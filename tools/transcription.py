
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from utils.log import log

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Energy verification: minimum fraction of a segment's duration that must
# have detectable vocal energy to keep it. Lowered for songs with long pauses.
_ENERGY_OVERLAP_THRESHOLD = 0.03

# RMS window size for energy calculation (seconds)
_RMS_WINDOW_S = 0.05

# dB below peak RMS to consider "silence". -45dB is safe for clean vocals
# while still catching quiet breathy singing.
_SILENCE_DB = -45.0

# Silence trimming for ASR: we trim internal/external silences longer than this.
# This prevents language detection bugs and prediction bias on silence.
_MAX_SILENCE_S = 3.0

# Padding to keep around detected vocal segments (seconds)
_VOCAL_PADDING_S = 0.5

# Common Whisper hallucinations in karaoke/silent contexts
_HALLUCINATION_BLACKLIST = [
    "ご視聴ありがとうございました",
    "視聴ありがとうございました",
    "ありがとうございました",
    "チャンネル登録",
    "Thank you for watching",
    "Please subscribe",
    "Thanks for watching",
]


class TimeMapper:
    """Maps timestamps from a concatenated (trimmed) audio back to original."""
    def __init__(self, segments: List[Tuple[float, float]]):
        self.segments = segments
        self.accumulated_durations = []
        curr = 0.0
        for s, e in segments:
            self.accumulated_durations.append(curr)
            curr += (e - s)
        self.total_trimmed_duration = curr

    def map_back(self, t: float) -> float:
        """Map trimmed time 't' to original time."""
        for i in range(len(self.segments) - 1, -1, -1):
            if t >= self.accumulated_durations[i]:
                start, end = self.segments[i]
                return start + (t - self.accumulated_durations[i])
        return self.segments[0][0] if self.segments else t


class TranscriptionEngine:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.model = None
        
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed. Please install it via uv.")
            
    def _ensure_model(self):
        if self.model is None:
            device_str = "cuda" if "cuda" in self.device else "cpu"
            log(f"[ASR] Loading Faster-Whisper ({self.model_size}) on {device_str}...")
            self.model = WhisperModel(
                self.model_size, 
                device=device_str, 
                compute_type=self.compute_type
            )

    # ------------------------------------------------------------------
    # Energy-based verification (uses clean vocal track from separation)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_energy_mask_from_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """Compute vocal energy mask from raw audio array."""
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        hop_samples = int(_RMS_WINDOW_S * sr)
        n_frames = len(audio) // hop_samples
        
        # Reshape for vectorized RMS calculation
        # Trim audio to be a multiple of hop_samples
        audio_trimmed = audio[:n_frames * hop_samples].reshape(n_frames, hop_samples)
        rms = np.sqrt(np.mean(audio_trimmed**2, axis=1))

        # Relative threshold: _SILENCE_DB below peak RMS
        peak_rms = rms.max() if rms.max() > 0 else 1e-10
        threshold = peak_rms * (10.0 ** (_SILENCE_DB / 20.0))

        voiced = rms > threshold
        return voiced, _RMS_WINDOW_S

    @staticmethod
    def _get_active_segments(voiced: np.ndarray, hop_s: float, total_duration: float) -> List[Tuple[float, float]]:
        """
        Group voiced frames into contiguous segments, merging silences < _MAX_SILENCE_S.
        Returns list of (start_s, end_s).
        """
        if not voiced.any():
            return []

        segments = []
        is_active = False
        start_idx = 0

        # Basic contiguous blocks
        for i, val in enumerate(voiced):
            if val and not is_active:
                is_active = True
                start_idx = i
            elif not val and is_active:
                is_active = False
                segments.append([start_idx * hop_s, i * hop_s])
        
        if is_active:
            segments.append([start_idx * hop_s, len(voiced) * hop_s])

        if not segments:
            return []

        # Merge segments separated by small silences
        merged = []
        curr_s, curr_e = segments[0]
        
        for next_s, next_e in segments[1:]:
            if next_s - curr_e < _MAX_SILENCE_S:
                curr_e = next_e
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged.append((curr_s, curr_e))

        # Add padding and clip
        padded = []
        for s, e in merged:
            ps = max(0, s - _VOCAL_PADDING_S)
            pe = min(total_duration, e + _VOCAL_PADDING_S)
            padded.append((ps, pe))

        # Re-merge if padding caused overlap
        final = []
        if not padded: return []
        curr_s, curr_e = padded[0]
        for next_s, next_e in padded[1:]:
            if next_s < curr_e:
                curr_e = max(curr_e, next_e)
            else:
                final.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        final.append((curr_s, curr_e))

        return final

    @staticmethod
    def _compute_energy_mask(audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Compute a per-frame boolean mask: True where vocal energy exceeds
        the silence threshold.
        """
        audio, sr = sf.read(audio_path, dtype="float32")
        return TranscriptionEngine._compute_energy_mask_from_audio(audio, sr)

    @staticmethod
    def _segment_has_energy(
        start: float,
        end: float,
        voiced_mask: np.ndarray,
        hop_s: float,
    ) -> float:
        """
        Compute what fraction of a time range has vocal energy.
        Returns 0.0 - 1.0.
        """
        i_start = int(start / hop_s)
        i_end = int(end / hop_s)
        i_start = max(0, min(i_start, len(voiced_mask) - 1))
        i_end = max(i_start + 1, min(i_end, len(voiced_mask)))

        span = voiced_mask[i_start:i_end]
        if len(span) == 0:
            return 0.0
        return float(span.sum()) / len(span)

    def _verify_with_energy(
        self,
        segments_data: List[Dict],
        words: List[Dict],
        voiced_mask: np.ndarray,
        hop_s: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Discard Whisper segments that land in silent regions of the vocal track.
        Also removes orphan words with no vocal energy.
        """
        verified_segments = []
        discarded_ranges = []

        for seg in segments_data:
            ratio = self._segment_has_energy(seg["start"], seg["end"], voiced_mask, hop_s)
            if ratio >= _ENERGY_OVERLAP_THRESHOLD:
                verified_segments.append(seg)
            else:
                discarded_ranges.append((seg["start"], seg["end"]))
                log(f"[ENERGY] Discarded segment [{seg['start']:.1f}–{seg['end']:.1f}]: "
                    f"\"{seg['text']}\" (energy={ratio:.0%})")

        if not discarded_ranges:
            return segments_data, words

        # Remove words in discarded segments + orphan words in silence
        verified_words = []
        for w in words:
            if any(s <= w["start"] < e for s, e in discarded_ranges):
                continue
            if self._segment_has_energy(w["start"], w["end"], voiced_mask, hop_s) == 0.0:
                continue
            verified_words.append(w)

        n_dropped = len(segments_data) - len(verified_segments)
        w_dropped = len(words) - len(verified_words)
        log(f"[ENERGY] Verification: dropped {n_dropped} segments, {w_dropped} words")

        return verified_segments, verified_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, return_timestamps: bool = True) -> Tuple[str, Dict]:
        """
        Transcribes audio file using Faster-Whisper with energy-based verification.
        Uses silence trimming to prevent hallucination on long instrumentals.
        """
        self._ensure_model()
        
        try:
            # 1. Load audio and ensure mono 16kHz (Standard for Whisper)
            # Logic: Whisper is most stable with 16kHz mono.
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Simple downsampling if sr != 16000
            if sr != 16000:
                log(f"[ASR] Resampling from {sr}Hz to 16000Hz...")
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    # Fallback to simple slice if librosa is missing (rudimentary)
                    # or better: use scipy.signal.resample
                    try:
                        from scipy.signal import resample
                        num_samples = int(len(audio) * 16000 / sr)
                        audio = resample(audio, num_samples)
                    except:
                        log("[WARN] librosa/scipy missing, skipping resampling. Transcription might be slow/unstable.")
                sr = 16000

            total_duration = len(audio) / sr
            
            # 2. Detect active vocal regions to avoid silent intros/outros/breaks
            voiced, hop_s = self._compute_energy_mask_from_audio(audio, sr)
            active_segments = self._get_active_segments(voiced, hop_s, total_duration)
            
            if not active_segments:
                log(f"[ASR] No vocal activity detected in {audio_path}")
                return "", {"language": "en", "duration": total_duration, "segments": [], "timestamps": []}

            # 3. Extract and concatenate active regions
            active_chunks = []
            for s, e in active_segments:
                active_chunks.append(audio[int(s*sr):int(e*sr)])
            
            trimmed_audio = np.concatenate(active_chunks) if len(active_chunks) > 1 else active_chunks[0]
            mapper = TimeMapper(active_segments)
            
            log(f"[ASR] Trimmed {total_duration:.1f}s -> {mapper.total_trimmed_duration:.1f}s "
                f"({len(active_segments)} segments)")

            # 4. Transcribe trimmed audio
            segments, info = self.model.transcribe(
                trimmed_audio, 
                word_timestamps=return_timestamps,
                vad_filter=False,                       # Disable Whisper VAD (we did our own)
                no_speech_threshold=0.8,
                compression_ratio_threshold=2.8,
                condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            )
            
            texts = []
            segments_data = []
            words = []
            
            for segment in segments:
                # Map timestamps back to original timeline
                start_orig = mapper.map_back(segment.start)
                end_orig = mapper.map_back(segment.end)
                
                texts.append(segment.text)
                segments_data.append({
                    "text": segment.text.strip(),
                    "start": round(start_orig, 3),
                    "end": round(end_orig, 3),
                })
                
                if return_timestamps and segment.words:
                    for w in segment.words:
                        ws_orig = mapper.map_back(w.start)
                        we_orig = mapper.map_back(w.end)
                        words.append({
                            "word": w.word,
                            "start": round(ws_orig, 3),
                            "end": round(we_orig, 3),
                            "probability": round(w.probability, 4)
                        })

            # 4. Energy verification (final safety check)
            # Re-compute voiced mask for full file if needed, but we already have it
            segments_data, words = self._verify_with_energy(
                segments_data, words, voiced, hop_s
            )

            # 5. Hallucination Filtering (Trailing blacklist)
            # Whisper often adds "Thank you for watching" etc at the end if it gets stuck.
            if segments_data:
                last_seg = segments_data[-1]
                if any(h in last_seg["text"] for h in _HALLUCINATION_BLACKLIST):
                    log(f"[ASR] Filtered trailing hallucination: \"{last_seg['text']}\"")
                    segments_data.pop()
                    # Also remove corresponding words
                    last_id = len(segments_data) # index after pop
                    words = [w for w in words if any(s <= w["start"] < e for s, e in ((seg["start"], seg["end"]) for seg in segments_data))]
                    # Re-calculate full text
                    
            full_text = " ".join(seg["text"] for seg in segments_data).strip()
            
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": total_duration,
                "segments": segments_data,
                "timestamps": words
            }
            
            return full_text, metadata
            
        except Exception as e:
            import traceback
            log(f"[ERR] Transcription failed: {e}\n{traceback.format_exc()}")
            return "", {}
