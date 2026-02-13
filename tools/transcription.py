
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
# have detectable vocal energy to keep it. Segments below this are
# discarded as likely hallucinations in silent regions.
_ENERGY_OVERLAP_THRESHOLD = 0.15

# RMS window size for energy calculation (seconds)
_RMS_WINDOW_S = 0.05

# dB below peak RMS to consider "silence". -40dB means anything quieter
# than 1/100th of the loudest moment is treated as no vocal activity.
_SILENCE_DB = -40.0


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
    def _compute_energy_mask(audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Compute a per-frame boolean mask: True where vocal energy exceeds
        the silence threshold.

        Uses RMS in short windows. The threshold is relative to the peak
        RMS of the file, making it adaptive to any recording level.

        Returns (voiced_mask, hop_seconds).
        """
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        hop_samples = int(_RMS_WINDOW_S * sr)
        n_frames = len(audio) // hop_samples

        # Compute RMS per window
        rms = np.array([
            np.sqrt(np.mean(audio[i * hop_samples:(i + 1) * hop_samples] ** 2))
            for i in range(n_frames)
        ])

        # Relative threshold: _SILENCE_DB below peak RMS
        peak_rms = rms.max() if rms.max() > 0 else 1e-10
        threshold = peak_rms * (10.0 ** (_SILENCE_DB / 20.0))

        voiced = rms > threshold
        log(f"[ENERGY] {voiced.sum()}/{n_frames} frames above threshold "
            f"({voiced.sum() * _RMS_WINDOW_S:.1f}s / {n_frames * _RMS_WINDOW_S:.1f}s)")

        return voiced, _RMS_WINDOW_S

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

        Pipeline:
            1. Whisper transcribes with VAD disabled (to not cut singing)
            2. RMS energy detection on the vocal track flags silent regions
            3. Segments in silent regions are discarded as hallucinations

        Returns (text, metadata) where metadata includes word-level timestamps.
        """
        self._ensure_model()
        
        try:
            # Karaoke-tuned parameters for song transcription
            segments, info = self.model.transcribe(
                audio_path, 
                word_timestamps=return_timestamps,
                vad_filter=False,                       # Disable VAD — sustained vowels are not silence
                no_speech_threshold=0.8,                 # Raise bar so quiet passages aren't dropped
                compression_ratio_threshold=2.8,         # Relaxed — songs have repetitive phrases
                condition_on_previous_text=False,         # Prevent cascading hallucinations
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Fallback chain on failed segments
            )
            
            texts = []
            segments_data = []
            words = []
            
            for segment in segments:
                texts.append(segment.text)
                segments_data.append({
                    "text": segment.text.strip(),
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                })
                if return_timestamps and segment.words:
                    for w in segment.words:
                        words.append({
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4)
                        })

            # --- Post-hoc energy verification ---
            voiced_mask, hop_s = self._compute_energy_mask(audio_path)
            segments_data, words = self._verify_with_energy(
                segments_data, words, voiced_mask, hop_s
            )

            # Rebuild text from verified segments only
            full_text = " ".join(seg["text"] for seg in segments_data).strip()
            
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_data,
                "timestamps": words
            }
            
            return full_text, metadata
            
        except Exception as e:
            log(f"[ERR] Transcription failed for {audio_path}: {e}")
            return "", {}
