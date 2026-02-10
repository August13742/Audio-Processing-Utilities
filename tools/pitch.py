"""
Pitch extraction and curve processing for karaoke scoring.

Design philosophy: **forgiving scoring**.  The pipeline is procedurally
generated and will make mistakes.  We'd rather give the singer free
points than punish them for model errors.

Core flow:
    1. FCPE pitch extraction  →  raw f0 (Hz) + voicing mask
    2. Curve cleaning          →  smoothed continuous-MIDI curve
    3. Per-word pitch summary  →  median MIDI note per word (for scoring)

All functions operate on NumPy arrays.  The PitchExtractor class wraps
the torch-level FCPE model and returns plain arrays for downstream use.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from utils.log import log

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FCPE_HOP_SAMPLES = 160          # FCPE native hop size
FCPE_SAMPLE_RATE = 16_000       # FCPE expects 16 kHz input
HOP_SECONDS = FCPE_HOP_SAMPLES / FCPE_SAMPLE_RATE   # 0.01 s = 10 ms

# Voicing / singing range
HUMAN_F0_MIN = 65.0             # C2  — low male range
HUMAN_F0_MAX = 1100.0           # C6  — high soprano / falsetto
MIN_VOICED_DURATION_S = 0.03    # Segments shorter than 30 ms are likely noise

# Smoothing
SAVGOL_WINDOW_S = 0.05          # 50 ms Savitzky-Golay window
MEDIAN_KERNEL_S = 0.03          # 30 ms median filter for spike removal


# ---------------------------------------------------------------------------
# Hz  ↔  Continuous MIDI  ↔  Note name
# ---------------------------------------------------------------------------

def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    """
    Convert Hz → continuous MIDI note number.

    Unvoiced (f0 ≤ 0) maps to 0.  Scoring math is linear in semitones:
    a difference of 1.0 in MIDI space = exactly 1 semitone, regardless of
    register.  This is *not* the case in Hz space (exponential).
    """
    midi = np.zeros_like(f0, dtype=np.float64)
    voiced = f0 > 0
    midi[voiced] = 12.0 * np.log2(f0[voiced] / 440.0) + 69.0
    return midi


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Continuous MIDI → Hz.  0 → 0 (unvoiced)."""
    hz = np.zeros_like(midi, dtype=np.float64)
    voiced = midi > 0
    hz[voiced] = 440.0 * 2.0 ** ((midi[voiced] - 69.0) / 12.0)
    return hz


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_val: float) -> str:
    """64.2 → 'E4'.  Returns '' for unvoiced (≤0)."""
    if midi_val <= 0:
        return ""
    note_int = int(round(midi_val))
    octave = (note_int // 12) - 1
    name = _NOTE_NAMES[note_int % 12]
    return f"{name}{octave}"


# ---------------------------------------------------------------------------
# Curve cleaning (operates in MIDI space)
# ---------------------------------------------------------------------------

def clean_pitch_curve(
    f0_hz: np.ndarray,
    voiced_mask: np.ndarray,
    hop_s: float = HOP_SECONDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean raw FCPE output into a smooth MIDI pitch curve.

    Steps:
        1. Zero out unvoiced frames.
        2. Remove micro-voiced segments (< MIN_VOICED_DURATION_S).
        3. Median filter in MIDI space to kill isolated spikes.
        4. Savitzky-Golay smoothing to preserve melodic shape while
           removing digital jitter.

    Returns:
        (midi_clean, voiced_clean)  — same length as input.
        midi_clean has 0 for unvoiced frames.
    """
    from scipy.signal import savgol_filter, medfilt

    midi_raw = hz_to_midi(f0_hz)

    # 1. Apply voicing mask
    midi_raw[~voiced_mask] = 0.0

    # 2. Remove micro-voiced segments
    min_frames = max(1, int(MIN_VOICED_DURATION_S / hop_s))
    midi_raw = _remove_short_segments(midi_raw, min_frames)

    voiced = midi_raw > 0

    if voiced.sum() < 3:
        return midi_raw, voiced

    # 3. Median filter (kills isolated spikes)
    kernel = _to_odd(int(MEDIAN_KERNEL_S / hop_s))
    if kernel >= 3:
        # Only filter voiced regions — leave zeros untouched
        midi_filtered = medfilt(midi_raw, kernel_size=kernel)
        # Restore unvoiced regions to zero
        midi_filtered[~voiced] = 0.0
        midi_raw = midi_filtered
        voiced = midi_raw > 0

    # 4. Savitzky-Golay smoothing on voiced regions only
    window = _to_odd(int(SAVGOL_WINDOW_S / hop_s))
    if window >= 5 and voiced.sum() >= window:
        # Smooth only the voiced values in-place
        # We extract voiced runs and smooth each independently
        midi_raw = _smooth_voiced_runs(midi_raw, voiced, window, polyorder=2)
        voiced = midi_raw > 0

    return midi_raw, voiced


def _remove_short_segments(midi: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out voiced segments shorter than min_frames."""
    voiced = midi > 0
    changes = np.diff(voiced.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for s, e in zip(starts, ends):
        if (e - s) < min_frames:
            midi[s:e] = 0.0
    return midi


def _smooth_voiced_runs(
    midi: np.ndarray, voiced: np.ndarray, window: int, polyorder: int
) -> np.ndarray:
    """Apply Savitzky-Golay to each contiguous voiced run independently."""
    from scipy.signal import savgol_filter

    changes = np.diff(voiced.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    out = midi.copy()
    for s, e in zip(starts, ends):
        length = e - s
        w = min(window, _to_odd(length))
        if w >= 5 and length >= w:
            out[s:e] = savgol_filter(midi[s:e], w, min(polyorder, w - 1))
    return out


def _to_odd(n: int) -> int:
    """Round up to nearest odd integer ≥ 1."""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# Per-word pitch assignment
# ---------------------------------------------------------------------------

def assign_pitch_to_words(
    words: List[Dict],
    midi_curve: np.ndarray,
    voiced_mask: np.ndarray,
    hop_s: float = HOP_SECONDS,
) -> List[Dict]:
    """
    For each word, compute the median pitched MIDI value within its time span.

    Words with no voiced frames get ``pitch_midi: 0`` — the game should
    treat these as **free points** (no pitch penalty).

    Mutates *copies* of the word dicts in-place; returns the new list.
    """
    enriched = []
    n_frames = len(midi_curve)

    for w in words:
        w = dict(w)  # shallow copy
        start_frame = int(w["start"] / hop_s)
        end_frame = int(w["end"] / hop_s)
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, n_frames))

        segment = midi_curve[start_frame:end_frame]
        seg_voiced = voiced_mask[start_frame:end_frame]

        voiced_vals = segment[seg_voiced]

        if len(voiced_vals) >= 1:
            pitch_midi = float(np.median(voiced_vals))
            pitch_hz = float(midi_to_hz(np.array([pitch_midi]))[0])
            note = midi_to_note_name(pitch_midi)
        else:
            # No pitch detected → free points for the singer
            pitch_midi = 0.0
            pitch_hz = 0.0
            note = ""

        w["pitch_midi"] = round(pitch_midi, 2)
        w["pitch_hz"] = round(pitch_hz, 1)
        w["note"] = note
        enriched.append(w)

    return enriched


# ---------------------------------------------------------------------------
# Compact pitch section for JSON output
# ---------------------------------------------------------------------------

def build_pitch_section(
    midi_curve: np.ndarray,
    voiced_mask: np.ndarray,
    hop_s: float = HOP_SECONDS,
    words: List[Dict] = None,
) -> Dict:
    """
    Build the top-level ``pitch`` dict for the karaoke JSON.

    Uses compact parallel arrays (not per-frame dicts) to keep file size
    manageable. Only includes frames that correspond to the lyrics time range.

    Args:
        midi_curve:  MIDI pitch values (or 0 for unvoiced)
        voiced_mask: Boolean mask of voiced frames
        hop_s:       Hop time in seconds (frame duration)
        words:       Optional list of word dicts with 'start'/'end' times.
                     If provided, trims pitch array to cover word time range.

    Game reconstructs timestamps as ``t = start_seconds + i * hop_seconds``.
    Unvoiced frames have value ``0``.
    """
    # Determine time range to output (only where words exist)
    if words and len(words) > 0:
        start_time = min(w["start"] for w in words)
        end_time = max(w["end"] for w in words)
        start_frame = max(0, int(start_time / hop_s))
        end_frame = min(len(midi_curve), int(np.ceil(end_time / hop_s)))
        
        midi_trimmed = midi_curve[start_frame:end_frame]
        values = np.round(midi_trimmed, 2).tolist()
        
        return {
            "hop_seconds": hop_s,
            "unit": "midi",
            "start_seconds": round(start_time, 2),
            "values": values,
        }
    else:
        # Fallback: output full curve if no words (shouldn't happen in practice)
        values = np.round(midi_curve, 2).tolist()
        return {
            "hop_seconds": hop_s,
            "unit": "midi",
            "start_seconds": 0.0,
            "values": values,
        }


# ---------------------------------------------------------------------------
# PitchExtractor  —  torch-level FCPE wrapper
# ---------------------------------------------------------------------------

class PitchExtractor:
    """
    Wraps torchfcpe for vocal pitch extraction.

    Usage::

        px = PitchExtractor(device="cuda")
        result = px.extract("vocals.wav")
        # result = {
        #     "f0_hz":      np.ndarray,   # (N,) raw Hz, 0 = unvoiced
        #     "voiced":     np.ndarray,   # (N,) bool mask
        #     "midi_clean": np.ndarray,   # (N,) cleaned continuous MIDI
        #     "hop_seconds": 0.01,
        # }
    """

    def __init__(self, device: str = "cuda"):
        from torchfcpe import spawn_bundled_infer_model

        self.device = device
        log(f"[Pitch] Loading FCPE model on {device}")
        self.model = spawn_bundled_infer_model(device=device)

    def extract(self, audio_path: str) -> Dict:
        """
        Extract pitch from an audio file.

        Returns a dict with raw and cleaned pitch data, all as numpy arrays
        aligned on the same 10 ms grid.
        """
        import soundfile as sf
        import torchaudio

        log(f"[Pitch] Loading audio: {audio_path}")
        # Use soundfile instead of torchaudio.load to avoid TorchCodec dependency
        wav_np, sr = sf.read(audio_path)
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]  # (1, n_samples)
        else:
            wav_np = wav_np.T  # (channels, n_samples)

        wav = torch.from_numpy(wav_np.astype(np.float32))

        # Mix to mono → (1, n_samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        if sr != FCPE_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, FCPE_SAMPLE_RATE)
            sr = FCPE_SAMPLE_RATE

        # FCPE expects (B, n_sample, 1)
        wav_in = wav.unsqueeze(-1).to(self.device)

        log("[Pitch] Running FCPE inference...")
        with torch.no_grad():
            # Get f0 and voicing separately
            # Don't use interp_uv since we're not interpolating
            f0 = self.model.infer(
                wav_in,
                sr=sr,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=HUMAN_F0_MIN,
                f0_max=HUMAN_F0_MAX,
                interp_uv=False,
                retur_uv=False,
            )

        # Squeeze to 1-D numpy
        f0_np = f0.squeeze().cpu().numpy().astype(np.float64)

        # Get voicing mask via a second pass with thresholding
        # Low-confidence frames are treated as unvoiced
        with torch.no_grad():
            # Re-infer to get confidence scores for voicing detection
            f0_with_conf = self.model.infer(
                wav_in,
                sr=sr,
                decoder_mode="local_argmax",
                threshold=0.006,  # This threshold determines voicing
                f0_min=HUMAN_F0_MIN,
                f0_max=HUMAN_F0_MAX,
                interp_uv=False,
                retur_uv=False,
            )
        
        # Unvoiced frames show as very low values or 0 after threshold
        # Simple heuristic: f0 > 0 means voiced
        voiced = f0_np > 0

        # Additionally zero out anything outside singing range
        out_of_range = (f0_np < HUMAN_F0_MIN) | (f0_np > HUMAN_F0_MAX)
        f0_np[out_of_range] = 0.0
        voiced[out_of_range] = False

        log(f"[Pitch] {len(f0_np)} frames, {voiced.sum()} voiced "
            f"({100 * voiced.mean():.1f}%)")

        # Clean
        midi_clean, voiced_clean = clean_pitch_curve(f0_np, voiced)

        log(f"[Pitch] After cleaning: {voiced_clean.sum()} voiced frames")

        return {
            "f0_hz": f0_np,
            "voiced": voiced_clean,
            "midi_clean": midi_clean,
            "hop_seconds": HOP_SECONDS,
        }
