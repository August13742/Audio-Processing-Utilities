"""
Pitch extraction and curve processing for karaoke scoring.

Design philosophy: **forgiving scoring**.  The pipeline is procedurally
generated and will make mistakes.  We'd rather give the singer free
points than punish them for model errors.

Core flow:
    1. FCPE pitch extraction  →  raw f0 (Hz) + voicing mask
    2. Curve cleaning          →  smoothed continuous-MIDI curve
    3. Sparse pitch events     →  median MIDI at ~100 ms intervals (voiced only)

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

# Sparse pitch output
PITCH_EVENT_HOP_S = 0.1         # 100 ms — coarse enough to hide jitter


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
# Sparse voiced-only pitch events
# ---------------------------------------------------------------------------

def build_sparse_pitch(
    midi_curve: np.ndarray,
    voiced_mask: np.ndarray,
    hop_s: float = HOP_SECONDS,
    event_hop_s: float = PITCH_EVENT_HOP_S,
    max_time: Optional[float] = None,
) -> List[Dict]:
    """
    Downsample the fine FCPE curve into sparse pitch events.

    Each event is a short time block (~100 ms) with a median MIDI value.
    Only blocks that contain voiced frames are emitted.

    Args:
        midi_curve:   Cleaned MIDI pitch curve.
        voiced_mask:  Voicing mask.
        hop_s:        Native hop size of the curve.
        event_hop_s:  Hop size for the sparse output (target interval).
        max_time:     Optional. If set, pitch events after this time are discarded.
                      Useful for trimming pitch to the range of the lyrics.
    """
    from scipy.signal import medfilt

    n_frames = len(midi_curve)
    block_size = max(1, int(event_hop_s / hop_s))  # frames per event block

    events: List[Dict] = []

    for block_start in range(0, n_frames, block_size):
        t = round(block_start * hop_s, 2)
        if max_time is not None and t > max_time:
            break

        block_end = min(block_start + block_size, n_frames)
        seg = midi_curve[block_start:block_end]
        seg_voiced = voiced_mask[block_start:block_end]
        voiced_vals = seg[seg_voiced]

        if len(voiced_vals) < 1:
            continue

        pitch_midi = int(round(float(np.median(voiced_vals))))

        events.append({
            "time": t,
            "midi": pitch_midi,
            "note": midi_to_note_name(pitch_midi),
        })

    # Post-hoc median filter across consecutive events to kill outlier blocks.
    # Operates on the midi values only; timestamps are untouched.
    if len(events) >= 3:
        midi_arr = np.array([e["midi"] for e in events])
        kernel = min(5, _to_odd(len(midi_arr)))
        if kernel >= 3:
            smoothed = medfilt(midi_arr, kernel_size=kernel)
            for i, e in enumerate(events):
                val = int(round(float(smoothed[i])))
                e["midi"] = val
                e["note"] = midi_to_note_name(val)

    log(f"[Pitch] Built {len(events)} sparse pitch events "
        f"(hop={event_hop_s}s, voiced only, whole-note)")

    return events


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

        # Voicing: f0 > 0 means voiced
        voiced = f0_np > 0

        # Zero out anything outside singing range
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
