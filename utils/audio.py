
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
import math

# Monkey patch for Speechbrain <-> Torchaudio compatibility if needed
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

EPS = 1e-9

def load_audio(path: str, target_sr: int = None, mono: bool = True) -> tuple[torch.Tensor, int]:
    """
    Loads audio using SoundFile (for robustness) and converts to Torch Tensor.
    Returns (Tensor, sr).
    Tensor shape: (Channels, Time) or (Time) ?
    We standardize to (Channels, Time) usually, but for mono 1D is often used.
    Lets standardize on returning (Time,) for Mono or (Channels, Time) for Stereo?
    Actually for Torch compatibility, (Channels, Time) is best.
    """
    try:
        y_np, sr = sf.read(path)
        y_np = y_np.astype(np.float32)
        
        # sf.read returns (Time, Channels) if stereo, (Time,) if mono.
        if y_np.ndim > 1:
            # Transpose to (Channels, Time)
            y_np = y_np.T
        else:
            # Expand to (1, Time)
            y_np = y_np[None, :]

        if mono and y_np.shape[0] > 1:
            y_np = y_np.mean(axis=0, keepdims=True)

        y = torch.from_numpy(y_np)

        if target_sr and sr != target_sr:
            y = torchaudio.functional.resample(y, sr, target_sr)
            sr = target_sr
            
        return y, sr
    except Exception as e:
        print(f"[ERR] Failed to load {path}: {e}")
        return torch.zeros(1, 0), 0

def save_audio(path: str, y: torch.Tensor, sr: int):
    """
    Saves audio using SoundFile.
    Expects y to be (Channels, Time) or (Time).
    """
    y_np = y.cpu().numpy()
    if y_np.ndim == 2:
        y_np = y_np.T # Soundfile wants (Time, Channels)
    
    sf.write(path, y_np, sr)

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)

class LoudnessNormalizer:
    def __init__(self, target_lufs: float = -23.0, max_gain_db: float = 10.0):
        self.target_lufs = target_lufs
        self.max_gain_db = max_gain_db
        try:
            import pyloudnorm as pyln
            self.pyln = pyln
        except ImportError:
            self.pyln = None

    def normalize(self, y: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Input: (Channels, Time)
        """
        if self.pyln is None:
            # Fallback to RMS
            return self._normalize_rms(y)
        
        meter = self.pyln.Meter(sr)
        # Pyloudnorm expects (Time, Channels)
        y_np = y.cpu().numpy().T
        
        try:
            loudness = meter.integrated_loudness(y_np)
            if loudness == -float('inf'):
                return y
                
            gain_db = self.target_lufs - loudness
            if gain_db > self.max_gain_db:
                gain_db = self.max_gain_db
            
            gain = 10 ** (gain_db / 20.0)
            return y * gain
        except Exception:
            return y

    def _normalize_rms(self, y: torch.Tensor, target_rms=0.1) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(y**2))
        if rms > 0:
            return y * (target_rms / rms)
        return y

def check_activity(path: str, threshold: float = 0.015, duration_s: float = 60.0) -> bool:
    """
    Checks if an audio file has significant energy (not silent).
    Scans the first 'duration_s' seconds.
    """
    if not os.path.exists(path):
        return False
    try:
        # Read a chunk to check activity
        y, sr = sf.read(path, frames=int(44100 * duration_s))
        if y.size == 0:
            return False
        peak = np.max(np.abs(y))
        return peak > threshold
    except Exception as e:
        print(f"[WARN] Activity check failed for {path}: {e}")
        return False

def match_audio_length(y: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pads or trims tensor to match target length.
    Input y: (Channels, Time)
    """
    curr_len = y.shape[-1]
    if curr_len == target_length:
        return y
    if curr_len < target_length:
        return torch.nn.functional.pad(y, (0, target_length - curr_len))
    else:
        return y[..., :target_length]
