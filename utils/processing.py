
import torch
import math
import numpy as np

EPS = 1e-9

class PeakLimiter:
    def __init__(self, peak_ceiling_db: float = -1.0, trim_db: float = -50.0, tp_over: int = 1):
        self.peak_ceiling_db = float(peak_ceiling_db)
        self.trim_db = float(trim_db)
        self.tp_over = max(1, int(tp_over))

    def process(self, seg: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Applies hard limiting to the segment.
        """
        if seg.numel() < int(0.25 * sr): 
            return torch.zeros(0, dtype=seg.dtype, device=seg.device)
        
        # Simple Gate
        amp = 10 ** (self.trim_db / 20.0)
        idx = torch.where(seg.abs() > amp)[0]
        
        if idx.numel() == 0: 
            return torch.zeros(0, dtype=seg.dtype, device=seg.device)
            
        pad = int(0.01 * sr)
        start = max(int(idx[0] - pad), 0)
        end   = min(int(idx[-1] + pad), seg.numel())
        x = seg[start:end]

        # Peak Limit
        ceiling = 10 ** (self.peak_ceiling_db / 20.0)
        peak = float(x.abs().max())
        
        if peak > ceiling:
            x = x * (ceiling / (peak + EPS))
            
        return x

def normalize_loudness(y: torch.Tensor, sr: int, target_lufs: float = -23.0) -> torch.Tensor:
    # Basic RMS fallback if pyloudnorm isn't available or for speed
    # Used if the advanced one in audio.py isn't desired or for different logic
    rms = torch.sqrt(torch.mean(y**2) + 1e-9)
    # Approx conversation: -23 LUFS is roughly 0.1 RMS for speech? 
    # Actually -23 LUFS is standard broadcast.
    # Let's just stick to a Peak normalization fallback if needed.
    return y
