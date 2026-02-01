
import torch
import numpy as np
import librosa
from pathlib import Path
from utils.audio import load_audio

class AudioAnalyzer:
    """
    Analyzes audio signals for various characteristics.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device

    def detect_polyphony(self, audio_path: str) -> dict:
        """
        Detects if audio is polyphonic (Harmonies/Choir) vs Monophonic.
        Uses Stereo Width and Spectral Desnity.
        """
        results = {
            "is_polyphonic": False,
            "stereo_width": 0.0,
            "harmonic_density": 0.0,
            "confidence": 0.0
        }
        
        try:
            # Load as stereo for width analysis
            y, sr = load_audio(audio_path, mono=False)
            
            # --- 1. Stereo Width Analysis ---
            if y.shape[0] > 1:
                L = y[0].cpu().numpy()
                R = y[1].cpu().numpy()
                
                mid = (L + R) / 2.0
                side = (L - R) / 2.0
                
                rmse_mid = np.sqrt(np.mean(mid**2) + 1e-9)
                rmse_side = np.sqrt(np.mean(side**2) + 1e-9)
                
                width_score = float(rmse_side / rmse_mid)
                results["stereo_width"] = width_score
            
            # --- 2. Spectral Density (CQT) ---
            # We use a mono version for this
            y_mono = y.mean(dim=0).cpu().numpy()
            
            # Use CQT to find harmonic peaks
            # Higher density of peaks often means harmonies
            cqt = np.abs(librosa.cqt(y_mono, sr=sr, n_bins=72, bins_per_octave=12))
            
            # Calculate average number of active bins per frame
            # A 'bin' is active if it's > threshold (e.g. 10% of max)
            threshold = np.max(cqt) * 0.1
            active_bins = np.sum(cqt > threshold, axis=0)
            avg_active = float(np.mean(active_bins))
            
            # Normalize density score (rough heuristic)
            density_score = avg_active / 12.0 # > 1.0 often indicates multi-voice
            results["harmonic_density"] = density_score
            
            # Final Decision
            if results["stereo_width"] > 0.25 or results["harmonic_density"] > 1.2:
                results["is_polyphonic"] = True
                results["confidence"] = min(1.0, max(results["stereo_width"] * 2, results["harmonic_density"] / 2))
                
            return results
        except Exception as e:
            print(f"[ERR] Analysis failed for {audio_path}: {e}")
            return results
