import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.transcription import TranscriptionEngine

def main():
    test_file = "TestData/黒田崇矢 - ばかみたい [Taxi Driver Edition].flac"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    print(f"Verifying silence trimming on: {test_file}")
    engine = TranscriptionEngine()
    
    # We want to see the "ASR Trimmed" log message
    # And we want to see the detected segments
    # Setting return_timestamps=False for speed if Whisper runs, but we might just check energy detection
    
    import soundfile as sf
    audio, sr = sf.read(test_file)
    total_duration = len(audio) / sr
    
    # Analyze energy (copied from TranscriptionEngine)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    _RMS_WINDOW_S = 0.05
    hop_samples = int(_RMS_WINDOW_S * sr)
    n_frames = len(audio) // hop_samples
    rms = np.array([
        np.sqrt(np.mean(audio[i * hop_samples:(i + 1) * hop_samples] ** 2))
        for i in range(n_frames)
    ])
    peak_rms = rms.max() if rms.max() > 0 else 1e-10
    
    # Print decibel levels for every 5 seconds to see intro characteristics
    print(f"Peak RMS: {peak_rms:.4f}")
    print("RMS profile (dB relative to peak):")
    for sec in range(0, min(60, int(total_duration)), 5):
        idx = int(sec / _RMS_WINDOW_S)
        if idx < len(rms):
            db = 20 * np.log10(rms[idx] / peak_rms + 1e-10)
            print(f"  {sec:2}s: {db:6.1f} dB")

    voiced, hop_s = engine._compute_energy_mask_from_audio(audio, sr)
    active_segments = engine._get_active_segments(voiced, hop_s, total_duration)
    
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Active segments ({len(active_segments)}):")
    for s, e in active_segments:
        print(f"  - {s:.2f}s to {e:.2f}s (dur: {e-s:.2f}s)")
    
    trimmed_dur = sum(e-s for s, e in active_segments)
    print(f"Final trimmed duration: {trimmed_dur:.2f}s")

if __name__ == "__main__":
    main()
