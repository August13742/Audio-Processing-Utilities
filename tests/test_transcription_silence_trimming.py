import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.transcription import TranscriptionEngine, TimeMapper

def test_time_mapper_logic():
    segments = [(10.0, 15.0), (25.0, 30.0)]
    mapper = TimeMapper(segments)
    
    assert mapper.total_trimmed_duration == 10.0
    assert round(mapper.map_back(0.0), 3) == 10.0
    assert round(mapper.map_back(2.5), 3) == 12.5
    # Cross boundary at 5.0
    assert round(mapper.map_back(5.1), 3) == 25.1
    assert round(mapper.map_back(10.0), 3) == 30.0

def test_silence_detection_synthetic():
    # Create a 40s signal: 
    # 0-10s silence, 10-15s noise (speech), 15-25s silence, 25-30s noise (speech), 30-40s silence
    sr = 16000
    duration = 40
    audio = np.zeros(duration * sr)
    # Speech pulses
    audio[int(10*sr):int(15*sr)] = np.random.uniform(-0.5, 0.5, int(5*sr))
    audio[int(25*sr):int(30*sr)] = np.random.uniform(-0.5, 0.5, int(5*sr))
    
    engine = TranscriptionEngine()
    voiced, hop_s = engine._compute_energy_mask_from_audio(audio, sr)
    active_segments = engine._get_active_segments(voiced, hop_s, duration)
    
    # We expect 2 segments around [10, 15] and [25, 30] 
    # With padding of 0.5s: [9.5, 15.5] and [24.5, 30.5]
    print(f"Detected segments: {active_segments}")
    
    assert len(active_segments) == 2
    # Check if they are roughly where we expect
    assert abs(active_segments[0][0] - 9.5) < 0.2
    assert abs(active_segments[0][1] - 15.5) < 0.2
    assert abs(active_segments[1][0] - 24.5) < 0.2
    assert abs(active_segments[1][1] - 30.5) < 0.2

if __name__ == "__main__":
    test_time_mapper_logic()
    print("Time mapping logic test passed!")
    test_silence_detection_synthetic()
    print("Silence detection synthetic test passed!")
