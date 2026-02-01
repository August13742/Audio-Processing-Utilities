
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

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
            print(f"[ASR] Loading Faster-Whisper ({self.model_size}) on {device_str}...")
            # Automatically handles downloading
            self.model = WhisperModel(
                self.model_size, 
                device=device_str, 
                compute_type=self.compute_type
            )

    def transcribe(self, audio_path: str, beam_size: int = 5) -> Tuple[str, Dict]:
        """
        Transcribes audio file.
        Returns (text, metadata).
        """
        self._ensure_model()
        
        try:
            segments, info = self.model.transcribe(
                audio_path, 
                beam_size=beam_size,
                vad_filter=True, # Built-in VAD is useful
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            texts = []
            logprobs = []
            
            for segment in segments:
                texts.append(segment.text)
                logprobs.append(segment.avg_logprob)
                
            full_text = " ".join(texts).strip()
            
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "avg_logprob": float(np.mean(logprobs)) if logprobs else -99.0
            }
            
            return full_text, metadata
            
        except Exception as e:
            print(f"[ERR] Transcription failed used {audio_path}: {e}")
            return "", {}
