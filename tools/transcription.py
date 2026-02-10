
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from utils.log import log

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
            log(f"[ASR] Loading Faster-Whisper ({self.model_size}) on {device_str}...")
            # Automatically handles downloading
            self.model = WhisperModel(
                self.model_size, 
                device=device_str, 
                compute_type=self.compute_type
            )

    def transcribe(self, audio_path: str, return_timestamps: bool = True) -> Tuple[str, Dict]:
        """
        Transcribes audio file using Faster-Whisper.
        Returns (text, metadata) where metadata includes word-level timestamps.
        """
        self._ensure_model()
        
        try:
            # More forgiving parameters for song transcription
            segments, info = self.model.transcribe(
                audio_path, 
                word_timestamps=return_timestamps,
                vad_filter=False, # Disable VAD to avoid cutting singing
                no_speech_threshold=0.8, # More forgiving than default 0.6
                compression_ratio_threshold=2.8, # More forgiving
                condition_on_previous_text=False, # Don't rely on early segments for songs
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
                
            full_text = "".join(texts).strip() # Joined for Japanese
            
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_data,
                "timestamps": words
            }
            
            return full_text, metadata
            
        except Exception as e:
            log(f"[ERR] Transcription failed used {audio_path}: {e}")
            return "", {}
