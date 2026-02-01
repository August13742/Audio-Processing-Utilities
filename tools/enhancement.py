
import os
import sys
import torch
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional
from utils.audio import load_audio, save_audio

class EnhancementEngine:
    """
    Handles audio enhancement using ClearVoice (MossFormer2).
    """
    def __init__(self, device: str = "cuda", verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.model = None
        self._setup_paths()

    def _setup_paths(self):
        # Path to ClearVoice repo
        self.repo_path = Path("models/ClearerVoice-Studio-main").resolve()
        self.clearvoice_src = self.repo_path / "clearvoice"
        
        # Inject into sys.path
        if self.repo_path.exists() and str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
            if self.verbose: print(f"[ENHANCE] Added {self.repo_path} to sys.path")

    def _init_model(self):
        if self.model is None:
            try:
                from clearvoice import ClearVoice
                if self.verbose: print("[ENHANCE] Initializing ClearVoice (MossFormer2_SE_48K)...")
                # MossFormer2_SE_48K initialization
                self.model = ClearVoice(
                    task='speech_enhancement',
                    model_names=['MossFormer2_SE_48K']
                )
            except ImportError as e:
                print(f"[ERR] ClearVoice dependencies missing: {e}")
                raise
            except Exception as e:
                print(f"[ERR] Failed to init ClearVoice: {e}")
                raise

    def process(self, input_path: str, output_path: str) -> bool:
        """
        Enhances audio and saves to output_path.
        """
        self._init_model()
        
        try:
            if self.verbose: print(f"[ENHANCE] Processing {Path(input_path).name}...")
            
            # ClearVoice call
            # Run ClearVoice enhancement
            output_wav = self.model(input_path=input_path, online_write=False)
            
            # Write output file
            self.model.write(output_wav, output_path=output_path)
            
            return True
        except Exception as e:
            print(f"[ERR] Enhancement failed for {input_path}: {e}")
            return False

def apply_postprocessing(audio_path: str):
    """
    Apply compression/normalization after ClearVoice if needed.
    """
    # TODO: Implement if needed using utils/audio.py
    pass
