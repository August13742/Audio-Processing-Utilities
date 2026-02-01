
import os
import shutil
from pathlib import Path
from typing import Optional, List
from .separation import SeparationEngine
from .enhancement import EnhancementEngine
from utils.audio import load_audio, save_audio, l2norm
from utils.processing import normalize_loudness

class CleaningEngine:
    def __init__(self, device: str = "cuda", verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.separator = SeparationEngine(device=device, verbose=verbose)
        self.enhancer = EnhancementEngine(device=device, verbose=verbose)

    def process(self, input_path: str, output_dir: str):
        """
        Full cleaning pipeline: 
        1. Separate Vocals/Inst
        2. Separate Lead/Backing from Vocals
        3. De-reverb/Denoise Lead
        4. Normalize
        """
        input_path = Path(input_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Separate Vocals/Inst
        if self.verbose: print(f"[CLEAN] Step 1: Separating Vocals/Inst for {input_path.name}...")
        vocals, inst = self.separator.separate_vocals(str(input_path), str(out_path))
        
        if not vocals:
            print(f"[ERR] Failed to extract vocals from {input_path.name}")
            return

        # 2. Separate Lead/Backing
        if self.verbose: print(f"[CLEAN] Step 2: Separating Lead/Backing...")
        lead, backing = self.separator.separate_lead_backing(vocals, str(out_path))
        
        if not lead:
            lead = vocals # Fallback if model fails or file is mono/simple
            
        # 3. Enhance Lead (De-reverb)
        clean_lead_name = f"{input_path.stem}_lead_clean.wav"
        clean_lead_path = out_path / clean_lead_name
        
        if self.verbose: print(f"[CLEAN] Step 3: Enhancing Lead...")
        success = self.enhancer.process(lead, str(clean_lead_path))
        
        if not success:
            print(f"[WARN] Enhancement failed, utilizing raw lead.")
            shutil.copy2(lead, clean_lead_path)

        # Cleanup of intermediate files could be added here if needed.
        
        print(f"[CLEAN] Finished. Output at {clean_lead_path}")
        return str(clean_lead_path)
