
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import torch
import soundfile as sf
import numpy as np
from audio_separator.separator import Separator
from utils.models import ensure_models
from utils.audio import load_audio, save_audio

class SeparationEngine:
    def __init__(self, model_dir: str = "models", device: str = "cuda", verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.device = device
        self.verbose = verbose
        # Ensure models exist
        ensure_models(self.model_dir, verbose=self.verbose)

    def separate_vocals(self, input_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Separates audio into (Vocals, Instrumental) using BS-Roformer.
        Returns paths to (vocals_path, inst_path).
        """
        fname = Path(input_path).stem
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Temp dir for this separation
        temp_dir = out_path / f"temp_{fname}_s1"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            if self.verbose: print(f"[SEP] Separating Vocals/Inst for {fname}...")
            
            # Initialize Separator
            sep = Separator(
                output_dir=str(temp_dir),
                model_file_dir=str(self.model_dir),
                output_format="WAV",
                log_level=40 # Error only
            )
            
            # Load BS-Roformer
            sep.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            
            # Run Separation
            sep.separate(input_path)
            
            # Find outputs
            vocals_path = None
            inst_path = None
            other_stems = []

            for f in temp_dir.glob("*.wav"):
                name = f.name.lower()
                dest = out_path / f"{fname}_{f.name}"
                
                # Move to final dir
                shutil.move(str(f), str(dest))
                
                if "(vocals)" in name:
                    vocals_path = str(dest)
            # If BS-Roformer is a STEM separator, we combine non-vocal stems for BGM
            stem_map = {}
            for f in out_path.glob(f"{fname}_*.wav"):
                if "(vocals)" in f.name.lower(): stem_map["vocals"] = f
                if "(drums)" in f.name.lower(): stem_map["drums"] = f
                if "(bass)" in f.name.lower(): stem_map["bass"] = f
                if "(other)" in f.name.lower(): stem_map["other"] = f
                if "(guitar)" in f.name.lower(): stem_map["guitar"] = f
                if "(piano)" in f.name.lower(): stem_map["piano"] = f

            if "vocals" in stem_map:
                vocals_path = str(stem_map["vocals"])
                
                # Combine others for BGM
                bgm_stems = [p for k, p in stem_map.items() if k != "vocals"]
                if bgm_stems:
                    if self.verbose: print(f"  [MIX] Combining {len(bgm_stems)} stems for BGM...")
                    
                    # Determine length
                    y_ref, sr = load_audio(str(bgm_stems[0]))
                    y_sum = torch.zeros_like(y_ref)
                    y_sum += y_ref
                    
                    for p in bgm_stems[1:]:
                        y_next, _ = load_audio(str(p), target_sr=sr)
                        # Handle length mismatch
                        if y_next.shape[-1] != y_sum.shape[-1]:
                            min_len = min(y_next.shape[-1], y_sum.shape[-1])
                            y_sum = y_sum[..., :min_len] + y_next[..., :min_len]
                        else:
                            y_sum += y_next
                            
                    inst_path = str(out_path / f"{fname}_instrumental.wav")
                    save_audio(inst_path, y_sum, sr)
            
            return vocals_path, inst_path
                
        except Exception as e:
            print(f"[ERR] Separation failed: {e}")
            return None, None
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def separate_stems(self, input_path: str, output_dir: str):
        """
        Runs BS-Roformer to get all 6 stems.
        """
        fname = Path(input_path).stem
        out_path = Path(output_dir) / fname
        out_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose: print(f"[SEP] 6-Stem Separation for {fname}...")
        
        try:
            sep = Separator(
                output_dir=str(out_path),
                model_file_dir=str(self.model_dir),
                output_format="WAV",
                log_level=40
            )
            sep.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            sep.separate(input_path)
            return str(out_path)
        except Exception as e:
            print(f"[ERR] Stems separation failed: {e}")
            return None

    def separate_lead_backing(self, vocals_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Separates Vocals into (Lead, Backing) using Mel-Roformer.
        """
        if not vocals_path or not os.path.exists(vocals_path):
            return None, None
            
        fname = Path(vocals_path).stem
        out_path = Path(output_dir)
        temp_dir = out_path / f"temp_{fname}_s2"
        temp_dir.mkdir(exist_ok=True)

        try:
            if self.verbose: print(f"[SEP] Separating Lead/Backing for {fname}...")
            
            sep = Separator(
                output_dir=str(temp_dir),
                model_file_dir=str(self.model_dir),
                output_format="WAV",
                log_level=40
            )
            
            sep.load_model('model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
            sep.separate(vocals_path)
            
            lead_path = None
            backing_path = None
            
            for f in temp_dir.glob("*.wav"):
                # Mel-Roformer scheme: "(Vocals)" -> Lead, "(Instrumental)" -> Backing
                
                dest = out_path / f"{fname}_{f.name}"
                shutil.move(str(f), str(dest))
                
                if "(Vocals)" in f.name:
                    lead_path = str(dest)
                elif "(Instrumental)" in f.name:
                    backing_path = str(dest)
            
            return lead_path, backing_path

        except Exception as e:
            print(f"[ERR] Lead/Backing separation failed: {e}")
            return None, None
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
