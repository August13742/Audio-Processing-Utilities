
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
        Separates audio into (Vocals, Instrumental) using BS-Roformer 6-stem model.
        Individual stems are recombined into a single instrumental mix.
        Output: songname_vocals.wav, songname_instrumental.wav
        Returns paths to (vocals_path, instrumental_path).
        """
        fname = Path(input_path).stem
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        temp_dir = out_path / f"temp_{fname}_s1"
        temp_dir.mkdir(exist_ok=True)

        try:
            if self.verbose: print(f"[SEP] Separating Vocals/Inst for {fname}...")

            sep = Separator(
                output_dir=str(temp_dir),
                model_file_dir=str(self.model_dir),
                output_format="WAV",
                log_level=40
            )
            sep.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            sep.separate(input_path)

            # Map separator outputs to clean songname_stemName.wav
            stem_keywords = ["vocals", "drums", "bass", "other", "guitar", "piano"]
            stem_files = {}

            for f in temp_dir.glob("*.wav"):
                name_lower = f.name.lower()
                for stem in stem_keywords:
                    if f"({stem})" in name_lower:
                        dest = out_path / f"{fname}_{stem}.wav"
                        shutil.move(str(f), str(dest))
                        stem_files[stem] = dest
                        break

            vocals_path = str(stem_files["vocals"]) if "vocals" in stem_files else None

            # Combine non-vocal stems into a single instrumental track
            bgm_stems = [p for k, p in stem_files.items() if k != "vocals"]
            inst_path = None

            if bgm_stems:
                if self.verbose: print(f"  [MIX] Combining {len(bgm_stems)} stems for instrumental...")
                y_ref, sr = load_audio(str(bgm_stems[0]))
                y_sum = y_ref.clone()

                for p in bgm_stems[1:]:
                    y_next, _ = load_audio(str(p), target_sr=sr)
                    min_len = min(y_next.shape[-1], y_sum.shape[-1])
                    y_sum[..., :min_len] += y_next[..., :min_len]
                    y_sum = y_sum[..., :min_len]

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

    def separate_karaoke(self, input_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        2-stem vocal/instrumental separation using Mel-Band Roformer (karaoke-optimized).
        Faster than 6-stem BS-Roformer and purpose-built for karaoke isolation.
        Output: songname_vocals.wav, songname_instrumental.wav
        Returns (vocals_path, instrumental_path).
        """
        fname = Path(input_path).stem
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        temp_dir = out_path / f"temp_{fname}_karaoke"
        temp_dir.mkdir(exist_ok=True)

        try:
            if self.verbose: print(f"[SEP] Karaoke 2-stem separation for {fname}...")

            sep = Separator(
                output_dir=str(temp_dir),
                model_file_dir=str(self.model_dir),
                output_format="WAV",
                log_level=40
            )
            sep.load_model('model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
            sep.separate(input_path)

            vocals_path = None
            inst_path = None

            for f in temp_dir.glob("*.wav"):
                name_lower = f.name.lower()
                if "(vocals)" in name_lower:
                    dest = out_path / f"{fname}_vocals.wav"
                    shutil.move(str(f), str(dest))
                    vocals_path = str(dest)
                elif "(instrumental)" in name_lower:
                    dest = out_path / f"{fname}_instrumental.wav"
                    shutil.move(str(f), str(dest))
                    inst_path = str(dest)

            return vocals_path, inst_path

        except Exception as e:
            print(f"[ERR] Karaoke separation failed: {e}")
            return None, None
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def separate_lead_backing(self, vocals_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Separates Vocals into (Lead, Backing) using Mel-Roformer.
        Output: songname_lead.wav, songname_backing.wav
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
                name_lower = f.name.lower()
                if "(vocals)" in name_lower:
                    dest = out_path / f"{fname}_lead.wav"
                    shutil.move(str(f), str(dest))
                    lead_path = str(dest)
                elif "(instrumental)" in name_lower:
                    dest = out_path / f"{fname}_backing.wav"
                    shutil.move(str(f), str(dest))
                    backing_path = str(dest)

            return lead_path, backing_path

        except Exception as e:
            print(f"[ERR] Lead/Backing separation failed: {e}")
            return None, None
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
