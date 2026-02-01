
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

def ensure_models(model_dir: Path, verbose: bool = False):
    """
    Ensures that the required BS-Roformer and Mel-Roformer models are present.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model definitions
    rofo_files = {
        "BS-Rofo-SW-Fixed.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-Rofo-SW-Fixed.yaml": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
    }
    
    viper_files = {
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml": "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
    }

    if verbose:
        print(f"[MODELS] Checking models in {model_dir}...")

    # Download BS-Roformer
    for remote, local in rofo_files.items():
        target = model_dir / local
        if not target.exists():
            try:
                if verbose: print(f"  [DL] Downloading {remote}...")
                hf_hub_download(repo_id="jarredou/BS-ROFO-SW-Fixed", filename=remote, local_dir=model_dir, local_dir_use_symlinks=False)
                
                # Handle renaming if HF doesn't save to the exact local name we want
                downloaded = model_dir / remote
                if downloaded.exists() and downloaded != target:
                    os.rename(downloaded, target)
            except Exception as e:
                print(f"  [ERR] Failed to download {remote}: {e}")

    # Download Viperx
    for remote, local in viper_files.items():
        target = model_dir / local
        if not target.exists():
            try:
                if verbose: print(f"  [DL] Downloading {remote}...")
                hf_hub_download(repo_id="jarredou/Mel-Band-Roformer-Karaoke-Aufr33-Viperx", filename=remote, local_dir=model_dir, local_dir_use_symlinks=False)
                # Viperx repo usually saves as the long filename, so less renamed needed, but good to check
            except Exception as e:
                print(f"  [ERR] Failed to download {remote}: {e}")
