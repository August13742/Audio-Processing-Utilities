
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from utils.audio import load_audio, save_audio

class Segmenter:
    def __init__(self, method: str = "silence", device: str = "cpu"):
        self.method = method
        self.device = device
        
        if method == "silence":
            # Load Silero VAD
            try:
                model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True)
                self.vad_model = model.to(device)
                self.get_ts = utils[0]
            except Exception as e:
                print(f"[ERR] Failed to load VAD: {e}")
                self.vad_model = None

    def segment_file(self, input_path: str, output_dir: str, min_dur: float = 2.0, max_dur: float = 15.0):
        fname = Path(input_path).stem
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load Audio (16k for VAD)
            y, sr = load_audio(input_path, target_sr=16000, mono=True)
            y = y.squeeze(0).to(self.device)
            
            if self.vad_model:
                ts_list = self.get_ts(y, self.vad_model, sampling_rate=16000)
                
                # Load high-quality for export
                y_hq, sr_hq = load_audio(input_path) 
                
                for i, ts in enumerate(ts_list):
                    start_s = ts['start'] / 16000
                    end_s = ts['end'] / 16000
                    dur = end_s - start_s
                    
                    if min_dur <= dur <= max_dur:
                        start_sample = int(start_s * sr_hq)
                        end_sample = int(end_s * sr_hq)
                        
                        chunk = y_hq[..., start_sample:end_sample]
                        
                        chunk_name = f"{fname}_seg{i:04d}.wav"
                        save_audio(str(out_path / chunk_name), chunk, sr_hq)
            else:
                print("[ERR] VAD model not loaded.")
                
        except Exception as e:
            print(f"[ERR] Segmentation failed for {input_path}: {e}")
