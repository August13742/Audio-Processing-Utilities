
import subprocess
import torch
import numpy as np
import soundfile as sf
import torchaudio
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from tools.separation import SeparationEngine
from tools.transcription import TranscriptionEngine
from utils.audio import load_audio, save_audio, l2norm
from utils.processing import PeakLimiter

try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    EncoderClassifier = None

class SegmentCandidate:
    def __init__(self, source_path: str, start: float, end: float, duration: float):
        self.source_path = source_path
        self.start = start
        self.end = end
        self.duration = duration
        self.embedding = None

class DatasetBuilder:
    def __init__(self, output_dir: str, device: str = "cuda"):
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(exist_ok=True)
        
        self.sep_engine = SeparationEngine(device=device)
        self.asr_engine = TranscriptionEngine(device=device)
        self.vad_model = None
        self.ecapa = None
        
    def _init_vad(self):
        if self.vad_model is None:
            print("[INIT] Loading Silero VAD...")
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True)
            self.get_ts = utils[0]
            self.vad_model = model

    def _init_ecapa(self):
        if self.ecapa is None and EncoderClassifier:
            print("[INIT] Loading ECAPA-TDNN...")
            self.ecapa = EncoderClassifier.from_hparams(
                "speechbrain/spkrec-ecapa-voxceleb", 
                run_opts={"device": self.device}
            )

    def process_dataset(self, input_dir: str, target_duration: float = 1800.0):
        """
        Full pipeline: Separate -> VAD -> Slice -> Select -> Transcribe -> Save
        """
        self._init_vad()
        
        input_path = Path(input_dir)
        files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.mp3")) + list(input_path.rglob("*.flac"))
        
        candidates = []
        
        print(f"[DATASET] Scanning {len(files)} files...")
        
        # Phase 1: Separation & VAD (Candidate Generation)
        for f in tqdm(files, desc="Scanning"):
            try:
                # Separate Vocals (Lead)
                # We use a temp cache for separation to avoid re-doing it.
                # Separation handles temp files internally but returns the final path.
                
                # Check if we have it cached in the output directory
                cache_dir = self.output_dir / "cache"
                cache_dir.mkdir(exist_ok=True)
                
                # Run separation to extract lead vocals
                
                vocals_path, _ = self.sep_engine.separate_vocals(str(f), str(cache_dir))
                if not vocals_path: continue
                
                lead_path, _ = self.sep_engine.separate_lead_backing(vocals_path, str(cache_dir))
                if not lead_path: lead_path = vocals_path # Fallback
                
                # 2. VAD
                # Silero expects 16k
                y, sr = load_audio(lead_path, target_sr=16000) # (1, T)
                y = y.squeeze(0) # (T)
                
                ts_list = self.get_ts(y, self.vad_model, sampling_rate=16000)
                
                for ts in ts_list:
                    start_s = ts['start'] / 16000
                    end_s = ts['end'] / 16000
                    dur = end_s - start_s
                    
                    if 2.0 <= dur <= 15.0:
                        cand = SegmentCandidate(str(lead_path), start_s, end_s, dur)
                        candidates.append(cand)
                        
            except Exception as e:
                print(f"[ERR] Failed to process {f.name}: {e}")

        # Phase 2: Selection
        # Implementing basic selection for now
        print(f"[DATASET] Found {len(candidates)} valid segments.")
        
        # Slice and Transcribe
        manifest = []
        
        limit_proc = PeakLimiter()
        
        print("[DATASET] Transcribing & Saving...")
        for i, cand in enumerate(tqdm(candidates[:])):
            try:
                # Load segment from LEAD path
                # Load at native rate for quality
                y, sr = load_audio(cand.source_path) # (Channel, Time)
                
                start_sample = int(cand.start * sr)
                end_sample = int(cand.end * sr)
                
                seg = y[..., start_sample:end_sample]
                
                # Peak Limit
                seg = limit_proc.process(seg.squeeze(0), sr).unsqueeze(0)
                
                if seg.abs().max() < 0.01: continue
                
                # Save
                fname = f"seg_{i:05d}.wav"
                out_path = self.wavs_dir / fname
                save_audio(str(out_path), seg, sr)
                
                # Transcribe
                text, idx = self.asr_engine.transcribe(str(out_path))
                
                if len(text) > 1:
                    manifest.append({
                        "file": str(out_path.name),
                        "text": text,
                        "duration": cand.duration
                    })
                    
            except Exception as e:
                pass

        # Save manifest
        import csv
        with open(self.output_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "text", "duration"])
            writer.writeheader()
            writer.writerows(manifest)
            
        print("[DONE] Dataset built.")

