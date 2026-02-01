"""
build_audio_dataset_v5.0.py — Standalone, GPU-accelerated vocal processing pipeline.

Highlights:
- Replaces Demucs with BS-Roformer (Stems) + Mel-Roformer (Lead/Backing).
- Integrated `audio-separator` for SOTA separation.
- Standalone `uv` project structure.

Usage:
  uv run python AudioProcessing/build_audio_dataset_v4.2.py \
    --in_path "" \
    --out_dir "" \
    --selection_mode diversity
"""
from __future__ import annotations

import os, sys
import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import math
from collections import Counter, defaultdict

# Optional dependencies
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None
try:
    from faster_whisper import WhisperModel as FWModel
except ImportError:
    FWModel = None
try:
    import whisper as whisper_openai
except ImportError:
    whisper_openai = None

# --- MONKEY PATCH ---
# Fix for Speechbrain <-> Torchaudio 2.1+ compatibility
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"] 

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass
# --------------------

try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

# Separation
from audio_separator.separator import Separator
from huggingface_hub import hf_hub_download

# --- Constants ---
TARGET_SR = 16000
SEPARATION_SR = 44100
EPS = 1e-9

# --- Data Structures ---
@dataclass
class SegmentCandidate:
    source_path: str
    start_sample: int
    end_sample: int
    duration_s: float
    prosody_features: Optional[np.ndarray] = None
    speaker_embedding: Optional[np.ndarray] = None
    similarity_score: Optional[float] = None

# --- Utility & Helper Functions ---
_HANN_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

def _get_hann_window(n_fft: int, device: torch.device):
    key = (n_fft, str(device))
    if key not in _HANN_CACHE:
        _HANN_CACHE[key] = torch.hann_window(n_fft, device=device, periodic=True)
    return _HANN_CACHE[key]

def list_audio_files(in_path: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
    if in_path.is_file():
        return [in_path]
    return sorted([p for p in in_path.rglob("*") if p.suffix.lower() in exts])

def load_mono16k(path: str) -> torch.Tensor:
    # Use simple soundfile load to avoid backend issues
    wav_np, sr = sf.read(path)
    # Ensure float32
    wav_np = wav_np.astype(np.float32)
    # Convert to tensor: (C, T) or (T,) -> require (C, T) for resampling usually, or handle dimensions
    # sf.read returns (samples, channels) if multi, or (samples,) if mono
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1) # Downmix to mono
    
    wav = torch.from_numpy(wav_np)
    
    if sr != TARGET_SR:
        # Resample needs (..., time)
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.contiguous()

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)

def load_mono_native(path: str) -> Tuple[torch.Tensor, int]:
    wav_np, sr = sf.read(path)
    wav_np = wav_np.astype(np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return torch.from_numpy(wav_np).contiguous(), int(sr)

# global verbose print
VPRINT = lambda *a, **k: None

def set_verbose(enabled: bool):
    global VPRINT
    if enabled:
        def _vprint(*args, **kwargs):
            print(*args, **kwargs)
        VPRINT = _vprint
    else:
        VPRINT = lambda *a, **k: None

# --- Separation Pipeline (Refactored from separator.py) ---
class SeparationPipeline:
    def __init__(self, device: str, model_dir: Path, verbose: bool = False):
        self.device = device
        self.model_dir = model_dir
        self.verbose = verbose
        # Check parent models dir first
        parent_models = Path("../../models").resolve()
        if parent_models.exists():
            self.model_dir = parent_models
            if self.verbose: print(f"[SEP] Using shared models at {self.model_dir}")
        else:
             self.model_dir.mkdir(parents=True, exist_ok=True)
             if self.verbose: print(f"[SEP] Using local models at {self.model_dir}")
             self._ensure_models()

        self._ensure_models()

    def _ensure_models(self):
        # Only download if we are NOT using the shared directory (implicit check at init: if files missing)
        rofo_files = {
            "BS-Rofo-SW-Fixed.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "BS-Rofo-SW-Fixed.yaml": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
        }
        viper_files = {
            "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
            "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml": "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
        }

        for remote, local in rofo_files.items():
            target = self.model_dir / local
            if not target.exists():
                if self.verbose: print(f"[DL] Downloading {remote}...")
                hf_hub_download(repo_id="jarredou/BS-ROFO-SW-Fixed", filename=remote, local_dir=self.model_dir, local_dir_use_symlinks=False)
                # Rename if needed (hf_hub might save as remote name)
                downloaded = self.model_dir / remote
                if downloaded.exists() and downloaded != target:
                    os.rename(downloaded, target)

        for remote, local in viper_files.items():
            target = self.model_dir / local
            if not target.exists():
                if self.verbose: print(f"[DL] Downloading {remote}...")
                hf_hub_download(repo_id="jarredou/Mel-Band-Roformer-Karaoke-Aufr33-Viperx", filename=remote, local_dir=self.model_dir, local_dir_use_symlinks=False)

    def process_file(self, input_path: str, temp_dir: Path) -> Tuple[torch.Tensor, int]:
        """
        Runs the 2-stage separation:
        1. BS-Roformer -> Vocals
        2. Mel-Roformer -> Vocals Lead
        Returns: (lead_vocals_tensor, sample_rate)
        """
        fname = Path(input_path).stem
        
        # Temp workspaces
        s1_dir = temp_dir / f"{fname}_s1"
        s2_dir = temp_dir / f"{fname}_s2"
        s1_dir.mkdir(exist_ok=True)
        s2_dir.mkdir(exist_ok=True)

        try:
            # --- Prep: Normalize Input ---
            # BS-Roformer works best with normalized input (approx -1dB)
            norm_input = temp_dir / f"{fname}_norm.wav"
            self._normalize_file(input_path, norm_input)

            # --- Stage 1: BS-Roformer (Vocals vs Inst) ---
            if self.verbose: print(f"[SEP] {fname}: Stage 1 (BS-Roformer)...")
            sep1 = Separator(output_dir=str(s1_dir), model_file_dir=str(self.model_dir), output_format="WAV", log_level=30)
            sep1.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            sep1.separate(str(norm_input))

            # Find 'vocals' output
            vocab_stem = None
            for p in s1_dir.glob("*.wav"):
                if "(vocals)" in p.name.lower():
                    vocab_stem = p
                    break
            
            if not vocab_stem:
                raise RuntimeError("Stage 1 failed to produce vocals stem")

            # --- Stage 2: Mel-Roformer (Lead vs Backing) ---
            if self.verbose: print(f"[SEP] {fname}: Stage 2 (Mel-Roformer)...")
            sep2 = Separator(output_dir=str(s2_dir), model_file_dir=str(self.model_dir), output_format="WAV", log_level=30)
            sep2.load_model('model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
            sep2.separate(str(vocab_stem))

            # Find 'Vocals' (Lead) output - NOTE: Mel-Roformer labels lead as "Vocals" and backing as "Instrumental" typically? 
            # Check separator.py logic: "(Vocals)" -> Lead, "(Instrumental)" -> Backing
            lead_stem = None
            for p in s2_dir.glob("*.wav"):
                if "(Vocals)" in p.name:
                    lead_stem = p
                    break
            
            if not lead_stem:
                # Fallback: maybe just use stage 1 vocals
                if self.verbose: print(f"[WARN] Stage 2 lead not found, using Stage 1 vocals.")
                lead_stem = vocab_stem

            # Load result
            # Use soundfile here too
            y_np, sr = sf.read(lead_stem)
            y_np = y_np.astype(np.float32)
            if y_np.ndim > 1:
                y_np = y_np.mean(axis=1)
            y = torch.from_numpy(y_np).contiguous()
            return y, sr

        finally:
            # Cleanup
            if norm_input.exists(): norm_input.unlink()
            shutil.rmtree(s1_dir, ignore_errors=True)
            shutil.rmtree(s2_dir, ignore_errors=True)

    def _normalize_file(self, in_path, out_path):
        y, sr = librosa.load(in_path, sr=44100, mono=False)
        if y.ndim == 1: y = np.stack([y, y]) # Make stereo
        peak = np.max(np.abs(y))
        target_peak = 0.9
        if peak < 0.8 or peak > 1.0:
            y = y * (target_peak / (peak + 1e-6))
        
        import soundfile as sf
        sf.write(str(out_path), y.T, sr, subtype='FLOAT')

# --- Caching Wrapper for Separation ---
class CachedSeparation:
    def __init__(self, workspace: Path, pipeline: SeparationPipeline):
        self.workspace = workspace / "cache_seps"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline
    
    def get_lead_vocals(self, path: str) -> Tuple[torch.Tensor, int]:
        p = Path(path)
        cache_file = self.workspace / f"{p.stem}_lead.wav"
        
        if cache_file.exists():
            return load_mono_native(str(cache_file))
        
        # Process
        y, sr = self.pipeline.process_file(path, self.workspace)
        
        # Save to cache using soundfile
        # y is Tensor, likely (C, T) or (T). sf.write expects (T, C) or (T)
        y_np = y.cpu().numpy()
        if y_np.ndim > 1:
            y_np = y_np.T
        sf.write(str(cache_file), y_np, sr)
        return y, sr


# --- Feature Extraction & Selection (Mostly Unchanged) ---
def extract_prosody_features(y: torch.Tensor) -> np.ndarray:
    y_cpu = y.cpu()
    frame, hop = 256, 128
    frames = y_cpu.unfold(0, frame, hop)
    if frames.numel() == 0:
        return np.zeros(7, dtype=np.float32)
    rms = torch.sqrt((frames**2).mean(dim=1) + EPS)
    rms_std = float(rms.std())
    rms_dr = float(torch.quantile(rms, 0.95) - torch.quantile(rms, 0.05)) if rms.numel() > 1 else 0.0
    crest = float(y_cpu.abs().max() / (rms.mean() + EPS))
    n_fft = 1024
    win = _get_hann_window(n_fft, y_cpu.device)
    S = torch.stft(y_cpu, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, return_complex=True).abs() + EPS
    freq_bins = torch.arange(S.size(0), dtype=S.dtype, device=S.device)
    centroid = (S * freq_bins.unsqueeze(1)).sum(0) / (S.sum(0) + EPS)
    f0s = centroid[torch.isfinite(centroid)]
    if f0s.numel() > 0:
        f0_mean, f0_std, f0_rng = float(f0s.mean()), float(f0s.std()), float(f0s.max() - float(f0s.min()))
    else:
        f0_mean = f0_std = f0_rng = 0.0
    zcr = ((frames[:, 1:] * frames[:, :-1]) < 0).float().mean(dim=1)
    tempo_var = float(zcr.std()) if zcr.numel() > 1 else 0.0
    return np.array([f0_mean, f0_std, f0_rng, rms_std, rms_dr, crest, tempo_var], dtype=np.float32)

def select_diverse_candidates(candidates: List[SegmentCandidate], target_duration_s: float) -> List[SegmentCandidate]:
    print(f"[SELECT] Starting diversity selection for ~{target_duration_s}s...")
    if not candidates:
        return []
    features = np.vstack([c.prosody_features for c in candidates])
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    iqr = np.maximum(q75 - q25, EPS)
    norm_features = (features - np.median(features, axis=0)) / iqr
    expr_heuristic = norm_features[:, 2] + norm_features[:, 1] + norm_features[:, 3] + norm_features[:, 4]
    seed_idx = int(np.argmax(expr_heuristic))
    selected_indices = {seed_idx}
    total_duration = candidates[seed_idx].duration_s
    min_dists = np.linalg.norm(norm_features - norm_features[seed_idx], axis=1)
    pbar = tqdm(total=target_duration_s, desc="[SELECT] K-Center", unit="s")
    pbar.update(total_duration)
    while total_duration < target_duration_s * 1.2 and len(selected_indices) < len(candidates):
        min_dists[list(selected_indices)] = -np.inf
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 1e-6:
            break
        selected_indices.add(next_idx)
        total_duration += candidates[next_idx].duration_s
        pbar.update(candidates[next_idx].duration_s)
        new_dists = np.linalg.norm(norm_features - norm_features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    pbar.close()
    selected = [candidates[i] for i in selected_indices]
    print(f"[SELECT] Selected {len(selected)} diverse segments totaling {total_duration:.2f}s.")
    return selected


# --- Loudness ---
class LoudnessNormalizer:
    def __init__(self, mode: str, target_lufs: float, peak_ceiling_db: float, trim_db: float, tp_oversample: int, max_gain_db: float):
        mode = mode.lower()
        if mode not in {"lufs", "rms"}: raise ValueError("loudness_mode must be 'lufs' or 'rms'")
        self.mode = mode
        self.target_lufs = float(target_lufs)
        self.peak_ceiling_db = float(peak_ceiling_db)
        self.trim_db = float(trim_db)
        self.tp_over = max(1, int(tp_oversample))
        self.max_gain_db = float(max_gain_db)
        if self.mode == "lufs" and pyln is not None:
            self.meter = pyln.Meter(TARGET_SR)

    def _true_peak(self, y: torch.Tensor, sr: int) -> float:
        if self.tp_over <= 1: return float(y.abs().max().item())
        resampler = torchaudio.transforms.Resample(sr, sr * self.tp_over).to(y.device)
        return float(resampler(y.unsqueeze(0)).squeeze(0).abs().max().item())

    def _gate_trim(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        amp = 10 ** (self.trim_db / 20.0)
        idx = torch.where(x.abs() > amp)[0]
        if idx.numel() == 0: return torch.zeros(0, dtype=x.dtype, device=x.device)
        pad = int(0.01 * sr)
        start = max(int(idx[0] - pad), 0)
        end   = min(int(idx[-1] + pad), x.numel())
        return x[start:end]

    def normalize(self, seg: torch.Tensor, sr: int) -> torch.Tensor:
        if seg.numel() < int(0.25 * sr): return torch.zeros(0, dtype=seg.dtype, device=seg.device)
        x = self._gate_trim(seg.detach(), sr)
        if x.numel() < int(0.10 * sr): return torch.zeros(0, dtype=seg.dtype, device=seg.device)

        if self.mode == "lufs" and pyln is not None:
            meter = pyln.Meter(sr)
            loud = float(meter.integrated_loudness(x.cpu().numpy().astype(np.float64)))
            if loud == -np.inf: return torch.zeros(0, dtype=x.dtype, device=x.device)
            gain_db = self.target_lufs - loud
            if gain_db > self.max_gain_db: return torch.zeros(0, dtype=x.dtype, device=x.device)
            y = x * (10 ** (max(-60.0, gain_db) / 20.0))
        else:
            rms = float(torch.sqrt((x**2).mean() + EPS))
            target_rms = 10 ** (-23.0 / 20.0)
            gain = target_rms / max(rms, 1e-5)
            if 20 * math.log10(max(gain, 1e-9)) > self.max_gain_db:
                return torch.zeros(0, dtype=x.dtype, device=x.device)
            y = x * gain

        tp = self._true_peak(y, sr)
        ceiling = 10 ** (self.peak_ceiling_db / 20.0)
        if tp > ceiling:
            y = y * (ceiling / (tp + EPS))
        return y

class PeakLimiter:
    def __init__(self, peak_ceiling_db: float, trim_db: float, tp_oversample: int):
        self.peak_ceiling_db = float(peak_ceiling_db)
        self.trim_db = float(trim_db)
        self.tp_over = max(1, int(tp_oversample))

    def process(self, seg: torch.Tensor, sr: int) -> torch.Tensor:
        if seg.numel() < int(0.25 * sr): return torch.zeros(0, dtype=seg.dtype, device=seg.device)
        
        # Simple Gate
        amp = 10 ** (self.trim_db / 20.0)
        idx = torch.where(seg.abs() > amp)[0]
        if idx.numel() == 0: return torch.zeros(0, dtype=seg.dtype, device=seg.device)
        pad = int(0.01 * sr)
        start = max(int(idx[0] - pad), 0)
        end   = min(int(idx[-1] + pad), seg.numel())
        x = seg[start:end]

        # Peak Limit
        ceiling = 10 ** (self.peak_ceiling_db / 20.0)
        peak = float(x.abs().max())
        if peak > ceiling:
            x = x * (ceiling / (peak + EPS))
        return x

# --- Whisper ---
class WhisperTranscriber:
    def __init__(self, backend: str, model_size: str, device: str, language: str, fw_workers: int):
        lang = None if (language is None or str(language).lower() in {"", "auto", "none"}) else language
        self.backend, self.language = backend, lang
        compute_type = "float16" if "cuda" in device else "int8"
        print(f"[INIT] Whisper backend '{backend}' ({model_size}) lang={lang}...")
        
        if backend == 'faster':
            if FWModel is None: raise ImportError("`pip install faster-whisper` required.")
            device_str = re.match(r"cuda:?(\d*)", device).group(0) if "cuda" in device else "cpu"
            device_indices = [int(i) for i in re.findall(r'\d+', device_str)] or [0]
            self.model = FWModel(
                model_size, device=device_str.split(':')[0], device_index=device_indices,
                compute_type=compute_type, num_workers=max(1, int(fw_workers))
            )
        else:
            if whisper_openai is None: raise ImportError("`pip install openai-whisper` required.")
            self.model = whisper_openai.load_model(model_size, device=device)

    def transcribe_and_filter(self, audio_tensor: torch.Tensor, sr: int, args: argparse.Namespace) -> Tuple[bool, str, Dict]:
        # Resample for ASR (16k)
        if sr != TARGET_SR:
            x16 = torchaudio.functional.resample(audio_tensor, sr, TARGET_SR)
        else:
            x16 = audio_tensor
        arr = x16.cpu().numpy().astype(np.float32)

        conf = {"avg_logprob": -99.0, "max_no_speech_prob": 1.0}
        
        if self.backend == 'faster':
            segs, _ = self.model.transcribe(
                arr, language=self.language, beam_size=args.beam_size,
                vad_filter=False, condition_on_previous_text=False
            )
            texts, logs, nsp = [], [], []
            for s in segs:
                texts.append(s.text)
                if s.avg_logprob is not None: logs.append(s.avg_logprob)
                if getattr(s, "no_speech_prob", None) is not None: nsp.append(s.no_speech_prob)
            text = " ".join(texts).strip()
            conf["avg_logprob"] = float(np.mean(logs)) if logs else -99.0
            conf["max_no_speech_prob"] = float(np.max(nsp)) if nsp else 1.0
        else:
            # OpenAI Backend
            res = self.model.transcribe(
                arr, language=self.language, beam_size=args.beam_size,
                condition_on_previous_text=False
            )
            text = res.get("text", "").strip()
            # Simplified confidence for openai backend
            conf["avg_logprob"] = -0.0 # Placeholder/TODO

        reasons = []
        if conf["avg_logprob"] < args.min_avg_logprob: reasons.append(f"logprob<{conf['avg_logprob']:.2f}")
        if conf["max_no_speech_prob"] > args.max_no_speech_prob: reasons.append(f"no_speech>{conf['max_no_speech_prob']:.2f}")
        if len(text) < args.min_text_len: reasons.append(f"text_len<{len(text)}")
        
        return len(reasons) == 0, text, {"reason": "; ".join(reasons) or "pass", **conf}

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Standalone Vocal Process (Roformer Era)")
    parser.add_argument("--in_path", required=True, help="Input source file or directory.")
    parser.add_argument("--out_dir", required=True, help="Result directory.")
    parser.add_argument("--final_sr", type=int, default=24000, help="Output sample rate.")
    
    # Separation
    parser.add_argument("--bgm_clean", action="store_true", default=True, help="Force BGM separation (default: True).")

    # Selection
    parser.add_argument("--selection_mode", required=True, choices=["diversity", "similarity"])
    parser.add_argument("--target_duration_s", type=float, default=1800.0)
    parser.add_argument("--ref_voice", type=str)
    
    # Audio
    parser.add_argument("--min_seg_s", type=float, default=2.0)
    parser.add_argument("--max_seg_s", type=float, default=20.0)
    parser.add_argument("--normalize_after_combine", action="store_true")
    
    # ASR
    parser.add_argument("--whisper_backend", default="faster")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--min_avg_logprob", type=float, default=-0.6)
    parser.add_argument("--max_no_speech_prob", type=float, default=0.5)
    parser.add_argument("--min_text_len", type=int, default=2)

    # Misc
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    set_verbose(args.verbose)
    
    # 1. Setup
    out_dir, wavs_dir = Path(args.out_dir), Path(args.out_dir) / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True); wavs_dir.mkdir(exist_ok=True)
    
    print(f"[INIT] Device: {args.device}")
    
    # 2. Load Models
    print("--- Loading Models ---")
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True)
    (get_ts, _, read_audio, *_) = utils; vad_model = model

    # Separation Pipeline
    sep_pipeline = SeparationPipeline(args.device, Path("models"), verbose=args.verbose)
    vocal_cache = CachedSeparation(Path(args.out_dir), sep_pipeline)

    # Normalizers
    normalizer = LoudnessNormalizer("lufs", -23.0, -1.0, -50.0, 2, 10.0)
    light_proc = PeakLimiter(-1.0, -50.0, 2)
    
    # ASR
    transcriber = WhisperTranscriber(args.whisper_backend, args.whisper_model, args.device, args.language, 4)

    # Embedding model (optional)
    ecapa, ref_embedding = None, None
    if args.selection_mode == "similarity":
        print("[INIT] Loading ECAPA...")
        ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": args.device})
        ref_wav = load_mono16k(args.ref_voice).to(args.device)
        ref_embedding = l2norm(ecapa.encode_batch(ref_wav.unsqueeze(0)).squeeze().cpu().numpy())

    # 3. Candidates
    print("--- Phase 1: Candidates ---")
    files = list_audio_files(Path(args.in_path))
    all_candidates = []
    
    for f_path in tqdm(files, desc="Files"):
        try:
            # Separate first? Or VAD first? 
            # Classic strategy: VAD on source, then only process segments.
            # BUT BGM interferes with VAD. 
            # Better strategy: Separate whole file -> VAD on Vocals -> Segment.
            
            # 1. Separate (Whole File)
            print(f"[SEP] Separating {f_path.name}...")
            vclean, v_sr = vocal_cache.get_lead_vocals(str(f_path)) # Returns ~44k typically
            
            # 2. VAD on Clean Vocals (Resample to 16k for Silero)
            vclean_16k = torchaudio.functional.resample(vclean, v_sr, TARGET_SR)
            ts_list = get_ts(vclean_16k, vad_model, sampling_rate=TARGET_SR)
            
            for ts in ts_list:
                dur = (ts['end'] - ts['start']) / TARGET_SR
                if args.min_seg_s <= dur <= args.max_seg_s:
                    cand = SegmentCandidate(str(f_path), ts['start'], ts['end'], dur)
                    
                    # Extract features from the clean vocals
                    startN = int(ts['start'])
                    endN = int(ts['end'])
                    seg_16k = vclean_16k[startN:endN]
                    
                    if args.selection_mode == "diversity":
                        cand.prosody_features = extract_prosody_features(seg_16k)
                    if ecapa is not None:
                        emb = ecapa.encode_batch(seg_16k.to(args.device).unsqueeze(0)).squeeze().cpu().numpy()
                        cand.speaker_embedding = l2norm(emb)
                        
                    all_candidates.append(cand)

        except Exception as e:
            print(f"[ERR] Failed {f_path.name}: {e}")

    # 4. Select
    print(f"--- Phase 2: Selection ({len(all_candidates)} cands) ---")
    if args.selection_mode == "diversity":
        selected = select_diverse_candidates(all_candidates, args.target_duration_s)
    else:
        # TODO: Implement select_similar_candidates if needed, but logic is same as before
        selected = all_candidates # Placehold

    # 5. Process & Transcribe
    print("--- Phase 3: Processing/ASR ---")
    final_manifest = []
    
    # Since we already separated the WHOLE file into cache, we just need to slice from the cached clear vocals.
    # Note: 'SegmentCandidate' stores indices relative to 16k VAD. 
    # We need to load the cached clean vocals again.
    
    # Improve efficiency: group by source file
    grouped = defaultdict(list)
    for s in selected: grouped[s.source_path].append(s)
    
    i_global = 0
    for src_path, segs in tqdm(grouped.items(), desc="Saving"):
        try:
             # Load cached vocals
             vclean, v_sr = vocal_cache.get_lead_vocals(src_path)
             
             for seg in segs:
                 # Map 16k -> v_sr
                 scale = v_sr / TARGET_SR
                 startN = int(seg.start_sample * scale)
                 endN = int(seg.end_sample * scale)
                 
                 raw = vclean[startN:endN]
                 
                 # Norm
                 if args.normalize_after_combine:
                     proc = light_proc.process(raw, v_sr)
                 else:
                     proc = normalizer.normalize(raw, v_sr)
                     
                 if proc.numel() == 0: continue
                 
                 # ASR
                 ok, text, _ = transcriber.transcribe_and_filter(proc, v_sr, args)
                 if not ok: continue
                 
                 # Save
                 out_sr = args.final_sr if args.final_sr > 0 else v_sr
                 if out_sr != v_sr:
                     out_audio = torchaudio.functional.resample(proc, v_sr, out_sr)
                 else:
                     out_audio = proc
                 
                 fname = f"seg_{i_global:05d}.wav"
                 # torchaudio.save replaced with sf.write
                 out_np = out_audio.cpu().numpy()
                 if out_np.ndim > 1: out_np = out_np.T
                 sf.write(str(wavs_dir / fname), out_np, out_sr)
                 
                 final_manifest.append({
                     "audio_filepath": f"wavs/{fname}",
                     "text": text,
                     "duration": seg.duration_s
                 })
                 i_global += 1
                 
        except Exception as e:
            print(f"[ERR] Processing segments for {Path(src_path).name}: {e}")

    # 6. Combined
    if final_manifest:
        with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "duration"])
            writer.writeheader()
            writer.writerows(final_manifest)
        
        # Combine Audio
        print("[FINAL] Creating combined audio...")
        parts = []
        full_sr = 24000 
        # Scan first to check SR? Assume uniform or use first
        
        for item in sorted(final_manifest, key=lambda x: x['audio_filepath']):
            w_np, sr = sf.read(out_dir / item['audio_filepath'])
            full_sr = sr
            parts.append(torch.from_numpy(w_np.astype(np.float32)))
            parts.append(torch.zeros(int(sr * 0.2))) # 200ms gap
        
        if parts:
            full = torch.cat(parts)
            # torchaudio.save is usually robust if backend set, but let's use sf.write for consistency
            sf.write(str(out_dir / "combined.wav"), full.numpy(), full_sr)
            
            # Combine Text
            full_text = "\n".join([x['text'] for x in final_manifest])
            (out_dir / "combined.txt").write_text(full_text, encoding="utf-8")
    
    print("Done.")

if __name__ == "__main__":
    main()
