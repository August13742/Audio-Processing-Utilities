#!/usr/bin/env python3
"""
Hybrid WhisperX Alignment Test on Kiritan Singing Database

Strategy: Use our TranscriptionEngine (faster-whisper with karaoke-tuned
params) for transcription, then feed those segments to WhisperX's
phoneme-aware forced alignment model.

This isolates the alignment quality question from WhisperX's broken
transcription on karaoke audio.

Requirements:
    - whisperx, faster-whisper, soundfile, numpy, torch

Usage:
    uv run python tests/test_whisperx_alignment.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import json
import numpy as np
import torch
import soundfile as sf

from tools.transcription import TranscriptionEngine

KIRITAN_DIR = Path("TestData/kiritan_singing")
WAV_DIR = KIRITAN_DIR / "wav"
LABEL_DIR = KIRITAN_DIR / "mono_label"

SAMPLE_IDS = ["01", "02", "03", "04", "05"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_phoneme_labels(lab_path: Path) -> list[dict]:
    """Parse kiritan .lab file → list of {start, end, phoneme}."""
    entries = []
    for line in lab_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            entries.append({
                "start": float(parts[0]),
                "end": float(parts[1]),
                "phoneme": parts[2],
            })
    return entries


def compute_boundary_errors(pred_words: list[dict], gt_phonemes: list[dict]) -> dict:
    if not pred_words or not gt_phonemes:
        return {"start_errors": [], "end_errors": [], "all_errors": []}
    gt_boundaries = sorted(set(
        [p["start"] for p in gt_phonemes] + [p["end"] for p in gt_phonemes]
    ))
    gt_arr = np.array(gt_boundaries)
    start_errors = []
    end_errors = []
    for w in pred_words:
        start_errors.append(float(np.min(np.abs(gt_arr - w["start"]))))
        end_errors.append(float(np.min(np.abs(gt_arr - w["end"]))))
    return {
        "start_errors": start_errors,
        "end_errors": end_errors,
        "all_errors": start_errors + end_errors,
    }


def alignment_report(errors: list[float], label: str) -> dict:
    if not errors:
        return {}
    errs = np.array(errors)
    report = {
        "label": label,
        "n": len(errs),
        "mean_ms": round(errs.mean() * 1000, 1),
        "median_ms": round(np.median(errs) * 1000, 1),
        "p90_ms": round(np.percentile(errs, 90) * 1000, 1),
    }
    for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
        report[f"within_{int(t*1000)}ms"] = round((errs <= t).mean() * 100, 1)
    return report


def main():
    print("=" * 70)
    print("HYBRID TEST: Our Whisper transcription → WhisperX alignment")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Samples: {SAMPLE_IDS}\n")

    # ── 1. Load models ───────────────────────────────────────────────
    import whisperx

    print("Loading TranscriptionEngine (faster-whisper, karaoke config)...")
    whisper_engine = TranscriptionEngine(model_size="large-v3", device=DEVICE)

    print("Loading WhisperX alignment model (Japanese)...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ja", device=DEVICE
    )
    print("✓ All models loaded.\n")

    # ── 2. Load ground truth ─────────────────────────────────────────
    gt_data = {}
    for sid in SAMPLE_IDS:
        gt_data[sid] = parse_phoneme_labels(LABEL_DIR / f"{sid}.lab")

    # ── 3. Process each sample: Whisper transcribe → WhisperX align ──
    whisper_all = []      # Whisper-only word boundary errors
    hybrid_all = []       # Hybrid (Whisper text → WhisperX align) errors

    print("=" * 70)
    print("Processing samples...")
    print("=" * 70)

    for sid in SAMPLE_IDS:
        wav_path = str(WAV_DIR / f"{sid}.wav")
        gt_ph = gt_data[sid]

        print(f"\n[{sid}] Transcribing with our TranscriptionEngine...")
        text, meta = whisper_engine.transcribe(wav_path)

        # ── A) Whisper-only word timestamps (baseline) ───────────────
        whisper_words = [
            {"text": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for w in meta.get("timestamps", [])
        ]
        w_errs = compute_boundary_errors(whisper_words, gt_ph)
        whisper_all.extend(w_errs["all_errors"])

        # ── B) Hybrid: feed Whisper segments → WhisperX alignment ────
        # Load audio as float32 numpy (what WhisperX expects)
        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]

        # Convert our segment format to what WhisperX expects
        wx_segments = [
            {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
            for seg in meta.get("segments", [])
        ]

        # Run WhisperX forced alignment on OUR transcription
        aligned = whisperx.align(
            wx_segments,
            align_model,
            align_metadata,
            audio_np,
            device=DEVICE,
            return_char_alignments=False,
        )

        hybrid_words = []
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                if "start" in w and "end" in w:  # skip unaligned words
                    hybrid_words.append({
                        "text": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })

        h_errs = compute_boundary_errors(hybrid_words, gt_ph)
        hybrid_all.extend(h_errs["all_errors"])

        w_med = np.median(w_errs["all_errors"]) * 1000 if w_errs["all_errors"] else 0
        h_med = np.median(h_errs["all_errors"]) * 1000 if h_errs["all_errors"] else 0
        delta = h_med - w_med

        print(f"  Text: {text[:70]}...")
        print(f"  Whisper words:  {len(whisper_words):>4}  |  median err: {w_med:>7.1f}ms")
        print(f"  Hybrid words:   {len(hybrid_words):>4}  |  median err: {h_med:>7.1f}ms"
              f"  ({delta:+.1f}ms)")

    # ── 4. Final comparison ──────────────────────────────────────────
    w_report = alignment_report(whisper_all, "Whisper-only")
    h_report = alignment_report(hybrid_all, "Hybrid (Whisper+WhisperX)")

    print("\n" + "=" * 70)
    print("RESULTS: Whisper-only vs Hybrid (Whisper text → WhisperX align)")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Whisper':>12} {'Hybrid':>12} {'Delta':>10}")
    print("-" * 56)
    for key in ["n", "mean_ms", "median_ms", "p90_ms",
                "within_50ms", "within_100ms", "within_200ms",
                "within_300ms", "within_500ms"]:
        wv = w_report.get(key, 0)
        hv = h_report.get(key, 0)
        unit = "%" if "within" in key else ("ms" if "ms" in key else "")
        if isinstance(wv, (int, float)) and isinstance(hv, (int, float)):
            delta = hv - wv
            sign = "+" if delta > 0 else ""
            # For "within" metrics, positive delta = better
            # For error metrics, negative delta = better
            print(f"{key:<20} {wv:>11}{unit} {hv:>11}{unit} {sign}{delta:>8.1f}{unit}")
        else:
            print(f"{key:<20} {wv:>11}{unit} {hv:>11}{unit}")

    w_med = w_report.get("median_ms", 999)
    h_med = h_report.get("median_ms", 999)
    w_300 = w_report.get("within_300ms", 0)
    h_300 = h_report.get("within_300ms", 0)

    print(f"\n--- VERDICT ---")
    if h_med < w_med:
        print(f"✓ HYBRID WINS on median:  {h_med}ms vs {w_med}ms ({w_med - h_med:.1f}ms better)")
    elif h_med == w_med:
        print(f"= TIE on median: both {w_med}ms")
    else:
        print(f"✗ WHISPER WINS on median: {w_med}ms vs {h_med}ms ({h_med - w_med:.1f}ms worse)")

    if h_300 > w_300:
        print(f"✓ HYBRID WINS on ±300ms:  {h_300}% vs {w_300}% ({h_300 - w_300:.1f}% better)")
    elif h_300 == w_300:
        print(f"= TIE on ±300ms: both {w_300}%")
    else:
        print(f"✗ WHISPER WINS on ±300ms: {w_300}% vs {h_300}% ({w_300 - h_300:.1f}% better)")

    # Save results
    output_file = Path("TestOutput/whisperx_hybrid_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "whisper_report": w_report,
            "hybrid_report": h_report,
            "samples": SAMPLE_IDS,
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
