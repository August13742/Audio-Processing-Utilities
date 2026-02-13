
import argparse
import sys
import json
from pathlib import Path
import torch

from utils.log import log

from tools.dataset import DatasetBuilder
from tools.separation import SeparationEngine
from tools.transcription import TranscriptionEngine
from tools.cleaning import CleaningEngine
from tools.segmentation import Segmenter
from tools.converters import AudioConverter
from tools.analysis import AudioAnalyzer
from tools.karaoke import KaraokePipeline

# ─── Subprocess output protocol ───────────────────────────────────────
#   stdout  → single JSON object on success (machine-readable for Godot)
#   stderr  → human-readable logs + optional JSON progress lines
#   exit 0  → success     exit 1  → failure
#
# Godot usage:
#   var output = []
#   OS.execute("uv", ["run", "main.py", "karaoke", audio_path, out_dir], output)
#   var result = JSON.parse_string(output[0])
# ──────────────────────────────────────────────────────────────────────


def _emit_result(data: dict):
    """Write the final machine-readable result to stdout."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


# === COMMAND IMPLEMENTATIONS ===

def cmd_clean(args):
    log(f"[CMD] Vocal Cleaning Pipeline for {args.input}")
    engine = CleaningEngine(device=args.device, verbose=args.verbose)
    engine.process(args.input, args.output_dir)
    _emit_result({"status": "ok", "output_dir": str(Path(args.output_dir).resolve())})

def cmd_transcribe(args):
    log(f"[CMD] Transcription Pipeline for {args.input}")
    sep_engine = SeparationEngine(device=args.device, verbose=args.verbose)
    asr_engine = TranscriptionEngine(model_size=args.model, device=args.device)
    
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if in_path.is_file():
        files = [in_path]
    else:
        files = list(in_path.rglob("*.wav")) + list(in_path.rglob("*.mp3")) + list(in_path.rglob("*.flac"))

    results = []
    for f in files:
        log(f"> Processing {f.name}...")
        vocals, _ = sep_engine.separate_vocals(str(f), str(out_dir))
        if vocals:
            lead, _ = sep_engine.separate_lead_backing(vocals, str(out_dir))
            target = lead if lead else vocals
            
            text, meta = asr_engine.transcribe(target)
            txt_path = out_dir / f"{f.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            results.append({
                "file": f.name,
                "transcript_path": str(txt_path.resolve()),
                "text": text,
                "language": meta.get("language", ""),
            })

    _emit_result({"status": "ok", "results": results})

def cmd_dataset(args):
    log(f"[CMD] Building Dataset from {args.input}")
    builder = DatasetBuilder(output_dir=args.output_dir, device=args.device)
    builder.process_dataset(args.input)
    _emit_result({"status": "ok", "output_dir": str(Path(args.output_dir).resolve())})

def cmd_segment(args):
    log(f"[CMD] Segmenting Audio for {args.input}")
    seg = Segmenter(method="silence", device="cpu")
    
    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = list(in_path.rglob("*.wav"))
        
    outputs = []
    for f in files:
        out_sub = Path(args.output_dir) / f.stem
        seg.segment_file(str(f), str(out_sub))
        outputs.append(str(out_sub.resolve()))

    _emit_result({"status": "ok", "segment_dirs": outputs})

def cmd_convert(args):
    log(f"[CMD] Converting Audio in {args.input}")
    conv = AudioConverter(verbose=args.verbose)
    
    if Path(args.input).is_file():
        dest = Path(args.output_dir) / Path(args.input).with_suffix(f".{args.format}").name
        conv.convert(args.input, str(dest), format=args.format)
        _emit_result({"status": "ok", "output_file": str(dest.resolve())})
    else:
        conv.batch_convert(args.input, args.output_dir, target_format=args.format)
        _emit_result({"status": "ok", "output_dir": str(Path(args.output_dir).resolve())})

def cmd_stems(args):
    log(f"[CMD] Extracting 6 Stems for {args.input}")
    engine = SeparationEngine(device=args.device, verbose=args.verbose)
    
    in_path = Path(args.input)
    outputs = []
    if in_path.is_file():
        out = engine.separate_stems(str(in_path), args.output_dir)
        if out: outputs.append(out)
    else:
        for f in in_path.glob("*.wav"):
            out = engine.separate_stems(str(f), args.output_dir)
            if out: outputs.append(out)

    _emit_result({"status": "ok", "stem_dirs": outputs})

def cmd_karaoke(args):
    log(f"[CMD] Starting Karaoke Pipeline for {args.input}")
    pipeline = KaraokePipeline(device=args.device)
    result = pipeline.process_song(args.input, args.output_dir)
    if not result:
        _emit_result({"status": "error", "message": "Karaoke pipeline failed."})
        sys.exit(1)
    # Emit only a status summary to stdout instead of the full 100KB+ JSON curve.
    # The full data is already saved to 'karaoke.json' in the song subfolder.
    _emit_result({
        "status": "ok", 
        "song": result.get("metadata", {}).get("title"),
        "output_dir": args.output_dir
    })

def main():
    parser = argparse.ArgumentParser(description="Audio Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Global Args
    gen_parser = argparse.ArgumentParser(add_help=False)
    gen_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    gen_parser.add_argument("--verbose", action="store_true")

    # 1. Clean
    p_clean = subparsers.add_parser("clean", parents=[gen_parser], help="Vocal Cleaning Pipeline")
    p_clean.add_argument("input", help="Input file")
    p_clean.add_argument("output_dir", help="Output directory")
    p_clean.set_defaults(func=cmd_clean)

    # 2. Transcribe
    p_trans = subparsers.add_parser("transcribe", parents=[gen_parser], help="Transcription Pipeline")
    p_trans.add_argument("input", help="Input file/dir")
    p_trans.add_argument("output_dir", help="Output directory")
    p_trans.add_argument("--model", default="large-v3", help="Whisper model")
    p_trans.set_defaults(func=cmd_transcribe)

    # 3. Dataset
    p_data = subparsers.add_parser("dataset", parents=[gen_parser], help="AIO Dataset Builder")
    p_data.add_argument("input", help="Input file/dir")
    p_data.add_argument("output_dir", help="Output directory")
    p_data.set_defaults(func=cmd_dataset)

    # 4. Segment
    p_seg = subparsers.add_parser("segment", parents=[gen_parser], help="Audio Segmentation")
    p_seg.add_argument("input", help="Input file/dir")
    p_seg.add_argument("output_dir", help="Output directory")
    p_seg.set_defaults(func=cmd_segment)

    # 5. Convert
    p_conv = subparsers.add_parser("convert", parents=[gen_parser], help="Format Conversion")
    p_conv.add_argument("input", help="Input file/dir")
    p_conv.add_argument("output_dir", help="Output directory")
    p_conv.add_argument("--format", default="ogg", help="Target format")
    p_conv.set_defaults(func=cmd_convert)

    # 6. Stems
    p_stem = subparsers.add_parser("stems", parents=[gen_parser], help="6-Stem Separation")
    p_stem.add_argument("input", help="Input file/dir")
    p_stem.add_argument("output_dir", help="Output directory")
    p_stem.set_defaults(func=cmd_stems)

    # 7. Karaoke
    p_kara = subparsers.add_parser("karaoke", parents=[gen_parser], help="Karaoke Pipeline (Sep -> Whisper -> FCPE)")
    p_kara.add_argument("input", help="Input file")
    p_kara.add_argument("output_dir", help="Output directory")
    p_kara.set_defaults(func=cmd_karaoke)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
