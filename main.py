
import argparse
import sys
from pathlib import Path
import torch

from tools.dataset import DatasetBuilder
from tools.separation import SeparationEngine
from tools.transcription import TranscriptionEngine
from tools.cleaning import CleaningEngine
from tools.segmentation import Segmenter
from tools.converters import AudioConverter
from tools.analysis import AudioAnalyzer

# === COMMAND IMPLEMENTATIONS ===

def cmd_clean(args):
    print(f"[CMD] Vocal Cleaning Pipeline for {args.input}")
    engine = CleaningEngine(device=args.device, verbose=args.verbose)
    engine.process(args.input, args.output_dir)

def cmd_transcribe(args):
    print(f"[CMD] Transcription Pipeline for {args.input}")
    # Logic: Separate -> Transcribe Lead
    sep_engine = SeparationEngine(device=args.device, verbose=args.verbose)
    asr_engine = TranscriptionEngine(model_size=args.model, device=args.device)
    
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle dir or file
    if in_path.is_file():
        files = [in_path]
    else:
        files = list(in_path.rglob("*.wav")) + list(in_path.rglob("*.mp3")) + list(in_path.rglob("*.flac"))

    for f in files:
        print(f"\n> Processing {f.name}...")
        # 1. Separate Lead
        vocals, _ = sep_engine.separate_vocals(str(f), str(out_dir))
        if vocals:
            lead, _ = sep_engine.separate_lead_backing(vocals, str(out_dir))
            target = lead if lead else vocals
            
            # 2. Transcribe
            text, _ = asr_engine.transcribe(target)
            txt_path = out_dir / f"{f.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            print(f"  Transcript: {text[:50]}...")

def cmd_dataset(args):
    print(f"[CMD] Building Dataset from {args.input}")
    builder = DatasetBuilder(output_dir=args.output_dir, device=args.device)
    # Note: DatasetBuilder implements the AIO pipeline (Separate -> VAD -> Slice -> Select -> Transcribe)
    builder.process_dataset(args.input)

def cmd_segment(args):
    print(f"[CMD] Segmenting Audio for {args.input}")
    seg = Segmenter(method="silence", device="cpu") # VAD is fast on CPU usually
    
    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = list(in_path.rglob("*.wav"))
        
    for f in files:
        out_sub = Path(args.output_dir) / f.stem
        seg.segment_file(str(f), str(out_sub))

def cmd_convert(args):
    print(f"[CMD] Converting Audio in {args.input}")
    conv = AudioConverter(verbose=args.verbose)
    
    if Path(args.input).is_file():
        dest = Path(args.output_dir) / Path(args.input).with_suffix(f".{args.format}").name
        conv.convert(args.input, str(dest), format=args.format)
    else:
        conv.batch_convert(args.input, args.output_dir, target_format=args.format)

def cmd_stems(args):
    print(f"[CMD] Extracting 6 Stems for {args.input}")
    engine = SeparationEngine(device=args.device, verbose=args.verbose)
    
    in_path = Path(args.input)
    if in_path.is_file():
        engine.separate_stems(str(in_path), args.output_dir)
    else:
        for f in in_path.glob("*.wav"): # Simple glob
             engine.separate_stems(str(f), args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Ultra Audio Toolkit")
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

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
