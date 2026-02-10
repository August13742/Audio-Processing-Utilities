
import os
import torch
import time
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def benchmark_asr(audio_path_str: str = "TestData/betelgeuse.mp3", comparison_filename: str = "ASR_Comparison.md"):
    audio_path = Path(audio_path_str).resolve()
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    print(f"Benchmarking ASR models on {audio_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}

    # 1. Faster-Whisper (Large-v3) - The "Anchor"
    print("\n--- Running Faster-Whisper (Large-v3) ---")
    start_time = time.time()
    whisper_words = []
    whisper_text = ""
    try:
        whisper_model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "float32")
        segments, info = whisper_model.transcribe(
            str(audio_path), 
            vad_filter=False, 
            word_timestamps=True,
            no_speech_threshold=0.8,
            compression_ratio_threshold=2.8,
            condition_on_previous_text=False
        )
        
        texts = []
        for s in segments:
            texts.append(s.text)
            if s.words:
                for w in s.words:
                    whisper_words.append({"word": w.word, "start": w.start, "end": w.end})
        
        whisper_text = "".join(texts).strip()
        whisper_duration = time.time() - start_time
        results["whisper"] = {"time": whisper_duration, "text": whisper_text, "words": whisper_words}
    except Exception as e:
        print(f"Whisper failed: {e}")
        return

    # 2. Qwen3-ForcedAligner-0.6B - Aligning Whisper's Text
    print(f"\n--- Running Qwen3-ForcedAligner (Aligning Whisper's Text) ---")
    start_time = time.time()
    try:
        from qwen_asr import Qwen3ForcedAligner
        aligner_path = Path("./Qwen3-ForcedAligner-0.6B").resolve()
        aligner_name = str(aligner_path) if aligner_path.exists() else "Qwen/Qwen3-ForcedAligner-0.6B"
        
        aligner = Qwen3ForcedAligner.from_pretrained(
            aligner_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        
        # Align Whisper's exact text
        qwen_results = aligner.align(
            audio=str(audio_path),
            text=whisper_text,
            language="Japanese", # Set explicitly to avoid NoneType error
        )
        qwen_duration = time.time() - start_time
        
        qwen_words = []
        if qwen_results and len(qwen_results) > 0:
            ts_items = qwen_results[0]
            if hasattr(ts_items, 'items'):
                ts_items = ts_items.items
            
            for item in ts_items:
                text = getattr(item, 'text', str(item))
                start = getattr(item, 'start_time', getattr(item, 'start', 0))
                end = getattr(item, 'end_time', getattr(item, 'end', 0))
                qwen_words.append({"word": text, "start": start, "end": end})

        results["qwen"] = {"time": qwen_duration, "text": whisper_text, "words": qwen_words}
    except Exception as e:
        print(f"Qwen Aligner failed: {e}")
        import traceback
        traceback.print_exc()

    # Output comparison as Markdown Table
    comparison_file = Path(comparison_filename).resolve()
    print(f"\n--- Saving Character-Level Timestamp Comparison to {comparison_file} ---")
    
    def get_char_timestamps(word_list):
        char_ts = []
        for w in word_list:
            text = w['word']
            start = w['start']
            end = w['end']
            if len(text) <= 1:
                char_ts.append({"char": text, "start": start, "end": end})
            else:
                # Distribute time equally for multi-char words (simple approximation)
                duration = end - start
                char_dur = duration / len(text)
                for i, char in enumerate(text):
                    char_ts.append({
                        "char": char, 
                        "start": start + i * char_dur, 
                        "end": start + (i + 1) * char_dur
                    })
        return char_ts

    w_chars = get_char_timestamps(results.get('whisper', {}).get('words', []))
    q_chars = get_char_timestamps(results.get('qwen', {}).get('words', []))

    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write(f"# ASR Character-Level Precision Comparison: {audio_path.name}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Whisper words are split into characters with interpolated timestamps for better side-by-side comparison.\n\n")
        f.write("| Index | Char | Whisper Start | Whisper End | Qwen Start | Qwen End | Delta |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        # We need to align the char lists since transcriptions might differ slightly
        # For simplicity in this benchmark, we'll just zip them if they are close in length,
        # or use a simple greedy matcher if the user wants.
        # But since Qwen aligned Whisper's text, they SHOULD be nearly identical in characters.
        
        max_idx = max(len(w_chars), len(q_chars))
        for i in range(min(max_idx, 300)):
            w = w_chars[i] if i < len(w_chars) else {"char": "-", "start": 0, "end": 0}
            q = q_chars[i] if i < len(q_chars) else {"char": "-", "start": 0, "end": 0}
            
            # Highlight if characters mismatch (unlikely if aligner worked)
            char_display = f"{w['char']} / {q['char']}" if w['char'] != q['char'] else w['char']
            diff = abs(w['start'] - q['start']) if i < min(len(w_chars), len(q_chars)) else 0
            f.write(f"| {i} | {char_display} | {w['start']:.3f} | {w['end']:.3f} | {q['start']:.3f} | {q['end']:.3f} | {diff:.3f}s |\n")


    print("\n--- Summary ---")
    print(f"{'Model':<20} | {'Time (s)':<10} | {'Text Length'}")
    print("-" * 50)
    for model, data in results.items():
        print(f"{model:<20} | {data['time']:<10.2f} | {len(data['text'])}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="TestData/betelgeuse.mp3")
    parser.add_argument("--out", type=str, default="ASR_Comparison.md")
    args = parser.parse_args()
    benchmark_asr(args.audio, args.out)
