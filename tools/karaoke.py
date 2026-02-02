
import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, List
from .separation import SeparationEngine
from .transcription import TranscriptionEngine
from qwen_asr import Qwen3ForcedAligner

class KaraokePipeline:
    def __init__(
        self, 
        model_dir: str = "models", 
        device: str = "cuda", 
        whisper_model: str = "large-v3",
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        self.separator = SeparationEngine(model_dir=model_dir, device=device)
        self.transcriber = TranscriptionEngine(model_size=whisper_model, device=device)
        
        # Check for local aligner model path to avoid download
        local_aligner = Path("./Qwen3-ForcedAligner-0.6B").resolve()
        final_aligner = str(local_aligner) if local_aligner.exists() else aligner_model
        
        # Load Qwen Aligner
        print(f"[Karaoke] Loading Qwen Forced Aligner ({final_aligner})...")
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            final_aligner,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )

    def process_song(self, audio_path: str, output_dir: str) -> str:
        """
        Full pipeline: Separate -> Transcribe (Whisper) -> Align (Qwen) -> Save JSON
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Separate Vocals
        print(f"\n--- [1/3] Separating Vocals: {audio_path.name} ---")
        vocal_path, instrumental_path = self.separator.separate_vocals(str(audio_path), str(output_dir))
        
        # 2. Transcribe cleanly with Whisper
        print(f"\n--- [2/3] Transcribing with Whisper ({self.transcriber.model_size}) ---")
        whisper_text, metadata = self.transcriber.transcribe(vocal_path)
        
        if not whisper_text:
            print("[ERR] Transcription failed.")
            return ""

        # 3. Align with Qwen for ultra-precision
        print(f"\n--- [3/3] Aligning Lyrics with Qwen Forced Aligner ---")
        # Detected language from Whisper
        lang_detected = metadata.get("language", "Japanese")
        # Map Whisper language codes to Qwen expected names if necessary
        lang_map = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
        lang_name = lang_map.get(lang_detected, "Japanese")
        
        try:
            align_results = self.aligner.align(
                audio=vocal_path,
                text=whisper_text,
                language=lang_name
            )
            
            # Extract timestamps
            karaoke_data = []
            if align_results:
                ts_items = align_results[0]
                if hasattr(ts_items, 'items'):
                    ts_items = ts_items.items
                
                for item in ts_items:
                    karaoke_data.append({
                        "text": getattr(item, 'text', str(item)),
                        "start": getattr(item, 'start_time', 0),
                        "end": getattr(item, 'end_time', 0)
                    })
        except Exception as e:
            print(f"[ERR] Alignment failed: {e}")
            # Fallback to Whisper timestamps if alignment fails
            karaoke_data = metadata.get("timestamps", [])

        # 4. Save Final Output
        result = {
            "song_path": os.path.abspath(audio_path),
            "vocal_path": os.path.abspath(vocal_path),
            "instrumental_path": os.path.abspath(instrumental_path),
            "language": lang_detected,
            "lyrics_text": whisper_text,
            "lyrics_aligned": karaoke_data
        }
        
        output_file = output_dir / f"{audio_path.stem}_karaoke.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"\n[DONE] Karaoke data saved to: {output_file}")
        return str(output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python karaoke.py <audio_file>")
        sys.exit(1)
        
    pipeline = KaraokePipeline()
    pipeline.process_song(sys.argv[1], "KaraokeOutput")
