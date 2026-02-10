
import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, List, Callable
from datetime import datetime, timezone
from .separation import SeparationEngine
from .transcription import TranscriptionEngine
from qwen_asr import Qwen3ForcedAligner

# Schema version — bump on breaking JSON structure changes.
# External karaoke renderers should check this for compatibility.
KARAOKE_SCHEMA_VERSION = "1.0.0"


class KaraokePipeline:
    """
    Full karaoke pipeline: Separate → Transcribe → Align → Structured JSON.

    Output JSON schema (v1.0.0):
        version         - schema version string
        metadata        - song title, language, duration, creation timestamp
        audio           - relative filenames for original, vocals, instrumental
        lines[]         - sentence-level timing (for scrolling lyrics display)
        words[]         - word-level timing with line_id (for per-word highlight/sweep)

    Designed for consumption by external karaoke renderer systems.
    Accepts an optional progress callback for real-time status reporting (API/WebSocket use).
    """

    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cuda",
        whisper_model: str = "large-v3",
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        self.separator = SeparationEngine(model_dir=model_dir, device=device)
        self.transcriber = TranscriptionEngine(model_size=whisper_model, device=device)

        # Prefer local model dir to avoid re-downloading
        local_aligner = Path("./Qwen3-ForcedAligner-0.6B").resolve()
        final_aligner = str(local_aligner) if local_aligner.exists() else aligner_model

        print(f"[Karaoke] Loading Qwen Forced Aligner ({final_aligner})...")
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            final_aligner,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_song(
        self,
        audio_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> Dict:
        """
        Full pipeline: Separate → Transcribe (Whisper) → Align (Qwen) → Save JSON.

        Args:
            audio_path:  Path to the input audio file.
            output_dir:  Directory for all output files.
            on_progress: Optional ``callback(stage: str, progress: float)``
                         where *stage* is one of ``"separation"``, ``"transcription"``,
                         ``"alignment"`` and *progress* is 0.0 → 1.0.

        Returns:
            The full karaoke result dict (same structure as the saved JSON),
            or an empty dict on failure.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _emit(stage: str, pct: float):
            if on_progress:
                on_progress(stage, pct)

        # --- 1. Separate (2-stem karaoke model) ---
        _emit("separation", 0.0)
        print(f"\n--- [1/3] Separating Vocals (Karaoke 2-stem): {audio_path.name} ---")
        vocal_path, instrumental_path = self.separator.separate_karaoke(
            str(audio_path), str(output_dir)
        )
        _emit("separation", 1.0)

        if not vocal_path:
            print("[ERR] Separation produced no vocal track.")
            return {}

        # --- 2. Transcribe with Whisper (segments + words) ---
        _emit("transcription", 0.0)
        print(f"\n--- [2/3] Transcribing with Whisper ({self.transcriber.model_size}) ---")
        whisper_text, metadata = self.transcriber.transcribe(vocal_path)
        _emit("transcription", 1.0)

        if not whisper_text:
            print("[ERR] Transcription returned empty text.")
            return {}

        # --- 3. Word-level alignment (Qwen Forced Aligner) ---
        _emit("alignment", 0.0)
        print("\n--- [3/3] Aligning Lyrics with Qwen Forced Aligner ---")

        lang_detected = metadata.get("language", "ja")
        lang_map = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
        lang_name = lang_map.get(lang_detected, "Japanese")

        aligned_words = self._align_lyrics(vocal_path, whisper_text, lang_name, metadata)
        _emit("alignment", 1.0)

        # --- 4. Build structured karaoke output ---
        lines = self._build_lines(metadata.get("segments", []))
        words = self._assign_words_to_lines(aligned_words, lines)

        result = {
            "version": KARAOKE_SCHEMA_VERSION,
            "metadata": {
                "title": audio_path.stem,
                "language": lang_detected,
                "duration_seconds": round(metadata.get("duration", 0), 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "audio": {
                "original": audio_path.name,
                "vocals": Path(vocal_path).name if vocal_path else None,
                "instrumental": Path(instrumental_path).name if instrumental_path else None,
            },
            "lines": lines,
            "words": words,
        }

        output_file = output_dir / f"{audio_path.stem}_karaoke.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n[DONE] Karaoke data saved to: {output_file}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_lyrics(
        self, vocal_path: str, text: str, lang_name: str, whisper_meta: Dict
    ) -> List[Dict]:
        """Run Qwen forced alignment; fall back to Whisper word timestamps on error."""
        try:
            align_results = self.aligner.align(
                audio=vocal_path,
                text=text,
                language=lang_name,
            )
            words: List[Dict] = []
            if align_results:
                ts_items = align_results[0]
                if hasattr(ts_items, "items"):
                    ts_items = ts_items.items
                for item in ts_items:
                    words.append(
                        {
                            "text": getattr(item, "text", str(item)),
                            "start": round(getattr(item, "start_time", 0), 3),
                            "end": round(getattr(item, "end_time", 0), 3),
                        }
                    )
            return words
        except Exception as e:
            print(f"[WARN] Alignment failed ({e}), falling back to Whisper word timestamps")
            return [
                {
                    "text": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
                for w in whisper_meta.get("timestamps", [])
            ]

    @staticmethod
    def _build_lines(segments: List[Dict]) -> List[Dict]:
        """Convert Whisper segments into line objects with placeholder word indices."""
        lines = []
        for i, seg in enumerate(segments):
            lines.append(
                {
                    "id": i,
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "end": seg["end"],
                    "word_start_index": 0,
                    "word_end_index": 0,
                }
            )
        return lines

    @staticmethod
    def _assign_words_to_lines(words: List[Dict], lines: List[Dict]) -> List[Dict]:
        """
        Assign each aligned word to a line based on timing overlap.

        Each word receives a ``line_id``.  Each line's ``word_start_index`` /
        ``word_end_index`` are set so that
        ``words[line.word_start_index : line.word_end_index]``
        yields exactly the words belonging to that line.
        """
        if not words:
            for line in lines:
                line["word_start_index"] = 0
                line["word_end_index"] = 0
            return []

        if not lines:
            # No segment info — put everything in a single virtual line
            return [{**w, "line_id": 0} for w in words]

        enriched: List[Dict] = []
        line_idx = 0
        lines[0]["word_start_index"] = 0

        for w in words:
            # Advance to the next line if this word starts at or after it
            while (
                line_idx < len(lines) - 1
                and w["start"] >= lines[line_idx + 1]["start"]
            ):
                lines[line_idx]["word_end_index"] = len(enriched)
                line_idx += 1
                lines[line_idx]["word_start_index"] = len(enriched)

            enriched.append(
                {
                    "text": w["text"],
                    "start": w["start"],
                    "end": w["end"],
                    "line_id": line_idx,
                }
            )

        # Close the last active line
        lines[line_idx]["word_end_index"] = len(enriched)

        # Mark any trailing empty lines
        for i in range(line_idx + 1, len(lines)):
            lines[i]["word_start_index"] = len(enriched)
            lines[i]["word_end_index"] = len(enriched)

        return enriched


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python karaoke.py <audio_file>")
        sys.exit(1)

    pipeline = KaraokePipeline()
    pipeline.process_song(sys.argv[1], "KaraokeOutput")
