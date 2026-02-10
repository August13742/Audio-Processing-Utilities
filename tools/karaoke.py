
import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, List, Callable
from datetime import datetime, timezone
from .separation import SeparationEngine
from .transcription import TranscriptionEngine
from .pitch import PitchExtractor, assign_pitch_to_words
from utils.log import log, emit_progress

# Schema version — bump on breaking JSON structure changes.
# External karaoke renderers should check this for compatibility.
KARAOKE_SCHEMA_VERSION = "1.1.0"


class KaraokePipeline:
    """
    Full karaoke pipeline: Separate → Transcribe → Pitch → Structured JSON.

    Output JSON schema (v1.1.0):
        version         - schema version string
        metadata        - song title, language, duration, creation timestamp
        audio           - relative filenames for original, vocals, instrumental
        lines[]         - sentence-level timing (for scrolling lyrics display)
        words[]         - word-level timing with line_id (for per-word highlight/sweep)

    Designed for consumption by external systems (Godot, etc.) via subprocess.
    Progress is emitted to stderr as JSON lines; final result goes to stdout.
    """

    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cuda",
        whisper_model: str = "large-v3",
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        self.separator = SeparationEngine(model_dir=model_dir, device=device)
        self.transcriber = TranscriptionEngine(model_size=whisper_model, device=device)

        log("[Karaoke] Loading Pitch Extractor (FCPE)...")
        self.pitch_extractor = PitchExtractor(device=device)

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
        Full pipeline: Separate → Transcribe (Whisper) → Pitch (FCPE) → Save JSON.

        Args:
            audio_path:  Path to the input audio file.
            output_dir:  Directory for all output files.
            on_progress: Optional ``callback(stage: str, progress: float)``.
                         If None, progress is emitted to stderr as JSON lines
                         (suitable for subprocess consumers like Godot).

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
            else:
                emit_progress(stage, pct)

        # --- 1. Separate (2-stem karaoke model) ---
        _emit("separation", 0.0)
        log(f"[1/4] Separating Vocals (Karaoke 2-stem): {audio_path.name}")
        vocal_path, instrumental_path = self.separator.separate_karaoke(
            str(audio_path), str(output_dir)
        )
        _emit("separation", 1.0)

        if not vocal_path:
            log("[ERR] Separation produced no vocal track.")
            return {}

        # --- 2. Transcribe with Whisper (segments + words) ---
        _emit("transcription", 0.0)
        log(f"[2/4] Transcribing with Whisper ({self.transcriber.model_size})")
        whisper_text, metadata = self.transcriber.transcribe(vocal_path)
        _emit("transcription", 1.0)

        if not whisper_text:
            log("[ERR] Transcription returned empty text.")
            return {}

        # --- 3. Word-level timestamps from Whisper ---
        _emit("alignment", 0.0)
        log("[3/4] Extracting word-level timestamps from Whisper")

        lang_detected = metadata.get("language", "ja")
        lang_map = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
        lang_name = lang_map.get(lang_detected, "Japanese")

        # Extract word-level timestamps from Whisper metadata
        # Whisper's built-in word alignment outperforms external forced alignment
        aligned_words = self._extract_whisper_words(metadata)
        _emit("alignment", 1.0)

        # --- 4. Pitch extraction (FCPE) ---
        _emit("pitch", 0.0)
        log("[4/4] Extracting pitch curve (FCPE)")
        pitch_data = self.pitch_extractor.extract(vocal_path)
        _emit("pitch", 1.0)

        # --- 5. Build structured karaoke output ---
        lines = self._build_lines(metadata.get("segments", []))
        words = self._assign_words_to_lines(aligned_words, lines)

        # Enrich words with per-word pitch (for scoring & melody guide)
        # Each word gets pitch_midi, pitch_hz, note — the game draws target
        # note boxes from (word.start, word.end, word.pitch_midi).
        # No frame-level pitch array needed — games use per-word rectangles.
        words = assign_pitch_to_words(
            words,
            pitch_data["midi_clean"],
            pitch_data["voiced"],
            pitch_data["hop_seconds"],
        )

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

        log(f"[DONE] Karaoke data saved to: {output_file}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_whisper_words(whisper_meta: Dict) -> List[Dict]:
        """Extract word-level timestamps from Whisper transcription metadata.
        
        Whisper's built-in word alignment is reliable for karaoke timing
        (§ kiritan validation: 98.9% within 300ms, 39.8ms median error).
        """
        words = [
            {
                "text": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            }
            for w in whisper_meta.get("timestamps", [])
        ]
        return words

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
