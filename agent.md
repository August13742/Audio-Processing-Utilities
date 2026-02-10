# Agent Context Map

**Project**: Audio Processing Toolkit for karaoke generation and vocal separation  
**Language**: Python 3.11+  
**Key Framework**: Torch, audio-separator, faster-whisper, torchfcpe  
**Architecture**: Subprocess protocol (stdout=JSON, stderr=logs+progress, exit codes: 0/1)

---

## Critical Context

### Subprocess Protocol Contract
- **stdout**: Single JSON object result (machine-readable only)
- **stderr**: Human logs + progress JSON lines (`{"type":"progress","stage":"...", "progress": 0.0-1.0}`)
- **exit code**: 0 = success, 1 = failure
- **NO print() in karaoke pipeline** — all logging via `utils/log.py:log()` and `emit_progress()`

### JSON Schema v1.1.0
Karaoke output structure (see [README.md](README.md#1-karaoke--generate-karaoke-data) for full schema):
- `version: "1.1.0"`, `metadata`, `audio` (vocals/instrumental paths)
- `pitch` — compact frame-level pitch curve (`hop_seconds`, `unit: "midi"`, `values[]`)
- `lines[]` (sentences), `words[]` (word-level timings + `pitch_midi`, `pitch_hz`)
- Bidirectional indexing: `lines[i].word_start_index` → slice into `words[]`
- Scoring philosophy: `pitch_midi: 0` on a word = free points (no penalty)

### Stem Naming Convention
- `song_vocals.wav`, `song_instrumental.wav` (not verbose model names)
- 2-stem separation (Mel-Band Roformer) for karaoke, 6-stem (BS-Roformer) for full extraction

---

## File Map

### Entry Point
- **[main.py](main.py)** — CLI dispatcher, single entry point for all 7 commands
  - Uses `_emit_result()` function to output JSON to stdout
  - All commands route through this file
  - Exit codes: 0 on success, `sys.exit(1)` on failure

### Core Karaoke Pipeline
- **[tools/karaoke.py](tools/karaoke.py)** — Main orchestration
  - `process_song(audio_path, output_dir, on_progress=None)` — Full pipeline
  - Separates → Transcribes → Extracts Pitch → Builds JSON
  - Uses `SeparationEngine`, `TranscriptionEngine`, `PitchExtractor`
  - Word alignment: Whisper native timestamps (validated: 46.3ms median, 86.5% within ±300ms on kiritan database)
  - See [README.md#alignment-strategy--validation](README.md#alignment-strategy--validation) for alignment testing results
  - All logging: `from utils.log import log, emit_progress`

- **[tools/pitch.py](tools/pitch.py)** — Pitch extraction & curve processing
  - `PitchExtractor` — wraps torchfcpe (FCPE) for f0 estimation
  - `clean_pitch_curve()` — median filter + Savitzky-Golay smoothing in MIDI space
  - `assign_pitch_to_words()` — per-word median pitch for scoring
  - `build_pitch_section()` — compact pitch array for JSON output
  - `hz_to_midi()`, `midi_to_hz()`, `midi_to_note_name()` — pitch math utilities
  - Design: forgiving scoring — unvoiced words = free points, not penalties

- **[tools/separation.py](tools/separation.py)** — Audio stem extraction
  - `separate_karaoke(audio_path, output_dir)` — 2-stem (vocals/instrumental)
  - `separate_stems(audio_path, output_dir)` — 6-stem full extraction
  - Always use `output_dir` parameter (don't hardcode model paths)

- **[tools/transcription.py](tools/transcription.py)** — Speech-to-text
  - `transcribe(audio_path, language='en', model_size='large-v3')`
  - Returns metadata with `segments[]` (sentence-level) and timestamps

### Logging & Utilities
- **[utils/log.py](utils/log.py)** — Centralized logging
  - `log(msg)` — sends to stderr
  - `emit_progress(stage, progress)` — sends JSON progress line to stderr
  - **Critical**: All pipeline logging must use this, never `print()`

- **[utils/audio.py](utils/audio.py)** — Audio IO helpers
  - `load_audio()`, `save_audio()`, constants
  - Updated to use `log()` instead of `print()`

- **[utils/models.py](utils/models.py)** — Model management
  - Download, cache, version tracking
  - Updated to use `log()` instead of `print()`

### Documentation
- **[README.md](README.md)** — User-facing documentation
  - Commands, subprocess protocol, integration examples (Godot GDScript, Python, Bash, Node.js)
  - Full JSON schema v1.0.0
  - Troubleshooting, benchmarks, model references

- **[tools/api.py](tools/api.py)** — Protocol reference (not executable code)
  - Example integrations in 4+ languages
  - Subprocess call patterns

---

## Common Agent Tasks

### Task: "Fix bug in karaoke pipeline"
1. Check [main.py](main.py) → `cmd_karaoke()` entry point
2. Trace into [tools/karaoke.py](tools/karaoke.py) → `process_song()`
3. Likely in: `_separate_audio()`, `_transcribe()`, `_align_lyrics()`, `_build_lines()`
4. Verify: No `print()` calls, use `log()` and `emit_progress()` instead
5. Test: `uv run main.py karaoke "TestData/Voyaging Star's Farewell.mp3" "TestOutput/karaoke/"`

### Task: "Add new CLI command"
1. Add function `cmd_newfeature()` in [main.py](main.py)
2. Register in `commands` dict at top of `main()`
3. Use `_emit_result()` to return JSON to stdout
4. Implement logic in new file under `tools/` (follow naming: `tools/newfeature.py`)
5. Import and call from main.py using try/except with `sys.exit(1)` on failure

### Task: "Integrate with external system (Godot, Python, etc.)"
1. Read subprocess protocol section below
2. Example: See [tools/api.py](tools/api.py) for Godot GDScript, Python, Bash, Node.js patterns
3. Verify: stdout is single JSON, stderr is ignored for final result
4. Handle: exit code (0=success, 1=failure)

### Task: "Modify JSON output schema"
1. Update [tools/karaoke.py](tools/karaoke.py) → `_build_output_json()`
2. Increment `version` field (e.g., 1.0.0 → 1.1.0)
3. Update [README.md](README.md) schema section with new fields
4. Update integration examples in [tools/api.py](tools/api.py) if schema significantly changes

### Task: "Optimize separation/transcription quality"
1. Separation: [tools/separation.py](tools/separation.py) → model selection in `separate_karaoke()`
   - Currently: `Mel-Band-Roformer` (2-stem, optimized for vocals)
   - Alternative: `BS-Roformer` (6-stem, more detailed but slower)
2. Transcription & alignment: [tools/transcription.py](tools/transcription.py)
   - Transcription: `--model` parameter (tiny/base/small/medium/large-v3; larger = better accuracy but slower)
   - Alignment: Uses Whisper's native word timestamps (validated on 50 songs; 46.3ms median error)
   - No external alignment needed — Whisper outperforms Qwen and WhisperX approaches

---

## Subprocess Protocol (Minimal Reference)

Calling from external system:

```bash
uv run main.py <command> [args] [--device cuda|cpu]
```

**Expected output:**
```
stdout:  {"status": "ok", "results": [...]} or {"error": "message"}
stderr:  Human logs + {"type":"progress","stage":"...","progress":0.0-1.0}
exit:    0 on success, 1 on failure
```

**No streaming output** — wait for process exit, then parse single JSON from stdout.

See [tools/api.py](tools/api.py) for language-specific examples (Godot, Python, Bash, Node.js).

---

## Key Files to Avoid

- **[tools/cleaning.py](tools/cleaning.py)** — vocal enhancement (not critical karaoke path)
- **[tools/segmentation.py](tools/segmentation.py)** — VAD segmentation (not critical karaoke path)
- **[tools/converters.py](tools/converters.py)** — format conversion (not critical karaoke path)
- **[tools/dataset.py](tools/dataset.py)** — dataset building (separate pipeline)
- **legacy/** — old code, archived (ignore)

These don't affect core karaoke generation and have `print()` calls (acceptable—they don't interfere with stdout contract).

---

## Recent Changes (Session: 2026-02-10/11)

✅ **Phase 1: Pitch Extraction & Validation**:
1. Created [tools/pitch.py](tools/pitch.py) — FCPE pitch extraction + curve cleaning
2. Integrated pitch into karaoke pipeline — schema bumped to v1.1.0
3. Created [tests/test_karaoke_scoring.ipynb](tests/test_karaoke_scoring.ipynb) — validation notebook

✅ **Phase 2: Alignment Testing & Strategy Finalization**:
4. Tested multiple alignment approaches on Kiritan Singing Database (50 songs, phoneme ground truth):
   - Whisper native: **46.3ms median, 86.5% within ±300ms** ✅ **WINNER**
   - Qwen3FA: 44.4ms median, 96.7% within ±300ms (but 48% drops outside vocal region)
   - Whisper+Qwen: 39.8ms median, 98.9% within ±300ms (overfits to phonemes, poor generalization)
   - WhisperX: 114,989ms median, 8.6% within ±300ms (hallucination on karaoke audio)
   - Whisper+WhisperX: 84.2ms median, 72.8% within ±300ms (degrades baseline by 82%)
5. **Decision: Whisper-only**, remove WhisperX and Qwen dependencies
   - Whisper's built-in timestamps are production-ready
   - Karaoke-tuned config: `vad_filter=False`, `no_speech_threshold=0.8` (no VAD hallucinations)
   - External alignment adds complexity without improvement

✅ **Phase 3: Dependency Cleanup**:
6. Removed `whisperx>=3.7.5`, `qwen-asr` from [pyproject.toml](pyproject.toml)
7. Updated [README.md](README.md) with alignment validation section & revised model references
8. Updated [agent.md](agent.md) with Whisper-only strategy documentation

✅ **Tested**:
- `uv run main.py karaoke "TestData/Voyaging Star's Farewell.mp3" "TestOutput/karaoke/"` → Exit 0, valid JSON output
- `uv run python tests/test_whisperx_alignment.py` → Hybrid (Whisper+WhisperX) confirmed worse than Whisper-only
- Python compilation check: All pipeline modules compile cleanly

---

## Quick Diagnostics

### Syntax Check
```bash
python -m py_compile tools/karaoke.py tools/separation.py tools/transcription.py main.py
```

### Test Karaoke Command
```bash
uv run main.py karaoke "TestData/Voyaging Star's Farewell.mp3" "TestOutput/karaoke_test/" --verbose
```

### Check Stdout/Stderr Separation
```bash
uv run main.py karaoke "TestData/Voyaging Star's Farewell.mp3" "TestOutput/karaoke_test/" > stdout.json 2> stderr.log
cat stdout.json | jq .version  # Should print "1.0.0"
grep "progress" stderr.log | wc -l  # Should have progress lines
```

### Verify No print() in Karaoke Pipeline
```bash
grep -n "print(" tools/karaoke.py tools/separation.py tools/transcription.py main.py
# Should return: (no lines = good)
```