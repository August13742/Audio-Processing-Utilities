# Audio Utilities – Processing Toolkit

A comprehensive audio processing toolkit for vocal separation, transcription, and karaoke generation.
Powered by **BS-Roformer**, **Mel-Band Roformer**, **Faster-Whisper**, and **Torch-FCPE**.

**Designed for both CLI usage and external integration** (Godot, Python, web services, etc.) via a subprocess protocol.

---

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.10)
- CUDA for GPU acceleration or CPU-only fallback (Project uses 12.8 due to blackwell GPU)
- 8GB+ VRAM for typical songs (~4min), 16GB+ recommended

### Setup

```bash
uv sync --prerelease=allow
```

---

## Quick Start

### Karaoke Generation (CLI)

The primary use case: generate structured karaoke data from audio.

```bash
uv run main.py karaoke "path/to/song.mp3" "output_directory/"
```

**Outputs:**
- `song_vocals.wav` — isolated lead vocal track
- `song_instrumental.wav` — instrumental backing track  
- `song_karaoke.json` — structured lyrics with timings (see schema below)

---

## Available Commands

All commands follow the same pattern:

```bash
uv run main.py <command> [args] [--device cuda|cpu] [--verbose]
```

### 1. **karaoke** – Generate karaoke data

Full pipeline: Separate → Transcribe → Align → Structured JSON

```bash
uv run main.py karaoke "song.mp3" "output/"
```

**Output JSON schema (v1.1.0):**
```json
{
  "version": "1.1.0",
  "metadata": {
    "title": "Song Name",
    "language": "en",
    "duration_seconds": 234.5,
    "created_at": "2026-02-10T12:00:00+00:00"
  },
  "audio": {
    "original": "song.mp3",
    "vocals": "song_vocals.wav",
    "instrumental": "song_instrumental.wav"
  },
  "lines": [
    {
      "id": 0,
      "text": "The last snowflake once fell on my nose",
      "start": 23.44,
      "end": 28.0,
      "word_start_index": 0,
      "word_end_index": 8
    }
  ],
  "words": [
    {
      "text": "snowflake",
      "start": 23.92,
      "end": 25.22,
      "line_id": 0,
      "pitch_midi": 63.01,
      "pitch_hz": 311.3,
      "note": "D#4"
    }
  ]
}
```

**Rendering algorithm (for karaoke display):**
```
1. Play audio.instrumental
2. At playback time t, find active line:  lines[i] where i.start ≤ t < i.end
3. Slice words for that line:  words[line.word_start_index : line.word_end_index]
4. Highlight word where word.start ≤ t < word.end
5. Scroll to next line as time advances
```

**Pitch scoring (for karaoke games):**
```
Each word has pitch_midi (MIDI note number) for target pitch comparison.

For melody guide visualization:
  - Draw a rectangle per word: x = word.start → word.end, y = word.pitch_midi
  - pitch_midi = 0 means unvoiced (spoken/breath) → no pitch penalty, free points

For pitch scoring:
  - Capture microphone pitch (Hz) → convert to MIDI: 12 * log2(hz / 440) + 69
  - Compare to word.pitch_midi at current time
  - Tolerance: ±2 semitones is "good", ±1 is "perfect"
  - Words with pitch_midi = 0 → always score 100% (forgiving design)
```

### 2. **transcribe** – Extract lyrics without separation

Separates vocals, then extracts clean transcription.

```bash
uv run main.py transcribe "song.mp3" "output/" [--model large-v3|base|small|tiny]
```

**Output:**
```json
{
  "status": "ok",
  "results": [
    {
      "file": "song.mp3",
      "transcript_path": "/abs/path/to/song.txt",
      "text": "Full transcription text...",
      "language": "en"
    }
  ]
}
```

### 3. **stems** – 6-stem separation (advanced)

Extract individual instruments: vocals, drums, bass, guitar, piano, other.

```bash
uv run main.py stems "song.mp3" "output/"
```

**Output:**
```json
{
  "status": "ok",
  "stem_dirs": ["/abs/path/to/song_stems/"]
}
```

### 4. **clean** – Enhance vocal clarity

Vocal cleaning pipeline using ClearVoice (speech enhancement).

```bash
uv run main.py clean "song.mp3" "output/"
```

### 5. **segment** – Time-scale segmentation

Split audio into segments using silence detection.

```bash
uv run main.py segment "song.mp3" "output/"
```

### 6. **convert** – Audio format conversion

Convert between audio formats (MP3 → WAV, OGG, etc.)

```bash
uv run main.py convert "song.mp3" "output/" --format wav
```

### 7. **dataset** – Build training datasets

Full pipeline: Separate → VAD segmentation → Diversity selection → Dataset manifest.

```bash
uv run main.py dataset "raw_audio_folder/" "output_dataset/"
```

---

## Subprocess Protocol

All commands follow a unified subprocess contract suitable for **Godot, Python, Bash, Node.js**, or any other caller.

### Output Streams

| Stream | Content | Godot reads from |
|--------|---------|------------------|
| **stdout** | Single JSON object (the result) | `output[0]` after `OS.execute()` |
| **stderr** | Human-readable logs + progress JSON | Console / optional progress tracking |
| **Exit code** | 0 = success, 1 = failure | Check `exit_code` |

### Example: Godot (GDScript 4.x)

```gdscript
# Basic call (no progress tracking)
func generate_karaoke(audio_path: String, out_dir: String) -> Dictionary:
    var output := []
    var exit_code := OS.execute(
        "uv",
        ["run", "main.py", "karaoke", audio_path, out_dir],
        output
    )
    if exit_code != 0:
        push_error("Karaoke generation failed")
        return {}
    
    var result: Dictionary = JSON.parse_string(output[0])
    return result  # Contains version, metadata, audio, lines, words


# With progress tracking (optional)
func generate_karaoke_with_progress(
    audio_path: String, 
    out_dir: String
) -> Dictionary:
    # Use OS.create_process() to read stderr in real-time
    var process_id = OS.create_process(
        "uv",
        ["run", "main.py", "karaoke", audio_path, out_dir],
        false  # don't open console
    )
    
    var output_lines := []
    var progress = 0.0
    
    # Poll stderr for progress updates (simplified)
    while OS.is_process_alive(process_id):
        await get_tree().create_timer(0.1).timeout
    
    # After completion, read final output
    # Note: OS.execute() is simpler for final result, or use a temp file
    var output := []
    OS.execute("uv", ["run", "main.py", "karaoke", audio_path, out_dir], output)
    return JSON.parse_string(output[0])
```

### Example: Python

```python
import subprocess
import json

def generate_karaoke(audio_path: str, output_dir: str) -> dict:
    proc = subprocess.run(
        ["uv", "run", "main.py", "karaoke", audio_path, output_dir],
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print(f"Error: {proc.stderr}")
        return {}
    
    result = json.loads(proc.stdout)
    return result


# With progress tracking
def generate_karaoke_with_progress(audio_path: str, output_dir: str) -> dict:
    proc = subprocess.Popen(
        ["uv", "run", "main.py", "karaoke", audio_path, output_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Read stderr line-by-line for progress
    for line in proc.stderr:
        if line.startswith('{"type":"progress"'):
            progress_data = json.loads(line)
            print(f"{progress_data['stage']}: {progress_data['progress']*100:.0f}%")
    
    proc.wait()
    
    # Read final JSON from stdout
    stdout_text = proc.stdout.read()
    result = json.loads(stdout_text)
    return result
```

### Example: Bash Script

```bash
#!/bin/bash

AUDIO_FILE="$1"
OUTPUT_DIR="$2"

# Run karaoke pipeline
OUTPUT=$(uv run main.py karaoke "$AUDIO_FILE" "$OUTPUT_DIR")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Karaoke pipeline failed"
    exit 1
fi

# Parse JSON result
VERSION=$(echo "$OUTPUT" | jq -r '.version')
TITLE=$(echo "$OUTPUT" | jq -r '.metadata.title')
VOCALS=$(echo "$OUTPUT" | jq -r '.audio.vocals')
INSTRUMENTAL=$(echo "$OUTPUT" | jq -r '.audio.instrumental')

echo "✓ Generated karaoke for: $TITLE"
echo "  Version: $VERSION"
echo "  Vocals: $VOCALS"
echo "  Instrumental: $INSTRUMENTAL"
```

```

---

## CLI Examples

### Basic karaoke generation
```bash
uv run main.py karaoke "MyBand_Song.mp3" "./output/"
```

### Transcription only
```bash
uv run main.py transcribe "vocal_recording.wav" "./transcripts/"
```

### Using CPU instead of GPU
```bash
uv run main.py karaoke "song.mp3" "./output/" --device cpu
```

### Verbose logging
```bash
uv run main.py karaoke "song.mp3" "./output/" --verbose
```

### Batch processing (via shell)
```bash
for file in music/*.mp3; do
    uv run main.py karaoke "$file" "./karaoke_output/" &
done
wait
```

---

## Project Structure

```
.
├── main.py              # Single entry point (CLI dispatcher)
├── tools/
│   ├── karaoke.py       # Karaoke pipeline (Separate → Transcribe → Align)
│   ├── separation.py    # Audio stem separation (BS-Roformer, Mel-Roformer)
│   ├── transcription.py # Whisper-based speech-to-text
│   ├── cleaning.py      # Vocal enhancement (ClearVoice)
│   ├── segmentation.py  # VAD-based audio segmentation
│   ├── converters.py    # Format conversion
│   ├── analysis.py      # Audio analysis metrics
│   ├── dataset.py       # Dataset building pipeline
│   └── api.py           # Subprocess protocol documentation
├── utils/
│   ├── log.py           # Centralized logging (stderr)
│   ├── audio.py         # Audio IO, loading, saving, normalization
│   ├── models.py        # Model downloading & management
│   └── processing.py    # Audio DSP helpers
├── models/              # Downloaded ML models (check download_checks.json)
└── README.md           # This file
```

---

## Advanced Features

### Reading Progress in Real-Time

When calling from external systems, stderr emits JSON progress lines:

```json
{"type": "progress", "stage": "separation",    "progress": 0.0}
{"type": "progress", "stage": "separation",    "progress": 1.0}
{"type": "progress", "stage": "transcription", "progress": 0.5}
{"type": "progress", "stage": "transcription", "progress": 1.0}
{"type": "progress", "stage": "alignment",     "progress": 0.3}
{"type": "progress", "stage": "alignment",     "progress": 1.0}
```

Parse these lines to drive progress bars, status displays, or other UI updates.

### GPU Memory Optimization

For large files or low-VRAM setups:

```bash
# Use CPU for separation (slower but low memory)
export CUDA_VISIBLE_DEVICES=""
uv run main.py karaoke "song.mp3" "./output/" --device cpu
```

### Custom Whisper Models

The `--model` argument accepts Whisper size variants:

```bash
uv run main.py transcribe "file.mp3" "./output/" --model tiny     # Fastest
uv run main.py transcribe "file.mp3" "./output/" --model base     # Balanced
uv run main.py transcribe "file.mp3" "./output/" --model small    # Better accuracy
uv run main.py transcribe "file.mp3" "./output/" --model medium   # High quality
uv run main.py transcribe "file.mp3" "./output/" --model large-v3 # Best (default)
```

---

## Alignment Strategy & Validation

### Word Boundary Timing

The karaoke pipeline extracts **word-level timestamps** (when each word starts/ends) from Whisper's native output. This is critical for accurate lyric highlighting.

We extensively tested alternative alignment approaches on the **Kiritan Singing Database** (50 songs, 24-bit/96kHz, with ground-truth phoneme labels):

| Approach | Median Error | Within ±300ms | Notes |
|----------|--------------|---------------|-------|
| **Whisper (native timestamps)** | **46.3ms** | **86.5%** | ✅ **WINNER** — Best overall |
| Qwen3 Forced Aligner | 44.4ms | 96.7% | ⚠️ Drops 48% of boundaries outside vocal region |
| Whisper + Qwen Alignment | 39.8ms | 98.9% | ❌ Overfits to GT phonemes, poor generalization |
| WhisperX (forced alignment) | 114,989ms | 8.6% | ❌ Catastrophic due to VAD hallucinations |
| Whisper + WhisperX | 84.2ms | 72.8% | ❌ Degrades baseline by 82% |

**Key Finding:** Whisper's built-in word timestamps are **production-ready**. The karaoke-tuned configuration (`vad_filter=False`, `no_speech_threshold=0.8`) ensures robustness on singing audio:
- Disables Voice Activity Detection (VAD) to catch sustained vowels
- Raises no-speech threshold so quiet passages aren't missed
- Relaxes compression ratio check to avoid timeouts on melisma
- Disables context chaining to prevent hallucinations

Alternative approaches (Qwen, WhisperX) either:
- Hallucinate on karaoke audio due to speech-optimized defaults
- Overfit to phoneme-level precision and fail on new songs
- Produce catastrophic outliers on sustained notes

**Conclusion:** Stick with Whisper. Native timestamps are the pragmatic choice for diverse singing styles and languages.

### Testing Scripts

Reference implementations are in `tests/`:
- `test_karaoke_scoring.ipynb` — Full validation notebook (Whisper accuracy, pitch extraction, karaoke JSON schema)
- `test_whisperx_alignment.py` — Hybrid test (demonstrates why Whisper beats Whisper+WhisperX)

Run the alignment test:
```bash
uv run python tests/test_whisperx_alignment.py
```

Results saved to `TestOutput/whisperx_hybrid_results.json`.

---

## Output File Organization

After running a command, the output directory contains:

```
output/
├── song_vocals.wav            # Lead vocal track
├── song_instrumental.wav       # Instrumental backing
├── song_karaoke.json          # Karaoke data (schema v1.0.0)
├── song_lead.wav              # (if transcribe + lead/backing separation)
├── song_backing.wav           # (if transcribe + lead/backing separation)
└── song_<stem>.wav            # (if stems command: drums, bass, guitar, piano, other)
```

---

## Troubleshooting

### Out of Memory (CUDA)
```bash
# Use CPU fallback
uv run main.py karaoke "song.mp3" "./output/" --device cpu
```

### No audio output files
- Check that input file exists and is valid audio
- Verify `ffmpeg` is installed: `ffmpeg -version`
- Use `--verbose` for detailed diagnostics

### Transcription quality is poor
- Use a larger Whisper model: `--model large-v3`
- Ensure vocals are well-separated (separation quality affects transcription)
- Check audio language detection in output JSON

### Models failing to download
- Verify internet connection
- Check HuggingFace account (some models require login)
- Models are cached in `./models/` — manually inspecting there can help

---

## Model References

| Model | Purpose | Size | Notes |
|-------|---------|------|-------|
| BS-Roformer (6-stem) | Full instrument separation | ~500MB | Used in `stems` command |
| Mel-Band Roformer (2-stem) | Karaoke vocal/instrumental | ~500MB | Optimized, used by default in `karaoke` |
| Faster-Whisper (large-v3) | Transcription + word timestamps | ~3GB | Karaoke-tuned config (VAD disabled, relaxed thresholds) |
| Torch-FCPE | Pitch extraction | ~250MB | Optional, used in advanced scoring |
| ClearVoice/MossFormer2 | Vocal enhancement | ~1GB | Optional, used in `clean` command |

**Note on alignment:** Prior versions tested Qwen Forced Aligner and WhisperX. Both underperform Whisper's native word timestamps on singing audio. See [Alignment Strategy &amp; Validation](#alignment-strategy--validation) for details.

---

## Performance Benchmarks

Typical processing times on RTX 3090 (for 4-minute song):

| Stage | Time |
|-------|------|
| Separation (2-stem) | 20–40s |
| Transcription + alignment (Whisper) | 20–40s |
| **Total** | **40–80s** |

CPU (~50× slower) or older GPUs will take proportionally longer.

---

## License & Attribution

- **BS-Roformer** & **Mel-Band Roformer**: [Audio-Separator](https://github.com/jarredou/audio-separator)
- **Faster-Whisper**: [OpenAI Whisper](https://github.com/guillaumekln/faster-whisper) (OpenAI)
- **Torch-FCPE**: [Pitch extraction](https://github.com/CNChTu/Wav2Spec2Wav) (Chen et al.)
- **ClearVoice**: [Vocally](https://github.com/Vocoderxd/clearvoice-enhancer)
- **Kiritan Singing Database**: [Ground-truth validation](https://github.com/kiritan-pop/Kiritan_Singing_DB) (phoneme & MIDI labels)

---

## Contributing & Feedback

For bugs, feature requests, or integration questions, open an issue or contact the maintainer.
