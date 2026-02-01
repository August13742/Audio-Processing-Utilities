# Rhythm Engine - Vocal Processing Toolkit

A comprehensive audio processing toolkit for separating, transcribing, and building vocal datasets.
Powered by BS-Roformer, Mel-Band Roformer, Faster-Whisper, and Audio-Separator.

## Setup

This project uses `uv` for dependency management.

```bash
uv sync --prerelease=allow
```

## Usage

The `main.py` script is the single entry point for all tools.

### 1. Separate Vocals & Instrumental (BGM)
Uses BS-Roformer to separate Lead Vocals from the backing track.

```bash
uv run main.py separate-bgm "path/to/song.mp3" "output_folder/"
```

### 2. Transcribe Audio
Uses Faster-Whisper (Large-v3 default) to transcribe audio files.

```bash
uv run main.py transcribe "path/to/audio_folder/" --model large-v3
```

### 3. Build Vocal Dataset (Slicing)
Full pipeline: Separate -> VAD Slicing -> Diversity Selection -> Transcription -> Dataset Manifest.
Ideal for TTS training or vocal analysis.

```bash
uv run main.py dataset "path/to/raw_audio/" "output_dataset/"
```

### 4. Process Whole Songs
Pipeline: Separate -> Isolate Lead -> Transcribe Whole Lead.

```bash
uv run main.py pipeline "path/to/song_folder/" "output_folder/"
```

## Project Structure

- `tools/`: Core logic modules
  - `separation.py`: BS-Roformer/Mel-Roformer wrapper
  - `transcription.py`: Faster-Whisper wrapper
  - `dataset.py`: Slicing and dataset building logic
- `utils/`: Helpers for audio IO and processing
- `legacy/`: Old scripts (separator.py, etc)
