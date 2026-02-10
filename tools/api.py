"""
Subprocess integration protocol for external callers (Godot, etc.)

════════════════════════════════════════════════════════════════════════
CALLING CONVENTION
════════════════════════════════════════════════════════════════════════

Every command follows the same contract:

    stdout  →  Single JSON object (the result).  Parse this.
    stderr  →  Human-readable log lines + optional JSON progress lines.
    exit 0  →  Success.  stdout contains a valid result.
    exit 1  →  Failure.  stdout may contain {"status": "error", "message": "..."}.

────────────────────────────────────────────────────────────────────────
GODOT EXAMPLE (GDScript 4.x)
────────────────────────────────────────────────────────────────────────

    func generate_karaoke(audio_path: String, out_dir: String) -> Dictionary:
        var output := []
        var exit_code := OS.execute(
            "uv",
            ["run", "main.py", "karaoke", audio_path, out_dir],
            output,    # stdout captured here
            true,      # read stderr (goes to Godot console)
            true       # open_console = true on Windows
        )
        if exit_code != 0:
            push_error("Karaoke pipeline failed")
            return {}
        return JSON.parse_string(output[0])

────────────────────────────────────────────────────────────────────────
READING PROGRESS  (optional — from stderr)
────────────────────────────────────────────────────────────────────────

If you need real-time progress (e.g. for a progress bar), read stderr
line-by-line using OS.execute_with_pipe() or a Thread + OS.create_process().
Lines that start with '{"type":"progress"' are JSON:

    {"type": "progress", "stage": "separation",    "progress": 0.0}
    {"type": "progress", "stage": "separation",    "progress": 1.0}
    {"type": "progress", "stage": "transcription", "progress": 0.0}
    ...
    {"type": "progress", "stage": "alignment",     "progress": 1.0}

All other stderr lines are plain human-readable log text (ignore them
or show them in a debug console).

════════════════════════════════════════════════════════════════════════
AVAILABLE COMMANDS
════════════════════════════════════════════════════════════════════════

┌─────────────┬─────────────────────────────────────────────────────────┐
│ Command     │ stdout result schema                                    │
├─────────────┼─────────────────────────────────────────────────────────┤
│ karaoke     │ {version, metadata, audio, lines[], words[]}            │
│ transcribe  │ {status, results: [{file, transcript_path, text, lang}]}│
│ stems       │ {status, stem_dirs: [path, ...]}                        │
│ clean       │ {status, output_dir}                                    │
│ segment     │ {status, segment_dirs: [path, ...]}                     │
│ convert     │ {status, output_file | output_dir}                      │
│ dataset     │ {status, output_dir}                                    │
└─────────────┴─────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
KARAOKE JSON SCHEMA  (v1.0.0)
════════════════════════════════════════════════════════════════════════

{
  "version": "1.0.0",
  "metadata": {
    "title":            "Song Name",
    "language":         "en",                    // ISO 639-1
    "duration_seconds": 234.5,
    "created_at":       "2026-02-10T12:00:00+00:00"
  },
  "audio": {
    "original":      "Song Name.mp3",            // input filename
    "vocals":        "Song Name_vocals.wav",     // relative to output_dir
    "instrumental":  "Song Name_instrumental.wav"
  },
  "lines": [                                     // sentence-level (scrolling display)
    {
      "id": 0,
      "text": "The last snowflake once fell on my nose",
      "start": 23.44,
      "end": 28.0,
      "word_start_index": 0,                     // index into words[]
      "word_end_index": 8                         // exclusive
    }
  ],
  "words": [                                     // word-level (highlight sweep)
    {
      "text": "The",
      "start": 23.44,
      "end": 23.68,
      "line_id": 0                               // back-reference to lines[]
    }
  ]
}

Rendering algorithm:
  1. Play audio.instrumental
  2. At current playback time t, find active line:  lines[i] where start <= t < end
  3. Slice words for that line:  words[line.word_start_index : line.word_end_index]
  4. Highlight word where word.start <= t < word.end
  5. Scroll to next line when t >= line.end
"""
