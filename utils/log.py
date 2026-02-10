"""
Centralised logging for the audio-utilities toolkit.

All internal modules should use ``from utils.log import log`` instead of
bare ``print()`` so that human-readable status messages go to **stderr**
while stdout stays clean for machine-readable output (JSON) that the
calling process (e.g. Godot) can parse.

Usage inside any module:
    from utils.log import log
    log("[SEP] Separating vocals...")      # → stderr
    log("[ERR] Something broke", err=True) # → stderr (same, but semantically marks errors)
"""

from __future__ import annotations

import sys
import json
from typing import Optional


def log(msg: str, *, err: bool = False) -> None:
    """Print a human-readable log line to stderr."""
    print(msg, file=sys.stderr, flush=True)


def emit_progress(stage: str, progress: float) -> None:
    """
    Write a machine-readable progress event to stderr as a single JSON line.

    Godot (or any caller) can optionally read stderr line-by-line and parse
    lines starting with ``{"type":"progress"`` to drive a progress bar.

    Format: {"type": "progress", "stage": "<name>", "progress": 0.0-1.0}
    """
    line = json.dumps(
        {"type": "progress", "stage": stage, "progress": round(progress, 3)},
        ensure_ascii=False,
    )
    print(line, file=sys.stderr, flush=True)
