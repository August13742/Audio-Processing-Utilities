"""
Optional HTTP + WebSocket API for external karaoke system integration.

Install server dependencies (not required for CLI usage):
    uv add fastapi uvicorn[standard]

Start the server:
    uv run main.py serve --host 0.0.0.0 --port 8000

─── REST ────────────────────────────────────────────────────────────────
POST /api/karaoke
    Body:  {"audio_path": "/path/to/song.mp3", "output_dir": "/path/to/out"}
    Reply: Full karaoke JSON (same schema as the file output)

GET /api/health
    Reply: {"status": "ok"}

─── WebSocket ───────────────────────────────────────────────────────────
ws://host:port/ws/karaoke
    Send:    {"audio_path": "...", "output_dir": "..."}
    Receive: {"type": "progress", "stage": "separation", "progress": 0.5}
             {"type": "progress", "stage": "transcription", "progress": 1.0}
             ...
             {"type": "result",   "data": { ...full karaoke JSON... }}
             {"type": "error",    "message": "..."}
"""

from __future__ import annotations

import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(device: str = "cuda") -> "FastAPI":
    """
    Factory that returns a configured FastAPI app.
    Lazily initialises the KaraokePipeline on first request so the server
    starts quickly and models are loaded only when needed.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Run:  uv add fastapi uvicorn[standard]"
        )

    app = FastAPI(title="Audio Utilities – Karaoke API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy singleton — heavy model loading deferred until first call
    _pipeline_holder: dict = {}

    def _get_pipeline():
        if "instance" not in _pipeline_holder:
            from .karaoke import KaraokePipeline

            _pipeline_holder["instance"] = KaraokePipeline(device=device)
        return _pipeline_holder["instance"]

    # ── Models ──────────────────────────────────────────────────────────

    class KaraokeRequest(BaseModel):
        audio_path: str
        output_dir: str

    # ── REST endpoints ──────────────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/karaoke")
    async def karaoke_rest(req: KaraokeRequest):
        pipeline = _get_pipeline()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.process_song(req.audio_path, req.output_dir),
        )
        if not result:
            return {"error": "Pipeline failed — check server logs."}
        return result

    # ── WebSocket endpoint ──────────────────────────────────────────────

    @app.websocket("/ws/karaoke")
    async def karaoke_ws(ws: WebSocket):
        await ws.accept()
        try:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            audio_path = msg["audio_path"]
            output_dir = msg["output_dir"]

            pipeline = _get_pipeline()
            loop = asyncio.get_event_loop()

            # Progress callback → sends JSON frames over the socket
            async def _send_progress(stage: str, progress: float):
                await ws.send_json(
                    {"type": "progress", "stage": stage, "progress": round(progress, 3)}
                )

            def _sync_progress(stage: str, progress: float):
                asyncio.run_coroutine_threadsafe(_send_progress(stage, progress), loop)

            result = await loop.run_in_executor(
                None,
                lambda: pipeline.process_song(
                    audio_path, output_dir, on_progress=_sync_progress
                ),
            )

            if result:
                await ws.send_json({"type": "result", "data": result})
            else:
                await ws.send_json({"type": "error", "message": "Pipeline returned empty result."})

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
            traceback.print_exc()
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, device: str = "cuda"):
    """Convenience wrapper to start Uvicorn programmatically."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Run:  uv add fastapi uvicorn[standard]"
        )
    import uvicorn

    app = create_app(device=device)
    uvicorn.run(app, host=host, port=port)
