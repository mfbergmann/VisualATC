"""FastAPI application for VisualATC."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio_ingest import AudioIngest, save_upload
from .models import (
    ATCEvent,
    EventType,
    ExtractedFields,
    InputMode,
    Mention,
    StartRequest,
    TranscriptSegment,
    WhisperModel,
)
from .nlp_extractor import ATCExtractor
from .session_manager import (
    SessionManager,
    add_bookmark,
    load_bookmarks,
    remove_bookmark,
)
from .transcriber import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("visualatc.app")

app = FastAPI(title="VisualATC", version="1.0.0")

# Serve static files
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
session_mgr = SessionManager()
transcriber: Optional[Transcriber] = None
extractor = ATCExtractor()
audio_ingest: Optional[AudioIngest] = None
transcription_task: Optional[asyncio.Task] = None
connected_websockets: list[WebSocket] = []


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

async def broadcast(msg: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    dead = []
    data = json.dumps(msg, default=str)
    for ws in connected_websockets:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_websockets.remove(ws)


# ---------------------------------------------------------------------------
# Transcription loop
# ---------------------------------------------------------------------------

async def transcription_loop(ingest: AudioIngest) -> None:
    """Main transcription loop: read audio chunks, transcribe, extract, broadcast."""
    global transcriber, session_mgr

    if not transcriber or not transcriber.is_loaded:
        logger.error("Transcriber not loaded")
        return

    chunk_count = 0
    base_ts = time.time()

    try:
        async for audio_chunk in ingest.chunks():
            if not session_mgr.state or not session_mgr.state.is_running:
                break

            if session_mgr.state.is_paused:
                await asyncio.sleep(0.5)
                continue

            chunk_count += 1
            chunk_duration = len(audio_chunk) / 16000  # samples / sample_rate
            session_mgr.state.total_audio_seconds += chunk_duration
            chunk_ts = time.time()

            # Transcribe in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(
                None, transcriber.transcribe_chunk, audio_chunk
            )

            for seg in segments:
                text = seg["text"]
                confidence = seg.get("confidence", 0.0)

                # Create transcript segment
                ts_segment = TranscriptSegment(
                    ts=chunk_ts,
                    text=text,
                    confidence=confidence,
                    raw=text,
                    duration=seg.get("end", 0) - seg.get("start", 0),
                )
                session_mgr.add_transcript_segment(ts_segment)

                # Extract entities
                extraction = extractor.extract_all(text, chunk_ts)

                # Process callsigns -> mentions
                for cs_info in extraction["callsigns"]:
                    mention = Mention(
                        ts=chunk_ts,
                        callsign_canonical=cs_info["canonical"],
                        aliases=[cs_info["alias"]],
                        extracted_fields=extraction["fields"],
                        raw_text=text,
                    )
                    session_mgr.add_mention(mention)

                # Process events
                for event in extraction["events"]:
                    session_mgr.add_event(event)

                # Broadcast update to all connected clients
                await broadcast({
                    "type": "transcript",
                    "segment": ts_segment.model_dump(),
                    "callsigns": extraction["callsigns"],
                    "fields": extraction["fields"].model_dump(),
                    "events": [e.model_dump() for e in extraction["events"]],
                })

            # Periodically broadcast full state update
            if chunk_count % 5 == 0:
                await broadcast({
                    "type": "state_update",
                    "flight_cards": {
                        cs: card.model_dump()
                        for cs, card in session_mgr.state.flight_cards.items()
                    },
                    "events": [e.model_dump() for e in session_mgr.state.events[-50:]],
                    "stats": session_mgr.get_state_snapshot(),
                })

    except asyncio.CancelledError:
        logger.info("Transcription loop cancelled")
    except Exception as e:
        logger.exception("Transcription loop error: %s", e)
        await broadcast({"type": "error", "message": str(e)})
    finally:
        await broadcast({"type": "stopped"})
        if session_mgr.state:
            session_mgr.state.is_running = False


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/start")
async def start_session(req: StartRequest):
    """Start a new transcription session from a stream URL."""
    global transcriber, audio_ingest, transcription_task

    # Stop any existing session
    await _stop_session()

    # Load model if needed
    if transcriber is None or transcriber.model_size != req.model_size:
        transcriber = Transcriber(model_size=req.model_size)
        await broadcast({"type": "status", "message": f"Loading {req.model_size} model..."})
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, transcriber.load_model)
        await broadcast({"type": "status", "message": "Model loaded."})

    # Create session
    session_mgr.create_session(req.url, req.mode, req.model_size)

    # Start audio ingest
    is_file = req.mode == InputMode.FILE
    audio_ingest = AudioIngest(
        source=req.url,
        chunk_seconds=req.chunk_duration,
        is_file=is_file,
    )
    await audio_ingest.start()

    # Start transcription loop
    transcription_task = asyncio.create_task(transcription_loop(audio_ingest))

    await broadcast({"type": "started", "session_id": session_mgr.state.session_id})

    return {"status": "started", "session_id": session_mgr.state.session_id}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), model_size: str = "small"):
    """Upload a local audio file and start transcription."""
    global transcriber, audio_ingest, transcription_task

    await _stop_session()

    # Save upload
    data = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix
    tmp_path = await save_upload(data, suffix=suffix)

    # Load model if needed
    wm = WhisperModel(model_size)
    if transcriber is None or transcriber.model_size != model_size:
        transcriber = Transcriber(model_size=model_size)
        await broadcast({"type": "status", "message": f"Loading {model_size} model..."})
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, transcriber.load_model)
        await broadcast({"type": "status", "message": "Model loaded."})

    # Create session
    session_mgr.create_session(file.filename or "uploaded_file", InputMode.FILE, wm)

    # Start audio ingest
    audio_ingest = AudioIngest(source=tmp_path, chunk_seconds=7.0, is_file=True)
    await audio_ingest.start()

    # Start transcription loop
    transcription_task = asyncio.create_task(transcription_loop(audio_ingest))

    await broadcast({"type": "started", "session_id": session_mgr.state.session_id})
    return {"status": "started", "session_id": session_mgr.state.session_id}


@app.post("/api/pause")
async def pause_session():
    if session_mgr.state:
        session_mgr.state.is_paused = not session_mgr.state.is_paused
        status = "paused" if session_mgr.state.is_paused else "resumed"
        await broadcast({"type": status})
        return {"status": status}
    return {"status": "no_session"}


@app.post("/api/stop")
async def stop_session():
    result = await _stop_session()
    return result


async def _stop_session() -> dict:
    """Stop the current session."""
    global audio_ingest, transcription_task

    if transcription_task and not transcription_task.done():
        transcription_task.cancel()
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass

    if audio_ingest:
        await audio_ingest.stop()
        audio_ingest = None

    result = session_mgr.finalize_session()
    await broadcast({"type": "stopped", **result})
    return {"status": "stopped", **result}


@app.get("/api/state")
async def get_state():
    """Get current session state."""
    if not session_mgr.state:
        return {"active": False}

    return {
        "active": True,
        **session_mgr.get_state_snapshot(),
        "transcript": [s.model_dump() for s in session_mgr.state.transcript[-100:]],
        "flight_cards": {
            cs: card.model_dump()
            for cs, card in session_mgr.state.flight_cards.items()
        },
        "events": [e.model_dump() for e in session_mgr.state.events[-100:]],
    }


@app.get("/api/transcript")
async def get_transcript():
    """Get full transcript."""
    if not session_mgr.state:
        return {"segments": []}
    return {
        "segments": [s.model_dump() for s in session_mgr.state.transcript]
    }


@app.get("/api/export/txt")
async def export_txt():
    """Export transcript as .txt file."""
    result = session_mgr.finalize_session()
    if "export_txt" in result:
        return FileResponse(
            result["export_txt"],
            media_type="text/plain",
            filename=f"visualatc_{session_mgr.state.session_id}.txt",
        )
    return JSONResponse({"error": "No active session"}, status_code=404)


@app.get("/api/export/json")
async def export_json():
    """Export entities/events as .json file."""
    result = session_mgr.finalize_session()
    if "export_json" in result:
        return FileResponse(
            result["export_json"],
            media_type="application/json",
            filename=f"visualatc_{session_mgr.state.session_id}.json",
        )
    return JSONResponse({"error": "No active session"}, status_code=404)


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

@app.get("/api/bookmarks")
async def get_bookmarks():
    return {"bookmarks": [b.model_dump() for b in load_bookmarks()]}


@app.post("/api/bookmarks")
async def create_bookmark(label: str = "", url: str = ""):
    if not label or not url:
        return JSONResponse({"error": "label and url required"}, status_code=400)
    bookmarks = add_bookmark(label, url)
    return {"bookmarks": [b.model_dump() for b in bookmarks]}


@app.delete("/api/bookmarks")
async def delete_bookmark(url: str = ""):
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    bookmarks = remove_bookmark(url)
    return {"bookmarks": [b.model_dump() for b in bookmarks]}


# ---------------------------------------------------------------------------
# Telephony map
# ---------------------------------------------------------------------------

@app.get("/api/telephony")
async def get_telephony():
    return extractor.telephony_map


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_websockets.append(ws)
    logger.info("WebSocket client connected (%d total)", len(connected_websockets))

    try:
        # Send current state on connect
        if session_mgr.state:
            await ws.send_text(json.dumps({
                "type": "state_update",
                "flight_cards": {
                    cs: card.model_dump()
                    for cs, card in session_mgr.state.flight_cards.items()
                },
                "events": [e.model_dump() for e in session_mgr.state.events[-50:]],
                "stats": session_mgr.get_state_snapshot(),
                "transcript": [s.model_dump() for s in session_mgr.state.transcript[-50:]],
            }, default=str))

        # Keep connection alive
        while True:
            try:
                data = await ws.receive_text()
                # Handle client messages if needed
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                break
    finally:
        if ws in connected_websockets:
            connected_websockets.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(connected_websockets))
