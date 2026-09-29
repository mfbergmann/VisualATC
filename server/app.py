"""FastAPI application for VisualATC."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio_ingest import AudioIngest, USER_AGENT, save_upload
from .models import InputMode, Mention, StartRequest, TranscriptSegment
from .nlp_extractor import ATCExtractor
from .segmenter import SAMPLE_RATE, Transmission, TransmissionSegmenter
from .session_manager import (
    SessionManager,
    add_bookmark,
    load_bookmarks,
    remove_bookmark,
)
from .transcriber import ASREngine, MODELS_BY_ID, check_available, create_engine, list_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("visualatc.app")

app = FastAPI(title="VisualATC", version="1.1.0")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Live streams: if transcription falls this many transmissions behind, the
# oldest queued ones are dropped so the transcript stays near real time.
MAX_STREAM_BACKLOG = 10
# Files: bounded queue gives back-pressure to ffmpeg instead of dropping.
FILE_QUEUE_SIZE = 4

# Global state
session_mgr = SessionManager()
transcriber: Optional[ASREngine] = None
extractor = ATCExtractor()
audio_ingest: Optional[AudioIngest] = None
transcription_task: Optional[asyncio.Task] = None
connected_websockets: list[WebSocket] = []
resolved_stream_url: Optional[str] = None  # actual stream URL after playlist resolution
upload_path: Optional[str] = None           # temp file for the current upload
model_lock = asyncio.Lock()


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
        if ws in connected_websockets:
            connected_websockets.remove(ws)


def _state_message() -> dict:
    return {
        "type": "state_update",
        "flight_cards": {
            cs: card.model_dump() for cs, card in session_mgr.state.flight_cards.items()
        },
        "events": [e.model_dump() for e in session_mgr.state.events[-50:]],
        "stats": session_mgr.get_state_snapshot(),
    }


# ---------------------------------------------------------------------------
# Pipeline: ffmpeg -> segmenter -> queue -> ASR -> extraction -> UI
# ---------------------------------------------------------------------------

async def reader_loop(ingest: AudioIngest, queue: asyncio.Queue, is_file: bool) -> None:
    """Read PCM from ffmpeg, cut it into transmissions and queue them."""
    segmenter = TransmissionSegmenter()
    got_audio = False
    wall_start = time.time()

    async def enqueue(tx: Transmission) -> None:
        state = session_mgr.state
        # Wall-clock time the transmission started. For files, decoding is
        # faster than real time, so use the position in the file instead.
        if is_file:
            ts = state.started_at + tx.start_s
        else:
            ts = time.time() - segmenter.seconds_since(tx.start_s)
        item = (tx, ts, time.time())
        if is_file:
            await queue.put(item)
            return
        if state.is_paused:
            return
        while queue.qsize() >= MAX_STREAM_BACKLOG:
            queue.get_nowait()
            state.dropped_transmissions += 1
            if state.dropped_transmissions in (1, 10) or state.dropped_transmissions % 50 == 0:
                await broadcast({
                    "type": "status",
                    "message": (
                        f"Transcription is falling behind the stream; skipped "
                        f"{state.dropped_transmissions} transmission(s). A faster model "
                        f"or a GPU will help."
                    ),
                })
        queue.put_nowait(item)

    try:
        async for block in ingest.blocks():
            if not got_audio:
                got_audio = True
                await broadcast({"type": "status", "message": f"Receiving audio from: {ingest.source}"})
            state = session_mgr.state
            if not state or not state.is_running:
                break
            state.total_audio_seconds += len(block) / SAMPLE_RATE
            for tx in segmenter.push(block):
                await enqueue(tx)

        for tx in segmenter.flush():
            await enqueue(tx)

        if not got_audio:
            detail = " | ".join(ingest.stderr_tail) or "no output from ffmpeg"
            await broadcast({
                "type": "error",
                "message": (
                    "No audio received. Check that the URL is a live audio stream "
                    f"(ffmpeg: {detail})"
                ),
            })
        else:
            logger.info("Audio input ended after %.1fs", time.time() - wall_start)
    except asyncio.CancelledError:
        raise
    except Exception:
        await queue.put(None)
        raise
    await queue.put(None)  # sentinel: no more audio


async def transcriber_loop(queue: asyncio.Queue, engine: ASREngine) -> None:
    """Transcribe queued transmissions and publish results."""
    loop = asyncio.get_running_loop()
    count = 0
    while True:
        item = await queue.get()
        if item is None:
            break
        tx, ts, queued_at = item
        state = session_mgr.state
        if not state or not state.is_running:
            break
        while state.is_paused and state.is_running:
            await asyncio.sleep(0.25)

        segments = await loop.run_in_executor(None, engine.transcribe_chunk, tx.audio)
        latency = time.time() - queued_at
        count += 1

        for seg in segments:
            raw = seg["text"]
            extraction = extractor.extract_all(raw, ts)
            ts_segment = TranscriptSegment(
                ts=ts + seg.get("start", 0.0),
                text=extraction["normalized"],
                raw=raw,
                confidence=seg.get("confidence"),
                duration=seg.get("end", 0.0) - seg.get("start", 0.0),
                audio_offset=tx.start_s + seg.get("start", 0.0),
                latency=round(latency, 2),
            )
            session_mgr.add_transcript_segment(ts_segment)

            for cs_info in extraction["callsigns"]:
                session_mgr.add_mention(Mention(
                    ts=ts_segment.ts,
                    callsign_canonical=cs_info["canonical"],
                    aliases=[cs_info["alias"]],
                    extracted_fields=extraction["fields"],
                    raw_text=ts_segment.text,
                ))
            for event in extraction["events"]:
                session_mgr.add_event(event)

            await broadcast({
                "type": "transcript",
                "segment": ts_segment.model_dump(),
                "callsigns": extraction["callsigns"],
                "fields": extraction["fields"].model_dump(),
                "events": [e.model_dump() for e in extraction["events"]],
            })

        if count % 3 == 0:
            await broadcast(_state_message())


async def run_pipeline(ingest: AudioIngest, engine: ASREngine, is_file: bool) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=FILE_QUEUE_SIZE if is_file else 0)
    reader = asyncio.create_task(reader_loop(ingest, queue, is_file))
    try:
        await transcriber_loop(queue, engine)
        if reader.done() and not reader.cancelled() and reader.exception():
            raise reader.exception()
    except asyncio.CancelledError:
        logger.info("Transcription pipeline cancelled")
        raise
    except Exception as e:
        logger.exception("Transcription pipeline error: %s", e)
        await broadcast({"type": "error", "message": str(e)})
    finally:
        if not reader.done():
            # The reader may be blocked on a full queue after a stop.
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        if session_mgr.state:
            session_mgr.state.is_running = False
            await broadcast(_state_message())
            await broadcast({"type": "finished"})


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
async def get_models():
    return {
        "models": list_models(),
        "loaded": transcriber.model_id if transcriber else None,
        "default": "small",
    }


async def _ensure_engine(model_id: str) -> ASREngine:
    """Load (or reuse) the engine for ``model_id``."""
    global transcriber
    spec = MODELS_BY_ID.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown model '{model_id}'")
    ok, reason = check_available(spec)
    if not ok:
        raise ValueError(f"{spec.label} is not installed. {reason}")

    async with model_lock:
        if transcriber is None or transcriber.model_id != model_id:
            engine = create_engine(model_id)
            await broadcast({"type": "status", "message": f"Loading {spec.label} (first use downloads {spec.size})..."})
            await asyncio.get_running_loop().run_in_executor(None, engine.load_model)
            transcriber = engine
            await broadcast({"type": "status", "message": f"Model loaded on {engine.device}."})
    return transcriber


async def _begin(source: str, label: str, mode: InputMode, model_id: str, radio_filter: bool) -> dict:
    global audio_ingest, transcription_task, resolved_stream_url

    try:
        engine = await _ensure_engine(model_id)
    except Exception as e:
        logger.exception("Model load failed")
        return {"status": "error", "error": f"Could not load model: {e or type(e).__name__}"}

    is_file = mode == InputMode.FILE
    ingest = AudioIngest(source=source, is_file=is_file, radio_filter=radio_filter)
    try:
        await ingest.start()
    except Exception as e:
        return {"status": "error", "error": f"Could not open audio: {e or type(e).__name__}"}

    session_mgr.create_session(label, mode, model_id)
    audio_ingest = ingest
    resolved_stream_url = None if is_file else ingest.source
    transcription_task = asyncio.create_task(run_pipeline(ingest, engine, is_file))

    info = {
        "session_id": session_mgr.state.session_id,
        "stream_url": resolved_stream_url,
        "mode": mode.value,
        "model": engine.spec.label,
        "device": engine.device,
    }
    await broadcast({"type": "started", **info})
    return {"status": "started", **info}


@app.post("/api/start")
async def start_session(req: StartRequest):
    """Start a new transcription session from a stream URL."""
    await _stop_session()
    if not req.url:
        return JSONResponse({"status": "error", "error": "url required"}, status_code=400)
    result = await _begin(req.url, req.url, req.mode, req.model_size, req.radio_filter)
    return JSONResponse(result, status_code=200 if result["status"] == "started" else 400)


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model_size: str = Form("small"),
    radio_filter: bool = Form(True),
):
    """Upload a local audio file and start transcription."""
    global upload_path
    await _stop_session()

    data = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    upload_path = await save_upload(data, suffix=suffix)

    result = await _begin(upload_path, file.filename or "uploaded_file", InputMode.FILE,
                          model_size, radio_filter)
    return JSONResponse(result, status_code=200 if result["status"] == "started" else 400)


@app.post("/api/pause")
async def pause_session():
    if session_mgr.state and session_mgr.state.is_running:
        session_mgr.state.is_paused = not session_mgr.state.is_paused
        status = "paused" if session_mgr.state.is_paused else "resumed"
        await broadcast({"type": status})
        return {"status": status}
    return {"status": "no_session"}


@app.post("/api/stop")
async def stop_session():
    return await _stop_session()


async def _stop_session() -> dict:
    """Stop the current session and write exports."""
    global audio_ingest, transcription_task, resolved_stream_url, upload_path

    if session_mgr.state:
        session_mgr.state.is_running = False

    if audio_ingest:
        await audio_ingest.stop()
        audio_ingest = None

    if transcription_task and not transcription_task.done():
        transcription_task.cancel()
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass
    transcription_task = None
    resolved_stream_url = None

    if upload_path:
        try:
            os.unlink(upload_path)
        except OSError:
            pass
        upload_path = None

    if not session_mgr.state:
        return {"status": "no_session"}
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
    """Export transcript as .txt (does not stop the session)."""
    result = session_mgr.write_exports()
    if "export_txt" in result:
        return FileResponse(
            result["export_txt"],
            media_type="text/plain",
            filename=f"visualatc_{session_mgr.state.session_id}.txt",
        )
    return JSONResponse({"error": "No active session"}, status_code=404)


@app.get("/api/export/json")
async def export_json():
    """Export entities/events as .json (does not stop the session)."""
    result = session_mgr.write_exports()
    if "export_json" in result:
        return FileResponse(
            result["export_json"],
            media_type="application/json",
            filename=f"visualatc_{session_mgr.state.session_id}.json",
        )
    return JSONResponse({"error": "No active session"}, status_code=404)


# ---------------------------------------------------------------------------
# Audio proxy – lets the browser play the stream in an <audio> element
# ---------------------------------------------------------------------------

@app.get("/api/audio-proxy")
async def audio_proxy():
    """Proxy the resolved audio stream so the browser can play it."""
    if not resolved_stream_url:
        return JSONResponse({"error": "No active stream"}, status_code=404)

    async def stream_audio():
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
                headers={"User-Agent": USER_AGENT},
            ) as client:
                async with client.stream("GET", resolved_stream_url) as resp:
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        yield chunk
        except Exception as e:
            logger.warning("Audio proxy error: %s", e)

    # Try to infer content type from URL
    url_lower = resolved_stream_url.lower()
    if ".aac" in url_lower:
        media_type = "audio/aac"
    elif ".ogg" in url_lower:
        media_type = "audio/ogg"
    else:
        media_type = "audio/mpeg"  # MP3 is most common for Icecast/Shoutcast

    return StreamingResponse(
        stream_audio(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/stream-url")
async def get_stream_url():
    """Return the resolved stream URL for the current session."""
    return {"stream_url": resolved_stream_url or ""}


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
                **_state_message(),
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
