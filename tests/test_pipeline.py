"""End-to-end: ffmpeg decode -> segmenter -> (fake) ASR -> extraction."""

import asyncio
import shutil
import wave

import numpy as np
import pytest

from server import app as app_mod
from server.audio_ingest import AudioIngest
from server.models import InputMode
from server.transcriber import ASREngine, ModelSpec

from .test_segmenter import noise, speech

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SCRIPT = [
    "Delta one four nine two descend and maintain three thousand",
    "Jazz eight eight five three go around",
]


class ScriptedEngine(ASREngine):
    def __init__(self):
        super().__init__(ModelSpec("scripted", "Scripted", "faster-whisper", "x", ""))
        self._loaded = True
        self.calls = []

    def _transcribe(self, audio):
        i = len(self.calls)
        self.calls.append(len(audio) / 16000)
        return [{"text": SCRIPT[i % len(SCRIPT)], "start": 0.0,
                 "end": len(audio) / 16000, "confidence": 0.9}]


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def test_file_pipeline(tmp_path, monkeypatch):
    audio = np.concatenate([noise(1), speech(2.5), noise(1.2), speech(2.0), noise(1)])
    wav = tmp_path / "atc.wav"
    write_wav(wav, audio)

    messages = []

    async def fake_broadcast(msg):
        messages.append(msg)

    monkeypatch.setattr(app_mod, "broadcast", fake_broadcast)
    monkeypatch.setattr("server.session_manager.SESSIONS_DIR", tmp_path / "sessions")

    async def run():
        engine = ScriptedEngine()
        mgr = app_mod.session_mgr
        mgr.create_session("atc.wav", InputMode.FILE, "scripted")
        ingest = AudioIngest(str(wav), is_file=True, radio_filter=True)
        await ingest.start()
        await asyncio.wait_for(app_mod.run_pipeline(ingest, engine, is_file=True), 30)
        await ingest.stop()
        return engine, mgr

    engine, mgr = asyncio.run(run())

    assert len(engine.calls) == 2, engine.calls
    texts = [s.text for s in mgr.state.transcript]
    assert texts[0] == "Delta 1492 descend and maintain 3000"
    assert set(mgr.state.flight_cards) == {"DAL1492", "JZA8853"}
    assert mgr.state.flight_cards["DAL1492"].latest_fields.altitude == "3000ft"
    assert [e.type.value for e in mgr.state.events] == ["GO_AROUND"]
    assert abs(mgr.state.transcript[1].audio_offset - 4.7) < 0.5
    assert any(m["type"] == "finished" for m in messages)
    # Export mid-session must not stop anything
    mgr.state.is_running = True
    mgr.write_exports()
    assert mgr.state.is_running
