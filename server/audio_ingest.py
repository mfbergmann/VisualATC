"""Audio ingest – decode streams/files to 16kHz mono PCM via ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger("visualatc.audio")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


class AudioIngest:
    """Async generator that yields PCM float32 chunks from a stream or file."""

    def __init__(self, source: str, chunk_seconds: float = 7.0, is_file: bool = False):
        self.source = source
        self.chunk_seconds = chunk_seconds
        self.is_file = is_file
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stopped = False
        self.chunk_size = int(SAMPLE_RATE * chunk_seconds * 4)  # float32 = 4 bytes

    def _build_ffmpeg_cmd(self) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        if not self.is_file:
            # For streams: reconnect options
            cmd += [
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-timeout", "10000000",  # 10s timeout in microseconds
            ]

        cmd += [
            "-i", self.source,
            "-vn",                     # no video
            "-acodec", "pcm_f32le",    # float32 little-endian
            "-ar", str(SAMPLE_RATE),   # 16kHz
            "-ac", str(CHANNELS),      # mono
            "-f", "f32le",             # raw PCM output
            "pipe:1",                  # stdout
        ]
        return cmd

    async def start(self) -> None:
        if not check_ffmpeg():
            raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

        cmd = self._build_ffmpeg_cmd()
        logger.info("Starting ffmpeg: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._process and self._process.returncode is None:
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Yield numpy float32 arrays of audio chunks."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("AudioIngest not started")

        buffer = b""
        while not self._stopped:
            try:
                data = await asyncio.wait_for(
                    self._process.stdout.read(self.chunk_size - len(buffer)),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                if self._stopped:
                    break
                if buffer:
                    # Yield what we have
                    arr = np.frombuffer(buffer, dtype=DTYPE).copy()
                    buffer = b""
                    yield arr
                continue

            if not data:
                # EOF
                if buffer:
                    arr = np.frombuffer(buffer, dtype=DTYPE).copy()
                    yield arr
                break

            buffer += data
            if len(buffer) >= self.chunk_size:
                arr = np.frombuffer(buffer[: self.chunk_size], dtype=DTYPE).copy()
                buffer = buffer[self.chunk_size:]
                yield arr

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and not self._stopped
        )


async def save_upload(data: bytes, suffix: str = ".wav") -> str:
    """Save uploaded file bytes to a temp file, return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
    tmp.write(data)
    tmp.close()
    return tmp.name
