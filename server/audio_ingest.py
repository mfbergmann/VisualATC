"""Audio ingest – decode streams/files to 16kHz mono PCM via ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import signal
import tempfile
from collections import deque
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import httpx
import numpy as np

logger = logging.getLogger("visualatc.audio")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


async def resolve_playlist_url(url: str) -> str:
    """Resolve .pls / .m3u / .m3u8 playlist URLs to the actual stream URL.

    If the URL is already a direct stream, returns it unchanged.
    """
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    # Check if this is a playlist file that needs resolving
    is_pls = path_lower.endswith(".pls")
    is_m3u = path_lower.endswith(".m3u") or path_lower.endswith(".m3u8")

    if not is_pls and not is_m3u:
        return url

    logger.info("Resolving playlist URL: %s", url)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.text
    except Exception as e:
        logger.warning("Failed to fetch playlist %s: %s. Passing URL directly to ffmpeg.", url, e)
        return url

    if is_pls:
        # PLS format: File1=http://...
        match = re.search(r"^File\d*\s*=\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
        if match:
            stream_url = match.group(1).strip()
            logger.info("Resolved PLS -> %s", stream_url)
            return stream_url
        logger.warning("Could not parse PLS playlist, passing original URL to ffmpeg")
        return url

    if is_m3u:
        # M3U format: lines starting with http (skip comments starting with #)
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                logger.info("Resolved M3U -> %s", line)
                return line
        logger.warning("Could not parse M3U playlist, passing original URL to ffmpeg")
        return url

    return url


# Band-limit to the AM airband voice channel. Removes hum, DC and hiss outside
# the speech band, and matches the audio chain ATC fine-tunes were trained on.
RADIO_FILTER = (
    "highpass=f=300:poles=2,highpass=f=300:poles=2,"
    "lowpass=f=3400:poles=2,lowpass=f=3400:poles=2"
)


class AudioIngest:
    """Decode a stream or file to 16 kHz mono float32 PCM with ffmpeg."""

    def __init__(self, source: str, is_file: bool = False, radio_filter: bool = True,
                 block_seconds: float = 0.1):
        self.source = source
        self.is_file = is_file
        self.radio_filter = radio_filter
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stopped = False
        self.block_bytes = int(SAMPLE_RATE * block_seconds) * 4  # float32 = 4 bytes
        self.stderr_tail: deque[str] = deque(maxlen=8)

    def _build_ffmpeg_cmd(self) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]

        if not self.is_file:
            cmd += [
                "-user_agent", USER_AGENT,
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_on_network_error", "1",
                "-reconnect_delay_max", "10",
                "-rw_timeout", "15000000",  # 15 s, microseconds
            ]

        cmd += ["-i", self.source, "-vn"]
        if self.radio_filter:
            cmd += ["-af", RADIO_FILTER]
        cmd += [
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-f", "f32le",
            "pipe:1",
        ]
        return cmd

    async def start(self) -> None:
        if not check_ffmpeg():
            raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

        if not self.is_file:
            self.source = await resolve_playlist_url(self.source)

        cmd = self._build_ffmpeg_cmd()
        logger.info("Starting ffmpeg: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._log_stderr())

    async def _log_stderr(self) -> None:
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.stderr_tail.append(text)
                    logger.warning("ffmpeg: %s", text)
        except Exception:
            pass

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
        if self._stderr_task:
            self._stderr_task.cancel()

    async def blocks(self) -> AsyncIterator[np.ndarray]:
        """Yield float32 sample blocks as ffmpeg produces them, until EOF."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("AudioIngest not started")

        remainder = b""
        while not self._stopped:
            data = await self._process.stdout.read(self.block_bytes)
            if not data:
                break
            data = remainder + data
            usable = len(data) - (len(data) % 4)
            remainder = data[usable:]
            if usable:
                yield np.frombuffer(data[:usable], dtype=DTYPE).copy()

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
