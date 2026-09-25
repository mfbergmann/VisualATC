"""Split continuous radio audio into individual transmissions.

Fixed-length chunks cut transmissions mid-word, often straight through the
callsign, and ATC-tuned models are trained on one transmission per clip.
This segmenter watches frame energy against an adaptive noise floor and cuts
at the squelch gaps between transmissions instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Transmission:
    audio: np.ndarray
    start_s: float          # offset from the start of the stream
    forced_split: bool = False

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class TransmissionSegmenter:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
        margin_db: float = 9.0,
        abs_floor_db: float = -60.0,
        start_ms: int = 60,
        end_silence_ms: int = 600,
        pre_roll_ms: int = 300,
        post_roll_ms: int = 200,
        min_speech_ms: int = 250,
        max_segment_s: float = 25.0,
        floor_window_s: float = 8.0,
        floor_percentile: float = 10.0,
    ):
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.margin_db = margin_db
        self.abs_floor_db = abs_floor_db
        self.start_frames = max(1, start_ms // frame_ms)
        self.end_frames = max(1, end_silence_ms // frame_ms)
        self.pre_frames = pre_roll_ms // frame_ms
        self.post_frames = post_roll_ms // frame_ms
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = int(max_segment_s * 1000 / frame_ms)
        self.floor_percentile = floor_percentile

        self._history: deque[float] = deque(maxlen=int(floor_window_s * 1000 / frame_ms))
        self._floor_db = abs_floor_db
        self._frames_since_floor = 0

        self._pending = np.zeros(0, dtype=np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, self.pre_frames))
        self._active = False
        self._seg: list[np.ndarray] = []
        self._seg_start_frame = 0
        self._voiced = 0
        self._silence_run = 0
        self._onset_run = 0
        self._frame_idx = 0

    # ------------------------------------------------------------------

    @property
    def noise_floor_db(self) -> float:
        return self._floor_db

    @property
    def consumed_s(self) -> float:
        return self._frame_idx * self.frame / self.sr

    def _update_floor(self, db: float) -> None:
        self._history.append(db)
        self._frames_since_floor += 1
        if self._frames_since_floor >= 10 and len(self._history) >= 10:
            self._floor_db = float(np.percentile(self._history, self.floor_percentile))
            self._frames_since_floor = 0

    def _threshold(self) -> float:
        return max(self._floor_db + self.margin_db, self.abs_floor_db)

    def _emit(self, trailing_silence: int, forced: bool) -> list[Transmission]:
        frames = self._seg
        if trailing_silence > self.post_frames:
            frames = frames[: len(frames) - (trailing_silence - self.post_frames)]
        out = []
        if self._voiced >= self.min_speech_frames and frames:
            out.append(Transmission(
                audio=np.concatenate(frames),
                start_s=self._seg_start_frame * self.frame / self.sr,
                forced_split=forced,
            ))
        self._seg = []
        self._voiced = 0
        self._silence_run = 0
        return out

    def push(self, samples: np.ndarray) -> list[Transmission]:
        """Feed audio; return any transmissions that completed."""
        if samples.size:
            self._pending = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        out: list[Transmission] = []
        n_frames = len(self._pending) // self.frame
        for i in range(n_frames):
            f = self._pending[i * self.frame:(i + 1) * self.frame]
            rms = float(np.sqrt(np.mean(f * f)))
            db = 20.0 * np.log10(rms + 1e-10)
            thr = self._threshold()
            self._update_floor(db)
            loud = db > thr

            if not self._active:
                self._pre.append(f)
                self._onset_run = self._onset_run + 1 if loud else 0
                if self._onset_run >= self.start_frames:
                    self._active = True
                    self._seg = list(self._pre)
                    self._seg_start_frame = self._frame_idx - len(self._pre) + 1
                    self._pre.clear()
                    self._voiced = self._onset_run
                    self._silence_run = 0
                    self._onset_run = 0
            else:
                self._seg.append(f)
                if loud:
                    self._voiced += 1
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                if self._silence_run >= self.end_frames:
                    out.extend(self._emit(self._silence_run, forced=False))
                    self._active = False
                elif len(self._seg) >= self.max_frames:
                    out.extend(self._emit(0, forced=True))
                    self._seg_start_frame = self._frame_idx + 1
            self._frame_idx += 1

        self._pending = self._pending[n_frames * self.frame:]
        return out

    def flush(self) -> list[Transmission]:
        """End of stream: emit whatever transmission is in progress."""
        out: list[Transmission] = []
        if self._active:
            out = self._emit(self._silence_run, forced=False)
            self._active = False
        self._pre.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        return out

    def seconds_since(self, start_s: float) -> float:
        return self.consumed_s - start_s
