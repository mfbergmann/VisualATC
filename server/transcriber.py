"""Whisper transcription wrapper using faster-whisper."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("visualatc.transcriber")


class Transcriber:
    """Wraps faster-whisper for incremental chunk transcription."""

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load_model(self) -> None:
        """Load the whisper model. Call once before transcribing."""
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model '%s' (device=%s, compute=%s)...",
                     self.model_size, self.device, self.compute_type)
        t0 = time.time()

        # Determine device and compute type
        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Model loaded in %.1fs (device=%s, compute=%s)",
                     time.time() - t0, device, compute_type)

    def transcribe_chunk(self, audio: np.ndarray) -> list[dict]:
        """
        Transcribe a chunk of float32 PCM audio at 16kHz.

        Returns list of segment dicts: {text, start, end, confidence}
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Clamp to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Skip near-silent chunks
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            return []

        segments_iter, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=3,
            best_of=3,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            without_timestamps=False,
            initial_prompt="Air traffic control communication. Callsigns, runways, altitudes, headings.",
        )

        results = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                results.append({
                    "text": text,
                    "start": seg.start,
                    "end": seg.end,
                    "confidence": round(1.0 - seg.no_speech_prob, 3) if hasattr(seg, 'no_speech_prob') else 0.0,
                })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
