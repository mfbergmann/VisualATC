"""Whisper transcription wrapper using faster-whisper."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("visualatc.transcriber")

# Realistic ATC prompt – gives Whisper context about the domain without
# being a sentence it will parrot back.  Using actual ATC-style phrasing
# steers the decoder toward correct vocabulary.
ATC_PROMPT = (
    "Delta 1492 runway two eight left cleared for takeoff. "
    "United 237 descend and maintain flight level two four zero. "
    "American 985 contact approach one two four point six. "
    "November seven two three alpha bravo turn right heading two seven zero. "
    "Jazz 8853 cleared ILS runway two zero right. "
    "Southwest 418 reduce speed to one eight zero knots. "
    "Air Canada 881 hold short runway three one. "
    "Go around, wind shear alert."
)

# Phrases that Whisper tends to hallucinate during silence on radio feeds.
# We strip these from output.
HALLUCINATION_PATTERNS = [
    re.compile(r"(?i)air traffic control communication"),
    re.compile(r"(?i)callsigns?,?\s*runways?,?\s*altitudes?,?\s*headings?"),
    re.compile(r"(?i)thank you for (?:watching|listening)"),
    re.compile(r"(?i)please (?:like and )?subscribe"),
    re.compile(r"(?i)^\s*you\s*$"),
    re.compile(r"(?i)^\s*\.+\s*$"),
    re.compile(r"(?i)^\s*\*+\s*$"),
]


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

        # Skip near-silent chunks (lower threshold for radio – lots of static)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.002:
            return []

        segments_iter, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            best_of=5,
            patience=1.5,
            # Do NOT carry previous context forward – prevents hallucination
            # cascading where one bad chunk poisons all subsequent ones
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                # Radio audio has background static; be more aggressive at
                # finding speech in noise but require a decent speech window
                threshold=0.35,
                min_speech_duration_ms=250,
                min_silence_duration_ms=200,
                speech_pad_ms=250,
                max_speech_duration_s=30,
            ),
            without_timestamps=False,
            initial_prompt=ATC_PROMPT,
            # Higher temperature schedule: try precise first, then broaden
            temperature=[0.0, 0.2, 0.4, 0.6],
            # Suppress common Whisper failure tokens
            suppress_blank=True,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
        )

        results = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            # Filter hallucinated text
            if self._is_hallucination(text):
                logger.debug("Filtered hallucination: %r", text)
                continue

            # Skip very short garbage segments
            if len(text) < 3:
                continue

            confidence = round(1.0 - seg.no_speech_prob, 3) if hasattr(seg, 'no_speech_prob') else 0.0

            # Skip low-confidence segments
            if confidence < 0.25:
                logger.debug("Skipping low-confidence (%.2f): %r", confidence, text)
                continue

            results.append({
                "text": text,
                "start": seg.start,
                "end": seg.end,
                "confidence": confidence,
            })

        return results

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Check if text matches known Whisper hallucination patterns."""
        for pattern in HALLUCINATION_PATTERNS:
            if pattern.search(text):
                return True
        # Repeated short phrases are often hallucinations
        words = text.split()
        if len(words) >= 4:
            # Check if it's just the same 1-2 words repeated
            unique = set(w.lower() for w in words)
            if len(unique) <= 2:
                return True
        return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
