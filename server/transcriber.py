"""Speech-recognition engines.

VisualATC can run several ASR back ends behind one interface:

* ``faster-whisper`` (CTranslate2) – installed by default; stock Whisper
  models plus a Whisper fine-tune trained on ATC audio.
* ``transformers`` – optional (torch + transformers + peft); NVIDIA Nemotron
  ASR Streaming and a US-ATC LoRA for Whisper large-v3-turbo.
* ``nemo`` – optional (NVIDIA NeMo); Parakeet TDT fine-tuned on ATC audio.

Every engine takes one radio transmission (float32 PCM, 16 kHz mono) and
returns a list of ``{text, start, end, confidence}`` dicts.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("visualatc.transcriber")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    backend: str               # "faster-whisper" | "transformers" | "nemo"
    repo: str                  # model name / Hugging Face repo id
    description: str
    size: str = ""
    atc_tuned: bool = False
    adapter: Optional[str] = None       # PEFT adapter repo (transformers only)
    cpu_compute_type: str = "int8"      # faster-whisper only


MODELS: list[ModelSpec] = [
    ModelSpec(
        id="tiny", label="Whisper tiny.en", backend="faster-whisper", repo="tiny.en",
        size="75 MB", description="Fastest and least accurate. Useful on slow machines.",
    ),
    ModelSpec(
        id="base", label="Whisper base.en", backend="faster-whisper", repo="base.en",
        size="145 MB", description="Fast; moderate accuracy.",
    ),
    ModelSpec(
        id="small", label="Whisper small.en", backend="faster-whisper", repo="small.en",
        size="485 MB", description="Stock Whisper baseline. Reasonable on CPU.",
    ),
    ModelSpec(
        id="large-v3-turbo", label="Whisper large-v3-turbo", backend="faster-whisper",
        repo="large-v3-turbo", size="1.6 GB",
        description="Strongest stock Whisper. A GPU is recommended for live streams.",
    ),
    ModelSpec(
        id="atc-medium", label="Whisper medium.en – ATC fine-tune (jacktol)",
        backend="faster-whisper",
        repo="jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper",
        size="3 GB", atc_tuned=True,
        description=(
            "Fine-tuned on ATCO2 and UWB-ATCC recordings. Reported 15% WER on its ATC "
            "test set, against 95% for stock medium.en. Outputs spoken-form numbers."
        ),
    ),
    ModelSpec(
        id="atc-turbo-us", label="Whisper large-v3-turbo – US ATC LoRA (thomaseibner)",
        backend="transformers", repo="openai/whisper-large-v3-turbo",
        adapter="thomaseibner/whisper-large-v3-turbo-us-atc-v2",
        size="1.6 GB + 13 MB", atc_tuned=True,
        description=(
            "LoRA trained on Minneapolis-area US ATC audio. Reported WER 56% -> 32% and "
            "callsign/identifier recall 53% -> 83% against stock turbo. GPU or Apple "
            "Silicon recommended."
        ),
    ),
    ModelSpec(
        id="nemotron-en", label="NVIDIA Nemotron ASR Streaming 0.6B (English)",
        backend="transformers", repo="nvidia/nemotron-speech-streaming-en-0.6b",
        size="2.5 GB",
        description=(
            "NVIDIA's 2026 cache-aware FastConformer-RNNT model. Strong general English "
            "accuracy and fast on CPU, but not trained on ATC phraseology."
        ),
    ),
    ModelSpec(
        id="parakeet-atc", label="NVIDIA Parakeet TDT 0.6B v3 – ATC fine-tune (qenneth)",
        backend="nemo", repo="qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC",
        size="2.5 GB", atc_tuned=True,
        description=(
            "Parakeet fine-tuned on the ATC-ASR dataset (European ATCO2/UWB-ATCC audio). "
            "Reported 6% WER on that dataset's test split."
        ),
    ),
]

MODELS_BY_ID = {m.id: m for m in MODELS}

INSTALL_HINTS = {
    "faster-whisper": "pip install faster-whisper",
    "transformers": "pip install -r requirements-transformers.txt",
    "nemo": "pip install -r requirements-nemo.txt",
}


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        m = re.match(r"\d+", part)
        out.append(int(m.group()) if m else 0)
    return tuple(out)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_available(spec: ModelSpec) -> tuple[bool, str]:
    """Return (available, reason) without importing heavy libraries."""
    if spec.backend == "faster-whisper":
        if not _has("faster_whisper"):
            return False, INSTALL_HINTS["faster-whisper"]
        return True, ""

    if spec.backend == "transformers":
        missing = [m for m in ("torch", "transformers") if not _has(m)]
        if spec.adapter and not _has("peft"):
            missing.append("peft")
        if missing:
            return False, f"Missing {', '.join(missing)}: {INSTALL_HINTS['transformers']}"
        if spec.id == "nemotron-en":
            ver = importlib.metadata.version("transformers")
            if _version_tuple(ver) < (5, 13):
                return False, f"Needs transformers >= 5.13 (found {ver})"
        return True, ""

    if spec.backend == "nemo":
        if not _has("nemo"):
            return False, INSTALL_HINTS["nemo"]
        return True, ""

    return False, f"Unknown backend {spec.backend}"


def list_models() -> list[dict]:
    out = []
    for spec in MODELS:
        ok, reason = check_available(spec)
        out.append({**asdict(spec), "available": ok, "unavailable_reason": reason})
    return out


# ---------------------------------------------------------------------------
# Output filtering shared by all engines
# ---------------------------------------------------------------------------

# Phrases Whisper-family models hallucinate on silence or static.
HALLUCINATION_PATTERNS = [
    re.compile(r"(?i)air traffic control communication"),
    re.compile(r"(?i)callsigns?,?\s*runways?,?\s*altitudes?,?\s*headings?"),
    re.compile(r"(?i)thank(?:s| you) for (?:watching|listening)"),
    re.compile(r"(?i)please (?:like and )?subscribe"),
    re.compile(r"(?i)subtitles? by"),
    re.compile(r"(?i)^\s*(?:you|bye|thank you)[.!]?\s*$"),
    re.compile(r"^\W*$"),
]

MIN_CONFIDENCE = 0.15


def is_hallucination(text: str) -> bool:
    """Known hallucination phrases and repetition loops.

    The loop rule (8+ words, and either a run of 5 identical words or under
    40% unique words) follows the one published with the
    thomaseibner/whisper-large-v3-turbo-us-atc-v2 model.
    """
    if any(p.search(text) for p in HALLUCINATION_PATTERNS):
        return True
    words = [w.lower().strip(".,!?") for w in text.split()]
    if len(words) >= 8:
        if len(set(words)) / len(words) < 0.4:
            return True
        run = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a == b else 1
            if run >= 5:
                return True
    elif len(words) >= 4 and len(set(words)) <= 1:
        return True
    return False


def prepare_audio(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Float32, finite, and RMS-normalised with a peak ceiling."""
    audio = np.nan_to_num(audio.astype(np.float32, copy=False))
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    gain = 10 ** (target_dbfs / 20) / rms
    peak = float(np.abs(audio).max())
    gain = min(gain, 0.99 / peak) if peak > 0 else gain
    return (audio * gain).astype(np.float32)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class ASREngine:
    """Base class. Subclasses implement ``_load`` and ``_transcribe``."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.device = "cpu"
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self.spec.id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        ok, reason = check_available(self.spec)
        if not ok:
            raise RuntimeError(f"{self.spec.label} is not available: {reason}")
        logger.info("Loading %s (%s backend)...", self.spec.label, self.spec.backend)
        t0 = time.time()
        self._load()
        self._loaded = True
        logger.info("Loaded %s on %s in %.1fs", self.spec.label, self.device, time.time() - t0)

    def transcribe_chunk(self, audio: np.ndarray) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        audio = prepare_audio(audio)
        if audio.size < SAMPLE_RATE * 0.2:
            return []
        results = []
        for seg in self._transcribe(audio):
            text = " ".join(seg["text"].split())
            if len(text) < 2 or is_hallucination(text):
                logger.debug("Filtered: %r", text)
                continue
            conf = seg.get("confidence")
            if conf is not None and conf < MIN_CONFIDENCE:
                logger.debug("Low confidence (%.2f): %r", conf, text)
                continue
            results.append({**seg, "text": text})
        return results

    def _load(self) -> None:
        raise NotImplementedError

    def _transcribe(self, audio: np.ndarray) -> list[dict]:
        raise NotImplementedError


class FasterWhisperEngine(ASREngine):

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        device = os.environ.get("VISUALATC_DEVICE", "auto")
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute_type = os.environ.get("VISUALATC_COMPUTE_TYPE") or (
            "float16" if device == "cuda" else self.spec.cpu_compute_type
        )
        self.device = f"{device} ({compute_type})"
        self._model = WhisperModel(self.spec.repo, device=device, compute_type=compute_type)

    def _transcribe(self, audio: np.ndarray) -> list[dict]:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            # Each call is one transmission; carrying text across calls lets
            # one bad decode poison the next.
            condition_on_previous_text=False,
            # No initial_prompt: a static ATC phraseology prompt measurably
            # hurts accuracy (+16.7 WER points on stock Whisper in the
            # thomaseibner ATC evaluation) and invites parroting.
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_speech_duration_ms=200,
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
            temperature=[0.0, 0.2, 0.4, 0.6],
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        out = []
        for seg in segments:
            conf = math.exp(seg.avg_logprob) * (1.0 - 0.5 * seg.no_speech_prob)
            out.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": round(conf, 3),
            })
        return out


class TransformersEngine(ASREngine):

    def _load(self) -> None:
        import torch
        from transformers import pipeline

        requested = os.environ.get("VISUALATC_DEVICE", "auto")
        if requested != "auto":
            device = requested
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.device = device

        if self.spec.adapter:
            from peft import PeftModel
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = WhisperProcessor.from_pretrained(self.spec.repo)
            # Merge in float32, then cast: merging at half precision rounds
            # the adapted weights twice.
            model = WhisperForConditionalGeneration.from_pretrained(
                self.spec.repo, dtype=torch.float32
            )
            model = PeftModel.from_pretrained(model, self.spec.adapter).merge_and_unload()
            model = model.to(device=device, dtype=dtype).eval()
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
            self._generate_kwargs = {
                "language": "en",
                "task": "transcribe",
                "max_new_tokens": 128,
                "repetition_penalty": 1.1,
            }
        else:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.spec.repo,
                device=device,
                dtype=dtype,
            )
            self._generate_kwargs = {}

    def _transcribe(self, audio: np.ndarray) -> list[dict]:
        kwargs = {"generate_kwargs": self._generate_kwargs} if self._generate_kwargs else {}
        result = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, **kwargs)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return [{
            "text": text,
            "start": 0.0,
            "end": len(audio) / SAMPLE_RATE,
            "confidence": None,
        }]


class NemoEngine(ASREngine):

    def _load(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        device = os.environ.get("VISUALATC_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.spec.repo.startswith("nvidia/"):
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.spec.repo)
        else:
            from huggingface_hub import hf_hub_download, list_repo_files

            nemo_files = [f for f in list_repo_files(self.spec.repo) if f.endswith(".nemo")]
            if not nemo_files:
                raise RuntimeError(f"No .nemo checkpoint in {self.spec.repo}")
            path = hf_hub_download(self.spec.repo, nemo_files[0])
            model = nemo_asr.models.ASRModel.restore_from(path, map_location=device)
        self._model = model.to(device).eval()

    def _transcribe(self, audio: np.ndarray) -> list[dict]:
        out = self._model.transcribe([audio], batch_size=1, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        if not out:
            return []
        hyp = out[0]
        text = getattr(hyp, "text", hyp)
        return [{
            "text": str(text),
            "start": 0.0,
            "end": len(audio) / SAMPLE_RATE,
            "confidence": None,
        }]


_ENGINE_CLASSES = {
    "faster-whisper": FasterWhisperEngine,
    "transformers": TransformersEngine,
    "nemo": NemoEngine,
}


def create_engine(model_id: str) -> ASREngine:
    spec = MODELS_BY_ID.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {', '.join(MODELS_BY_ID)}")
    return _ENGINE_CLASSES[spec.backend](spec)
