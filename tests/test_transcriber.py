import numpy as np
import pytest

from server.transcriber import (
    MODELS, ASREngine, ModelSpec, create_engine, is_hallucination, list_models, prepare_audio,
)


@pytest.mark.parametrize("text", [
    "Thank you for watching!",
    "you",
    "point point point point point point point point",
    "roger roger roger roger roger wilco one two",
    "...",
])
def test_hallucinations(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", [
    "Delta 1492 cleared to land runway 28 left",
    "roger",
    "two two two zero",
])
def test_real_speech_kept(text):
    assert not is_hallucination(text)


def test_registry_ids_unique_and_default_present():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))
    assert "small" in ids


def test_list_models_reports_availability():
    for m in list_models():
        assert isinstance(m["available"], bool)
        if not m["available"]:
            assert m["unavailable_reason"]


def test_unknown_model():
    with pytest.raises(ValueError):
        create_engine("nope")


def test_prepare_audio_normalises_and_limits():
    quiet = (np.sin(np.linspace(0, 1000, 16000)) * 0.001).astype(np.float32)
    out = prepare_audio(quiet)
    assert 0.05 < np.sqrt(np.mean(out ** 2)) < 0.15
    assert np.abs(out).max() <= 0.99


class FakeEngine(ASREngine):
    def __init__(self, outputs):
        super().__init__(ModelSpec("fake", "Fake", "faster-whisper", "fake", ""))
        self._outputs = outputs
        self._loaded = True

    def _transcribe(self, audio):
        return self._outputs


def test_engine_filters_output():
    eng = FakeEngine([
        {"text": " Delta  1492 roger ", "start": 0, "end": 1, "confidence": 0.9},
        {"text": "Thank you for watching", "start": 1, "end": 2, "confidence": 0.9},
        {"text": "mumble", "start": 2, "end": 3, "confidence": 0.05},
        {"text": "United 5 go around", "start": 3, "end": 4, "confidence": None},
    ])
    out = eng.transcribe_chunk(np.ones(16000, dtype=np.float32) * 0.1)
    assert [o["text"] for o in out] == ["Delta 1492 roger", "United 5 go around"]
