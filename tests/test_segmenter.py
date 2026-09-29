import numpy as np

from server.segmenter import SAMPLE_RATE, TransmissionSegmenter

rng = np.random.default_rng(0)


def noise(seconds, level=0.003):
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * level).astype(np.float32)


def speech(seconds, level=0.2):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    # 4 Hz amplitude modulation, roughly syllable rate
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (np.sin(2 * np.pi * 700 * t) * env * level).astype(np.float32) + noise(seconds)


def feed(seg, audio, block=1600):
    out = []
    for i in range(0, len(audio), block):
        out.extend(seg.push(audio[i:i + block]))
    out.extend(seg.flush())
    return out


def test_splits_on_squelch_gaps():
    audio = np.concatenate([noise(2), speech(2.0), noise(1.5), speech(3.0), noise(2)])
    txs = feed(TransmissionSegmenter(), audio)
    assert len(txs) == 2
    assert abs(txs[0].start_s - 2.0) < 0.4
    assert abs(txs[0].duration_s - 2.0) < 0.8
    assert abs(txs[1].start_s - 5.5) < 0.4
    assert abs(txs[1].duration_s - 3.0) < 0.8


def test_ignores_static_bursts():
    audio = np.concatenate([noise(2), speech(0.1), noise(2)])
    assert feed(TransmissionSegmenter(), audio) == []


def test_digital_silence_between_transmissions():
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    audio = np.concatenate([silence, speech(1.5), silence, speech(1.5), silence])
    assert len(feed(TransmissionSegmenter(), audio)) == 2


def test_long_transmission_is_split():
    audio = np.concatenate([noise(1), speech(12), noise(1)])
    txs = feed(TransmissionSegmenter(max_segment_s=5), audio)
    assert len(txs) >= 3
    assert all(t.duration_s <= 5.01 for t in txs)


def test_flush_emits_in_progress():
    seg = TransmissionSegmenter()
    seg.push(np.concatenate([noise(1), speech(2)]))
    assert len(seg.flush()) == 1
