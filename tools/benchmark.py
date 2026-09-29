#!/usr/bin/env python3
"""Compare VisualATC speech models on labelled ATC audio.

Reports word error rate (after number normalisation), callsign recall and
real-time factor for each model, so you can pick an engine on evidence from
your own kind of audio rather than on published numbers.

Two data sources:

  # Your own clips: foo.wav + foo.txt pairs (one transmission per clip).
  # Recommended: a few dozen clips from the feed you actually listen to.
  python tools/benchmark.py --dir my_clips/ --models small atc-medium nemotron-en

  # A public ATC test set from Hugging Face (needs pyarrow + huggingface_hub).
  python tools/benchmark.py --hf jacktol/ATC-ASR-Dataset --limit 200 \
      --models small large-v3-turbo atc-medium

Caveat: jacktol/ATC-ASR-Dataset is European audio (ATCO2 + UWB-ATCC) and is
the training source for the atc-medium and parakeet-atc fine-tunes, so it
flatters those two. Your own clips are the fairer test.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.audio_ingest import RADIO_FILTER  # noqa: E402
from server.nlp_extractor import ATCExtractor  # noqa: E402
from server.text_normalizer import normalize_numbers  # noqa: E402
from server.transcriber import MODELS_BY_ID, check_available, create_engine  # noqa: E402

SAMPLE_RATE = 16000


def decode(data: bytes | str, radio_filter: bool) -> np.ndarray:
    """Decode a file path or encoded bytes to 16 kHz mono float32 with ffmpeg."""
    src = data if isinstance(data, str) else "pipe:0"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src]
    if radio_filter:
        cmd += ["-af", RADIO_FILTER]
    cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, input=None if isinstance(data, str) else data,
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_dir(path: Path, limit: int) -> list[tuple[str, bytes | str]]:
    items = []
    for audio in sorted(path.iterdir()):
        if audio.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            continue
        ref = audio.with_suffix(".txt")
        if ref.exists():
            items.append((ref.read_text().strip(), str(audio)))
    return items[:limit] if limit else items


def load_hf(repo: str, split: str, limit: int) -> list[tuple[str, bytes | str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
             if f.endswith(".parquet") and f"/{split}/" in f]
    if not files:
        sys.exit(f"No parquet files for split '{split}' in {repo}")
    items: list[tuple[str, bytes | str]] = []
    for f in sorted(files):
        table = pq.read_table(hf_hub_download(repo, f, repo_type="dataset",
                                              revision="refs/convert/parquet"))
        cols = table.column_names
        text_col = next((c for c in ("text", "transcription", "sentence", "transcript") if c in cols), None)
        if text_col is None or "audio" not in cols:
            sys.exit(f"Unexpected columns in {f}: {cols}")
        for row in table.select([text_col, "audio"]).to_pylist():
            items.append((row[text_col], row["audio"]["bytes"]))
            if limit and len(items) >= limit:
                return items
    return items


def words(text: str) -> list[str]:
    text = normalize_numbers(text).lower()
    text = re.sub(r"[^\w\s.]|(?<!\d)\.|\.(?!\d)", " ", text)
    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="folder of audio + .txt reference pairs")
    src.add_argument("--hf", help="Hugging Face dataset repo with audio/text columns")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=["small"], choices=sorted(MODELS_BY_ID))
    ap.add_argument("--no-radio-filter", action="store_true")
    ap.add_argument("--show", type=int, default=3, help="print N example transcripts per model")
    args = ap.parse_args()

    items = load_dir(args.dir, args.limit) if args.dir else load_hf(args.hf, args.split, args.limit)
    if not items:
        sys.exit("No labelled clips found.")
    print(f"Decoding {len(items)} clips...")
    clips = [(ref, decode(data, not args.no_radio_filter)) for ref, data in items]
    audio_s = sum(len(a) for _, a in clips) / SAMPLE_RATE

    extractor = ATCExtractor()
    ref_calls = [{c["canonical"] for c in extractor.extract_all(ref, 0)["callsigns"]} for ref, _ in clips]

    rows = []
    for model_id in args.models:
        ok, reason = check_available(MODELS_BY_ID[model_id])
        if not ok:
            print(f"\n[{model_id}] skipped: {reason}")
            continue
        engine = create_engine(model_id)
        engine.load_model()
        errors = ref_words = cs_hit = cs_total = 0
        t0 = time.time()
        for i, (ref, audio) in enumerate(clips):
            hyp = " ".join(s["text"] for s in engine.transcribe_chunk(audio))
            r, h = words(ref), words(hyp)
            errors += edit_distance(r, h)
            ref_words += len(r)
            hyp_calls = {c["canonical"] for c in extractor.extract_all(hyp, 0)["callsigns"]}
            cs_hit += len(ref_calls[i] & hyp_calls)
            cs_total += len(ref_calls[i])
            if i < args.show:
                print(f"\n[{model_id}] REF: {ref}\n[{model_id}] HYP: {hyp}")
        elapsed = time.time() - t0
        rows.append((model_id, engine.device, errors / max(ref_words, 1),
                     cs_hit / cs_total if cs_total else float("nan"), elapsed / audio_s))

    print(f"\n{len(clips)} clips, {audio_s / 60:.1f} min of audio\n")
    print(f"{'model':<16} {'device':<18} {'WER':>7} {'callsign recall':>16} {'RTF':>6}")
    for model_id, device, wer, recall, rtf in rows:
        print(f"{model_id:<16} {device:<18} {wer:>6.1%} {recall:>16.1%} {rtf:>6.2f}")
    print("\nRTF < 1 means faster than real time. Callsign recall counts reference "
          "callsigns (as parsed by VisualATC) that the model reproduced exactly.")


if __name__ == "__main__":
    main()
