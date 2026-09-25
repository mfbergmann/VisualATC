# VisualATC – ATC Stream Transcriber

A **local-first** Air Traffic Control audio transcriber with near-real-time visualization. No cloud services required.

> **Disclaimer:** Transcripts may be inaccurate and are not for operational use. Ensure you have rights to use any audio source.

## Features

- **Local transcription** with a choice of engines — no API keys needed:
  stock Whisper, ATC fine-tuned Whisper, NVIDIA Nemotron ASR and an ATC fine-tuned Parakeet
- **Transmission-aware segmentation** — audio is cut at the squelch gaps between transmissions,
  so callsigns aren't split across chunks and results arrive about a second after each transmission ends
- **Radio voice-band filter** (300–3400 Hz) matching AM airband audio
- **Callsign detection** with airline telephony mapping, NATO phonetics, US (`N123AB`) and Canadian (`C-GABC`) registrations
- **Number normalisation** — "delta one four nine two", "Delta fourteen ninety-two" and "Delta 1-4-9-2" all become `Delta 1492`
- **Flight cards** showing accumulated mentions, extracted fields (runway, altitude, heading, speed, frequency, squawk)
- **Events timeline** tracking GO-AROUND, DIVERT, HOLD, EMERGENCY, WINDSHEAR, and more
- **Mention frequency sparklines** per callsign
- **Cross-linking** to FlightAware (or custom tracker URL)
- **Session export** to `.txt` transcript and `.json` structured data
- **Feed bookmarks** — save and label your own stream sources
- **Audio playback** in the browser when possible

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH
- 0.1–3 GB disk per model (downloaded from Hugging Face on first use)

## Quick Start

```bash
git clone https://github.com/mfbergmann/VisualATC.git
cd VisualATC
bash setup.sh
source venv/bin/activate
python run.py
```

Then open **http://localhost:8765** in your browser.

## Usage

1. Paste an audio stream URL you have permission to use (Icecast, Shoutcast, HLS, or direct HTTP audio).
2. Or upload a local audio file (.mp3, .wav, .ogg, .flac).
3. Choose a speech model (see below). Models that need an optional back end are listed but disabled until it is installed.
4. Confirm you have permission to use the audio source.
5. Click **Start Transcribing**.

### Live View

- **Transcript panel** — scrolling feed with timestamps
- **Flight cards** — grouped by canonical callsign, showing aliases, extracted fields, event badges
- **Events timeline** — filterable by callsign, newest first
- **Controls** — pause/resume transcription, stop session, export

### Feed Bookmarks

Save frequently used stream URLs with a custom label for quick access.

## Audio Sources

This app does **not** include or link to any third-party stream directories. You must provide your own stream URL that you have rights to use.

Supported formats:
- HTTP audio streams (Icecast/Shoutcast MP3/AAC)
- HLS streams (.m3u8)
- Direct audio file URLs
- Local file upload (.mp3, .wav, .ogg, .flac, .m4a)

## Speech Models

| ID | Model | Back end | Notes |
|----|-------|----------|-------|
| `tiny`, `base`, `small` | Whisper `*.en` | faster-whisper | Stock Whisper. `small` is the default. |
| `large-v3-turbo` | Whisper large-v3-turbo | faster-whisper | Best stock Whisper; GPU recommended for live use. |
| `atc-medium` | [Whisper medium.en ATC fine-tune](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper) | faster-whisper | Trained on ATCO2 + UWB-ATCC. Reported 15% WER on its ATC test set vs 95% for stock medium.en. |
| `atc-turbo-us` | [Whisper large-v3-turbo + US ATC LoRA](https://huggingface.co/thomaseibner/whisper-large-v3-turbo-us-atc-v2) | transformers | Trained on Minneapolis-area US ATC. Reported WER 56% → 32%, identifier recall 53% → 83%. |
| `nemotron-en` | [NVIDIA Nemotron ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | transformers | NVIDIA's 2026 streaming English model. Strong general accuracy, fast on CPU, **not** ATC-trained. |
| `parakeet-atc` | [Parakeet TDT 0.6B v3 ATC fine-tune](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC) | NeMo | Reported 6% WER on the ATC-ASR test split (European audio, same source it was trained on). |

The faster-whisper models work out of the box. The others need an optional install:

```bash
bash setup.sh --transformers   # nemotron-en, atc-turbo-us (torch + transformers + peft)
bash setup.sh --nemo           # parakeet-atc (NVIDIA NeMo; Linux + NVIDIA GPU recommended)
```

**Which to use?** Domain fine-tuning matters more than model generation for ATC: general-purpose
models, Nemotron included, stumble on callsigns, phonetics and clipped phraseology. Start with
`atc-medium` (no extra install), try `atc-turbo-us` if you have a GPU or Apple Silicon, and use
the benchmark below on clips from your own feed to decide. All published figures above come from
the model authors and were measured on different test sets, so they are not directly comparable.

### Benchmarking on your own audio

```bash
pip install -r requirements-dev.txt
# folder of clip.wav + clip.txt pairs, one transmission per clip
python tools/benchmark.py --dir my_clips/ --models small atc-medium nemotron-en
# or a public ATC test set
python tools/benchmark.py --hf jacktol/ATC-ASR-Dataset --limit 200 --models small atc-medium
```

It reports word error rate (after number normalisation), callsign recall and real-time factor.

### Environment variables

| Variable | Effect |
|----------|--------|
| `VISUALATC_DEVICE` | Force `cpu`, `cuda` or `mps` instead of auto-detecting. |
| `VISUALATC_COMPUTE_TYPE` | faster-whisper precision, e.g. `float32` or `int8_float32`. The default on CPU is `int8`, which is fastest; the author of the US-ATC LoRA saw occasional digit errors with int8, so try `float32` if accuracy matters more than speed. |

## How it works

```
ffmpeg (decode, 16 kHz mono, voice-band filter)
  → transmission segmenter (adaptive noise floor, cuts at squelch gaps, 25 s max)
  → queue (live streams drop the oldest transmissions if the model falls behind)
  → ASR engine (one transmission per call; hallucination and repetition-loop filter)
  → number normalisation → callsign / field / event extraction → WebSocket → browser
```

Whisper is run without an initial prompt: a static ATC phraseology prompt made stock Whisper
16.7 WER points worse in the evaluation published with the US-ATC LoRA.

## Project Structure

```
VisualATC/
├── run.py                  # Entry point
├── requirements.txt
├── setup.sh                # One-step setup
├── server/
│   ├── app.py              # FastAPI application
│   ├── audio_ingest.py     # Audio stream/file handling (ffmpeg)
│   ├── transcriber.py      # ASR engines + model registry
│   ├── segmenter.py        # Splits audio into transmissions
│   ├── text_normalizer.py  # Spoken numbers → digits
│   ├── nlp_extractor.py    # Callsign & entity extraction
│   ├── session_manager.py  # Session data & persistence
│   ├── models.py           # Pydantic data models
│   └── data/
│       └── airline_telephony.json
├── tools/benchmark.py      # Compare models on labelled ATC clips
├── tests/                  # pytest suite
├── static/                 # Frontend (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── sessions/               # Auto-created per session
```

## Configuration

Edit `server/data/airline_telephony.json` to add or modify airline telephony-to-ICAO mappings.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

## License

MIT
