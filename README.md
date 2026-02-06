# VisualATC – ATC Stream Transcriber

A **local-first** Air Traffic Control audio transcriber with near-real-time visualization. No cloud services required.

> **Disclaimer:** Transcripts may be inaccurate and are not for operational use. Ensure you have rights to use any audio source.

## Features

- **Local transcription** using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — no API keys needed
- **Near-real-time** streaming transcription in 5–10 second chunks
- **Callsign detection** with airline telephony mapping and NATO phonetic normalization
- **Flight cards** showing accumulated mentions, extracted fields (runway, altitude, heading, speed, frequency)
- **Events timeline** tracking GO-AROUND, DIVERT, HOLD, EMERGENCY, WINDSHEAR, and more
- **Mention frequency sparklines** per callsign
- **Cross-linking** to FlightAware (or custom tracker URL)
- **Session export** to `.txt` transcript and `.json` structured data
- **Feed bookmarks** — save and label your own stream sources
- **Audio playback** in the browser when possible

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH
- ~1–2 GB disk for Whisper model files (downloaded on first run)

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
3. Select a Whisper model size (tiny / base / small — larger = more accurate but slower).
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

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | ~75 MB | Fastest | Lower |
| base  | ~150 MB | Fast | Medium |
| small | ~500 MB | Medium | Good |

Models are downloaded automatically on first use via faster-whisper.

## Project Structure

```
VisualATC/
├── run.py                  # Entry point
├── requirements.txt
├── setup.sh                # One-step setup
├── server/
│   ├── app.py              # FastAPI application
│   ├── audio_ingest.py     # Audio stream/file handling (ffmpeg)
│   ├── transcriber.py      # faster-whisper wrapper
│   ├── nlp_extractor.py    # Callsign & entity extraction
│   ├── session_manager.py  # Session data & persistence
│   ├── models.py           # Pydantic data models
│   └── data/
│       └── airline_telephony.json
├── static/                 # Frontend (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── sessions/               # Auto-created per session
```

## Configuration

Edit `server/data/airline_telephony.json` to add or modify airline telephony-to-ICAO mappings.

## License

MIT
