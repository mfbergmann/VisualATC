"""Session management: in-memory state + disk persistence."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    ATCEvent,
    BookmarkEntry,
    ExtractedFields,
    FlightCard,
    InputMode,
    Mention,
    SessionState,
    TranscriptSegment,
    WhisperModel,
)

logger = logging.getLogger("visualatc.session")

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
BOOKMARKS_FILE = Path(__file__).parent.parent / "bookmarks.json"

# Number of sparkline buckets (each ~30s window)
SPARKLINE_BUCKETS = 60
SPARKLINE_WINDOW = 30.0  # seconds per bucket


class SessionManager:
    """Manages the current session state and persists data to disk."""

    def __init__(self):
        self.state: Optional[SessionState] = None
        self._session_dir: Optional[Path] = None
        # Sparkline tracking: callsign -> list of timestamps
        self._mention_timestamps: dict[str, list[float]] = {}

    def create_session(
        self,
        source_url: str,
        source_mode: InputMode,
        model_size: WhisperModel,
    ) -> SessionState:
        """Create a new session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = SESSIONS_DIR / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self.state = SessionState(
            session_id=session_id,
            started_at=time.time(),
            is_running=True,
            model_size=model_size,
            source_url=source_url,
            source_mode=source_mode,
        )
        self._mention_timestamps = {}

        logger.info("Created session %s at %s", session_id, self._session_dir)
        return self.state

    def add_transcript_segment(self, segment: TranscriptSegment) -> None:
        """Add a transcript segment and persist to disk."""
        if not self.state:
            return
        self.state.transcript.append(segment)
        self._append_jsonl("transcript.jsonl", segment.model_dump())

    def add_mention(self, mention: Mention) -> None:
        """Add a mention and update the corresponding flight card."""
        if not self.state:
            return

        cs = mention.callsign_canonical

        # Update or create flight card
        if cs not in self.state.flight_cards:
            self.state.flight_cards[cs] = FlightCard(
                callsign=cs,
                first_seen=mention.ts,
                last_seen=mention.ts,
            )
            self._mention_timestamps[cs] = []

        card = self.state.flight_cards[cs]
        card.last_seen = mention.ts
        card.mention_count += 1

        # Add aliases
        for alias in mention.aliases:
            if alias not in card.aliases:
                card.aliases.append(alias)

        # Keep last 20 mentions in memory
        card.mentions.append(mention)
        if len(card.mentions) > 20:
            card.mentions = card.mentions[-20:]

        # Update latest fields (only overwrite non-None)
        if mention.extracted_fields.runway:
            card.latest_fields.runway = mention.extracted_fields.runway
        if mention.extracted_fields.altitude:
            card.latest_fields.altitude = mention.extracted_fields.altitude
        if mention.extracted_fields.heading:
            card.latest_fields.heading = mention.extracted_fields.heading
        if mention.extracted_fields.speed:
            card.latest_fields.speed = mention.extracted_fields.speed
        if mention.extracted_fields.frequency:
            card.latest_fields.frequency = mention.extracted_fields.frequency

        # Track sparkline
        self._mention_timestamps.setdefault(cs, []).append(mention.ts)
        card.sparkline = self._compute_sparkline(cs)

        # Persist
        self._append_jsonl("mentions.jsonl", mention.model_dump())

    def add_event(self, event: ATCEvent) -> None:
        """Add an event."""
        if not self.state:
            return
        self.state.events.append(event)

        # Also attach to flight card if callsign is known
        if event.callsign_canonical and event.callsign_canonical in self.state.flight_cards:
            card = self.state.flight_cards[event.callsign_canonical]
            card.events.append(event)

        self._append_jsonl("events.jsonl", event.model_dump())

    def _compute_sparkline(self, callsign: str) -> list[int]:
        """Compute mention frequency sparkline for a callsign."""
        timestamps = self._mention_timestamps.get(callsign, [])
        if not timestamps or not self.state:
            return []

        now = time.time()
        start = self.state.started_at
        elapsed = now - start
        num_buckets = min(SPARKLINE_BUCKETS, max(1, int(elapsed / SPARKLINE_WINDOW) + 1))

        buckets = [0] * num_buckets
        for ts in timestamps:
            bucket_idx = int((ts - start) / SPARKLINE_WINDOW)
            if 0 <= bucket_idx < num_buckets:
                buckets[bucket_idx] += 1

        return buckets

    def finalize_session(self) -> dict:
        """Finalize session: write export files, return paths."""
        if not self.state or not self._session_dir:
            return {}

        self.state.is_running = False

        # Write export.txt
        txt_path = self._session_dir / "export.txt"
        with open(txt_path, "w") as f:
            f.write(f"VisualATC Session: {self.state.session_id}\n")
            f.write(f"Source: {self.state.source_url}\n")
            f.write(f"Model: {self.state.model_size}\n")
            f.write(f"Started: {datetime.fromtimestamp(self.state.started_at).isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            for seg in self.state.transcript:
                ts_str = self._format_ts(seg.ts - self.state.started_at)
                f.write(f"[{ts_str}] {seg.text}\n")

        # Write export.json
        json_path = self._session_dir / "export.json"
        export_data = {
            "session_id": self.state.session_id,
            "source_url": self.state.source_url,
            "model_size": self.state.model_size,
            "started_at": self.state.started_at,
            "total_audio_seconds": self.state.total_audio_seconds,
            "flight_cards": {
                cs: card.model_dump() for cs, card in self.state.flight_cards.items()
            },
            "events": [e.model_dump() for e in self.state.events],
            "transcript_segments": len(self.state.transcript),
        }
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info("Session finalized: %s", self._session_dir)
        return {
            "session_dir": str(self._session_dir),
            "export_txt": str(txt_path),
            "export_json": str(json_path),
        }

    def _append_jsonl(self, filename: str, data: dict) -> None:
        """Append a JSON line to a session file."""
        if not self._session_dir:
            return
        path = self._session_dir / filename
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    @staticmethod
    def _format_ts(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def get_state_snapshot(self) -> dict:
        """Get a JSON-serializable snapshot of current state for the UI."""
        if not self.state:
            return {"active": False}

        return {
            "active": True,
            "session_id": self.state.session_id,
            "is_running": self.state.is_running,
            "is_paused": self.state.is_paused,
            "model_size": self.state.model_size,
            "source_url": self.state.source_url,
            "total_audio_seconds": round(self.state.total_audio_seconds, 1),
            "transcript_count": len(self.state.transcript),
            "flight_card_count": len(self.state.flight_cards),
            "event_count": len(self.state.events),
        }


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

def load_bookmarks() -> list[BookmarkEntry]:
    """Load bookmarks from disk."""
    if not BOOKMARKS_FILE.exists():
        return []
    try:
        with open(BOOKMARKS_FILE, "r") as f:
            data = json.load(f)
        return [BookmarkEntry(**b) for b in data]
    except Exception as e:
        logger.warning("Failed to load bookmarks: %s", e)
        return []


def save_bookmarks(bookmarks: list[BookmarkEntry]) -> None:
    """Save bookmarks to disk."""
    with open(BOOKMARKS_FILE, "w") as f:
        json.dump([b.model_dump() for b in bookmarks], f, indent=2)


def add_bookmark(label: str, url: str) -> list[BookmarkEntry]:
    """Add a bookmark and return updated list."""
    bookmarks = load_bookmarks()
    # Avoid duplicates by URL
    bookmarks = [b for b in bookmarks if b.url != url]
    bookmarks.append(BookmarkEntry(label=label, url=url))
    save_bookmarks(bookmarks)
    return bookmarks


def remove_bookmark(url: str) -> list[BookmarkEntry]:
    """Remove a bookmark by URL and return updated list."""
    bookmarks = load_bookmarks()
    bookmarks = [b for b in bookmarks if b.url != url]
    save_bookmarks(bookmarks)
    return bookmarks
