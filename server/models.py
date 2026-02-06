"""Pydantic data models for VisualATC."""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InputMode(str, Enum):
    STREAM = "stream"
    FILE = "file"


class WhisperModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"


class EventType(str, Enum):
    GO_AROUND = "GO_AROUND"
    DIVERT = "DIVERT"
    HOLD = "HOLD"
    HOLD_SHORT = "HOLD_SHORT"
    RUNWAY_CHANGE = "RUNWAY_CHANGE"
    WINDSHEAR = "WINDSHEAR"
    MINIMUM_FUEL = "MINIMUM_FUEL"
    EMERGENCY = "EMERGENCY"
    MAYDAY = "MAYDAY"
    PAN_PAN = "PAN_PAN"


class StartRequest(BaseModel):
    url: str = ""
    mode: InputMode = InputMode.STREAM
    model_size: WhisperModel = WhisperModel.SMALL
    chunk_duration: float = Field(default=7.0, ge=3.0, le=15.0)


class TranscriptSegment(BaseModel):
    ts: float
    text: str
    confidence: float = 0.0
    raw: str = ""
    duration: float = 0.0


class ExtractedFields(BaseModel):
    runway: Optional[str] = None
    altitude: Optional[str] = None
    heading: Optional[str] = None
    speed: Optional[str] = None
    frequency: Optional[str] = None


class Mention(BaseModel):
    ts: float
    callsign_canonical: str
    aliases: list[str] = Field(default_factory=list)
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    raw_text: str = ""


class ATCEvent(BaseModel):
    ts: float
    type: EventType
    callsign_canonical: Optional[str] = None
    details: str = ""
    confidence: float = 0.0


class FlightCard(BaseModel):
    callsign: str
    aliases: list[str] = Field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    mention_count: int = 0
    mentions: list[Mention] = Field(default_factory=list)
    events: list[ATCEvent] = Field(default_factory=list)
    latest_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    sparkline: list[int] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str = ""
    started_at: float = Field(default_factory=time.time)
    is_running: bool = False
    is_paused: bool = False
    model_size: WhisperModel = WhisperModel.SMALL
    source_url: str = ""
    source_mode: InputMode = InputMode.STREAM
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    flight_cards: dict[str, FlightCard] = Field(default_factory=dict)
    events: list[ATCEvent] = Field(default_factory=list)
    total_audio_seconds: float = 0.0


class BookmarkEntry(BaseModel):
    label: str
    url: str
    added_at: float = Field(default_factory=time.time)
