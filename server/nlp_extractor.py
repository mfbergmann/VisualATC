"""NLP extraction for ATC transcripts: callsigns, entities, events.

All extraction runs on text that has been through
:func:`text_normalizer.normalize_numbers`, so numbers are always digit
strings regardless of which ASR engine produced the transcript.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import ATCEvent, EventType, ExtractedFields
from .text_normalizer import normalize_numbers

logger = logging.getLogger("visualatc.nlp")

# NATO phonetic alphabet
NATO_PHONETIC = {
    "alfa": "A", "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliet": "J", "juliett": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W", "whisky": "W",
    "x-ray": "X", "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}
_NATO_RE = "|".join(sorted((re.escape(k) for k in NATO_PHONETIC), key=len, reverse=True))

RUNWAY_SUFFIX = {"left": "L", "right": "R", "center": "C", "centre": "C", "l": "L", "r": "R", "c": "C"}

# Built-in telephony aliases, including common ASR mishearings. Entries in
# data/airline_telephony.json take precedence.
AIRLINE_ALIASES = {
    "national air": "ACA",     # "Air Canada" misheard
    "air canada": "ACA",
    "canadi'n": "ACA",
    "jazz": "JZA",
    "jazz air": "JZA",
    "delta": "DAL",
    "american": "AAL",
    "united": "UAL",
    "southwest": "SWA",
    "south west": "SWA",
    "jetblue": "JBU",
    "jet blue": "JBU",
    "alaska": "ASA",
    "spirit": "NKS",
    "frontier": "FFT",
    "westjet": "WJA",
    "west jet": "WJA",
    "endeavor": "EDV",
    "envoy": "ENY",
    "republic": "RPA",
    "brickyard": "RPA",
    "skywest": "SKW",
    "sky west": "SKW",
    "mesa": "ASH",
    "cactus": "AWE",
    "speedbird": "BAW",
    "air france": "AFR",
    "lufthansa": "DLH",
    "emirates": "UAE",
    "fedex": "FDX",
    "ups": "UPS",
    "giant": "GTI",
    "atlas": "GTI",
    "atlas air": "GTI",
    "hawaiian": "HAL",
    "allegiant": "AAY",
    "sun country": "SCX",
    "porter": "POE",
    "sunwing": "SWG",
    "flair": "FLE",
    "air transat": "TSC",
}

# Prefixes that look like ICAO callsigns but are not.
_NOT_CALLSIGN_PREFIXES = ("FL", "RWY", "ILS", "RNAV", "VOR", "DME", "QNH", "ATIS")

# ---------------------------------------------------------------------------
# Event keyword patterns
# ---------------------------------------------------------------------------

EVENT_PATTERNS: list[tuple[str, EventType]] = [
    (r"\bgo(?:ing)?[\s\-]?around\b", EventType.GO_AROUND),
    (r"\bmissed\s+approach\b(?!\s+(?:procedure|instructions))", EventType.GO_AROUND),
    (r"\bdivert(?:ing|ed|s)?\b|\bdiversion\b|\bproceed(?:ing)?\s+to\s+(?:the|our|my)\s+alternate\b",
     EventType.DIVERT),
    (r"\bhold(?:ing)?\s+short\b", EventType.HOLD_SHORT),
    (r"\bholding\s+pattern\b|\benter(?:ing)?\s+(?:the\s+)?hold\b"
     r"|\bhold\s+(?:as\s+published|over|at|(?:north|south|east|west)(?:east|west)?\s+of)\b"
     r"|\bexpect\s+further\s+clearance\b",
     EventType.HOLD),
    (r"\brunway\s+change\b|\bchange\s+(?:of\s+)?runway\b", EventType.RUNWAY_CHANGE),
    (r"\bwind\s*shear\b|\bmicroburst\b", EventType.WINDSHEAR),
    (r"\bminimum\s+fuel\b", EventType.MINIMUM_FUEL),
    (r"\bmayday\b", EventType.MAYDAY),
    (r"\bpan[\s\-]?pan\b", EventType.PAN_PAN),
    (r"\bdeclar(?:e|es|ed|ing)\s+(?:an\s+)?emergency\b|\bemergency\b|\bsquawk(?:ing)?\s+7700\b",
     EventType.EMERGENCY),
]


def _nato_to_letters(words: str) -> str:
    return "".join(NATO_PHONETIC.get(w.lower(), "") for w in words.split())


class ATCExtractor:
    """Extract ATC entities from transcribed text."""

    def __init__(self, telephony_path: Optional[str] = None):
        if telephony_path is None:
            telephony_path = str(Path(__file__).parent / "data" / "airline_telephony.json")

        self.telephony_map = self._load_telephony(telephony_path)
        for alias, icao in AIRLINE_ALIASES.items():
            self.telephony_map.setdefault(alias, icao)

        names = sorted(self.telephony_map.keys(), key=len, reverse=True)
        name_re = "|".join(re.escape(n) for n in names) or r"(?!x)x"

        # "Air Canada 8853", "Delta 1492 heavy", "Speedbird 27A"
        self._telephony_re = re.compile(
            r"\b(" + name_re + r")\s+(\d{1,4}(?-i:[A-Z])?)\b(?:\s+(heavy|super))?",
            re.IGNORECASE,
        )
        # US GA: "November 123 alpha bravo" / "N123AB"
        self._ga_us_re = re.compile(
            r"\bnovember\s+(\d{1,5})((?:\s+(?:" + _NATO_RE + r")){0,2})\b",
            re.IGNORECASE,
        )
        # Canadian GA: "Charlie Golf Alpha Bravo Charlie" / "Golf Alpha Bravo Charlie"
        self._ga_ca_re = re.compile(
            r"\b(?:charlie\s+)?(golf|foxtrot)((?:\s+(?:" + _NATO_RE + r")){3})\b",
            re.IGNORECASE,
        )
        # Callsigns already written as ICAO codes: "ACA8853", "N123AB", "DAL123"
        self._alpha_callsign_re = re.compile(r"\b([A-Z]{2,4}\d{1,5}[A-Z]{0,2})\b")

        self._event_compiled = [
            (re.compile(pat, re.IGNORECASE), evt_type)
            for pat, evt_type in EVENT_PATTERNS
        ]

    @staticmethod
    def _load_telephony(path: str) -> dict[str, str]:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return {k.lower(): v for k, v in data.items() if not k.startswith("_")}
        except Exception as e:
            logger.warning("Failed to load telephony map from %s: %s", path, e)
            return {}

    # ------------------------------------------------------------------
    # Callsigns
    # ------------------------------------------------------------------

    def extract_callsigns(self, text: str) -> list[dict]:
        """Extract callsigns from (normalised) text.

        Returns a list of ``{canonical, alias, span}`` dicts, where ``alias``
        is the exact substring of ``text`` that matched.
        """
        results: list[dict] = []
        taken: list[tuple[int, int]] = []

        def claim(span: tuple[int, int]) -> bool:
            if any(span[0] < e and s < span[1] for s, e in taken):
                return False
            taken.append(span)
            return True

        def add(canonical: str, m: re.Match, span: Optional[tuple[int, int]] = None) -> None:
            span = span or (m.start(), m.end())
            if claim(span):
                results.append({
                    "canonical": canonical,
                    "alias": text[span[0]:span[1]].strip(),
                    "span": span,
                })

        # 1. Airline telephony + flight number
        for m in self._telephony_re.finditer(text):
            name = m.group(1).lower()
            icao = self.telephony_map.get(name, name.upper()[:3])
            number = m.group(2).upper()
            add(f"{icao}{number}", m)

        # 2. US GA registrations
        for m in self._ga_us_re.finditer(text):
            add(f"N{m.group(1)}{_nato_to_letters(m.group(2))}", m)

        # 3. Canadian GA registrations
        for m in self._ga_ca_re.finditer(text):
            letters = _nato_to_letters(m.group(1) + m.group(2))
            add(f"C-{letters}", m)

        # 4. ICAO-style codes already present in the text
        for m in self._alpha_callsign_re.finditer(text.upper()):
            cs = m.group(1)
            if cs.startswith(_NOT_CALLSIGN_PREFIXES):
                continue
            add(cs, m)

        results.sort(key=lambda r: r["span"][0])
        return results

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    @staticmethod
    def extract_runway(text: str) -> Optional[str]:
        m = re.search(
            r"\brunways?\s+(\d{1,2})\s*(left|right|center|centre|[LRC]\b)?",
            text, re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r"\b(?:cleared\s+(?:to\s+land|for\s+(?:takeoff|the\s+option|(?:the\s+)?(?:visual|ils)(?:\s+approach)?))"
                r"|line\s+up\s+and\s+wait)\s+(?:runway\s+)?(\d{1,2})\s*(left|right|center|centre|[LRC]\b)?",
                text, re.IGNORECASE,
            )
        if not m:
            return None
        num = int(m.group(1))
        if not 1 <= num <= 36:
            return None
        suffix = RUNWAY_SUFFIX.get((m.group(2) or "").lower(), "")
        return f"{num:02d}{suffix}"

    @staticmethod
    def extract_altitude(text: str) -> Optional[str]:
        m = re.search(r"\b(?:flight\s+level|FL)\s*(\d{2,3})\b", text, re.IGNORECASE)
        if m:
            return f"FL{int(m.group(1)):03d}"

        m = re.search(
            r"\b(?:(?:climb|descend)(?:\s+and)?\s+maintain|climb(?:\s+to)?|descend(?:\s+to)?|maintain)\s+"
            r"(\d{3,5})\b(\s*feet)?(?!\s*(?:knots|kts))",
            text, re.IGNORECASE,
        )
        if m:
            val = int(m.group(1))
            # Bare three-digit values after "maintain" are usually speeds.
            if 1000 <= val <= 60000 or (100 <= val < 1000 and m.group(2)):
                return f"{val}ft"
        return None

    @staticmethod
    def extract_heading(text: str) -> Optional[str]:
        m = re.search(r"\bheading\s+(\d{1,3})\b", text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 360:
                return f"{val:03d}°"
        return None

    @staticmethod
    def extract_speed(text: str) -> Optional[str]:
        m = re.search(
            r"\b(?:(?:reduce|increase|slow)\s+(?:speed\s+)?(?:to\s+)?|speed\s+(?:to\s+)?)(\d{2,3})\b",
            text, re.IGNORECASE,
        ) or re.search(r"\b(\d{2,3})\s*(?:knots|kts)\b", text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 60 <= val <= 400:
                return f"{val}kts"
        return None

    @staticmethod
    def extract_frequency(text: str) -> Optional[str]:
        """VHF airband frequency, 118.000–136.975 MHz."""
        band = r"(1(?:1[89]|2\d|3[0-6]))"
        # "contact departure 124.85" / "124 85" / "12485"
        m = re.search(
            r"\b(?:contact|monitor|frequency|switch(?:ing)?\s+to)\b[^.]{0,40}?\b" + band + r"(?:\.|\s)?(\d{1,3})\b",
            text, re.IGNORECASE,
        ) or re.search(r"\b" + band + r"\.(\d{1,3})\b", text)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        return None

    @staticmethod
    def extract_squawk(text: str) -> Optional[str]:
        m = re.search(r"\bsquawk(?:ing)?\s+(?:code\s+)?([0-7]{4})\b", text, re.IGNORECASE)
        return m.group(1) if m else None

    def extract_events(self, text: str, ts: float, callsign: Optional[str] = None) -> list[ATCEvent]:
        events = []
        for pattern, event_type in self._event_compiled:
            m = pattern.search(text)
            if not m:
                continue
            # "hold short" should not also count as a generic hold, and an
            # explicit mayday should not also be a generic emergency.
            if event_type == EventType.EMERGENCY and any(
                e.type in (EventType.MAYDAY, EventType.PAN_PAN) for e in events
            ):
                continue
            events.append(ATCEvent(
                ts=ts,
                type=event_type,
                callsign_canonical=callsign,
                details=m.group(0),
                confidence=0.8,
            ))
        return events

    def extract_all(self, text: str, ts: float) -> dict:
        """Full extraction pipeline on a text segment.

        Returns ``{normalized, callsigns, fields, events}``. Callsign aliases
        refer to substrings of ``normalized``.
        """
        norm = normalize_numbers(text)
        callsigns = self.extract_callsigns(norm)

        fields = ExtractedFields(
            runway=self.extract_runway(norm),
            altitude=self.extract_altitude(norm),
            heading=self.extract_heading(norm),
            speed=self.extract_speed(norm),
            frequency=self.extract_frequency(norm),
            squawk=self.extract_squawk(norm),
        )

        primary_cs = callsigns[0]["canonical"] if callsigns else None
        events = self.extract_events(norm, ts, primary_cs)

        return {
            "normalized": norm,
            "callsigns": callsigns,
            "fields": fields,
            "events": events,
        }
