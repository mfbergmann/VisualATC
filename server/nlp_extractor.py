"""NLP extraction for ATC transcripts: callsigns, entities, events."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import ATCEvent, EventType, ExtractedFields, Mention

logger = logging.getLogger("visualatc.nlp")

# ---------------------------------------------------------------------------
# NATO / ATC number word mapping
# ---------------------------------------------------------------------------

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0",
    "one": "1", "won": "1",
    "two": "2",
    "three": "3", "tree": "3",
    "four": "4",
    "five": "5", "fife": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ait": "8",
    "nine": "9", "niner": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "00",
    "thousand": "000",
}

# Numbers 0-9 as words for pattern matching
SINGLE_DIGIT_WORDS = (
    r"(?:zero|oh|one|two|three|tree|four|five|fife|six|seven|eight|ait|nine|niner)"
)

# Number-like token: digit words, actual digits, or digits with hyphens
NUMBER_TOKEN = (
    r"(?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d|\d)"
)

# NATO phonetic alphabet
NATO_PHONETIC = {
    "alfa": "A", "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "echo": "E",
    "foxtrot": "F", "fox trot": "F",
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
    "x-ray": "X", "xray": "X", "x ray": "X",
    "yankee": "Y",
    "zulu": "Z",
}

# Direction suffixes for runways
RUNWAY_SUFFIX = {"left": "L", "right": "R", "center": "C", "centre": "C"}

# Common Whisper mistranscriptions of airline names
AIRLINE_ALIASES = {
    # Whisper often hears these wrong
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

# ---------------------------------------------------------------------------
# Event keyword patterns
# ---------------------------------------------------------------------------

EVENT_PATTERNS: list[tuple[str, EventType]] = [
    (r"\bgo[\s\-]?around\b", EventType.GO_AROUND),
    (r"\bmissed\s+approach\b", EventType.GO_AROUND),
    (r"\bdivert(?:ing|ed|s)?\b|\bdiversion\b|\balternate\b", EventType.DIVERT),
    (r"\bhold(?:ing)?\s+short\b", EventType.HOLD_SHORT),
    (r"\b(?:hold(?:ing)?(?:\s+pattern)?|enter(?:ing)?\s+hold)\b", EventType.HOLD),
    (r"\brunway\s+change\b|\blanding\s+runway\b|\bchange\s+runway\b", EventType.RUNWAY_CHANGE),
    (r"\bwind\s*shear\b|\bmicroburst\b", EventType.WINDSHEAR),
    (r"\bminimum\s+fuel\b", EventType.MINIMUM_FUEL),
    (r"\bmayday\b", EventType.MAYDAY),
    (r"\bpan[\s\-]?pan\b", EventType.PAN_PAN),
    (r"\bemergency\b|\bdeclare(?:s|d)?\s+emergency\b|\bsquawk(?:ing)?\s+7700\b", EventType.EMERGENCY),
]


def _strip_hyphens_to_digits(s: str) -> str:
    """Convert '1-3-2-0' or '31-92' to '13200' or '3192'."""
    return re.sub(r"[\-\s]", "", s)


def _mixed_to_digits(text: str) -> str:
    """Convert a mixed sequence of digit words and actual digits to a digit string.

    Handles: 'eight eight five three', '8853', '1-3-2-0', 'one eight zero',
    'eight 8 five 3', etc.
    """
    # First strip hyphens from digit groups
    text = re.sub(r"(\d)[\-](\d)", r"\1\2", text)
    tokens = text.lower().split()
    result = []
    for token in tokens:
        if token in WORD_TO_DIGIT:
            result.append(WORD_TO_DIGIT[token])
        elif re.match(r"^\d+$", token):
            result.append(token)
        elif token in NATO_PHONETIC:
            result.append(NATO_PHONETIC[token])
        # Skip unrecognized
    return "".join(result)


class ATCExtractor:
    """Extract ATC entities from transcribed text."""

    def __init__(self, telephony_path: Optional[str] = None):
        if telephony_path is None:
            telephony_path = str(Path(__file__).parent / "data" / "airline_telephony.json")

        # Load telephony from file and merge with built-in aliases
        self.telephony_map = self._load_telephony(telephony_path)
        for alias, icao in AIRLINE_ALIASES.items():
            if alias not in self.telephony_map:
                self.telephony_map[alias] = icao

        # Build regex for telephony names (longest first to avoid partial matches)
        names = sorted(self.telephony_map.keys(), key=len, reverse=True)
        if names:
            escaped = [re.escape(n) for n in names]
            # Match: airline name + (digit words / actual digits / hyphenated digits)
            self._telephony_re = re.compile(
                r"\b(" + "|".join(escaped) + r")\s+"
                r"((?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*(?:\d)?"
                r")(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*(?:\d)?))*"
                r"(?:\s+(?:heavy|super))?)",
                re.IGNORECASE,
            )
        else:
            self._telephony_re = None

        # Pattern for alphanumeric callsigns already in text: "ACA8853", "N123AB", "DAL123"
        self._alpha_callsign_re = re.compile(
            r"\b([A-Z]{2,4}\d{1,5}[A-Z]{0,2})\b"
        )

        # Pattern for "November" style GA callsigns
        nato_letter_words = "|".join(
            sorted(NATO_PHONETIC.keys(), key=len, reverse=True)
        )
        self._ga_callsign_re = re.compile(
            r"\b(november)\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?|" + nato_letter_words + r")"
            r"(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?|" + nato_letter_words + r"))*)",
            re.IGNORECASE,
        )

        # Pattern for standalone flight-number-like patterns when preceded by
        # ATC verbiage: e.g., "cleared to land 697" or "contact ground 697"
        # We look for a 3-4 digit number that could be a flight number
        self._standalone_flight_re = re.compile(
            r"(?:cleared|contact|taxi|hold|turn|descend|climb|maintain|roger|copy"
            r"|squawk|ident|say again|read back|good day)"
            r"[^.]{0,30}?\b(\d{3,4})\b",
            re.IGNORECASE,
        )

        # Precompile event patterns
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

    def extract_callsigns(self, text: str) -> list[dict]:
        """
        Extract callsigns from text.

        Returns list of {canonical, alias, span} dicts.
        """
        results = []
        seen_spans = set()

        def _overlaps(span):
            for s in seen_spans:
                if s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1]:
                    return True
                if span[0] <= s[0] < span[1] or span[0] < s[1] <= span[1]:
                    return True
            return False

        # 1. Airline telephony + numbers: "Air Canada eight eight five three"
        #    or "Delta 1-3-2-0" or "United 237"
        if self._telephony_re:
            for m in self._telephony_re.finditer(text):
                span = (m.start(), m.end())
                if _overlaps(span):
                    continue

                airline_name = m.group(1).lower()
                number_part = m.group(2)

                # Strip "heavy"/"super" suffix
                clean_number = re.sub(r"\s+(?:heavy|super)\s*$", "", number_part, flags=re.IGNORECASE)
                weight_class = number_part[len(clean_number):].strip()

                icao_prefix = self.telephony_map.get(airline_name, airline_name.upper()[:3])
                digits = _mixed_to_digits(clean_number)

                if digits:
                    canonical = f"{icao_prefix}{digits}"
                    alias = m.group(0).strip()
                    seen_spans.add(span)
                    results.append({
                        "canonical": canonical,
                        "alias": alias,
                        "span": span,
                    })

        # 2. GA callsigns: "November one two three alpha bravo"
        for m in self._ga_callsign_re.finditer(text):
            span = (m.start(), m.end())
            if _overlaps(span):
                continue

            rest = m.group(2)
            suffix = _mixed_to_digits(rest)
            if suffix:
                canonical = f"N{suffix}"
                alias = m.group(0).strip()
                seen_spans.add(span)
                results.append({
                    "canonical": canonical,
                    "alias": alias,
                    "span": span,
                })

        # 3. Alphanumeric callsigns in text: "ACA8853", "N123AB", "DAL123"
        for m in self._alpha_callsign_re.finditer(text.upper()):
            start = 0
            search_text = text.upper()
            # Find position in original text
            pos = search_text.find(m.group(0), start)
            if pos == -1:
                continue
            span = (pos, pos + len(m.group(0)))
            if _overlaps(span):
                continue

            cs = m.group(1)
            # Must have both letters and digits
            if re.search(r"[A-Z]", cs) and re.search(r"\d", cs):
                # Filter out things that look like runway designations or frequencies
                if re.match(r"^\d{2,3}[LRC]?$", cs):
                    continue
                seen_spans.add(span)
                results.append({
                    "canonical": cs,
                    "alias": cs,
                    "span": span,
                })

        # 4. Whisper often writes "airline + space + digits" without the exact
        #    telephony match working (e.g., "National Air 1320").
        #    Scan for "<Known Airline> <digits>" with the digits possibly
        #    containing hyphens or spaces
        for m in re.finditer(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d[\d\-\s]{0,10}\d|\d{2,5})\b",
            text,
        ):
            span = (m.start(), m.end())
            if _overlaps(span):
                continue

            name = m.group(1).lower()
            num_raw = m.group(2)

            # Check if this name is a known airline
            icao = self.telephony_map.get(name)
            if not icao:
                # Try two-word lookup
                continue

            digits = _strip_hyphens_to_digits(num_raw)
            if digits and len(digits) >= 2:
                canonical = f"{icao}{digits}"
                alias = m.group(0).strip()
                seen_spans.add(span)
                results.append({
                    "canonical": canonical,
                    "alias": alias,
                    "span": span,
                })

        return results

    def extract_runway(self, text: str) -> Optional[str]:
        """Extract runway designation from text."""
        text_lower = text.lower()

        # Match: "runway" + number words/digits + optional L/R/C
        # Also handle Whisper oddities like "runway 2S" → "runway 25"
        m = re.search(
            r"runway\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?)"
            r"(?:[\s\-]+(?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?))*)"
            r"(?:[\s\-]*(left|right|center|centre|l|r|c))?",
            text_lower,
        )
        if m:
            num = _mixed_to_digits(m.group(1))
            suffix = ""
            if m.group(2):
                s = m.group(2).lower()
                suffix = RUNWAY_SUFFIX.get(s, s.upper()[0] if s else "")
            if num:
                return f"{num}{suffix}"

        # Also try to match bare digit-based runway mentions: "two zero left"
        # after "cleared to land", "cleared for takeoff", etc.
        m = re.search(
            r"(?:cleared (?:to land|for (?:takeoff|the option|the visual))|"
            r"departing|landing)\s+(?:runway\s+)?"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?)"
            r"(?:[\s\-]+(?:" + SINGLE_DIGIT_WORDS + r"|\d[\d\-]*\d?))*)"
            r"(?:[\s\-]*(left|right|center|centre|l|r|c))?",
            text_lower,
        )
        if m:
            num = _mixed_to_digits(m.group(1))
            suffix = ""
            if m.group(2):
                s = m.group(2).lower()
                suffix = RUNWAY_SUFFIX.get(s, s.upper()[0] if s else "")
            if num and 1 <= len(num) <= 2:
                return f"{num}{suffix}"

        return None

    def extract_altitude(self, text: str) -> Optional[str]:
        """Extract altitude from text."""
        text_lower = text.lower()

        # Flight level
        m = re.search(
            r"flight\s+level\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)",
            text_lower,
        )
        if m:
            digits = _mixed_to_digits(m.group(1))
            if digits:
                return f"FL{digits}"

        # "maintain/climb and maintain/descend and maintain" + altitude
        m = re.search(
            r"(?:maintain|climb(?:\s+and\s+maintain)?|descend(?:\s+and\s+maintain)?)\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d|thousand|hundred))*)",
            text_lower,
        )
        if m:
            raw = m.group(1)
            digits = _mixed_to_digits(raw)
            if digits:
                if "thousand" in raw:
                    return f"{digits}ft"
                return f"{digits}ft"

        return None

    def extract_heading(self, text: str) -> Optional[str]:
        """Extract heading from text."""
        text_lower = text.lower()
        m = re.search(
            r"(?:fly\s+)?heading\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)",
            text_lower,
        )
        if m:
            digits = _mixed_to_digits(m.group(1))
            if digits:
                return f"{digits}\u00b0"
        return None

    def extract_speed(self, text: str) -> Optional[str]:
        """Extract speed from text."""
        text_lower = text.lower()
        m = re.search(
            r"(?:speed|reduce\s+speed(?:\s+to)?|maintain\s+(?:\w+\s+)?speed|increase\s+speed(?:\s+to)?)\s+"
            r"(?:to\s+)?"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)"
            r"(?:\s*(?:knots|kts))?",
            text_lower,
        )
        if m:
            digits = _mixed_to_digits(m.group(1))
            if digits:
                return f"{digits}kts"
        return None

    def extract_frequency(self, text: str) -> Optional[str]:
        """Extract frequency from text."""
        text_lower = text.lower()

        # "contact [facility] on 124.6" or "contact [facility] one two four point six"
        m = re.search(
            r"(?:contact|monitor)\s+\w+(?:\s+\w+)?\s+(?:on\s+)?"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:[\s\.](?:" + SINGLE_DIGIT_WORDS + r"|\d|point|decimal))*)",
            text_lower,
        )
        if not m:
            # Try bare frequency pattern: "one two four point six"
            m = re.search(
                r"\b(\d{2,3}[\.\s]\d{1,2})\b",
                text_lower,
            )

        if m:
            raw = m.group(1)
            raw = re.sub(r"\bpoint\b|\bdecimal\b", ".", raw)
            tokens = raw.split()
            freq_str = ""
            for t in tokens:
                if t == ".":
                    freq_str += "."
                elif t in WORD_TO_DIGIT:
                    freq_str += WORD_TO_DIGIT[t]
                elif re.match(r"^[\d\.]+$", t):
                    freq_str += t
            if freq_str:
                return freq_str
        return None

    def extract_events(self, text: str, ts: float, callsign: Optional[str] = None) -> list[ATCEvent]:
        """Extract ATC events from text."""
        events = []
        for pattern, event_type in self._event_compiled:
            m = pattern.search(text)
            if m:
                events.append(ATCEvent(
                    ts=ts,
                    type=event_type,
                    callsign_canonical=callsign,
                    details=m.group(0),
                    confidence=0.8,
                ))
        return events

    def extract_all(self, text: str, ts: float) -> dict:
        """
        Full extraction pipeline on a text segment.

        Returns {callsigns: [...], fields: ExtractedFields, events: [...]}
        """
        callsigns = self.extract_callsigns(text)

        fields = ExtractedFields(
            runway=self.extract_runway(text),
            altitude=self.extract_altitude(text),
            heading=self.extract_heading(text),
            speed=self.extract_speed(text),
            frequency=self.extract_frequency(text),
        )

        # Associate events with the first callsign found (if any)
        primary_cs = callsigns[0]["canonical"] if callsigns else None
        events = self.extract_events(text, ts, primary_cs)

        return {
            "callsigns": callsigns,
            "fields": fields,
            "events": events,
        }
