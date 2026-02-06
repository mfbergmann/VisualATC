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
    "two": "2", "to": "2", "too": "2",
    "three": "3", "tree": "3",
    "four": "4", "for": "4", "fore": "4",
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

# NATO phonetic alphabet
NATO_PHONETIC = {
    "alfa": "A", "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "delta letter": "D",  # "delta" alone is airline
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

# Reverse: letter -> NATO word (for matching "november one two three alpha bravo")
NATO_LETTERS = {v: k for k, v in NATO_PHONETIC.items()}

# Direction suffixes for runways
RUNWAY_SUFFIX = {"left": "L", "right": "R", "center": "C", "centre": "C"}

# ---------------------------------------------------------------------------
# Event keyword patterns
# ---------------------------------------------------------------------------

EVENT_PATTERNS: list[tuple[str, EventType]] = [
    (r"\bgo[\s-]?around\b", EventType.GO_AROUND),
    (r"\bdivert(?:ing|ed|s)?\b|\bdiversion\b|\balternate\b", EventType.DIVERT),
    (r"\bhold(?:ing)?\s+short\b", EventType.HOLD_SHORT),
    (r"\b(?:hold(?:ing)?|enter(?:ing)?\s+hold)\b", EventType.HOLD),
    (r"\brunway\s+change\b|\blanding\s+runway\b", EventType.RUNWAY_CHANGE),
    (r"\bwind\s*shear\b", EventType.WINDSHEAR),
    (r"\bminimum\s+fuel\b", EventType.MINIMUM_FUEL),
    (r"\bmayday\b", EventType.MAYDAY),
    (r"\bpan[\s-]?pan\b", EventType.PAN_PAN),
    (r"\bemergency\b|\bdeclare(?:s|d)?\s+emergency\b", EventType.EMERGENCY),
]


class ATCExtractor:
    """Extract ATC entities from transcribed text."""

    def __init__(self, telephony_path: Optional[str] = None):
        if telephony_path is None:
            telephony_path = str(Path(__file__).parent / "data" / "airline_telephony.json")
        self.telephony_map = self._load_telephony(telephony_path)
        # Build regex for telephony names (longest first to avoid partial matches)
        names = sorted(self.telephony_map.keys(), key=len, reverse=True)
        if names:
            escaped = [re.escape(n) for n in names]
            self._telephony_re = re.compile(
                r"\b(" + "|".join(escaped) + r")\s+"
                r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)+(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)",
                re.IGNORECASE,
            )
        else:
            self._telephony_re = None

        # Pattern for alphanumeric callsigns: AAA1234, N123AB
        self._alpha_callsign_re = re.compile(
            r"\b([A-Z]{2,4}\d{1,5}[A-Z]{0,2})\b"
        )

        # Pattern for "November" style GA callsigns: November 1 2 3 Alpha Bravo
        nato_letter_words = "|".join(
            sorted([k for k in NATO_PHONETIC.keys() if k != "delta letter"],
                   key=len, reverse=True)
        )
        self._ga_callsign_re = re.compile(
            r"\b(november)\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d|" + nato_letter_words + r")"
            r"(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d|" + nato_letter_words + r"))*)",
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
            # Remove comment keys
            return {k.lower(): v for k, v in data.items() if not k.startswith("_")}
        except Exception as e:
            logger.warning("Failed to load telephony map from %s: %s", path, e)
            return {}

    def _words_to_digits(self, text: str) -> str:
        """Convert a sequence of spoken number words to a digit string."""
        tokens = text.lower().split()
        digits = []
        for token in tokens:
            if token in WORD_TO_DIGIT:
                d = WORD_TO_DIGIT[token]
                digits.append(d)
            elif token.isdigit():
                digits.append(token)
            elif token in NATO_PHONETIC:
                digits.append(NATO_PHONETIC[token])
            # Skip unrecognized words
        return "".join(digits)

    def _words_to_number_string(self, text: str) -> str:
        """Convert spoken ATC number words to concatenated digits.

        'one eight' => '18', 'two zero' => '20', 'niner' => '9'
        """
        tokens = text.lower().split()
        result = []
        for token in tokens:
            if token in WORD_TO_DIGIT:
                val = WORD_TO_DIGIT[token]
                result.append(val)
            elif token.isdigit():
                result.append(token)
        return "".join(result)

    def extract_callsigns(self, text: str) -> list[dict]:
        """
        Extract callsigns from text.

        Returns list of {canonical, alias, span} dicts.
        """
        results = []
        seen_spans = set()

        # 1. Airline telephony + numbers: "Air Canada eight eight five three"
        if self._telephony_re:
            for m in self._telephony_re.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                airline_name = m.group(1).lower()
                number_part = m.group(2)
                icao_prefix = self.telephony_map.get(airline_name, airline_name.upper()[:3])
                digits = self._words_to_number_string(number_part)
                if digits:
                    canonical = f"{icao_prefix}{digits}"
                    alias = m.group(0).strip()
                    results.append({
                        "canonical": canonical,
                        "alias": alias,
                        "span": span,
                    })

        # 2. GA callsigns: "November one two three alpha bravo"
        for m in self._ga_callsign_re.finditer(text):
            span = (m.start(), m.end())
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                continue
            seen_spans.add(span)

            rest = m.group(2)
            suffix = self._words_to_digits(rest)
            canonical = f"N{suffix}"
            alias = m.group(0).strip()
            results.append({
                "canonical": canonical,
                "alias": alias,
                "span": span,
            })

        # 3. Alphanumeric callsigns already in text: "ACA8853", "N123AB", "DAL123"
        for m in self._alpha_callsign_re.finditer(text.upper()):
            # Map to original text positions
            upper_text = text.upper()
            start = text.upper().find(m.group(0))
            if start == -1:
                continue
            span = (start, start + len(m.group(0)))
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                continue
            seen_spans.add(span)

            cs = m.group(1)
            # Only accept if it looks like a real callsign (has both letters and digits)
            if re.search(r"[A-Z]", cs) and re.search(r"\d", cs):
                results.append({
                    "canonical": cs,
                    "alias": cs,
                    "span": span,
                })

        return results

    def extract_runway(self, text: str) -> Optional[str]:
        """Extract runway designation from text."""
        text_lower = text.lower()

        # "runway two zero left" or "runway 20L"
        m = re.search(
            r"runway\s+"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)"
            r"(?:\s*(left|right|center|centre|l|r|c))?",
            text_lower,
        )
        if m:
            num = self._words_to_number_string(m.group(1))
            suffix = ""
            if m.group(2):
                s = m.group(2).lower()
                suffix = RUNWAY_SUFFIX.get(s, s.upper()[0] if s else "")
            if num:
                return f"{num}{suffix}"
        return None

    def extract_altitude(self, text: str) -> Optional[str]:
        """Extract altitude from text."""
        text_lower = text.lower()

        patterns = [
            r"(?:maintain|climb(?:\s+and\s+maintain)?|descend(?:\s+and\s+maintain)?|at)\s+"
            r"((?:flight\s+level\s+)?(?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d|thousand|hundred))*)",
        ]
        for pat in patterns:
            m = re.search(pat, text_lower)
            if m:
                raw = m.group(1)
                if "flight level" in raw:
                    raw = raw.replace("flight level", "").strip()
                    digits = self._words_to_number_string(raw)
                    if digits:
                        return f"FL{digits}"
                else:
                    digits = self._words_to_number_string(raw)
                    if digits:
                        # If "thousand" was in original, multiply
                        if "thousand" in m.group(1).lower():
                            return f"{digits}ft"
                        return f"{digits}ft" if len(digits) <= 3 else f"{digits}ft"
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
            digits = self._words_to_number_string(m.group(1))
            if digits:
                return f"{digits}°"
        return None

    def extract_speed(self, text: str) -> Optional[str]:
        """Extract speed from text."""
        text_lower = text.lower()
        m = re.search(
            r"(?:speed|reduce\s+speed(?:\s+to)?|maintain\s+speed|increase\s+speed(?:\s+to)?)\s+"
            r"(?:to\s+)?"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d))*)"
            r"(?:\s*(?:knots|kts))?",
            text_lower,
        )
        if m:
            digits = self._words_to_number_string(m.group(1))
            if digits:
                return f"{digits}kts"
        return None

    def extract_frequency(self, text: str) -> Optional[str]:
        """Extract frequency from text."""
        text_lower = text.lower()
        m = re.search(
            r"(?:contact|monitor|frequency)\s+\w+\s+(?:on\s+)?"
            r"((?:" + SINGLE_DIGIT_WORDS + r"|\d)(?:\s+(?:" + SINGLE_DIGIT_WORDS + r"|\d|point|decimal|\.))*)",
            text_lower,
        )
        if m:
            raw = m.group(1)
            # Replace "point" / "decimal" with "."
            raw = re.sub(r"\bpoint\b|\bdecimal\b", ".", raw)
            tokens = raw.split()
            freq_str = ""
            for t in tokens:
                if t == ".":
                    freq_str += "."
                elif t in WORD_TO_DIGIT:
                    freq_str += WORD_TO_DIGIT[t]
                elif t.replace(".", "").isdigit():
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
