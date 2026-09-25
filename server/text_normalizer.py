"""Spoken-number normalisation for ATC transcripts.

Different ASR engines render numbers differently:

* stock Whisper / Nemotron / Parakeet usually write digits ("Delta 1492"),
  sometimes hyphenated ("1-4-9-2") or with thousands separators ("3,500");
* ATC fine-tunes trained on ATCO2/UWB-ATCC transcripts write digit words
  ("delta one four nine two", "three thousand five hundred");
* some mix the two ("eight 8 five 3").

`normalize_numbers` folds all of these into plain digit strings so the
extractor only has to deal with one form.
"""

from __future__ import annotations

import re

DIGIT_WORDS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "tree": 3,
    "four": 4, "five": 5, "fife": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "niner": 9,
}
TEEN_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
MAGNITUDE_WORDS = {"hundred": 100, "thousand": 1000}
DECIMAL_WORDS = {"point", "decimal", "dayseemal"}

# Words that are only numeric when they sit next to another number token.
_AMBIGUOUS = {"oh", "one"}

_NUMBER_WORDS = set(DIGIT_WORDS) | set(TEEN_WORDS) | set(TENS_WORDS) | set(MAGNITUDE_WORDS)

_TOKEN_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+(?!\d)"          # 3,500 / 11,000
    r"|\d+(?:-\d+)+"                     # 1-4-9-2 / 31-92
    r"|\d+(?:\.\d+)?"                    # 1492 / 124.85
    r"|[A-Za-z]+(?:[-'][A-Za-z]+)*"      # words, incl. ninety-two
    r"|\s+"
    r"|."
)


def _is_number_token(tok: str) -> bool:
    if tok[0].isdigit():
        return True
    low = tok.lower()
    if low in _NUMBER_WORDS:
        return True
    if "-" in low:
        return all(p in _NUMBER_WORDS for p in low.split("-"))
    return False


def _expand(tokens: list[str]) -> list[str]:
    """Split hyphenated words and lower-case; digit tokens are cleaned."""
    out: list[str] = []
    for tok in tokens:
        if tok[0].isdigit():
            out.append(tok.replace(",", "").replace("-", ""))
        else:
            out.extend(tok.lower().split("-"))
    return out


def _concat_value(parts: list[str]) -> str:
    """Digit-by-digit or grouped form ("fourteen ninety two") -> "1492"."""
    out = ""
    i = 0
    while i < len(parts):
        p = parts[i]
        if p[0].isdigit():
            out += p
        elif p in DIGIT_WORDS:
            out += str(DIGIT_WORDS[p])
        elif p in TEEN_WORDS:
            out += str(TEEN_WORDS[p])
        elif p in TENS_WORDS:
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt in DIGIT_WORDS and DIGIT_WORDS[nxt] > 0 and nxt != "oh":
                out += str(TENS_WORDS[p] + DIGIT_WORDS[nxt])
                i += 1
            else:
                out += str(TENS_WORDS[p])
        i += 1
    return out


def _magnitude_value(parts: list[str]) -> str:
    """Handle "three thousand five hundred", "one one thousand", "two hundred"."""
    total = 0
    for word, mult in (("thousand", 1000), ("hundred", 100)):
        if word in parts:
            idx = parts.index(word)
            left = _concat_value(parts[:idx]) or "1"
            total += int(left) * mult
            parts = parts[idx + 1:]
    rest = _concat_value(parts)
    if rest:
        total += int(rest)
    return str(total)


def _convert_run(tokens: list[str]) -> str:
    parts = _expand(tokens)
    # Split on decimal words ("one two four point eight five")
    groups: list[list[str]] = [[]]
    for p in parts:
        if p in DECIMAL_WORDS:
            groups.append([])
        else:
            groups[-1].append(p)
    converted = []
    for g in groups:
        if any(p in MAGNITUDE_WORDS for p in g):
            converted.append(_magnitude_value(g))
        else:
            converted.append(_concat_value(g))
    return ".".join(converted)


def normalize_numbers(text: str) -> str:
    """Rewrite spoken or fragmented numbers in ``text`` as digit strings.

    Adjacent number tokens separated only by whitespace are merged, so
    "United 2 3 7" and "united two three seven" both become "United 237".
    """
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _is_number_token(tok):
            out.append(tok)
            i += 1
            continue

        # Gather a run: number tokens joined by whitespace, with decimal
        # words allowed between two number tokens.
        run = [tok]
        j = i + 1
        while j < n:
            if tokens[j].isspace() and j + 1 < n:
                nxt = tokens[j + 1]
                if _is_number_token(nxt):
                    run.append(nxt)
                    j += 2
                    continue
                if (nxt.lower() in DECIMAL_WORDS and j + 3 < n
                        and tokens[j + 2].isspace() and _is_number_token(tokens[j + 3])):
                    run.extend([nxt, tokens[j + 3]])
                    j += 4
                    continue
            break

        words_only = [t.lower() for t in run if not t[0].isdigit()]
        if len(run) == 1 and words_only and words_only[0] in _AMBIGUOUS:
            out.append(tok)  # lone "one"/"oh" – leave it alone
        elif all(w in MAGNITUDE_WORDS for w in words_only) and len(words_only) == len(run):
            out.extend(tokens[i:j])  # bare "hundred"/"thousand"
        else:
            out.append(_convert_run(run))
        i = j
    return "".join(out)
