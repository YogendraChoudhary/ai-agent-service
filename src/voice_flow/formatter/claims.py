"""
Claim retention for Polished style.

Polishing rewrites sentences, and a rewrite can quietly lose something the speaker said: in a trial it dropped "plug-and-play"
and renamed "points" as "advantages". This check lists the terms a rewrite must keep, and the engine falls back to the Clean
style when one is missing. The terms are the ones that carry a claim and are easy to see in text:

  acronyms (USP, SDK), hyphenated terms (plug-and-play), numbers other than 0 and 1 (10 minutes, 3 times, $5,000),
  and capitalised names in the middle of a sentence (Editor, Sarah).

List-cue numbers ("point 5") and numbers the speaker retracted ("5 seconds, wait no, 10") are not claims and are skipped.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.voice_flow.formatter.rules import _POINT_CUE, _clean_speech, _repair_asr

_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                 "hundred": 100, "thousand": 1000}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def _digits(text: str) -> str:
    """"three times" -> "3 times", so a rewrite may spell a number either way."""
    return _NUMBER_WORD_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text)


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


ALL_KINDS = frozenset({"acronym", "hyphen", "number", "name"})
# What Clean style must keep. Numbers and names are left out: a speaker retracts them ("send it to Bob, make that Alice"), and
# asking the model to "keep Bob" would be wrong. Acronyms and hyphenated terms are essentially never retracted.
CLEAN_KINDS = frozenset({"acronym", "hyphen"})
# Polished may shorten, and an acronym that only labelled an area ("in the USP area") can go; forcing it back produced "establishing
# our USP area". What it must keep are the claims: hyphenated terms and numbers.
POLISHED_KINDS = frozenset({"hyphen", "number"})
_PATTERNS = (("acronym", r"\b[A-Z]{2,}s?\b"),
             ("hyphen", r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b"),
             ("number", r"\$?\b\d[\d,.]*\d\b|\b[2-9]\b"),          # not 0 or 1
             ("name", r"(?<=[a-z,;] )[A-Z][a-z]{2,}\b"))                # capitalised in the middle of a sentence


def key_terms(raw: str, kinds: frozenset = ALL_KINDS) -> List[str]:
    text = _POINT_CUE.sub(" ", _clean_speech(_repair_asr(" ".join(raw.split()))))
    text = _digits(text)
    terms: List[str] = []
    for kind, pattern in _PATTERNS:
        if kind not in kinds:
            continue
        for m in re.finditer(pattern, text):
            term = m.group(0).strip(".,")
            if term and term.lower() not in {t.lower() for t in terms}:
                terms.append(term)
    return terms


def missing_claims(raw: str, out: str, kinds: frozenset = ALL_KINDS) -> List[str]:
    """Every key term of `raw` that `out` no longer contains."""
    squashed = _squash(_digits(out))
    tokens = set(re.findall(r"[a-z0-9]+", _digits(out).lower()))
    missing: List[str] = []
    for term in key_terms(raw, kinds):
        norm = _squash(term)
        if not norm:
            continue
        if "-" in term or any(ch.isdigit() for ch in norm):
            if norm in squashed:
                continue
        elif norm in tokens or norm.rstrip("s") in tokens or norm + "s" in tokens:
            continue
        missing.append(term)
    return missing


def check_claims(raw: str, out: str) -> Optional[str]:
    """The first key term of `raw` that `out` no longer contains, or None."""
    missing = missing_claims(raw, out)
    return missing[0] if missing else None
