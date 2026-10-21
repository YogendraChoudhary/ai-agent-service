"""
Official tone: the rules the prompt states, and the code that checks them afterwards.

The prompt asks; lint() verifies. Errors send the template back for one repair pass; warnings go to the
editor so the author can decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .schema import PREHEADER_TARGET, SUBJECT_TARGET_MAX, TEXT_FIELDS

TONE_PROFILES: dict[str, str] = {
    "formal": (
        "Formal: measured, precise and impersonal. Use 'Dear <name>,' and close with 'Yours sincerely,' or "
        "'Kind regards,'. No contractions. No exclamation marks."
    ),
    "professional": (
        "Professional: clear, courteous and direct. Contractions are acceptable but sparing. "
        "Close with 'Kind regards,' or 'Best regards,'. No exclamation marks."
    ),
    "professional_warm": (
        "Professional and warm: friendly and appreciative while remaining businesslike. "
        "At most one exclamation mark in the whole email. Close with 'Warm regards,' or 'Kind regards,'."
    ),
}

BASE_RULES: tuple[str, ...] = (
    "Write for an official, organisational audience. Plain, precise language; short sentences "
    "(aim for 25 words or fewer); active voice.",
    "State the purpose of the email in the first paragraph and ask for at most one action.",
    "No emojis, slang, idioms, rhetorical questions, or ALL-CAPS words (acronyms are fine).",
    "No hype, superlatives or false urgency (for example 'amazing', 'don't miss out', 'act now').",
    "NEVER invent facts: dates, times, amounts, prices, percentages, names, reference numbers, venues, "
    "statistics or policy details. If a fact the email needs is not in the brief, write [[fill: <what is "
    "needed>]] in its place.",
    "Do not add plans, promises, agenda items, next steps or contact routes the brief does not state (for "
    "example a Q&A session, a calendar invite, 'further updates will follow', 'contact the X team'). Say only "
    "what the brief says, clearly and completely.",
    "Make no legal, medical or financial commitments beyond what the brief states.",
    "Use inclusive, neutral language and the spelling and date conventions of the requested language.",
)

# Phrases that read as hype, spam or pressure in an official email. Matched case-insensitively on word
# boundaries. Extend per deployment rather than in the prompt.
BANNED_PHRASES: tuple[str, ...] = (
    "act now", "don't miss out", "dont miss out", "limited time only", "once in a lifetime", "100% free",
    "guaranteed", "risk-free", "risk free", "click here", "amazing", "awesome", "revolutionary",
    "game-changer", "game changer", "mind-blowing", "unbelievable", "insane", "hurry",
)
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in BANNED_PHRASES) + r")\b", re.IGNORECASE)

# Words in capitals that are acronyms, not shouting. Words of 2-4 capitals are treated as acronyms anyway.
ACRONYM_ALLOWLIST = frozenset({"ASAP", "FAQ", "GDPR", "HIPAA", "NOTE", "RSVP", "UTC", "GMT", "IST", "AGM"})
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{5,}\b")

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF⭐⬆↔-↪]"
)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*[a-z0-9_]+\s*\}\}|\[\[fill:[^\]]*\]\]|\(link:[a-z0-9_]+\)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,:/]\d+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
LONG_SENTENCE_WORDS = 35


@dataclass(frozen=True)
class Issue:
    kind: str
    message: str
    block_id: Optional[str] = None
    field: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class LintResult:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)


def _walk(node, path: str, key: Optional[str]) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k, k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]", key)
    elif isinstance(node, str) and key in TEXT_FIELDS:
        yield path, node


def iter_text(template: dict) -> Iterator[tuple[Optional[str], str, str]]:
    """(block_id, field path, text) for every piece of reader-visible copy, including subject and preheader."""
    meta = template.get("meta") or {}
    for i, s in enumerate(meta.get("subject_options") or []):
        if isinstance(s, str):
            yield None, f"subject_options[{i}]", s
    if isinstance(meta.get("preheader"), str):
        yield None, "preheader", meta["preheader"]
    for block in template.get("blocks") or []:
        if isinstance(block, dict):
            for path, text in _walk(block, "", None):
                yield block.get("id"), path, text


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text)


def numbers_in(texts: Iterable[str]) -> set[str]:
    out = set()
    for t in texts:
        out.update(_NUMBER_RE.findall(_strip_placeholders(t or "")))
    return out


def lint(template: dict, tone: str, source_texts: Iterable[str]) -> LintResult:
    """
    Check tone rules on a schema-valid template. source_texts is everything the user supplied (brief,
    sender, audience...): numbers in the email that appear nowhere in it are flagged as possibly invented.
    """
    result = LintResult()
    allowed_numbers = numbers_in(source_texts)
    exclamations: list[tuple[Optional[str], str]] = []
    subject_exclamations: list[tuple[Optional[str], str]] = []

    for bid, fld, raw in iter_text(template):
        text = _strip_placeholders(raw)
        if _EMOJI_RE.search(text):
            result.errors.append(Issue("emoji", "Remove emojis.", bid, fld))
        for m in _BANNED_RE.finditer(text):
            result.errors.append(Issue("banned_phrase", f"Replace '{m.group(0)}' with neutral wording.", bid, fld))
        for m in _CAPS_WORD_RE.finditer(text):
            if m.group(0) not in ACRONYM_ALLOWLIST:
                result.errors.append(Issue("all_caps", f"Write '{m.group(0)}' in normal case.", bid, fld))
        marks = [(bid, fld)] * text.count("!")
        if fld.startswith("subject_options[") and bid is None:
            # Only one subject is sent; check the candidate with the most punctuation.
            if len(marks) > len(subject_exclamations):
                subject_exclamations = marks
        else:
            exclamations.extend(marks)
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if len(sentence.split()) > LONG_SENTENCE_WORDS:
                result.warnings.append(Issue(
                    "long_sentence", f"Sentence over {LONG_SENTENCE_WORDS} words; consider splitting.", bid, fld))
        for n in _NUMBER_RE.findall(text):
            if n not in allowed_numbers:
                result.warnings.append(Issue(
                    "possible_invented_fact",
                    f"'{n}' does not appear in the brief. Confirm it or replace it with a [[fill: ...]] slot.",
                    bid, fld))

    allowed_bangs = 1 if tone == "professional_warm" else 0
    exclamations.extend(subject_exclamations)
    if len(exclamations) > allowed_bangs:
        bid, fld = exclamations[0]
        result.errors.append(Issue(
            "exclamation",
            f"Use at most {allowed_bangs} exclamation mark(s) in the whole email for the '{tone}' tone "
            f"(found {len(exclamations)}).",
            bid, fld))

    meta = template.get("meta") or {}
    for i, s in enumerate(meta.get("subject_options") or []):
        if isinstance(s, str) and len(s) > SUBJECT_TARGET_MAX:
            result.warnings.append(Issue(
                "subject_length", f"Subject option over {SUBJECT_TARGET_MAX} characters may be cut off.",
                None, f"subject_options[{i}]"))
    pre = meta.get("preheader") or ""
    if not PREHEADER_TARGET[0] <= len(pre) <= PREHEADER_TARGET[1]:
        result.warnings.append(Issue(
            "preheader_length", f"Preheader works best at {PREHEADER_TARGET[0]}-{PREHEADER_TARGET[1]} characters.",
            None, "preheader"))
    return result
