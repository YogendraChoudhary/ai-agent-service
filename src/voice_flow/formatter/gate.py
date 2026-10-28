"""
Does this utterance need the model?

A model call costs about a second whatever the prompt (measured: a one-word prompt takes 1.05 s on gemini-3.5-flash-lite,
the full prompt 1.16 s), and most dictation is a plain sentence that Whisper has already punctuated and capitalized. This
gate keeps the model for what only it does well: lists, self-corrections, fillers, repeats, spoken punctuation, and (when
app formats are switched on) email layout. It errs towards the model: a false positive costs a second, a false negative
costs a wrong sentence.
"""

from __future__ import annotations

import re
from typing import Optional

from src.voice_flow.formatter.patterns import GREETING_DETECT_RE, SIGN_OFF_CLOSING_RE, SPOKEN_COMMANDS_RE
from src.voice_flow.formatter.rules import LIST_CUES
from src.voice_flow.schemas import AppContext, CursorContext, effective_context

# Whisper decodes one 30 s window at a time, and a speaker at ~90 words per minute fills about 45 words of it. Above that the
# text spans several windows: paragraphing, sentence breaks across the window edges and list structure need the model.
MAX_FAST_WORDS = 45

_CORRECTION = re.compile(r"\b(?:wait|actually|scratch\s+that|i\s+mean|sorry|make\s+that|never\s?mind|or\s+rather|rather|correction)\b", re.IGNORECASE)
_FILLER = re.compile(r"\b(?:um+|uh+|er+m?|hmm+|mm+|ah+|you\s+know|basically|sort\s+of|kind\s+of|i\s+guess)\b|,\s*like\b|\blike\s*,", re.IGNORECASE)
_REPEAT = re.compile(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", re.IGNORECASE)          # "the the", "we need to we need to"
_ASR_QUIRK = re.compile(r"\bmarket\s*(?:\.|point)\s*\d", re.IGNORECASE)       # "mark it point 1" heard as "market.1"


def needs_model(raw: str, app_context: AppContext, cursor: Optional[CursorContext] = None) -> Optional[str]:
    """The reason the model is needed, or None when rules are enough."""
    context = effective_context(app_context)
    words = raw.split()
    if context in (AppContext.CODE_COMMENT, AppContext.MARKDOWN_DOC):
        return f"{context.value} context"
    if len(words) > MAX_FAST_WORDS:
        return "long utterance"
    if len(words) >= 6 and not re.search(r"[.!?]", raw):
        return "unpunctuated text"                       # not from Whisper; capitalization and sentence breaks are guesses
    for name, pattern in (("list cue", LIST_CUES), ("correction", _CORRECTION), ("filler", _FILLER),
                          ("spoken command", SPOKEN_COMMANDS_RE), ("repeat", _REPEAT), ("asr quirk", _ASR_QUIRK)):
        if pattern.search(raw):
            return name
    if context is AppContext.EMAIL and (GREETING_DETECT_RE.search(raw) or SIGN_OFF_CLOSING_RE.search(" ".join(words[-6:]))):
        return "email greeting or sign-off"
    return None
