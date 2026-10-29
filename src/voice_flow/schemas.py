"""
Data contracts for voice dictation: the `context` part of the upload and the response.

Free-text fields are capped, the way api.py caps `query`: the formatter runs once per utterance on a small model, so an
unbounded body is a cost and latency hole.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator

from config import settings

DICTIONARY_MAX_TERMS = 100
DICTIONARY_TERM_MAX_CHARS = 60
MAX_CURSOR_CHARS = 150

Style = Literal["clean", "polished"]
_LANGUAGE_PATTERN = r"^[A-Za-z]{2,3}([-_][A-Za-z]{2,4})?$"


class AppContext(str, Enum):
    """The application surface the text is for. SWITCHED OFF for now (VOICE_APP_CONTEXTS=false): every value is treated as
    `general`, so clients may keep sending them. When switched on they select the rules below."""
    GENERAL = "general"             # prose: complete sentences, paragraph breaks only when the speaker asks
    EMAIL = "email"                 # a dictated greeting and sign-off go on their own lines
    SLACK = "slack_chat"            # casual wording, contractions kept, no forced final period on fragments
    CODE_COMMENT = "code_editor"    # identifiers kept as spoken, no Markdown, no final period on a one-line subject
    MARKDOWN_DOC = "markdown_docs"  # Markdown allowed: headings, inline code


def effective_context(context: "AppContext | str | None") -> AppContext:
    """What the formatter actually applies: `general` unless app-specific formats are switched on."""
    if not settings.voice_app_contexts_enabled:
        return AppContext.GENERAL
    return context if isinstance(context, AppContext) else AppContext(context or "general")


def _slice_preceding(value: Any) -> Any:
    return value[-MAX_CURSOR_CHARS:] if isinstance(value, str) else value


def _slice_following(value: Any) -> Any:
    return value[:MAX_CURSOR_CHARS] if isinstance(value, str) else value


class CursorContext(BaseModel):
    """The text on either side of the caret. Without it, spacing and capitalisation at the insertion point are guesses.
    A longer window is accepted and trimmed to the 150 characters nearest the caret."""
    preceding_text: Annotated[str, BeforeValidator(_slice_preceding)] = Field(
        default="", max_length=MAX_CURSOR_CHARS,
        description="Up to 150 chars immediately before the cursor (trimmed from the far end if longer)")
    following_text: Annotated[str, BeforeValidator(_slice_following)] = Field(
        default="", max_length=MAX_CURSOR_CHARS,
        description="Up to 150 chars immediately after the cursor (trimmed from the far end if longer)")


def _clean_dictionary(terms: List[str]) -> List[str]:
    seen, out = set(), []
    for term in terms:
        term = " ".join(term.split())[:DICTIONARY_TERM_MAX_CHARS]
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out[:DICTIONARY_MAX_TERMS]


class DictateContext(BaseModel):
    """The optional `context` part of a dictation upload (JSON). Every field may be left out."""
    app_context: AppContext = AppContext.GENERAL
    cursor_context: Optional[CursorContext] = None
    custom_dictionary: List[str] = Field(default_factory=list)
    style: Optional[Style] = Field(None, description="clean keeps the speaker's wording; polished tightens list points. None = server default")
    language: Optional[str] = Field(None, pattern=_LANGUAGE_PATTERN, description="Whisper language, e.g. en; None = server default")

    @field_validator("app_context")
    @classmethod
    def _context(cls, value: AppContext) -> AppContext:
        return effective_context(value)

    @field_validator("custom_dictionary")
    @classmethod
    def _dictionary(cls, value: List[str]) -> List[str]:
        return _clean_dictionary(value)


class Timings(BaseModel):
    audio_ms: float = 0.0       # length of the speech uploaded
    asr_ms: float = 0.0         # Whisper
    llm_ms: float = 0.0         # the model (0 when rules answered)
    post_ms: float = 0.0        # deterministic post-processing
    total_ms: float = 0.0       # server time: from the upload being received to the response being ready (the upload itself is not counted)


class DictateResponse(BaseModel):
    raw_text: str
    # Exactly what to insert at the caret. It already starts with a space when `leading_space_required` is true and ends
    # with one when `trailing_space_required` is true; the flags only say why.
    formatted_text: str
    leading_space_required: bool
    trailing_space_required: bool = False
    # llm = the model formatted it; fast = a plain sentence, formatted by rules without a model call (VOICE_LLM_MODE=smart);
    # rules = the model was down, too slow or rejected; see fallback_reason
    source: Literal["llm", "fast", "rules"] = "llm"
    fallback_reason: Optional[str] = None
    timings: Timings
