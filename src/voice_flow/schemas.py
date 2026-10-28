"""
Data contracts for voice dictation: the REST bodies and the WebSocket frames.

Free-text fields are capped, the way api.py caps `query`: the formatter runs once per utterance on a small model, so an
unbounded body is a cost and latency hole.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator

from config import settings

RAW_TEXT_MAX_CHARS = 8000          # ~1,400 words: far beyond one dictated utterance
DICTIONARY_MAX_TERMS = 100
DICTIONARY_TERM_MAX_CHARS = 60
AUDIO_B64_MAX_CHARS = 14_000_000   # ~10 MB of audio
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


class _DictationContext(BaseModel):
    """Fields every dictation request carries."""
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


class FormatRequest(_DictationContext):
    raw_text: str = Field(max_length=RAW_TEXT_MAX_CHARS)


class FormatResponse(BaseModel):
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
    asr_latency_ms: float = 0.0
    llm_latency_ms: float
    total_latency_ms: float


class DictateRequest(_DictationContext):
    """Audio in, formatted text out. For clients that record a whole utterance (and for scripted tests)."""
    audio_b64: str = Field(max_length=AUDIO_B64_MAX_CHARS, description="Any container PyAV can decode (wav, webm, m4a, ...)")


class DictateResponse(FormatResponse):
    audio_ms: float = 0.0


# ---------------------------------------------------------------------------
# WebSocket frames. Client -> server: one JSON "start", then binary PCM16 frames, then JSON "end" (or "cancel").
# ---------------------------------------------------------------------------
class StartFrame(_DictationContext):
    type: Literal["start"] = "start"
    sample_rate: int = Field(16000, ge=8000, le=48000)
    # push_to_talk: the utterance ends when the client sends "end". auto: the server also ends it after a pause.
    mode: Literal["push_to_talk", "auto"] = "push_to_talk"
    token: str = ""
    # auto mode: how long a silence ends the dictation. None = the server setting, 0 = never (the client sends "end").
    silence_ms: Optional[int] = Field(None, ge=0, le=60000)

    @field_validator("silence_ms")
    @classmethod
    def _silence(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and 0 < value < 500:
            raise ValueError("silence_ms must be 0 or at least 500")
        return value


class ContextUpdateFrame(BaseModel):
    """Send only what changed, e.g. the cursor moved: fields left out keep their value (the dictionary is not resent)."""
    type: Literal["context"] = "context"
    app_context: Optional[AppContext] = None
    cursor_context: Optional[CursorContext] = None
    custom_dictionary: Optional[List[str]] = None
    style: Optional[Style] = None
    language: Optional[str] = Field(None, pattern=_LANGUAGE_PATTERN)

    @field_validator("app_context")
    @classmethod
    def _context(cls, value: Optional[AppContext]) -> Optional[AppContext]:
        return None if value is None else effective_context(value)

    @field_validator("custom_dictionary")
    @classmethod
    def _dictionary(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return None if value is None else _clean_dictionary(value)


class Timings(BaseModel):
    audio_ms: float = 0.0       # length of the speech sent
    asr_ms: float = 0.0         # Whisper
    llm_ms: float = 0.0         # the model (0 when rules answered)
    post_ms: float = 0.0        # deterministic post-processing
    total_ms: float = 0.0       # from "end" (or the pause) to the final frame


class FinalFrame(BaseModel):
    type: Literal["final"] = "final"
    raw_text: str
    formatted_text: str
    leading_space_required: bool
    trailing_space_required: bool = False
    source: Literal["llm", "fast", "rules"] = "llm"
    fallback_reason: Optional[str] = None
    timings: Timings


def error_frame(message: str, code: str = "error") -> Dict[str, str]:
    return {"type": "error", "code": code, "message": message}
