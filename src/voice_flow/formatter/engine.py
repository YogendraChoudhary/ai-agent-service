"""
The semantic compiler: a low-end model (Gemini by default, OpenAI by setting) turns a rough transcript into formatted text, with code around it.

  transcript -> gate (plain sentence: fast path, no model) -> [breaker?] -> model (timeout) -> guards -> text
                                                                  |              |             |
                                                                  +--------------+-------------+--> rule_format()   (the speaker's words are never lost)

The guards exist because a small model can answer a dictated question instead of formatting it, add a preamble, or cut
the text short. A rejected answer is replaced by the rule-based result, never shown.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple

from config import settings
from src.models import AUTOCOMPLETE_MODEL_IDS, ensure_allowed_model
from src.voice_flow.formatter.claims import CLEAN_KINDS, POLISHED_KINDS, missing_claims
from src.voice_flow.formatter.gate import needs_model
from src.voice_flow.formatter.patterns import GREETING_DETECT_RE
from src.voice_flow.formatter.prompts import build_system_prompt, build_turns, to_gemini_contents, to_openai_messages
from src.voice_flow.formatter.rules import (
    LIST_CUES, apply_dictionary, continue_sentence, detect_list, expected_list_items, list_lead_in, list_style, rule_format)
from src.voice_flow.schemas import AppContext, CursorContext, effective_context

logger = logging.getLogger(__name__)

PERMANENT_CODES = frozenset({401, 402, 403, 404})      # key, credit or model problems: retrying within a minute is pointless
PERMANENT_COOLDOWN_S = 60.0
TRANSIENT_COOLDOWN_S = 10.0
TRANSIENT_STRIKES = 3

Turns = List[Dict[str, str]]       # backend-neutral: {"role": "user" | "assistant", "content": "..."}


class LlmError(Exception):
    def __init__(self, reason: str, permanent: bool = False, is_content_block: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.permanent = permanent
        self.is_content_block = is_content_block


class LlmBackend(Protocol):
    name: str

    async def generate(self, system: str, turns: Turns, *, max_output_tokens: int, timeout: float) -> str: ...


@dataclass(frozen=True)
class CompileResult:
    formatted_text: str              # before PostProcessor: no edge spaces yet
    processing_time_ms: float
    source: Literal["llm", "fast", "rules"]
    fallback_reason: Optional[str] = None


_WARM_TURNS: Turns = [{"role": "user", "content": "ok"}]
_WARM_SYSTEM = "Reply with the single word: ok"


_MODEL_THINKING_VARIANTS: Dict[str, int] = {}


def default_thinking_variant_for_model(model_name: str) -> int:
    """Gemini 3 models use ThinkingLevel.MINIMAL (variant 0); Gemini 2 / 2.5 models use thinking_budget = 0 (variant 1)."""
    m = model_name.lower()
    if "gemini-2" in m:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
class GeminiBackend:
    """google-genai, async. One client for the process, so the TLS connection is reused between utterances."""
    name = "gemini"
    _model_thinking_variants: Dict[str, int] = _MODEL_THINKING_VARIANTS

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.voice_gemini_api_key or settings.default_gemini_api_key
        self.model = model or settings.voice_format_model or AUTOCOMPLETE_MODEL_IDS["gemini"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LlmError("no Gemini API key (VOICE_GEMINI_API_KEY / DEFAULT_GEMINI_API_KEY)", permanent=True)
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))      # a retry inside the SDK would hide a slow call
        return self._client

    def _thinking_config(self) -> Tuple[Any, int, int]:
        from google.genai import types
        variants = (lambda: types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                    lambda: types.ThinkingConfig(thinking_budget=0),
                    lambda: None)
        variant_idx = self._model_thinking_variants.get(self.model, default_thinking_variant_for_model(self.model))
        return variants[min(variant_idx, len(variants) - 1)](), len(variants), variant_idx

    async def generate(self, system: str, turns: Turns, *, max_output_tokens: int, timeout: float) -> str:
        from google.genai import errors, types
        client = self._get_client()
        contents = to_gemini_contents(turns)
        safety = [types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT, types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)]
        deadline = time.monotonic() + timeout
        while True:
            thinking, total_variants, current_variant = self._thinking_config()
            config = types.GenerateContentConfig(system_instruction=system, temperature=0.0, candidate_count=1,
                                                 max_output_tokens=max_output_tokens, safety_settings=safety,
                                                 thinking_config=thinking)
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(model=self.model, contents=contents, config=config),
                    timeout=max(0.2, deadline - time.monotonic()))
            except TimeoutError:
                raise LlmError(f"timeout after {timeout:.1f}s") from None
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code == 400 and "thinking" in str(exc).lower() and current_variant < total_variants - 1:
                    type(self)._model_thinking_variants[self.model] = current_variant + 1
                    continue
                raise LlmError(f"gemini {code} {getattr(exc, 'status', '')}".strip(), permanent=code in PERMANENT_CODES) from None
            type(self)._model_thinking_variants[self.model] = current_variant
            break

        if not response.candidates:
            block = getattr(response.prompt_feedback, "block_reason", None) or "no candidates"
            raise LlmError(f"blocked ({block})", is_content_block=True)
        candidate = response.candidates[0]
        finish = str(getattr(candidate, "finish_reason", ""))
        if any(term in finish.upper() for term in ("SAFETY", "RECITATION", "BLOCK")):
            raise LlmError(f"blocked ({finish})", is_content_block=True)
        if "MAX_TOKENS" in finish:
            raise LlmError("truncated at max_output_tokens")
        text = (response.text or "").strip()
        if not text:
            raise LlmError(f"empty response ({finish})", is_content_block="SAFETY" in finish.upper())
        return text

    async def warm(self) -> None:
        """One real, tiny generation: opens the connection, settles the thinking-config spelling, and proves the key has
        credit (a metadata call such as models.get passes even when generation is refused). Raises LlmError on failure."""
        await self.generate(_WARM_SYSTEM, _WARM_TURNS, max_output_tokens=64, timeout=8.0)


# ---------------------------------------------------------------------------
# OpenAI (chat completions). Same prompt, same guards; chosen with VOICE_FORMAT_PROVIDER=openai.
# ---------------------------------------------------------------------------
class OpenAIBackend:
    name = "openai"
    # Model names are not stable (gpt-4o, gpt-4.1, gpt-5 ...), so a prefix list is only the first guess: a model that rejects
    # `temperature` is learned from the 400 at run time and remembered for the life of the process.
    _KNOWN_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
    _no_temperature_models: Set[str] = set()

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, *, base_url: Optional[str] = None):
        self.api_key = api_key or settings.default_openai_api_key
        self.model = model or settings.voice_format_model or AUTOCOMPLETE_MODEL_IDS["openai"]
        self.base_url = base_url
        self._client = None
        self._effort_variant = 0         # reasoning models: ask for "minimal" reasoning, fall back to the default on a 400

    @property
    def _reasons(self) -> bool:
        return self.model.startswith(self._KNOWN_REASONING_PREFIXES)           # these spend hidden tokens and reject temperature

    def _supports_temperature(self) -> bool:
        return self.model not in self._no_temperature_models and not self._reasons

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LlmError("no OpenAI API key (DEFAULT_OPENAI_API_KEY)", permanent=True)
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, max_retries=0, base_url=self.base_url)
        return self._client

    async def generate(self, system: str, turns: Turns, *, max_output_tokens: int, timeout: float) -> str:
        import openai
        client = self._get_client()
        messages = to_openai_messages(system, turns)
        deadline = time.monotonic() + timeout
        while True:
            token_param = "max_tokens" if self.base_url else "max_completion_tokens"
            params: Dict[str, Any] = {"model": self.model, "messages": messages, token_param: max_output_tokens}
            if self._reasons and self._effort_variant == 0:
                params["reasoning_effort"] = "minimal"
            if self._supports_temperature():
                params["temperature"] = 0
            try:
                response = await asyncio.wait_for(client.chat.completions.create(**params),
                                                  timeout=max(0.2, deadline - time.monotonic()))
            except TimeoutError:
                raise LlmError(f"timeout after {timeout:.1f}s") from None
            except openai.APIStatusError as exc:
                code, detail = exc.status_code, str(getattr(exc, "body", "") or exc).lower()
                if code == 400 and "temperature" in params and "temperature" in detail and (
                        "unsupported" in detail or "not supported" in detail):
                    type(self)._no_temperature_models.add(self.model)         # learned once, remembered
                    continue
                if code == 400 and "reasoning" in detail and "reasoning_effort" in params:
                    self._effort_variant = 1          # this model does not take "minimal"; remembered
                    continue
                quota = code == 429 and "insufficient_quota" in detail
                raise LlmError(f"{self.name} {code}", permanent=code in PERMANENT_CODES or quota) from None
            except openai.APIConnectionError:
                raise LlmError(f"{self.name} connection error") from None
            break

        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise LlmError("content filter", is_content_block=True)
        if choice.finish_reason == "length":
            raise LlmError("truncated at max_output_tokens")
        text = (choice.message.content or "").strip()
        if not text:
            raise LlmError(f"empty response ({choice.finish_reason})")
        return text

    async def warm(self) -> None:
        await self.generate(_WARM_SYSTEM, _WARM_TURNS, max_output_tokens=64, timeout=8.0)


def build_backend() -> "LlmBackend":
    return OpenAIBackend() if settings.voice_format_provider == "openai" else GeminiBackend()


class ClaudeBackend:
    name = "claude"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model
        self._client = None

    async def generate(self, system: str, turns: Turns, *, max_output_tokens: int, timeout: float) -> str:
        import anthropic
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0)
        try:
            response = await self._client.messages.create(
                model=self.model, system=system, messages=turns, max_tokens=max_output_tokens, temperature=0, timeout=timeout)
        except anthropic.APIStatusError as exc:
            raise LlmError(f"claude {exc.status_code}", permanent=exc.status_code in PERMANENT_CODES) from None
        except anthropic.APIConnectionError:
            raise LlmError("claude connection error") from None
        if response.stop_reason == "max_tokens":
            raise LlmError("truncated at max_output_tokens")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LlmError("empty response")
        return text


def build_backend_for(provider: str, api_key: str) -> "LlmBackend":
    """BYOK always uses the provider's light model and an explicit personal key."""
    if not api_key or not api_key.strip():
        raise ValueError("A personal API key is required")
    model = ensure_allowed_model(AUTOCOMPLETE_MODEL_IDS[provider])
    if provider == "gemini":
        return GeminiBackend(api_key=api_key, model=model)
    if provider == "claude":
        return ClaudeBackend(api_key, model)
    from src.text_processing_agent import OPENAI_COMPATIBLE_BASES
    backend = OpenAIBackend(api_key=api_key, model=model, base_url=OPENAI_COMPATIBLE_BASES.get(provider))
    backend.name = provider
    return backend


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class Breaker:
    """After a key/credit/model error, skip the model for a minute; after repeated timeouts, for ten seconds."""

    def __init__(self, backend_name: str = "model") -> None:
        self.backend_name = backend_name
        self._open_until = 0.0
        self._strikes = 0
        self.last_reason = ""

    def allow(self) -> bool:
        return time.monotonic() >= self._open_until

    def success(self) -> None:
        self._strikes = 0

    def failure(self, error: LlmError) -> None:
        self.last_reason = error.reason
        if error.is_content_block:
            # Content safety or prompt blocks are utterance-specific, not an infrastructure/connectivity failure.
            # Do not register a strike or pause the model.
            return
        if error.permanent:
            self._open_until = time.monotonic() + PERMANENT_COOLDOWN_S
            logger.warning("voice_flow: %s backend disabled for %.0fs (%s)", self.backend_name, PERMANENT_COOLDOWN_S, error.reason)
            return
        self._strikes += 1
        if self._strikes >= TRANSIENT_STRIKES:
            self._open_until = time.monotonic() + TRANSIENT_COOLDOWN_S
            self._strikes = 0
            logger.warning("voice_flow: %s backend paused for %.0fs after repeated failures (%s)",
                           self.backend_name, TRANSIENT_COOLDOWN_S, error.reason)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9']+")
_ALPHA = re.compile(r"[A-Za-z]{3,}")
_LIST_LINE = re.compile(r"(?m)^\s*(?:\d{1,3}[.)]|[-*•])\s+\S")
# What a chat model says when it answers instead of formatting. Not a problem when the speaker dictated those words.
_CHATBOT = re.compile(r"\A\s*(?:sure\b|certainly\b|of course\b|here(?:'s|\s+is|\s+are)\b|i\s+can(?:not|'t)\b|as\s+an\s+ai\b|okay,?\s+here\b|the formatted)",
                      re.IGNORECASE)
NEAR_MATCH = 0.72              # a homophone or contraction is a near-match of the word it fixes: their / they're, its / it's
MIN_NOVEL_WORDS = 5            # fewer new words than this can be a homophone chain, not an answer
NOVEL_RATIO = 0.45
MIN_WORDS_FOR_NOVELTY_CHECK = 8


def _near_any(word: str, pool: Sequence[str]) -> bool:
    for known in pool:
        if abs(len(known) - len(word)) > 3:
            continue
        matcher = SequenceMatcher(None, word, known)
        if matcher.real_quick_ratio() >= NEAR_MATCH and matcher.quick_ratio() >= NEAR_MATCH and matcher.ratio() >= NEAR_MATCH:
            return True
    return False


def check_output(raw: str, out: str, dictionary: Sequence[str] = ()) -> Optional[str]:
    """None when `out` looks like a formatting of `raw`; otherwise the reason it does not."""
    if not out.strip():
        return "empty_output"
    raw_words, out_words = _WORD.findall(raw), _WORD.findall(out)
    if _CHATBOT.match(out) and not _CHATBOT.match(raw):
        return "answered_instead_of_formatting"
    if len(out_words) > len(raw_words) * 1.6 + 6:
        return "too_long"
    if len(raw_words) >= 10 and len(out_words) < len(raw_words) * 0.4:
        return "dropped_content"
    # The speech counts through items but the answer is not a list (Gemini does this when text precedes the caret) - or a
    # list appears where nothing asked for one.
    if detect_list(raw) and not _LIST_LINE.search(out):
        return "missing_list"
    expected = expected_list_items(raw)
    if expected >= 2 and len(_LIST_LINE.findall(out)) != expected:
        return "list_count_mismatch"
    if _LIST_LINE.search(out) and not LIST_CUES.search(raw):
        return "unrequested_list"

    out_alpha = [w.lower() for w in _ALPHA.findall(out)]
    if len(out_alpha) > MIN_WORDS_FOR_NOVELTY_CHECK:           # a short fix-up ("their going" -> "they're going") is never an answer
        known = {w.lower() for w in _ALPHA.findall(raw)} | {w.lower() for term in dictionary for w in _ALPHA.findall(term)}
        pool = sorted(known)
        novel = [w for w in out_alpha if w not in known and not _near_any(w, pool)]
        # Fixing homophones adds a few words; answering a question adds most of the sentence.
        if len(novel) >= MIN_NOVEL_WORDS and len(novel) / len(out_alpha) > NOVEL_RATIO:
            return "answered_instead_of_formatting"
    return None


_CUE_PHRASE = re.compile(r"\s*\b(?:make|mark)\s+(?:it|this|that|these)\s+(?:as\s+)?(?:a\s+)?(?:numbered\s+list|bullet\s+(?:list|points?)|bullets?|list|points?)\b[.:,]?", re.IGNORECASE)


def strip_spoken_cues(raw: str, out: str) -> str:
    """"Make it bullets." is a command, not text. Gemini sometimes keeps it in the sentence above the list."""
    if not _CUE_PHRASE.search(raw) or not _LIST_LINE.search(out):
        return out
    lines = out.split("\n")
    first = next(i for i, line in enumerate(lines) if _LIST_LINE.match(line))
    for i in range(first):
        lines[i] = _CUE_PHRASE.sub("", lines[i]).rstrip()
    return "\n".join(lines)


_ITEM_MARK = re.compile(r"^(\s*)(?:\d{1,3}[.)]|[-*•])\s+(.*)$")
_ORDINAL_ITEM = re.compile(r"^(\s*(?:\d{1,3}[.)]|[-*•])\s+)(?:first|second|third|fourth|fifth|firstly|secondly|thirdly|next|finally|lastly)\b[,:]?\s+(\S)", re.IGNORECASE)


def normalize_list(raw: str, out: str) -> str:
    """Three slips the model makes on lists: bullets where the speaker counted ("point 1, point 2"), the counting words
    ("First review Q3") left at the start of items, and a lead-in nobody said ("Here are the tasks:"). The speech is the authority."""
    lines = out.split("\n")
    items = [i for i, line in enumerate(lines) if _ITEM_MARK.match(line)]
    if len(items) < 2:
        return out
    lead = list_lead_in(raw)
    if lead is not None and not lead.strip() and items[0] > 0 and any(l.strip() for l in lines[:items[0]]):
        lines = lines[items[0]:]                       # nothing was said before the first cue: "Here are the tasks:" is invented
        items = [i for i, line in enumerate(lines) if _ITEM_MARK.match(line)]
    if sum(bool(_ORDINAL_ITEM.match(lines[i])) for i in items) >= 2:
        for i in items:
            lines[i] = _ORDINAL_ITEM.sub(lambda m: m.group(1) + m.group(2).upper(), lines[i])
    style = list_style(raw)
    if style:
        for n, i in enumerate(items, 1):
            m = _ITEM_MARK.match(lines[i])
            lines[i] = f"{m.group(1)}{n}. {m.group(2)}" if style == "number" else f"{m.group(1)}- {m.group(2)}"
    return "\n".join(lines)


def strip_context_echo(raw: str, out: str, cursor: Optional[CursorContext]) -> str:
    """Small models sometimes re-type the text before the caret ("We need to focus on the onboarding flow" after
    "We need to focus on"). Drop that echo, unless the speaker's own words begin with it (then it was dictated)."""
    if not cursor:
        return out
    lines = [l.strip() for l in cursor.preceding_text.split("\n") if l.strip()]
    words = lines[-1].split() if lines else []
    low_out, low_raw = out.lower(), " ".join(raw.lower().split())
    for n in range(min(len(words), 10), 0, -1):
        phrase = " ".join(words[-n:]).lower()
        if n == 1 and len(words) > 1:
            break                                    # one word of a longer line is too likely to be dictated again
        if low_out.startswith(phrase) and not low_raw.startswith(phrase.rstrip(",.:;")):
            rest = out[len(phrase):].lstrip(" \t\n,;:")
            if rest:
                out = rest
            break
    head = cursor.following_text.strip().split()[:8]
    for n in range(len(head), 1, -1):
        phrase = " ".join(head[:n]).lower()
        if out.lower().rstrip(" .").endswith(phrase) and not low_raw.rstrip(" .").endswith(phrase):
            trimmed = out.rstrip(" .")[: -len(phrase)].rstrip()
            if trimmed:
                out = trimmed
            break
    return out


def _keeps_opening_greeting(raw: str, out: str) -> bool:
    """A dictation that opens "Hey, hi there, good morning" still opens that way: a small model sometimes drops it as filler."""
    if not GREETING_DETECT_RE.search(raw):
        return True
    wanted = set(re.findall(r"[a-z']+", raw.lower())[:2])
    return bool(wanted & set(re.findall(r"[a-z']+", out.lower())[:12]))


def _max_output_tokens(raw: str) -> int:
    # Hidden "thinking" tokens, if the model spends any, come out of this too, so there is a floor under short utterances.
    return min(4096, max(512, int(len(raw.split()) * 3.0) + 300))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class VoiceFormattingService:
    def __init__(self, backend: Optional[LlmBackend] = None, timeout: Optional[float] = None,
                 breaker: Optional[Breaker] = None, mode: Optional[str] = None):
        self.backend: LlmBackend = backend or build_backend()
        self.mode = mode or settings.voice_llm_mode
        self.timeout = timeout or settings.voice_format_timeout_seconds
        self.breaker = breaker or Breaker(getattr(self.backend, "name", "model"))

    async def _ask(self, raw: str, context: AppContext, cursor: Optional[CursorContext], dictionary: Sequence[str],
                   style: str, timeout: float, retry: Optional[Tuple[str, str]] = None) -> str:
        """One model call, with the clean-ups for the slips small models make, ready for the guards.
        `retry` = (the answer that was wrong, what was wrong with it): asked again with that correction."""
        turns = build_turns(raw, context, cursor, dictionary, style)
        if retry:
            turns += [{"role": "assistant", "content": retry[0]}, {"role": "user", "content": retry[1]}]
        out = await self.backend.generate(build_system_prompt(style), turns, max_output_tokens=_max_output_tokens(raw), timeout=timeout)
        out = normalize_list(raw, strip_spoken_cues(raw, strip_context_echo(raw, out, cursor)))
        return continue_sentence(out, cursor)          # mid-sentence caret: "the Redis pool", not "The Redis pool", whoever wrote it

    async def _ask_counted(self, raw: str, context: AppContext, cursor: Optional[CursorContext], dictionary: Sequence[str],
                           style: str, timeout: float) -> str:
        """_ask, and when the speaker counted N points but the answer has another number of items (a small model leaves point 1
        in the introduction, or invents an item), ask once more with that exact correction."""
        out = await self._ask(raw, context, cursor, dictionary, style, timeout)
        expected = expected_list_items(raw)
        got = len(_LIST_LINE.findall(out))
        if expected >= 2 and got != expected:
            logger.info("voice_flow: answer has %d list items, the speaker counted %d; asking again", got, expected)
            note = (f"Your list has {got} items but the speaker counted {expected} points. Write the whole text again with exactly "
                    f"{expected} numbered items, starting at 1, the point with the first cue being item 1. Output only the text.")
            out = await self._ask(raw, context, cursor, dictionary, style, timeout, retry=(out, note))
        return out

    async def _compose(self, raw: str, context: AppContext, cursor: Optional[CursorContext], dictionary: Sequence[str],
                       style: str, timeout: float) -> Tuple[str, Optional[str], Optional[str]]:
        """The best answer the styles can give: (text, what the guards still object to or None, a note for the result).

        Clean is one call. Polished tightens the writing, so it can also lose something the speaker said; then it gets ONE repair
        that names exactly what was lost, and only if that fails too is the plain Clean text used. Never straight to the rules,
        which would throw away what the model understood."""
        kinds = POLISHED_KINDS if style == "polished" else CLEAN_KINDS

        def verdict(text: str) -> Tuple[Optional[str], List[str]]:
            problem = check_output(raw, text, dictionary)
            if problem:
                return problem, []
            lost = missing_claims(raw, text, kinds)
            if not _keeps_opening_greeting(raw, text):
                lost.append("the opening greeting")
            return None, lost

        def repair_note(lost: List[str]) -> str:
            keep = "hyphenated term and number" if style == "polished" else "acronym and hyphenated term"
            return (f"Your answer dropped {', '.join(repr(t) for t in lost)} from what the speaker said. Write the text again keeping every "
                    f"{keep}; keep the same structure. Output only the text.")

        if style != "polished":
            out = await self._ask_counted(raw, context, cursor, dictionary, "clean", timeout)
            problem, lost = verdict(out)
            if problem or not lost:
                return out, problem, None
            fixed = await self._ask(raw, context, cursor, dictionary, "clean", timeout, retry=(out, repair_note(lost)))
            problem2, lost2 = verdict(fixed)
            if problem2:
                return out, None, f"kept despite dropping {', '.join(repr(t) for t in lost)}"     # the repair made it worse: keep the first
            return fixed, None, (f"repaired (it had dropped {', '.join(repr(t) for t in lost)})" if not lost2 else
                                 f"kept despite dropping {', '.join(repr(t) for t in lost2)}")

        out = await self._ask_counted(raw, context, cursor, dictionary, "polished", timeout)
        problem, lost = verdict(out)
        if not problem and not lost:
            return out, None, None
        why = problem
        if lost:
            names = ", ".join(repr(t) for t in lost)
            fixed = await self._ask(raw, context, cursor, dictionary, "polished", timeout, retry=(out, repair_note(lost)))
            problem2, lost2 = verdict(fixed)
            if not problem2 and not lost2:
                return fixed, None, f"polished repaired (it had dropped {names})"
            why = problem2 or f"dropped {lost2[0]!r}"
        logger.info("voice_flow: polished output refused (%s); using the clean style", why)
        out = await self._ask_counted(raw, context, cursor, dictionary, "clean", timeout)
        return out, check_output(raw, out, dictionary), f"polished refused ({why}); used clean"

    async def compile(self, raw_text: str, app_context: AppContext = AppContext.GENERAL,
                      cursor_context: Optional[CursorContext] = None, custom_dictionary: Sequence[str] = (),
                      style: Optional[str] = None) -> CompileResult:
        t0 = time.perf_counter()
        raw = " ".join(raw_text.split())
        context = effective_context(app_context)
        style = style or settings.voice_default_style

        def done(text: str, source: Literal["llm", "fast", "rules"], reason: Optional[str] = None) -> CompileResult:
            return CompileResult(apply_dictionary(text, custom_dictionary), (time.perf_counter() - t0) * 1000.0, source, reason)

        def rules(reason: str) -> CompileResult:
            return done(rule_format(raw, cursor_context, custom_dictionary), "rules", reason)

        if not raw:
            return done("", "rules", "empty transcript")
        # A 255-word dictation takes Gemini ~1.8 s; a 1,000-word one needs longer than the 4 s that suits a sentence.
        timeout = min(40.0, max(self.timeout, 3.0 + 0.02 * len(raw.split())))
        if self.mode == "smart" and needs_model(raw, context, cursor_context) is None:
            return done(rule_format(raw, cursor_context, custom_dictionary), "fast")
        if not self.breaker.allow():
            return rules(f"model paused ({self.breaker.last_reason})")
        try:
            out, problem, reason = await self._compose(raw, context, cursor_context, custom_dictionary, style, timeout)
        except LlmError as exc:
            self.breaker.failure(exc)
            return rules(exc.reason)
        except Exception as exc:                                        # noqa: BLE001 - a formatter must never take dictation down
            logger.exception("voice_flow: unexpected formatter error")
            self.breaker.failure(LlmError(type(exc).__name__))
            return rules(f"unexpected {type(exc).__name__}")

        if problem:
            logger.warning("voice_flow: model output rejected (%s)", problem)
            return rules(f"guard: {problem}")
        self.breaker.success()
        return done(out, "llm", reason)
