"""
The model call every long-form stage goes through (the blog pipeline in src/blog, the refinement pipeline in this package).

`LLM` is injectable, like the email service's: tests and other callers never need a provider key. `AgnoLLM` is the
real one: an agno agent per call (non-streaming unless the call asks to be streamed, because every part is checked before
anyone sees it), the provider candidates and fallback rules of the main streaming loop, usage recorded per call.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallSpec:
    """How one call should run. The stage and unit let a scripted test LLM answer per call, in any order."""
    stage: str                      # blog: outline | outline_repair | unit | unit_repair; refinement: chunk | chunk_repair
    temperature: float
    max_tokens: int                 # VISIBLE output tokens wanted; get_model adds headroom for reasoning models
    use_web: bool = False
    unit: Optional[str] = None      # blog: intro | s1.. | conclusion; refinement: c1, c2, ...
    # Streams the call: receives everything written so far after each chunk (a failed provider restarts it from nothing on
    # the next candidate). The reply returned is the same either way; a listener that raises never fails the call.
    on_text: Optional[Callable[[str], None]] = field(default=None, compare=False, repr=False)


# (system prompt, messages, spec) -> reply text. messages alternate user/assistant and end with a user turn.
LLM = Callable[[str, List[dict], CallSpec], str]


def call_with_timeout(fn: Callable[[], object], seconds: Optional[float], what: str = "The model call"):
    """
    `fn()`, but a call that has not answered after `seconds` raises TimeoutError("... timed out ...") instead of holding the request
    for ever (seen live: a Gemini call that stayed open and silent for minutes). That error is one the fallback rules retry on the next
    provider. The stuck call cannot be interrupted: its daemon thread ends when the socket does. None or 0 waits as long as it takes.
    """
    if not seconds:
        return fn()
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as error:               # handed to the caller below, with its traceback
            outcome["error"] = error

    worker = threading.Thread(target=run, name="llm-call", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"{what} timed out after {seconds:g}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def model_id_for(candidate, use_web: bool = False) -> str:
    """The model id a candidate runs: its tier's id, and DeepSeek's chat model when tools are attached."""
    from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS, DEEPSEEK_TOOL_MODEL_ID

    if candidate.provider == "deepseek" and use_web:
        return DEEPSEEK_TOOL_MODEL_ID
    return (ADVANCED_MODEL_IDS if candidate.advanced else BASIC_MODEL_IDS)[candidate.provider]


class AgnoLLM:
    """
    candidates: the providers to draft with, in order (task_router.Candidate); key_for(provider) resolves keys lazily.
    outline_candidates: optional chain for the FIRST outline attempt (a faster model, then the drafting providers as
    fallbacks). The outline repair and every part always use `candidates`: a cheap first pass costs time when it is
    wrong, never quality.

    Fallback follows process_user_query_stream: a rate limit is retried once on the same provider; a retryable or
    dead-key error moves to the next candidate; bad requests and safety blocks fail at once (they would fail
    everywhere). Parts run in parallel, so a model that failed is remembered (across both chains): later calls skip it
    instead of every part paying for the same failure. The last candidate of a chain is always tried.
    """

    def __init__(self, candidates: Sequence, key_for: Callable[[str], Optional[str]], *,
                 outline_candidates: Optional[Sequence] = None, request_id: Optional[str] = None,
                 user_id: Optional[str] = None, is_byok: bool = False, image_ctx: Optional[dict] = None,
                 agent_type: str = "blog_writer", timeout: Optional[float] = None):
        self.agent_type = agent_type                         # the AGENT_CONFIGS entry every call is made with
        self.timeout = timeout                               # seconds one call may take before it counts as a retryable failure
        self.candidates = list(candidates)
        self.outline_candidates = list(outline_candidates) if outline_candidates else None
        self.key_for = key_for
        self.request_id = request_id
        self.user_id = user_id
        self.is_byok = is_byok
        self.image_ctx = image_ctx
        self.used: Optional[Tuple[str, str]] = None          # (provider, model id) of the latest successful part
        self.outline_used: Optional[Tuple[str, str]] = None
        self.fallbacks: List[dict] = []
        self.calls: List[dict] = []                          # every model call: where the time of a post went
        self._failed: Set[Tuple[str, bool]] = set()          # (provider, advanced) that could not serve this request
        self._lock = threading.Lock()

    def drafting_model(self) -> Optional[Tuple[str, str]]:
        """(provider, model id) the parts will be drafted by, as far as is known before they run."""
        with self._lock:
            live = [c for c in self.candidates if (c.provider, c.advanced) not in self._failed]
        pick = (live or self.candidates or [None])[0]
        return (pick.provider, model_id_for(pick)) if pick is not None else None

    def _chain(self, spec: CallSpec) -> list:
        return self.outline_candidates if spec.stage == "outline" and self.outline_candidates else self.candidates

    def __call__(self, system: str, messages: List[dict], spec: CallSpec) -> str:
        from src.observability import trace_set
        from src.provider_errors import can_fall_back, is_provider_unusable, is_rate_limit, retry_delay, status_code
        from config import settings

        log = f"[{self.request_id}] " if self.request_id else ""
        chain = self._chain(spec)
        for index, candidate in enumerate(chain):
            last = index + 1 == len(chain)
            with self._lock:
                if not last and (candidate.provider, candidate.advanced) in self._failed:
                    continue
            key = self.key_for(candidate.provider)
            if not key:
                if not last:
                    continue
                raise ValueError(f"No API key configured for {candidate.provider}")
            retry = 0
            while True:
                started = time.perf_counter()
                try:
                    reply = call_with_timeout(lambda: self._run(candidate, key, system, messages, spec), self.timeout,
                                              f"The {candidate.provider} call")
                except Exception as error:
                    self._record(spec, candidate, started, error)
                    if not settings.provider_fallback_enabled or not can_fall_back(error):
                        raise
                    if is_provider_unusable(error):
                        logger.error("%s%s key or model unusable (%s); falling back", log, candidate.provider,
                                     status_code(error) or type(error).__name__)
                    if is_rate_limit(error) and retry == 0:
                        retry += 1
                        delay = retry_delay(error)
                        logger.warning("%sRate limit: retrying %s once after %.2fs", log, candidate.provider, delay)
                        time.sleep(delay)
                        continue
                    if last:
                        raise
                    reason = str(status_code(error) or type(error).__name__)
                    with self._lock:
                        self._failed.add((candidate.provider, candidate.advanced))
                        self.fallbacks.append({"from": candidate.provider, "to": chain[index + 1].provider,
                                               "error": reason})
                        trace_set(self.request_id, fallbacks=list(self.fallbacks))
                    logger.warning("%sProvider fallback %s -> %s (%s)", log, candidate.provider,
                                   chain[index + 1].provider, reason)
                    break
                else:
                    self._record(spec, candidate, started)       # outside the try: logging can never read as a provider error
                    return reply
        raise RuntimeError("No provider could serve the request")

    def _record(self, spec: CallSpec, candidate, started: float, error: Optional[Exception] = None) -> None:
        """One line per attempt in the trace (`llm_calls`): stage, part, model, seconds, and the error if it failed."""
        from src.observability import trace_set
        from src.provider_errors import status_code

        entry = {"stage": spec.stage, "unit": spec.unit, "model": f"{candidate.provider}/{model_id_for(candidate, spec.use_web)}",
                 "seconds": round(time.perf_counter() - started, 2)}
        if error is not None:
            entry["error"] = str(status_code(error) or type(error).__name__)
        with self._lock:
            self.calls.append(entry)
            trace_set(self.request_id, llm_calls=list(self.calls))

    def _run(self, candidate, key: str, system: str, messages: List[dict], spec: CallSpec) -> str:
        from agno.models.message import Message

        from src import text_processing_agent as tpa
        from src.email_template.service import is_failed_run
        from src.observability import record_text_usage, trace_set
        from src.request_plan import RequestPlan

        agent = tpa.create_agent(
            self.agent_type, candidate.provider, key, self.request_id, user_id=self.user_id, is_byok=self.is_byok,
            image_ctx=self.image_ctx, plan=RequestPlan(use_web=spec.use_web, advanced_model=candidate.advanced),
            instructions=system, temperature=spec.temperature, max_tokens=spec.max_tokens,
        )
        history = [Message(role=m["role"], content=m["content"]) for m in messages]
        if spec.on_text is None:
            response = agent.run(history, stream=False)
            # agno does not raise on provider failures: the run comes back failed with the error text as content.
            if is_failed_run(response):
                raise RuntimeError(tpa._agent_error_message(candidate.provider, getattr(response, "content", None)))
            content, metrics = response.content, getattr(response, "metrics", None)
        else:
            content, metrics = self._stream(tpa, agent, candidate, history, spec.on_text)
        model_id = model_id_for(candidate, spec.use_web)
        record_text_usage(self.request_id, model_id, metrics)
        served = (candidate.provider, model_id)
        with self._lock:
            if spec.stage == "outline":
                self.outline_used = served
                trace_set(self.request_id, outline_model=f"{served[0]}/{served[1]}")
            else:
                # Attribution (billing, latency groups) follows the model that WRITES the post, not a fast first pass.
                first_part = self.used is None
                self.used = served
                if first_part:
                    trace_set(self.request_id, model_used=f"{served[0]}/{served[1]}")
        return content if isinstance(content, str) else str(content or "")

    @staticmethod
    def _stream(tpa, agent, candidate, history, on_text: Callable[[str], None]) -> Tuple[str, object]:
        """The same call, streamed the way the chat path streams: (text, metrics). A failure raises as a failed run does."""
        from agno.agent import RunEvent

        text, metrics = "", None
        for event in agent.run(history, stream=True, stream_events=True):
            if getattr(event, "metrics", None) is not None:
                metrics = event.metrics
            kind = getattr(event, "event", None)
            if kind == RunEvent.run_error:
                raise RuntimeError(tpa._agent_error_message(candidate.provider, getattr(event, "content", None)))
            if kind == RunEvent.run_content and event.content:
                text += event.content if isinstance(event.content, str) else str(event.content)
                try:
                    on_text(text)
                except Exception:                       # a listener must never fail the call that feeds it
                    logger.warning("blog on_text listener failed", exc_info=True)
        return text, metrics
