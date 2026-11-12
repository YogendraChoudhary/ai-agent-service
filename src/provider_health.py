"""
Provider and model health: what answered, what failed, and how.

Why it exists: a dead Anthropic key hid behind silent provider fallback for weeks (the request succeeded on OpenAI, the only trace was one
ERROR line), and two Claude request-shape bugs (`temperature`, `content[0].text`) and a withdrawn image model would have been invisible the
same way. This keeps a rolling window of outcomes per provider/model in memory, turns the ones that mean "a rollout is broken" into alerts,
logs each alert loudly once per hour (`ROLLOUT_ALERT ...`, easy to alarm on), and serves the lot at GET /agent/health/providers.

No request or response text is stored, only counts, status classes and a short error snippet (never a key).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 15 * 60
ALERT_LOG_INTERVAL = 60 * 60
_MAX_EVENTS = 20_000

# kinds where the call produced no answer at all (truncated / refused / empty have their own alerts)
HARD_FAILURE_KINDS = ("key_rejected", "model_missing", "request_rejected", "rate_limited", "server_error")

_KEY = re.compile(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+|(AIza[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]+|(xai-[A-Za-z0-9]{4})[A-Za-z0-9]+")


def _scrub(text: str, limit: int = 160) -> str:
    return _KEY.sub(lambda m: (m.group(1) or m.group(2) or m.group(3)) + "...", str(text)).replace("\n", " ")[:limit]


def classify_error(error: BaseException) -> str:
    """What a provider error means for the rollout, from its status and text."""
    from .provider_errors import is_rate_limit, status_code
    status = status_code(error)
    text = (type(error).__name__ + " " + str(error)).lower()
    if any(s in text for s in ("model_not_found", "does not exist", "model not found", "no longer available", "decommissioned")) or status == 404:
        return "model_missing"
    if status in (401, 403) or any(s in text for s in ("invalid x-api-key", "invalid api key", "incorrect api key", "authentication", "permission denied")):
        return "key_rejected"
    if status == 402 or "credit balance" in text or "insufficient_quota" in text or "prepayment" in text:
        return "key_rejected"                        # the key works but cannot spend: the same fix, for the same people
    if is_rate_limit(error):
        return "rate_limited"
    if status in (400, 422) or any(s in text for s in ("unsupported parameter", "unsupported value", "not supported with this model",
                                                       "deprecated for this model", "invalid_request_error")):
        return "request_rejected"                    # our request shape is wrong for this model (a parameter it no longer takes, ...)
    return "server_error"


class ProviderHealth:
    def __init__(self, window: int = WINDOW_SECONDS):
        self.window = window
        self._events: Deque[Tuple[float, str, str, str, bool, str]] = deque(maxlen=_MAX_EVENTS)   # ts, provider, model, kind, byok, detail
        self._fallbacks: Deque[Tuple[float, str, str, str]] = deque(maxlen=2000)
        self._lock = threading.Lock()
        self._alerted: Dict[Tuple[str, str, str], float] = {}
        self.preflight: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ recording
    def record(self, provider: str, model: str, kind: str, *, byok: bool = False, detail: str = "") -> None:
        now = time.time()
        with self._lock:
            self._events.append((now, str(provider), str(model), kind, bool(byok), _scrub(detail) if detail else ""))
        if byok:
            return
        if kind in ("key_rejected", "model_missing"):
            self._log_alert(kind, provider, model, detail, now)         # structural: the first one is already a finding
        elif kind == "request_rejected":
            # a single 400 is often the user's own input (too long, bad image); three in the window mean our request shape is wrong
            with self._lock:
                recent = sum(1 for e in self._events if e[0] >= now - self.window and e[1] == provider and e[2] == model
                             and e[3] == "request_rejected" and not e[4])
            if recent >= 3:
                self._log_alert(kind, provider, model, detail, now)

    def record_error(self, provider: str, model: str, error: BaseException, *, byok: bool = False) -> str:
        kind = classify_error(error)
        self.record(provider, model, kind, byok=byok, detail=f"{type(error).__name__}: {error}")
        return kind

    def record_fallback(self, provider: str, to: str, reason: str) -> None:
        with self._lock:
            self._fallbacks.append((time.time(), provider, to, str(reason)))

    def _log_alert(self, kind: str, provider: str, model: str, detail: str, now: float) -> None:
        key = (kind, provider, model)
        with self._lock:
            if now - self._alerted.get(key, 0) < ALERT_LOG_INTERVAL:
                return
            self._alerted[key] = now
        logger.error("ROLLOUT_ALERT %s provider=%s model=%s %s", _ALERT_TEXT[kind], provider, model, _scrub(detail, 200))

    # ------------------------------------------------------------------ reading
    def _window(self, now: float) -> List[Tuple[float, str, str, str, bool, str]]:
        cutoff = now - self.window
        with self._lock:
            return [e for e in self._events if e[0] >= cutoff]

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        events = self._window(now)
        per: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "ok": 0, "failures": defaultdict(int), "last_error": {}})
        for _, provider, model, kind, byok, detail in events:
            row = per[f"{provider}/{model}"]
            row["calls"] += 1
            if kind == "ok":
                row["ok"] += 1
            else:
                row["failures"][kind] += 1
                if detail and not byok:
                    row["last_error"][kind] = detail
        models = {name: {"calls": r["calls"], "ok": r["ok"], "failures": dict(r["failures"]), "last_error": r["last_error"]} for name, r in sorted(per.items())}
        with self._lock:
            fallbacks = [f for f in self._fallbacks if f[0] >= now - self.window]
        pairs: Dict[str, int] = defaultdict(int)
        for _, a, b, reason in fallbacks:
            pairs[f"{a}->{b} ({reason})"] += 1
        return {"window_seconds": self.window, "models": models, "fallbacks": dict(pairs), "alerts": self.alerts(events),
                "preflight": self.preflight}

    def alerts(self, events=None) -> List[Dict[str, Any]]:
        events = self._window(time.time()) if events is None else events
        platform = [e for e in events if not e[4]]                 # a user's own (BYOK) key failing is their problem, not a rollout alert
        out: List[Dict[str, Any]] = []
        by_model: Dict[Tuple[str, str], List[Tuple[float, str, str, str, bool, str]]] = defaultdict(list)
        for e in platform:
            by_model[(e[1], e[2])].append(e)
        for (provider, model), rows in sorted(by_model.items()):
            kinds: Dict[str, int] = defaultdict(int)
            for r in rows:
                kinds[r[3]] += 1
            calls = len(rows)

            def alert(code: str, severity: str, count: int) -> None:
                sample = next((r[5] for r in reversed(rows) if r[3] == code and r[5]), "")
                out.append({"code": code, "severity": severity, "provider": provider, "model": model, "count": count, "calls": calls,
                            "what": _ALERT_TEXT.get(code, code), "sample": sample})

            if kinds["key_rejected"]:
                alert("key_rejected", "critical", kinds["key_rejected"])
            if kinds["model_missing"]:
                alert("model_missing", "critical", kinds["model_missing"])
            if kinds["request_rejected"] >= 3 or (calls >= 10 and kinds["request_rejected"] / calls > 0.02):
                alert("request_rejected", "critical", kinds["request_rejected"])
            if calls >= 20 and kinds["truncated"] / calls > 0.05:
                alert("truncated", "warning", kinds["truncated"])
            if kinds["refused"] >= 3:
                alert("refused", "warning", kinds["refused"])
            if calls >= 20 and kinds["empty"] / calls > 0.05:
                alert("empty", "warning", kinds["empty"])
            failed = sum(kinds[k] for k in HARD_FAILURE_KINDS)
            if calls >= 10 and failed / calls > 0.2 and not {"key_rejected", "model_missing", "request_rejected"} & set(k for k, v in kinds.items() if v):
                alert("high_failure_rate", "warning", failed)
        if self.preflight and not self.preflight.get("ok", True):
            for problem in self.preflight.get("problems", []):
                out.append({"code": "preflight", "severity": "critical", "provider": problem.get("provider"), "model": problem.get("model"),
                            "count": 1, "calls": 0, "what": "failed the startup check", "sample": problem.get("detail", "")})
        return out

    def reset(self) -> None:
        with self._lock:
            self._events.clear(); self._fallbacks.clear(); self._alerted.clear(); self.preflight = None


_ALERT_TEXT = {
    "key_rejected": "the platform key was rejected or cannot spend (401/403/402); requests silently fall back to another provider",
    "model_missing": "the model no longer exists (404 / does not exist)",
    "request_rejected": "the model rejects our request shape (400); a parameter or setting it no longer accepts",
    "truncated": "answers are being cut off at max_tokens",
    "refused": "the model is refusing requests",
    "empty": "the model returns empty answers",
    "high_failure_rate": "more than 20% of calls are failing",
}

HEALTH = ProviderHealth()


def record_ok(provider: str, model: str, *, byok: bool = False) -> None:
    HEALTH.record(provider, model, "ok", byok=byok)


def record_failure(provider: str, model: str, error: BaseException, *, byok: bool = False) -> str:
    return HEALTH.record_error(provider, model, error, byok=byok)
