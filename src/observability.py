"""
Request observability: what came in, what went out, and where the time went.

One RequestTrace per /agent/* request records timing marks (auth, routing, first token, text done,
each image, all images) and attributes (agent type, route, model, plan). When the request finishes the
trace is emitted as:

  * a human line   "⏱ [rid] ... total=11.8s | auth 0.01s | first_token 0.9s | image 7.9s ..."
  * a JSON line    "REQUEST_SUMMARY {...}"  (and appended to REQUEST_LOG_FILE if set)
  * an in-memory ring served by GET /agent/metrics when METRICS_ENDPOINT_ENABLED=true

Request/response BODIES are logged only when LOG_REQUEST_BODIES=true (they contain user text). API keys,
licence keys and tokens are redacted from anything logged, always.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger("observability")

_SECRET_FRAGMENTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def redact(obj: Any) -> Any:
    """Copy of obj with every secret-looking field replaced. Never mutates the input."""
    if isinstance(obj, dict):
        return {
            k: ("***redacted***" if any(f in str(k).lower() for f in _SECRET_FRAGMENTS) and v else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def _clip(text: Any, limit: Optional[int] = None) -> str:
    limit = limit or int(settings.log_body_max_chars)
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return s if len(s) <= limit else f"{s[:limit]}… [+{len(s) - limit} chars]"


class RequestTrace:
    def __init__(self, request_id: str, endpoint: str):
        self.request_id = request_id
        self.endpoint = endpoint
        self.started_at = time.time()
        self._t0 = time.perf_counter()
        self.marks: Dict[str, float] = {}
        self.attrs: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def mark(self, name: str, **attrs: Any) -> None:
        with self._lock:
            self.marks.setdefault(name, round(self.elapsed(), 3))   # first occurrence wins
            self.attrs.update(attrs)

    def set(self, **attrs: Any) -> None:
        with self._lock:
            self.attrs.update(attrs)


_TRACES: Dict[str, RequestTrace] = {}
_TRACES_LOCK = threading.Lock()
_RING: deque = deque(maxlen=max(10, int(settings.trace_ring_size)))
_FILE_LOCK = threading.Lock()


def start_trace(request_id: str, endpoint: str, **attrs: Any) -> RequestTrace:
    t = RequestTrace(request_id, endpoint)
    t.attrs.update(attrs)
    with _TRACES_LOCK:
        _TRACES[request_id] = t
    return t


def get_trace(request_id: Optional[str]) -> Optional[RequestTrace]:
    if not request_id:
        return None
    with _TRACES_LOCK:
        return _TRACES.get(request_id)


def trace_mark(request_id: Optional[str], name: str, **attrs: Any) -> None:
    """No-op when the request has no trace (tests, scripts), so call sites never need a guard."""
    t = get_trace(request_id)
    if t is not None:
        t.mark(name, **attrs)


def trace_set(request_id: Optional[str], **attrs: Any) -> None:
    t = get_trace(request_id)
    if t is not None:
        t.set(**attrs)


def summarize_images(pending_images: Optional[dict]) -> List[Dict[str, Any]]:
    out = []
    for key, e in (pending_images or {}).items():
        if str(key).startswith("__") or not isinstance(e, dict):
            continue
        out.append({
            "status": e.get("status"),
            "provider": e.get("provider"),
            "model": e.get("model"),
            "seconds": e.get("seconds"),
            "adapter_seconds": (e.get("timings") or {}).get("adapter_seconds"),
            "upload_seconds": (e.get("timings") or {}).get("upload_seconds"),
            "width": e.get("width"),
            "height": e.get("height"),
            "url": e.get("cdn_url"),
            "error": e.get("error"),
        })
    return out


def finish_trace(
    request_id: str,
    status: str = "ok",
    error: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    http_status: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    with _TRACES_LOCK:
        t = _TRACES.pop(request_id, None)
    if t is None:
        return None
    total = round(t.elapsed(), 3)
    summary = {
        "request_id": request_id,
        "endpoint": t.endpoint,
        "status": status,
        "http_status": http_status,
        "started_at": datetime.fromtimestamp(t.started_at, tz=timezone.utc).isoformat(timespec="milliseconds"),
        "total_seconds": total,
        "marks": dict(sorted(t.marks.items(), key=lambda kv: kv[1])),
        **t.attrs,
        "images": images or [],
    }
    if error:
        summary["error"] = _clip(error, 400)
    _RING.append(summary)

    parts = [f"total={total:.2f}s"]
    for name, secs in summary["marks"].items():
        parts.append(f"{name} {secs:.2f}s")
    for i, img in enumerate(summary["images"], 1):
        parts.append(f"image#{i} {img.get('provider')}/{img.get('model')} {img.get('status')} {img.get('seconds')}s")
    route = summary.get("route") or "-"
    logger.info(
        f"⏱ [{request_id}] {t.endpoint} type={summary.get('type')} chat={summary.get('chat_model')} "
        f"route={route} status={status} | " + " | ".join(parts)
    )
    logger.info("REQUEST_SUMMARY " + json.dumps(summary, ensure_ascii=False, default=str))

    path = (settings.request_log_file or "").strip()
    if path:
        try:
            with _FILE_LOCK, open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:  # logging must never break a request
            logger.warning(f"Could not write REQUEST_LOG_FILE: {exc}")
    return summary


# ---------------------------------------------------------------------------
# Bodies (opt-in)
# ---------------------------------------------------------------------------

def log_request_body(request_id: str, endpoint: str, payload: Dict[str, Any]) -> None:
    if settings.log_request_bodies:
        logger.info(f"📥 [{request_id}] {endpoint} REQUEST  {_clip(redact(payload))}")


def log_response_body(request_id: str, endpoint: str, body: Any) -> None:
    if settings.log_request_bodies:
        logger.info(f"📤 [{request_id}] {endpoint} RESPONSE {_clip(redact(body))}")


def record_text_usage(request_id, model_id, metrics):
    """Accumulate model/repair usage. A missing price or token count remains unknown."""
    from .models import TEXT_PRICES_PER_MTOK
    trace = get_trace(request_id)
    if trace is None or metrics is None:
        return
    data = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
    if not isinstance(data, dict):
        return
    tokens_in, tokens_out = data.get('input_tokens'), data.get('output_tokens')
    if tokens_in is None or tokens_out is None:
        return
    prices = {**TEXT_PRICES_PER_MTOK, **json.loads(settings.text_prices_per_mtok)}
    price = prices.get(model_id)
    cost = ((tokens_in * price[0] + tokens_out * price[1]) / 1_000_000) if price else None
    with trace._lock:
        usage = trace.attrs.get('usage', []) + [dict(model=model_id, tokens_in=tokens_in, tokens_out=tokens_out, est_cost_usd=cost)]
        trace.attrs.update(usage=usage, tokens_in=sum(u['tokens_in'] for u in usage),
                           tokens_out=sum(u['tokens_out'] for u in usage),
                           est_cost_usd=sum(u['est_cost_usd'] for u in usage) if all(u['est_cost_usd'] is not None for u in usage) else None)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _pct(values: List[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)


def recent_summaries(limit: int = 50) -> List[Dict[str, Any]]:
    return list(_RING)[-max(1, limit):][::-1]


def latency_stats() -> Dict[str, Any]:
    """p50/p95/max of total_seconds per endpoint+route, plus per image model and per long-form mode (blog)."""
    groups: Dict[str, List[float]] = {}
    images: Dict[str, List[float]] = {}
    tasks: Dict[str, List[float]] = {}
    modes: Dict[str, List[float]] = {}
    first_content: Dict[str, List[float]] = {}
    for s in list(_RING):
        if s.get("status") == "ok":
            groups.setdefault(f"{s['endpoint']} | {s.get('route') or '-'}", []).append(s["total_seconds"])
            if s.get('task'):
                tasks.setdefault(f"{s['task']} | {s.get('model_used') or s.get('chat_model')}", []).append(s['total_seconds'])
            if s.get('mode'):
                key = f"{s['mode']} | {s.get('model_used') or s.get('chat_model')}"
                modes.setdefault(key, []).append(s['total_seconds'])
                first = s.get('marks', {}).get('first_token')
                if first is not None:
                    first_content.setdefault(key, []).append(first)
        for img in s.get("images", []):
            if img.get("status") == "done" and img.get("seconds") is not None:
                images.setdefault(f"{img.get('provider')}/{img.get('model')}", []).append(float(img["seconds"]))

    def fmt(d):
        return {k: {"count": len(v), "p50": _pct(v, 0.5), "p95": _pct(v, 0.95), "max": round(max(v), 3)} for k, v in d.items()}

    return {"requests": fmt(groups), "tasks": fmt(tasks), "modes": fmt(modes), "mode_first_content": fmt(first_content),
            "image_generation": fmt(images), "samples": len(_RING)}
