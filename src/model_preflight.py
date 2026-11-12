"""
Is every model the app is configured to call actually there, and does each platform key work? Free calls only (model lookups, no generation).

Run at startup (a background thread, MODEL_PREFLIGHT_ON_START) and before a deploy (scripts/preflight_models.py). It would have caught, on the day
they happened, a dead Anthropic key and a withdrawn image model (`gpt-image-1`), both of which only showed up as quiet fallbacks.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from . import models as M
from .provider_health import HEALTH

logger = logging.getLogger(__name__)

CHECKED_PROVIDERS = ("claude", "openai", "gemini")


def configured_models() -> Dict[str, Dict[str, List[str]]]:
    """provider -> {model id: the jobs it is configured for}."""
    out: Dict[str, Dict[str, List[str]]] = {}

    def add(provider: str, model_id: Optional[str], role: str) -> None:
        if model_id:
            out.setdefault(provider, {}).setdefault(model_id, []).append(role)

    for provider, model_id in M.ADVANCED_MODEL_IDS.items():
        add(provider, model_id, "advanced tier")
    for provider, model_id in M.BASIC_MODEL_IDS.items():
        add(provider, model_id, "basic tier")
    for provider, model_id in M.AUTOCOMPLETE_MODEL_IDS.items():
        add(provider, model_id, "autocomplete / classifier / voice")
    for provider, model_id in M.DESIGN_MODEL_IDS.items():
        add(provider, model_id, "design canvas")
    for (provider, agent), model_id in M.AGENT_MODEL_PINS.items():
        add(provider, model_id, f"pinned: {agent}")
    for provider, model_id in M.IMAGE_MODEL_IDS.items():
        add(provider, model_id, "image default")
    for provider, fallbacks in M.IMAGE_MODEL_FALLBACKS.items():
        for model_id in fallbacks:
            add(provider, model_id, "image fallback")
    return out


_OPENAI_IDS: Dict[int, tuple] = {}
_OPENAI_LOCK = threading.Lock()


def _openai_model_ids(key: str, timeout: float) -> set:
    """What this OpenAI account can use. One list call, shared by the parallel lookups of a run."""
    import openai
    with _OPENAI_LOCK:
        cached = _OPENAI_IDS.get(hash(key))
        if cached and time.time() - cached[0] < 30:
            return cached[1]
        ids = {m.id for m in openai.OpenAI(api_key=key, max_retries=0, timeout=timeout).models.list()}
        _OPENAI_IDS[hash(key)] = (time.time(), ids)
        return ids


def _lookup(provider: str, key: str, model_id: str, timeout: float) -> None:
    """Raises if the key is rejected or the model is not there."""
    if provider == "openai":
        # models.retrieve is not usable here: OpenAI still answers 200 for the withdrawn gpt-image-1 (probed 2026-11-12), while the
        # list leaves it out and every call to it fails with "does not exist". The list is what the account can actually use.
        if model_id not in _openai_model_ids(key, timeout):
            raise LookupError(f"The model '{model_id}' does not exist in this account's model list")
    elif provider == "claude":
        import anthropic
        anthropic.Anthropic(api_key=key, max_retries=0, timeout=timeout).models.retrieve(model_id)
    elif provider == "gemini":
        from google import genai
        from google.genai import types
        with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))) as client:
            client.models.get(model=model_id)
    else:
        raise ValueError(f"no check for provider {provider}")


def check_provider(provider: str, key: Optional[str], models: Dict[str, List[str]], timeout: float = 20.0) -> Dict[str, Any]:
    from .provider_health import classify_error
    if not key:
        return {"key": "not configured", "models": {}, "problems": []}
    rows: Dict[str, Dict[str, Any]] = {}
    problems: List[Dict[str, str]] = []
    key_state = "accepted"

    def one(model_id: str):
        try:
            _lookup(provider, key, model_id, timeout)
            return model_id, None
        except Exception as error:                                              # noqa: BLE001 - every kind of failure is a finding
            return model_id, error

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(models)))) as pool:
        results = list(pool.map(one, sorted(models)))
    for model_id, error in results:
        roles = models[model_id]
        if error is None:
            rows[model_id] = {"ok": True, "roles": roles}
            continue
        kind = classify_error(error)
        if kind == "key_rejected":
            key_state = "rejected"
        elif kind == "server_error" and any(s in type(error).__name__.lower() + str(error).lower() for s in ("connect", "timeout", "timed out")):
            key_state = "unreachable" if key_state == "accepted" else key_state
            rows[model_id] = {"ok": None, "roles": roles, "detail": f"could not reach {provider}: {type(error).__name__}"}
            continue
        detail = f"{kind}: {str(error)[:140]}".replace("\n", " ")
        rows[model_id] = {"ok": False, "roles": roles, "detail": detail}
        problems.append({"provider": provider, "model": model_id, "detail": detail})
    if key_state == "rejected":
        problems = [{"provider": provider, "model": "(key)", "detail": "the platform key was rejected (401/403)"}]    # one finding, not one per model
    return {"key": key_state, "models": rows, "problems": problems}


def run_preflight(keys: Optional[Dict[str, Optional[str]]] = None, timeout: float = 20.0, record: bool = True) -> Dict[str, Any]:
    """Check every checked provider that has a platform key. Result is stored for /agent/health/providers and problems are logged as alerts."""
    if keys is None:
        from config import settings
        keys = {"claude": settings.default_anthropic_api_key, "openai": settings.default_openai_api_key, "gemini": settings.default_gemini_api_key}
    wanted = configured_models()
    providers: Dict[str, Any] = {}
    problems: List[Dict[str, str]] = []
    for provider in CHECKED_PROVIDERS:
        result = check_provider(provider, keys.get(provider), wanted.get(provider, {}), timeout)
        providers[provider] = {"key": result["key"], "models": result["models"]}
        problems += result["problems"]
    report = {"ok": not problems, "checked_at": time.time(), "providers": providers, "problems": problems}
    if record:
        HEALTH.preflight = report
        for p in problems:
            logger.error("ROLLOUT_ALERT preflight provider=%s model=%s %s", p["provider"], p["model"], p["detail"])
        if not problems:
            logger.info("Model preflight OK: %s", ", ".join(f"{p}={v['key']}" for p, v in providers.items()))
    return report
