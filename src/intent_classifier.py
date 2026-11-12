"""Small, bounded classifier; it never sends BYOK text through a platform key."""
import concurrent.futures
import threading
import time

from config import settings
from .models import AUTOCOMPLETE_MODEL_IDS, claude_accepts_temperature, claude_light_request_params, ensure_allowed_model

_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='code_intent')
_SLOTS = threading.BoundedSemaphore(4)
# classify_label (document_action "auto") has its own workers: a burst of those must not starve the code classifier, nor the reverse.
LABEL_CONCURRENCY = settings.document_action_classifier_concurrency
_LABEL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=LABEL_CONCURRENCY, thread_name_prefix='label_intent')
_LABEL_SLOTS = threading.BoundedSemaphore(LABEL_CONCURRENCY)
_SYSTEM = 'Classify the request. Reply code for requests to produce or edit code; otherwise other. One word only.'


def _answer(response, provider, model_id, run_status):
    """The classifier's reply, with a failed run counted for the rollout health view (the caller only ever sees '' -> 'other')."""
    from .provider_health import HEALTH, classify_error, record_ok
    if getattr(response, 'status', None) == run_status.error:
        HEALTH.record(provider, model_id, classify_error(RuntimeError(str(getattr(response, 'content', '')))), detail=str(getattr(response, 'content', '')))
        return ''
    record_ok(provider, model_id)
    return getattr(response, 'content', '')


def _call(text, provider, key, timeout, request_id=None, system=_SYSTEM, max_tokens=5, sdk_deadline_floor=0.0):
    import httpx
    from agno.agent import Agent
    from agno.run.base import RunStatus
    from .text_processing_agent import get_model
    from .observability import record_text_usage
    model = get_model(provider, key, use_advanced=False, cache_model=False)
    model.id = ensure_allowed_model(AUTOCOMPLETE_MODEL_IDS[provider])
    # get_model() built this model for the basic tier, whose Claude ID may reject temperature; the ID was just replaced
    model.temperature = None if provider == 'claude' and not claude_accepts_temperature(model.id) else 0
    if provider == 'gemini':
        model.max_output_tokens = max_tokens
    else:
        model.max_tokens = max_tokens
    if provider == 'claude':
        # Haiku 5.x thinks by default and counts it against max_tokens; the answer is one word, so no thinking, lowest effort.
        # The cap is 8, not 5: the same word is ~30-40% more tokens than on Haiku 4.5 (measured 0.95 accuracy at 8).
        light = claude_light_request_params(model.id)
        model.thinking, model.output_config = light.get('thinking'), light.get('output_config')
        if light:
            model.max_tokens = max(max_tokens, 8)
    # SDK clients all receive a transport timeout and no automatic retries.
    with httpx.Client(timeout=timeout) as client:
        if provider == 'gemini':
            from google import genai
            from google.genai import types
            with genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=int(max(timeout, sdk_deadline_floor) * 1000), retry_options=types.HttpRetryOptions(attempts=1))) as gemini_client:
                model.client = gemini_client
                response = Agent(model=model, instructions=system, telemetry=False).run(text)
                record_text_usage(request_id, model.id, getattr(response, 'metrics', None))
                return _answer(response, provider, model.id, RunStatus)
        else:
            model.client_params = {'http_client': client, 'max_retries': 0}
        response = Agent(model=model, instructions=system, telemetry=False).run(text)
        record_text_usage(request_id, model.id, getattr(response, 'metrics', None))
        return _answer(response, provider, model.id, RunStatus)


def classify_intent(text, provider, api_key, is_byok=False, platform_key=None, call=None, request_id=None):
    provider, key = (provider, api_key) if is_byok else ('claude', platform_key)
    if not key or not _SLOTS.acquire(blocking=False):
        return 'other'
    timeout = settings.code_classifier_timeout_seconds
    future = (_POOL.submit(call, text[:1000], provider, key, timeout) if call else
              _POOL.submit(_call, text[:1000], provider, key, timeout, request_id))
    future.add_done_callback(lambda _: _SLOTS.release())
    try:
        result = future.result(timeout=timeout)
        return result.strip().lower() if isinstance(result, str) and result.strip().lower() in ('code', 'other') else 'other'
    except Exception:
        future.cancel()
        return 'other'


# Gemini refuses a request deadline under 10 s ("Manually set deadline 1.5s is too short"). The caller still stops waiting at `timeout`.
GEMINI_MIN_DEADLINE = 10.0


def classify_label(text, provider, api_key, *, system, labels, is_byok=False, platform_key=None, timeout=None, call=None, request_id=None):
    """
    One word from `labels`, or None when the call failed, timed out, found no free slot within the timeout or answered something else.
    Same key rules as classify_intent: a BYOK request stays on its own key and provider, anything else uses the platform's Claude key
    (light tier). A burst waits briefly for a slot (the caller is a worker thread) rather than giving up at once.
    """
    provider, key = (provider, api_key) if is_byok else ('claude', platform_key)
    if not key:
        return None
    timeout = timeout or settings.code_classifier_timeout_seconds
    started = time.monotonic()
    if not _LABEL_SLOTS.acquire(timeout=timeout):
        return None
    remaining = max(0.05, timeout - (time.monotonic() - started))
    future = (_LABEL_POOL.submit(call, text[:1000], provider, key, remaining) if call else
              _LABEL_POOL.submit(_call, text[:1000], provider, key, remaining, request_id, system, 8, GEMINI_MIN_DEADLINE))
    future.add_done_callback(lambda _: _LABEL_SLOTS.release())
    try:
        result = future.result(timeout=remaining)
    except Exception:
        future.cancel()
        return None
    answer = result.strip().strip('.').lower() if isinstance(result, str) else ''
    return answer if answer in labels else None
