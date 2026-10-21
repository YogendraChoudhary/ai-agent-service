"""Small, bounded classifier; it never sends BYOK text through a platform key."""
import concurrent.futures
import threading

from config import settings
from .models import AUTOCOMPLETE_MODEL_IDS, ensure_allowed_model

_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='code_intent')
_SLOTS = threading.BoundedSemaphore(4)
_SYSTEM = 'Classify the request. Reply code for requests to produce or edit code; otherwise other. One word only.'


def _call(text, provider, key, timeout, request_id=None):
    import httpx
    from agno.agent import Agent
    from agno.run.base import RunStatus
    from .text_processing_agent import get_model
    from .observability import record_text_usage
    model = get_model(provider, key, use_advanced=False, cache_model=False)
    model.id = ensure_allowed_model(AUTOCOMPLETE_MODEL_IDS[provider])
    model.temperature = 0
    if provider == 'gemini':
        model.max_output_tokens = 5
    else:
        model.max_tokens = 5
    # SDK clients all receive a transport timeout and no automatic retries.
    with httpx.Client(timeout=timeout) as client:
        if provider == 'gemini':
            from google import genai
            from google.genai import types
            with genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))) as gemini_client:
                model.client = gemini_client
                response = Agent(model=model, instructions=_SYSTEM, telemetry=False).run(text)
                record_text_usage(request_id, model.id, getattr(response, 'metrics', None))
                return '' if getattr(response, 'status', None) == RunStatus.error else getattr(response, 'content', '')
        else:
            model.client_params = {'http_client': client, 'max_retries': 0}
        response = Agent(model=model, instructions=_SYSTEM, telemetry=False).run(text)
        record_text_usage(request_id, model.id, getattr(response, 'metrics', None))
        return '' if getattr(response, 'status', None) == RunStatus.error else getattr(response, 'content', '')


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
