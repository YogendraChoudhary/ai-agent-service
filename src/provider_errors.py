"""Classify SDK errors and Agno error strings without retrying auth, validation or safety failures."""
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def status_code(err):
    status = (getattr(err, 'status_code', None) or getattr(getattr(err, 'response', None), 'status_code', None)
              or getattr(err, 'code', None))
    if isinstance(status, int):
        return status
    match = re.search(r'\b(400|401|403|404|408|422|429|5\d\d)\b', str(err))
    return int(match[1]) if match else None


def is_retryable(err):
    status = status_code(err)
    text = (type(err).__name__ + ' ' + str(err)).lower()
    if status in (400, 401, 403, 404, 422) or any(s in text for s in ('safety', 'content_filter', 'invalid api key', 'authentication', 'permission denied')):
        return False
    return status == 429 or status == 408 or (status is not None and 500 <= status < 600) or any(
        s in text for s in ('overloaded', 'timeout', 'timed out', 'connection reset', 'connection error',
                           'connectionreseterror', 'connectionerror', 'connecterror', 'apiconnectionerror',
                           'resource_exhausted', 'rate limit', 'ratelimit'))


# Our code and the installed SDK disagree about a call (an argument the SDK version does not have, ...). agno turns the TypeError into
# its run's error text, so this is matched on text too. Nothing the user did; another provider's SDK can still serve the request.
_SDK_CONTRACT_RE = re.compile(r"unexpected keyword argument|missing \d+ required (?:keyword-only |positional )?argument"
                              r"|takes \d+ positional arguments? but \d+ (?:were|was) given")
_PROVIDER_PREFIX_RE = re.compile(r"^(openai|claude|gemini|grok|deepseek) request failed:", re.IGNORECASE)
# Python errors from our own code: their text means nothing to a user and can reveal internals.
_INTERNAL_ERRORS = (TypeError, AttributeError, KeyError, NameError, IndexError, AssertionError, UnboundLocalError,
                    ZeroDivisionError, ImportError, NotImplementedError)


def is_sdk_contract_error(err):
    return bool(_SDK_CONTRACT_RE.search(str(err)))


def public_error_message(err, provider=None):
    """
    What a user may see for an error. Errors they can act on (bad key, rate limit, bad input, provider outage) keep their text; an SDK
    mismatch or a Python error from our own code gets a plain message instead. Log the real error yourself (with the request id).
    """
    text = str(err)
    match = _PROVIDER_PREFIX_RE.match(text)
    provider = provider or (match.group(1).lower() if match else None)
    if is_sdk_contract_error(err):
        name = {'claude': 'Claude', 'openai': 'OpenAI', 'gemini': 'Gemini', 'grok': 'Grok', 'deepseek': 'DeepSeek'}.get(provider or '')
        return (f"The {name} model is temporarily unavailable because of a problem on our side. Please try again shortly or choose "
                "another model." if name else "The AI service ran into a problem on our side. Please try again shortly.")
    if isinstance(err, _INTERNAL_ERRORS):
        return "Something went wrong on our side. Please try again."
    return text


def is_provider_unusable(err):
    """A broken key, missing permission or retired model: this provider cannot serve, but another one can."""
    status = status_code(err)
    text = (type(err).__name__ + ' ' + str(err)).lower()
    if any(s in text for s in ('safety', 'content_filter')):
        return False
    if is_sdk_contract_error(err):
        return True
    return status in (401, 403, 404) or any(
        s in text for s in ('invalid api key', 'invalid x-api-key', 'authentication', 'permission denied',
                           'model not found', 'model_not_found'))


def can_fall_back(err):
    """Errors that move a request to the next provider. Bad requests and safety blocks would fail everywhere."""
    return is_retryable(err) or is_provider_unusable(err)


def is_rate_limit(err):
    return status_code(err) == 429 or any(s in str(err).lower() for s in ('rate limit', 'resource_exhausted'))


def retry_delay(err):
    headers = getattr(getattr(err, 'response', None), 'headers', {}) or {}
    # Agno can drop SDK response headers; use the cap rather than retrying too early.
    value = headers.get('retry-after', '2')
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            seconds = 2.0
    return min(2.0, max(0.0, seconds))
