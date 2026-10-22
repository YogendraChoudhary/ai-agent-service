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


def is_provider_unusable(err):
    """A broken key, missing permission or retired model: this provider cannot serve, but another one can."""
    status = status_code(err)
    text = (type(err).__name__ + ' ' + str(err)).lower()
    if any(s in text for s in ('safety', 'content_filter')):
        return False
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
