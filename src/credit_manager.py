"""
Credit reservation manager for the AI agent.

Reads subscription:{userId} (written by the Supabase edge function) and
atomically holds in-flight credits via credits_reserved:{userId} using
Lua scripts. See docs/superpowers/specs/2026-05-02-agent-side-credit-
precheck-design.md for full design rationale.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Tuple

logger = logging.getLogger(__name__)

# Scale factor used internally so sub-1 credit costs (autocomplete = 0.2)
# can be expressed as integers — Redis INCRBY/DECRBY operate on integers.
CREDIT_SCALE = 10
CHARS_PER_CREDIT = 1000

# Costs in scaled units (multiply raw credits by CREDIT_SCALE)
IMAGE_CREDITS_SCALED = 50            # 5 credits per image
AUTOCOMPLETE_CREDITS_SCALED = 2      # 0.2 credits per acceptance

# Per-model image prices in scaled units, keyed "provider/model" or just "provider". Empty on purpose:
# every model costs the flat IMAGE_CREDITS_SCALED until a price is set here or through the
# IMAGE_CREDITS_BY_MODEL env var (JSON, e.g. {"openai/gpt-image-1": 80}). The backend still computes the
# charge from image_count; these numbers drive the agent-side reservation and are sent in the webhook.
IMAGE_CREDITS_BY_MODEL: dict = {}

# Reservation key auto-expires so an agent crash cannot permanently lock
# credits. 600s gives plenty of slack for slow image generation jobs.
RESERVATION_TTL_SECONDS = 600


def image_credits_scaled(provider: str | None, model: str | None) -> int:
    """Scaled credits for one image from this provider/model: exact model, then provider, then the flat default."""
    import json

    table = dict(IMAGE_CREDITS_BY_MODEL)
    try:
        from config import settings
        raw = getattr(settings, "image_credits_by_model", "") or ""
        if raw.strip():
            table.update(json.loads(raw))
    except Exception as exc:  # a bad override must never block image generation
        logger.warning("Ignoring invalid IMAGE_CREDITS_BY_MODEL: %s", exc)
    p, m = (provider or "").lower(), (model or "").lower()
    for key in (f"{p}/{m}", p):
        if key in table:
            try:
                return max(0, int(table[key]))
            except (TypeError, ValueError):
                break
    return IMAGE_CREDITS_SCALED


def text_credits_scaled(input_chars: int, output_chars: int) -> int:
    """
    Scaled credit cost for a text operation.
    Mirrors apps/backend/src/utils/creditCalculator.ts:calculateTextCredits.
    """
    total_chars = max(0, input_chars) + max(0, output_chars)
    raw_credits = max(1, math.ceil(total_chars / CHARS_PER_CREDIT))
    return raw_credits * CREDIT_SCALE


# Lua: atomic check-and-reserve.
# Returns {success_flag, available_after_scaled}.
_RESERVE_LUA = """
local sub  = redis.call('HMGET', KEYS[1], 'remaining_credits', 'purchased_credits')
local rem  = tonumber(sub[1]) or 0
local pur  = tonumber(sub[2]) or 0
local total_scaled = (rem + pur) * 10

local reserved = tonumber(redis.call('GET', KEYS[2])) or 0
local need = tonumber(ARGV[1])

if total_scaled - reserved < need then
  return {0, total_scaled - reserved}
end

local new_reserved = redis.call('INCRBY', KEYS[2], need)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
return {1, total_scaled - new_reserved}
"""

# Lua: atomic release. Capped at 0; deletes the key on full release so
# inactive users do not accumulate stale keys.
_RELEASE_LUA = """
local current = tonumber(redis.call('GET', KEYS[1])) or 0
local amount  = tonumber(ARGV[1])

if current <= amount then
  redis.call('DEL', KEYS[1])
  return 0
end

return redis.call('DECRBY', KEYS[1], amount)
"""


class CreditManager:
    """Atomic credit reservation against Redis."""

    def __init__(self, redis_client, reservation_ttl: int = RESERVATION_TTL_SECONDS):
        self._redis = redis_client
        self._reservation_ttl = reservation_ttl
        # Register the script once. Redis client caches the SHA so subsequent
        # calls go over the wire as EVALSHA (smaller than EVAL with the body).
        self._reserve_script = self._redis.register_script(_RESERVE_LUA)
        self._release_script = self._redis.register_script(_RELEASE_LUA)

    def reserve(self, user_id: str, scaled_amount: int) -> Tuple[bool, int]:
        """
        Atomically check-and-reserve `scaled_amount` for the user.

        Returns:
            (True,  available_after_scaled) on success.
            (False, available_now_scaled)   on insufficient credits or error.
        """
        if scaled_amount <= 0:
            return True, 0

        try:
            result = self._reserve_script(
                keys=[
                    f"subscription:{user_id}",
                    f"credits_reserved:{user_id}",
                ],
                args=[scaled_amount, self._reservation_ttl],
            )
        except Exception as exc:
            logger.error(
                "CreditManager.reserve failed for user=%s amount=%s: %s",
                user_id, scaled_amount, exc,
            )
            # Fail-closed: caller treats this as insufficient credits.
            return False, 0

        ok_flag, available_after = result
        return bool(int(ok_flag)), int(available_after)

    def release(self, user_id: str, scaled_amount: int) -> int:
        """
        Atomically release `scaled_amount`. Returns the new reserved value.
        Releasing more than is held is safe — the key is deleted and 0 returned.
        """
        if scaled_amount <= 0:
            return self._read_reserved(user_id)

        try:
            new_reserved = self._release_script(
                keys=[f"credits_reserved:{user_id}"],
                args=[scaled_amount],
            )
        except Exception as exc:
            logger.error(
                "CreditManager.release failed for user=%s amount=%s: %s",
                user_id, scaled_amount, exc,
            )
            return self._read_reserved(user_id)

        return int(new_reserved)

    def schedule_release(
        self,
        user_id: str,
        scaled_amount: int,
        delay_seconds: float,
    ) -> None:
        """
        Release `scaled_amount` after `delay_seconds` in a daemon thread.
        Non-blocking — returns immediately. Used after the webhook fires so
        the backend has time to update the DB before the reservation drops.
        """
        if scaled_amount <= 0:
            return

        def _run():
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            self.release(user_id, scaled_amount)

        threading.Thread(target=_run, daemon=True).start()

    def peek_available(self, user_id: str) -> int:
        """
        Read the user's currently-available scaled credits without
        mutating anything. Useful for friendly error messages.
        """
        try:
            sub = self._redis.hmget(
                f"subscription:{user_id}",
                "remaining_credits",
                "purchased_credits",
            )
            rem = int(sub[0]) if sub[0] else 0
            pur = int(sub[1]) if sub[1] else 0
            reserved = self._read_reserved(user_id)
            available = (rem + pur) * CREDIT_SCALE - reserved
            return max(0, available)
        except Exception as exc:
            logger.error("CreditManager.peek_available failed for user=%s: %s", user_id, exc)
            return 0

    def _read_reserved(self, user_id: str) -> int:
        try:
            raw = self._redis.get(f"credits_reserved:{user_id}")
            return int(raw) if raw else 0
        except Exception:
            return 0
