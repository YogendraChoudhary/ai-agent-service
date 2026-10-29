"""
Who may dictate, and what a dictation costs.

POST /v1/voice/dictate is billed like /agent/stream. The Eddyter licence key (`Authorization: Bearer <key>`) is validated
against Redis, credits are reserved before the work and settled after it, and the backend is told what happened through the
same webhook, which is where the charge is computed.

Price: the same as AI text, so the backend needs no change. A dictation is reported as `text` usage with the formatted text as
output: 1 credit per 1,000 characters, at least 1. Nothing is charged when nothing was heard or when the request fails.

BYOK plans are never charged credits and format with their own supported provider key. A managed plan's "use my own keys"
setting also requires a personal key but retains credit billing. Neither can fall back to platform provider keys.

The gate is optional. Without one (scripts/voice_demo.py, unit tests) the endpoint is open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

from fastapi import BackgroundTasks, HTTPException
from starlette.concurrency import run_in_threadpool

from config import settings
from src.credit_manager import text_credits_scaled
from src.models import AUTOCOMPLETE_MODEL_IDS

logger = logging.getLogger(__name__)

# The providers whose light model the formatter can drive (src/voice_flow/formatter/engine.py).
FORMATTER_PROVIDERS = tuple(AUTOCOMPLETE_MODEL_IDS)
# Fast talkers say about 17 characters a second. The reservation is only a hold until the real charge is known, so a generous
# estimate costs nothing. Top up against the actual output before delivering it.
SPEECH_CHARS_PER_SECOND = 20


@dataclass(frozen=True)
class Caller:
    """Who is dictating, as far as the formatter and the bill are concerned."""
    user_id: Optional[str] = None
    billable: bool = False                         # False for accounts on their own keys, and when there is no gate
    own_key: Optional[Tuple[str, str]] = None      # (provider, key): the formatter runs on this instead of the platform's


class Gate(Protocol):
    async def admit(self, licence_key: str, request_id: str, background: BackgroundTasks) -> Caller: ...
    def reserve(self, caller: Caller, audio_seconds: float, *, output_chars: int = 0, reserved: int = 0) -> int: ...
    def settle(self, caller: Caller, reserved: int, *, output_chars: int, duration_ms: int, provider: str,
               model: Optional[str]) -> None: ...
    def refund(self, caller: Caller, reserved: int, *, error_code: Optional[str] = None, duration_ms: int = 0) -> None: ...


class LicenceGate:
    """The production gate. `host` is the api module: its redis_manager and credit_manager are set at startup, and its webhook
    functions are looked up at call time (scripts/run_local_server.py replaces them), so none of them is copied here."""

    def __init__(self, host: Any):
        self.host = host

    # -- before the work --------------------------------------------------------------------------------------------------
    async def admit(self, licence_key: str, request_id: str, background: BackgroundTasks) -> Caller:
        redis = self.host.redis_manager
        if not redis:
            logger.error(f"[{request_id}] Redis manager not available")
            raise HTTPException(status_code=503, detail="License validation service unavailable")

        result = await run_in_threadpool(redis.validate_license, licence_key)
        if not result["valid"]:
            code = result.get("code")
            logger.warning(f"[{request_id}] voice: licence validation failed: {result['error']}")
            if code == "LICENSE_SERVICE_UNAVAILABLE":
                raise HTTPException(status_code=503, detail=result["error"])
            message = result["error"] if code else f"Invalid license: {result['error']}"
            raise HTTPException(status_code=403, detail={"error": message, "code": code} if code else message)

        user_id, info = result["user_id"], result.get("license") or {}
        background.add_task(redis.update_heartbeat, licence_key)
        byok = bool(info.get("requires_user_api_keys", False))
        if info.get("subscription_status") == "past_due" and not byok:
            raise HTTPException(status_code=402, detail="Your payment is past due. Please update your payment method to continue using AI features.")

        keys = await run_in_threadpool(redis.get_user_api_keys, user_id)
        if keys.get("_error_code") == "LICENSE_SERVICE_UNAVAILABLE":
            raise HTTPException(status_code=503, detail="API key service unavailable")
        own_key = None
        if byok or keys.get("use_own_keys"):
            own_key = self._own_formatter_key(redis, {**keys, "use_own_keys": True})
            if own_key is None:
                raise HTTPException(status_code=400, detail={
                    "error": "No API key configured for voice dictation. Add a Gemini, OpenAI, Claude, Grok or DeepSeek key in settings.",
                    "code": "NO_API_KEY"})
        return Caller(user_id=user_id, billable=not byok, own_key=own_key)

    @staticmethod
    def _own_formatter_key(redis: Any, keys: dict) -> Optional[Tuple[str, str]]:
        """The user's key for the first provider the formatter supports, the platform's configured provider first."""
        order = sorted(FORMATTER_PROVIDERS, key=lambda p: p != settings.voice_format_provider)
        for provider in order:
            key = redis.resolve_api_key_from_user_keys(keys, provider, None)
            if key and key.strip():
                return provider, key
        return None

    def reserve(self, caller: Caller, audio_seconds: float, *, output_chars: int = 0, reserved: int = 0) -> int:
        """Hold the credits a dictation of this length could cost. Returns the scaled amount held (0 when nothing is held)."""
        h = self.host
        if not (caller.billable and h.CREDIT_PRECHECK_ENABLED):
            return 0
        if h.credit_manager is None:
            raise HTTPException(status_code=503, detail="Credit service unavailable")
        needed = text_credits_scaled(0, max(output_chars, int(audio_seconds * SPEECH_CHARS_PER_SECOND)))
        extra = max(0, needed - reserved)
        if not extra:
            return reserved
        ok, available = h.credit_manager.reserve(caller.user_id, extra)
        if not ok:
            logger.warning(f"voice: insufficient credits for user={caller.user_id}: needed={needed} available={available}")
            raise HTTPException(status_code=402, detail={
                "error": "Insufficient credits", "code": "INSUFFICIENT_CREDITS", "available": available, "needed": extra})
        return reserved + extra

    # -- after the work ---------------------------------------------------------------------------------------------------
    def settle(self, caller: Caller, reserved: int, *, output_chars: int, duration_ms: int, provider: str,
               model: Optional[str]) -> None:
        """A dictation was delivered: tell the backend, which charges it, and let the reservation go."""
        if not caller.billable:
            return
        h = self.host
        if h.CREDIT_PRECHECK_ENABLED and h.credit_manager is not None:
            h.fire_credit_deduction_and_release(
                user_id=caller.user_id, input_chars=0, output_chars=output_chars, operation_type="text", image_count=0,
                reserved_scaled=reserved, credit_manager=h.credit_manager, settle_delay_seconds=h.CREDIT_SETTLE_DELAY_SECONDS,
                provider=provider, model=model, duration_ms=duration_ms, success=True)
        else:
            h.fire_credit_deduction(caller.user_id, 0, output_chars, "text", 0, provider=provider, model=model,
                                    duration_ms=duration_ms, success=True)

    def refund(self, caller: Caller, reserved: int, *, error_code: Optional[str] = None, duration_ms: int = 0) -> None:
        """Nothing was delivered (nothing heard, or a failure): let the reservation go. A failure is also reported, for analytics."""
        h = self.host
        if reserved and h.credit_manager is not None:
            h.credit_manager.schedule_release(caller.user_id, reserved, delay_seconds=0)
        if error_code and caller.billable:
            h.fire_credit_deduction(caller.user_id, 0, 0, "text", 0, duration_ms=duration_ms, success=False, error_code=error_code)
