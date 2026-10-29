"""
HTTP surface for voice dictation.

  POST /v1/voice/dictate   multipart/form-data, one recorded utterance in, formatted text out
                             audio    the recording: any container PyAV reads (webm/opus, mp4/aac, wav, ogg ...). Required.
                             context  optional JSON text: style, language, cursor_context, custom_dictionary, app_context
                           header Authorization: Bearer <Eddyter licence key>   (when a gate is installed: src/voice_flow/access.py)
                           header X-Voice-Token when VOICE_FLOW_TOKEN is set
  GET  /v1/voice/status    what is loaded, and whether the model is currently reachable

One dictation gives ONE response, however long it is and however many pauses it holds: Whisper decodes the whole recording
and the formatter sees the whole text once, so lists and context are never cut in two.

Answers: 200 DictateResponse | 400 bad upload or context, or no key for an account on its own keys | 401 no licence key or wrong token |
402 no credits or plan past due | 403 invalid licence | 413 too large or too long | 429 server busy | 503 licence service down.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from collections import OrderedDict
from typing import List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from config import settings
from src.voice_flow.asr.audio import WHISPER_RATE, decode_audio_bytes
from src.voice_flow.asr.whisper_client import get_transcriber
from src.voice_flow.access import Caller, Gate
from src.voice_flow.formatter.engine import CompileResult, LlmError, VoiceFormattingService, build_backend_for
from src.voice_flow.formatter.post_processor import PostProcessor, Stitched
from src.voice_flow.schemas import AppContext, CursorContext, DictateContext, DictateResponse, Timings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/voice", tags=["Voice Dictation"])

AUDIO_MAX_BYTES = 10 * 1024 * 1024    # the uploaded file; a proxy in front should refuse bodies a little over this before they arrive
CONTEXT_MAX_CHARS = 50_000            # the JSON context part (the individual fields are capped much lower by DictateContext)
MIN_UTTERANCE_SECONDS = 0.05          # shorter than this is a key bounce, not speech: Whisper is never called for it
MAX_DICTATIONS_IN_FLIGHT = 8          # accepted at once (decoding, waiting for the speech model, or formatting); the next gets 429

OWN_FORMATTERS_MAX = 256              # formatters kept for accounts on their own keys, least recently used dropped first

_service: Optional[VoiceFormattingService] = None
_gate: Optional[Gate] = None          # licence + credits (src/voice_flow/access.py); None = the endpoint is open
_own_formatters: "OrderedDict[str, VoiceFormattingService]" = OrderedDict()
_in_flight = 0


def get_service() -> VoiceFormattingService:
    global _service
    if _service is None:
        _service = VoiceFormattingService()
    return _service


def set_gate(gate: Optional[Gate]) -> None:
    global _gate
    _gate = gate


def _formatter_for(caller: Caller) -> VoiceFormattingService:
    """The platform's formatter, or one on the caller's own key. Each own key has its own circuit breaker, so one
    account's expired key never pauses the model for anyone else."""
    if caller.own_key is None:
        return get_service()
    provider, key = caller.own_key
    ident = f"{provider}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"          # the key itself is never kept as a name or logged
    service = _own_formatters.get(ident)
    if service is None:
        service = _own_formatters[ident] = VoiceFormattingService(backend=build_backend_for(provider, key))
        while len(_own_formatters) > OWN_FORMATTERS_MAX:
            _own_formatters.popitem(last=False)
    else:
        _own_formatters.move_to_end(ident)
    return service


def _licence_key(request: Request) -> str:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise HTTPException(status_code=401, detail="Licence key required: send 'Authorization: Bearer <licence key>'")
    return value.strip()


def _verify_bearer_token(provided: str, expected: str) -> bool:
    """A shared-secret check (not a keyed MAC): constant-time, on UTF-8 bytes so a non-ASCII token cannot raise."""
    return not expected or secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


async def warm_up() -> None:
    """Load Whisper and open the Gemini connection before the first dictation. Failures are logged, never raised."""
    try:
        seconds = await get_transcriber().aload()
        logger.info("voice_flow: speech model ready (%.1fs)", seconds)
    except Exception:                                                 # noqa: BLE001
        logger.exception("voice_flow: speech model failed to load; dictation will retry on first use")
    service = get_service()
    if hasattr(service.backend, "warm"):
        try:
            await service.backend.warm()
            logger.info("voice_flow: formatting model %s ready", getattr(service.backend, "model", ""))
        except LlmError as exc:
            service.breaker.failure(exc)      # skip the doomed call on every utterance until the cool-down ends
            logger.warning("voice_flow: formatting model unavailable (%s); dictation will use the rule-based formatter", exc.reason)


async def _format(service: VoiceFormattingService, raw_text: str, app_context: AppContext, cursor: Optional[CursorContext],
                  dictionary: List[str], style: Optional[str] = None) -> Tuple[Stitched, CompileResult, float]:
    result = await service.compile(raw_text, app_context, cursor, dictionary, style)
    t = time.perf_counter()
    stitched = PostProcessor.stitch(result.formatted_text, cursor, app_context)
    return stitched, result, (time.perf_counter() - t) * 1000.0


def _parse_context(raw: str) -> DictateContext:
    if not raw.strip():
        return DictateContext()
    if len(raw) > CONTEXT_MAX_CHARS:
        raise HTTPException(status_code=400, detail="context is too long")
    try:
        return DictateContext.model_validate_json(raw)
    except ValidationError as exc:
        loc = exc.errors()[0]["loc"]
        raise HTTPException(status_code=400, detail=f"invalid context: {loc[-1]}" if loc else "invalid context: not a JSON object")


def _nothing_heard(audio_ms: float, asr_ms: float, started: float) -> DictateResponse:
    return DictateResponse(
        raw_text="", formatted_text="", leading_space_required=False, source="rules", fallback_reason="no speech detected",
        timings=Timings(audio_ms=audio_ms, asr_ms=asr_ms, total_ms=(time.perf_counter() - started) * 1000.0))


async def _dictate(data: bytes, ctx: DictateContext, caller: Caller) -> DictateResponse:
    started = time.perf_counter()
    try:
        audio = await run_in_threadpool(decode_audio_bytes, data)
    except Exception:                                                 # noqa: BLE001 - PyAV raises many types for bad input
        raise HTTPException(status_code=400, detail="could not decode the audio")
    if audio.size > settings.voice_max_utterance_seconds * WHISPER_RATE:
        raise HTTPException(status_code=413, detail=f"audio is longer than {settings.voice_max_utterance_seconds}s")
    audio_ms = audio.size / WHISPER_RATE * 1000.0
    if audio.size < MIN_UTTERANCE_SECONDS * WHISPER_RATE:
        return _nothing_heard(audio_ms, 0.0, started)                 # nothing is reserved or charged for a key bounce

    reserved = _gate.reserve(caller, audio_ms / 1000.0) if _gate else 0
    settled = False
    try:
        asr = await get_transcriber().transcribe(audio, ctx.custom_dictionary, ctx.language)
        if not asr.text.strip():
            if _gate:
                _gate.refund(caller, reserved)
                settled = True
            return _nothing_heard(asr.audio_ms, asr.asr_ms, started)
        service = _formatter_for(caller)
        stitched, result, post_ms = await _format(service, asr.text, ctx.app_context, ctx.cursor_context, ctx.custom_dictionary, ctx.style)
        total = (time.perf_counter() - started) * 1000.0
        llm_ms = result.processing_time_ms if result.source == "llm" else 0.0
        logger.info("voice_flow: %.1fs audio -> asr %.0f ms, %s %.0f ms, post %.2f ms, total %.0f ms%s",
                    asr.audio_ms / 1000, asr.asr_ms, result.source, result.processing_time_ms, post_ms, total,
                    f" ({result.fallback_reason})" if result.fallback_reason else "")
        response = DictateResponse(
            raw_text=asr.text, formatted_text=stitched.insert_text, leading_space_required=stitched.leading_space,
            trailing_space_required=stitched.trailing_space, source=result.source, fallback_reason=result.fallback_reason,
            timings=Timings(audio_ms=asr.audio_ms, asr_ms=asr.asr_ms, llm_ms=llm_ms, post_ms=post_ms, total_ms=total))
        if _gate:
            if not response.formatted_text.strip():
                _gate.refund(caller, reserved)
                settled = True
                return response
            reserved = _gate.reserve(caller, 0, output_chars=len(response.formatted_text), reserved=reserved)
            backend = service.backend
            _gate.settle(caller, reserved, output_chars=len(response.formatted_text), duration_ms=int(total),
                         provider=getattr(backend, "name", "unknown"), model=getattr(backend, "model", None) if result.source == "llm" else None)
            settled = True                                            # the webhook now owns the reservation
        return response
    except (Exception, asyncio.CancelledError) as exc:
        if _gate and not settled:
            _gate.refund(caller, reserved, error_code=type(exc).__name__, duration_ms=int((time.perf_counter() - started) * 1000))
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/dictate", response_model=DictateResponse)
async def dictate(request: Request, background: BackgroundTasks, audio: Optional[UploadFile] = File(None),
                  context: str = Form("")) -> DictateResponse:
    global _in_flight
    if not _verify_bearer_token(request.headers.get("x-voice-token", ""), settings.voice_flow_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    caller = Caller()
    if _gate is not None:
        caller = await _gate.admit(_licence_key(request), request.headers.get("x-request-id") or uuid.uuid4().hex[:8], background)
    if audio is None:
        raise HTTPException(status_code=400, detail="send the recording as the multipart part 'audio'")
    ctx = _parse_context(context)
    if audio.size is not None and audio.size > AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"the recording is larger than {AUDIO_MAX_BYTES // (1024 * 1024)} MB")
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="the recording is empty")
    if len(data) > AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"the recording is larger than {AUDIO_MAX_BYTES // (1024 * 1024)} MB")

    if _in_flight >= MAX_DICTATIONS_IN_FLIGHT:                        # a client that cannot keep up must not pile work on the speech model
        raise HTTPException(status_code=429, detail="too many dictations are being processed; try again in a moment",
                            headers={"Retry-After": "2"})
    _in_flight += 1
    try:
        return await _dictate(data, ctx, caller)
    finally:
        _in_flight -= 1


@router.get("/status")
async def status() -> dict:
    service, transcriber = get_service(), get_transcriber()
    return {
        "whisper_model": transcriber.model_name, "whisper_loaded": transcriber._model is not None,
        "compute_type": transcriber.compute_type,
        "format_provider": getattr(service.backend, "name", type(service.backend).__name__),
        "format_model": getattr(service.backend, "model", type(service.backend).__name__),
        "format_model_available": service.breaker.allow(), "last_format_error": service.breaker.last_reason or None,
        "default_style": settings.voice_default_style, "app_contexts_enabled": settings.voice_app_contexts_enabled,
    }
