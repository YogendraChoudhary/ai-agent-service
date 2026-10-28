"""
HTTP and WebSocket surface for voice dictation.

  POST /v1/voice/format    raw transcript + caret context -> formatted text      (client does its own speech recognition)
  POST /v1/voice/dictate   one recorded utterance (any audio container) -> formatted text
  WS   /v1/voice/stream    PCM16 audio frames in, one "final" frame per utterance out
  GET  /v1/voice/status    what is loaded, and whether the model is currently reachable
  GET  /v1/voice/demo      a microphone test page

Stream protocol (JSON text frames and binary audio frames). One dictation gives ONE final frame, however long it is and
however many pauses it holds:
  client: {"type":"start", sample_rate, mode:"push_to_talk"|"auto", silence_ms (auto: 0 = only "end" finishes), style:"clean"|"polished",
          language, cursor_context, custom_dictionary, token}
          <binary PCM16 mono little-endian frames>
          {"type":"context", ...}   optional, only the fields that changed (cursor_context, custom_dictionary, language, style) for the NEXT utterance
          {"type":"end"}            push-to-talk release (also accepted in auto mode)  |  {"type":"cancel"}
  server: {"type":"ready"} {"type":"speech_start"} {"type":"final", raw_text, formatted_text, ..., timings} {"type":"error"}
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import secrets
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from config import settings
from src.voice_flow.asr.audio import WHISPER_RATE, decode_audio_bytes, pcm16_to_float, resample_audio
from src.voice_flow.asr.vad import SPEECH_START, UTTERANCE_END, EndpointDetector
from src.voice_flow.asr.whisper_client import Transcriber, get_transcriber
from src.voice_flow.formatter.engine import CompileResult, LlmError, VoiceFormattingService
from src.voice_flow.formatter.post_processor import PostProcessor, Stitched
from src.voice_flow.schemas import (
    AppContext, ContextUpdateFrame, CursorContext, DictateRequest, DictateResponse, FinalFrame, FormatRequest,
    FormatResponse, StartFrame, Timings, error_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/voice", tags=["Voice Dictation"])

MAX_FRAME_BYTES = 1_000_000
MIN_UTTERANCE_SECONDS = 0.05          # shorter than this is a key bounce, not speech: Whisper is never called for it
MAX_QUEUED_UTTERANCES = 8             # utterances waiting for the speech model; more than this is a client that cannot keep up
PRE_ROLL_MS = 400                     # audio kept from before the speech started, in auto mode
_DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"

_service: Optional[VoiceFormattingService] = None


def get_service() -> VoiceFormattingService:
    global _service
    if _service is None:
        _service = VoiceFormattingService()
    return _service


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


async def _format(raw_text: str, app_context: AppContext, cursor: Optional[CursorContext],
                  dictionary: List[str], style: Optional[str] = None) -> Tuple[Stitched, CompileResult, float]:
    result = await get_service().compile(raw_text, app_context, cursor, dictionary, style)
    t = time.perf_counter()
    stitched = PostProcessor.stitch(result.formatted_text, cursor, app_context)
    return stitched, result, (time.perf_counter() - t) * 1000.0


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@router.post("/format", response_model=FormatResponse)
async def format_transcript(req: FormatRequest, request: Request) -> FormatResponse:
    if not _verify_bearer_token(request.headers.get("x-voice-token", ""), settings.voice_flow_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    t0 = time.perf_counter()
    stitched, result, _ = await _format(req.raw_text, req.app_context, req.cursor_context, req.custom_dictionary, req.style)
    return FormatResponse(
        raw_text=req.raw_text, formatted_text=stitched.insert_text,
        leading_space_required=stitched.leading_space, trailing_space_required=stitched.trailing_space,
        source=result.source, fallback_reason=result.fallback_reason,
        llm_latency_ms=result.processing_time_ms, total_latency_ms=(time.perf_counter() - t0) * 1000.0)


@router.post("/dictate", response_model=DictateResponse)
async def dictate(req: DictateRequest, request: Request) -> DictateResponse:
    if not _verify_bearer_token(request.headers.get("x-voice-token", ""), settings.voice_flow_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    t0 = time.perf_counter()
    try:
        data = base64.b64decode(req.audio_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="audio_b64 is not valid base64")
    try:
        audio = await run_in_threadpool(decode_audio_bytes, data)
    except Exception:                                                 # noqa: BLE001 - PyAV raises many types for bad input
        raise HTTPException(status_code=400, detail="could not decode the audio")
    if audio.size > settings.voice_max_utterance_seconds * WHISPER_RATE:
        raise HTTPException(status_code=413, detail=f"audio is longer than {settings.voice_max_utterance_seconds}s")

    asr = await get_transcriber().transcribe(audio, req.custom_dictionary, req.language)
    stitched, result, _ = await _format(asr.text, req.app_context, req.cursor_context, req.custom_dictionary, req.style)
    return DictateResponse(
        raw_text=asr.text, formatted_text=stitched.insert_text,
        leading_space_required=stitched.leading_space, trailing_space_required=stitched.trailing_space,
        source=result.source, fallback_reason=result.fallback_reason or ("no speech detected" if not asr.text else None),
        audio_ms=asr.audio_ms, asr_latency_ms=asr.asr_ms, llm_latency_ms=result.processing_time_ms,
        total_latency_ms=(time.perf_counter() - t0) * 1000.0)


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


@router.get("/demo", response_class=HTMLResponse)
async def demo() -> HTMLResponse:
    return HTMLResponse(_DEMO_PAGE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@dataclass
class _Context:
    app_context: AppContext = AppContext.GENERAL
    cursor: Optional[CursorContext] = None
    dictionary: List[str] = field(default_factory=list)
    language: Optional[str] = None
    style: Optional[str] = None


@dataclass
class _Utterance:
    pcm: bytes
    sample_rate: int
    context: _Context
    t_end: float


class _Session:
    """Audio and caret context for one WebSocket connection. Utterances are decoded in order by a single worker."""

    def __init__(self) -> None:
        self.sample_rate = WHISPER_RATE
        self.mode = "push_to_talk"
        self.context = _Context()
        self.buffer = bytearray()
        self.detector: Optional[EndpointDetector] = None
        self.started = False

    def start(self, frame: StartFrame) -> None:
        self.sample_rate, self.mode, self.started = frame.sample_rate, frame.mode, True
        self.context = _Context(frame.app_context, frame.cursor_context, frame.custom_dictionary, frame.language, frame.style)
        self.buffer = bytearray()
        silence = settings.voice_endpoint_silence_ms if frame.silence_ms is None else frame.silence_ms
        self.detector = EndpointDetector(frame.sample_rate, silence) if frame.mode == "auto" else None

    def snapshot(self) -> _Utterance:
        utterance = _Utterance(bytes(self.buffer), self.sample_rate,
                               replace(self.context, dictionary=list(self.context.dictionary)), time.perf_counter())
        self.buffer = bytearray()
        if self.detector:
            self.detector.reset()
        return utterance

    def trim_pre_roll(self) -> None:
        keep = self.sample_rate * 2 * PRE_ROLL_MS // 1000
        if self.detector and not self.detector.speaking and len(self.buffer) > keep:
            del self.buffer[:-keep]

    @property
    def seconds(self) -> float:
        return len(self.buffer) / 2 / self.sample_rate


async def _process(ws: WebSocket, utterance: _Utterance, transcriber: Transcriber) -> None:
    seconds = len(utterance.pcm) / 2 / utterance.sample_rate
    if seconds < MIN_UTTERANCE_SECONDS:
        # A bounced key or an "end" with nothing before it: answer at once, without a trip to Whisper's worker thread.
        # It still goes through the queue, so results arrive in the order the "end" frames did.
        await ws.send_json(FinalFrame(
            raw_text="", formatted_text="", leading_space_required=False, source="rules", fallback_reason="no speech detected",
            timings=Timings(audio_ms=seconds * 1000.0, total_ms=(time.perf_counter() - utterance.t_end) * 1000.0)).model_dump())
        return
    audio = resample_audio(pcm16_to_float(utterance.pcm), utterance.sample_rate)
    ctx = utterance.context
    asr = await transcriber.transcribe(audio, ctx.dictionary, ctx.language)
    if not asr.text:
        total = (time.perf_counter() - utterance.t_end) * 1000.0
        await ws.send_json(FinalFrame(
            raw_text="", formatted_text="", leading_space_required=False, source="rules", fallback_reason="no speech detected",
            timings=Timings(audio_ms=asr.audio_ms, asr_ms=asr.asr_ms, total_ms=total)).model_dump())
        return
    stitched, result, post_ms = await _format(asr.text, ctx.app_context, ctx.cursor, ctx.dictionary, ctx.style)
    total = (time.perf_counter() - utterance.t_end) * 1000.0
    llm_ms = result.processing_time_ms if result.source == "llm" else 0.0
    logger.info("voice_flow: %.1fs audio -> asr %.0f ms, %s %.0f ms, post %.2f ms, total %.0f ms%s",
                asr.audio_ms / 1000, asr.asr_ms, result.source, result.processing_time_ms, post_ms, total,
                f" ({result.fallback_reason})" if result.fallback_reason else "")
    await ws.send_json(FinalFrame(
        raw_text=asr.text, formatted_text=stitched.insert_text, leading_space_required=stitched.leading_space,
        trailing_space_required=stitched.trailing_space, source=result.source, fallback_reason=result.fallback_reason,
        timings=Timings(audio_ms=asr.audio_ms, asr_ms=asr.asr_ms, llm_ms=llm_ms, post_ms=post_ms, total_ms=total),
    ).model_dump())


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    session = _Session()
    queue: "asyncio.Queue[Optional[_Utterance]]" = asyncio.Queue(maxsize=MAX_QUEUED_UTTERANCES)
    transcriber = get_transcriber()

    async def worker() -> None:
        while (utterance := await queue.get()) is not None:
            try:
                await _process(ws, utterance, transcriber)
            except WebSocketDisconnect:
                return
            except Exception:                                         # noqa: BLE001 - one bad utterance must not end the session
                logger.exception("voice_flow: utterance failed")
                try:
                    await ws.send_json(error_frame("could not process that utterance", "processing_failed"))
                except Exception:                                     # noqa: BLE001 - the client is already gone
                    return

    worker_task = asyncio.create_task(worker())

    async def finish_utterance() -> None:
        utterance = session.snapshot()                    # also empties the buffer, so a refused utterance is dropped, not kept
        try:
            queue.put_nowait(utterance)
        except asyncio.QueueFull:
            await ws.send_json(error_frame("too many utterances are waiting; wait for results before sending more", "rate_limited"))

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                if not session.started:
                    await ws.send_json(error_frame("send a start frame before audio", "no_session"))
                    continue
                if len(chunk) > MAX_FRAME_BYTES:
                    await ws.send_json(error_frame("audio frame too large", "frame_too_large"))
                    continue
                session.buffer.extend(chunk)
                if session.detector:
                    for event in session.detector.feed(chunk):
                        if event == SPEECH_START:
                            await ws.send_json({"type": "speech_start"})
                        elif event == UTTERANCE_END:
                            await finish_utterance()
                    session.trim_pre_roll()
                if session.seconds > settings.voice_max_utterance_seconds:
                    # A memory guard, not a chunk size: say so, so a cut-off dictation is never a silent surprise.
                    await ws.send_json({"type": "notice", "code": "max_duration",
                                        "message": f"Reached the {settings.voice_max_utterance_seconds}s limit; formatting what was captured."})
                    await finish_utterance()
                continue

            try:
                frame = json.loads(message.get("text") or "")
                kind = frame.get("type")
            except (ValueError, AttributeError):
                await ws.send_json(error_frame("frames must be JSON objects or binary audio", "bad_frame"))
                continue

            if kind == "start":
                try:
                    start = StartFrame.model_validate(frame)
                except ValidationError as exc:
                    await ws.send_json(error_frame(f"invalid start frame: {exc.errors()[0]['loc'][-1]}", "bad_start"))
                    continue
                if not _verify_bearer_token(start.token, settings.voice_flow_token):
                    await ws.send_json(error_frame("unauthorized", "unauthorized"))
                    await ws.close(code=4401)
                    break
                session.start(start)
                await ws.send_json({"type": "ready"})
            elif kind == "context" and session.started:
                try:
                    update = ContextUpdateFrame.model_validate(frame)
                except ValidationError:
                    await ws.send_json(error_frame("invalid context frame", "bad_context"))
                    continue
                current = session.context                  # only what the frame carries changes; the dictionary is not resent
                session.context = replace(
                    current,
                    app_context=current.app_context if update.app_context is None else update.app_context,
                    cursor=current.cursor if update.cursor_context is None else update.cursor_context,
                    dictionary=current.dictionary if update.custom_dictionary is None else update.custom_dictionary,
                    language=current.language if update.language is None else update.language,
                    style=current.style if update.style is None else update.style)
            elif kind == "end" and session.started:
                await finish_utterance()
            elif kind == "cancel":
                session.buffer = bytearray()
                if session.detector:
                    session.detector.reset()
                await ws.send_json({"type": "cancelled"})
            else:
                await ws.send_json(error_frame(f"unexpected frame: {kind!r}", "unexpected_frame"))
    except WebSocketDisconnect:
        pass
    finally:
        queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker_task, timeout=5)
        except (asyncio.TimeoutError, Exception):                     # noqa: BLE001
            worker_task.cancel()
