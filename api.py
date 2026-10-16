# ---------------------------------------------------------------------------
# Standard library — ALL imports at the top (fixes Bug #8: import threading
# was previously declared mid-module after constants)
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from config import settings
from src.credit_manager import (
    AUTOCOMPLETE_CREDITS_SCALED,
    CREDIT_SCALE,
    IMAGE_CREDITS_SCALED,
    CreditManager,
    text_credits_scaled,
)
from src.redis_manager import get_redis_manager
from src.image_gen import sanitize_image_params
from src.text_processing_agent import (
    VISION_CAPABLE_PROVIDERS,
    get_autocomplete_suggestion,
    process_user_query,
    process_user_query_stream,
    supports_vision,
)
from src.webhook import (
    fire_credit_deduction,
    fire_credit_deduction_and_release,
    fire_prompt_history_save,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — derived from pydantic-settings singleton (Enhancement #3)
# ---------------------------------------------------------------------------
CREDIT_PRECHECK_ENABLED: bool = settings.credit_precheck_enabled
CREDIT_SETTLE_DELAY_SECONDS: float = settings.credit_settle_delay_seconds
MAX_RESERVATION_PER_REQUEST_SCALED: int = settings.max_reservation_per_request * CREDIT_SCALE
MAX_REFERENCE_IMAGES: int = settings.max_reference_images
CDN_URL: str = settings.cdn_url

DEFAULT_API_KEYS = {
    'openai_api_key': settings.default_openai_api_key,
    'anthropic_api_key': settings.default_anthropic_api_key,
    'grok_api_key': settings.default_grok_api_key,
    'gemini_api_key': settings.default_gemini_api_key,
    'deepseek_api_key': settings.default_deepseek_api_key,
}

# ---------------------------------------------------------------------------
# Supported providers / agent types
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ['openai', 'claude', 'grok', 'deepseek', 'gemini']

# Use a frozenset internally but keep a sorted list for deterministic error
# messages (fixes Bug #12: set gives non-deterministic ordering in error msgs)
VALID_AGENT_TYPES: frozenset[str] = frozenset({
    'fix_grammar', 'simplify', 'complete', 'emojify', 'translate',
    'make_longer', 'make_shorter', 'enhance_with_ai', 'change_tone',
    'general', 'generate_email_template', 'extract_image_text',
})
_SORTED_AGENT_TYPES = ', '.join(sorted(VALID_AGENT_TYPES))

# ---------------------------------------------------------------------------
# Cancellation registry (Lock-Free per-request threading.Event)
# Active entries have no expiry; pre-cancelled entries expire after the TTL.
# Eliminates global mutex contention in the hot streaming loop.
# ---------------------------------------------------------------------------
_active_cancellations: dict[str, tuple[threading.Event, Optional[float]]] = {}
_cancel_registry_lock = threading.Lock()
CANCEL_ENTRY_TTL = 300  # seconds before a stale entry is purged


def register_cancellation_event(request_id: str) -> threading.Event:
    """Register and return a new threading.Event for this request."""
    with _cancel_registry_lock:
        if request_id in _active_cancellations:
            existing_event, _ = _active_cancellations[request_id]
            _active_cancellations[request_id] = (existing_event, None)
            return existing_event
        event = threading.Event()
        _active_cancellations[request_id] = (event, None)
        return event


def cancel_request_by_id(request_id: str) -> bool:
    """Mark a request as cancelled by setting its event."""
    with _cancel_registry_lock:
        if request_id in _active_cancellations:
            event, _ = _active_cancellations[request_id]
            event.set()
        else:
            # Pre-set event in case cancel arrives before registration
            event = threading.Event()
            event.set()
            _active_cancellations[request_id] = (event, time.time())
    return True


def unregister_cancellation_event(request_id: str) -> None:
    """Clean up request event when stream completes."""
    with _cancel_registry_lock:
        _active_cancellations.pop(request_id, None)


def cleanup_stale_cancellations() -> None:
    """Purge stale cancellation records older than CANCEL_ENTRY_TTL."""
    now = time.time()
    with _cancel_registry_lock:
        stale = [
            key for key, (_, created_at) in _active_cancellations.items()
            if created_at is not None and now - created_at > CANCEL_ENTRY_TTL
        ]
        for k in stale:
            _active_cancellations.pop(k, None)
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale cancellation entries")


# ---------------------------------------------------------------------------
# Global service instances — populated by lifespan (Enhancement #4)
# ---------------------------------------------------------------------------
redis_manager = None
credit_manager = None


# ---------------------------------------------------------------------------
# LIFESPAN  (Enhancement #4)
# Replaces module-level initialisation so startup/shutdown are explicit and
# the app is easier to test (no side effects on import).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Redis, CreditManager lifecycle, and pre-warm AI models."""
    global redis_manager, credit_manager

    # ---- startup ----
    try:
        redis_manager = get_redis_manager()
        logger.info("✅ Redis manager initialised successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialise Redis manager: {e}")
        logger.error("⚠️  API will not be able to validate licences!")
        redis_manager = None

    if redis_manager is not None:
        try:
            credit_manager = CreditManager(redis_manager.redis_client)
            logger.info("✅ CreditManager initialised successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialise CreditManager: {e}")
            credit_manager = None

    # Pre-warm SDK model instances at startup (Bottleneck 4)
    try:
        from src.text_processing_agent import prewarm_models
        prewarm_count = prewarm_models(DEFAULT_API_KEYS)
        logger.info(f"✅ Pre-warmed {prewarm_count} model configurations")
    except Exception as e:
        logger.warning(f"Model pre-warming encountered an issue (non-fatal): {e}")

    yield

    # ---- shutdown ----
    if redis_manager is not None:
        try:
            redis_manager.close()
            logger.info("Redis connection closed on shutdown")
        except Exception:
            pass

    try:
        from src.text_processing_agent import close_shared_web_client
        close_shared_web_client()
        logger.info("Shared web client closed on shutdown")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(title="Text Processing Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """Ensure every response carries an X-Request-ID header."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    response = await call_next(request)
    if "X-Request-ID" not in response.headers:
        response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# EXCEPTION HANDLERS
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error = exc.errors()[0]
    loc = error.get("loc", [])
    field = loc[-1] if loc else "field"
    msg = error.get("msg", "Validation error")
    if msg == "Field required":
        error_msg = f"Missing required fields: {field}"
    else:
        error_msg = f"{field}: {msg}" if field != "body" else msg
    return JSONResponse(status_code=400, content={"error": error_msg})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---------------------------------------------------------------------------
# REQUEST MODELS
# max_length added to query/text_before fields (fixes missing input cap)
# VALID_AGENT_TYPES error message now deterministic (sorted list)
# ---------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    type: str = Field(..., min_length=1)
    query: str = Field(..., max_length=100_000)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    secondary_prompt: Optional[str] = ""

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v not in VALID_AGENT_TYPES:
            raise ValueError(f'Invalid type. Must be one of: {_SORTED_AGENT_TYPES}')
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v.lower() not in SUPPORTED_MODELS:
            raise ValueError(f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}')
        return v.lower()


class StreamRequest(BaseModel):
    type: str = Field(..., min_length=1)
    query: str = Field(..., max_length=100_000)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    secondary_prompt: Optional[str] = ""
    original_prompt: Optional[str] = ""
    image_urls: Optional[List[str]] = None
    # Scopes the "previous image" memory so a follow-up in one document never picks up
    # an image generated in another.
    document_id: Optional[str] = Field(None, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')
    # Explicit image settings; they outrank anything inferred from the prompt text.
    # Keys: aspect_ratio, quality, resolution, format, transparent_bg.
    image_params: Optional[Dict[str, Any]] = None

    @field_validator('image_params')
    @classmethod
    def validate_image_params(cls, v):
        try:
            return sanitize_image_params(v) or None
        except ValueError as e:
            raise ValueError(str(e))

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v not in VALID_AGENT_TYPES:
            raise ValueError(f'Invalid type. Must be one of: {_SORTED_AGENT_TYPES}')
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v.lower() not in SUPPORTED_MODELS:
            raise ValueError(f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}')
        return v.lower()


class AutocompleteRequest(BaseModel):
    text_before: str = Field(..., min_length=1, max_length=5_000)
    text_after: Optional[str] = ""
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v.lower() not in SUPPORTED_MODELS:
            raise ValueError(f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}')
        return v.lower()


class AbortRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# RESPONSE MODELS  (Enhancement #2)
# Typed so FastAPI generates correct OpenAPI schema and auto-validates output.
# ---------------------------------------------------------------------------
class ProcessResponse(BaseModel):
    success: bool
    markdown_response: str
    html_response: Optional[str] = None


class AutocompleteResponse(BaseModel):
    success: bool
    suggestion: str


class AbortResponse(BaseModel):
    success: bool


class HealthResponse(BaseModel):
    status: str
    redis: bool
    timestamp: float


class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

# Full query-prefix map — now includes ALL valid agent types (fixes Bug #9:
# generate_email_template and extract_image_text were missing, causing KeyError)
TYPE_PREFIX_MAP: dict[str, str] = {
    'fix_grammar':            'Fix grammar: ',
    'simplify':               'Simplify: ',
    'complete':               'Complete: ',
    'emojify':                'Emojify: ',
    'translate':              'Translate: ',
    'make_longer':            'Make longer: ',
    'make_shorter':           'Make shorter: ',
    'enhance_with_ai':        'Enhance: ',
    'change_tone':            'Change tone: ',
    'general':                '',
    'generate_email_template': '',
    'extract_image_text':     '',
}


def build_full_query(agent_type: str, user_query: str, secondary_prompt: str) -> str:
    """Build the full prompt string sent to the AI model."""
    if agent_type == 'enhance_with_ai' and secondary_prompt:
        return f"Enhance: {secondary_prompt}, Text: {user_query}"
    if agent_type == 'change_tone' and secondary_prompt:
        return f"Change tone to {secondary_prompt}: {user_query}"
    if agent_type == 'translate' and secondary_prompt:
        return f"Translate to {secondary_prompt}: {user_query}"
    return f"{TYPE_PREFIX_MAP[agent_type]}{user_query}"


def validate_reference_images(raw, request_id: str) -> list:
    """
    Validate the image_urls field.
    Returns a (possibly empty) list, or raises HTTPException on error.
    """
    if raw in (None, [], ""):
        return []

    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="image_urls must be an array of URLs")

    if len(raw) > MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_REFERENCE_IMAGES} reference images are allowed",
        )

    if not CDN_URL:
        logger.error(f"[{request_id}] CDN_URL is not configured — rejecting reference images")
        raise HTTPException(status_code=500, detail="Image uploads are not configured on this server")

    allowed_prefix = CDN_URL.rstrip("/") + "/"
    for url in raw:
        if not isinstance(url, str) or not url.startswith(allowed_prefix):
            logger.warning(f"[{request_id}] Rejected reference image from disallowed source")
            raise HTTPException(
                status_code=400,
                detail="Reference images must be uploaded through the editor before use",
            )

    return raw


async def async_validate_and_get_api_key(
    license_key: str,
    model_provider: str,
    request_id: str,
    background_tasks: Optional[BackgroundTasks] = None,
) -> tuple[str, dict]:
    """
    Validate license key and return (ai_api_key, license_meta) asynchronously.

    High Impact Latency Optimization:
    If user_id is in the license cache, validate_license() and get_user_api_keys()
    execute in parallel using asyncio.gather(), cutting ~5-12 ms off every request
    by overlapping the Redis network RTTs and avoiding sequential thread scheduling.
    """
    if not redis_manager:
        logger.error(f"[{request_id}] Redis manager not available")
        raise HTTPException(status_code=503, detail='License validation service unavailable')

    logger.info(f"[{request_id}] Validating licence key...")
    cached_user_id = redis_manager.get_cached_user_id(license_key)

    if cached_user_id:
        # Run license validation and user API key retrieval in parallel!
        license_task = run_in_threadpool(redis_manager.validate_license, license_key)
        keys_task = run_in_threadpool(redis_manager.get_user_api_keys, cached_user_id)
        license_result, user_keys = await asyncio.gather(license_task, keys_task)
    else:
        # First request on cold cache: validate license first, then retrieve keys
        license_result = await run_in_threadpool(redis_manager.validate_license, license_key)
        user_keys = None

    if not license_result['valid']:
        logger.warning(f"[{request_id}] ❌ Licence validation failed: {license_result['error']}")
        license_error_code = license_result.get('code')
        if license_error_code == 'LICENSE_SERVICE_UNAVAILABLE':
            raise HTTPException(status_code=503, detail=license_result['error'])
        error_msg = (
            license_result['error'] if license_error_code
            else f"Invalid license: {license_result['error']}"
        )
        detail = {'error': error_msg, 'code': license_error_code} if license_error_code else error_msg
        raise HTTPException(status_code=403, detail=detail)

    user_id = license_result['user_id']
    license_info = license_result.get('license', {})
    logger.info(f"[{request_id}] ✅ Licence valid for user: {user_id}")

    # If cold lookup or user_id changed, fetch keys
    if user_keys is None or user_id != cached_user_id:
        user_keys = await run_in_threadpool(redis_manager.get_user_api_keys, user_id)
    if user_keys.get('_error_code') == 'LICENSE_SERVICE_UNAVAILABLE':
        raise HTTPException(status_code=503, detail='API key service unavailable')

    try:
        if background_tasks:
            background_tasks.add_task(redis_manager.update_heartbeat, license_key)
        else:
            redis_manager.update_heartbeat(license_key)
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to update heartbeat: {e}")

    subscription_status = license_info.get('subscription_status', '')
    requires_user_api_keys = license_info.get('requires_user_api_keys', False)

    if subscription_status == 'past_due' and not requires_user_api_keys:
        logger.warning(f"[{request_id}] ❌ Blocked: subscription is past_due on managed plan")
        raise HTTPException(
            status_code=402,
            detail='Your payment is past due. Please update your payment method to continue using AI features.',
        )

    logger.info(f"[{request_id}] Resolving API key for model: {model_provider}")
    api_key = redis_manager.resolve_api_key_from_user_keys(
        user_keys=user_keys,
        model_provider=model_provider,
        default_keys=None if requires_user_api_keys else DEFAULT_API_KEYS,
    )

    if not api_key:
        logger.error(f"[{request_id}] ❌ No API key available for model: {model_provider}")
        raise HTTPException(
            status_code=400,
            detail=f'No API key configured for {model_provider}. Please add your API key in settings.',
        )

    logger.info(f"[{request_id}] ✅ API key retrieved successfully")

    license_meta = {
        'user_id': user_id,
        'requires_user_api_keys': requires_user_api_keys,
    }
    return api_key, license_meta


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get('/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and uptime monitors."""
    return HealthResponse(
        status="ok",
        redis=redis_manager is not None,
        timestamp=time.time(),
    )


@app.post('/agent/process', response_model=ProcessResponse)
async def process_text(
    request_data: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Non-streaming text processing endpoint.

    Fixes applied vs. original implementation:
      - Bug #1: api_key field is now treated as a LICENCE key; licence
        validation + API-key lookup run before any AI call.
      - Licence validation and API-key lookup stay off the event loop.
      - Bug #9: TYPE_PREFIX_MAP now covers all valid agent types.
      - Enhancement #2: returns X-Request-ID header.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== NEW REQUEST ==========")

    try:
        logger.info(f"[{request_id}] Type: {request_data.type}")
        logger.info(f"[{request_id}] Model: {request_data.model}")
        query_preview = (
            request_data.query[:100] + '...' if len(request_data.query) > 100
            else request_data.query
        )
        logger.info(f"[{request_id}] Query preview: {query_preview}")

        agent_type = request_data.type
        model_provider = request_data.model
        license_key = request_data.api_key

        # Validate licence and resolve the actual AI provider API key.
        # Uses async_validate_and_get_api_key to run Redis calls concurrently via asyncio.gather.
        api_key, license_meta = await async_validate_and_get_api_key(
            license_key, model_provider, request_id, background_tasks,
        )

        full_query = build_full_query(
            agent_type, request_data.query, request_data.secondary_prompt or ""
        )

        forced_agent_types = {
            'generate_email_template': 'email_template',
            'extract_image_text':      'text_extraction',
        }

        logger.info(f"[{request_id}] Processing: agent={agent_type}, model={model_provider}")

        ai_start = time.time()
        response = await run_in_threadpool(
            process_user_query,
            full_query,
            model_provider=model_provider,
            api_key=api_key,
            request_id=request_id,
            force_agent_type=forced_agent_types.get(agent_type),
            is_byok=license_meta['requires_user_api_keys'],
        )
        logger.info(f"[{request_id}] AI response received in {time.time() - ai_start:.2f}s")

        # Strip markdown code-fence wrappers the model sometimes emits
        if response.startswith('```markdown\n'):
            response = response[12:]
        elif response.startswith('```md\n'):
            response = response[6:]
        elif response.startswith('```\n'):
            response = response[4:]
        if response.endswith('\n```'):
            response = response[:-4]

        total_duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS — {total_duration:.2f}s")
        logger.info(f"[{request_id}] ================================")

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'markdown_response': response,
                'html_response': response,
            },
            headers={"X-Request-ID": request_id},
        )

    except HTTPException:
        raise  # let FastAPI's exception handler format the response
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] FAILED — {e} ({total_duration:.2f}s)")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)},
            headers={"X-Request-ID": request_id},
        )


@app.post('/agent/stream')
async def stream_text(
    request_data: StreamRequest,
    background_tasks: BackgroundTasks,
):
    """
    Streaming endpoint — returns Server-Sent Events with markdown chunks.

    Licence validation and API-key lookup stay off the event loop.
    Enhancement #2: X-Request-ID added to SSE response headers.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== NEW STREAM REQUEST ==========")

    try:
        logger.info(f"[{request_id}] Type: {request_data.type}")
        logger.info(f"[{request_id}] Model: {request_data.model}")
        query_preview = (
            request_data.query[:100] + '...' if len(request_data.query) > 100
            else request_data.query
        )
        logger.info(f"[{request_id}] Query preview: {query_preview}")

        agent_type = request_data.type
        model_provider = request_data.model
        license_key = request_data.api_key

        # Validate licence and resolve the actual AI provider API key.
        # Uses async_validate_and_get_api_key to run Redis calls concurrently via asyncio.gather (Point 1).
        api_key, license_meta = await async_validate_and_get_api_key(
            license_key, model_provider, request_id, background_tasks,
        )

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']
        original_prompt = (
            request_data.original_prompt.strip() if request_data.original_prompt else ''
        )

        image_urls = validate_reference_images(request_data.image_urls, request_id)

        if image_urls and not supports_vision(model_provider):
            logger.warning(f"[{request_id}] {model_provider} cannot accept image input")
            usable = ", ".join(sorted(VISION_CAPABLE_PROVIDERS))
            raise HTTPException(
                status_code=400,
                detail={
                    'error': f"The '{model_provider}' model cannot read images. Switch to one of: {usable}.",
                    'code': 'VISION_NOT_SUPPORTED',
                },
            )

        forced_agent_types = {
            'generate_email_template': 'email_template',
            'extract_image_text':      'text_extraction',
        }

        full_query = build_full_query(
            agent_type, request_data.query, request_data.secondary_prompt or ""
        )
        input_chars = len(request_data.query)

        # Upfront credit reservation (skipped for BYOK users)
        upfront_reserved_scaled = 0
        if CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
            upfront_reserved_scaled = text_credits_scaled(input_chars, 0)
            if image_urls:
                upfront_reserved_scaled += IMAGE_CREDITS_SCALED * len(image_urls)
            ok, available_scaled = credit_manager.reserve(user_id, upfront_reserved_scaled)
            if not ok:
                logger.warning(
                    f"[{request_id}] Insufficient credits: "
                    f"needed={upfront_reserved_scaled} available={available_scaled}"
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "Insufficient credits",
                        "code": "INSUFFICIENT_CREDITS",
                        "available": available_scaled,
                        "needed": upfront_reserved_scaled,
                    },
                )

        async def generate():
            output_chars = 0
            had_error = False
            error_code = None
            pending_images: dict = {}
            resolved_image_count = 0
            extra_reserved_scaled = 0
            credit_limit_hit = False

            # Register lock-free cancellation event for this request (Point 6)
            cancel_event = register_cancellation_event(request_id)
            completion_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            def notify_image_completion() -> None:
                try:
                    loop.call_soon_threadsafe(completion_event.set)
                except RuntimeError:
                    pass

            try:
                yield f"data: {json.dumps({'type': 'start', 'request_id': request_id})}\n\n"

                next_reserve_threshold = (upfront_reserved_scaled // CREDIT_SCALE) * 1000

                async for chunk in iterate_in_threadpool(process_user_query_stream(
                    full_query,
                    model_provider=model_provider,
                    api_key=api_key,
                    request_id=request_id,
                    user_id=user_id,
                    credit_manager=credit_manager if CREDIT_PRECHECK_ENABLED else None,
                    is_byok=is_byok,
                    image_urls=image_urls,
                    force_agent_type=forced_agent_types.get(agent_type),
                    image_completion_callback=notify_image_completion,
                    # Only free-form chat may be routed to image generation by regex;
                    # text operations carry user text that can mention images freely.
                    image_routing=(agent_type == 'general'),
                    document_id=request_data.document_id,
                    image_params=request_data.image_params,
                )):
                    # Lock-free cancellation check on every streamed token (Point 6)
                    if cancel_event.is_set():
                        logger.info(
                            f"[{request_id}] Generator stopped by client abort "
                            f"(output_chars={output_chars})"
                        )
                        break

                    if isinstance(chunk, dict) and "__pending_images__" in chunk:
                        pending_images = chunk["__pending_images__"]
                        continue

                    output_chars += len(chunk)

                    # Per-1000-char incremental reservation (non-BYOK only)
                    if CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
                        while (input_chars + output_chars) > next_reserve_threshold:
                            if (
                                upfront_reserved_scaled + extra_reserved_scaled
                                >= MAX_RESERVATION_PER_REQUEST_SCALED
                            ):
                                credit_limit_hit = True
                                break
                            ok, _ = credit_manager.reserve(user_id, CREDIT_SCALE)
                            if not ok:
                                credit_limit_hit = True
                                break
                            extra_reserved_scaled += CREDIT_SCALE
                            next_reserve_threshold += 1000

                        if credit_limit_hit:
                            yield f"data: {json.dumps({'type': 'credit_limit_reached', 'output_chars': output_chars})}\n\n"
                            break

                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                # Done event
                total_duration = time.time() - start_time
                actual_images = {k: v for k, v in pending_images.items() if k != "__completion_callback__"}
                pending_count = len(actual_images)
                logger.info(
                    f"[{request_id}] STREAM SUCCESS — {total_duration:.2f}s, "
                    f"pending images: {pending_count}"
                )
                yield f"data: {json.dumps({'type': 'done', 'duration': round(total_duration, 2), 'pending_images': pending_count})}\n\n"

                # Wait for background image completions via event notification (Point 7)
                if pending_count > 0:
                    resolved: set[str] = set()
                    poll_start = time.time()
                    max_wait = 120  # seconds

                    while len(resolved) < pending_count and (time.time() - poll_start) < max_wait:
                        if cancel_event.is_set():
                            logger.info(
                                f"[{request_id}] Image waiting stopped by client abort "
                                f"({len(resolved)}/{pending_count} resolved)"
                            )
                            break

                        for pid, info in actual_images.items():
                            if pid in resolved:
                                continue
                            if info["status"] == "done":
                                resolved.add(pid)
                                resolved_image_count += 1
                                logger.info(
                                    f"[{request_id}] Image ready: {pid} -> {info['cdn_url']} "
                                    f"({info.get('width', '?')}x{info.get('height', '?')}, {info.get('format', '?')})"
                                )
                                ready_payload = {
                                    'type': 'image_ready',
                                    'placeholder_id': pid,
                                    'url': info['cdn_url'],
                                    'prompt': info.get('prompt', ''),
                                }
                                if info.get('width'):
                                    ready_payload['width'] = info['width']
                                if info.get('height'):
                                    ready_payload['height'] = info['height']
                                if info.get('format'):
                                    ready_payload['format'] = info['format']
                                yield f"data: {json.dumps(ready_payload)}\n\n"
                            elif info["status"] == "failed":
                                resolved.add(pid)
                                logger.error(f"[{request_id}] Image failed: {pid} -> {info['error']}")
                                yield f"data: {json.dumps({'type': 'image_failed', 'placeholder_id': pid, 'error': info['error']})}\n\n"

                        if len(resolved) >= pending_count:
                            break

                        # Event notification wait: unblocks immediately (<1ms) when worker signals set()
                        remaining = max_wait - (time.time() - poll_start)
                        if remaining <= 0:
                            break
                        wait_slice = min(remaining, 1.5)
                        completion_event.clear()
                        if any(
                            pid not in resolved and info["status"] in ("done", "failed")
                            for pid, info in actual_images.items()
                        ):
                            continue
                        try:
                            await asyncio.wait_for(completion_event.wait(), timeout=wait_slice)
                        except asyncio.TimeoutError:
                            pass

                    # Timeout any still-pending images
                    for pid in actual_images:
                        if pid not in resolved:
                            logger.error(f"[{request_id}] Image timed out: {pid}")
                            yield f"data: {json.dumps({'type': 'image_failed', 'placeholder_id': pid, 'error': 'Image generation timed out'})}\n\n"

                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

            except Exception as e:
                had_error = True
                error_code = type(e).__name__
                logger.error(f"[{request_id}] STREAM FAILED — {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            except GeneratorExit:
                unregister_cancellation_event(request_id)
                logger.info(
                    f"[{request_id}] Generator closed by client disconnect "
                    f"(output_chars={output_chars})"
                )

            finally:
                unregister_cancellation_event(request_id)
                duration_ms = int((time.time() - start_time) * 1000)

                if not had_error and not is_byok:
                    if resolved_image_count > 0 and output_chars > 0:
                        op_type = 'mixed'
                    elif resolved_image_count > 0:
                        op_type = 'image'
                    else:
                        op_type = 'text'

                    if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                        total_reserved_scaled = (
                            upfront_reserved_scaled
                            + extra_reserved_scaled
                            + resolved_image_count * IMAGE_CREDITS_SCALED
                        )
                        fire_credit_deduction_and_release(
                            user_id=user_id,
                            input_chars=input_chars,
                            output_chars=output_chars,
                            operation_type=op_type,
                            image_count=resolved_image_count,
                            reserved_scaled=total_reserved_scaled,
                            credit_manager=credit_manager,
                            settle_delay_seconds=CREDIT_SETTLE_DELAY_SECONDS,
                            provider=model_provider,
                            model=None,
                            duration_ms=duration_ms,
                            success=True,
                        )
                    else:
                        fire_credit_deduction(
                            user_id, input_chars, output_chars, op_type,
                            resolved_image_count,
                            provider=model_provider, model=None,
                            duration_ms=duration_ms, success=True,
                        )

                elif had_error and not is_byok:
                    if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                        total_reserved_scaled = (
                            upfront_reserved_scaled
                            + extra_reserved_scaled
                            + resolved_image_count * IMAGE_CREDITS_SCALED
                        )
                        credit_manager.schedule_release(
                            user_id, total_reserved_scaled, delay_seconds=0,
                        )
                    fire_credit_deduction(
                        user_id, input_chars, output_chars, 'text',
                        resolved_image_count,
                        provider=model_provider, model=None,
                        duration_ms=duration_ms, success=False, error_code=error_code,
                    )

                if not had_error and original_prompt:
                    fire_prompt_history_save(user_id, original_prompt)

        return StreamingResponse(
            generate(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'X-Request-ID': request_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] STREAM FAILED — {e} ({total_duration:.2f}s)")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)},
            headers={"X-Request-ID": request_id},
        )


@app.post('/agent/autocomplete', response_model=AutocompleteResponse)
async def autocomplete_text(
    request_data: AutocompleteRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Inline text autocomplete — returns a short predicted continuation.

    Bug #2 fix: user_id, is_byok, and autocomplete_reserved are now
    initialised BEFORE the try block so the except handler can safely
    reference them even if validate_and_get_api_key raises.

    Licence validation and API-key lookup stay off the event loop.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== AUTOCOMPLETE REQUEST ==========")

    # Initialise before try so the except block can reference them safely.
    # (Bug #2 fix)
    user_id: Optional[str] = None
    is_byok: bool = False
    model_provider: str = request_data.model  # already validated by Pydantic
    autocomplete_reserved: bool = False

    try:
        text_before = request_data.text_before
        text_after = request_data.text_after or ""
        license_key = request_data.api_key

        # Validate licence and resolve API key concurrently via asyncio.gather (Point 1)
        api_key, license_meta = await async_validate_and_get_api_key(
            license_key, model_provider, request_id, background_tasks,
        )

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']

        if CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
            ok, available = credit_manager.reserve(user_id, AUTOCOMPLETE_CREDITS_SCALED)
            if not ok:
                logger.warning(
                    f"[{request_id}] Autocomplete blocked: insufficient credits "
                    f"(available={available})"
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "Insufficient credits",
                        "code": "INSUFFICIENT_CREDITS",
                        "available": available,
                        "needed": AUTOCOMPLETE_CREDITS_SCALED,
                    },
                )
            autocomplete_reserved = True

        # Truncate context to keep requests fast
        text_before = text_before[-500:]
        text_after = text_after[:200] if text_after else ""

        logger.info(
            f"[{request_id}] Autocomplete: model={model_provider}, "
            f"context_before={len(text_before)} chars"
        )

        suggestion = await run_in_threadpool(
            get_autocomplete_suggestion,
            text_before=text_before,
            text_after=text_after,
            model_provider=model_provider,
            api_key=api_key,
            request_id=request_id,
        )

        total_duration = time.time() - start_time
        logger.info(
            f"[{request_id}] Autocomplete SUCCESS — {total_duration:.2f}s, "
            f"suggestion_len={len(suggestion)}"
        )

        duration_ms = int((time.time() - start_time) * 1000)
        if not is_byok:
            if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                if suggestion:
                    fire_credit_deduction_and_release(
                        user_id=user_id,
                        input_chars=0,
                        output_chars=0,
                        operation_type="autocomplete",
                        image_count=0,
                        reserved_scaled=AUTOCOMPLETE_CREDITS_SCALED,
                        credit_manager=credit_manager,
                        settle_delay_seconds=CREDIT_SETTLE_DELAY_SECONDS,
                        provider=model_provider,
                        model=None,
                        duration_ms=duration_ms,
                        success=True,
                    )
                    logger.info(f"[{request_id}] Autocomplete credit deduction fired for user={user_id}")
                else:
                    # Empty suggestion — release reservation without deducting
                    credit_manager.schedule_release(
                        user_id, AUTOCOMPLETE_CREDITS_SCALED, delay_seconds=0,
                    )
                autocomplete_reserved = False
            elif suggestion:
                fire_credit_deduction(
                    user_id, 0, 0, 'autocomplete',
                    provider=model_provider, model=None,
                    duration_ms=duration_ms, success=True,
                )
                logger.info(f"[{request_id}] Autocomplete credit deduction fired for user={user_id}")

        return JSONResponse(
            status_code=200,
            content={'success': True, 'suggestion': suggestion},
            headers={"X-Request-ID": request_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] Autocomplete FAILED — {total_duration:.2f}s: {e}")

        # Release any held reservation so the user isn't blocked until TTL expires
        try:
            if autocomplete_reserved and credit_manager is not None and user_id:
                credit_manager.schedule_release(
                    user_id, AUTOCOMPLETE_CREDITS_SCALED, delay_seconds=0,
                )
        except Exception:
            pass

        # Log failed telemetry
        try:
            if user_id and model_provider and not is_byok:
                fire_credit_deduction(
                    user_id, 0, 0, 'autocomplete',
                    provider=model_provider, model=None,
                    duration_ms=int(total_duration * 1000),
                    success=False, error_code=type(e).__name__,
                )
        except Exception:
            pass

        return JSONResponse(
            status_code=500,
            content={'error': str(e)},
            headers={"X-Request-ID": request_id},
        )


@app.post('/agent/abort', response_model=AbortResponse)
async def abort_request(request_data: AbortRequest) -> JSONResponse:
    """Cancel an in-flight stream request so its generator exits early."""
    req_id = request_data.request_id
    license_key = request_data.api_key

    if not license_key or not redis_manager:
        raise HTTPException(status_code=401, detail='Unauthorized')

    license_result = await run_in_threadpool(redis_manager.validate_license, license_key)
    if not license_result['valid']:
        if license_result.get('code') == 'LICENSE_SERVICE_UNAVAILABLE':
            raise HTTPException(status_code=503, detail=license_result['error'])
        raise HTTPException(status_code=401, detail='Unauthorized')

    cleanup_stale_cancellations()
    cancel_request_by_id(req_id)
    logger.info(f"[{req_id}] Request cancelled by client")
    return JSONResponse(
        status_code=200,
        content={'success': True},
        headers={"X-Request-ID": req_id},
    )


if __name__ == '__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=False)
