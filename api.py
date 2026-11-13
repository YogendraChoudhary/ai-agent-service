# ---------------------------------------------------------------------------
# Standard library — ALL imports at the top (fixes Bug #8: import threading
# was previously declared mid-module after constants)
# ---------------------------------------------------------------------------
import asyncio
import base64
import binascii
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from config import settings
from src.blog.plan import BRIEF_MAX_CHARS
from src.charts.router import router as chart_router
from src.charts.schema import GraphSuggestion
from src.document_edit.router import router as document_router
from src.document_upload.router import router as document_upload_router
from src.document_import.router import router as document_import_router
from src.document_edit.schema import DocumentEditResult
from src.document_context import AnalysisMode, DocumentAction, DocumentActionRequest, DocumentScopeRequest, DocumentContext, ContextMetadata, CONTEXT_ERROR, prepare_context, source_index
from src.document_analysis import AnalysisResult
from src.document_intent import IntentSuggestion, suggest_document_intent
from src.key_points import HighlightResult, KeyPointValidationError, MAX_TEXT_CHARS
from src.code_validate import CodeValidationError
from src.credit_manager import (
    AUTOCOMPLETE_CREDITS_SCALED,
    CREDIT_SCALE,
    IMAGE_CREDITS_SCALED,
    CreditManager,
    design_credits_scaled,
    text_credits_scaled,
)
from src.redis_manager import get_redis_manager
from src.image_gen import sanitize_image_params
from src.text_processing_agent import (
    FREE_FORM_AGENT,
    VISION_CAPABLE_PROVIDERS,
    agent_for_request_type,
    get_autocomplete_suggestion,
    process_user_query,
    process_user_query_with_images,
    process_user_query_stream,
    supports_vision,
)
from src.observability import (
    get_trace,
    finish_trace,
    log_request_body,
    log_response_body,
    start_trace,
    summarize_images,
    trace_mark,
    trace_set,
)
from src.webhook import (
    fire_credit_deduction,
    fire_credit_deduction_and_release,
    fire_prompt_history_save,
    start_billing_delivery,
    stop_billing_delivery,
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
    'generate_code', 'explain_code', 'generate_blog',
    'generate_design', 'edit_design', 'highlight_key_points',
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

    if not settings.agent_backend_secret:
        logger.warning("⚠️  AGENT_BACKEND_SECRET is not set: /agent/process accepts a provider key from any caller. "
                       "Set the same value on the backend (X-Agent-Backend-Secret header) and here to restrict it.")

    if redis_manager is not None:
        try:
            credit_manager = CreditManager(redis_manager.redis_client)
            start_billing_delivery(redis_manager.redis_client)
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

    from src.models import MODEL_ENV_OVERRIDES
    if MODEL_ENV_OVERRIDES:
        logger.warning("Model overrides from the environment are active: %s", MODEL_ENV_OVERRIDES)
    if settings.model_preflight_on_start:
        import threading
        from src.model_preflight import run_preflight
        threading.Thread(target=run_preflight, name="model-preflight", daemon=True).start()

    if settings.voice_flow_enabled:
        # Whisper's first decode compiles kernels and Gemini's first call opens a TLS connection; do both before a user waits.
        from src.voice_flow.router import warm_up as warm_up_voice
        await warm_up_voice()

    yield

    # ---- shutdown ----
    stop_billing_delivery()
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
app.include_router(chart_router)
app.include_router(document_router)
app.include_router(document_upload_router)
app.include_router(document_import_router)

if settings.voice_flow_enabled:       # /v1/voice/*: local Whisper dictation with licence validation and text billing
    from src.voice_flow.access import LicenceGate
    from src.voice_flow.router import router as voice_router, set_gate
    set_gate(LicenceGate(sys.modules[__name__]))      # licence key + credits, like /agent/stream
    app.include_router(voice_router)

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
    t0 = time.perf_counter()
    response = await call_next(request)
    if "X-Request-ID" not in response.headers:
        response.headers["X-Request-ID"] = request_id
    # For streams this is time until the response headers are ready (time-to-first-byte), not the whole stream.
    response.headers["X-Response-Time-Ms"] = str(int((time.perf_counter() - t0) * 1000))
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
        return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


# ---------------------------------------------------------------------------
# REQUEST MODELS
# max_length added to query/text_before fields (fixes missing input cap)
# VALID_AGENT_TYPES error message now deterministic (sorted list)
# ---------------------------------------------------------------------------
class BlogOptions(BaseModel):
    """Optional knobs for type=generate_blog. Every field is optional; a misspelt one is an error."""
    model_config = {'extra': 'forbid'}
    length: Optional[Literal['short', 'standard', 'long']] = None     # ~800 / ~1,450 / ~2,200 words
    images: Optional[Union[Annotated[int, Field(ge=0, le=10)], Literal['auto']]] = None
    tone: Optional[str] = Field(None, min_length=1, max_length=60)
    audience: Optional[str] = Field(None, min_length=1, max_length=200)


def check_blog_fields(request_type: str, query: str, blog_options: Optional[BlogOptions]) -> None:
    """generate_blog takes a brief of bounded size; blog_options belongs to generate_blog and nothing else."""
    if request_type == 'generate_blog':
        if len(query.strip()) < 3:
            raise ValueError('query must contain a topic or brief for the blog post')
        if len(query) > BRIEF_MAX_CHARS:
            raise ValueError(f'query is too long for a blog brief (at most {BRIEF_MAX_CHARS} characters)')
    elif blog_options is not None:
        raise ValueError('blog_options is only valid with type generate_blog')


def check_document_fields(request) -> None:
    if request.document_context is None and (request.document_action is not None or request.document_scope is not None or request.document_analysis is not None or request.suggest_document_intent):
        raise ValueError('Document actions, scope, analysis and intent suggestions require document_context')
    if request.document_context is not None:
        if request.document_text is not None:
            raise ValueError('Supply only one of document_context and document_text')
        if request.type != 'general':
            raise ValueError('document_context is only valid with type general')
        action, scope = request.document_action or 'ask', request.document_scope or 'document'
        if scope != 'document' and action not in ('ask', 'auto'):      # auto resolves such a request to Ask
            raise ValueError('Scoped retrieval is only available for Ask')
        if scope == 'selection' and (request.document_context.selection is None or request.document_context.selection.start_utf16 == request.document_context.selection.end_utf16):
            raise ValueError('Selection scope requires a nonempty selection')
        if request.document_analysis is not None and (action not in ('ask', 'auto') or scope != 'document'):
            raise ValueError('Analysis requires Ask with document scope')
        limit = 2000 if request.document_action == 'edit' else 8000
        if not request.query.strip() or len(request.query) > limit or request.secondary_prompt:
            raise ValueError(f'With document_context, query must contain 1–{limit} characters; omit secondary_prompt')
        try:
            request.query.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError('query must contain valid Unicode') from None
        if '\x00' in request.query:
            raise ValueError('query must not contain NUL characters')
        if request.document_analysis is not None and len(json.dumps(request.query, ensure_ascii=False).encode('utf-8')) > 2000:
            raise ValueError('An analysis instruction must fit 2,000 serialized UTF-8 bytes')
        if request.existing_design_ast is not None or getattr(request, 'image_urls', None) or request.image_params or request.image_provider or request.image_model or request.blog_options is not None:
            raise ValueError('Document context does not accept design, image or blog inputs')
        if request.document_action == 'edit':
            from src.document_edit.blocks import document_blocks
            document_blocks(request.document_context.text)
        return
    if request.document_text is None:
        return
    if request.type not in ('general', 'enhance_with_ai'):
        raise ValueError('document_text is only valid with type general or enhance_with_ai')
    if not request.query.strip() or len(request.query) > 2000 or request.secondary_prompt:
        raise ValueError('With document_text, query must be the editing instruction (1–2,000 characters); omit secondary_prompt')
    if request.existing_design_ast is not None or getattr(request, 'image_urls', None) or request.image_params or request.image_provider or request.image_model:
        raise ValueError('Document edits do not accept design or image-generation inputs')
    from src.document_edit.blocks import document_blocks
    document_blocks(request.document_text)


def check_highlight_fields(request) -> None:
    if request.type != 'highlight_key_points':
        return
    if not request.query.strip() or len(request.query) > MAX_TEXT_CHARS:
        raise ValueError(f'highlight_key_points needs a non-empty text portion of at most {MAX_TEXT_CHARS} characters')
    request.query.encode('utf-8')
    if request.secondary_prompt or request.existing_design_ast is not None or getattr(request, 'image_urls', None) or request.image_params or request.image_provider or request.image_model:
        raise ValueError('highlight_key_points accepts text only; omit secondary_prompt and image/design inputs')


def write_needs_document_for(request) -> Optional[dict]:
    """Write mode (type general, nothing but the prompt) that refers to a document it did not send; see document_intent."""
    if (request.type != 'general' or request.document_context is not None or request.document_text is not None
            or request.existing_design_ast is not None or getattr(request, 'image_urls', None)):
        return None
    from src.document_intent import write_needs_document
    return write_needs_document(request.query)


def chart_request_for(request) -> Optional[dict]:
    """
    Enhance with an instruction that explicitly asks for a chart ("improve this and add a bar chart"): the enhanced
    prose is followed by exactly one graph_suggestion or graph_suggestion_unavailable, never a table. None keeps the
    optional suggestion. Write mode (general) keeps its Mermaid output and other types never chart (editor contract).
    """
    if (request.type != 'enhance_with_ai' or not request.suggest_graphs or not settings.chart_suggestions_enabled
            or not request.secondary_prompt or getattr(request, 'image_urls', None)):
        return None
    from src.charts.request import asks_for_chart, requested_chart_type
    if not asks_for_chart(request.secondary_prompt, allow_writing=True):
        return None
    return {'chart_type': requested_chart_type(request.secondary_prompt)}


async def document_request_inputs(request):
    if request.document_context is None:
        return request.document_text, None
    if not settings.document_context_enabled:
        raise HTTPException(status_code=503, detail={'error': 'Document context is disabled', 'code': 'DOCUMENT_CONTEXT_DISABLED'})
    if request.document_action == 'edit':
        return request.document_context.text, None
    prepared = await run_in_threadpool(prepare_context, request.document_context, request.query, request.document_action or 'ask',
                                     scope=request.document_scope or 'document', analysis=request.document_analysis)
    return None, prepared


def auto_action_by_rules(request):
    """document_action "auto" decided by the request and its wording alone (no model, no key), or None when that is not enough."""
    from src.document_action import decide_by_rules
    if not settings.document_context_enabled:
        # Same answer an explicit action gets, and before any classifier call is spent on a request that cannot run.
        raise HTTPException(status_code=503, detail={'error': 'Document context is disabled', 'code': 'DOCUMENT_CONTEXT_DISABLED'})
    return decide_by_rules(request.query, request.document_context.text, scope=request.document_scope or 'document',
                           analysis=request.document_analysis)


def classify_auto_action(request, *, provider, api_key, is_byok, platform_key, request_id):
    """The light-tier decision for an "auto" request the rules could not settle. Blocking: run it off the event loop."""
    from src.document_action import decide_with_model
    from src.intent_classifier import classify_label
    selection = request.document_context.selection
    has_selection = bool(selection and selection.start_utf16 != selection.end_utf16)

    def classify(text, system, labels):
        return classify_label(text, provider, api_key, system=system, labels=labels, is_byok=is_byok, platform_key=platform_key,
                              timeout=settings.document_action_classifier_timeout_seconds, request_id=request_id)
    return decide_with_model(request.query, request.document_context.text, has_selection=has_selection, classify=classify)


def request_for_action(request, decision):
    """
    The request the pipeline runs once "auto" has been decided. Ask, Draft and Edit keep the document and gain the chosen action; Write
    is the plain prompt, so the document is not sent to it. An Edit gets the checks an explicit Edit gets at validation time.
    """
    if decision.action == 'write':
        return request.model_copy(update={'document_action': None, 'document_context': None, 'document_scope': None,
                                          'document_analysis': None, 'suggest_document_intent': False})
    if decision.action == 'edit':
        if len(request.query) > 2000:
            raise HTTPException(status_code=400, detail='With document_context, query must contain 1–2000 characters for Edit; omit secondary_prompt')
        from src.document_edit.blocks import document_blocks
        try:
            document_blocks(request.document_context.text)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
    # The decision event replaces the advisory suggestion: there is nothing left to suggest.
    return request.model_copy(update={'document_action': decision.action, 'suggest_document_intent': False})


class ProcessRequest(BaseModel):
    type: str = Field(..., min_length=1)
    query: str = Field(..., max_length=100_000)
    model: str = Field(..., min_length=1)
    model_source: Literal['default', 'user'] = 'user'
    api_key: str = Field(..., min_length=1)
    secondary_prompt: Optional[str] = ""
    image_provider: Optional[str] = Field(None, max_length=32)
    image_model: Optional[str] = Field(None, max_length=64)
    image_params: Optional[Dict[str, Any]] = None
    blog_options: Optional[BlogOptions] = None      # generate_blog only
    existing_design_ast: Optional[Union[Dict[str, Any], str]] = None
    suggest_graphs: bool = True       # Enhance only; also requires CHART_SUGGESTIONS_ENABLED
    document_text: Optional[str] = Field(default=None, min_length=1, max_length=100_000)
    document_context: Optional[DocumentContext] = None
    document_action: Optional[DocumentActionRequest] = None
    document_scope: Optional[DocumentScopeRequest] = None
    document_analysis: Optional[AnalysisMode] = None
    suggest_document_intent: bool = Field(False, strict=True)

    @model_validator(mode='before')
    @classmethod
    def validate_context_attachments(cls, values):
        # Process has no image_urls field; do not silently discard attachments on the new path.
        if isinstance(values, dict) and values.get('document_context') is not None and values.get('image_urls'):
            raise ValueError('Document context does not accept image attachments')
        return values

    @field_validator('image_provider')
    @classmethod
    def validate_image_provider(cls, v):
        if v is not None and v.lower() not in ('gemini', 'openai', 'grok'):
            raise ValueError('Invalid image_provider. Must be one of: gemini, openai, grok')
        return v.lower() if v is not None else None

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

    @model_validator(mode='after')
    def validate_blog_fields(self):
        check_blog_fields(self.type, self.query, self.blog_options)
        check_document_fields(self)
        check_highlight_fields(self)
        return self


class StreamRequest(BaseModel):
    type: str = Field(..., min_length=1)
    query: str = Field(..., max_length=100_000)
    model: str = Field(..., min_length=1)
    model_source: Literal['default', 'user'] = 'user'
    api_key: str = Field(..., min_length=1)
    secondary_prompt: Optional[str] = ""
    original_prompt: Optional[str] = ""
    image_urls: Optional[List[str]] = None
    # Scopes the "previous image" memory so a follow-up in one document never picks up
    # an image generated in another.
    document_id: Optional[str] = Field(None, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')
    image_provider: Optional[str] = Field(None, max_length=32)
    image_model: Optional[str] = Field(None, max_length=64)
    # Explicit image settings; they outrank anything inferred from the prompt text.
    # Keys: aspect_ratio, quality, resolution, format, transparent_bg, provider, model.
    image_params: Optional[Dict[str, Any]] = None
    blog_options: Optional[BlogOptions] = None      # generate_blog only
    existing_design_ast: Optional[Union[Dict[str, Any], str]] = None
    suggest_graphs: bool = True
    document_text: Optional[str] = Field(default=None, min_length=1, max_length=100_000)
    document_context: Optional[DocumentContext] = None
    document_action: Optional[DocumentActionRequest] = None
    document_scope: Optional[DocumentScopeRequest] = None
    document_analysis: Optional[AnalysisMode] = None
    suggest_document_intent: bool = Field(False, strict=True)

    @field_validator('image_provider')
    @classmethod
    def validate_image_provider(cls, v):
        if v is not None and v.lower() not in ('gemini', 'openai', 'grok'):
            raise ValueError('Invalid image_provider. Must be one of: gemini, openai, grok')
        return v.lower() if v is not None else None

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

    @model_validator(mode='after')
    def validate_blog_fields(self):
        check_blog_fields(self.type, self.query, self.blog_options)
        check_document_fields(self)
        check_highlight_fields(self)
        return self


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


_REF_KEY = r'^[a-z0-9_]{1,40}$'


class EmailThemeRequest(BaseModel):
    """Look of the email, picked by the user (or their brand kit). Passed through to the template for the renderer."""
    model_config = {'extra': 'forbid'}
    accent_color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')
    hero: Optional[Dict[str, Any]] = None     # {"style": light|dark|brand|custom, "background"?, "text"?}
    footer: Optional[Dict[str, Any]] = None

    @field_validator('hero', 'footer')
    @classmethod
    def validate_section(cls, v):
        if v is None:
            return v
        from src.email_template.schema import SectionTheme
        return SectionTheme.model_validate(v).model_dump(exclude_none=True)


class EmailGenerateRequest(BaseModel):
    model_config = {'extra': 'forbid'}   # a misspelt field (socialLinks) is an error, not silently ignored
    model: str = Field(..., min_length=1)
    model_source: Literal['default', 'user'] = 'user'
    api_key: str = Field(..., min_length=1)
    brief: str = Field(..., min_length=3, max_length=5000)
    category: Optional[str] = Field(None, max_length=40)       # omitted -> routed from the brief
    tone: Optional[str] = Field(None, pattern=r'^(formal|professional|professional_warm)$')
    audience: Optional[str] = Field(None, max_length=300)
    language: str = Field('en-GB', pattern=r'^[a-z]{2,3}(-[A-Z]{2})?$')
    length: str = Field('standard', pattern=r'^(short|standard|detailed)$')
    allowed_merge_tags: List[str] = Field(default_factory=list, max_length=50)
    links: Dict[str, str] = Field(default_factory=dict)        # key -> https URL; only keys reach the model
    assets: Dict[str, str] = Field(default_factory=dict)       # key -> short description
    sender: Dict[str, str] = Field(default_factory=dict)       # name, title, org, contact
    org_name: Optional[str] = Field(None, max_length=80)
    brand_kit_id: Optional[str] = Field(None, max_length=64)
    variants: int = Field(1, ge=1, le=3)
    # omitted -> category default; social_links imply "none" (signature and social icons are exclusive)
    sign_off: Optional[str] = Field(None, pattern=r'^(signature|none)$')
    social_links: Dict[str, str] = Field(default_factory=dict)            # platform -> URL, max 4, footer order
    theme: Optional[EmailThemeRequest] = None                             # chosen by the user, never the model

    @field_validator('brief')
    @classmethod
    def validate_brief(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError('brief must contain at least 3 characters of text')
        return v

    @field_validator('social_links')
    @classmethod
    def validate_social_links(cls, v):
        from src.email_template.schema import SOCIAL_MAX, SOCIAL_PLATFORMS
        if len(v) > SOCIAL_MAX:
            raise ValueError(f'At most {SOCIAL_MAX} social links')
        for platform, url in v.items():
            if platform not in SOCIAL_PLATFORMS:
                raise ValueError(f"Unknown social platform '{platform}'. Use: {', '.join(SOCIAL_PLATFORMS)}")
            ok = re.fullmatch(r'mailto:[^\s<>"@]+@[^\s<>"]+', url) if platform == 'email' \
                else re.fullmatch(r'https://[^\s<>"]{1,2000}', url)
            if not ok:
                raise ValueError(f"Social link '{platform}' must be an https URL (mailto: for email)")
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v.lower() not in SUPPORTED_MODELS:
            raise ValueError(f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}')
        return v.lower()

    @field_validator('allowed_merge_tags')
    @classmethod
    def validate_merge_tags(cls, v):
        for tag in v:
            if not re.fullmatch(_REF_KEY, tag):
                raise ValueError(f"Invalid merge tag '{tag}': use lowercase letters, digits and _")
        return v

    @field_validator('links')
    @classmethod
    def validate_links(cls, v):
        if len(v) > 20:
            raise ValueError('At most 20 links')
        for key, url in v.items():
            if not re.fullmatch(_REF_KEY, key):
                raise ValueError(f"Invalid link key '{key}': use lowercase letters, digits and _")
            if key.startswith('social_'):
                raise ValueError(f"Link key '{key}': the social_ prefix is reserved for social_links")
            if not re.fullmatch(r'https://[^\s<>"]{1,2000}', url):
                raise ValueError(f"Link '{key}' must be an https URL")
        return v

    @field_validator('assets')
    @classmethod
    def validate_assets(cls, v):
        if len(v) > 20:
            raise ValueError('At most 20 assets')
        for key, desc in v.items():
            if not re.fullmatch(_REF_KEY, key) or len(desc) > 200:
                raise ValueError(f"Invalid asset '{key}': key must be lowercase/digits/_ and description <= 200 chars")
        return v

    @field_validator('sender')
    @classmethod
    def validate_sender(cls, v):
        allowed = {'name', 'title', 'org', 'contact'}
        if set(v) - allowed or any(len(x) > 120 for x in v.values()):
            raise ValueError(f"sender may only contain {sorted(allowed)}, each <= 120 chars")
        return v


class EmailImageRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    api_key: str = Field(..., min_length=1)                     # Eddyter licence key (provider key only with the backend secret)
    slot: str = Field(..., pattern=r'^(hero|inline|logo)$')
    action: str = Field('fit', pattern=r'^fit$')                # refine / generate arrive in phase 2
    image_base64: str = Field(..., min_length=16, max_length=21_000_000)  # ~15 MB decoded
    focal_point: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    # S3 folder + storage quota. Licence callers always get their licence's user; only a backend-secret caller may set it.
    user_id: Optional[str] = Field(None, pattern=r'^[A-Za-z0-9_-]{1,64}$')

    @field_validator('focal_point')
    @classmethod
    def validate_focal_point(cls, v):
        if v is not None and not all(0.0 <= x <= 1.0 for x in v):
            raise ValueError('focal_point must be [x, y] with each value between 0 and 1')
        return v


# ---------------------------------------------------------------------------
# RESPONSE MODELS  (Enhancement #2)
# Typed so FastAPI generates correct OpenAPI schema and auto-validates output.
# ---------------------------------------------------------------------------
class ProcessResponse(BaseModel):
    success: bool
    markdown_response: str
    html_response: Optional[str] = None
    # Things the server fixed or could not do (generate_blog): [{"kind", "detail", "unit"?}]. Empty when clean.
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    design: Optional[Dict[str, Any]] = None
    graph_suggestions: List[GraphSuggestion] = Field(default_factory=list)
    graph_suggestion_error: Optional[Dict[str, str]] = None
    document_edit: Optional[DocumentEditResult] = None
    document_context: Optional[ContextMetadata] = None
    document_analysis: Optional[AnalysisResult] = None
    document_intent: Optional[IntentSuggestion] = None
    highlighting: Optional[HighlightResult] = None


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
    'generate_code': '',
    'explain_code': '',
    'fix_grammar':            'Fix grammar: ',
    'highlight_key_points':   '',
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
    'generate_blog':          '',
    'generate_design':        '',
    'edit_design':            '',
}


def image_billing(pending_images: Optional[dict]) -> tuple[int, list]:
    """
    (credits reserved for ALL images started, details of the images that finished).
    Every reservation is released at settlement, so a failed image no longer holds credits until the
    reservation key expires. details = [{"provider", "model", "credits_scaled"}] for the webhook.
    """
    entries = [v for k, v in (pending_images or {}).items() if not str(k).startswith("__") and isinstance(v, dict)]
    reserved = sum(int(e.get("reserved_scaled") or 0) for e in entries)
    done = [
        {"provider": e.get("provider"), "model": e.get("model"), "credits_scaled": int(e.get("credits_scaled") or 0)}
        for e in entries if e.get("status") == "done"
    ]
    return reserved, done


def billed_provider_model(model_provider: str, op_type: str, details: list) -> tuple[str, Optional[str]]:
    """Image-only requests are attributed to the image model that ran, not the (unused) chat model."""
    if op_type == 'image' and details and len({(d["provider"], d["model"]) for d in details}) == 1:
        return details[0]["provider"] or model_provider, details[0]["model"]
    return model_provider, None


def resolve_request_image_params(request_data, agent_type: str, model_provider: str, license_meta: dict, is_byok: bool):
    """
    Image params for a request, identical on /agent/stream and /agent/process:
    explicit API fields win; otherwise a model named in the user's own words ("using grok", regex only,
    never the LLM); and the matching user key is looked up when that provider is not the chat provider.
    """
    merged = dict(request_data.image_params or {})
    if request_data.image_provider:
        merged["provider"] = request_data.image_provider
    if request_data.image_model:
        merged["model"] = request_data.image_model

    if agent_type == 'general' and not merged.get("provider"):
        from src.image_gen import detect_image_model_request
        cue_provider, cue_model = detect_image_model_request(request_data.query)
        if cue_provider:
            merged["provider"] = cue_provider
            if cue_model and not merged.get("model"):
                merged["model"] = cue_model

    target = merged.get("provider")
    if target and target != model_provider and "api_key" not in merged:
        resolved_key = redis_manager.resolve_api_key_from_user_keys(
            user_keys=license_meta.get("user_keys"),
            model_provider=target,
            default_keys=None if is_byok else DEFAULT_API_KEYS,
        )
        if resolved_key:
            merged["api_key"] = resolved_key
    return merged


def build_full_query(agent_type: str, user_query: str, secondary_prompt: str) -> str:
    """Build the full prompt string sent to the AI model."""
    if agent_type == 'enhance_with_ai' and secondary_prompt:
        # Tags keep the instruction apart from the text: run together ("Enhance: X, Text: Y") the model sometimes copied
        # the instruction into the document.
        return f"<instruction>{secondary_prompt}</instruction>\n<text>\n{user_query}\n</text>"
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
    allow_missing_key: bool = False,
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
    if requires_user_api_keys:
        user_keys = {**user_keys, 'use_own_keys': True}

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
        default_keys=None if requires_user_api_keys or user_keys.get('use_own_keys') else DEFAULT_API_KEYS,
    )

    if not api_key and not allow_missing_key:
        logger.error(f"[{request_id}] ❌ No API key available for model: {model_provider}")
        raise HTTPException(
            status_code=400,
            detail=f'No API key configured for {model_provider}. Please add your API key in settings.',
        )

    logger.info(f"[{request_id}] ✅ API key retrieved successfully")

    license_meta = {
        'user_id': user_id,
        'requires_user_api_keys': requires_user_api_keys,
        'user_keys': user_keys,
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
async def process_text(request_data: ProcessRequest, request: Request) -> JSONResponse:
    """
    Non-streaming text processing endpoint.

    Called server-to-server by the backend (apps/backend agentService.ts),
    which resolves the provider key itself (user BYOK or platform key) and
    deducts credits after the call. So api_key here is the PROVIDER key, not
    a licence key — validating it as a licence would 403 every backend call,
    and the agent must not reserve credits on top of the backend's deduction.

    Fixes applied vs. original implementation:
      - Bug #9: TYPE_PREFIX_MAP now covers all valid agent types.
      - Enhancement #2: returns X-Request-ID header.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    start_trace(
        request_id, '/agent/process', type=request_data.type, chat_model=request_data.model,
        query_chars=len(request_data.query),
        document_context_request=request_data.document_context is not None,
    )
    # api_key here is a raw provider key and nothing is billed, so only the backend may call this route.
    if settings.agent_backend_secret and not backend_secret_valid(request):
        logger.warning(f"[{request_id}] /agent/process rejected: missing or wrong X-Agent-Backend-Secret")
        finish_trace(request_id, 'rejected', error='backend secret required', http_status=401)
        return JSONResponse(status_code=401, headers={"X-Request-ID": request_id},
                            content={'error': 'Unauthorized', 'code': 'BACKEND_SECRET_REQUIRED'})
    log_request_body(request_id, '/agent/process', request_data.model_dump())

    logger.info(f"[{request_id}] ========== NEW REQUEST ==========")

    try:
        had_document_context = request_data.document_context is not None      # auto -> Write drops the document, not the fact that this was one
        action_resolution = None
        if request_data.document_action == 'auto':
            # The backend's provider key serves the classifier as it would a BYOK key: the document task stays with that provider.
            decision = auto_action_by_rules(request_data) or await run_in_threadpool(
                classify_auto_action, request_data, provider=request_data.model, api_key=request_data.api_key, is_byok=True,
                platform_key=None, request_id=request_id)
            action_resolution = decision.event()
            trace_set(request_id, document_action=decision.action, document_action_source=decision.source)
            request_data = request_for_action(request_data, decision)
        chart_request = chart_request_for(request_data)
        needs_document = None if action_resolution and request_data.document_context is None else write_needs_document_for(request_data)
        document_text, prepared_context = await document_request_inputs(request_data)
        logger.info(f"[{request_id}] Type: {request_data.type}")
        logger.info(f"[{request_id}] Model: {request_data.model}")
        query_preview = (
            request_data.query[:100] + '...' if len(request_data.query) > 100
            else request_data.query
        )
        logger.info(f"[{request_id}] Query preview: " + ('[document task]' if had_document_context else query_preview))

        agent_type = request_data.type
        model_provider = request_data.model
        api_key = request_data.api_key
        if agent_type == 'generate_blog' and not settings.blog_ai_enabled:
            finish_trace(request_id, 'rejected', error='blog generation disabled', http_status=503)
            return JSONResponse(status_code=503, content={'error': 'AI blog generation is disabled'},
                                headers={"X-Request-ID": request_id})

        # Backend contract (see docstring): api_key is the provider key the backend
        # already resolved, and the backend bills the call. Treat it as BYOK so the
        # agent never attaches platform keys (DEFAULT_API_KEYS) or reserves credits.
        license_meta = {'user_id': None, 'requires_user_api_keys': True, 'user_keys': None}
        trace_mark(request_id, 'auth_done', user='backend')

        full_query = build_full_query(
            agent_type, request_data.query, request_data.secondary_prompt or ""
        )

        logger.info(f"[{request_id}] Processing: agent={agent_type}, model={model_provider}")

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']
        merged_image_params = resolve_request_image_params(
            request_data, agent_type, model_provider, license_meta, is_byok
        )

        ai_start = time.time()
        meta: dict = {}
        # Same routing, planner, direct image route and model selection as /agent/stream; images are
        # awaited and their URLs written into the text, so there are never dead placeholders.
        response, pending_images = await run_in_threadpool(
            process_user_query_with_images,
            full_query,
            model_provider=model_provider,
            api_key=api_key,
            request_id=request_id,
            force_agent_type=agent_for_request_type(agent_type),
            # Backend bills this call; also keeps caller keys out of the model cache.
            is_byok=True,
            user_id=None,
            credit_manager=None,
            image_routing=(agent_type == 'general'),
            image_params=merged_image_params or None,
            model_source=request_data.model_source,
            instruction=request_data.secondary_prompt or None,
            blog_options=request_data.blog_options.model_dump(exclude_none=True) if request_data.blog_options else None,
            meta=meta,
            user_text=request_data.query,
            existing_design_ast=request_data.existing_design_ast,
            operation_type=agent_type,
            suggest_graphs=request_data.suggest_graphs,
            document_text=document_text,
            prepared_context=prepared_context,
            match_document_style=request_data.document_context is not None and request_data.document_action == 'edit',
            document_intent=suggest_document_intent(request_data.query, request_data.document_action).model_dump() if request_data.suggest_document_intent else None,
            action_resolution=action_resolution,
            chart_request=chart_request,
            needs_document=needs_document,
        )
        warnings = meta.get('warnings', [])
        logger.info(f"[{request_id}] AI response received in {time.time() - ai_start:.2f}s")
        trace_mark(request_id, 'pipeline_done')

        # Images are the only thing this endpoint meters (text is not metered here, as before).
        image_reserved_scaled, image_details = image_billing(pending_images)
        if not is_byok and (image_details or image_reserved_scaled):
            bill_provider, bill_model = billed_provider_model(model_provider, 'image', image_details)
            duration_ms = int((time.time() - start_time) * 1000)
            if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                fire_credit_deduction_and_release(
                    user_id=user_id, input_chars=0, output_chars=0, operation_type='image',
                    image_count=len(image_details), reserved_scaled=image_reserved_scaled,
                    credit_manager=credit_manager, settle_delay_seconds=CREDIT_SETTLE_DELAY_SECONDS,
                    provider=bill_provider, model=bill_model, duration_ms=duration_ms,
                    success=True, images=image_details,
                )
            elif image_details:
                fire_credit_deduction(
                    user_id, 0, 0, 'image', len(image_details),
                    provider=bill_provider, model=bill_model, duration_ms=duration_ms,
                    success=True, images=image_details,
                )

        # Strip markdown code-fence wrappers the model sometimes emits. Code answers are fences by nature, and a document refined
        # in verified chunks (src/longform) had its wrappers removed there: a fence at its start or end is the document's own.
        trace = get_trace(request_id)
        trace_attrs = trace.attrs if trace is not None else {}
        is_code = agent_type in ('generate_code', 'explain_code') or trace_attrs.get('agent_type') == 'code_writer'
        keeps_fences = is_code or trace_attrs.get('mode') in ('longform', 'document_edit', 'document_context')
        if not keeps_fences and response.startswith('```markdown\n'):
            response = response[12:]
        elif not keeps_fences and response.startswith('```md\n'):
            response = response[6:]
        elif not keeps_fences and response.startswith('```\n'):
            response = response[4:]
        if not keeps_fences and response.endswith('\n```'):
            response = response[:-4]

        total_duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS — {total_duration:.2f}s")
        logger.info(f"[{request_id}] ================================")
        log_response_body(request_id, '/agent/process', {'markdown_response': response, 'images': summarize_images(pending_images)})
        finish_trace(request_id, 'ok', images=summarize_images(pending_images), http_status=200)

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'markdown_response': response,
                'html_response': response,
                'warnings': warnings,
                'design': meta.get('design_ready') if meta else None,
                'graph_suggestions': [meta['graph_suggestion']] if meta.get('graph_suggestion') else [],
                'graph_suggestion_error': meta.get('graph_suggestion_unavailable'),
                'document_edit': meta.get('document_edit_ready'),
                'document_context': meta.get('document_context_ready'),
                'document_analysis': meta.get('document_analysis_ready'),
                'document_intent': meta.get('document_intent_suggested'),
                'document_action_resolved': meta.get('document_action_resolved'),
                'highlighting': meta.get('highlight_ready'),
                'needs_context': meta.get('needs_context'),
            },
            headers={"X-Request-ID": request_id},
        )

    except KeyPointValidationError as exc:
        finish_trace(request_id, 'rejected', error=str(exc), http_status=422)
        return JSONResponse(status_code=422, content={'error': str(exc)}, headers={'X-Request-ID': request_id})
    except CodeValidationError as exc:
        finish_trace(request_id, 'rejected', error=str(exc), http_status=422)
        return JSONResponse(status_code=422, content={'error': str(exc), 'issues': exc.errors},
                            headers={'X-Request-ID': request_id})
    except HTTPException as exc:
        finish_trace(request_id, 'rejected', error=str(exc.detail), http_status=exc.status_code)
        raise  # let FastAPI's exception handler format the response
    except Exception as e:
        error = CONTEXT_ERROR if had_document_context else str(e)
        finish_trace(request_id, 'error', error=error, http_status=500)
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] FAILED — {error} ({total_duration:.2f}s)")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': error},
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
    start_trace(
        request_id, '/agent/stream', type=request_data.type, chat_model=request_data.model,
        query_chars=len(request_data.query),
        document_context_request=request_data.document_context is not None,
    )
    log_request_body(request_id, '/agent/stream', request_data.model_dump())

    logger.info(f"[{request_id}] ========== NEW STREAM REQUEST ==========")

    try:
        had_document_context = request_data.document_context is not None
        action_resolution, defer_action = None, False
        if request_data.document_action == 'auto':
            decision = auto_action_by_rules(request_data)
            if decision is None:
                defer_action = True      # unclear wording: the classifier needs the resolved key, so it runs right after the licence check
            else:
                action_resolution = decision.event()
                trace_set(request_id, document_action=decision.action, document_action_source=decision.source)
                request_data = request_for_action(request_data, decision)
        if defer_action:
            chart_request = needs_document = document_text = prepared_context = None
            context_no_model = False
        else:
            chart_request = chart_request_for(request_data)
            needs_document = None if action_resolution and request_data.document_context is None else write_needs_document_for(request_data)
            document_text, prepared_context = await document_request_inputs(request_data)
            context_no_model = (prepared_context is not None and prepared_context.message is not None) or needs_document is not None
        logger.info(f"[{request_id}] Type: {request_data.type}")
        logger.info(f"[{request_id}] Model: {request_data.model}")
        query_preview = (
            request_data.query[:100] + '...' if len(request_data.query) > 100
            else request_data.query
        )
        logger.info(f"[{request_id}] Query preview: " + ('[document task]' if had_document_context else query_preview))

        agent_type = request_data.type
        model_provider = request_data.model
        license_key = request_data.api_key
        if agent_type == 'generate_blog' and not settings.blog_ai_enabled:
            finish_trace(request_id, 'rejected', error='blog generation disabled', http_status=503)
            return JSONResponse(status_code=503, content={'error': 'AI blog generation is disabled'},
                                headers={"X-Request-ID": request_id})


        # Validate licence and resolve the actual AI provider API key.
        # Uses async_validate_and_get_api_key to run Redis calls concurrently via asyncio.gather (Point 1).
        api_key, license_meta = await async_validate_and_get_api_key(
            license_key, model_provider, request_id, background_tasks,
            **({'allow_missing_key': True} if settings.task_routing_enabled or settings.provider_fallback_enabled else {}),
        )
        trace_mark(request_id, 'auth_done', user=license_meta['user_id'])

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']
        text_byok = is_byok or bool((license_meta['user_keys'] or {}).get('use_own_keys'))
        if defer_action:
            # Same key rules as the code classifier: a BYOK request stays on its own key, anything else uses the platform's Claude key.
            decision = await run_in_threadpool(
                classify_auto_action, request_data, provider=model_provider, api_key=api_key, is_byok=text_byok,
                platform_key=DEFAULT_API_KEYS.get('anthropic_api_key') if not text_byok else None, request_id=request_id)
            action_resolution = decision.event()
            trace_set(request_id, document_action=decision.action, document_action_source=decision.source)
            request_data = request_for_action(request_data, decision)
            chart_request = chart_request_for(request_data)
            needs_document = None if request_data.document_context is None else write_needs_document_for(request_data)
            document_text, prepared_context = await document_request_inputs(request_data)
            context_no_model = (prepared_context is not None and prepared_context.message is not None) or needs_document is not None
        if not context_no_model and CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is None:
            raise HTTPException(status_code=503, detail='Credit service unavailable')
        key_resolver = None
        if not context_no_model and (settings.task_routing_enabled or settings.provider_fallback_enabled):
            from src.task_router import KeyResolver
            key_resolver = KeyResolver(license_meta['user_keys'], None if text_byok else DEFAULT_API_KEYS,
                                       redis_manager.resolve_api_key_from_user_keys)
            if not key_resolver.available_providers:
                raise HTTPException(status_code=400, detail='No API key configured for an eligible provider')
        original_prompt = '' if request_data.document_context is not None else (
            request_data.original_prompt.strip() if request_data.original_prompt else ''
        )

        image_urls = validate_reference_images(request_data.image_urls, request_id)

        if image_urls and not supports_vision(model_provider) and key_resolver is None:
            logger.warning(f"[{request_id}] {model_provider} cannot accept image input")
            usable = ", ".join(sorted(VISION_CAPABLE_PROVIDERS))
            raise HTTPException(
                status_code=400,
                detail={
                    'error': f"The '{model_provider}' model cannot read images. Switch to one of: {usable}.",
                    'code': 'VISION_NOT_SUPPORTED',
                },
            )

        full_query = build_full_query(
            agent_type, request_data.query, request_data.secondary_prompt or ""
        )
        input_chars = (prepared_context.metadata.billable_input_chars if prepared_context is not None
                       else len(request_data.query) + len(document_text or ''))

        # Mode 8 is billed at a flat price, not by characters. Same gate, same inputs as the pipeline's priority-1
        # design check, so the price reserved here is the price of what the pipeline runs.
        design_plan = None
        if document_text is None and prepared_context is None and agent_for_request_type(agent_type) == FREE_FORM_AGENT:
            from src.request_plan import plan_design_request
            design_plan = plan_design_request(
                request_data.query, agent_type, bool(image_urls), bool(request_data.existing_design_ast))
        design_price_scaled = design_credits_scaled(design_plan.design_sub_mode) if design_plan else 0

        # Upfront credit reservation (skipped for BYOK users)
        upfront_reserved_scaled = 0
        if not context_no_model and CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
            expected_output_chars = 0
            reservation_input_chars = input_chars
            if prepared_context is not None and prepared_context.analysis is not None:
                reservation_input_chars = prepared_context.metadata.reserved_input_chars
                expected_output_chars = prepared_context.metadata.reserved_output_chars
            if document_text is not None:
                expected_output_chars = len(document_text)
            if agent_type == 'generate_blog':
                # A post costs the platform its full parallel draft before the user has read a word: reserve for it.
                from src.blog.plan import BlogRequest, plan_blog
                expected_output_chars = 6 * plan_blog(BlogRequest.from_options(
                    request_data.query, request_data.secondary_prompt,
                    request_data.blog_options.model_dump(exclude_none=True) if request_data.blog_options else None),
                    image_available=False).target_words
            upfront_reserved_scaled = design_price_scaled or text_credits_scaled(reservation_input_chars, expected_output_chars)
            if (document_text is not None or prepared_context is not None) and upfront_reserved_scaled > MAX_RESERVATION_PER_REQUEST_SCALED:
                finish_trace(request_id, 'rejected', error='document request exceeds credit limit', http_status=402)
                return JSONResponse(status_code=402, content={
                    'error': 'This document exceeds the per-request credit limit. Use a smaller document.',
                    'code': 'REQUEST_CREDIT_LIMIT'})
            if image_urls and not design_plan:
                upfront_reserved_scaled += IMAGE_CREDITS_SCALED * len(image_urls)
            ok, available_scaled = credit_manager.reserve(user_id, upfront_reserved_scaled)
            if not ok:
                logger.warning(
                    f"[{request_id}] Insufficient credits: "
                    f"needed={upfront_reserved_scaled} available={available_scaled}"
                )
                finish_trace(request_id, 'rejected', error='insufficient credits', http_status=402)
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
            stream_warnings: list = []
            design_billing = None   # set when a Mode 8 design is delivered: its flat price replaces the char charge
            analysis_billing = None
            edit_result = None

            # Register lock-free cancellation event for this request (Point 6)
            cancel_event = register_cancellation_event(request_id)
            completion_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            def notify_image_completion() -> None:
                try:
                    loop.call_soon_threadsafe(completion_event.set)
                except RuntimeError:
                    pass

            trace_status = 'ok'
            trace_error = None
            out_preview: list[str] = []
            out_preview_len = 0
            used_provider, used_model = model_provider, None
            try:
                yield f"data: {json.dumps({'type': 'start', 'request_id': request_id})}\n\n"

                next_reserve_threshold = (upfront_reserved_scaled // CREDIT_SCALE) * 1000

                merged_image_params = resolve_request_image_params(
                    request_data, agent_type, model_provider, license_meta, is_byok
                )

                async for chunk in iterate_in_threadpool(process_user_query_stream(
                    full_query,
                    model_provider=model_provider,
                    api_key=api_key,
                    request_id=request_id,
                    user_id=user_id,
                    credit_manager=credit_manager if CREDIT_PRECHECK_ENABLED else None,
                    is_byok=is_byok,
                    image_urls=image_urls,
                    force_agent_type=agent_for_request_type(agent_type),
                    image_completion_callback=notify_image_completion,
                    # Only free-form chat may be routed to image generation by regex;
                    # text operations carry user text that can mention images freely.
                    image_routing=(agent_type == 'general'),
                    document_id=request_data.document_id,
                    image_params=merged_image_params or None,
                    model_source=request_data.model_source,
                    key_resolver=key_resolver,
                    report_model=True,
                    classifier_platform_key=DEFAULT_API_KEYS.get('anthropic_api_key') if not text_byok else None,
                    text_byok=text_byok,
                    instruction=request_data.secondary_prompt or None,
                    blog_options=request_data.blog_options.model_dump(exclude_none=True) if request_data.blog_options else None,
                    user_text=request_data.query,
                    existing_design_ast=request_data.existing_design_ast,
                    operation_type=agent_type,
                    cancel_event=cancel_event,
                    suggest_graphs=request_data.suggest_graphs,
                    document_text=document_text,
                    prepared_context=prepared_context,
                    match_document_style=request_data.document_context is not None and request_data.document_action == 'edit',
                    document_intent=suggest_document_intent(request_data.query, request_data.document_action).model_dump() if request_data.suggest_document_intent else None,
                    action_resolution=action_resolution,
                    chart_request=chart_request,
                    needs_document=needs_document,
                )):
                    # Lock-free cancellation check on every streamed token (Point 6)
                    if cancel_event.is_set():
                        if prepared_context is not None or document_text is not None:
                            trace_status = 'aborted'
                        logger.info(
                            f"[{request_id}] Generator stopped by client abort "
                            f"(output_chars={output_chars})"
                        )
                        break

                    if isinstance(chunk, dict) and "__pending_images__" in chunk:
                        pending_images = chunk["__pending_images__"]
                        continue

                    if isinstance(chunk, dict) and '__model_used__' in chunk:
                        used_provider, used_model = chunk['__model_used__']
                        continue
                    if isinstance(chunk, dict) and '__event__' in chunk:
                        ev = chunk['__event__']
                        if isinstance(ev, dict) and 'event' in ev and 'data' in ev:
                            if ev['event'] == 'document_analysis_ready':
                                analysis_billing = ev['data']['billing']
                            if ev['event'] == 'document_edit_ready':
                                edit_result = ev['data']
                            if ev['event'] == 'design_ready' and design_plan is not None:
                                design_billing = {'sub_mode': design_plan.design_sub_mode,
                                                  'credits_scaled': design_price_scaled}
                            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
                        else:
                            yield f"data: {json.dumps(ev)}\n\n"
                        continue
                    if isinstance(chunk, dict) and '__warnings__' in chunk:
                        stream_warnings = chunk['__warnings__']
                        continue
                    replacing = isinstance(chunk, dict) and '__content_replace__' in chunk
                    if replacing:
                        chunk = chunk['__content_replace__']

                    next_output_chars = len(chunk) if replacing else output_chars + len(chunk)

                    # Per-1000-char incremental reservation (non-BYOK only)
                    # An Edit streams the whole document snapshot back; it is billed on its patches, not on these characters.
                    if not context_no_model and analysis_billing is None and document_text is None and CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
                        while (input_chars + next_output_chars) > next_reserve_threshold:
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

                    if output_chars == 0 and chunk:
                        trace_mark(request_id, 'first_token')
                    output_chars = next_output_chars
                    if replacing:
                        out_preview, out_preview_len = [chunk[:20000]], min(len(chunk), 20000)
                    elif out_preview_len < 20000:
                        out_preview.append(chunk)
                        out_preview_len += len(chunk)

                    yield f"data: {json.dumps({'type': 'content_replace' if replacing else 'content', 'content': chunk})}\n\n"

                trace_mark(request_id, 'text_done')

                # Done event
                total_duration = time.time() - start_time
                actual_images = {k: v for k, v in pending_images.items() if k != "__completion_callback__"}
                pending_count = len(actual_images)
                logger.info(
                    f"[{request_id}] STREAM SUCCESS — {total_duration:.2f}s, "
                    f"pending images: {pending_count}"
                )
                done_event = {'type': 'done', 'duration': round(total_duration, 2), 'pending_images': pending_count}
                if (prepared_context is not None or document_text is not None) and cancel_event.is_set():
                    done_event['aborted'] = True
                    trace_status = 'aborted'
                if stream_warnings:
                    done_event['warnings'] = stream_warnings
                yield f"data: {json.dumps(done_event)}\n\n"

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

                    trace_mark(request_id, 'images_done')
                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

            except Exception as e:
                had_error = True
                trace_status, trace_error = 'error', CONTEXT_ERROR if had_document_context else str(e)
                error_code = type(e).__name__
                logger.error(f"[{request_id}] STREAM FAILED — {trace_error}")
                yield f"data: {json.dumps({'type': 'error', 'error': trace_error})}\n\n"

            except GeneratorExit:
                trace_status = 'aborted'
                unregister_cancellation_event(request_id)
                logger.info(
                    f"[{request_id}] Generator closed by client disconnect "
                    f"(output_chars={output_chars})"
                )

            finally:
                if prepared_context is not None and prepared_context.analysis is not None:
                    cancel_event.set()
                unregister_cancellation_event(request_id)
                context_trace = get_trace(request_id)
                context_used_model = (prepared_context is None and document_text is None and needs_document is None) or bool(context_trace and context_trace.attrs.get('document_model_started'))
                if prepared_context is not None and prepared_context.analysis is not None and analysis_billing is None and not had_error:
                    context_used_model = False
                if edit_result is not None and edit_result.get('status') != 'proposed':
                    # No proposal (no_change, needs_clarification, failed): nothing reaches the document, nothing is charged.
                    context_used_model = False
                billed_input_chars = analysis_billing['input_chars'] if analysis_billing is not None else input_chars
                billed_output_chars = analysis_billing['output_chars'] if analysis_billing is not None else output_chars
                if edit_result is not None:
                    # The stream carries the whole document snapshot; only the rewritten blocks are generated output.
                    billed_output_chars = sum(len(p.get('after', '')) for p in edit_result.get('patches', []))
                duration_ms = int((time.time() - start_time) * 1000)
                log_response_body(request_id, '/agent/stream', {
                    'text': ''.join(out_preview), 'output_chars': output_chars,
                    'images': summarize_images(pending_images),
                })
                finish_trace(
                    request_id, trace_status, error=trace_error,
                    images=summarize_images(pending_images), http_status=200,
                )

                if not context_used_model and not is_byok and credit_manager is not None and upfront_reserved_scaled:
                    credit_manager.schedule_release(user_id, upfront_reserved_scaled + extra_reserved_scaled, delay_seconds=0)

                if context_used_model and not had_error and not is_byok:
                    if resolved_image_count > 0 and output_chars > 0:
                        op_type = 'mixed'
                    elif resolved_image_count > 0:
                        op_type = 'image'
                    else:
                        op_type = 'text'

                    image_reserved_scaled, image_details = image_billing(pending_images)
                    bill_provider, bill_model = billed_provider_model(used_provider, op_type, image_details)
                    if op_type != 'image':
                        bill_model = used_model
                    if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                        total_reserved_scaled = (
                            upfront_reserved_scaled
                            + extra_reserved_scaled
                            + image_reserved_scaled
                        )
                        fire_credit_deduction_and_release(
                            user_id=user_id,
                            input_chars=billed_input_chars,
                            output_chars=billed_output_chars,
                            operation_type=op_type,
                            image_count=resolved_image_count,
                            reserved_scaled=total_reserved_scaled,
                            credit_manager=credit_manager,
                            settle_delay_seconds=CREDIT_SETTLE_DELAY_SECONDS,
                            provider=bill_provider,
                            model=bill_model,
                            duration_ms=duration_ms,
                            success=True,
                            images=image_details,
                            design=design_billing,
                        )
                    else:
                        fire_credit_deduction(
                            user_id, billed_input_chars, billed_output_chars, op_type,
                            resolved_image_count,
                            provider=bill_provider, model=bill_model,
                            duration_ms=duration_ms, success=True,
                            images=image_details, design=design_billing,
                        )

                elif context_used_model and had_error and not is_byok:
                    if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                        total_reserved_scaled = (
                            upfront_reserved_scaled
                            + extra_reserved_scaled
                            + image_billing(pending_images)[0]
                        )
                        credit_manager.schedule_release(
                            user_id, total_reserved_scaled, delay_seconds=0,
                        )
                    fire_credit_deduction(
                        user_id, input_chars, output_chars, 'text',
                        resolved_image_count,
                        provider=used_provider, model=used_model,
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

    except HTTPException as exc:
        finish_trace(request_id, 'rejected', error=str(exc.detail), http_status=exc.status_code)
        raise
    except Exception as e:
        error = CONTEXT_ERROR if request_data.document_context is not None else str(e)
        finish_trace(request_id, 'error', error=error, http_status=500)
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] STREAM FAILED — {error} ({total_duration:.2f}s)")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': error},
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
            is_byok=is_byok,
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


@app.get('/agent/metrics')
async def metrics(limit: int = 50) -> JSONResponse:
    """Recent request summaries + latency percentiles. Off unless METRICS_ENDPOINT_ENABLED=true (local/dev)."""
    if not settings.metrics_endpoint_enabled:
        raise HTTPException(status_code=404, detail='Not found')
    from src.observability import latency_stats, recent_summaries
    return JSONResponse(content={'stats': latency_stats(), 'recent': recent_summaries(min(max(limit, 1), 200))})


@app.get('/agent/health/providers')
async def provider_health(request: Request) -> JSONResponse:
    """
    What each provider/model answered or failed with in the last 15 minutes, the active alerts (key rejected, model missing, request shape
    rejected, truncation, refusals, failure rate) and the startup model check. Counts only: no content, no keys. For monitors and
    rollouts; allowed with the backend secret (X-Agent-Backend-Secret) or when the metrics endpoint is enabled.
    """
    if not (settings.metrics_endpoint_enabled or backend_secret_valid(request)):
        raise HTTPException(status_code=404, detail='Not found')
    from src.models import MODEL_ENV_OVERRIDES
    from src.provider_health import HEALTH
    snapshot = HEALTH.snapshot()
    snapshot['model_overrides'] = MODEL_ENV_OVERRIDES
    snapshot['status'] = 'alert' if any(a['severity'] == 'critical' for a in snapshot['alerts']) else ('warning' if snapshot['alerts'] else 'ok')
    return JSONResponse(content=snapshot)


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


# ---------------------------------------------------------------------------
# AI email templates (src/email_template). Same contract as /agent/stream: api_key is the Eddyter licence key; BYOK
# users run on their saved keys, everyone else on the platform key, and non-BYOK calls are billed through the credit
# webhook as text. A server-to-server caller holding AGENT_BACKEND_SECRET may still send a provider key instead (no
# licence check, no billing here). Output is template JSON; the editor renders HTML from it.
# ---------------------------------------------------------------------------
EMAIL_OUTPUT_CHARS_PER_VARIANT = 1500   # upfront reservation only; the charge uses the real output
_EMAIL_NON_TEXT_KEYS = {'id', 'type', 'variant', 'align', 'alignment', 'size', 'width', 'height', 'href', 'url', 'link',
                        'src', 'theme', 'color', 'colour', 'background', 'style', 'slot', 'platform', 'layout', 'role'}


def backend_secret_valid(request: Request) -> bool:
    """True only for a caller that sends the configured AGENT_BACKEND_SECRET (constant-time comparison)."""
    import hmac
    secret = settings.agent_backend_secret
    supplied = request.headers.get('X-Agent-Backend-Secret', '')
    return bool(secret) and bool(supplied) and hmac.compare_digest(supplied.encode(), secret.encode())


def email_input_chars(request_data) -> int:
    """The user's own text the model reads: the brief and the free-text fields around it."""
    texts = [request_data.brief, request_data.audience or '', request_data.org_name or '',
             *request_data.sender.values(), *request_data.assets.values()]
    return sum(len(t) for t in texts)


def email_output_chars(result: dict) -> int:
    """Characters of readable text in the returned templates (subjects, headings, body, buttons), not the JSON around it."""
    def walk(value, key=''):
        if isinstance(value, dict):
            return sum(walk(v, k) for k, v in value.items())
        if isinstance(value, list):
            return sum(walk(v, key) for v in value)
        if isinstance(value, str) and key not in _EMAIL_NON_TEXT_KEYS and not key.endswith(('_asset', '_url', '_color', '_id')) \
                and not re.match(r'^(?:asset:|link:|https?://|mailto:|#[0-9A-Fa-f]{3,8}$)', value):
            return len(value)
        return 0
    return sum(walk(t.get('template', {})) for t in result.get('templates', []))


async def resolve_email_caller(request: Request, request_data, request_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Who is calling and with which keys. Licence callers go through exactly the /agent/stream checks (invalid or
    blocked licence 403, past-due managed plan 402, no key 400, Redis down 503). Returns user_id, is_byok and, for
    model calls, candidates + key_for.
    """
    if backend_secret_valid(request):
        from src.task_router import Candidate
        return {'backend': True, 'user_id': None, 'is_byok': True, 'text_byok': True,
                'candidates': [Candidate(getattr(request_data, 'model', ''), True)], 'key_for': lambda _p: request_data.api_key}
    model = getattr(request_data, 'model', None) or 'openai'
    routed = settings.task_routing_enabled or settings.provider_fallback_enabled
    # A route without a model call (image fitting) needs a valid licence but no provider key.
    needs_key = hasattr(request_data, 'model') and not routed
    api_key, license_meta = await async_validate_and_get_api_key(
        request_data.api_key, model, request_id, background_tasks, **({} if needs_key else {'allow_missing_key': True}))
    trace_mark(request_id, 'auth_done', user=license_meta['user_id'])
    is_byok = license_meta['requires_user_api_keys']
    text_byok = is_byok or bool((license_meta['user_keys'] or {}).get('use_own_keys'))
    caller = {'backend': False, 'user_id': license_meta['user_id'], 'is_byok': is_byok, 'text_byok': text_byok}
    if not hasattr(request_data, 'model'):
        return caller
    from src.task_router import Candidate, KeyResolver, resolve_route
    if routed:
        key_resolver = KeyResolver(license_meta['user_keys'], None if text_byok else DEFAULT_API_KEYS,
                                   redis_manager.resolve_api_key_from_user_keys)
        candidates = resolve_route('writing', model, request_data.model_source, key_resolver.available_providers, True)
        if not candidates:
            raise HTTPException(status_code=400, detail='No API key configured for an eligible provider')
        caller.update(candidates=candidates, key_for=key_resolver)
    else:
        caller.update(candidates=[Candidate(model, True)], key_for=lambda _p: api_key)
    return caller
@app.get('/agent/email/categories')
async def email_categories() -> JSONResponse:
    """Email types the frontend can offer, in routing order. Only enabled types (with an example) are listed."""
    from src.email_template.categories import enabled_categories

    if not settings.email_ai_enabled:
        return JSONResponse(status_code=503, content={'error': 'AI email templates are disabled'})
    return JSONResponse(content={'categories': [
        {'id': c.id, 'label': c.label, 'default_tone': c.default_tone, 'image_slots': list(c.image_slots),
         'needs_button': c.needs_button, 'needs_signature': c.needs_signature, 'bulk': c.bulk}
        for c in enabled_categories().values()
    ]})


@app.post('/agent/email/generate')
async def email_generate(request_data: EmailGenerateRequest, request: Request,
                         background_tasks: BackgroundTasks) -> JSONResponse:
    from src.email_template.service import (
        EmailGenerationError, EmailProviderError, GenerateRequest, generate, make_routed_llm,
    )
    from src.models import ADVANCED_MODEL_IDS

    request_id = str(uuid.uuid4())[:8]
    headers = {"X-Request-ID": request_id}
    if not settings.email_ai_enabled:
        return JSONResponse(status_code=503, content={'error': 'AI email templates are disabled'}, headers=headers)
    start_trace(request_id, '/agent/email/generate', chat_model=request_data.model,
                query_chars=len(request_data.brief))
    log_request_body(request_id, '/agent/email/generate', request_data.model_dump())
    start_time = time.time()
    try:
        caller = await resolve_email_caller(request, request_data, request_id, background_tasks)
    except HTTPException as e:
        finish_trace(request_id, 'rejected', error=str(e.detail), http_status=e.status_code)
        raise
    user_id, billed = caller['user_id'], not caller['backend'] and not caller['is_byok']
    if billed and CREDIT_PRECHECK_ENABLED and credit_manager is None:
        finish_trace(request_id, 'rejected', error='credit service unavailable', http_status=503)
        raise HTTPException(status_code=503, detail='Credit service unavailable')
    input_chars = email_input_chars(request_data)
    reserved_scaled = 0
    if billed and CREDIT_PRECHECK_ENABLED and credit_manager is not None:
        reserved_scaled = text_credits_scaled(input_chars, EMAIL_OUTPUT_CHARS_PER_VARIANT * request_data.variants)
        ok, available_scaled = credit_manager.reserve(user_id, reserved_scaled)
        if not ok:
            finish_trace(request_id, 'rejected', error='insufficient credits', http_status=402)
            return JSONResponse(status_code=402, headers=headers, content={
                'error': 'Insufficient credits', 'code': 'INSUFFICIENT_CREDITS',
                'available': available_scaled, 'needed': reserved_scaled})

    def release_reservation():
        if reserved_scaled:
            credit_manager.schedule_release(user_id, reserved_scaled, delay_seconds=0)

    req = GenerateRequest(
        brief=request_data.brief, category=request_data.category, tone=request_data.tone,
        audience=request_data.audience, language=request_data.language, length=request_data.length,
        allowed_merge_tags=tuple(request_data.allowed_merge_tags), links=request_data.links,
        assets=request_data.assets, sender=request_data.sender, org_name=request_data.org_name,
        brand_kit_id=request_data.brand_kit_id, variants=request_data.variants,
        sign_off=request_data.sign_off, social_links=request_data.social_links,
        theme=request_data.theme.model_dump(exclude_none=True) if request_data.theme else None,
    )
    llm = make_routed_llm(caller['candidates'], caller['key_for'], request_id=request_id, is_byok=caller['text_byok'])
    try:
        result = await run_in_threadpool(generate, req, llm, request_id)
    except ValueError as e:
        release_reservation()   # a failed generation is not charged, as on /agent/stream
        finish_trace(request_id, 'rejected', error=str(e), http_status=400)
        return JSONResponse(status_code=400, content={'error': str(e)}, headers=headers)
    except EmailGenerationError as e:
        release_reservation()
        finish_trace(request_id, 'rejected', error=str(e), http_status=422)
        return JSONResponse(
            status_code=422,
            content={'error': 'The generated email did not pass validation',
                     'issues': [i.as_dict() for i in e.errors]},
            headers=headers,
        )
    except EmailProviderError as e:
        release_reservation()
        logger.error(f"[{request_id}] email generate provider error — {e}")
        finish_trace(request_id, 'error', error=str(e), http_status=502)
        return JSONResponse(status_code=502, content={'error': str(e)}, headers=headers)
    except Exception as e:
        release_reservation()
        logger.error(f"[{request_id}] email generate FAILED — {e}")
        finish_trace(request_id, 'error', error=str(e), http_status=500)
        return JSONResponse(status_code=500, content={'error': str(e)}, headers=headers)

    output_chars = email_output_chars(result)
    provider = llm.used.get('provider') or request_data.model
    model_id = ADVANCED_MODEL_IDS.get(provider)
    trace_set(request_id, model_used=f'{provider}/{model_id}', email_output_chars=output_chars,
              email_billed=billed)
    if billed:
        # operation_type "text", priced by characters like any text request: the brief and its fields in, the
        # templates' readable text out (all variants). BYOK and backend-secret calls are not billed here.
        duration_ms = int((time.time() - start_time) * 1000)
        if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
            fire_credit_deduction_and_release(
                user_id=user_id, input_chars=input_chars, output_chars=output_chars, operation_type='text',
                image_count=0, reserved_scaled=reserved_scaled, credit_manager=credit_manager,
                settle_delay_seconds=CREDIT_SETTLE_DELAY_SECONDS, provider=provider, model=model_id,
                duration_ms=duration_ms, success=True)
        else:
            fire_credit_deduction(user_id, input_chars, output_chars, 'text', 0, provider=provider, model=model_id,
                                  duration_ms=duration_ms, success=True)

    log_response_body(request_id, '/agent/email/generate', result)
    finish_trace(request_id, 'ok', http_status=200)
    return JSONResponse(status_code=200, content={'success': True, **result}, headers=headers)


@app.post('/agent/email/image')
async def email_image(request_data: EmailImageRequest, request: Request,
                      background_tasks: BackgroundTasks) -> JSONResponse:
    """Crop and compress an image to a fixed email slot size, then upload it. Phase 1: action 'fit' only."""
    from src.email_template.images import SlotImageError, fit_to_slot
    from src.s3_service import StorageQuotaExceededError, upload_image_from_bytes_detailed

    request_id = str(uuid.uuid4())[:8]
    headers = {"X-Request-ID": request_id}
    if not settings.email_ai_enabled:
        return JSONResponse(status_code=503, content={'error': 'AI email templates are disabled'}, headers=headers)
    # Licence check as on /agent/stream. No model runs and nothing is billed; the upload goes to the licence's user.
    caller = await resolve_email_caller(request, request_data, request_id, background_tasks)
    upload_user = request_data.user_id if caller['backend'] else caller['user_id']
    try:
        b64 = request_data.image_base64
        if b64.startswith('data:'):  # browsers' FileReader.readAsDataURL output
            b64 = b64.partition(',')[2]
        raw = base64.b64decode(re.sub(r'\s+', '', b64), validate=True)
        focal = tuple(request_data.focal_point) if request_data.focal_point else None
        fitted = await run_in_threadpool(fit_to_slot, raw, request_data.slot, focal)
        uploaded = await run_in_threadpool(
            upload_image_from_bytes_detailed, fitted.data, fitted.content_type, upload_user, request_id,
        )
    except (binascii.Error, ValueError) as e:  # SlotImageError is a ValueError
        status = 400 if not isinstance(e, SlotImageError) or 'compress' not in str(e) else 422
        return JSONResponse(status_code=status, content={'error': f'Invalid image: {e}'}, headers=headers)
    except StorageQuotaExceededError as e:
        return JSONResponse(status_code=413, content={'error': str(e)}, headers=headers)
    except Exception as e:
        logger.error(f"[{request_id}] email image FAILED — {e}")
        return JSONResponse(status_code=500, content={'error': str(e)}, headers=headers)

    return JSONResponse(
        status_code=200,
        content={
            'success': True,
            'slot': request_data.slot,
            'url': uploaded['cdn_url'],
            'width': fitted.width, 'height': fitted.height,
            'display_width': fitted.display_width, 'display_height': fitted.display_height,
            'content_type': fitted.content_type,
            'size_bytes': len(fitted.data),
        },
        headers=headers,
    )


if __name__ == '__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=False)
