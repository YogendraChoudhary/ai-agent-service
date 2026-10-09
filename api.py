from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.text_processing_agent import (
    process_user_query,
    process_user_query_stream,
    get_autocomplete_suggestion,
    supports_vision,
    VISION_CAPABLE_PROVIDERS,
)
from src.redis_manager import get_redis_manager
from src.webhook import (
    fire_credit_deduction,
    fire_credit_deduction_and_release,
    fire_prompt_history_save,
)
from src.credit_manager import (
    CreditManager,
    text_credits_scaled,
    AUTOCOMPLETE_CREDITS_SCALED,
    IMAGE_CREDITS_SCALED,
    CREDIT_SCALE,
)
import logging
import time
import uuid
import json
import os
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Processing Agent API")

# Enable CORS for all routes (matching Flask CORS configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Supported AI models
SUPPORTED_MODELS = ['openai', 'claude', 'grok', 'deepseek', 'gemini']

# Track cancelled requests so generators can stop early
# Maps request_id -> timestamp of cancellation
cancelled_requests = {}
# Maximum age (seconds) for cancelled_requests entries before cleanup
CANCEL_ENTRY_TTL = 300  # 5 minutes

# Load default API keys from environment
DEFAULT_API_KEYS = {
    'openai_api_key': os.getenv('DEFAULT_OPENAI_API_KEY'),
    'anthropic_api_key': os.getenv('DEFAULT_ANTHROPIC_API_KEY'),
    'grok_api_key': os.getenv('DEFAULT_GROK_API_KEY'),
    'gemini_api_key': os.getenv('DEFAULT_GEMINI_API_KEY'),
}

# Initialize Redis manager
try:
    redis_manager = get_redis_manager()
    logger.info("✅ Redis manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Redis manager: {str(e)}")
    logger.error("⚠️  API will not be able to validate licenses!")
    redis_manager = None

# Credit pre-check configuration
CREDIT_PRECHECK_ENABLED = os.getenv("CREDIT_PRECHECK_ENABLED", "true").lower() == "true"
CREDIT_SETTLE_DELAY_SECONDS = float(os.getenv("CREDIT_SETTLE_DELAY_SECONDS", "3"))
MAX_RESERVATION_PER_REQUEST_SCALED = (
    int(os.getenv("MAX_RESERVATION_PER_REQUEST", "100")) * CREDIT_SCALE
)

# Initialize CreditManager (depends on redis_manager)
credit_manager = None
if redis_manager is not None:
    try:
        credit_manager = CreditManager(redis_manager.redis_client)
        logger.info("✅ CreditManager initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize CreditManager: {str(e)}")
        credit_manager = None


# Reference images the user attaches for the model to look at. Capped because
# each one costs input tokens and a design brief needs very few.
MAX_REFERENCE_IMAGES = int(os.getenv("MAX_REFERENCE_IMAGES", "3"))

# Only our own CDN is accepted as an image source. Without this the endpoint
# would fetch any URL a caller supplies, turning it into an SSRF vector against
# anything reachable from the agent host (including EC2 instance metadata).
CDN_URL = os.getenv("CDN_URL", "")


def validate_reference_images(raw, request_id: str):
    """
    Validate the image_urls field from a request.

    Returns (image_urls, error) where error is a dict ready to jsonify, or None.
    """
    if raw in (None, [], ""):
        return [], None

    if not isinstance(raw, list):
        return None, {"error": "image_urls must be an array of URLs", "status": 400}

    if len(raw) > MAX_REFERENCE_IMAGES:
        return None, {
            "error": f"At most {MAX_REFERENCE_IMAGES} reference images are allowed",
            "status": 400,
        }

    if not CDN_URL:
        logger.error(f"[{request_id}] CDN_URL is not configured — rejecting reference images")
        return None, {"error": "Image uploads are not configured on this server", "status": 500}

    allowed_prefix = CDN_URL.rstrip("/") + "/"
    for url in raw:
        if not isinstance(url, str) or not url.startswith(allowed_prefix):
            logger.warning(f"[{request_id}] Rejected reference image from disallowed source")
            return None, {
                "error": "Reference images must be uploaded through the editor before use",
                "status": 400,
            }

    return raw, None


def validate_and_get_api_key(license_key: str, model_provider: str, request_id: str):
    """
    Validate license key and get the appropriate API key for the model

    Args:
        license_key: The license key to validate (from request)
        model_provider: The AI model provider (openai, claude, etc.)
        request_id: Request ID for logging

    Returns:
        tuple: (api_key, error_response, license_meta)
            - If successful: (api_key_string, None, {'user_id': ..., 'requires_user_api_keys': ...})
            - If failed: (None, error_dict, None)
    """
    if not redis_manager:
        logger.error(f"[{request_id}] Redis manager not available")
        return None, {
            'error': 'License validation service unavailable',
            'status': 503
        }, None

    # Step 1: Validate license
    logger.info(f"[{request_id}] Validating license key...")
    license_result = redis_manager.validate_license(license_key)

    if not license_result['valid']:
        logger.warning(f"[{request_id}] ❌ License validation failed: {license_result['error']}")
        # A result carrying a 'code' already has a complete, user-facing message
        # (e.g. an admin-suspended account) — don't bury it under "Invalid license:".
        license_error_code = license_result.get('code')
        return None, {
            'error': license_result['error'] if license_error_code
                     else f"Invalid license: {license_result['error']}",
            'code': license_error_code,
            'status': 403
        }, None

    user_id = license_result['user_id']
    license_info = license_result.get('license', {})
    logger.info(f"[{request_id}] ✅ License valid for user: {user_id}")

    # Update heartbeat asynchronously (don't wait for result)
    try:
        redis_manager.update_heartbeat(license_key)
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to update heartbeat: {str(e)}")

    # Step 1.5: Check subscription status for past_due managed plans
    # Managed plans (requires_user_api_keys=false) use platform API keys
    # Block AI access during past_due to prevent unpaid usage of platform resources
    subscription_status = license_info.get('subscription_status', '')
    requires_user_api_keys = license_info.get('requires_user_api_keys', False)

    if subscription_status == 'past_due' and not requires_user_api_keys:
        logger.warning(f"[{request_id}] ❌ Blocked: subscription is past_due on managed plan")
        return None, {
            'error': 'Your payment is past due. Please update your payment method to continue using AI features.',
            'status': 402
        }, None

    # Step 2: Get API key for the model
    logger.info(f"[{request_id}] Fetching API key for model: {model_provider}")
    api_key = redis_manager.get_api_key_for_model(
        user_id=user_id,
        model_provider=model_provider,
        default_keys=DEFAULT_API_KEYS
    )

    if not api_key:
        logger.error(f"[{request_id}] ❌ No API key available for model: {model_provider}")
        return None, {
            'error': f'No API key configured for {model_provider}. Please add your API key in settings.',
            'status': 400
        }, None

    logger.info(f"[{request_id}] ✅ API key retrieved successfully")

    license_meta = {
        'user_id': user_id,
        'requires_user_api_keys': requires_user_api_keys,
    }

    return api_key, None, license_meta


@app.post('/agent/process')
async def process_text(request: Request):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== NEW REQUEST ==========")

    try:
        try:
            data = await request.json()
        except Exception:
            data = None

        if not data or not isinstance(data, dict):
            logger.warning(f"[{request_id}] VALIDATION FAILED - Missing request body")
            return JSONResponse(
                status_code=400,
                content={'error': 'Missing required fields: type, query, model, api_key'}
            )

        # Log request details (without sensitive data)
        logger.info(f"[{request_id}] Request received")
        logger.info(f"[{request_id}] Type: {data.get('type', 'N/A')}")
        logger.info(f"[{request_id}] Model: {data.get('model', 'N/A')}")
        query_preview = data.get('query', '')[:100] + '...' if len(data.get('query', '')) > 100 else data.get('query', '')
        logger.info(f"[{request_id}] Query preview: {query_preview}")

        # Validate required fields
        required_fields = ['type', 'query', 'model', 'api_key']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Missing fields: {missing_fields}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Missing required fields: {", ".join(missing_fields)}'}
            )

        agent_type = data['type']
        user_query = data['query']
        secondary_prompt = data.get('secondary_prompt', '')
        model_provider = data['model'].lower()
        api_key = data['api_key']

        # Validate model provider
        if model_provider not in SUPPORTED_MODELS:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid model: {model_provider}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}'}
            )

        type_mapping = {
            'fix_grammar': 'Fix grammar: ',
            'simplify': 'Simplify: ',
            'complete': 'Complete: ',
            'emojify': 'Emojify: ',
            'translate': 'Translate: ',
            'make_longer': 'Make longer: ',
            'make_shorter': 'Make shorter: ',
            'enhance_with_ai': 'Enhance: ',
            'change_tone': 'Change tone: ',
            'general': ''
        }

        if agent_type not in type_mapping:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid type: {agent_type}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid type. Must be one of: {", ".join(type_mapping.keys())}'}
            )

        # Build query based on type with secondary_prompt support
        if agent_type == 'enhance_with_ai' and secondary_prompt:
            full_query = f"Enhance: {secondary_prompt}, Text: {user_query}"
        elif agent_type == 'change_tone' and secondary_prompt:
            full_query = f"Change tone to {secondary_prompt}: {user_query}"
        elif agent_type == 'translate' and secondary_prompt:
            full_query = f"Translate to {secondary_prompt}: {user_query}"
        else:
            full_query = f"{type_mapping[agent_type]}{user_query}"

        logger.info(f"[{request_id}] Processing with agent type: {agent_type}, model: {model_provider}")
        logger.info(f"[{request_id}] Calling AI model...")

        ai_start_time = time.time()
        response = await run_in_threadpool(
            process_user_query,
            full_query,
            model_provider=model_provider,
            api_key=api_key,
            request_id=request_id
        )
        ai_duration = time.time() - ai_start_time

        logger.info(f"[{request_id}] AI response received in {ai_duration:.2f}s")

        # Clean up any markdown code block wrappers
        if response.startswith('```markdown\n'):
            response = response[12:]
        elif response.startswith('```md\n'):
            response = response[6:]
        elif response.startswith('```\n'):
            response = response[4:]
        if response.endswith('\n```'):
            response = response[:-4]

        total_duration = time.time() - start_time
        response_preview = response[:100] + '...' if len(response) > 100 else response
        logger.info(f"[{request_id}] Response preview: {response_preview}")
        logger.info(f"[{request_id}] SUCCESS - Total time: {total_duration:.2f}s")
        logger.info(f"[{request_id}] ================================")

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'markdown_response': response
            }
        )

    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] FAILED - Error: {str(e)}")
        logger.error(f"[{request_id}] Total time: {total_duration:.2f}s")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post('/agent/stream')
async def stream_text(request: Request):
    """
    Streaming endpoint for AI text processing.
    Returns Server-Sent Events (SSE) with markdown content chunks.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== NEW STREAM REQUEST ==========")

    try:
        try:
            data = await request.json()
        except Exception:
            data = None

        if not data or not isinstance(data, dict):
            logger.warning(f"[{request_id}] VALIDATION FAILED - Missing request body")
            return JSONResponse(
                status_code=400,
                content={'error': 'Missing required fields: type, query, model, api_key'}
            )

        # Log request details (without sensitive data)
        logger.info(f"[{request_id}] Stream request received")
        logger.info(f"[{request_id}] Type: {data.get('type', 'N/A')}")
        logger.info(f"[{request_id}] Model: {data.get('model', 'N/A')}")
        query_preview = data.get('query', '')[:100] + '...' if len(data.get('query', '')) > 100 else data.get('query', '')
        logger.info(f"[{request_id}] Query preview: {query_preview}")

        # Validate required fields
        required_fields = ['type', 'query', 'model', 'api_key']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Missing fields: {missing_fields}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Missing required fields: {", ".join(missing_fields)}'}
            )

        agent_type = data['type']
        user_query = data['query']
        secondary_prompt = data.get('secondary_prompt', '')
        model_provider = data['model'].lower()
        license_key = data['api_key']  # This is actually the license key

        # Validate model provider
        if model_provider not in SUPPORTED_MODELS:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid model: {model_provider}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}'}
            )

        # Validate license and get appropriate API key
        api_key, error, license_meta = validate_and_get_api_key(license_key, model_provider, request_id)
        if error:
            error_content = {'error': error['error']}
            if error.get('code'):
                error_content['code'] = error['code']
            return JSONResponse(
                status_code=error['status'],
                content=error_content
            )

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']
        original_prompt = data.get('original_prompt', '').strip()

        image_urls, image_error = validate_reference_images(data.get('image_urls'), request_id)
        if image_error:
            return JSONResponse(
                status_code=image_error['status'],
                content={'error': image_error['error']}
            )

        if image_urls and not supports_vision(model_provider):
            logger.warning(f"[{request_id}] VALIDATION FAILED - {model_provider} cannot accept image input")
            usable = ", ".join(sorted(VISION_CAPABLE_PROVIDERS))
            return JSONResponse(
                status_code=400,
                content={
                    'error': f"The '{model_provider}' model cannot read images. "
                             f"Switch to one of: {usable}.",
                    'code': 'VISION_NOT_SUPPORTED',
                }
            )

        type_mapping = {
            'fix_grammar': 'Fix grammar: ',
            'simplify': 'Simplify: ',
            'complete': 'Complete: ',
            'emojify': 'Emojify: ',
            'translate': 'Translate: ',
            'make_longer': 'Make longer: ',
            'make_shorter': 'Make shorter: ',
            'enhance_with_ai': 'Enhance: ',
            'change_tone': 'Change tone: ',
            'general': '',
            'generate_email_template': '',
            'extract_image_text': '',
        }

        forced_agent_types = {
            'generate_email_template': 'email_template',
            'extract_image_text': 'text_extraction',
        }

        if agent_type not in type_mapping:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid type: {agent_type}")
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid type. Must be one of: {", ".join(type_mapping.keys())}'}
            )

        # Build query based on type with secondary_prompt support
        if agent_type == 'enhance_with_ai' and secondary_prompt:
            full_query = f"Enhance: {secondary_prompt}, Text: {user_query}"
        elif agent_type == 'change_tone' and secondary_prompt:
            full_query = f"Change tone to {secondary_prompt}: {user_query}"
        elif agent_type == 'translate' and secondary_prompt:
            full_query = f"Translate to {secondary_prompt}: {user_query}"
        else:
            full_query = f"{type_mapping[agent_type]}{user_query}"

        logger.info(f"[{request_id}] Streaming with agent type: {agent_type}, model: {model_provider}")

        input_chars = len(user_query)

        # Upfront credit reservation (skipped for BYOK users)
        upfront_reserved_scaled = 0
        if CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
            upfront_reserved_scaled = text_credits_scaled(input_chars, 0)
            if image_urls:
                upfront_reserved_scaled += IMAGE_CREDITS_SCALED * len(image_urls)
            ok, available_scaled = credit_manager.reserve(user_id, upfront_reserved_scaled)
            if not ok:
                logger.warning(
                    f"[{request_id}] Insufficient credits: needed={upfront_reserved_scaled} "
                    f"available={available_scaled}"
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "Insufficient credits",
                        "code": "INSUFFICIENT_CREDITS",
                        "available": available_scaled,
                        "needed": upfront_reserved_scaled,
                    }
                )

        def generate():
            output_chars = 0
            had_error = False
            error_code = None
            pending_images = {}
            resolved_image_count = 0
            extra_reserved_scaled = 0
            credit_limit_hit = False
            try:
                # Send initial event
                yield f"data: {json.dumps({'type': 'start', 'request_id': request_id})}\n\n"

                # Threshold (in raw total chars) past which we need 1 more credit
                # reserved. Initial threshold equals the input contribution we've
                # already covered with upfront_reserved_scaled.
                next_reserve_threshold = (upfront_reserved_scaled // CREDIT_SCALE) * 1000

                # Stream the AI response
                for chunk in process_user_query_stream(
                    full_query,
                    model_provider=model_provider,
                    api_key=api_key,
                    request_id=request_id,
                    user_id=user_id,
                    credit_manager=credit_manager if CREDIT_PRECHECK_ENABLED else None,
                    is_byok=is_byok,
                    image_urls=image_urls,
                    force_agent_type=forced_agent_types.get(agent_type),
                ):
                    # Check if client cancelled this request
                    if request_id in cancelled_requests:
                        cancelled_requests.pop(request_id, None)
                        logger.info(f"[{request_id}] Generator stopped early due to client abort (output_chars={output_chars})")
                        break
                    # Check for sentinel dict with pending images
                    if isinstance(chunk, dict) and "__pending_images__" in chunk:
                        pending_images = chunk["__pending_images__"]
                        continue
                    output_chars += len(chunk)

                    # Per-1000-chars incremental reservation (non-BYOK only)
                    if (
                        CREDIT_PRECHECK_ENABLED
                        and not is_byok
                        and credit_manager is not None
                    ):
                        while (input_chars + output_chars) > next_reserve_threshold:
                            # Hard ceiling — never lock more than the configured
                            # per-request maximum, even if the AI runs away.
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

                # Send completion event with pending image count
                total_duration = time.time() - start_time
                pending_count = len(pending_images)
                logger.info(f"[{request_id}] STREAM SUCCESS - Total time: {total_duration:.2f}s, pending images: {pending_count}")
                yield f"data: {json.dumps({'type': 'done', 'duration': round(total_duration, 2), 'pending_images': pending_count})}\n\n"

                # Poll for background image completions
                if pending_count > 0:
                    resolved = set()
                    poll_start = time.time()
                    max_wait = 120  # seconds

                    while len(resolved) < pending_count and (time.time() - poll_start) < max_wait:
                        # Check if client cancelled this request
                        if request_id in cancelled_requests:
                            cancelled_requests.pop(request_id, None)
                            logger.info(f"[{request_id}] Image polling stopped early due to client abort (resolved={len(resolved)}/{pending_count})")
                            break
                        time.sleep(0.5)
                        for pid, info in pending_images.items():
                            if pid in resolved:
                                continue
                            if info["status"] == "done":
                                resolved.add(pid)
                                resolved_image_count += 1
                                logger.info(f"[{request_id}] Image ready: {pid} -> {info['cdn_url']}")
                                yield f"data: {json.dumps({'type': 'image_ready', 'placeholder_id': pid, 'url': info['cdn_url'], 'prompt': info.get('prompt', '')})}\n\n"
                            elif info["status"] == "failed":
                                resolved.add(pid)
                                logger.error(f"[{request_id}] Image failed: {pid} -> {info['error']}")
                                yield f"data: {json.dumps({'type': 'image_failed', 'placeholder_id': pid, 'error': info['error']})}\n\n"

                    # Timeout any unresolved images
                    for pid in pending_images:
                        if pid not in resolved:
                            logger.error(f"[{request_id}] Image timed out: {pid}")
                            yield f"data: {json.dumps({'type': 'image_failed', 'placeholder_id': pid, 'error': 'Image generation timed out'})}\n\n"

                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

            except Exception as e:
                had_error = True
                error_code = type(e).__name__
                logger.error(f"[{request_id}] STREAM FAILED - Error: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            except GeneratorExit:
                # Client disconnected — generator is being closed.
                # Clean up the cancel entry if present.
                cancelled_requests.pop(request_id, None)
                logger.info(f"[{request_id}] Generator closed by client disconnect (output_chars={output_chars})")

            finally:
                # Total elapsed time — used as duration_ms in the telemetry webhook.
                duration_ms = int((time.time() - start_time) * 1000)

                # Fire credit deduction webhook (non-blocking) for non-BYOK users on success
                if not had_error and not is_byok:
                    # Determine operation type based on content generated
                    if resolved_image_count > 0 and output_chars > 0:
                        op_type = 'mixed'
                    elif resolved_image_count > 0:
                        op_type = 'image'
                    else:
                        op_type = 'text'

                    if CREDIT_PRECHECK_ENABLED and credit_manager is not None:
                        # Total reservation = upfront text + per-1000-chars increments
                        # + per-image (50 each) reserved inside generate_image_tool.
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
                            user_id,
                            input_chars,
                            output_chars,
                            op_type,
                            resolved_image_count,
                            provider=model_provider,
                            model=None,
                            duration_ms=duration_ms,
                            success=True,
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
                        user_id,
                        input_chars,
                        output_chars,
                        'text',
                        resolved_image_count,
                        provider=model_provider,
                        model=None,
                        duration_ms=duration_ms,
                        success=False,
                        error_code=error_code,
                    )
                # Save prompt to history (non-blocking) if original_prompt was provided
                if not had_error and original_prompt:
                    fire_prompt_history_save(user_id, original_prompt)

        return StreamingResponse(
            generate(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] STREAM FAILED - Error: {str(e)}")
        logger.error(f"[{request_id}] Total time: {total_duration:.2f}s")
        logger.info(f"[{request_id}] ================================")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post('/agent/autocomplete')
async def autocomplete_text(request: Request):
    """
    Lightweight endpoint for inline text autocomplete suggestions.
    Returns a short predicted text continuation based on cursor context.
    Uses fast/basic model variants for sub-second response times.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] ========== AUTOCOMPLETE REQUEST ==========")

    try:
        try:
            data = await request.json()
        except Exception:
            data = None

        if not data or not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={'error': 'Missing required fields: text_before, model, api_key'}
            )

        text_before = data.get('text_before', '')
        text_after = data.get('text_after', '')
        model_provider = data.get('model', '').lower()
        license_key = data.get('api_key', '')

        # Validate required fields
        if not text_before or not model_provider or not license_key:
            return JSONResponse(
                status_code=400,
                content={'error': 'Missing required fields: text_before, model, api_key'}
            )

        if model_provider not in SUPPORTED_MODELS:
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid model. Must be one of: {", ".join(SUPPORTED_MODELS)}'}
            )

        # Validate license and get API key
        api_key, error, license_meta = validate_and_get_api_key(license_key, model_provider, request_id)
        if error:
            error_content = {'error': error['error']}
            if error.get('code'):
                error_content['code'] = error['code']
            return JSONResponse(
                status_code=error['status'],
                content=error_content
            )

        user_id = license_meta['user_id']
        is_byok = license_meta['requires_user_api_keys']

        # Track whether we hold an autocomplete reservation so the outer
        # exception handler can release it if anything below raises.
        autocomplete_reserved = False

        # Reservation (skipped for BYOK)
        if CREDIT_PRECHECK_ENABLED and not is_byok and credit_manager is not None:
            ok, available = credit_manager.reserve(user_id, AUTOCOMPLETE_CREDITS_SCALED)
            if not ok:
                logger.warning(
                    f"[{request_id}] Autocomplete blocked: insufficient credits "
                    f"available={available}"
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "Insufficient credits",
                        "code": "INSUFFICIENT_CREDITS",
                        "available": available,
                        "needed": AUTOCOMPLETE_CREDITS_SCALED,
                    }
                )
            autocomplete_reserved = True

        # Truncate context to keep requests fast
        text_before = text_before[-500:]
        text_after = text_after[:200] if text_after else ""

        logger.info(f"[{request_id}] Autocomplete: model={model_provider}, context_before={len(text_before)} chars")

        suggestion = await run_in_threadpool(
            get_autocomplete_suggestion,
            text_before=text_before,
            text_after=text_after,
            model_provider=model_provider,
            api_key=api_key,
            request_id=request_id,
        )

        total_duration = time.time() - start_time
        logger.info(f"[{request_id}] Autocomplete SUCCESS - {total_duration:.2f}s, suggestion_len={len(suggestion)}")

        # Deduct credits + release reservation (non-BYOK users only)
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
                    # Empty suggestion: release reservation without firing webhook
                    credit_manager.schedule_release(
                        user_id, AUTOCOMPLETE_CREDITS_SCALED, delay_seconds=0,
                    )
                # Reservation has been handed off (or scheduled for release) —
                # don't double-release in the outer except handler.
                autocomplete_reserved = False
            elif suggestion:
                fire_credit_deduction(
                    user_id,
                    0,
                    0,
                    'autocomplete',
                    provider=model_provider,
                    model=None,
                    duration_ms=duration_ms,
                    success=True,
                )
                logger.info(f"[{request_id}] Autocomplete credit deduction fired for user={user_id}")

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'suggestion': suggestion,
            }
        )

    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"[{request_id}] Autocomplete FAILED - {total_duration:.2f}s: {str(e)}")
        # Release any held reservation so the user isn't blocked by a transient
        # AI failure until the 600s TTL expires.
        local_vars = locals()
        try:
            if (
                local_vars.get('autocomplete_reserved')
                and credit_manager is not None
                and local_vars.get('user_id')
            ):
                credit_manager.schedule_release(
                    local_vars['user_id'],
                    AUTOCOMPLETE_CREDITS_SCALED,
                    delay_seconds=0,
                )
        except Exception:
            pass

        try:
            if (
                local_vars.get('user_id')
                and local_vars.get('model_provider')
                and not local_vars.get('is_byok', False)
            ):
                fire_credit_deduction(
                    local_vars['user_id'],
                    0,
                    0,
                    'autocomplete',
                    provider=local_vars['model_provider'],
                    model=None,
                    duration_ms=int(total_duration * 1000),
                    success=False,
                    error_code=type(e).__name__,
                )
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post('/agent/abort')
async def abort_request(request: Request):
    """Cancel an in-flight stream request so the generator stops early."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    req_id = data.get('request_id', '')
    license_key = data.get('api_key', '')

    if not req_id:
        return JSONResponse(status_code=400, content={'error': 'Missing request_id'})

    # Validate license key to prevent unauthenticated abuse
    if not license_key or not redis_manager:
        return JSONResponse(status_code=401, content={'error': 'Unauthorized'})

    license_result = redis_manager.validate_license(license_key)
    if not license_result['valid']:
        return JSONResponse(status_code=401, content={'error': 'Unauthorized'})

    # Clean up stale entries to prevent memory leaks
    _cleanup_stale_cancellations()

    cancelled_requests[req_id] = time.time()
    logger.info(f"[{req_id}] Request cancelled by client")
    return JSONResponse(status_code=200, content={'success': True})


def _cleanup_stale_cancellations():
    """Remove cancelled_requests entries older than CANCEL_ENTRY_TTL."""
    now = time.time()
    stale_keys = [k for k, ts in cancelled_requests.items() if now - ts > CANCEL_ENTRY_TTL]
    for k in stale_keys:
        cancelled_requests.pop(k, None)
    if stale_keys:
        logger.info(f"Cleaned up {len(stale_keys)} stale cancellation entries")


if __name__ == '__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=False)
