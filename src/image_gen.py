"""
Unified Image Generation & Editing Adapter

Single source of truth for:
  - Model selection and fallbacks (from src.models)
  - Parameter parsing and prompt regex triggers (ImageParams)
  - Magic-byte MIME sniffing and format conversion (via Pillow & src.s3_service)
  - Resolution & dimension tracking
  - Multi-turn prompt and image memory in Redis
"""

import io
import os
import re
import time
import uuid
import base64
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import requests
from config import settings
from src.models import (
    IMAGE_MODEL_IDS,
    IMAGE_MODEL_FALLBACKS,
    get_image_model_id,
    get_image_model_fallbacks,
)
from src.s3_service import (
    sniff_image_bytes,
    convert_image_format,
    get_image_dimensions,
    upload_image_from_bytes_detailed,
    upload_image_from_url_detailed,
)

logger = logging.getLogger(__name__)


_VALID_ASPECTS = ("1:1", "16:9", "9:16", "4:3", "3:4")
_VALID_QUALITIES = ("low", "medium", "high")
_VALID_RESOLUTIONS = ("1K", "2K", "4K")
_VALID_FORMATS = ("jpeg", "png", "webp")
ALLOWED_PARAM_KEYS = ("aspect_ratio", "quality", "resolution", "format", "transparent_bg", "size")


def sanitize_image_params(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a client-supplied params dict (explicit API field). Raises ValueError on
    unknown keys or out-of-range values so bad input is rejected instead of ignored.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("image_params must be an object")

    unknown = set(raw) - set(ALLOWED_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown image_params key(s): {', '.join(sorted(unknown))}")

    out: Dict[str, Any] = {}
    if raw.get("aspect_ratio") is not None:
        v = str(raw["aspect_ratio"])
        if v not in _VALID_ASPECTS:
            raise ValueError(f"aspect_ratio must be one of {', '.join(_VALID_ASPECTS)}")
        out["aspect_ratio"] = v
    if raw.get("quality") is not None:
        v = str(raw["quality"]).lower()
        if v not in _VALID_QUALITIES:
            raise ValueError(f"quality must be one of {', '.join(_VALID_QUALITIES)}")
        out["quality"] = v
    if raw.get("resolution") is not None:
        v = str(raw["resolution"]).upper()
        if v not in _VALID_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {', '.join(_VALID_RESOLUTIONS)}")
        out["resolution"] = v
    if raw.get("format") is not None:
        v = str(raw["format"]).lower().replace("jpg", "jpeg")
        if v not in _VALID_FORMATS:
            raise ValueError(f"format must be one of {', '.join(_VALID_FORMATS)}")
        out["format"] = v
    if raw.get("transparent_bg") is not None:
        out["transparent_bg"] = bool(raw["transparent_bg"])
    if raw.get("size") is not None:
        out["size"] = str(raw["size"])[:16]
    return out


@dataclass
class ImageParams:
    """
    Standardized image parameters across all providers.
    Precedence: explicit API field > prompt regex > inherited (previous image) > default.
    """
    aspect_ratio: str = "1:1"        # "1:1", "16:9", "9:16", "4:3", "3:4"
    quality: str = "medium"          # "low", "medium", "high"
    resolution: str = "1K"           # "1K", "2K", "4K"
    format: str = "jpeg"             # "jpeg", "png", "webp"
    transparent_bg: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_prompt_and_overrides(
        cls,
        prompt: str = "",
        explicit_params: Optional[Dict[str, Any]] = None,
        inherited_params: Optional[Dict[str, Any]] = None,
    ) -> "ImageParams":
        """
        Derive parameters. Only phrases that clearly ask for a setting count; style words an
        LLM habitually writes into prompts ("cinematic", "detailed", "story") do not.
        """
        explicit = explicit_params or {}
        inherited = inherited_params or {}
        p_lower = (prompt or "").lower()

        # 1. Defaults, then values inherited from the previous image
        aspect = inherited.get("aspect_ratio") or "1:1"
        quality = inherited.get("quality") or "medium"
        resolution = inherited.get("resolution") or "1K"
        fmt = inherited.get("format") or getattr(settings, "image_default_format", "jpeg").lower()
        transparent = bool(inherited.get("transparent_bg", False))

        # 2. Prompt triggers
        if re.search(r"\b(16:9|widescreen|wide[- ](?:format|screen)|banner|horizontal|"
                     r"in\s+landscape|landscape[- ](?:image|photo|picture|format|orientation|mode|layout|"
                     r"aspect|version|shape|ratio))\b", p_lower):
            aspect = "16:9"
        elif re.search(r"\b(9:16|vertical|instagram\s+stor(?:y|ies)|reels?|tiktok|in\s+portrait|"
                       r"portrait[- ](?:image|photo|picture|format|orientation|mode|layout|aspect|"
                       r"version|shape|ratio))\b", p_lower):
            aspect = "9:16"
        elif re.search(r"\b4:3\b", p_lower):
            aspect = "4:3"
        elif re.search(r"\b3:4\b", p_lower):
            aspect = "3:4"
        elif re.search(r"\b(square|1:1)\b", p_lower):
            aspect = "1:1"

        if re.search(r"\b(draft|low[- ]?res|low[- ]?quality|lowest\s+quality|fast\s+render)\b", p_lower):
            quality = "low"
        elif re.search(r"\b(high[- ]?quality|highest\s+quality|best\s+quality|maximum\s+quality|"
                       r"ultra[- ]?detailed|hq)\b", p_lower):
            quality = "high"

        if re.search(r"\b(4k|uhd|8k)\b", p_lower):
            resolution = "4K"
        elif re.search(r"\b(2k|qhd|1440p|high[- ]?res|highres)\b", p_lower):
            resolution = "2K"
        elif re.search(r"\b(1080p?|fhd|hd)\b", p_lower):
            resolution = "1K"

        if re.search(r"\b(transparent(\s+background|\s+bg)?|transparency|alpha\s+channel|cutout)\b", p_lower):
            fmt = "png"
            transparent = True
        elif re.search(r"\b(webp)\b", p_lower):
            fmt = "webp"
        elif re.search(r"\b(png)\b", p_lower):
            fmt = "png"
        elif re.search(r"\b(jpeg|jpg)\b", p_lower):
            fmt = "jpeg"

        # 3. Explicit overrides (highest precedence)
        if "size" in explicit:
            s = str(explicit["size"]).lower()
            if "1536x1024" in s:
                aspect = "16:9"
            elif "1024x1536" in s:
                aspect = "9:16"
            elif "1024x1024" in s:
                aspect = "1:1"

        if explicit.get("aspect_ratio"):
            aspect = str(explicit["aspect_ratio"])
        if explicit.get("quality"):
            quality = str(explicit["quality"]).lower()
        if explicit.get("resolution"):
            resolution = str(explicit["resolution"]).upper()
        if explicit.get("format"):
            fmt = str(explicit["format"]).lower().replace("jpg", "jpeg")
        if "transparent_bg" in explicit and explicit["transparent_bg"] is not None:
            transparent = bool(explicit["transparent_bg"])
        if transparent:
            fmt = "png"

        return cls(
            aspect_ratio=aspect,
            quality=quality,
            resolution=resolution,
            format=fmt,
            transparent_bg=transparent,
        )

    def to_openai_size(self) -> str:
        """Map aspect ratio to OpenAI supported sizes: 1024x1024, 1536x1024, 1024x1536."""
        if self.aspect_ratio in ("16:9", "4:3"):
            return "1536x1024"
        elif self.aspect_ratio in ("9:16", "3:4"):
            return "1024x1536"
        return "1024x1024"

    def to_openai_quality(self) -> str:
        """gpt-image-1 quality levels map one to one: low / medium / high."""
        return self.quality if self.quality in _VALID_QUALITIES else "medium"


def _is_model_not_found_error(err: Exception) -> bool:
    """True only for errors that say the MODEL is missing or retired (not bad params)."""
    msg = str(err).lower()
    if "model_not_found" in msg:
        return True
    return "model" in msg and any(
        t in msg
        for t in ("not found", "does not exist", "404", "deprecated", "no longer available", "decommissioned")
    )


_OPTIONAL_PARAM_NAMES = ("output_format", "background", "input_fidelity", "output_compression", "quality")


def _is_optional_param_error(err: Exception) -> bool:
    """True when the provider rejected one of the optional tuning params we send."""
    msg = str(err).lower()
    rejected = any(
        w in msg
        for w in ("unknown parameter", "unrecognized", "unexpected keyword", "unsupported", "not supported",
                  "extra inputs", "invalid value", "invalid_value")
    )
    return rejected and any(n in msg for n in _OPTIONAL_PARAM_NAMES)


# ======================================================================
# PROVIDER SELECTION
# ======================================================================

def resolve_image_provider(
    model_provider: str,
    api_key: Optional[str],
    is_byok: bool = False,
) -> Optional[Tuple[str, str]]:
    """
    Map the chat provider to (image_provider, api_key), or None if images cannot be
    generated for this request.

    Claude and DeepSeek have no image endpoint, so they use the platform's fallback
    provider key. That key is never used for bring-your-own-key users, whose requests
    must not be billed to the platform.
    """
    provider = (model_provider or "").lower().strip()
    if provider in ("claude", "deepseek"):
        if is_byok:
            return None
        fallback = getattr(settings, "image_default_provider_fallback", "gemini").lower()
        key = getattr(settings, f"default_{fallback}_api_key", None)
        return (fallback, key) if key else None
    if provider in ("openai", "gemini", "grok"):
        return (provider, api_key) if api_key else None
    return None


def image_route_available(model_provider: str, api_key: Optional[str], is_byok: bool = False) -> bool:
    return resolve_image_provider(model_provider, api_key, is_byok) is not None


# ======================================================================
# REDIS PROMPT MEMORY & REUSE
# ======================================================================

MAX_MODIFICATIONS = 3
_REDIS_RETRY_AFTER = 0.0
_REDIS_RETRY_COOLDOWN_SECONDS = 30.0


def _get_redis_client():
    """
    Shared Redis client from the process-wide singleton. After a failure the lookup is
    skipped for a cooldown, so an unreachable Redis cannot add its connect timeout to
    every image request.
    """
    global _REDIS_RETRY_AFTER
    if time.time() < _REDIS_RETRY_AFTER:
        return None
    try:
        from src.redis_manager import get_redis_manager
        return get_redis_manager().redis_client
    except Exception as e:
        _REDIS_RETRY_AFTER = time.time() + _REDIS_RETRY_COOLDOWN_SECONDS
        logger.warning(f"Redis unavailable for image context (retry in {_REDIS_RETRY_COOLDOWN_SECONDS:.0f}s): {e}")
        return None


def save_image_context(
    user_id: str,
    final_prompt: str,
    params: ImageParams,
    cdn_url: str,
    provider: str,
    document_id: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    request_id: Optional[str] = None,
    base_prompt: Optional[str] = None,
    modifications: Optional[List[str]] = None,
) -> None:
    """
    Store the last successful image so a follow-up can reuse it (TTL-bound).
    base_prompt and modifications keep chained edits bounded instead of nesting
    "A. Modification: B. Modification: C" forever.
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    if not user_id or not getattr(settings, "image_memory_enabled", True):
        return

    client = _get_redis_client()
    if not client:
        return

    import json
    ttl = getattr(settings, "image_memory_ttl_seconds", 3600)
    data = {
        "final_prompt": final_prompt,
        "base_prompt": base_prompt or final_prompt,
        "modifications": list(modifications or []),
        "params": params.to_dict(),
        "cdn_url": cdn_url,
        "provider": provider,
        "width": width,
        "height": height,
        "ts": time.time(),
    }

    try:
        payload = json.dumps(data)
        if document_id:
            client.set(f"image_context:{user_id}:doc:{document_id}", payload, ex=ttl)
        client.set(f"image_context:{user_id}:global", payload, ex=ttl)
        logger.info(f"{log_prefix}Stored image context for user={user_id} doc={document_id} (TTL={ttl}s)")
    except Exception as e:
        logger.warning(f"{log_prefix}Failed to store image context in Redis (non-fatal): {e}")


def get_previous_image_context(
    user_id: str,
    document_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch the previous image context. When the client names a document, only that
    document's entry is considered: falling back to the per-user entry would let an
    image from one document leak into another.
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    if not user_id or not getattr(settings, "image_memory_enabled", True):
        return None

    client = _get_redis_client()
    if not client:
        return None

    import json
    try:
        key = (
            f"image_context:{user_id}:doc:{document_id}"
            if document_id
            else f"image_context:{user_id}:global"
        )
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"{log_prefix}Failed to read image context from Redis: {e}")
        return None


def compose_followup_prompt(
    prev_ctx: Dict[str, Any], query: str, mode: str
) -> Tuple[str, str, List[str]]:
    """
    Build the prompt for a follow-up.

    mode "redo":  reuse the previous prompt verbatim.
    mode "delta": base prompt plus the last few modifications (bounded).
    Returns (final_prompt, base_prompt, modifications).
    """
    prev_final = prev_ctx.get("final_prompt", "") or ""
    base = prev_ctx.get("base_prompt") or prev_final
    mods = list(prev_ctx.get("modifications") or [])

    if mode == "redo":
        return prev_final or query, base, mods

    mods = (mods + [query.strip()])[-MAX_MODIFICATIONS:]
    final = f"{base}. Modifications: {'; '.join(mods)}" if base else query
    return final, base, mods


# ======================================================================
# PROVIDER IMPLEMENTATIONS
# ======================================================================

def _call_gemini_with_fallback(
    api_key: str,
    prompt: str,
    params: ImageParams,
    reference_image_bytes: Optional[bytes] = None,
    request_id: Optional[str] = None,
) -> Tuple[bytes, str, str]:
    """
    Call Google Gemini GenAI SDK with retry on fallback models.
    Returns (raw_bytes, mime_type, model_used).
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    from google import genai
    from google.genai import types

    primary_model = get_image_model_id("gemini")
    fallback_models = get_image_model_fallbacks("gemini")
    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    client = genai.Client(api_key=api_key)

    img_config = None
    try:
        img_config = types.ImageConfig(
            aspect_ratio=params.aspect_ratio,
            image_size=params.resolution,
        )
    except Exception as e:
        logger.debug(f"{log_prefix}types.ImageConfig not supported, omitting: {e}")

    gen_config_kwargs = {"response_modalities": ["IMAGE"]}
    if img_config:
        gen_config_kwargs["image_config"] = img_config

    config = types.GenerateContentConfig(**gen_config_kwargs)

    if reference_image_bytes:
        ref_mime, _ = sniff_image_bytes(reference_image_bytes)
        contents = [types.Part.from_bytes(data=reference_image_bytes, mime_type=ref_mime), prompt]
    else:
        contents = prompt

    last_error = None
    for idx, model_id in enumerate(models_to_try):
        try:
            logger.info(f"{log_prefix}Calling Gemini image generation (model={model_id})...")
            response = client.models.generate_content(model=model_id, contents=contents, config=config)

            if not response.candidates:
                block_reason = getattr(response, "prompt_feedback", None)
                raise Exception(f"Gemini image blocked by safety filters: {block_reason}")

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                finish = getattr(candidate, "finish_reason", "unknown")
                raise Exception(f"Gemini returned no image data (finish_reason: {finish})")

            for part in candidate.content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    mime = part.inline_data.mime_type or "image/png"
                    return part.inline_data.data, mime, model_id

            raise Exception("No inline image bytes in Gemini response candidate parts")

        except Exception as e:
            last_error = e
            if _is_model_not_found_error(e) and idx < len(models_to_try) - 1:
                logger.warning(
                    f"{log_prefix}Gemini model '{model_id}' failed ({e}). "
                    f"Retrying with fallback model '{models_to_try[idx + 1]}'..."
                )
                continue
            raise last_error

    raise last_error or Exception("Gemini generation failed on all models")


def _call_openai_with_fallback(
    api_key: str,
    prompt: str,
    params: ImageParams,
    reference_image_bytes: Optional[bytes] = None,
    request_id: Optional[str] = None,
) -> Tuple[bytes, str, str]:
    """
    Call OpenAI SDK with retry on fallback models.
    Sends size, quality (low/medium/high), output_format, and background for transparency.
    If the API rejects one of the optional tuning params, retries once with the bare request.
    Returns (raw_bytes, mime_type, model_used).
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    from openai import OpenAI

    primary_model = get_image_model_id("openai")
    fallback_models = get_image_model_fallbacks("openai")
    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    client = OpenAI(api_key=api_key)
    size = params.to_openai_size()
    extras: Dict[str, Any] = {
        "quality": params.to_openai_quality(),
        "output_format": params.format,
    }
    if params.transparent_bg:
        extras["background"] = "transparent"

    def _request(model_id: str, tuning: Dict[str, Any], first_attempt: bool):
        if reference_image_bytes:
            _, ref_ext = sniff_image_bytes(reference_image_bytes)
            buf = io.BytesIO(reference_image_bytes)
            buf.name = f"edit_input.{ref_ext or 'png'}"
            kwargs: Dict[str, Any] = dict(model=model_id, image=buf, prompt=prompt, size=size, n=1, **tuning)
            if first_attempt and tuning:
                kwargs["input_fidelity"] = "high"  # keep the user's subject recognisable
            return client.images.edit(**kwargs)
        return client.images.generate(model=model_id, prompt=prompt, size=size, n=1, **tuning)

    last_error = None
    for idx, model_id in enumerate(models_to_try):
        try:
            logger.info(
                f"{log_prefix}Calling OpenAI image {'edit' if reference_image_bytes else 'generation'} "
                f"(model={model_id}, size={size}, quality={extras['quality']}, format={params.format})..."
            )
            try:
                resp = _request(model_id, extras, True)
                used_format = params.format
            except Exception as param_err:
                if not _is_optional_param_error(param_err):
                    raise
                logger.warning(f"{log_prefix}OpenAI rejected tuning params ({param_err}); retrying bare request")
                resp = _request(model_id, {}, False)
                used_format = "png"

            data_obj = resp.data[0]
            b64_json = getattr(data_obj, "b64_json", None)
            url = getattr(data_obj, "url", None)

            if b64_json:
                return base64.b64decode(b64_json), f"image/{used_format}", model_id
            if url:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                return r.content, r.headers.get("Content-Type", "image/png"), model_id

            raise Exception("OpenAI returned neither b64_json nor url")

        except Exception as e:
            last_error = e
            if _is_model_not_found_error(e) and idx < len(models_to_try) - 1:
                logger.warning(
                    f"{log_prefix}OpenAI model '{model_id}' failed ({e}). "
                    f"Retrying with fallback '{models_to_try[idx + 1]}'..."
                )
                continue
            raise last_error

    raise last_error or Exception("OpenAI generation failed on all models")


GROK_EDIT_URL = "https://api.x.ai/v1/images/edits"
GROK_MAX_INPUT_BYTES = 20 * 1024 * 1024  # xAI documents a 20 MiB limit, jpg/jpeg/png only


def _grok_data_uri(reference_image_bytes: bytes) -> str:
    """xAI accepts jpg/png only: transcode anything else (e.g. webp) to PNG first."""
    mime, ext = sniff_image_bytes(reference_image_bytes)
    data = reference_image_bytes
    if ext not in ("png", "jpg"):
        data, mime, ext = convert_image_format(reference_image_bytes, "png")
    if len(data) > GROK_MAX_INPUT_BYTES:
        raise ValueError("Reference image exceeds xAI's 20 MiB input limit")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _grok_download(url: str) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "image/jpeg")


def _call_grok_edit(
    api_key: str,
    prompt: str,
    reference_image_bytes: bytes,
    models_to_try: List[str],
    request_id: Optional[str] = None,
) -> Tuple[bytes, str, str]:
    """
    Edit an image through xAI's documented JSON endpoint (POST /v1/images/edits).
    The OpenAI SDK's images.edit sends multipart/form-data, which this endpoint
    rejects (HTTP 415), so it is called directly with a JSON body.
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    data_uri = _grok_data_uri(reference_image_bytes)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = None
    for idx, model_id in enumerate(models_to_try):
        try:
            logger.info(f"{log_prefix}Calling Grok image edit (model={model_id})...")
            r = requests.post(
                GROK_EDIT_URL,
                headers=headers,
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "image": {"url": data_uri, "type": "image_url"},
                },
                timeout=120,
            )
            if r.status_code >= 400:
                raise Exception(f"xAI image edit failed (HTTP {r.status_code}) for model {model_id}: {r.text[:300]}")

            body = r.json()
            items = body.get("data") or []
            item = items[0] if items else body
            b64 = item.get("b64_json")
            url = item.get("url")
            if b64:
                return base64.b64decode(b64), "image/jpeg", model_id
            if url:
                content, mime = _grok_download(url)
                return content, mime, model_id
            raise Exception("xAI image edit response had neither url nor b64_json")

        except Exception as e:
            last_error = e
            if _is_model_not_found_error(e) and idx < len(models_to_try) - 1:
                logger.warning(f"{log_prefix}Grok edit model '{model_id}' failed ({e}); trying next...")
                continue
            raise last_error

    raise last_error or Exception("Grok edit failed on all models")


def _call_grok_with_fallback(
    api_key: str,
    prompt: str,
    params: ImageParams,
    reference_image_bytes: Optional[bytes] = None,
    request_id: Optional[str] = None,
) -> Tuple[bytes, str, str]:
    """
    xAI generation (OpenAI-compatible) or edit (JSON endpoint) with model fallback.
    Aspect ratio and resolution are not sent: xAI's docs list them as options but do not
    publish the field names, and an unknown field would fail the request.
    Returns (raw_bytes, mime_type, model_used).
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    primary_model = get_image_model_id("grok")
    fallback_models = get_image_model_fallbacks("grok")
    models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

    if reference_image_bytes:
        return _call_grok_edit(api_key, prompt, reference_image_bytes, models_to_try, request_id)

    from openai import OpenAI
    client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)

    last_error = None
    for idx, model_id in enumerate(models_to_try):
        try:
            logger.info(f"{log_prefix}Calling Grok image generation (model={model_id})...")
            resp = client.images.generate(model=model_id, prompt=prompt, n=1)
            content, mime = _grok_download(resp.data[0].url)
            return content, mime, model_id

        except Exception as e:
            last_error = e
            if _is_model_not_found_error(e) and idx < len(models_to_try) - 1:
                logger.warning(
                    f"{log_prefix}Grok model '{model_id}' failed ({e}). "
                    f"Retrying with fallback '{models_to_try[idx + 1]}'..."
                )
                continue
            raise last_error

    raise last_error or Exception("Grok generation failed on all models")


# ======================================================================
# UNIFIED ADAPTER ENTRY POINT
# ======================================================================

def generate_image_unified(
    prompt: str,
    model_provider: str = "openai",
    api_key: Optional[str] = None,
    explicit_params: Optional[Dict[str, Any]] = None,
    reference_image_bytes: Optional[bytes] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    document_id: Optional[str] = None,
    inherited_params: Optional[Dict[str, Any]] = None,
    context_extra: Optional[Dict[str, Any]] = None,
    is_byok: bool = False,
) -> Dict[str, Any]:
    """
    Unified entry point for image generation & editing across all providers.

    1. Resolves the image provider (Claude/DeepSeek use the platform fallback, never for BYOK)
    2. Parses ImageParams (explicit > prompt regex > inherited > default)
    3. Runs the provider call with model fallback
    4. A Grok edit that fails falls back to Gemini when a platform key exists; otherwise it
       raises. The reference image is never silently dropped.
    5. Sniffs real bytes, converts format if needed, uploads, measures dimensions
    6. Stores context for follow-ups (scoped to document_id when given)
    """
    start_time = time.time()
    log_prefix = f"[{request_id}] " if request_id else ""
    provider = (model_provider or "openai").lower().strip()

    if provider in ("claude", "deepseek"):
        resolved = resolve_image_provider(provider, api_key, is_byok)
        if not resolved:
            raise ValueError(
                f"Provider '{provider}' has no image endpoint and no platform fallback key is available"
            )
        logger.info(f"{log_prefix}Provider '{provider}' has no image endpoint; routing to '{resolved[0]}'")
        provider, api_key = resolved

    if not api_key:
        raise ValueError(f"No API key available for image provider '{provider}'")

    params = ImageParams.from_prompt_and_overrides(prompt, explicit_params, inherited_params)
    logger.info(
        f"{log_prefix}Image params: aspect={params.aspect_ratio}, quality={params.quality}, "
        f"res={params.resolution}, format={params.format}, edit={bool(reference_image_bytes)}"
    )

    adapter_t0 = time.time()
    result = None

    if provider == "grok" and reference_image_bytes:
        try:
            result = _call_grok_with_fallback(api_key, prompt, params, reference_image_bytes, request_id)
        except Exception as grok_err:
            gemini_key = getattr(settings, "default_gemini_api_key", None)
            if is_byok or not gemini_key:
                raise ValueError(f"Grok image edit failed and no fallback is available: {grok_err}") from grok_err
            logger.warning(f"{log_prefix}Grok edit failed ({grok_err}); falling back to Gemini")
            provider, api_key = "gemini", gemini_key

    if result is None:
        if provider == "gemini":
            result = _call_gemini_with_fallback(api_key, prompt, params, reference_image_bytes, request_id)
        elif provider == "openai":
            result = _call_openai_with_fallback(api_key, prompt, params, reference_image_bytes, request_id)
        elif provider == "grok":
            result = _call_grok_with_fallback(api_key, prompt, params, None, request_id)
        else:
            raise ValueError(f"Unsupported image provider: '{provider}'")

    raw_bytes, raw_mime, model_used = result
    adapter_duration = time.time() - adapter_t0
    logger.info(f"{log_prefix}Provider {provider} returned {len(raw_bytes)} bytes in {adapter_duration:.2f}s")

    upload_t0 = time.time()
    upload_res = upload_image_from_bytes_detailed(
        image_bytes=raw_bytes,
        content_type=raw_mime,
        user_id=user_id,
        request_id=request_id,
        target_format=params.format,
    )
    upload_duration = time.time() - upload_t0

    total_duration = time.time() - start_time
    cdn_url = upload_res["cdn_url"]
    width = upload_res["width"]
    height = upload_res["height"]

    logger.info(
        f"{log_prefix}Image ready: {cdn_url} | dims: {width}x{height} | format: {upload_res['format']} | "
        f"total {total_duration:.2f}s (adapter {adapter_duration:.2f}s, upload {upload_duration:.2f}s)"
    )

    extra = context_extra or {}
    if user_id:
        save_image_context(
            user_id=user_id,
            final_prompt=prompt,
            params=params,
            cdn_url=cdn_url,
            provider=provider,
            document_id=document_id,
            width=width,
            height=height,
            request_id=request_id,
            base_prompt=extra.get("base_prompt"),
            modifications=extra.get("modifications"),
        )

    return {
        "cdn_url": cdn_url,
        "width": width,
        "height": height,
        "format": upload_res["format"],
        "content_type": upload_res["content_type"],
        "provider": provider,
        "model": model_used,
        "prompt": prompt,
        "params": params.to_dict(),
        "timings": {
            "adapter_seconds": round(adapter_duration, 3),
            "upload_seconds": round(upload_duration, 3),
            "total_seconds": round(total_duration, 3),
        },
    }
