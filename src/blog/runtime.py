"""
Binds the blog pipeline to this app: the provider candidates and keys of the request, the image tool (provider
routing, credits, background upload), web-source tracking and observability. service.py stays free of all of it.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from config import OUTLINE_MODEL_PATTERN, settings

from .llm import AgnoLLM, model_id_for
from .media import get_source
from .plan import BlogRequest, plan_blog
from .service import generate_blog_stream
from .sources import WebSources

logger = logging.getLogger(__name__)

# Defaults for a post's pictures; the user's explicit image_params still win.
BLOG_IMAGE_DEFAULTS = {"aspect_ratio": "16:9", "quality": "medium"}


def parse_outline_model(value: Optional[str]) -> Optional[Tuple[str, bool]]:
    """BLOG_OUTLINE_MODEL "provider[:tier]" -> (provider, advanced). Tier defaults to basic: a fast model is the point."""
    match = re.fullmatch(OUTLINE_MODEL_PATTERN, (value or "").strip().lower())
    return (match.group(1), match.group(2) == "advanced") if match else None


def outline_chain(setting: Optional[str], candidates: Sequence, available: Optional[Iterable[str]]) -> Optional[List]:
    """
    The providers to try for the FIRST outline attempt: the configured fast model, then the drafting candidates as
    fallbacks. None when nothing is configured, when the request may not use that provider (`available` is what the
    account's keys allow: a BYOK account's own keys, otherwise the platform's; with no router, only the one provider
    in use), when the model is blocked, or when it is the model that drafts anyway.
    """
    from src.models import BLOCKED_MODELS
    from src.task_router import Candidate

    parsed = parse_outline_model(setting)
    if parsed is None or not candidates:
        return None
    provider, advanced = parsed
    allowed = set(available) if available is not None else {candidates[0].provider}
    fast = Candidate(provider, advanced)
    if provider not in allowed or model_id_for(fast) in BLOCKED_MODELS:
        return None
    if candidates[0] == fast:
        return None
    return [fast] + list(candidates)


def stream_blog(
    query: str,
    instruction: Optional[str],
    blog_options: Optional[dict],
    *,
    candidates: Sequence,
    key_for: Callable[[str], Optional[str]],
    is_byok: bool,
    user_id: Optional[str],
    credit_manager,
    pending_images: dict,
    image_params: Optional[dict],
    document_id: Optional[str],
    request_id: Optional[str],
    report_model: bool = False,
    available_providers: Optional[Iterable[str]] = None,
) -> Iterator[Union[str, dict]]:
    """
    The blog pipeline for one request. Pictures it starts are registered in `pending_images`, the same dict (and the
    same completion callback) the chat path uses, so the stream endpoint's image_ready / billing flow applies as is.
    """
    from src import text_processing_agent as tpa
    from src.request_plan import ToolBudget

    sources = WebSources()
    # One context for the whole request: how pictures are drawn, the web budgets every part shares, and the
    # record of which pages were read (the lint needs it to allow an attributed quote).
    ctx = {
        "document_id": document_id,
        "explicit_params": {**BLOG_IMAGE_DEFAULTS, **(image_params or {})},
        "user_query": query,
        "web_sources": sources,
    }
    request = BlogRequest.from_options(query, instruction, blog_options)

    # Pictures are drawn by the first candidate whose account can draw them (a chat provider without an image
    # endpoint falls back to the platform's image key; a BYOK account on such a provider cannot draw at all).
    drawer = None
    for candidate in candidates:
        key = key_for(candidate.provider)
        if key and tpa._image_tool_available(candidate.provider, key, is_byok, ctx):
            drawer = (candidate, key)
            break
    plan = plan_blog(request, image_available=drawer is not None)
    logger.info(f"[{request_id}] blog plan: {plan.target_words} words, {plan.sections} sections, {plan.tables} table(s), "
                f"{plan.quotes} quote(s), {plan.images} image(s), web={plan.use_web} | {plan.reasons}")

    media = None
    if plan.images and drawer is not None:
        candidate, key = drawer
        tool = tpa._build_image_tool(candidate.provider, key, pending_images, request_id, user_id, credit_manager,
                                     is_byok, ctx, ToolBudget(plan.images, "image"))
        media = get_source(settings.blog_image_source, tool)

    fast_outline = outline_chain(settings.blog_outline_model, candidates, available_providers)
    llm = AgnoLLM(candidates, key_for, outline_candidates=fast_outline, request_id=request_id, user_id=user_id,
                  is_byok=is_byok, image_ctx=ctx)
    if fast_outline:
        logger.info(f"[{request_id}] blog outline model: {fast_outline[0].provider} "
                    f"({'advanced' if fast_outline[0].advanced else 'basic'}), then the drafting model")
    reported = handed_over = False
    for chunk in generate_blog_stream(request, plan, llm, media, sources=sources, request_id=request_id):
        if report_model and not reported:
            reported = True
            # Attribution follows the model that writes the post, which is known before it has run a single part.
            if llm.drafting_model() is not None:
                yield {"__model_used__": llm.drafting_model()}
        if isinstance(chunk, dict) and "__pictures_started__" in chunk:
            # Plumbing between the pipeline and this hook, never sent on. The title can go out before any picture exists,
            # so the next chunk may be a whole part away: hand the record of the pictures over now.
            if not handed_over and any(not str(k).startswith("__") for k in pending_images):
                handed_over = True
                yield {"__pending_images__": pending_images}
            continue
        if isinstance(chunk, str) and not handed_over and any(not str(k).startswith("__") for k in pending_images):
            # Pictures are billed work that starts before any text. Hand the dict over now, not at the end, so a
            # client that leaves or a credit limit that stops the stream still releases what was reserved.
            handed_over = True
            yield {"__pending_images__": pending_images}
        yield chunk
