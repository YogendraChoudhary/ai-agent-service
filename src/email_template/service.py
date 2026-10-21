"""
Generate email templates: route -> prompt -> model -> validate -> (one repair) -> validated JSON.

The model call is injectable (`llm`) so tests and other callers never need a provider key.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import settings

from .categories import Category
from .prompts import repair_message, system_prompt, user_message
from .routing import resolve_category
from .schema import SOCIAL_MAX
from .tone import Issue
from .validate import Context, Result, format_errors, validate

logger = logging.getLogger(__name__)

# (system_prompt, messages) -> reply text. messages alternate user/assistant, ending with a user turn.
LLM = Callable[[str, list[dict]], str]


class EmailProviderError(RuntimeError):
    """The model call itself failed (bad key, provider down, SDK error). Not the model's output."""


class EmailGenerationError(Exception):
    """No variant passed validation, even after repair."""

    def __init__(self, errors: list[Issue]):
        super().__init__(format_errors(errors))
        self.errors = errors


@dataclass
class GenerateRequest:
    brief: str
    category: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    language: str = "en-GB"
    length: str = "standard"
    allowed_merge_tags: tuple[str, ...] = ()
    links: dict[str, str] = field(default_factory=dict)       # key -> URL (URLs never go to the model)
    assets: dict[str, str] = field(default_factory=dict)      # key -> short description
    sender: dict[str, str] = field(default_factory=dict)
    org_name: Optional[str] = None
    brand_kit_id: Optional[str] = None
    variants: int = 1
    sign_off: Optional[str] = None                             # "signature" | "none" | None (category default)
    social_links: dict[str, str] = field(default_factory=dict)  # platform -> URL, shown in the footer in this order
    theme: Optional[dict] = None                                # accent_color, hero / footer section styles


# "a banner image" is one picture, but the shared counter sees two image nouns.
_COMPOUND_IMAGE_RE = re.compile(
    r"\b(?:banner|hero|header|cover|feature|featured|thumbnail)\s+(image|picture|photo|graphic|visual)(s?)\b", re.I)
_HERO_CUE_RE = re.compile(r"\b(banner|hero|header image|cover|at the top|top of the email)\b", re.I)
_INLINE_CUE_RE = re.compile(r"\b(inline|in the body|in the middle|below the text|before the button)\b", re.I)


def requested_image_slots(brief: str, category: Category) -> tuple[str, ...]:
    """
    Slots the brief asks to fill, capped at what the category has. Counting reuses the shared detector, so
    only an explicit request ("add an image of...", "with a banner") opens a slot. One image goes to the slot
    the brief names, else to the category's first slot (hero when it has one).
    """
    from src.request_plan import extract_image_count

    count = extract_image_count(_COMPOUND_IMAGE_RE.sub(r"\1\2", brief or ""))
    if count <= 0 or not category.image_slots:
        return ()
    if count >= 2:
        return category.image_slots[:2]
    wants_hero, wants_inline = _HERO_CUE_RE.search(brief), _INLINE_CUE_RE.search(brief)
    if wants_inline and not wants_hero and "inline" in category.image_slots:
        return ("inline",)
    if wants_hero and "hero" in category.image_slots:
        return ("hero",)
    return category.image_slots[:1]


def _model_request(req: GenerateRequest, image_slots: tuple[str, ...], sign_off: Optional[str]) -> dict:
    data = {
        "brief": req.brief,
        "audience": req.audience,
        "language": req.language,
        "length": req.length,
        "allowed_merge_tags": list(req.allowed_merge_tags),
        "links": sorted(req.links),
        "assets": req.assets,
        "sender": req.sender or None,
        "org_name": req.org_name,
        "image_slots": list(image_slots),
        "sign_off": sign_off,
    }
    return {k: v for k, v in data.items() if v not in (None, {}, "")}


def _source_texts(req: GenerateRequest) -> tuple[str, ...]:
    return (req.brief, req.audience or "", req.org_name or "", *req.sender.values(), *req.assets.values())


def is_failed_run(response) -> bool:
    """True for an agno run that failed: status ERROR (non-streamed runs) or a RunError event."""
    from agno.agent import RunEvent
    from agno.run.base import RunStatus

    return getattr(response, "status", None) == RunStatus.error or getattr(response, "event", None) == RunEvent.run_error


def make_agno_llm(model_provider: str, api_key: str, request_id: Optional[str] = None,
                  is_byok: bool = True, temperature: float = 0.3) -> LLM:
    """The default model call: an agno agent on the advanced tier, no tools, plain text (not Markdown) out."""
    from agno.agent import Agent
    from agno.models.message import Message

    from src.text_processing_agent import _agent_error_message, get_model

    def call(system: str, messages: list[dict]) -> str:
        model = get_model(model_provider, api_key, use_advanced=True, request_id=request_id,
                          needs_tools=False, temperature=temperature, cache_model=not is_byok)
        agent = Agent(name="Email Template Writer", model=model, instructions=system, markdown=False,
                      telemetry=False)
        history = [Message(role=m["role"], content=m["content"]) for m in messages]
        response = agent.run(history, stream=False)
        # agno does not raise on provider failures (bad key, outage): the run comes back with status ERROR and
        # the error text as content. Never mistake that for the model's reply.
        if is_failed_run(response):
            raise EmailProviderError(_agent_error_message(model_provider, getattr(response, "content", None)))
        return response.content if isinstance(response.content, str) else str(response.content or "")

    return call


def _generate_one(llm: LLM, system: str, first_user: str, ctx: Context, request_id: Optional[str]) -> Result:
    """One variant. Provider errors propagate; generate() decides whether other variants can carry on."""
    messages = [{"role": "user", "content": first_user}]
    reply = llm(system, messages)
    result = validate(reply, ctx)
    if result.ok or not settings.email_ai_repair_enabled:
        return result
    logger.info(f"[{request_id}] email template needs repair: {[e.kind for e in result.errors]}")
    messages += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": repair_message(reply, format_errors(result.errors))}]
    repaired = validate(llm(system, messages), ctx)
    fixed = ", ".join(sorted({e.kind for e in result.errors}))
    repaired.warnings.append(Issue("repaired", f"The first draft was corrected automatically ({fixed})."))
    return repaired


def effective_sign_off(req: GenerateRequest) -> Optional[str]:
    """
    An email ends with either a signature or a footer with social icons, never both. Social links therefore
    mean no signature; asking for both is a request error.
    """
    if req.social_links:
        if req.sign_off == "signature":
            raise ValueError("Choose either a signature or footer social links: an email cannot have both.")
        return "none"
    return req.sign_off


def generate(req: GenerateRequest, llm: LLM, request_id: Optional[str] = None) -> dict:
    """Validated templates for the request. Raises ValueError (bad input) or EmailGenerationError."""
    sign_off = effective_sign_off(req)
    category, routing = resolve_category(req.brief, req.category)
    tone = req.tone or category.default_tone
    image_slots = requested_image_slots(req.brief, category)
    ctx = Context(
        category=category, routing=routing, tone=tone, language=req.language,
        allowed_merge_tags=frozenset(req.allowed_merge_tags), link_keys=frozenset(req.links),
        asset_keys=frozenset(req.assets), image_slots=image_slots, brand_kit_id=req.brand_kit_id,
        source_texts=_source_texts(req), sign_off=sign_off,
        social=tuple(req.social_links)[:SOCIAL_MAX], theme=req.theme,
    )
    system = system_prompt(category, tone)
    first_user = user_message(_model_request(req, image_slots, sign_off))
    n = max(1, min(req.variants, settings.email_ai_max_variants))

    def one(_):
        try:
            return _generate_one(llm, system, first_user, ctx, request_id)
        except EmailProviderError as e:
            return e

    if n == 1:
        outcomes = [one(0)]
    else:
        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="email_variant") as pool:
            outcomes = list(pool.map(one, range(n)))
    results = [o for o in outcomes if isinstance(o, Result)]
    provider_errors = [o for o in outcomes if isinstance(o, EmailProviderError)]
    if not results:  # every variant failed at the provider: report that, not a validation failure
        raise provider_errors[0]

    good = [r for r in results if r.ok]
    logger.info(f"[{request_id}] email templates: category={category.id} routing={routing} tone={tone} "
                f"ok={len(good)}/{n}")
    if not good:
        raise EmailGenerationError(results[0].errors)
    return {
        "resolved_category": category.id,
        "routing": routing,
        "tone": tone,
        "templates": [
            {"template": r.template, "warnings": [w.as_dict() for w in r.warnings], "fill_slots": r.fill_slots}
            for r in good
        ],
    }
