"""
Email category registry.

A category is data: its skeleton, which blocks it requires and allows, its default tone, which image slots it
may use, footer rules, and the regex that routes a brief to it. Adding a category means adding an entry here
plus one few-shot example in examples/<id>.json; the engine and the shared prompt do not change.

A category is offered (to routing and to the API) only when `enabled` and its example exists. The others are
defined so their patterns and rules are reviewed together, and are switched on once their example is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .schema import BLOCK_TYPES, PROMO_BLOCKS

EXAMPLES_DIR = Path(__file__).parent / "examples"

_ALL_BLOCKS = frozenset(BLOCK_TYPES)
_CORE = _ALL_BLOCKS - PROMO_BLOCKS          # letter-style emails: no launch hero, no feature columns
_NO_IMAGE = _CORE - {"image"}


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    skeleton: tuple[str, ...]           # recommended order; the model may drop optional blocks or repeat text blocks
    required: frozenset[str]
    allowed: frozenset[str]
    default_tone: str
    image_slots: tuple[str, ...]        # which fixed image slots this category may use
    needs_button: bool = False          # the email asks the reader to do something
    needs_signature: bool = False
    bulk: bool = False                  # sent to lists: footer must carry unsubscribe
    pattern: Optional[str] = None       # routing regex (case-insensitive); None = never matched by regex
    enabled: bool = True


# Order matters: routing ties are broken by this order (urgent and sensitive categories first).
_CATEGORIES: tuple[Category, ...] = (
    Category(
        id="incident_apology", label="Incident / apology",
        skeleton=("header", "greeting", "paragraph", "list", "paragraph", "signature", "footer"),
        required=frozenset({"header", "greeting", "signature", "footer"}),
        allowed=_NO_IMAGE,
        default_tone="formal", image_slots=(), needs_signature=True,
        pattern=r"\b(apolog(y|ies|ise|ize)|sorry|outage|downtime|disruption|incident|service (issue|interruption))\b",
        enabled=False,
    ),
    Category(
        id="confirmation", label="Confirmation (order, booking, registration)",
        skeleton=("header", "heading", "paragraph", "key_value", "button", "footer"),
        required=frozenset({"header", "key_value", "footer"}),
        allowed=_NO_IMAGE,
        default_tone="professional", image_slots=(),
        pattern=(r"\border (#|no\.|number|confirmation)|\byour order\b|\breceipt\b|\binvoice\b"
                 r"|\bbooking confirmation\b|\bregistration confirmation\b|\bpayment (received|confirmation)\b"
                 r"|\bpassword reset\b"),
        enabled=False,
    ),
    Category(
        id="reminder_deadline", label="Reminder / deadline",
        skeleton=("header", "greeting", "notice", "paragraph", "key_value", "image", "button", "signature", "footer"),
        required=frozenset({"header", "greeting", "key_value", "footer"}),
        allowed=_CORE,
        default_tone="formal", image_slots=("inline",), needs_button=True, needs_signature=True,
        pattern=r"\b(reminder|remind|deadline|due (date|by|on)|last date|expir(es|y|ing)|overdue)\b",
    ),
    Category(
        id="policy_notice", label="Policy / notice",
        skeleton=("header", "greeting", "paragraph", "key_value", "list", "image", "button", "paragraph",
                  "signature", "footer"),
        required=frozenset({"header", "greeting", "key_value", "signature", "footer"}),
        allowed=_CORE,
        default_tone="formal", image_slots=("inline",), needs_signature=True,
        pattern=(r"\b(polic(y|ies)|guidelines?|compliance|regulations?|terms (of service|and conditions)"
                 r"|code of conduct|procedures?)\b"),
    ),
    Category(
        id="meeting_request", label="Meeting request",
        skeleton=("header", "greeting", "paragraph", "key_value", "paragraph", "signature", "footer"),
        required=frozenset({"header", "greeting", "key_value", "signature", "footer"}),
        allowed=_NO_IMAGE,
        default_tone="professional", image_slots=(), needs_signature=True,
        pattern=r"\b(meeting|schedule a (call|meeting)|book a (call|slot)|your availability|calendar invite)\b",
        enabled=False,
    ),
    Category(
        id="follow_up", label="Follow-up / thank you",
        skeleton=("header", "greeting", "paragraph", "paragraph", "signature", "footer"),
        required=frozenset({"header", "greeting", "signature", "footer"}),
        allowed=_NO_IMAGE,
        default_tone="professional_warm", image_slots=(), needs_signature=True,
        pattern=r"\b(follow[- ]?up|following up|thank you for|thanks for|feedback)\b",
        enabled=False,
    ),
    Category(
        id="invitation", label="Invitation",
        skeleton=("header", "image", "heading", "greeting", "paragraph", "key_value", "image", "button",
                  "signature", "footer"),
        required=frozenset({"header", "key_value", "button", "footer"}),
        allowed=_CORE,
        default_tone="professional_warm", image_slots=("hero", "inline"), needs_button=True,
        pattern=(r"\b(invit(e|ed|ation|ing)|rsvp|save the date|join us (for|at)|webinar|ceremony|gala"
                 r"|seminar|conference)\b"),
    ),
    Category(
        id="onboarding_welcome", label="Onboarding / welcome",
        skeleton=("header", "image", "heading", "greeting", "paragraph", "list", "button", "signature", "footer"),
        required=frozenset({"header", "list", "footer"}),
        allowed=_CORE,
        default_tone="professional_warm", image_slots=("hero", "inline"), needs_button=True,
        pattern=r"\b(welcome|onboard(ing)?|first day|new (employee|joiner|member|customer)s?)\b",
        enabled=False,
    ),
    Category(
        id="product_launch", label="New product launch",
        skeleton=("header", "hero_banner", "image", "features", "paragraph", "button", "footer"),
        required=frozenset({"header", "hero_banner", "footer"}),
        allowed=_ALL_BLOCKS - {"greeting", "notice", "key_value"},
        default_tone="professional", image_slots=("hero", "inline"), needs_button=True, bulk=True,
        pattern=(r"\b(launch(es|ed|ing)?|new product|now available|introducing|introduc(e|es) (our|the|a) new"
                 r"|unveil(s|ed|ing)?)\b"),
    ),
    Category(
        id="feature_update", label="New feature / product update",
        skeleton=("header", "hero_banner", "image", "paragraph", "features", "list", "button", "footer"),
        required=frozenset({"header", "hero_banner", "footer"}),
        allowed=_ALL_BLOCKS - {"notice"},
        default_tone="professional", image_slots=("hero", "inline"), needs_button=True, bulk=True,
        pattern=(r"\b(new features?|feature (update|release)|roll(ing|s|ed)? out|rollout|release notes"
                 r"|product update|now supports?|what'?s new|improvements?)\b"),
    ),
    Category(
        id="announcement", label="Announcement",
        skeleton=("header", "image", "heading", "paragraph", "paragraph", "image", "button", "signature", "footer"),
        required=frozenset({"header", "heading", "footer"}),
        allowed=_CORE,
        default_tone="professional", image_slots=("hero", "inline"),
        pattern=(r"\b(announc(e|es|ement|ing)|pleased to (share|inform)|relocat(e|es|ion|ing)"
                 r"|office (move|closure)|will be closed|new (office|branch|appointment|campus))\b"),
    ),
    Category(
        id="newsletter", label="Newsletter",
        skeleton=("header", "image", "heading", "paragraph", "heading", "paragraph", "button", "footer"),
        required=frozenset({"header", "heading", "footer"}),
        allowed=_ALL_BLOCKS,
        default_tone="professional", image_slots=("hero", "inline"), bulk=True,
        pattern=r"\b(newsletter|digest|roundup|bulletin|(monthly|quarterly|weekly) update)\b",
        enabled=False,
    ),
    Category(
        id="general", label="General",
        skeleton=("header", "greeting", "paragraph", "paragraph", "signature", "footer"),
        required=frozenset({"header", "footer"}),
        allowed=_CORE,
        default_tone="professional", image_slots=("hero", "inline"),
        pattern=None,
    ),
)

GENERAL = "general"
ALL_CATEGORIES: dict[str, Category] = {c.id: c for c in _CATEGORIES}


def example_path(category_id: str) -> Path:
    return EXAMPLES_DIR / f"{category_id}.json"


@lru_cache(maxsize=None)
def load_example(category_id: str) -> Optional[dict]:
    """The category's few-shot example ({"input": ..., "output": ...}), or None if it has none yet."""
    path = example_path(category_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def enabled_categories() -> dict[str, Category]:
    """Categories offered to users: switched on and with a few-shot example. Order is routing precedence."""
    return {cid: c for cid, c in ALL_CATEGORIES.items() if c.enabled and example_path(cid).exists()}


def get_category(category_id: str) -> Category:
    """An enabled category by id; ValueError for unknown or disabled ones (the API turns that into a 400)."""
    enabled = enabled_categories()
    if category_id not in enabled:
        raise ValueError(f"Unknown email category '{category_id}'. Available: {', '.join(enabled)}")
    return enabled[category_id]
