"""
Deterministic plan for a blog post.

How long it is, how many sections it has, which visuals it must contain, how many images may be generated and
whether web tools exist are all decided here, from the user's own words and options, BEFORE the model writes a
word. The outline assigns the visuals; the lint checks them. The model cannot "forget" a table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import settings
from src.request_plan import extract_word_count, image_count_requested, needs_web

LENGTH_WORDS = {"short": 800, "standard": 1450, "long": 2200}
DEFAULT_LENGTH = "standard"
# A brief can name any number ("a 50,000-word post"). Outside these bounds the post is not a blog post and the
# bill is not the user's to run up by accident, so the plan clamps the length and says so in `reasons`.
MIN_WORDS, MAX_WORDS = 300, 4000
BRIEF_MAX_CHARS = 5000      # "a topic or a short brief": the request is rejected above this, the regex ignores it

_NO_IMAGES = re.compile(
    r"\b(?:no|without|zero|skip|omit|exclude)\s+(?:any\s+|the\s+)?"
    r"(?:images?|pictures?|photos?|photographs?|illustrations?|visuals?|graphics?)\b|\btext[- ]only\b",
    re.I,
)


@dataclass(frozen=True)
class BlogRequest:
    """What the user asked for. `brief` and `style_notes` are the user's text: content, never instructions."""
    brief: str
    style_notes: str = ""
    length: Optional[str] = None        # short | standard | long
    images: Optional[int] = None        # None = auto
    tone: Optional[str] = None
    audience: Optional[str] = None

    @classmethod
    def from_options(cls, brief: str, style_notes: Optional[str], options: Optional[dict]) -> "BlogRequest":
        options = options or {}
        images = options.get("images")
        return cls(
            brief=(brief or "").strip(),
            style_notes=(style_notes or "").strip(),
            length=options.get("length"),
            images=images if isinstance(images, int) and not isinstance(images, bool) else None,
            tone=options.get("tone"),
            audience=options.get("audience"),
        )


@dataclass(frozen=True)
class BlogPlan:
    target_words: int
    length: str                 # short | standard | long: picks the few-shot example
    sections: int               # body sections; the intro and the conclusion are extra
    intro_words: int
    conclusion_words: int
    tables: int                 # at least this many
    quotes: int                 # at least this many
    images: int                 # exactly this many (each one costs money)
    use_web: bool
    max_prose_run: int          # words allowed between visual breaks
    images_asked: Optional[int] = None   # what the user explicitly asked for, before caps (None = they left it to us)
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def section_words(self) -> int:
        """Words across the body sections."""
        return self.target_words - self.intro_words - self.conclusion_words


def length_bucket(target_words: int) -> str:
    return "short" if target_words <= 1000 else "standard" if target_words <= 1600 else "long"


def section_count(target_words: int) -> int:
    return 4 if target_words <= 1000 else 5 if target_words <= 1600 else 6 if target_words <= 2200 else 7


def plan_blog(request: BlogRequest, *, image_available: bool = True) -> BlogPlan:
    """
    The plan for a request. `image_available` is whether any image provider can serve this account (decided by
    the caller, which knows the keys); without one the post is planned with no images and the lint does not ask
    for them.
    """
    reasons: Dict[str, str] = {}
    cap = max(0, int(settings.image_max_per_request))

    # length: the brief's own number, else the option, else standard
    asked = extract_word_count(request.brief)
    if asked is not None:
        target, reasons["length"] = asked, f"brief asked for {asked} words"
    else:
        bucket = request.length if request.length in LENGTH_WORDS else DEFAULT_LENGTH
        target, reasons["length"] = LENGTH_WORDS[bucket], f"option {bucket}" if request.length in LENGTH_WORDS else "default"
    clamped = max(MIN_WORDS, min(MAX_WORDS, target))
    if clamped != target:
        reasons["length"] += f", clamped to {clamped}"
    target = clamped

    sections = section_count(target)
    edge = min(180, max(80, round(target * 0.085 / 10) * 10))
    tables = 2 if target >= 2000 else 1
    quotes = 3 if target >= 2000 else 2

    # images: "no images" in the brief, else the brief's count, else the option, else 1 + one per 700 words
    asked: Optional[int] = None
    if _NO_IMAGES.search(request.brief):
        images, reasons["images"] = 0, "brief asked for none"
    elif image_count_requested(request.brief) > 0:
        asked = images = image_count_requested(request.brief)
        reasons["images"] = "count from the brief"
    elif request.images is not None:
        asked = images = request.images
        reasons["images"] = "option"
    else:
        images, reasons["images"] = 1 + target // 700, "auto"
    images = max(0, min(images, cap, sections))
    source = (settings.blog_image_source or "").strip().lower()
    if images and source != "generate":
        images, reasons["images"] = 0, f"BLOG_IMAGE_SOURCE={source or 'unset'}: no image backend"
    elif images and not image_available:
        images, reasons["images"] = 0, "no usable image provider"

    use_web = needs_web(request.brief)
    reasons["web"] = "cue in the brief" if use_web else "no freshness/research cue"
    return BlogPlan(
        target_words=target, length=length_bucket(target), sections=sections, intro_words=edge,
        conclusion_words=edge, tables=tables, quotes=quotes, images=images, use_web=use_web,
        max_prose_run=int(settings.blog_max_prose_run), images_asked=asked, reasons=reasons,
    )
