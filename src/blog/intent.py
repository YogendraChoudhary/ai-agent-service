"""
Fail-safe blog intent for free-form chat ("write a blog post about ...").

The declared request `type` decides the agent; text is only guessed from once the logs prove the regex, so this
ships off (BLOG_INTENT_REGEX_ENABLED) and in shadow mode first (BLOG_INTENT_SHADOW_MODE), like code intent.
Anchored at the START of the request after the shared polite-filler prefix, so text that merely mentions a blog
("my blog post about X needs a fix", "summarise this article") never matches. Image and code intent are checked
by the caller before this runs.
"""

from __future__ import annotations

import re
from typing import Dict

from src.intent import POLITE_PREFIX

from .plan import BRIEF_MAX_CHARS

# Up to three free words before the form ("an SEO blog post", "an informative blog post", "a blog article"), and the usual ways
# to name the subject after it ("for my bakery about…", "titled…", "explaining how…"): live audit 2026-11-20 missed all of these.
_ADJECTIVES = r"(?:[\w'-]+\s+){0,3}?"
_FORM = r"(?:blog(?:\s+(?:post|article|entry))?|article|guide|long[- ]form\s+(?:piece|article|post))"
_BLOG = re.compile(
    POLITE_PREFIX + rf"(?:write|draft|create)\s+(?:me\s+)?(?:an?\s+){_ADJECTIVES}{_FORM}\s+"
    r"(?:about|on|covering|for|titled|called|named|explaining|describing|that\s+(?:explains|covers|describes))\b",
    re.I,
)


def detect_blog_intent(text: str) -> Dict[str, str]:
    text = text or ""
    if len(text) <= BRIEF_MAX_CHARS and _BLOG.search(text):
        return {"kind": "blog", "rule": "anchored_blog"}
    return {"kind": "other", "rule": "no_blog_cue"}
