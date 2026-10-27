"""
Deterministic request planning for the free-form ("smart global") agent.

The LLM writes text and picture prompts. It does not decide, per request:
  * how many images exist             -> extract_image_count()   (user's words, capped in code)
  * whether web tools are available   -> needs_web()             (freshness / research cues)
  * which model tier answers          -> needs_advanced_model()

Everything here is regex on the USER's own text. Tools the plan does not grant are never attached, so
the model cannot call them; tools it does grant are budgeted (see ToolBudget).

Each decision fails toward the cheaper, safer side only when no cue is present, and every gate has a
kill switch in config.py so behaviour can be rolled back without a deploy.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import settings

# ---------------------------------------------------------------------------
# Image count
# ---------------------------------------------------------------------------

_IMAGE_NOUN = (
    r"(?:images?|pictures?|photos?|photographs?|illustrations?|infographics?|graphics?|diagrams?|"
    r"banners?|logos?|icons?|thumbnails?|visuals?|artworks?|drawings?|paintings?|renders?)"
)
_IMAGE_NOUN_RE = re.compile(rf"^{_IMAGE_NOUN}$", re.I)
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "1": 1,
    "two": 2, "pair": 2, "couple": 2, "both": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "few": 3, "several": 3, "some": 3, "multiple": 3, "many": 3,
}
# "image generators", "image compression": the noun is the topic, not a request for pictures.
_TOPIC_FOLLOW = {
    "generator", "generators", "generation", "compression", "file", "files", "format", "formats", "size",
    "sizes", "editing", "editor", "editors", "quality", "search", "processing", "recognition", "model",
    "models", "tool", "tools", "app", "apps", "software", "library", "libraries", "resolution", "seo",
    "optimization", "optimisation", "classification", "analysis", "detection", "dataset", "datasets",
    "hosting", "upload", "uploads", "url", "urls", "tag", "tags", "attribute", "attributes", "design",
    "designer", "designers", "design", "editing",
}
# Words that, standing between a quantifier and an image noun, show the noun is a topic ("an article about image").
_TOPIC_BREAK = {"about", "on", "of", "for", "regarding", "to", "that", "which", "with", "in", "from", "using", "by"}
_LEAD_VERBS = {"with", "include", "including", "includes", "add", "adding", "plus", "featuring", "containing",
               "and", "insert", "embed", "generate", "create", "make", "produce", "needs", "need", "want",
               "illustrate", "illustrated", "show", "showing"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.!?;:\n]")


def extract_image_count(text: Optional[str]) -> int:
    """
    Pictures the user asked for inside a text request ("an article with 3 images" -> 3).
    0 when none were asked for. Capped at IMAGE_MAX_PER_REQUEST. See image_count_requested for the raw number.
    """
    return min(image_count_requested(text), max(0, int(settings.image_max_per_request)))


def image_count_requested(text: Optional[str]) -> int:
    toks = _TOKEN_RE.findall(text or "")
    low = [t.lower() for t in toks]
    total = 0
    i = 0
    while i < len(low):
        tok = low[i]
        if not _IMAGE_NOUN_RE.match(tok):
            i += 1
            continue
        nxt = low[i + 1] if i + 1 < len(low) else ""
        if nxt in _TOPIC_FOLLOW:
            i += 1
            continue
        # A picture is only requested when a request verb ("with", "include", "add", "create"...) leads
        # into it; "how to take a good photo" or "top 5 photos of 2020" are topics, not requests.
        lead_window = low[max(0, i - 7):i]
        if not any(w in _LEAD_VERBS for w in lead_window):
            i += 1
            continue
        count = 0
        # look back up to 3 tokens for a quantifier ("2", "a couple of", "several")
        for back in range(1, 4):
            j = i - back
            if j < 0 or low[j] in ".!?;:\n":
                break
            word = low[j]
            if word == "of" and j > 0 and low[j - 1] in ("couple", "pair", "few"):
                continue
            if word in _TOPIC_BREAK:
                break
            if word.isdigit():
                count = int(word)
                break
            if word in _NUMBER_WORDS:
                count = _NUMBER_WORDS[word]
                break
        if count == 0:
            # bare plural ("with images") gets the default; bare singular ("add image") is one
            count = max(1, int(settings.image_default_plural_count)) if tok.endswith("s") else 1
        total += count
        i += 1
    return total


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

_WEB_CUES = re.compile(
    r"(?:\b(?:latest|current|currently|recent|recently|newest|up[- ]to[- ]date|breaking|today|tonight|"
    r"yesterday|this\s+(?:week|month|year|quarter)|last\s+(?:week|month|year|\d+\s+years?)|"
    r"news|headlines?|trending|trends?|statistics|stats|market\s+(?:data|size|share|cap)|forecast|"
    r"research|search(?:\s+the)?\s+(?:web|internet|online)|google|look\s*up|find\s+out|fact[- ]check|"
    r"sources?|cite|citations?|references?|according\s+to|price\s+of|prices|stock|earnings|"
    r"revenue|competitors?|benchmarks?|versus|vs\.?|compare|comparison|who\s+is\s+the\s+current|"
    r"as\s+of)\b|\b20[2-9]\d\b|https?://|www\.)",
    re.I,
)


def needs_web(text: Optional[str]) -> bool:
    """True when the request needs fresh or sourced facts. Kill switch: WEB_TOOLS_GATE_ENABLED=false."""
    if not settings.web_tools_gate_enabled:
        return True
    return bool(_WEB_CUES.search(text or ""))


# ---------------------------------------------------------------------------
# Model tier
# ---------------------------------------------------------------------------

_LONGFORM = re.compile(
    r"\b(?:article|blog(?:\s+post)?|essay|report|white\s?paper|newsletter|case\s+study|guide|tutorial|"
    r"documentation|proposal|press\s+release|chapter|ebook|e-book|speech|presentation|analysis|"
    r"strategy|business\s+plan|lesson\s+plan|thesis|dissertation|long[- ]form|in[- ]depth|detailed|"
    r"comprehensive|step[- ]by[- ]step|code|script|function|sql|regex)\b",
    re.I,
)
_WORDCOUNT = re.compile(r"\b(\d[\d,]{2,5})[- ]?(?:words?|word\s+count)\b", re.I)


def extract_word_count(text: Optional[str]) -> Optional[int]:
    """The length the user asked for in their own words ("a 1,500-word post" -> 1500), else None."""
    m = _WORDCOUNT.search(text or "")
    return int(m.group(1).replace(",", "")) if m else None


def needs_advanced_model(text: Optional[str], use_web: bool, image_count: int) -> bool:
    """
    Reasoning-grade model for tool use, long-form or code requests, long inputs; the cheap fast
    tier for short conversational writes. Kill switch: MODEL_TIER_ROUTING_ENABLED=false.
    """
    if not settings.model_tier_routing_enabled:
        return True
    t = text or ""
    if use_web or image_count > 0:
        return True
    if len(t) > int(settings.tier_long_query_chars):
        return True
    if _LONGFORM.search(t):
        return True
    m = _WORDCOUNT.search(t)
    if m and int(m.group(1).replace(",", "")) >= 500:
        return True
    return False


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestPlan:
    image_count: int = 0          # images this request may generate (0 = the image tool is not attached)
    image_requested: int = 0      # what the user asked for before the cap
    use_web: bool = False         # web search/read tools attached
    advanced_model: bool = True   # reasoning tier vs fast tier
    execution_mode: str = "general"  # "general" | "design_canvas" | …
    design_sub_mode: Optional[str] = None  # "create_from_prompt" | "create_from_vision" | "edit_existing_ast"
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def use_tools(self) -> bool:
        return self.use_web or self.image_count > 0

    @property
    def image_capped(self) -> bool:
        return self.image_requested > self.image_count


def plan_design_request(
    user_text: Optional[str],
    operation_type: str = "general",
    has_image_attachment: bool = False,
    has_existing_design_ast: bool = False,
) -> Optional[RequestPlan]:
    """
    Mode 8 Design Canvas plan, or None when this is not a design request (or the engine is off).

    Zero LLM calls. The stream calls it before the image router, the code classifier and the blog regex,
    so a design request never reaches them; /agent/stream calls it again to reserve the design price.
    Image and web tools are disabled and the model tier is forced to advanced.
    """
    if not settings.design_canvas_enabled:
        return None
    from src.design_intent import detect_design_intent
    decision = detect_design_intent(
        query=user_text or "",
        operation_type=operation_type,
        has_image_attachment=has_image_attachment,
        has_existing_design_ast=has_existing_design_ast,
    )
    if not decision.is_design_request:
        return None
    return RequestPlan(
        image_count=0,
        image_requested=0,
        use_web=False,
        advanced_model=True,
        execution_mode="design_canvas",
        design_sub_mode=decision.sub_mode,
        reasons={
            "images": "disabled=design_canvas_mode",
            "web": "disabled=design_canvas_mode",
            "tier": "advanced",
            "design_intent": decision.reason,
            "design_sub_mode": decision.sub_mode or "",
        },
    )


def plan_smart_request(
    user_text: Optional[str],
    operation_type: str = "general",
    has_image_attachment: bool = False,
    has_existing_design_ast: bool = False,
) -> RequestPlan:
    """
    Deterministic request plan.

    Priority 1 (before any other gate): Mode 8 Design Canvas detection (plan_design_request).
    """
    design = plan_design_request(user_text, operation_type, has_image_attachment, has_existing_design_ast)
    if design is not None:
        return design

    requested = image_count_requested(user_text)
    count = min(requested, max(0, int(settings.image_max_per_request)))
    web = needs_web(user_text)
    adv = needs_advanced_model(user_text, web, count)
    return RequestPlan(
        image_count=count,
        image_requested=requested,
        use_web=web,
        advanced_model=adv,
        execution_mode="general",
        reasons={
            "images": f"requested={requested} granted={count}",
            "web": "cue" if web else "no freshness/research cue",
            "tier": "advanced" if adv else "fast",
        },
    )


# ---------------------------------------------------------------------------
# Per-request budgets
# ---------------------------------------------------------------------------

class ToolBudget:
    """Thread-safe counter. take() grants a call or refuses it; the tool then returns an error string."""

    def __init__(self, limit: int, label: str = "tool"):
        self.limit = max(0, int(limit))
        self.label = label
        self._used = 0
        self._lock = threading.Lock()

    def take(self, n: int = 1) -> bool:
        with self._lock:
            if self._used + n > self.limit:
                return False
            self._used += n
            return True

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self._used)
