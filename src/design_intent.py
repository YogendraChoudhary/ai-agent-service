"""
Deterministic intent classifier for Mode 8: Editable UI Design Engine.

Runs in < 0.05 ms inside src/request_plan.py **before** any model routing.
Returns a frozen DesignIntentDecision with no LLM calls, no network I/O, and no
stochastic decisions — only compiled regex state machines and boolean logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# 1. Start-anchored UI / design action verbs
# ---------------------------------------------------------------------------
_DESIGN_ACTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
    r"(?:design|create|build|make|generate|recreate|replicate|clone|convert|turn|draft|mock\s*up|wireframe|edit|update|modify|change)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 2. Target UI / editable canvas artifacts
# ---------------------------------------------------------------------------
_DESIGN_ARTIFACT_RE = re.compile(
    r"\b(?:"
    r"ui|user\s+interface|ux|wireframe|mockup|mock-up|"
    r"hero\s+section|landing\s+section|onboarding\s+card|feature\s+card|pricing\s+card|callout\s+card|card\s+layout|"
    r"banner\s+design|carousel\s+slide|modal\s+dialog|popup\s+layout|dashboard\s+widget|"
    r"editable\s+design|editable\s+ui|editable\s+card|editable\s+layout|editable\s+banner|canvas\s+design|"
    r"screen\s+design|app\s+screen|web\s+section"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 3. Multimodal vision recreation cues (user attaches a reference UI image)
# ---------------------------------------------------------------------------
_VISION_RECREATE_RE = re.compile(
    r"\b(?:"
    r"like\s+this\s+(?:image|photo|screenshot|design|ui|picture)|"
    r"from\s+this\s+(?:image|photo|screenshot|design|ui)|"
    r"match(?:ing)?\s+this\s+(?:design|layout|style|screenshot)|"
    r"recreate\s+this|replicate\s+this|clone\s+this|make\s+it\s+editable"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 4. Hard negative guards (prevent collisions with Mode 5 Code or Mode 7 Raster)
# ---------------------------------------------------------------------------
_NEGATIVE_CODE_GUARD_RE = re.compile(
    r"\b(?:python|typescript|javascript|rust|golang|sql|bash|dockerfile|react\s+hooks?|functions?|classes|algorithms?|regex|api\s+endpoints?|unit\s+tests?)\b",
    re.IGNORECASE,
)

_NEGATIVE_RASTER_PHOTO_GUARD_RE = re.compile(
    r"\b(?:photorealistic|oil\s+painting|watercolor|8k\s+photo|cinematic\s+lighting|portrait\s+of|landscape\s+photo|anime\s+style)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 5. Edit wording, used only when the request carries the design being edited (existing_design_ast). The editor
#    sends it only while a design is selected, so a request that reads like a change to it is an edit, whatever
#    verb it opens with. Plain chat ("hello", "thanks") and questions about the design ("what font is this?")
#    stay out.
# ---------------------------------------------------------------------------
_EDIT_VERB_RE = re.compile(
    r"^\s*(?:(?:please|pls|kindly|now|also|and|then|ok(?:ay)?|just|let'?s|can\s+you|could\s+you|would\s+you|"
    r"i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to)[\s,]+)*"
    r"(?:switch|swap|remove|delete|drop|add|insert|include|replace|translate|set|use|move|rename|recolou?r|"
    r"resize|increase|decrease|reduce|enlarge|shrink|align|cent(?:er|re)|darken|lighten|hide|show|put|give|apply|"
    r"rewrite|reword|shorten|lengthen|flip|invert|round|bold|italici[sz]e|underline|adjust|tweak|restyle|"
    r"redesign|redo|fix|duplicate|split|merge|stack|widen|narrow|tighten|bump|tone\s+down|try|"
    r"design|create|build|make|generate|recreate|convert|turn|edit|update|modify|change)\b",
    re.IGNORECASE,
)

_DESIGN_PART_RE = re.compile(
    r"\b(?:buttons?|cta|titles?|headlines?|headings?|header|subtitles?|sub-?headings?|copy|text|labels?|fonts?|"
    r"typography|colou?rs?|palette|theme|background|borders?|corners?|radius|padding|margins?|spacing|gaps?|"
    r"layout|alignment|illustrations?|icons?|arrows?|images?|dividers?|cards?|sections?|footer|logo|bubbles?|"
    r"shapes?|layers?|columns?|dark\s+mode|light\s+mode)\b",
    re.IGNORECASE,
)

_STYLE_CHANGE_RE = re.compile(
    r"\b(?:bigger|smaller|larger|darker|lighter|brighter|bolder|thinner|thicker|wider|narrower|taller|rounder|"
    r"softer|warmer|cooler|(?:more|less)\s+(?:space|spacing|padding|contrast|room|whitespace)|"
    r"too\s+(?:dark|light|big|small|bright|busy|plain|crowded|cramped|bold|faint|loud))\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(r"^\s*(?:what|why|how|which|who|when|where|is|are|does|do|did|was|were)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DesignIntentDecision:
    """Immutable result of the deterministic design intent classifier."""

    is_design_request: bool
    sub_mode: Optional[Literal["create_from_prompt", "create_from_vision", "edit_existing_ast"]] = None
    reason: str = ""


def detect_design_intent(
    query: str,
    operation_type: str = "general",
    has_image_attachment: bool = False,
    has_existing_design_ast: bool = False,
) -> DesignIntentDecision:
    """
    Deterministic, zero-LLM intent classifier for Mode 8 Editable UI Design.

    Priority ladder (first rule that fires wins):
      0. Explicit operation override or active AST edit.
      1. Negative guards (code / photorealistic raster images take precedence).
      1b. Design attached + edit wording (any edit verb, a design part, or a style change; not a question).
      2. Multimodal reference screenshot + UI/recreation cue.
      3. Start-anchored design verb + UI artifact keyword.
      4. Fallthrough: not a design request.

    Args:
        query:                  The raw user query string.
        operation_type:         Explicit request type from the API payload
                                (e.g. "generate_design", "edit_design", "general").
        has_image_attachment:   True when the request includes an attached image.
        has_existing_design_ast: True when the request body includes an existing
                                 DesignDocumentAST for iterative editing.

    Returns:
        DesignIntentDecision (frozen dataclass).
    """
    clean_q = (query or "").strip()

    # ------------------------------------------------------------------
    # Rule 0: Explicit operation override or active AST selection in the editor
    # ------------------------------------------------------------------
    if operation_type in ("generate_design", "edit_design"):
        mode: Literal["create_from_prompt", "create_from_vision", "edit_existing_ast"]
        if has_existing_design_ast:
            mode = "edit_existing_ast"
        elif has_image_attachment:
            mode = "create_from_vision"
        else:
            mode = "create_from_prompt"
        return DesignIntentDecision(True, mode, "explicit_operation_type")

    if has_existing_design_ast and _DESIGN_ACTION_RE.search(clean_q):
        return DesignIntentDecision(True, "edit_existing_ast", "active_design_ast_mutation")

    # ------------------------------------------------------------------
    # Rule 1: Negative Guards (code & photorealistic raster take precedence)
    # ------------------------------------------------------------------
    if _NEGATIVE_CODE_GUARD_RE.search(clean_q):
        return DesignIntentDecision(False, None, "blocked_by_code_guard")
    if _NEGATIVE_RASTER_PHOTO_GUARD_RE.search(clean_q):
        return DesignIntentDecision(False, None, "blocked_by_raster_photo_guard")

    # ------------------------------------------------------------------
    # Rule 1b: the selected design + wording that changes it ("Switch this to a light theme", "the button
    # should be green", "more padding"). Questions about the design are chat.
    # ------------------------------------------------------------------
    if has_existing_design_ast and not _QUESTION_RE.search(clean_q) and (
        _EDIT_VERB_RE.search(clean_q) or _DESIGN_PART_RE.search(clean_q) or _STYLE_CHANGE_RE.search(clean_q)
    ):
        return DesignIntentDecision(True, "edit_existing_ast", "design_ast_edit_wording")

    # ------------------------------------------------------------------
    # Rule 2: Multimodal reference screenshot + UI/recreation cue
    # ------------------------------------------------------------------
    if has_image_attachment and (
        _VISION_RECREATE_RE.search(clean_q) or _DESIGN_ARTIFACT_RE.search(clean_q)
    ):
        return DesignIntentDecision(True, "create_from_vision", "vision_attachment_plus_ui_cue")

    # ------------------------------------------------------------------
    # Rule 3: Start-anchored design verb + UI artifact keyword
    # ------------------------------------------------------------------
    if _DESIGN_ACTION_RE.search(clean_q) and _DESIGN_ARTIFACT_RE.search(clean_q):
        sub = "create_from_vision" if has_image_attachment else "create_from_prompt"
        return DesignIntentDecision(True, sub, "action_verb_plus_ui_artifact")

    return DesignIntentDecision(False, None, "no_design_pattern_matched")
