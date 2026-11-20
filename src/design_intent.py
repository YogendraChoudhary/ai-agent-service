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
    r"screen\s+design|app\s+screen|web\s+section|"
    # whole pages and screens people ask to have designed (live audit 2026-11-20: "Design a landing page for a coffee
    # shop", "Design a login screen", "Make a signup form" went to plain text)
    r"landing\s+page|home\s*page|web\s*page|website|web\s+site|pricing\s+page|product\s+page|about\s+page|"
    r"contact\s+page|login\s+(?:screen|page|form)|sign[- ]?(?:up|in)\s+(?:screen|page|form)|checkout\s+(?:screen|page)|"
    r"settings\s+(?:screen|page)|profile\s+(?:screen|page|card)|(?:mobile|app)\s+(?:screen|ui)|"
    r"(?:signup|login|contact|subscribe|newsletter|booking|registration)\s+form"
    r")\b(?!\s+(?:copy|content|text|description|plan|strategy|seo|ideas?|names?|outline|brief|audit|review|"
    r"designers?|roles?|skills?|research(?:ers?)?|teams?|engineers?|writ(?:er|ing)|jobs?|positions?|interviews?|courses?|"
    r"careers?|principles|trends|best\s+practices|laws|feedback|findings|issues|bugs|tests?|testing|case\s+stud(?:y|ies))\b)",
    re.IGNORECASE,
)

# A text deliverable named before the UI word ("Draft an email to the team about the new UI", "Create a job description
# for a senior UX designer", "Make a list of UX interview questions"): the request is that text, not a design.
_TEXT_DELIVERABLE_RE = re.compile(
    r"\b(?:emails?|lists?|plans?|descriptions?|notes|resumes?|cv|cover\s+letters?|letters?|posts?|blog|articles?|"
    r"reports?|summar(?:y|ies)|questions|checklists?|proposals?|briefs?|case\s+stud(?:y|ies)|presentations?|strategy|"
    r"guides?|outlines?|faqs?|specs?|specifications?|requirements|surveys?|essays?|tweets?|threads?|captions?|"
    r"announcements?|newsletters?|scripts?|stor(?:y|ies)|poems?|bios?|quiz(?:zes)?|agendas?|memos?|messages?|"
    r"release\s+notes|changelogs?|documentation|docs|tickets?|user\s+stories|personas?|interview\s+guides?)\b",
    re.IGNORECASE,
)


# Code asked for, not code words as content: "write a python function to validate the form" is code, "Add an FAQ about our
# JavaScript bootcamp" edits the design.
_CODE_REQUEST_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:write|create|generate|build|make|give\s+me|implement|code|"
    r"convert\s+(?:it|this|the\s+design)\s+(?:to|into))\b[^.?!]{0,40}?\b(?:python|typescript|javascript|rust|golang|sql|"
    r"bash|dockerfile|react|html|css|tailwind|jsx|tsx|vue|svelte|angular|flutter|swiftui|kotlin|functions?|script|code|"
    r"class|regex|api\s+endpoints?|unit\s+tests?|algorithm)\b", re.IGNORECASE)


# A separate piece of writing or an explanation ("Write a blog post about dogs", "Explain the color theory behind this",
# "Brainstorm names for the app"), unless it names a part of the design ("Write a tagline for the header").
_OTHER_TASK_RE = re.compile(
    r"^\s*(?:(?:please|pls|kindly|now|ok(?:ay)?|can\s+you|could\s+you|would\s+you)[\s,]+)*"
    r"(?:write|draft|compose|summari[sz]e|explain|describe|tell\s+me|brainstorm|suggest|list|give\s+me\s+(?:some\s+)?ideas|"
    r"research|compare|analy[sz]e|review|critique|proofread|find|search|look\s+up|calculate|count|check|define|"
    r"bash|shell|terminal|command)\b", re.IGNORECASE)


def _asks_for_something_else(query: str) -> bool:
    """With a design attached: code, a new picture or a separate piece of writing, not a change to the design."""
    if _CODE_REQUEST_RE.search(query):
        return True
    if _OTHER_TASK_RE.search(query) and not _DESIGN_PART_RE.search(query):
        return True
    from src.intent import detect_image_intent
    return detect_image_intent(query)["kind"] == "generate"


def _asks_for_design(query: str) -> bool:
    """A design verb at the start and a UI artifact as what is made, not a text deliverable that only mentions UI."""
    if not _DESIGN_ACTION_RE.search(query):
        return False
    artifact = _DESIGN_ARTIFACT_RE.search(query)
    if not artifact:
        return False
    deliverable = _TEXT_DELIVERABLE_RE.search(query[:artifact.start()])
    return deliverable is None

# ---------------------------------------------------------------------------
# 3. Multimodal vision recreation cues (user attaches a reference UI image)
# ---------------------------------------------------------------------------
_VISION_RECREATE_RE = re.compile(
    r"\b(?:"
    r"like\s+this\s+(?:image|photo|screenshot|design|ui|picture)|"
    r"from\s+this\s+(?:image|photo|screenshot|design|ui)|"
    r"match(?:ing)?\s+this\s+(?:design|layout|style|screenshot)|"
    r"recreate\s+this|replicate\s+this|clone\s+this|make\s+(?:it|this)\s+(?:into\s+(?:an?\s+)?)?editable|"
    # a screenshot as the layout to copy (live audit 2026-11-20: these went to plain text or to an image edit)
    r"rebuild\s+this|copy\s+this\s+(?:page|layout|design|ui|screen|website|site|template)|(?:the\s+)?same\s+layout|"
    r"use\s+(?:this|it)\s+as\s+(?:a|the|my)\s+(?:template|layout|base|starting\s+point)|(?:in|with)\s+this\s+layout"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 4. Hard negative guards (prevent collisions with Mode 5 Code or Mode 7 Raster)
# ---------------------------------------------------------------------------
_NEGATIVE_CODE_GUARD_RE = re.compile(
    r"\b(?:python|typescript|javascript|rust|golang|sql|bash|dockerfile|react(?:\s+hooks?)?|functions?|classes|algorithms?|regex|api\s+endpoints?|unit\s+tests?|"
    r"html|css|tailwind|jsx|tsx|vue|svelte|angular|next\.?js|flutter|swiftui|kotlin)\b",
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
# Questions that are requests ("Could it be more minimal?", "Is it possible to make the button green?", "How about a darker
# background?"), and plain acknowledgements that ask for nothing ("thanks", "looks great").
_REQUEST_QUESTION_RE = re.compile(
    r"^\s*(?:(?:could|can|would|will)\s+(?:it|this|that|the)\b|is\s+it\s+possible|how\s+about|what\s+about|"
    r"why\s+not|what\s+if\s+(?:we|you|it)|do\s+you\s+mind)", re.IGNORECASE)
# Verb-less edits of the attached design: "More modern", "Less clutter", "Pizzas instead of burgers", "Same thing but for a
# sushi place", "Ok now for a coffee shop", "In French please", "The price should read $12".
_INTERJECTION_RE = re.compile(r"^\s*(?:(?:hmm+|ok(?:ay)?|well|so|now|and|also|actually|maybe|alright|right|nice|good)[\s,.!]+)+",
                              re.IGNORECASE)
_IMPLICIT_EDIT_RE = re.compile(
    r"^\s*(?:more|less|too|a\s+bit|slightly|much|way|even|bit)\b|\binstead\s+of\b|\bbut\s+(?:for|with|in|as)\b|"
    r"^\s*(?:same|similar)\b|^\s*for\s+an?\b|"
    r"\bin\s+(?:english|french|spanish|german|italian|portuguese|dutch|hindi|arabic|chinese|japanese|korean|russian|turkish)\b|"
    r"\bshould\s+(?:be|say|read|look|feel|show)\b", re.IGNORECASE)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(?:thanks?(?:\s+you)?|thank\s+you|ty|ok(?:ay)?|great|cool|nice|perfect|awesome|good|hi|hello|hey|yes|no|"
    r"wow|lol|hmm+|looks?\s+(?:great|good|nice|perfect)|love\s+it|that'?s\s+(?:great|good|nice|perfect))(?:[\s,!.]+(?:thanks?|"
    r"thank\s+you|a\s+lot|so\s+much))*[\s!.?]*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 5b. A screenshot as a template: "Design a page like this for pizzas", "based on this", "in this style". Needs a
#     design verb at the start and a page-type noun, so "make a picture like this" stays an image request.
# ---------------------------------------------------------------------------
_TEMPLATE_CUE_RE = re.compile(
    r"\b(?:like\s+this|like\s+that|similar\s+to\s+this|based\s+on\s+this|in\s+this\s+style|inspired\s+by\s+this|"
    r"same\s+(?:layout|style|design|structure|look)|this\s+(?:layout|style|template|screenshot|design))\b",
    re.IGNORECASE,
)
_PAGE_NOUN_RE = re.compile(
    r"\b(?:page|pages|screen|screens|website|site|homepage|landing|layout|section|listing|menu|ui|interface|"
    r"card|cards|banner|design|template|dashboard|app)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 5c. Same layout, new content: the screenshot is a template, not the copy. Its words are then not required in the
#     design (live 2026-11-19: "...but for a page of pizzas ... instead of burgers" was checked for every burger
#     name, which cost a 48 s repair asking to put them back and returned 24 false warnings).
# ---------------------------------------------------------------------------
_NEW_CONTENT_RE = re.compile(
    r"\b(?:instead\s+of|rather\s+than|but\s+(?:for|with|about|showing|featuring)|inspired\s+by|"
    r"(?:in|with)\s+(?:the\s+)?(?:same|this|that)\s+(?:style|layout|look|structure|format|design)|"
    r"same\s+(?:style|layout|look|structure|format|design)\s+(?:but|for|with)|"
    r"(?:like|similar\s+to|based\s+on)\s+this\b[^.?!]{0,40}?\b(?:for|about|featuring|showing|with)\b|"
    r"(?:this|the)\s+(?:design|layout|page|screenshot|template)\s+for\s+(?:my|our|a|an)\b|"
    r"(?:change|replace|swap)\s+(?:the\s+|all\s+the\s+)?(?:content|text|copy|items|products|dishes|names))",
    re.IGNORECASE,
)


def wants_new_content(query: str) -> bool:
    """A screenshot request that keeps the layout and style but asks for different content."""
    return bool(_NEW_CONTENT_RE.search(query or ""))


# ---------------------------------------------------------------------------
# 6. Photo fill: with a design selected, a request for real photos/images/pictures goes to the image generation
#    flow (fill the design's ImageNode placeholders), not to a layout edit, which cannot make photos. The wording
#    rule lives with the fill code: src/design_canvas/images.asks_for_new_photos.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignIntentDecision:
    """Immutable result of the deterministic design intent classifier."""

    is_design_request: bool
    sub_mode: Optional[Literal["create_from_prompt", "create_from_vision", "edit_existing_ast",
                               "fill_design_images"]] = None
    reason: str = ""


def detect_design_intent(
    query: str,
    operation_type: str = "general",
    has_image_attachment: bool = False,
    has_existing_design_ast: bool = False,
    existing_design_ast=None,
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
        existing_design_ast:    That design itself (dict or JSON), when the caller has it: a selected photo
                                 ("make it a pizza") is then a photo request, and a selected section without photos
                                 is not ("add a photo here" adds a slot: a layout edit).

    Returns:
        DesignIntentDecision (frozen dataclass).
    """
    clean_q = (query or "").strip()

    # ------------------------------------------------------------------
    # Rule F: a selected design + a request for photos. Checked before every edit rule: "add real photos" would
    # otherwise be a layout edit. Questions ("can you add photos?") are answered as chat like other questions.
    # ------------------------------------------------------------------
    if has_existing_design_ast and not _QUESTION_RE.search(clean_q):
        from src.design_canvas.images import asks_for_new_photos, selected_photo_scope
        from src.design_canvas.text_edit import split_selection
        instruction, _ = split_selection(clean_q)
        _, holds = selected_photo_scope(existing_design_ast, clean_q) if existing_design_ast is not None else (None, None)
        # 2026-11-20: a photo word anywhere ("Change the button text to 'Upload images'", "the image captions") is not a
        # photo request; the photo must be what the request acts on (images.asks_for_new_photos).
        if asks_for_new_photos(instruction, holds):
            return DesignIntentDecision(True, "fill_design_images",
                                        "design_photo_fill_wording" if holds != "image" else "design_selected_photo")

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
    # Rule 1a: a design is attached (the editor sends it only while one is selected). Anything that is not a real question
    # or a plain acknowledgement changes it, whatever the wording: "More modern", "Pizzas instead of burgers", "Could it
    # be more minimal?", "Add an FAQ about our JavaScript bootcamp" (live audit 2026-11-20: these became chat, and the
    # design was dropped). Checked before the code and photo guards, which are about requests for something new.
    # ------------------------------------------------------------------
    if has_existing_design_ast and clean_q and not _ACKNOWLEDGEMENT_RE.match(clean_q) and (
            _REQUEST_QUESTION_RE.match(clean_q) or not _QUESTION_RE.search(clean_q)) and not _asks_for_something_else(clean_q):
        # Code or photo words as content only when an edit verb opens the request ("Add an FAQ about our JavaScript bootcamp");
        # otherwise they mean code or a picture, as before ("Count words in text in python").
        guarded = _NEGATIVE_CODE_GUARD_RE.search(clean_q) or _NEGATIVE_RASTER_PHOTO_GUARD_RE.search(clean_q)
        explicit = (_EDIT_VERB_RE.search(clean_q) or (not guarded and (
            _DESIGN_PART_RE.search(clean_q) or _STYLE_CHANGE_RE.search(clean_q))))
        implicit = len(clean_q.split()) <= 12 and (_REQUEST_QUESTION_RE.match(clean_q) or _IMPLICIT_EDIT_RE.search(
            _INTERJECTION_RE.sub("", clean_q))) and not _NEGATIVE_CODE_GUARD_RE.search(clean_q) \
            and not _NEGATIVE_RASTER_PHOTO_GUARD_RE.search(clean_q)
        if explicit or implicit:
            return DesignIntentDecision(True, "edit_existing_ast", "design_ast_edit_wording")

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

    # Rule 2b: a screenshot used as a template ("Design a page like this for pizzas").
    if (has_image_attachment and _DESIGN_ACTION_RE.search(clean_q) and _TEMPLATE_CUE_RE.search(clean_q)
            and _PAGE_NOUN_RE.search(clean_q)):
        return DesignIntentDecision(True, "create_from_vision", "vision_attachment_as_template")

    # ------------------------------------------------------------------
    # Rule 3: Start-anchored design verb + UI artifact keyword
    # ------------------------------------------------------------------
    if _asks_for_design(clean_q):
        sub = "create_from_vision" if has_image_attachment else "create_from_prompt"
        return DesignIntentDecision(True, sub, "action_verb_plus_ui_artifact")

    return DesignIntentDecision(False, None, "no_design_pattern_matched")
