"""
Image Intent Routing Engine

Regex router that decides whether a query is an image generation / edit request
that can skip the LLM agent and go straight to the image adapter.

Design rules (all of them exist because the opposite caused a real misroute):

  * Fail safe. Anything ambiguous returns "none" and falls through to the normal
    agent router, which still handles images through the image tool (slower, but
    correct). A false positive generates and charges for an unwanted image; a
    false negative only costs latency.
  * Verbs are anchored to the START of the request (after polite filler such as
    "please" or "can you"), so text that merely mentions images ("write about how
    to create a logo") never matches.
  * Follow-ups ("make it night", "regenerate") and edits that point at an
    earlier image only count when the caller confirms an image exists to refer
    to. Without one the result is "none" with requires_context=True, and the
    caller may look up the previous image and call again with
    has_previous_image=True.
  * The caller must additionally gate on the request type: only the free-form
    "general" chat should ever reach this module. Text operations such as
    fix_grammar or simplify carry user text that can mention images freely.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

IMAGE_NOUNS = (
    r"(?:images?|pictures?|photos?|photographs?|logos?|posters?|banners?|icons?|"
    r"illustrations?|artworks?|sketch(?:es)?|wallpapers?|drawings?|paintings?|"
    r"graphics?|infographics?|avatars?|thumbnails?|renders?|portraits?)"
)

CREATION_VERBS = r"(?:draw|create|generate|design|render|illustrate|paint|sketch|produce|make|craft)"

EDIT_VERBS = (
    r"(?:refine|redraw|restyle|recolou?r|re-colou?r|retouch|touch\s+up|polish|enhance|improve|"
    r"upscale|modify|edit|expand|sharpen|brighten|darken|colou?rize|vectorize)"
)

# Verbs that, when the user has attached an image, almost always mean "edit it".
REF_EDIT_VERBS = (
    r"(?:refine|redraw|restyle|turn|transform|edit|modify|change|replace|recolou?r|re-colou?r|"
    r"upscale|enhance|improve|touch\s+up|retouch|convert|add|make|give|apply|use|remove|erase|"
    r"fix|blur|sharpen|brighten|darken|put|adjust)"
)

# Optional polite filler before the real verb: "please", "can you", "i want you to"...
_PREFIX = (
    r"^[\s\W]*(?:(?:please|pls|kindly|hey|hi|ok|okay|now|then|also|and|just|go\s+ahead\s+and|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|you\s+can|help\s+me|try\s+to|let'?s|"
    r"i\s+(?:want|need|would\s+like|'d\s+like|wanna)(?:\s+you)?(?:\s+to)?)[\s,]+)*"
)

# Words that mean the user is producing or editing TEXT, not a picture.
TEXT_CUES = re.compile(
    r"\b(?:summary|summari[sz]e|list|table|caption|description|text|essay|article|email|post|"
    r"poem|story|title|slogan|tagline|paragraph|sentence|bullet|translate|translation|grammar|"
    r"spelling|wording|heading|headline|tone|formal|casual|concise|shorter|longer|outline|"
    r"script|prompt|markdown|html|csv|json|code|template|document|checklist|ideas?|names?)\b",
    re.I,
)

# Nouns that, directly after an image noun, show it is a modifier ("graphics discussion").
_TEXTY_FOLLOW = (
    r"(?:discussion|section|list|paragraph|text|description|caption|file|folder|library|gallery|"
    r"names?|ideas?|concepts?|trends?|guidelines?|requirements?|brief|copy|content|article|post|page)"
)

# ---------------------------------------------------------------------------
# Compiled rules
# ---------------------------------------------------------------------------

NEGATIVE_PATTERNS = [
    re.compile(r"\b(alt[- ]?text|caption)\b", re.I),
    re.compile(r"\b(extract\s+text|read\s+text|transcribe|ocr|scan\s+text)\b", re.I),
    re.compile(r"\b(summari[sz]e\s+(the\s+|this\s+)?(image|picture|photo|chart|infographic|diagram))", re.I),
    re.compile(r"\b(describe|analy[sz]e|explain|identify|interpret)\s+(what\s+is\s+(in|happening\s+in)\s+)?"
               r"(this|the|that)\s+(image|picture|photo|chart|diagram|scan|screenshot)", re.I),
    re.compile(r"\bwhat('s|\s+is|\s+does|\s+are)\s+(in\s+)?(this|the)\s+(image|picture|photo)", re.I),
    re.compile(r"\b(describe|analy[sz]e|explain)\s+(this|the)\s+\w+\s+(scan|diagram|chart)", re.I),
]

COMPOSITE_PATTERNS = [
    re.compile(
        r"\b(article|blog\s+post|story|essay|newsletter|report|email)\b.*?"
        r"\b(with|including|having|featuring)\b.*?"
        r"\b(\d+|two|three|four|multiple|an?)\s+" + IMAGE_NOUNS,
        re.I,
    ),
    re.compile(
        r"\b(write|draft|author)\b.*?\b(article|blog|story|post|essay|newsletter)\b.*?"
        r"\b(generate|create|add|and)\b.*?" + IMAGE_NOUNS,
        re.I,
    ),
]

REDO_PATTERNS = [
    re.compile(_PREFIX + r"(?:regenerate|redo|try\s+again|once\s+more|again|another\s+version|"
                         r"give\s+me\s+another(?:\s+one)?|generate\s+another|same\s+again)\s*[.!]?\s*$", re.I),
]

_VISUAL = (
    r"(?:darker|brighter|lighter|night(?:time)?|day(?:time)?|sunset|sunrise|dusk|dawn|golden\s+hour|"
    r"rain(?:y|ier)|snow(?:y|ier)|fog(?:gy)?|misty|sunny|cloudy|stormy|colou?rful|vibrant|muted|"
    r"(?:de)?saturated|warmer|cooler|bigger|smaller|sharper|blurr(?:y|ier)|realistic|photorealistic|"
    r"cartoon(?:ish)?|anime|watercolou?r|oil\s+painting|sketch|pixel(?:ated)?|minimal(?:ist)?|vintage|"
    r"retro|futuristic|cyberpunk|moody|dramatic|cinematic|black\s+and\s+white|monochrome|contrast|"
    r"lighting|background|foreground|sky|colou?rs?|palette|textures?|shadows?|highlights?|autumn|"
    r"winter|summer|spring|16:9|9:16|banner|wide|portrait|landscape|square|zoomed|closer|farther)"
)

DELTA_PATTERNS = [
    re.compile(_PREFIX + r"(?:make|turn|render|paint|draw)\s+(?:it|this|them|that)\b.*?\b" + _VISUAL + r"\b", re.I),
    re.compile(_PREFIX + r"(?:the\s+)?same\s+(?:image|one|thing|picture|scene)?\s*but\b", re.I),
    re.compile(_PREFIX + r"(?:change|swap|switch|replace|adjust|alter)\s+the\s+"
                         r"(?:colou?rs?|background|style|lighting|mood|vibe|sky|weather|season|palette|"
                         r"time\s+of\s+day|aspect\s+ratio|angle|pose)\b", re.I),
    re.compile(_PREFIX + r"(?:more|less|add\s+more|use\s+more|with\s+more)\s+(?:\w+\s+)?" + _VISUAL + r"\b", re.I),
    re.compile(_PREFIX + r"(?:give\s+it|add)\s+(?:an?\s+|some\s+)?(?:\w+\s+){0,2}"
                         r"(?:glow|filter|vibe|look|feel|effect|texture|border|frame|shadow|tint)\b", re.I),
]

# "refine the image ...", "enhance this photo ...": edit verb, determiner, image noun.
STRICT_EDIT = re.compile(
    _PREFIX + EDIT_VERBS + r"\s+(?:[\w'-]+\s+){0,2}?(?:this|the|that|my|these|our|your)\s+"
    r"(?:[\w-]+\s+){0,2}?" + IMAGE_NOUNS + r"\b(?!\s+" + _TEXTY_FOLLOW + r"\b)",
    re.I,
)

CONVERT_EDIT = re.compile(
    _PREFIX + r"(?:turn|convert|transform|change)\s+(?:this|the|that|my)\s+(?:[\w-]+\s+){0,2}?"
    r"" + IMAGE_NOUNS + r"\s+(?:into|to|in)\b",
    re.I,
)

BACKGROUND_EDIT = re.compile(
    _PREFIX + r"(?:remove|cut\s*out|delete|erase)\s+(?:the\s+|its\s+)?background\b", re.I
)

STYLE_EDIT = re.compile(
    _PREFIX + r"(?:add|give|apply)\s+(?:(?:a|an|some|the)\s+)?[\w\s'-]{0,40}?"
    r"\b(?:vibe|style|look|feel|effect|filter|aesthetic)\b",
    re.I,
)

REF_EDIT_START = re.compile(_PREFIX + REF_EDIT_VERBS + r"\b", re.I)
QUESTION_START = re.compile(
    r"^\s*(?:what|who|where|why|how|when|which|is\s+this|are\s+these|describe|tell\s+me|identify)\b", re.I
)

# "create a poster design brief" asks for a document about a poster, not the poster.
_DOC_FOLLOW = (
    r"(?:brief|copy|guidelines?|concepts?|ideas?|descriptions?|prompts?|proposals?|strategy|"
    r"requirements?|specs?|template|checklist)"
)

VERB_NEAR_NOUN = re.compile(
    _PREFIX + CREATION_VERBS + r"\b(?:\s+[\w'-]+){0,5}?\s+" + IMAGE_NOUNS + r"\b(?!\s+(?:\w+\s+)?" + _DOC_FOLLOW + r"\b)",
    re.I,
)
NOUN_OF_START = re.compile(
    _PREFIX + r"(?:(?:an?|the)\s+)?(?:[\w-]+\s+){0,2}?(?:image|picture|photo|illustration|drawing|painting)\s+of\b",
    re.I,
)
DRAW_START = re.compile(
    _PREFIX + r"(?:draw|sketch|paint|illustrate|render)\s+(?:me\s+|us\s+)?(?:an?|the|some|my|our)\b", re.I
)

# Words between the verb and noun that show the output is text ("make a list of icons").
_TEXT_ARTIFACT = re.compile(
    r"\b(?:list|summary|outline|plan|script|description|caption|article|post|email|ideas?|"
    r"names?|titles?|checklist|bullet|paragraph|essay|prompts?|markdown|html|csv|json|code|"
    r"template|page|document|spreadsheet)\b"
    # "table" is text only as a document table ("a table of results", "create a table"), not "a wooden table"
    r"|\btable\s+(?:of|with|showing|comparing|listing|format)\b"
    r"|\b(?:create|make|build|generate|add|insert)\s+(?:an?\s+|the\s+)?(?:\w+\s+){0,2}table\b",
    re.I,
)


def _result(kind: str, rule: str, query: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "kind": kind,
        "rule": rule,
        "is_followup": False,
        "followup_mode": None,
        "uses_previous_image": False,
        "requires_context": False,
        "raw_query": query,
    }
    out.update(extra)
    return out


def _edit_result(
    rule: str, q: str, has_reference: bool, has_previous_image: bool, log_prefix: str
) -> Dict[str, Any]:
    """Edits need a picture to edit: an attached one, or the previously generated one."""
    if has_reference:
        logger.info(f"{log_prefix}Intent rule matched: {rule} -> edit (attached reference)")
        return _result("edit", rule, q)
    if has_previous_image:
        logger.info(f"{log_prefix}Intent rule matched: {rule} -> edit (previous image)")
        return _result("edit", rule, q, uses_previous_image=True)
    logger.info(f"{log_prefix}Intent rule matched: {rule} but no image to edit; deferring")
    return _result("none", "edit_without_reference", q, requires_context=True)


def detect_image_intent(
    query: str,
    has_reference: bool = False,
    has_previous_image: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a free-form chat query.

    Returns a dict with:
        kind                 "generate" | "edit" | "none"
        rule                 identifier of the rule that decided
        is_followup          True for redo / delta phrases tied to a previous image
        followup_mode        "redo" | "delta" | None
        uses_previous_image  True when the edit should use the last generated image
        requires_context     True when the query only makes sense with a previous
                             image and none was available; the caller may look one
                             up and call again with has_previous_image=True
        raw_query            the stripped query
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    q = (query or "").strip()
    if not q:
        return _result("none", "empty_query", q)

    # 1. Composite requests stay on the agent path
    for p in COMPOSITE_PATTERNS:
        if p.search(q):
            logger.info(f"{log_prefix}Intent rule matched: composite_article_with_image (agent path)")
            return _result("none", "composite_article_with_image", q)

    # 2. Captions, OCR, summaries, vision Q&A
    for p in NEGATIVE_PATTERNS:
        if p.search(q):
            logger.info(f"{log_prefix}Intent rule matched: negative_rule")
            return _result("none", f"negative_pattern:{p.pattern[:30]}", q)

    if has_reference and QUESTION_START.search(q):
        logger.info(f"{log_prefix}Intent rule matched: vision_qa_with_reference")
        return _result("none", "vision_qa_with_reference", q)

    text_cue = bool(TEXT_CUES.search(q))

    # 3. Follow-ups: only meaningful when there is a previous image and the user
    #    did not attach a new one.
    if not has_reference and not text_cue:
        for p in REDO_PATTERNS:
            if p.search(q):
                if has_previous_image:
                    logger.info(f"{log_prefix}Intent rule matched: followup_pure_redo")
                    return _result("generate", "followup_pure_redo", q, is_followup=True,
                                   followup_mode="redo", uses_previous_image=False)
                return _result("none", "followup_without_context", q, requires_context=True)

        for p in DELTA_PATTERNS:
            if p.search(q):
                if has_previous_image:
                    logger.info(f"{log_prefix}Intent rule matched: followup_delta_edit")
                    return _result("edit", "followup_delta_edit", q, is_followup=True,
                                   followup_mode="delta", uses_previous_image=True)
                return _result("none", "followup_without_context", q, requires_context=True)

    # 4. Edits that name the image explicitly
    for rule, pattern in (
        ("strict_edit_image_object", STRICT_EDIT),
        ("convert_image_object", CONVERT_EDIT),
        ("remove_background", BACKGROUND_EDIT),
    ):
        # "convert this list of icons to a table" names an image noun but edits text.
        if text_cue and rule != "remove_background":
            continue
        if pattern.search(q):
            return _edit_result(rule, q, has_reference, has_previous_image, log_prefix)

    if (has_reference or has_previous_image) and not text_cue and STYLE_EDIT.search(q):
        return _edit_result("style_edit", q, has_reference, has_previous_image, log_prefix)

    # 5. Reference attached + an edit verb leading the request
    if has_reference and not text_cue and REF_EDIT_START.search(q):
        logger.info(f"{log_prefix}Intent rule matched: reference_attached_with_edit_verb -> edit")
        return _result("edit", "reference_attached_with_edit_verb", q)

    # 6. Generation, anchored to the start of the request
    m = VERB_NEAR_NOUN.search(q)
    if m and not _TEXT_ARTIFACT.search(m.group(0)):
        logger.info(f"{log_prefix}Intent rule matched: verb_near_image_noun -> generate")
        return _result("generate", "verb_near_image_noun", q)

    if NOUN_OF_START.search(q) and not text_cue:
        logger.info(f"{log_prefix}Intent rule matched: image_of_subject -> generate")
        return _result("generate", "image_of_subject", q)

    m = DRAW_START.search(q)
    if m and not _TEXT_ARTIFACT.search(q):
        logger.info(f"{log_prefix}Intent rule matched: standalone_draw_command -> generate")
        return _result("generate", "standalone_draw_command", q)

    return _result("none", "no_image_intent_matched", q)
