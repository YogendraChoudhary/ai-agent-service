"""
Photo fill for the design canvas (docs/design_image_fill_plan.md).

With a 1.1 design selected, "add real photos" sends the design's ImageNode placeholders through the image
generation flow the rest of the agent uses (text_processing_agent._build_image_tool): credits, provider and model
choice, CDN upload and the image_ready stream events all come from there. Nothing here calls a model.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

# The words that ask for every photo to be made again, not only the empty slots.
_REPLACE_RE = re.compile(r"\b(?:replace|swap|regenerate|redo|re-?generate|new|different|change|other)\b", re.I)
PHOTO_STYLE = "Photograph. No text, no lettering, no logos, no watermark."
_PLACEHOLDER_RE = re.compile(r"placeholder::([0-9a-f-]{36})")

# One layout edit, run only when the selected design has no ImageNode yet (made before photo slots existed).
CONVERT_INSTRUCTION = (
    "Turn every area that stands for a photograph or picture (a vector_illustration, or a coloured block drawn in "
    "place of a photo) into an ImageNode: keep its id and position, give it a short alt, a prompt describing the "
    "photo it should show (subject, setting, colours, lighting, framing; no brand names, logos or text), the "
    "aspect_ratio closest to the area's shape, and no src. Change nothing else."
)


def walk(node: dict) -> Iterator[tuple[dict, list[dict]]]:
    """(node, ancestors) in reading order."""
    stack = [(node, [])]
    while stack:
        current, ancestors = stack.pop()
        yield current, ancestors
        for child in reversed(current.get("children", ())):
            stack.append((child, ancestors + [current]))


def image_nodes(ast: dict) -> list[dict]:
    return [node for node, _ in walk(ast["root"]) if node["node_type"] == "image"]


def _node_text(node: dict) -> str:
    if node["node_type"] == "text":
        return "".join(span["text"] for span in node.get("spans", ()))
    return node.get("label", "") if node["node_type"] == "button" else ""


def _card_texts(ancestors: list[dict]) -> list[str]:
    """Texts of the nearest ancestor that has any: the card or panel the photo belongs to."""
    for ancestor in reversed(ancestors):
        texts = [t for n, _ in walk(ancestor) if (t := " ".join(_node_text(n).split()))]
        if texts:
            return texts
    return []


def fill_targets(ast: dict, query: str, cap: int) -> tuple[list[dict], dict]:
    """
    The ImageNodes to generate, in reading order, and a summary. With a section or photo selected in the editor, the
    photos in it. Otherwise empty slots, or every slot when the user asks for new/replacement photos. When the request names a card by one of its own short texts ("only the KFC one"), only
    that card's photos. At most ``cap``.
    """
    from .text_edit import split_selection

    query, selected_id = split_selection(query)
    slots = [(node, ancestors) for node, ancestors in walk(ast["root"]) if node["node_type"] == "image"]
    if selected_id:
        # The editor's selection: only the photos in it, redrawn whether or not they already have one.
        chosen = [n for n, a in slots if n["id"] == selected_id or any(x.get("id") == selected_id for x in a)]
        if chosen:
            summary = {"slots": len(slots), "replace": True, "named": False, "selected": selected_id,
                       "over_cap": max(0, len(chosen) - cap)}
            return chosen[:cap], summary
    replace = bool(_REPLACE_RE.search(query))
    asked = " " + " ".join(re.findall(r"[\w'&!]+", query.casefold())) + " "
    candidates = [(n, a) for n, a in slots if replace or not n.get("src")]
    named = []
    for node, ancestors in candidates:
        for text in _card_texts(ancestors):
            words = " ".join(re.findall(r"[\w'&!]+", text.casefold()))
            if 3 <= len(words) <= 40 and f" {words} " in asked:
                named.append(node)
                break
    targets = named or [n for n, _ in candidates]
    summary = {"slots": len(slots), "replace": replace, "named": bool(named),
               "over_cap": max(0, len(targets) - cap)}
    return targets[:cap], summary


def photo_prompt(node: dict) -> str:
    return f"{node['prompt'].rstrip(' .')}. {PHOTO_STYLE}"


def placeholder_id(tool_result: str) -> str | None:
    """The uuid in the image tool's "Image URL: placeholder::<uuid>" answer, or None when it refused."""
    match = _PLACEHOLDER_RE.search(tool_result or "")
    return match.group(1) if match else None


def has_image_slots(existing_design_ast) -> bool:
    """
    Whether a design sent by the editor already has ImageNodes; prices the request. Validated with the same schema
    the fill path uses: an image node the schema drops (e.g. under a text node) must not price a conversion edit at 0.
    """
    from pydantic import ValidationError
    from .schema_v11 import DesignDocumentASTV11
    try:
        if isinstance(existing_design_ast, str):
            ast = DesignDocumentASTV11.model_validate_json(existing_design_ast)
        else:
            ast = DesignDocumentASTV11.model_validate(existing_design_ast)
        return bool(image_nodes(ast.model_dump()))
    except (ValidationError, ValueError, TypeError):
        return False


def apply_finished_images(ast: dict, pending_images: dict) -> dict:
    """
    The design with each generated photo in place: src = its CDN URL, placeholder_id cleared. A photo that failed
    or is still generating keeps its previous src (none for an empty slot). For callers that wait for the images
    themselves (/agent/process); the streaming editor applies image_ready events instead.
    """
    for node in image_nodes(ast):
        entry = (pending_images or {}).get(node.get("placeholder_id") or "")
        if not entry:
            continue
        if entry.get("status") == "done" and entry.get("cdn_url"):
            node["src"] = entry["cdn_url"]
        node["placeholder_id"] = None
    return ast


# Wording that changes how a photo sits in the layout (size, crop, corners, position): a layout edit, not a new photo.
_PHOTO_LAYOUT_RE = re.compile(
    r"\b(?:bigger|smaller|larger|wider|narrower|taller|shorter|size|resize|width|height|crop|cropped|aspect|ratio"
    r"|round(?:ed)?|circle|circular|corners?|radius|border|frame|shadow|align(?:ed)?|cent(?:er|re)(?:ed)?|left|right"
    r"|move|position|padding|margins?|spacing|gap|full[- ]width|stretch|fit|contain|cover|remove|delete|hide)\b",
    re.IGNORECASE,
)


def is_photo_layout_wording(text: str) -> bool:
    return bool(_PHOTO_LAYOUT_RE.search(text or ""))


def _parse_design(existing_design_ast) -> dict | None:
    if isinstance(existing_design_ast, dict):
        return existing_design_ast
    if isinstance(existing_design_ast, str):
        try:
            value = json.loads(existing_design_ast)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None


def selected_photo_scope(existing_design_ast, query: str) -> tuple[str | None, str | None]:
    """
    (selected node id, what it holds) for the editor's selection sentence in ``query``: "image" when the selected node
    is a photo, "contains_images" when photos sit inside it, "no_images" otherwise, None when nothing usable is selected.
    Reads the raw design without validating it: called on every request to price and route it.
    """
    from .text_edit import split_selection

    _, selected_id = split_selection(query)
    design = _parse_design(existing_design_ast)
    if not selected_id or design is None or not isinstance(design.get("root"), dict):
        return selected_id, None
    for node, _ in walk(design["root"]):
        if isinstance(node, dict) and node.get("id") == selected_id:
            if node.get("node_type") == "image":
                return selected_id, "image"
            inside = any(isinstance(n, dict) and n.get("node_type") == "image" for n, _ in walk(node))
            return selected_id, "contains_images" if inside else "no_images"
    return selected_id, None


# A plain fill ("add real photos", "fill the photo slots") draws each slot from the prompt the design already has.
# Anything else ("replace this image with a pizza", "make the photos brighter", "use a different picture") says what
# the new photo should show: the slots' prompts are rewritten from the request first (one small model call).
_PLAIN_FILL_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:add|use|put|insert|fill(?:\s+in)?|generate|create|make|show"
    r"|give|get|place|load)\b(?:\s+(?:in|me|it|them|the|all|some|real|actual|proper|stock|realistic|photo|image|picture"
    r"|slots?|placeholders?|for|with|to|this|these|design|page|here|of|every|each|empty))*\s+"
    r"(?:photos?|images?|pictures?|pics|photographs?|imagery)\b(?:\s+(?:slots?|placeholders?))?(?:\s+(?:in|to|for|on|into)\s+(?:the\s+|this\s+|my\s+)?"
    r"(?:design|page|slots?|placeholders?|cards?|it|them))?\s*(?:please)?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def needs_prompt_rewrite(query: str) -> bool:
    """Whether the request says something about what the photos should show (anything but a plain fill)."""
    from .text_edit import split_selection

    instruction, _ = split_selection(query)
    return not _PLAIN_FILL_RE.match(instruction)


REWRITE_SYSTEM_PROMPT = """You write the descriptions that photos in a UI design are generated from.
You get the user's request and the photo slots it applies to. Each slot has an `id`, its `alt` text, its current
`prompt` (the description its photo was or will be generated from) and `context`: the texts of the card or section the
photo sits in.

Answer {"photos": {"<id>": {"prompt": "...", "alt": "..."}}} with one entry for EVERY slot given:
- `prompt`: a concrete photo description (subject, setting, colours, lighting, framing) for the photo the user wants.
  Apply the request: a new subject ("a pizza"), a look ("brighter", "top-down", "on a dark table"), or, for "a
  different photo", the same subject shown differently. Keep everything the request does not change from the current
  prompt. No brand names, logos, text or lettering. At most 500 characters.
- `alt`: a short description of the new photo, at most 120 characters.

If the request cannot be done by generating new photos for these slots, answer {"not_photo_edit": true}.
Answer with the JSON object only."""


def rewrite_photo_prompts(ast: dict, targets: list[dict], query: str, llm) -> dict[str, dict] | None:
    """
    {id: {"prompt", "alt"}} for the targets, from the request; None when the model declines or the answer does not hold.
    Provider errors propagate.
    """
    from src.json_reply import strip_json_fence
    from src.longform.llm import CallSpec
    from .text_edit import split_selection

    instruction, _ = split_selection(query)
    ids = {node["id"] for node in targets}
    context = {node["id"]: _card_texts(ancestors)[:8] for node, ancestors in walk(ast["root"]) if node["id"] in ids}
    slots = [{"id": node["id"], "alt": node.get("alt", ""), "prompt": node.get("prompt", ""),
              "context": context.get(node["id"], [])} for node in targets]
    user = (f"Design: {json.dumps(ast.get('title') or '', ensure_ascii=False)}\n"
            f"Request: {json.dumps(instruction, ensure_ascii=False)}\n"
            f"Slots: {json.dumps(slots, ensure_ascii=False)}")
    reply = llm(REWRITE_SYSTEM_PROMPT, [{"role": "user", "content": user}],
                CallSpec(stage="design_photo_prompt", temperature=0.3, max_tokens=min(4000, 300 + 220 * len(targets))))
    try:
        answer = json.loads(strip_json_fence(reply or "").strip())
    except ValueError:
        return None
    if not isinstance(answer, dict) or answer.get("not_photo_edit") or not isinstance(answer.get("photos"), dict):
        return None
    out = {}
    for node_id, value in answer["photos"].items():
        if node_id not in ids or not isinstance(value, dict):
            continue
        prompt, alt = value.get("prompt"), value.get("alt")
        if not isinstance(prompt, str) or not 3 <= len(prompt.strip()) <= 600:
            continue
        out[node_id] = {"prompt": prompt.strip(),
                        "alt": alt.strip()[:300] if isinstance(alt, str) and alt.strip() else None}
    return out if set(out) == ids else None
