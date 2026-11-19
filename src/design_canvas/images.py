"""
Photo fill for the design canvas (docs/design_image_fill_plan.md).

With a 1.1 design selected, "add real photos" sends the design's ImageNode placeholders through the image
generation flow the rest of the agent uses (text_processing_agent._build_image_tool): credits, provider and model
choice, CDN upload and the image_ready stream events all come from there. Nothing here calls a model.
"""

from __future__ import annotations

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
    The ImageNodes to generate, in reading order, and a summary. Empty slots, or every slot when the user asks for
    new/replacement photos. When the request names a card by one of its own short texts ("only the KFC one"), only
    that card's photos. At most ``cap``.
    """
    replace = bool(_REPLACE_RE.search(query))
    asked = " " + " ".join(re.findall(r"[\w'&!]+", query.casefold())) + " "
    slots = [(node, ancestors) for node, ancestors in walk(ast["root"]) if node["node_type"] == "image"]
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
    """Whether a design sent by the editor already has ImageNodes (no validation: used to price the request)."""
    import json
    try:
        ast = json.loads(existing_design_ast) if isinstance(existing_design_ast, str) else existing_design_ast
        return bool(isinstance(ast, dict) and isinstance(ast.get("root"), dict) and image_nodes(ast))
    except (ValueError, TypeError, KeyError, AttributeError):
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
