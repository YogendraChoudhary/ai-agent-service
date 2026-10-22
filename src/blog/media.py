"""
Images for a post. The model never writes image markup (D11): it describes a picture in the outline, and from
there code owns everything. As soon as the outline validates, each picture is started in the background through the
app's existing image tool (same provider routing, credit reservation, upload and `image_ready` flow as a chat
request), and the drafted text gets `![alt](placeholder::<uuid>)` at the position code chose.

Image backends sit behind one small interface so stock photos or generated diagrams can be added later (open items
O4, O5 of the plan) without touching the pipeline: `fulfil(directive) -> Placed`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from .schema import ImageDirective, Outline

logger = logging.getLogger(__name__)

# Appended to every picture prompt by code, so the rule holds whatever the model wrote.
IMAGE_STYLE_SUFFIX = "No text, letters, numbers or logos in the picture. Main subject centred with space around it."
_PLACEHOLDER_RE = re.compile(r"placeholder::([0-9A-Za-z-]+)")
_REASON_PREFIX_RE = re.compile(r"^(?:ERROR|Image generation skipped)\s*:?\s*", re.I)


@dataclass(frozen=True)
class Placed:
    placeholder_id: Optional[str]
    reason: Optional[str] = None          # why there is no picture, when placeholder_id is None


class ImageSource(Protocol):
    def fulfil(self, directive: ImageDirective) -> Placed: ...


class GeneratedImages:
    """Draws each picture with the app's image tool: `tool(prompt)` returns 'Image URL: placeholder::<uuid>' or an error."""

    def __init__(self, tool: Callable[[str], str]):
        self.tool = tool

    def fulfil(self, directive: ImageDirective) -> Placed:
        prompt = f"{directive.prompt.rstrip('. ')}. {IMAGE_STYLE_SUFFIX}"
        try:
            result = self.tool(prompt)
        except Exception as e:                              # a picture must never take the post down
            logger.warning(f"image tool failed: {e}")
            return Placed(None, str(e)[:120])
        match = _PLACEHOLDER_RE.search(result or "")
        if match:
            return Placed(match.group(1))
        return Placed(None, _REASON_PREFIX_RE.sub("", (result or "no image was started").strip())[:120])


def get_source(name: str, tool: Optional[Callable[[str], str]]) -> Optional[ImageSource]:
    """The backend for BLOG_IMAGE_SOURCE. Only 'generate' exists today; anything else means no images."""
    name = (name or "").strip().lower()
    if name == "generate" and tool is not None:
        return GeneratedImages(tool)
    if name not in ("generate", "none", ""):
        logger.warning(f"BLOG_IMAGE_SOURCE={name!r} has no backend yet; the post will have no images")
    return None


def start_images(source: Optional[ImageSource], outline: Outline) -> Dict[str, Placed]:
    """Start every picture the outline asks for, in order. Keyed by section id (one picture per section at most)."""
    placed: Dict[str, Placed] = {}
    for section in outline.sections:
        for directive in section.of_kind("image"):
            placed[section.id] = source.fulfil(directive) if source is not None else Placed(None, "no image backend")
    return placed
