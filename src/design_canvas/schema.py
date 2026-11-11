"""
Discriminated Pydantic AST for Mode 8: Editable UI Design Engine.

Every visual and textual element in the returned design is decomposed into a
typed node with a unique ``id`` so the Eddyter Lexical ``HtmlCanvasNode``
can perform click-to-edit mutations without re-generating the entire graphic.

Node Types
----------
TextNode              – rich text block with multi-span (dual-tone) support.
DividerNode           – horizontal rule.
VectorIllustrationNode – multi-layer SVG illustration (each layer is selectable).
IconButtonNode        – navigation arrow / action icon.
ButtonNode            – labelled call-to-action button, with an optional safe ``href``.
ContainerNode         – recursive flex/grid layout container (``columns`` for grid).
DesignDocumentAST     – top-level document root.
"""

from __future__ import annotations

import re

from typing import Annotated, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

HEX_COLOR_PATTERN = r"^#(?:[0-9A-Fa-f]{3,4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$"
NODE_ID_PATTERN = r"^[a-z0-9_]{2,40}$"
LAYER_ID_PATTERN = r"^[a-z0-9_]{2,40}$"
MAX_HREF_LENGTH = 2048
SAFE_HREF_SCHEMES = ("http", "https", "mailto", "tel")


def safe_href(value: Optional[str]) -> Optional[str]:
    """
    A button link the editor can render as-is: an absolute http(s) URL with a host, a mailto:/tel: link, or an
    in-page ``#anchor``. Anything else (``javascript:``, ``data:``, ``vbscript:``, protocol-relative ``//host``,
    relative paths, backslashes, whitespace or control characters used to disguise a scheme) is rejected.
    An empty string means "no link".
    """
    if value is None or value == "":
        return None
    if len(value) > MAX_HREF_LENGTH:
        raise ValueError(f"href must be at most {MAX_HREF_LENGTH} characters")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 or ch == "\\" for ch in value):
        raise ValueError("href must not contain whitespace, control characters or backslashes")
    if value.startswith("#"):
        if not re.fullmatch(r"#[A-Za-z0-9_\-:.]{1,200}", value):
            raise ValueError("An in-page href must look like #section-id")
        return value
    scheme = value.split(":", 1)[0].lower() if ":" in value else ""
    if scheme not in SAFE_HREF_SCHEMES:
        raise ValueError("href must be an http, https, mailto or tel link, or an in-page #anchor")
    if scheme in ("http", "https"):
        parts = urlsplit(value)
        if not value.lower().startswith(scheme + "://") or not parts.hostname:
            raise ValueError("An http(s) href must be an absolute URL with a host")
    elif len(value) <= len(scheme) + 1:
        raise ValueError(f"A {scheme}: href needs an address after the colon")
    return value


class TextSpan(BaseModel):
    """A run of text with its own colour and weight — enables dual-tone headers."""

    text: str = Field(..., min_length=1)
    color: str = Field("#FFFFFF", pattern=HEX_COLOR_PATTERN)
    font_weight: Literal["normal", "medium", "semibold", "bold"] = "normal"
    italic: bool = False
    underline: bool = False


class SpacingBox(BaseModel):
    """CSS-style padding / margin in pixels."""

    top: int = Field(0, ge=0, le=256)
    right: int = Field(0, ge=0, le=256)
    bottom: int = Field(0, ge=0, le=256)
    left: int = Field(0, ge=0, le=256)


class NodeStyle(BaseModel):
    """Visual style properties shared across multiple node types."""

    background_color: Optional[str] = Field(None, pattern=HEX_COLOR_PATTERN)
    padding: SpacingBox = Field(default_factory=SpacingBox)
    border_radius: int = Field(0, ge=0, le=999)
    border_color: Optional[str] = Field(None, pattern=HEX_COLOR_PATTERN)
    border_width: int = Field(0, ge=0, le=16)
    max_width: Optional[int] = Field(None, ge=40, le=1440)
    width_percent: Optional[int] = Field(None, ge=10, le=100)


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class TextNode(BaseModel):
    node_type: Literal["text"] = "text"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    tag: Literal["h1", "h2", "h3", "p", "span", "caption"]
    spans: List[TextSpan] = Field(..., min_length=1)
    align: Literal["left", "center", "right"] = "center"
    font_size_px: int = Field(16, ge=10, le=96)
    line_height: float = Field(1.5, ge=1.0, le=2.5)
    style: NodeStyle = Field(default_factory=NodeStyle)
    editable: bool = True


class DividerNode(BaseModel):
    node_type: Literal["divider"] = "divider"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    color: str = Field("#2C2C2E", pattern=HEX_COLOR_PATTERN)
    thickness_px: int = Field(1, ge=1, le=8)
    width_percent: int = Field(75, ge=10, le=100)
    margin_y_px: int = Field(16, ge=0, le=128)
    editable: bool = True


class SVGLayer(BaseModel):
    """One selectable vector layer inside a VectorIllustrationNode."""

    layer_id: str = Field(..., pattern=LAYER_ID_PATTERN)
    label: str
    svg_element: Literal["path", "circle", "rect", "ellipse"]
    attributes: Dict[str, Union[str, int, float]]
    fill_color: str = Field(..., pattern=HEX_COLOR_PATTERN)
    stroke_color: Optional[str] = Field(None, pattern=HEX_COLOR_PATTERN)
    stroke_width: float = Field(0.0, ge=0.0, le=20.0)
    opacity: float = Field(1.0, ge=0.0, le=1.0)


class VectorIllustrationNode(BaseModel):
    node_type: Literal["vector_illustration"] = "vector_illustration"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    view_box: str = Field("0 0 200 180", pattern=r"^\d+\s+\d+\s+\d+\s+\d+$")
    width_px: int = Field(180, ge=24, le=800)
    height_px: int = Field(160, ge=24, le=800)
    layers: List[SVGLayer] = Field(..., min_length=1, max_length=24)
    editable: bool = True


class IconButtonNode(BaseModel):
    node_type: Literal["icon_button"] = "icon_button"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    icon_name: Literal[
        "arrow_left", "arrow_right", "chevron_left", "chevron_right",
        "plus", "close", "check", "sparkles", "search", "send",
    ]
    color: str = Field("#FFFFFF", pattern=HEX_COLOR_PATTERN)
    size_px: int = Field(24, ge=12, le=64)
    aria_label: str = "Action"
    style: NodeStyle = Field(default_factory=NodeStyle)
    editable: bool = True


class ButtonNode(BaseModel):
    node_type: Literal["button"] = "button"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    label: str = Field(..., min_length=1)
    text_color: str = Field("#000000", pattern=HEX_COLOR_PATTERN)
    font_size_px: int = Field(15, ge=12, le=32)
    font_weight: Literal["normal", "medium", "semibold", "bold"] = "semibold"
    href: Optional[str] = Field(
        None, max_length=MAX_HREF_LENGTH,
        description="Optional link the button opens: an absolute https:// or http:// URL, mailto:, tel:, or an "
                    "in-page #anchor. Omit it when the user gave no destination; never invent a URL.")
    style: NodeStyle = Field(default_factory=NodeStyle)
    editable: bool = True

    @field_validator("href", mode="before")
    @classmethod
    def _safe_href(cls, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("href must be a string")
        return safe_href(value)


# ---------------------------------------------------------------------------
# Recursive container (forward-declared for circular reference)
# ---------------------------------------------------------------------------

class ContainerNode(BaseModel):
    node_type: Literal["container"] = "container"
    id: str = Field(..., pattern=NODE_ID_PATTERN)
    layout: Literal["flex_col", "flex_row", "grid"] = "flex_col"
    align_items: Literal["start", "center", "end", "stretch"] = "center"
    justify_content: Literal[
        "start", "center", "end", "space_between", "space_around"
    ] = "center"
    gap_px: int = Field(16, ge=0, le=128)
    columns: Optional[int] = Field(
        None, ge=1, le=12,
        description="Number of equal grid columns (1–12). Only for layout \"grid\"; omit for flex layouts.")
    style: NodeStyle = Field(default_factory=NodeStyle)
    children: List["DesignNode"] = Field(..., min_length=1, max_length=25)

    @model_validator(mode="after")
    def _columns_need_grid(self):
        if self.columns is not None and self.layout != "grid":
            raise ValueError("columns is only valid when layout is \"grid\"")
        return self


# Discriminated union — the ``node_type`` literal discriminator lets Pydantic
# pick the correct class without ambiguity.
DesignNode = Annotated[
    Union[
        ContainerNode,
        TextNode,
        DividerNode,
        VectorIllustrationNode,
        IconButtonNode,
        ButtonNode,
    ],
    Field(discriminator="node_type"),
]

# Required: rebuild ContainerNode's forward reference after DesignNode is defined.
ContainerNode.model_rebuild()


# ---------------------------------------------------------------------------
# Document root
# ---------------------------------------------------------------------------

class DesignDocumentAST(BaseModel):
    """Top-level validated design document returned by execute_design_canvas_harness()."""

    block_type: Literal["eddyter_design_canvas"] = "eddyter_design_canvas"
    version: Literal["1.0"] = "1.0"
    design_id: str
    title: str
    theme_palette: List[str] = Field(..., min_length=2, max_length=10)
    root: ContainerNode
