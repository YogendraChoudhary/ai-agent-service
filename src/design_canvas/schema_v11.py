"""Additive Design Canvas 1.1 models. The 1.0 models also define the legacy wire format."""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from . import schema as v10

HexColor = Annotated[str, Field(pattern=v10.HEX_COLOR_PATTERN)]
FontRole = Literal["display", "heading", "body", "mono"]
FontFamily = Literal[
    "Inter", "Inter Tight", "DM Sans", "Manrope", "Plus Jakarta Sans", "Outfit", "Sora", "Space Grotesk",
    "Instrument Sans", "Geist", "Urbanist", "Poppins", "Montserrat", "Playfair Display", "Fraunces",
    "DM Serif Display", "Instrument Serif", "JetBrains Mono", "IBM Plex Mono", "Space Mono", "Geist Mono",
]
ALLOWED_FONT_FAMILIES = get_args(FontFamily)
FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "extrabold"]
Version = Literal["1.0", "1.1"]
PanelWeight = Annotated[int, Field(ge=1, le=100)]


class FontToken(BaseModel):
    family: FontFamily
    fallback: Literal["sans-serif", "serif", "monospace"] = "sans-serif"


class DesignFonts(BaseModel):
    display: FontToken | None = None
    heading: FontToken | None = None
    body: FontToken | None = None
    mono: FontToken | None = None


class LinearGradient(BaseModel):
    angle_deg: int = Field(90, ge=0, le=360)
    stops: list[HexColor] = Field(min_length=2, max_length=5)


WeightValue = Annotated[int, Field(ge=100, le=1000)]


class TextSpan(v10.TextSpan):
    font_weight: FontWeight = "normal"
    # Exact variable-font weight (e.g. 550) measured from a reference; font_weight stays the nearest named weight.
    font_weight_value: WeightValue | None = None
    gradient: LinearGradient | None = None
    baseline: Literal["normal", "super", "sub"] = "normal"
    size_scale: float = Field(1.0, ge=0.3, le=1.0)


class NodeStyle(v10.NodeStyle):
    background_gradient: LinearGradient | None = None
    shadow: Literal["none", "soft", "glow"] = "none"
    shadow_color: HexColor | None = None
    flex_grow: PanelWeight | None = None


def _clamped_size(low: int, high: int):
    """A size just outside the range (a 9 px "NEW" badge) is clamped, not sent back to the model: live 2026-11-19
    one such size cost a 44 s repair call."""
    def clamp(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(low, min(high, round(value)))
        return value
    return clamp


class TextNode(v10.TextNode):
    spans: list[TextSpan] = Field(min_length=1)
    style: NodeStyle = Field(default_factory=NodeStyle)
    font_size_px: int = Field(16, ge=10, le=200)
    _clamp_size = field_validator("font_size_px", mode="before")(_clamped_size(10, 200))
    font_role: FontRole = "body"
    letter_spacing_em: float = Field(0.0, ge=-0.1, le=0.5)
    text_transform: Literal["none", "uppercase"] = "none"


class DividerNode(v10.DividerNode):
    orientation: Literal["horizontal", "vertical"] = "horizontal"


class IconButtonNode(v10.IconButtonNode):
    icon_name: Literal[
        "arrow_left", "arrow_right", "chevron_left", "chevron_right",
        "plus", "close", "check", "sparkles", "search", "send",
        "copy", "arrow_up_right", "chevron_down",
    ]
    style: NodeStyle = Field(default_factory=NodeStyle)


class ButtonNode(v10.ButtonNode):
    _clamp_size = field_validator("font_size_px", mode="before")(_clamped_size(12, 32))
    font_role: FontRole = "body"
    font_weight: FontWeight = "semibold"
    font_weight_value: WeightValue | None = None
    letter_spacing_em: float = Field(0.0, ge=-0.1, le=0.5)
    text_transform: Literal["none", "uppercase"] = "none"
    trailing_icon: Literal["arrow_up_right", "arrow_right", "copy", "chevron_down"] | None = None
    style: NodeStyle = Field(default_factory=NodeStyle)


IMAGE_ASPECTS = ("1:1", "4:3", "3:4", "16:9", "9:16")     # the image generation flow's own aspect ratios
PLACEHOLDER_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def cdn_image_url(value):
    """An image src the editor may load: https on the configured CDN host (where generated images are uploaded).
    The model never supplies an arbitrary URL; an empty value means "not generated yet"."""
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 2048 or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("src must be a URL without whitespace, at most 2048 characters")
    from config import settings
    cdn = urlsplit(settings.cdn_url or "")
    parts = urlsplit(value)
    # Exactly the CDN's host[:port]: https://attacker.example\@<cdn>/a.png parses here as the CDN, but browsers turn
    # the backslash into a slash and load attacker.example (audit 2026-11-20). No userinfo, no other port.
    if ("\\" in value or parts.scheme != "https" or not cdn.hostname or parts.hostname != cdn.hostname
            or parts.netloc.lower() != cdn.netloc.lower()):
        raise ValueError("src must be an https URL on the image CDN; leave it empty for a placeholder")
    return value


class ImageNode(BaseModel):
    """A photo slot. ``prompt`` says what the photo shows; ``src`` is set once the image flow has generated it."""
    node_type: Literal["image"] = "image"
    id: str = Field(..., pattern=v10.NODE_ID_PATTERN)
    alt: str = Field(..., min_length=1, max_length=300)
    prompt: str = Field(..., min_length=3, max_length=600)
    src: str | None = None
    placeholder_id: str | None = Field(None, pattern=PLACEHOLDER_ID_PATTERN)   # set while the photo is generating
    # The slot's own shape, e.g. "140:93" for a 280x186 photo. Generation uses the nearest of IMAGE_ASPECTS and the
    # photo fills the slot with object-fit (a slight crop), so the layout keeps the reference's exact proportions.
    aspect_ratio: str = Field("4:3", pattern=r"^[1-9][0-9]{0,4}:[1-9][0-9]{0,4}$")
    width_percent: int = Field(100, ge=10, le=100)
    object_fit: Literal["cover", "contain"] = "cover"
    source_box: list[Annotated[int, Field(ge=0, le=20000)]] | None = Field(None, min_length=4, max_length=4)
    style: NodeStyle = Field(default_factory=NodeStyle)
    editable: bool = True

    @field_validator("src", mode="before")
    @classmethod
    def _cdn_only(cls, value):
        return cdn_image_url(value)

    @field_validator("aspect_ratio")
    @classmethod
    def _sane_shape(cls, value):
        w, h = (int(v) for v in value.split(":"))
        if not 0.2 <= w / h <= 5:
            raise ValueError("aspect_ratio must be between 1:5 and 5:1")
        return value


def generation_aspect(aspect_ratio: str) -> str:
    """The image flow's supported aspect closest to a slot's shape."""
    w, h = (int(v) for v in aspect_ratio.split(":"))
    return min(IMAGE_ASPECTS, key=lambda a: abs(w / h - int(a.split(":")[0]) / int(a.split(":")[1])))


class ContainerNode(v10.ContainerNode):
    style: NodeStyle = Field(default_factory=NodeStyle)
    children: list[DesignNode] = Field(min_length=1, max_length=25)
    column_ratios: list[PanelWeight] | None = Field(None, min_length=2, max_length=6)
    wrap: bool = False

    @model_validator(mode="after")
    def _layout_fields(self):
        if self.column_ratios and (self.layout != "grid" or len(self.column_ratios) != self.columns):
            raise ValueError("column_ratios requires grid layout and one ratio for each column")
        for i, child in enumerate(self.children):
            if not isinstance(child, DividerNode) or child.orientation != "vertical":
                continue
            if (self.layout != "flex_row" or self.wrap or i in (0, len(self.children) - 1) or
                isinstance(self.children[i - 1], DividerNode) or isinstance(self.children[i + 1], DividerNode)):
                raise ValueError("A vertical divider must sit between two panels in a non-wrapping flex_row")
        return self


DesignNode = Annotated[
    ContainerNode | TextNode | DividerNode | v10.VectorIllustrationNode | IconButtonNode | ButtonNode | ImageNode,
    Field(discriminator="node_type"),
]
ContainerNode.model_rebuild()


class DesignDocumentASTV11(v10.DesignDocumentAST):
    version: Version = "1.1"
    fonts: DesignFonts | None = None
    root: ContainerNode


def downgrade_to_v10(ast: v10.DesignDocumentAST) -> v10.DesignDocumentAST:
    """Keep compatible rendering values and let the 1.0 model drop 1.1 fields."""
    if type(ast) is v10.DesignDocumentAST:
        return ast
    weights = {"light": "normal", "extrabold": "bold"}

    def downgrade_node(node: dict) -> None:
        style = node.get("style")
        if style and style.get("background_gradient") and not style["background_color"]:
            style["background_color"] = style["background_gradient"]["stops"][0]
        if node["node_type"] == "text":
            node["font_size_px"] = min(node["font_size_px"], 96)
            for span in node["spans"]:
                if span.get("gradient"):
                    span["color"] = span["gradient"]["stops"][0]
                if node.get("text_transform") == "uppercase":
                    span["text"] = span["text"].upper()
                span["font_weight"] = weights.get(span["font_weight"], span["font_weight"])
        if node["node_type"] == "button":
            node["font_weight"] = weights.get(node["font_weight"], node["font_weight"])
            if node.get("text_transform") == "uppercase":
                node["label"] = node["label"].upper()
        if node.get("icon_name") in ("copy", "arrow_up_right", "chevron_down"):
            glyph = {"copy": "⧉", "arrow_up_right": "↗", "chevron_down": "⌄"}[node["icon_name"]]
            node.update(node_type="text", tag="span", font_size_px=node["size_px"],
                        spans=[{"text": glyph, "color": node["color"]}])
        if node["node_type"] == "image":
            # 1.0 has no image node: a placeholder rectangle of the same shape, in the slot's colour.
            w, h = (int(v) for v in node["aspect_ratio"].split(":"))
            width = 320
            height = max(24, min(800, round(width * h / w)))
            fill = (node.get("style") or {}).get("background_color") or "#D9D9DE"
            node_id, alt = node["id"], node["alt"]
            node.clear()
            node.update(node_type="vector_illustration", id=node_id, view_box=f"0 0 {width} {height}",
                        width_px=width, height_px=height, editable=True, layers=[{
                            "layer_id": (node_id[:34] + "_photo"), "label": alt[:120], "svg_element": "rect",
                            "attributes": {"x": 0, "y": 0, "width": width, "height": height, "rx": 12},
                            "fill_color": fill}])
        if node.get("column_ratios"):
            node["columns"] = len(node["column_ratios"])
        if "children" in node:
            node["children"] = [child for child in node["children"] if child.get("orientation") != "vertical"]
            for child in node["children"]:
                downgrade_node(child)

    raw = ast.model_dump()
    raw["version"] = "1.0"
    downgrade_node(raw["root"])
    return v10.DesignDocumentAST.model_validate(raw)
