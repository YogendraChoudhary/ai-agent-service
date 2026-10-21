"""
Email template format, schema v1.

The model writes a Template as JSON; code validates it here and the editor's renderer turns it into HTML.
Every model forbids unknown fields, so a field the model invents is an error rather than silently kept.
Limits live in this module only: the prompt spec (prompts.py) is generated from them.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MIN_BLOCKS, MAX_BLOCKS = 3, 20
SUBJECT_OPTIONS = 3
SUBJECT_HARD_MAX = 60          # schema rejects above this
SUBJECT_TARGET_MAX = 50        # lint warns above this
PREHEADER_HARD_MAX = 100
PREHEADER_TARGET = (40, 90)    # lint warns outside this
HEADING_MAX = 80
PARAGRAPH_MAX = 600
LIST_ITEMS = (2, 8)
LIST_ITEM_MAX = 200
KV_ROWS = (1, 10)
BUTTON_LABEL_MAX = 25
ALT_MAX = 150
IMAGE_PROMPT_MAX = 600
EYEBROW_MAX = 20
HERO_HEADLINE_MAX = 60
HERO_SUMMARY_MAX = 140
FEATURE_COLUMNS = (2, 3)
FEATURE_TITLE_MAX = 20
FEATURE_TEXT_MAX = 60
SOCIAL_MAX = 4
SOCIAL_PLATFORMS = ("linkedin", "x", "facebook", "instagram", "youtube", "email")

Tone = Literal["formal", "professional", "professional_warm"]
SocialPlatform = Literal["linkedin", "x", "facebook", "instagram", "youtube", "email"]
SectionStyle = Literal["light", "dark", "brand", "custom"]
Routing = Literal["selected", "matched", "general"]
ImageSlot = Literal["hero", "inline"]

# References. The model only ever writes keys; URLs and files are resolved by the renderer.
LINK_REF = r"^link:[a-z0-9_]{1,40}$"
ASSET_REF = r"^asset:[a-z0-9_]{1,40}$"
BLOCK_ID = r"^[a-z0-9_]{1,16}$"
HEX_COLOR = r"^#[0-9A-Fa-f]{6}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Block(_Strict):
    id: str = Field(pattern=BLOCK_ID)


class HeaderBlock(_Block):
    type: Literal["header"]
    logo_asset: Optional[str] = Field(None, pattern=ASSET_REF)
    org_name: Optional[str] = Field(None, max_length=80)


class GreetingBlock(_Block):
    type: Literal["greeting"]
    text: str = Field(min_length=1, max_length=120)


class HeadingBlock(_Block):
    type: Literal["heading"]
    text: str = Field(min_length=1, max_length=HEADING_MAX)
    level: Literal[1, 2] = 1


class ParagraphBlock(_Block):
    type: Literal["paragraph"]
    # Inline markup allowed: **bold**, *italic*, [text](link:key). Everything else is escaped by the renderer.
    text: str = Field(min_length=1, max_length=PARAGRAPH_MAX)


class ListBlock(_Block):
    type: Literal["list"]
    style: Literal["bulleted", "numbered"] = "bulleted"
    items: List[Annotated[str, Field(min_length=1, max_length=LIST_ITEM_MAX)]] = Field(
        min_length=LIST_ITEMS[0], max_length=LIST_ITEMS[1]
    )


class KeyValueRow(_Strict):
    label: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=200)


class KeyValueBlock(_Block):
    type: Literal["key_value"]
    rows: List[KeyValueRow] = Field(min_length=KV_ROWS[0], max_length=KV_ROWS[1])


class NoticeBlock(_Block):
    type: Literal["notice"]
    variant: Literal["info", "important"] = "info"
    text: str = Field(min_length=1, max_length=300)


class ButtonFields(_Strict):
    label: str = Field(min_length=1, max_length=BUTTON_LABEL_MAX)
    link: str = Field(pattern=r"^(?:link:[a-z0-9_]{1,40}|\[\[fill: [^\]\r\n]{1,120}\]\])$")


class HeroBannerBlock(_Block):
    """Opening section on a coloured background: short label, headline, one-line summary, optional button."""
    type: Literal["hero_banner"]
    eyebrow: Optional[str] = Field(None, max_length=EYEBROW_MAX)
    headline: str = Field(min_length=1, max_length=HERO_HEADLINE_MAX)
    summary: Optional[str] = Field(None, max_length=HERO_SUMMARY_MAX)
    button: Optional[ButtonFields] = None   # counts as the email's one call to action


class FeatureItem(_Strict):
    title: str = Field(min_length=1, max_length=FEATURE_TITLE_MAX)
    text: str = Field(min_length=1, max_length=FEATURE_TEXT_MAX)


class FeaturesBlock(_Block):
    """2-3 short benefit columns side by side (stacked on mobile)."""
    type: Literal["features"]
    items: List[FeatureItem] = Field(min_length=FEATURE_COLUMNS[0], max_length=FEATURE_COLUMNS[1])


class ButtonBlock(_Block, ButtonFields):
    type: Literal["button"]


class ImageBlock(_Block):
    type: Literal["image"]
    slot: ImageSlot
    # null = empty slot. "asset" = user's file, "refined" = AI-edited user file, "generated" = AI-made file.
    source: Optional[Literal["asset", "refined", "generated"]] = None
    asset: Optional[str] = Field(None, pattern=ASSET_REF)
    generation_prompt: Optional[str] = Field(None, max_length=IMAGE_PROMPT_MAX)
    alt: str = Field(min_length=1, max_length=ALT_MAX)


class DividerBlock(_Block):
    type: Literal["divider"]


class SpacerBlock(_Block):
    type: Literal["spacer"]
    size: Literal["small", "medium", "large"] = "medium"


class SignatureBlock(_Block):
    type: Literal["signature"]
    closing: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=80)
    title: Optional[str] = Field(None, max_length=80)
    org: Optional[str] = Field(None, max_length=80)
    contact: Optional[str] = Field(None, max_length=120)


class SocialLink(_Strict):
    platform: SocialPlatform
    link: str = Field(pattern=LINK_REF)


class FooterBlock(_Block):
    type: Literal["footer"]
    address: Optional[str] = Field(None, max_length=200)
    legal: Optional[str] = Field(None, max_length=300)
    unsubscribe: bool = False
    # Filled by code from the request's social links; the model never writes these.
    social: List[SocialLink] = Field(default_factory=list, max_length=SOCIAL_MAX)


Block = Annotated[
    Union[
        HeaderBlock, HeroBannerBlock, GreetingBlock, HeadingBlock, ParagraphBlock, ListBlock, KeyValueBlock,
        NoticeBlock, FeaturesBlock, ButtonBlock, ImageBlock, DividerBlock, SpacerBlock, SignatureBlock, FooterBlock,
    ],
    Field(discriminator="type"),
]

BLOCK_TYPES: tuple[str, ...] = (
    "header", "hero_banner", "greeting", "heading", "paragraph", "list", "key_value", "notice", "features",
    "button", "image", "divider", "spacer", "signature", "footer",
)
# Launch-style blocks, offered only by the categories that announce products and features.
PROMO_BLOCKS = frozenset({"hero_banner", "features"})


class Meta(_Strict):
    category: str = Field(min_length=1, max_length=40)
    language: str = Field("en-GB", pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")
    tone: Tone = "professional"
    routing: Routing = "selected"
    subject_options: List[Annotated[str, Field(min_length=1, max_length=SUBJECT_HARD_MAX)]] = Field(
        min_length=SUBJECT_OPTIONS, max_length=SUBJECT_OPTIONS
    )
    subject: Optional[str] = Field(None, max_length=SUBJECT_HARD_MAX)
    preheader: str = Field(min_length=1, max_length=PREHEADER_HARD_MAX)


class SectionTheme(_Strict):
    """How one section looks. Chosen by the user, never by the model; the renderer applies it."""
    style: SectionStyle = "light"
    background: Optional[str] = Field(None, pattern=HEX_COLOR)   # used when style is "custom"
    text: Optional[str] = Field(None, pattern=HEX_COLOR)

    @model_validator(mode="after")
    def _custom_needs_colour(self):
        if self.style == "custom" and not self.background:
            raise ValueError("a custom section style needs a background colour")
        return self


class Theme(_Strict):
    brand_kit_id: Optional[str] = Field(None, max_length=64)
    accent_color: Optional[str] = Field(None, pattern=HEX_COLOR)   # buttons, links, eyebrow
    hero: Optional[SectionTheme] = None
    footer: Optional[SectionTheme] = None


class Template(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    meta: Meta
    theme: Theme = Field(default_factory=Theme)
    blocks: List[Block] = Field(min_length=MIN_BLOCKS, max_length=MAX_BLOCKS)


# Fields that hold copy a reader sees (and so get tone-linted and sanitised). Everything else is a key,
# enum or reference.
TEXT_FIELDS = frozenset({
    "org_name", "text", "items", "label", "value", "alt", "closing", "name", "title", "org", "contact",
    "address", "legal", "eyebrow", "headline", "summary",
})
