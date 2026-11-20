"""
A caller-chosen layout for /agent/email/generate: an ordered list of block types the email must have, exactly.

The layout replaces the category's skeleton and its required/allowed blocks; the model only writes the copy.
Entries are block types (schema.BLOCK_TYPES) or a few aliases. An image entry becomes "image:hero" or
"image:inline": "image" alone is the hero slot when it opens the email (first, or right after the header or
hero banner), otherwise the inline slot.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .schema import BLOCK_TYPES, MAX_BLOCKS, MIN_BLOCKS

ALIASES: dict[str, str] = {
    "hero": "hero_banner",
    "banner": "hero_banner",
    "cta": "button",
    "hero_image": "image:hero",
    "image_hero": "image:hero",
    "inline_image": "image:inline",
    "image_inline": "image:inline",
    "image:hero": "image:hero",
    "image:inline": "image:inline",
}
_SINGLE = ("header", "footer", "greeting", "notice", "signature", "hero_banner", "features", "button",
           "image:hero", "image:inline")


def block_type(entry: str) -> str:
    """'image:hero' -> 'image'; anything else is already a block type."""
    return entry.partition(":")[0]


def image_slot(entry: str) -> Optional[str]:
    return entry.partition(":")[2] or None


def parse_layout(items: Iterable[str]) -> tuple[str, ...]:
    """Canonical layout entries, or ValueError with a message the API returns as a 400."""
    given = [str(x) for x in items]
    raw = [x.strip().lower().replace("-", "_").replace(" ", "_") for x in given]
    if not MIN_BLOCKS <= len(raw) <= MAX_BLOCKS:
        raise ValueError(f"layout must list {MIN_BLOCKS}-{MAX_BLOCKS} blocks")
    out: list[str] = []
    for i, name in enumerate(raw):
        entry = ALIASES.get(name, name)
        if entry == "image":
            opens = not out or out[-1] in ("header", "hero_banner")
            entry = "image:hero" if opens and "image:hero" not in out else "image:inline"
        if block_type(entry) not in BLOCK_TYPES:
            raise ValueError(f"layout[{i}]: unknown block type '{given[i]}'. Use: "
                             f"{', '.join(BLOCK_TYPES)} (aliases: {', '.join(sorted(ALIASES))})")
        out.append(entry)

    for single in _SINGLE:
        if out.count(single) > 1:
            raise ValueError(f"layout: at most one '{single}' block")
    if "header" in out and out[0] != "header":
        raise ValueError("layout: the header must be the first block")
    if "footer" in out and out[-1] != "footer":
        raise ValueError("layout: the footer must be the last block")
    if "hero_banner" in out and out.index("hero_banner") != (1 if out[0] == "header" else 0):
        raise ValueError("layout: the hero_banner comes first, or directly after the header")
    return tuple(out)


def image_slots(layout: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(image_slot(e) for e in layout if block_type(e) == "image")


def check_sign_off(layout: tuple[str, ...], sign_off: Optional[str]) -> None:
    """A fixed layout decides the signature; a sign_off option that contradicts it is a request error."""
    has_signature = "signature" in layout
    if sign_off == "none" and has_signature:
        raise ValueError("The layout has a signature block, but sign_off is \"none\" (or social links were sent). "
                         "Remove 'signature' from the layout or drop that option.")
    if sign_off == "signature" and not has_signature:
        raise ValueError("sign_off is \"signature\" but the layout has no signature block.")
