"""Flat reference analysis and fidelity checks shared by the harness and live eval."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from math import dist
from typing import Annotated, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .colors import delta_e, hex_rgb, lab
from .schema_v11 import FontFamily, FontRole, FontWeight, HexColor
from .typography_match import family_class, named_weight, tracking_range  # noqa: F401  (re-exported)


class SpecText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1, max_length=2000)
    role: Literal["eyebrow", "display", "heading", "subheading", "body", "label", "button", "caption", "numeral"]
    approx_size_px: int = Field(ge=1, le=1500)
    color_hex: HexColor
    gradient_hex: list[HexColor] | None = Field(None, min_length=2, max_length=5)
    font_class: Literal["geometric_sans", "grotesk_sans", "humanist_sans", "serif", "display_serif", "mono"]
    weight: FontWeight
    uppercase: bool
    letter_spacing: Literal["tight", "normal", "wide"]
    raised_suffix: str | None = Field(None, min_length=1, max_length=12)
    rule_before: bool = False   # a thin vertical rule (|) separates this item from the previous one on its line
    # Tight pixel bounds [x0, y0, x1, y1] of the glyphs (first line only) in the supplied image, for measuring type.
    box: list[Annotated[int, Field(ge=0, le=20000)]] | None = Field(None, min_length=4, max_length=4)

    @model_validator(mode="after")
    def _box_order(self):
        if self.box and not (self.box[0] < self.box[2] and self.box[1] < self.box[3]):
            raise ValueError("box must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1")
        return self


class SpecRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    parent_id: str | None = None
    # row: a line of small items (chips, tabs, links) listed in text_ids; side_by_side and grid: child panel regions.
    arrangement: Literal["stack", "row", "side_by_side", "grid"]
    ratio: list[Annotated[int, Field(ge=1, le=100)]] | None = Field(None, min_length=2, max_length=6)
    columns: int | None = Field(None, ge=1, le=12)          # grid only: panels per row
    separator: Literal["none", "vertical_rule", "gap", "card_edges"] = "none"
    background_hex: HexColor | None = None
    has_border: bool = False
    corner_radius_px: int = Field(0, ge=0, le=999)
    text_ids: list[str]
    # Pixel bounds [x0, y0, x1, y1] in the supplied image: how far down the page the description reaches.
    box: list[Annotated[int, Field(ge=0, le=20000)]] | None = Field(None, min_length=4, max_length=4)


class SpecImage(BaseModel):
    """A photo or picture in the reference: becomes an ImageNode placeholder the photo-fill flow can generate."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]{2,40}$")
    region_id: str | None = None            # the panel or card region that holds it
    box: list[Annotated[int, Field(ge=0, le=20000)]] | None = Field(None, min_length=4, max_length=4)
    description: str = Field(min_length=3, max_length=600)    # what it shows, for an image generator
    aspect: str = "4:3"

    @model_validator(mode="after")
    def _shape_from_box(self):
        """
        The slot's shape comes from the photo's box, reduced (280x186 -> "140:93"), never from the model's guess:
        asked to pick from a list, it once wrote '"aspect":"3:2".length?"16:9":"16:9"' and broke the whole JSON.
        Without a box, a readable "W:H" is kept and anything else becomes 4:3.
        """
        from math import gcd
        if self.box and self.box[2] > self.box[0] and self.box[3] > self.box[1]:
            w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
        else:
            try:
                w, h = (round(float(v)) for v in str(self.aspect).split(":"))
            except ValueError:
                w, h = 4, 3
        if w <= 0 or h <= 0 or not 0.2 <= w / h <= 5:
            w, h = 4, 3
        g = gcd(w, h)
        self.aspect = f"{w // g}:{h // g}"
        return self


class ReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canvas_background_hex: HexColor
    regions: list[SpecRegion] = Field(min_length=1, max_length=100)
    texts: list[SpecText] = Field(max_length=200)
    font_guess: dict[FontRole, FontFamily]
    images: list[SpecImage] = Field(default_factory=list, max_length=60)

    @model_validator(mode="before")
    @classmethod
    def _normalise_hints(cls, data):
        """
        Layout hints the describe call gets slightly wrong are corrected rather than costing the whole description
        (live 2026-11-19: a chip row sent as side_by_side with seven text_ids and no panels rejected the spec, so no
        layout checks and no typography measurement ran). A side_by_side region without two child panels becomes a
        row (two or more texts) or a stack; a ratio that does not have one weight per panel is dropped.
        """
        if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
            return data
        regions = [r for r in data["regions"] if isinstance(r, dict)]
        # Text placement is bookkeeping, not judgement (live 2026-11-19 it cost a 38 s repair call): a text id listed
        # twice keeps its first region, an unknown id is dropped, and texts no region lists go to a catch-all region.
        text_ids = [t.get("id") for t in data.get("texts") or () if isinstance(t, dict)]
        known, placed = set(text_ids), set()
        for region in regions:
            if isinstance(region.get("text_ids"), list):
                kept = []
                for tid in region["text_ids"]:
                    if tid in known and tid not in placed:
                        kept.append(tid)
                        placed.add(tid)
                region["text_ids"] = kept
        unplaced = [tid for tid in text_ids if tid not in placed]
        if unplaced:
            taken = {r.get("id") for r in regions}
            catch_all = next(f"unplaced{n or ''}" for n in range(100) if f"unplaced{n or ''}" not in taken)
            data["regions"].append({"id": catch_all, "arrangement": "stack", "text_ids": unplaced})
            regions.append(data["regions"][-1])
        panels = Counter(r.get("parent_id") for r in regions)
        for region in regions:
            count = panels.get(region.get("id"), 0)
            if region.get("arrangement") == "side_by_side" and count < 2:
                region["arrangement"] = "row" if len(region.get("text_ids") or ()) >= 2 else "stack"
            if region.get("ratio") is not None and (region.get("arrangement") != "side_by_side"
                                                    or not isinstance(region["ratio"], list)
                                                    or len(region["ratio"]) != count):
                region["ratio"] = None
        return data

    @model_validator(mode="after")
    def _valid_references(self):
        regions = {region.id: region for region in self.regions}
        texts = {text.id for text in self.texts}
        if len(regions) != len(self.regions) or len(texts) != len(self.texts):
            raise ValueError("Region and text ids must each be unique")
        if len({image.id for image in self.images}) != len(self.images):
            raise ValueError("Image ids must be unique")
        for image in self.images:
            if image.region_id is not None and image.region_id not in regions:
                image.region_id = None            # a hint, not structure: an unknown region only loses placement
        used_text = Counter(text_id for region in self.regions for text_id in region.text_ids)
        if used_text.keys() != texts or any(count != 1 for count in used_text.values()):
            raise ValueError("Assign each text id to exactly one region")
        panel_counts = Counter(region.parent_id for region in self.regions)
        for region in self.regions:
            if region.arrangement == "side_by_side" and (
                panel_counts[region.id] < 2 or region.ratio and len(region.ratio) != panel_counts[region.id]
            ):
                raise ValueError("A side-by-side region needs one child region per panel and matching ratios")
            parent = region.parent_id
            seen = {region.id}
            while parent is not None:
                if parent not in regions or parent in seen:
                    raise ValueError("Region parents must exist and cannot form cycles")
                seen.add(parent)
                parent = regions[parent].parent_id
        return self


def _plain(text: str) -> str:
    """
    Whitespace-folded text without symbol glyphs (arrows, check marks, ⧉, |): a design may draw those as an icon,
    a trailing_icon or a divider, so "Find my plan ↗" in the reference matches a "Find my plan" button with an arrow.
    """
    return " ".join("".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Sm")).split())


def _read_text(node: dict) -> str:
    text = "".join(span["text"] for span in node.get("spans", ())) if node["node_type"] == "text" else node.get("label", "")
    if node.get("text_transform") == "uppercase":
        text = text.upper()
    return _plain(text)


def _walk(node: dict):
    yield node
    for child in node.get("children", ()):
        yield from _walk(child)


def _own_nodes(nodes: list[dict], text: str) -> list[dict]:
    """Nodes whose whole text is this reference text: the nodes its typography belongs to."""
    wanted = _plain(text).casefold()
    return [node for node in nodes if node["node_type"] in ("text", "button") and _read_text(node).casefold() == wanted]


def _parents(root: dict) -> dict[int, dict]:
    return {id(child): node for node in _walk(root) for child in node.get("children", ())}


def _row_position(node: dict, parents: dict[int, dict]) -> tuple[dict, int] | None:
    """The nearest flex_row holding ``node`` and the index of the child on the path to it."""
    child, parent = node, parents.get(id(node))
    while parent is not None:
        if parent.get("layout") == "flex_row":
            return parent, next(i for i, c in enumerate(parent["children"]) if c is child)
        child, parent = parent, parents.get(id(parent))
    return None


def _has_rule_before(node: dict, parents: dict[int, dict]) -> bool:
    position = _row_position(node, parents)
    if position is None or position[1] == 0:
        return False
    previous = position[0]["children"][position[1] - 1]
    return previous["node_type"] == "divider" and previous.get("orientation") == "vertical"


def _measured_nodes(nodes: list[dict], entry: "SpecText") -> list[dict]:
    """The nodes a measured text styles: its whole text, else (a multi-line text split by the model) its first line."""
    return _own_nodes(nodes, entry.text) or _own_nodes(nodes, entry.text.split("\n")[0])


def _set_measured(node: dict, measured) -> bool:
    """Font role, exact weight and spacing from a typography measurement. True when anything changed."""
    before = json.dumps(node, sort_keys=True)
    node["font_role"] = measured.role
    node["letter_spacing_em"] = measured.letter_spacing_em
    targets = [node] if node["node_type"] == "button" else node["spans"]
    for target in targets:
        target["font_weight_value"] = measured.weight
        target["font_weight"] = named_weight(measured.weight)
    return json.dumps(node, sort_keys=True) != before


def _is_measured(node: dict, measured, fonts: dict) -> bool:
    token = fonts.get(node.get("font_role", "body")) or {}
    targets = [node] if node["node_type"] == "button" else node.get("spans", ())
    return (token.get("family") == measured.family
            and abs(node.get("letter_spacing_em", 0) - measured.letter_spacing_em) <= 0.005
            and all(t.get("font_weight_value") == measured.weight for t in targets))


def heal_reference_details(ast: dict, spec: ReferenceSpec, typography=None) -> list[str]:
    """
    Deterministic fixes on a validated 1.1 AST dump, in place, with no model call. Returns what changed.
    * With a typography match (typography_match.match_typography): the document font tokens it measured, and on each
      measured text its font role, exact weight (font_weight_value, plus the nearest named weight) and spacing.
    * letter_spacing_em of any other reference text outside tracking_range() moves to the nearest edge of it.
    * A reference item with rule_before that sits in a non-wrapping flex_row without one gets a vertical divider
      (the colour of an existing divider, else the item's own colour at 35% opacity).
    Nodes are matched by their whole text, so a short reference text never restyles a sentence that contains it.
    """
    nodes = list(_walk(ast["root"]))
    parents = _parents(ast["root"])
    taken = {node["id"] for node in nodes}
    dividers = [node for node in nodes if node["node_type"] == "divider"]
    rule_colours = [node["color"] for node in sorted(dividers, key=lambda node: node.get("orientation") != "vertical")]
    changes = []
    measured = typography.texts if typography else {}
    if typography and typography.fonts:
        fonts = ast.get("fonts") or {}
        for role, family in typography.fonts.items():
            if (fonts.get(role) or {}).get("family") != family:
                fonts[role] = {"family": family, "fallback": {"mono": "monospace", "serif": "serif"}.get(
                    family_class(family), "sans-serif")}
                changes.append(f"font {role}: {family} (measured)")
        ast["fonts"] = fonts
    for entry in spec.texts:
        if entry.id in measured:
            for node in _measured_nodes(nodes, entry):
                if _set_measured(node, measured[entry.id]):
                    m = measured[entry.id]
                    changes.append(f"typography {node['id']}: {m.family} {m.weight}, {m.letter_spacing_em:+.3f} em (measured)")
        for node in _own_nodes(nodes, entry.text):
            low, high = tracking_range(entry.letter_spacing, node["font_size_px"])
            current = node.get("letter_spacing_em", 0.0)
            if entry.id not in measured and not low <= current <= high:
                node["letter_spacing_em"] = round(min(max(current, low), high), 3)
                changes.append(f"tracking {node['id']}: {current:+.3f} -> {node['letter_spacing_em']:+.3f} em "
                               f"({entry.letter_spacing}, {node['font_size_px']} px)")
            if not entry.rule_before or _has_rule_before(node, parents):
                continue
            position = _row_position(node, parents)
            if position is None or position[1] == 0:
                continue
            row, index = position
            if row.get("wrap") or len(row["children"]) >= 25 or row["children"][index - 1]["node_type"] == "divider":
                continue
            own = node["text_color"] if node["node_type"] == "button" else node["spans"][0]["color"]
            colour = rule_colours[0] if rule_colours else own[:7] + "59"
            rule_id = next(candidate for n in range(1, 100)
                           if (candidate := f"{node['id'][:30]}_rule{'' if n == 1 else n}") not in taken)
            taken.add(rule_id)
            row["children"].insert(index, {
                "node_type": "divider", "id": rule_id, "orientation": "vertical", "color": colour,
                "thickness_px": 1, "width_percent": 100, "margin_y_px": 0, "editable": True,
            })
            changes.append(f"separator {rule_id} before {node['id']} in {row['id']}")
    changes.extend(_heal_images(ast, spec))
    return changes


def _region_texts(spec: ReferenceSpec, region_id: str) -> list[str]:
    ids = [region_id]
    for rid in ids:
        ids.extend(r.id for r in spec.regions if r.parent_id == rid)
    texts = {t.id: t.text for t in spec.texts}
    return [texts[tid] for r in spec.regions if r.id in ids for tid in r.text_ids if tid in texts]


def _region_container(root: dict, spec: ReferenceSpec, region_id: str) -> dict | None:
    """The smallest container holding every text of a reference region (a card, a panel)."""
    wanted = [w for w in (_plain(t) for t in _region_texts(spec, region_id)) if w]
    if not wanted:
        return None
    best = None
    for node in _walk(root):
        if node["node_type"] != "container":
            continue
        text = " ".join(filter(None, map(_read_text, _walk(node))))
        if all(w in text for w in wanted):
            size = sum(1 for _ in _walk(node))
            if best is None or size < best[0]:
                best = (size, node)
    return best[1] if best else None


def _image_alt(description: str) -> str:
    first = re.split(r"[,.;]", description, maxsplit=1)[0].strip()
    return (first or description)[:120]


def _heal_images(ast: dict, spec: ReferenceSpec) -> list[str]:
    """
    Photo slots for the reference's photos, with no model call: an ImageNode with the photo's id gets its source
    box; where the model still drew a photo as a vector_illustration inside the photo's card, that illustration
    becomes an ImageNode (same id, so a selection survives) carrying the description as its generation prompt.
    """
    changes = []
    nodes = list(_walk(ast["root"]))
    by_id = {n["id"]: n for n in nodes if n["node_type"] == "image"}
    for image in spec.images:
        node = by_id.get(image.id)
        if node is None and image.region_id:
            holder = _region_container(ast["root"], spec, image.region_id)
            if holder is not None and not any(n["node_type"] == "image" for n in _walk(holder)):
                stand_in = next((n for n in _walk(holder) if n["node_type"] == "vector_illustration"), None)
                if stand_in is not None:
                    keep_id = stand_in["id"]
                    stand_in.clear()
                    stand_in.update({
                        "node_type": "image", "id": keep_id, "alt": _image_alt(image.description),
                        "prompt": image.description, "src": None, "placeholder_id": None,
                        "aspect_ratio": image.aspect, "width_percent": 100, "object_fit": "cover",
                        "source_box": image.box, "style": {"border_radius": 12}, "editable": True})
                    changes.append(f"image {keep_id}: drawn stand-in replaced by a photo placeholder ({image.id})")
                    node = stand_in
        if node is not None and image.box and node.get("source_box") != image.box:
            node["source_box"] = list(image.box)
    return changes


def _painted_colours(node: dict):
    style = node.get("style") or {}
    gradient = style.get("background_gradient")
    if gradient:
        yield from gradient["stops"]
    elif style.get("background_color"):
        yield style["background_color"]
    if style.get("border_width") and style.get("border_color"):
        yield style["border_color"]
    if style.get("shadow") == "glow" and style.get("shadow_color"):
        yield style["shadow_color"]
    for field in ("color", "text_color"):
        if node.get(field):
            yield node[field]
    for span in node.get("spans", ()):
        yield from (span.get("gradient") or {}).get("stops", (span["color"],))
    for layer in node.get("layers", ()):
        yield layer["fill_color"]
        if layer.get("stroke_color") and layer.get("stroke_width"):
            yield layer["stroke_color"]


def _matches_layout(node: dict, patterns: list[list[re.Pattern]], region: SpecRegion, v11: bool) -> bool:
    children = node.get("children", ())
    dividers = [i for i, child in enumerate(children) if child.get("orientation") == "vertical"]
    panels = [child for i, child in enumerate(children) if i not in dividers]
    if len(panels) != len(patterns):
        return False
    is_grid = node.get("layout") == "grid"
    if is_grid:
        if node["columns"] != len(panels) or dividers:
            return False
        widths = node.get("column_ratios") or [1] * len(panels)
    elif node.get("layout") == "flex_row" and not node.get("wrap"):
        styles = [panel.get("style") or {} for panel in panels]
        if v11 and region.ratio and not all(style.get("flex_grow") for style in styles):
            return False
        widths = [style.get("flex_grow") or style.get("width_percent") or 1 for style in styles]
    else:
        return False
    if v11 and region.separator == "vertical_rule" and dividers != list(range(1, len(children), 2)):
        return False
    if v11 and region.separator in ("vertical_rule", "card_edges") and is_grid:
        return False
    if v11 and region.separator in ("gap", "none") and not is_grid:
        return False
    if region.ratio and any(abs(a / sum(widths) - b / sum(region.ratio)) > 0.1 for a, b in zip(widths, region.ratio)):
        return False
    for panel, group in zip(panels, patterns):
        text = " ".join(filter(None, map(_read_text, _walk(panel))))
        if not all(pattern.search(text) for pattern in group):
            return False
    return True


def _child_holding(node: dict, group: list[re.Pattern]) -> int | None:
    for i, child in enumerate(node.get("children", ())):
        text = " ".join(filter(None, map(_read_text, _walk(child))))
        if all(pattern.search(text) for pattern in group):
            return i
    return None


def _matches_flow(node: dict, groups: list[list[re.Pattern]], region: SpecRegion) -> bool:
    """
    A row: every item in a direct child of one flex_row or grid, spread over two or more children (a chip with its
    badge may share a child). A grid: layout grid, the reference's column count when given, one panel per child.
    """
    layout = node.get("layout")
    holders = [_child_holding(node, group) for group in groups]
    if any(h is None for h in holders):
        return False
    if region.arrangement == "row":
        return layout in ("flex_row", "grid") and len(set(holders)) >= 2
    if layout != "grid" or len(set(holders)) != len(holders):
        return False
    return node.get("columns") == region.columns if region.columns else (node.get("columns") or 0) >= 2


def _check_structure(ast: dict, spec: ReferenceSpec, nodes: list[dict], metrics: dict, errors: list[str],
                     v11: bool) -> None:
    """Template checks for new content: the reference's grids, side-by-side panels, photo slots and fonts."""
    containers = [n for n in nodes if n["node_type"] == "container"]
    for region in spec.regions:
        panels = [r for r in spec.regions if r.parent_id == region.id]
        if region.arrangement == "grid" and region.columns:
            metrics["layout_regions"] += 1
            if any(n.get("layout") == "grid" and n.get("columns") == region.columns and len(n["children"]) >= 2
                   for n in containers):
                metrics["layout_matches"] += 1
            else:
                errors.append(f"FIDELITY_LAYOUT_STACKED: Keep the reference's {region.columns}-column grid "
                              f"(region {region.id!r}): layout grid with columns {region.columns}.")
        elif region.arrangement == "side_by_side" and len(panels) >= 2:
            metrics["layout_regions"] += 1

            def side_by_side(n):
                kids = [c for c in n["children"] if c.get("orientation") != "vertical"]
                rules = len(n["children"]) - len(kids)
                row = n.get("layout") == "grid" or (n.get("layout") == "flex_row" and not n.get("wrap"))
                return row and len(kids) == len(panels) and (region.separator != "vertical_rule" or rules or not v11)

            if any(side_by_side(n) for n in containers):
                metrics["layout_matches"] += 1
            else:
                errors.append(f"FIDELITY_LAYOUT_STACKED: Region {region.id!r} needs {len(panels)} panels in one row "
                              f"with separator {region.separator!r}.")
    if spec.images:
        slots = sum(1 for n in nodes if n["node_type"] == "image")
        metrics["images_expected"] = len(spec.images)
        metrics["images_matched"] = min(slots, len(spec.images))
        if v11 and slots < len(spec.images):
            errors.append(f"FIDELITY_IMAGE_MISSING: The reference has {len(spec.images)} photos; the design has "
                          f"{slots} ImageNode slots. Add the missing photo slots.")
    fonts = ast.get("fonts") or {}
    metrics["fonts_set"] = all(fonts.get(role) for role in spec.font_guess) if spec.font_guess else bool(fonts)
    if v11 and spec.font_guess and not metrics["fonts_set"]:
        errors.append("FIDELITY_FONTS_MISSING: Set fonts for the roles in reference_spec.font_guess.")


COVERAGE_GAP_WITH_REGION_BOXES = 0.15   # share of the page height from the lowest described box to the last content
COVERAGE_GAP_TEXT_BOXES_ONLY = 0.50     # only large texts and photos are boxed: lower ones are unboxed, not missing


def description_coverage(spec: ReferenceSpec, image_bytes: bytes) -> dict | None:
    """
    How far down the supplied image the description reaches, against where the image's content ends.

    Live 2026-11-19 a description stopped after the page header (11 texts, no cards, no photos) and every later check
    passed vacuously. ``complete`` is False when the span from the lowest described box to the last content exceeds
    COVERAGE_GAP_WITH_REGION_BOXES of the page height, or COVERAGE_GAP_TEXT_BOXES_ONLY when no region has a box.
    None when nothing is boxed or the image is blank: then coverage cannot be judged.
    """
    import io
    import statistics

    from PIL import Image

    region_boxes = [r.box for r in spec.regions if r.box]
    boxes = region_boxes + [t.box for t in spec.texts if t.box] + [i.box for i in spec.images if i.box]
    if not boxes:
        return None
    with Image.open(io.BytesIO(image_bytes)) as img:
        full_w, full_h = img.size
        scale = min(1.0, 300 / full_w)
        small = img.convert("RGB").resize((max(1, round(full_w * scale)), max(1, round(full_h * scale))),
                                          Image.Resampling.BOX)
    w, h = small.size
    pixels = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
    border = pixels[:w] + pixels[-w:] + pixels[::w] + pixels[w - 1::w]
    bg = tuple(int(statistics.median(p[c] for p in border)) for c in range(3))
    content_rows = [y for y in range(h) if sum(
        1 for p in pixels[y * w:(y + 1) * w] if max(abs(p[0] - bg[0]), abs(p[1] - bg[1]), abs(p[2] - bg[2])) > 30
    ) > 0.004 * w]
    if not content_rows:
        return None
    described_bottom = max(b[3] for b in boxes)
    content_bottom = (content_rows[-1] + 1) / scale
    # The page height between the lowest described box and the last content: rows of sparse text count fully
    # (counting only rows with ink scored a header-only description of a text page 0.23).
    uncovered = max(0.0, content_bottom - described_bottom) / full_h
    threshold = COVERAGE_GAP_WITH_REGION_BOXES if region_boxes else COVERAGE_GAP_TEXT_BOXES_ONLY
    return {"described_bottom": described_bottom, "content_bottom": round(content_bottom),
            "height": full_h, "uncovered": round(uncovered, 3), "basis": "regions" if region_boxes else "texts",
            "complete": uncovered <= threshold}


def _font_class(family: str) -> str:
    if "Mono" in family:
        return "mono"
    if "Serif" in family or family in {"Playfair Display", "Fraunces"}:
        return "serif"
    return "sans" if family else "unknown"


def check_fidelity(
    ast: dict,
    spec: ReferenceSpec | None,
    quantized_hex_colors: Iterable[str] = (),
    expected_accents: Iterable[str] = (),
    typography=None,
    content: str = "reference",
) -> tuple[dict, list[str]]:
    """
    Score the rendered fields and return repair hints. Inputs are validated AST dumps. ``typography`` is a
    typography_match result: its texts are checked for the measured family, weight and spacing instead of a range.
    ``content="new"``: the screenshot is a template for new content (design_intent.wants_new_content), so nothing
    that depends on its words is checked; structure (grid columns, panel counts, photo slots), fonts and colours are.
    """
    nodes = list(_walk(ast["root"]))
    colour_nodes = {color: node["id"] for node in nodes for color in _painted_colours(node)}
    v11 = ast["version"] == "1.1"
    errors = []
    metrics = {
        "text_recall": None, "layout_matches": 0, "layout_regions": 0, "fonts_set": False,
        "font_class_matches": 0, "font_class_texts": 0, "accent_coverage": None,
        "separator_matches": 0, "separator_texts": 0, "typography_matches": 0, "typography_texts": 0,
        "images_matched": 0, "images_expected": 0,
    }
    measured = typography.texts if typography else {}
    parents = _parents(ast["root"])
    if spec and content == "new":
        _check_structure(ast, spec, nodes, metrics, errors, v11)
    elif spec:
        node_texts = [(node, text) for node in nodes if (text := _read_text(node))]
        all_text = " ".join(text for _, text in node_texts)
        patterns = {
            entry.id: re.compile(r"(?<!\w)" + re.escape(_plain(entry.text)) + r"(?!\w)")
            for entry in spec.texts if _plain(entry.text)
        }
        found = 0
        fonts = ast.get("fonts") or {}
        expected_counts = Counter()
        for entry in spec.texts:
            if entry.id not in patterns:
                continue
            pattern = patterns[entry.id]
            expected_counts[pattern] += 1
            if len(pattern.findall(all_text)) < expected_counts[pattern]:
                errors.append(f"FIDELITY_TEXT_MISSING: Preserve the reference text {entry.text[:80]!r}.")
                continue
            found += 1
            matching = [node for node, text in node_texts if pattern.search(text)]
            if matching:
                metrics["font_class_texts"] += 1
                expected_class = ("serif" if entry.font_class.endswith("serif") else
                                  "mono" if entry.font_class == "mono" else "sans")
                if any(_font_class((fonts.get(node.get("font_role", "body")) or fonts.get("body") or {}).get("family", ""))
                       == expected_class for node in matching):
                    metrics["font_class_matches"] += 1
                elif v11:
                    errors.append(f"FIDELITY_FONT_CLASS: Use a {expected_class} font token for {entry.text[:60]!r}.")
            if not v11:
                continue
            if entry.gradient_hex and not any(span.get("gradient") for node in matching for span in node.get("spans", ())):
                errors.append(f"FIDELITY_GRADIENT_MISSING: Use gradient text for {entry.text[:60]!r}.")
            if entry.raised_suffix and not any(
                span.get("baseline") == "super" and span["text"].strip() == entry.raised_suffix
                for node in matching for span in node.get("spans", ())
            ):
                errors.append(f"FIDELITY_BASELINE: Set baseline to super for {entry.raised_suffix!r} in {entry.text[:60]!r}.")
            own = _own_nodes(nodes, entry.text)
            if entry.id in measured:
                m = measured[entry.id]
                metrics["typography_texts"] += 1
                styled = _measured_nodes(nodes, entry)
                if styled and all(_is_measured(node, m, fonts) for node in styled):
                    metrics["typography_matches"] += 1
                else:
                    errors.append(f"FIDELITY_TYPOGRAPHY: Set {entry.text[:60]!r} in {m.family} weight {m.weight} with "
                                  f"letter_spacing_em {m.letter_spacing_em:+.3f} (measured from the image).")
            elif own and not all(
                tracking_range(entry.letter_spacing, node["font_size_px"])[0] - 1e-9
                <= node.get("letter_spacing_em", 0) <= tracking_range(entry.letter_spacing, node["font_size_px"])[1] + 1e-9
                for node in own
            ):
                low, high = tracking_range(entry.letter_spacing, own[0]["font_size_px"])
                errors.append(f"FIDELITY_TRACKING: Use {entry.letter_spacing} letter spacing ({low:+.3f} to "
                              f"{high:+.3f} em) for {entry.text[:60]!r}.")
            if entry.rule_before and own:
                metrics["separator_texts"] += 1
                if any(_has_rule_before(node, parents) for node in own):
                    metrics["separator_matches"] += 1
                else:
                    errors.append(f"FIDELITY_SEPARATOR_MISSING: Put a vertical divider directly before "
                                  f"{entry.text[:60]!r} in its flex_row.")
        metrics["text_recall"] = found / len(patterns) if patterns else None

        def panel_patterns(panel_id):
            ids = [panel_id]
            for region_id in ids:
                ids.extend(region.id for region in spec.regions if region.parent_id == region_id)
            return [patterns[text_id] for region in spec.regions if region.id in ids
                    for text_id in region.text_ids if text_id in patterns]

        containers = [node for node in nodes if node["node_type"] == "container"]
        for region in spec.regions:
            if region.arrangement in ("row", "grid"):
                if region.arrangement == "row":
                    groups = [[patterns[t]] for t in region.text_ids if t in patterns]
                else:
                    groups = [panel_patterns(panel.id) for panel in spec.regions if panel.parent_id == region.id]
                if len(groups) < 2 or not all(groups):
                    continue
                metrics["layout_regions"] += 1
                if any(_matches_flow(node, groups, region) for node in containers):
                    metrics["layout_matches"] += 1
                elif region.arrangement == "row":
                    errors.append(f"FIDELITY_LAYOUT_STACKED: The {len(groups)} items of region {region.id!r} sit on "
                                  f"one line: put them in one flex_row.")
                else:
                    errors.append(f"FIDELITY_LAYOUT_STACKED: Region {region.id!r} is a grid of {len(groups)} panels"
                                  f"{f' in {region.columns} columns' if region.columns else ''}: use layout grid with "
                                  f"one panel per cell.")
                continue
            if region.arrangement != "side_by_side":
                continue
            groups = [panel_patterns(panel.id) for panel in spec.regions if panel.parent_id == region.id]
            if not all(groups):
                continue
            metrics["layout_regions"] += 1
            if any(_matches_layout(node, groups, region, v11) for node in nodes if node["node_type"] == "container"):
                metrics["layout_matches"] += 1
            else:
                errors.append(f"FIDELITY_LAYOUT_STACKED: Region {region.id!r} needs {len(groups)} panels in one row "
                              f"with ratios {region.ratio or 'as shown'} and separator {region.separator!r}.")

        if spec.images:
            image_nodes = [n for n in nodes if n["node_type"] == "image"]
            image_ids = {n["id"] for n in image_nodes}
            metrics["images_expected"] = len(spec.images)
            for image in spec.images:
                found = image.id in image_ids
                if not found and image.region_id:
                    holder = _region_container(ast["root"], spec, image.region_id)
                    found = holder is not None and any(n["node_type"] == "image" for n in _walk(holder))
                elif not found:
                    found = len(image_nodes) >= len(spec.images)
                if found:
                    metrics["images_matched"] += 1
                elif v11:
                    errors.append(f"FIDELITY_IMAGE_MISSING: Add an ImageNode (photo placeholder) for "
                                  f"{image.description[:60]!r} where the reference shows it.")

        metrics["fonts_set"] = all(fonts.get(role) for role in spec.font_guess) if spec.font_guess else bool(fonts)
        if v11 and spec.font_guess and not metrics["fonts_set"]:
            errors.append("FIDELITY_FONTS_MISSING: Set fonts for the roles in reference_spec.font_guess.")
    palette_lab = {color: lab(hex_rgb(color)) for color in quantized_hex_colors}
    for color in sorted(colour_nodes):
        if not palette_lab:
            break
        color_lab = lab(hex_rgb(color))
        nearest = min(palette_lab, key=lambda ref: dist(color_lab, palette_lab[ref]))
        if dist(color_lab, palette_lab[nearest]) > 12:
            errors.append(f"FIDELITY_COLOUR_OFF_PALETTE: Node {colour_nodes[color]!r} uses {color}. "
                          f"Nearest measured shade is {nearest}. Keep {color} if it is visibly present.")
    accents = list(expected_accents)
    if accents:
        metrics["accent_coverage"] = sum(any(delta_e(actual, wanted) <= 8 for actual in colour_nodes)
                                         for wanted in accents) / len(accents)
    priority = {"FIDELITY_LAYOUT_STACKED": 0, "FIDELITY_TEXT_MISSING": 1, "FIDELITY_FONTS_MISSING": 2}
    return metrics, sorted(errors, key=lambda error: priority.get(error.split(":", 1)[0], 3))
