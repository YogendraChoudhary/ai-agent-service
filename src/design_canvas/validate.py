"""
Deterministic validator and auto-healer for Mode 8 DesignDocumentAST.

validate_design_ast()  – runs the deterministic checks and returns a list
                          of human-readable error strings (empty = valid).
auto_heal_design_ast() – last-resort code auto-healer: its output always passes
                          validate_design_ast(), so a 500 never fires if the
                          AST parsed successfully.

Checks performed:
  1. Unique node ``id`` and ``layer_id`` across the entire AST tree.
  2. SVG attribute whitelist (blocks XSS / script injection).
  3. WCAG 2.1 relative-luminance contrast ≥ MIN_TEXT_CONTRAST (3.0:1) for text
     spans vs their nearest ancestor's background colour.
  4. Maximum tree depth ≤ MAX_TREE_DEPTH (6).
  5. Non-empty containers (guaranteed by Pydantic).
  6. Sibling ID preservation when editing an existing AST: a node that is
     unchanged apart from its id must keep its original id. Deleting or
     replacing nodes is a legitimate edit and is not flagged.
"""

from __future__ import annotations

import json
import re
from math import dist, hypot
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import ValidationError

from src.design_canvas.schema import (
    ContainerNode,
    DesignDocumentAST,
    DesignNode,
    NodeStyle,
    TextNode,
    VectorIllustrationNode,
)
from src.design_canvas.colors import hex_rgb, lab, linearize

if TYPE_CHECKING:
    from src.design_canvas.vision_extractor import VisionMetadata

MIN_TEXT_CONTRAST = 3.0   # slightly below AA 4.5:1 to allow large heading text
MAX_TREE_DEPTH = 6
MAX_CHILDREN = 25         # ContainerNode.children max_length
MAX_ID_LENGTH = 40        # NODE_ID_PATTERN / LAYER_ID_PATTERN

# ---------------------------------------------------------------------------
# SVG attribute whitelist (geometry-only; no event handlers, no hrefs)
# ---------------------------------------------------------------------------
_SVG_ALLOWED_ATTRS: Set[str] = {
    "d", "cx", "cy", "r", "rx", "ry",
    "x", "y", "x1", "y1", "x2", "y2",
    "width", "height", "transform", "points",
}
_SVG_BLOCKED_PATTERN = re.compile(
    r"(?:on\w+|href|xlink:href|src|action|formaction|data:|javascript:)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# WCAG 2.1 luminance helpers
# ---------------------------------------------------------------------------

def _relative_luminance(hex_color: str) -> Optional[float]:
    """Return WCAG 2.1 relative luminance [0,1], or None on parse error."""
    if len(hex_color.removeprefix("#")) not in (3, 4, 6, 8):
        return None
    try:
        r, g, b = map(linearize, hex_rgb(hex_color))
    except ValueError:
        return None
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(lum_text: float, lum_bg: float) -> float:
    lighter = max(lum_text, lum_bg)
    darker = min(lum_text, lum_bg)
    return (lighter + 0.05) / (darker + 0.05)


def _background_colors(style: NodeStyle) -> tuple[str, ...]:
    gradient = getattr(style, "background_gradient", None)
    solid = (style.background_color,) if style.background_color else ()
    return (*solid, *gradient.stops) if gradient else solid


def _low_contrast_spans(node: TextNode, backgrounds: tuple[str, ...]) -> Iterator[tuple[int, int | None, float, str]]:
    """Yield span/stop indices, worst ratio and background for each colour that fails."""
    if not backgrounds:
        return
    luminances = [(bg, _relative_luminance(bg)) for bg in backgrounds]
    for i, span in enumerate(node.spans):
        gradient = getattr(span, "gradient", None)
        colors = [(None, span.color)]
        if gradient:
            colors.extend(enumerate(gradient.stops))
        for stop, color in colors:
            lum_text = _relative_luminance(color)
            ratio, bg_hex = min((_contrast_ratio(lum_text, lum_bg), bg) for bg, lum_bg in luminances)
            if ratio < MIN_TEXT_CONTRAST:
                yield i, stop, ratio, bg_hex


# ---------------------------------------------------------------------------
# Tree walker helpers
# ---------------------------------------------------------------------------

def _collect_all_nodes(
    node: DesignNode,
    depth: int = 0,
    backgrounds: tuple[str, ...] = (),
) -> Iterator[tuple[DesignNode, int, tuple[str, ...]]]:
    """Walk nodes with the nearest painted background colours."""
    if isinstance(node, (TextNode, ContainerNode)):
        backgrounds = _background_colors(node.style) or backgrounds
    yield node, depth, backgrounds
    if isinstance(node, ContainerNode):
        for child in node.children:
            yield from _collect_all_nodes(child, depth + 1, backgrounds)


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

def validate_design_ast(
    ast: DesignDocumentAST,
    vision_meta: "Optional[VisionMetadata]" = None,
    existing_ast_json: Optional[str] = None,
) -> List[str]:
    """
    Run all deterministic domain checks on a parsed DesignDocumentAST.

    Args:
        ast:              Parsed and Pydantic-validated AST.
        vision_meta:      Vision metadata (used in future checks; reserved).
        existing_ast_json: Original AST JSON string for sibling-ID preservation
                           check in edit mode.

    Returns:
        A list of human-readable error strings.  Empty list = all checks passed.
    """
    errors: List[str] = []

    all_nodes = list(_collect_all_nodes(ast.root))

    # ------------------------------------------------------------------
    # Check 1: Unique node IDs
    # ------------------------------------------------------------------
    seen_node_ids: Dict[str, int] = {}
    for node, _, _ in all_nodes:
        nid = getattr(node, "id", None)
        if nid:
            seen_node_ids[nid] = seen_node_ids.get(nid, 0) + 1
    for nid, count in seen_node_ids.items():
        if count > 1:
            errors.append(f"DUPLICATE_NODE_ID: `{nid}` appears {count} times in the AST")

    # ------------------------------------------------------------------
    # Check 2: Unique SVG layer IDs + attribute whitelist
    # ------------------------------------------------------------------
    seen_layer_ids: Dict[str, int] = {}
    for node, _, _ in all_nodes:
        if isinstance(node, VectorIllustrationNode):
            for layer in node.layers:
                lid = layer.layer_id
                seen_layer_ids[lid] = seen_layer_ids.get(lid, 0) + 1
                for attr_key, attr_val in layer.attributes.items():
                    # Key whitelist
                    if attr_key.lower() not in _SVG_ALLOWED_ATTRS:
                        errors.append(
                            f"SVG_FORBIDDEN_ATTR: layer `{lid}` attribute `{attr_key}` is not in the geometry whitelist"
                        )
                    # Value XSS scan
                    val_str = str(attr_val)
                    if _SVG_BLOCKED_PATTERN.search(val_str):
                        errors.append(
                            f"SVG_XSS_ATTR_VALUE: layer `{lid}` attribute `{attr_key}` contains a blocked pattern"
                        )

    for lid, count in seen_layer_ids.items():
        if count > 1:
            errors.append(f"DUPLICATE_LAYER_ID: `{lid}` appears {count} times in VectorIllustrationNodes")

    # ------------------------------------------------------------------
    # Check 3: WCAG 2.1 contrast — text spans vs nearest ancestor background. In edit mode only what the edit made:
    # a colour the original already had on the same background is the user's choice.
    # ------------------------------------------------------------------
    kept = _preexisting_low_contrast(ast, existing_ast_json) if existing_ast_json else set()
    for node, _, ancestor_bg in all_nodes:
        if not isinstance(node, TextNode):
            continue
        for i, stop, ratio, bg_hex in _low_contrast_spans(node, ancestor_bg):
            if _contrast_key(node, i, stop, ancestor_bg) in kept:
                continue
            target = f" gradient stop {stop + 1}" if stop is not None else ""
            errors.append(
                f"WCAG_LOW_CONTRAST: node `{node.id}` span \"{node.spans[i].text[:20]}\"{target} "
                f"has contrast ratio {ratio:.2f}:1 against background {bg_hex} "
                f"(minimum {MIN_TEXT_CONTRAST}:1)"
            )

    # ------------------------------------------------------------------
    # Check 4: Maximum tree depth ≤ 6
    # ------------------------------------------------------------------
    for node, depth, _ in all_nodes:
        if depth > MAX_TREE_DEPTH:
            nid = getattr(node, "id", "?")
            errors.append(f"MAX_DEPTH_EXCEEDED: node `{nid}` is at depth {depth} (max {MAX_TREE_DEPTH})")

    # ------------------------------------------------------------------
    # Check 5: Sibling ID preservation (edit mode only). Deleting or replacing a node is a legitimate edit;
    # giving an unchanged node a new id is not (the editor loses the user's selection on it).
    # ------------------------------------------------------------------
    if existing_ast_json:
        for node, original_id in _renamed_nodes(ast, existing_ast_json):
            errors.append(
                f"RENAMED_NODE_ID: node `{node.id}` is unchanged from original node `{original_id}`; "
                f"keep its original id `{original_id}`"
            )

    return errors


def _contrast_key(node: TextNode, span: int, stop: Optional[int], backgrounds: tuple[str, ...]) -> tuple:
    """A low-contrast colour as (node id, colour, backgrounds), case-insensitive, to compare with the original's."""
    color = node.spans[span].color if stop is None else node.spans[span].gradient.stops[stop]
    return node.id, color.upper(), tuple(bg.upper() for bg in backgrounds)


def _original_ast(ast: DesignDocumentAST, existing_ast_json: str) -> Optional[DesignDocumentAST]:
    try:
        return type(ast).model_validate_json(existing_ast_json)
    except (ValidationError, ValueError, TypeError):
        return None


def _preexisting_low_contrast(ast: DesignDocumentAST, existing_ast_json: str) -> Set[tuple]:
    """
    (node id, colour, backgrounds) of every low-contrast span colour the original design already had. An edit leaves
    these alone: audit 2026-11-20, a label the user set to #3A3A40 on #18100C came back #FFFFFF after "make the button
    blue". The same node with a new colour, or on a new background, is the edit's doing and is still checked.
    """
    original = _original_ast(ast, existing_ast_json)
    if original is None:
        return set()
    found = set()
    for node, _, backgrounds in _collect_all_nodes(original.root):
        if isinstance(node, TextNode):
            for i, stop, _, _ in _low_contrast_spans(node, backgrounds):
                found.add(_contrast_key(node, i, stop, backgrounds))
    return found


def _node_signature(node: DesignNode) -> str:
    """Everything about a node except its id and its children, for spotting a node that only got a new id."""
    return json.dumps(node.model_dump(exclude={"id", "children"}), sort_keys=True)


def _renamed_nodes(ast: DesignDocumentAST, existing_ast_json: str) -> List[Tuple[DesignNode, str]]:
    """
    (node, original id) for each node of ``ast`` that has an id the original AST did not have but is otherwise
    identical to an original node whose id is gone. Paired one to one, in tree order. An original that does
    not parse cannot be compared, so it yields nothing.
    """
    original = _original_ast(ast, existing_ast_json)
    if original is None:
        return []
    original_nodes = [n for n, _, _ in _collect_all_nodes(original.root)]
    new_nodes = [n for n, _, _ in _collect_all_nodes(ast.root)]
    original_ids = {n.id for n in original_nodes}
    new_ids = {n.id for n in new_nodes}

    dropped: Dict[str, List[str]] = {}
    for n in original_nodes:
        if n.id not in new_ids:
            dropped.setdefault(_node_signature(n), []).append(n.id)

    renamed: List[Tuple[DesignNode, str]] = []
    for n in new_nodes:
        if n.id in original_ids:
            continue
        candidates = dropped.get(_node_signature(n))
        if candidates:
            renamed.append((n, candidates.pop(0)))
    return renamed


def keep_selected_id(ast: DesignDocumentAST, existing_ast_json: str, selected_id: Optional[str]) -> DesignDocumentAST:
    """
    The edited design with the editor's selected node under its original id. A model that edits the selected node
    sometimes renames it (audit 2026-11-20: `badge_container` came back as `badge_container_new`); _renamed_nodes only
    pairs nodes that are otherwise identical, so the editor lost its selection. The node of the same type at the
    selected node's place (same parent, nearest position) with an id the original did not have takes the id back.
    """
    if not selected_id:
        return ast
    original = _original_ast(ast, existing_ast_json)
    if original is None:
        return ast
    placed = list(_placed(ast.root))
    if any(n.id == selected_id for n, _, _ in placed):
        return ast
    where = next(((parent, index, n.node_type) for n, parent, index in _placed(original.root) if n.id == selected_id), None)
    if where is None or where[0] is None:
        return ast
    old_parent, old_index, node_type = where
    original_ids = {n.id for n, _, _ in _placed(original.root)}
    candidates = [(n, index) for n, parent, index in placed
                  if parent == old_parent and n.node_type == node_type and n.id not in original_ids]
    if not candidates:
        return ast
    node = min(candidates, key=lambda c: abs(c[1] - old_index))[0]
    healed = ast.model_copy(deep=True)
    next(n for n, _, _ in _placed(healed.root) if n.id == node.id).id = selected_id
    return type(ast).model_validate(healed.model_dump())


# ---------------------------------------------------------------------------
# Auto-healer (last resort — guarantees zero 500s after successful Pydantic parse)
# ---------------------------------------------------------------------------

def auto_heal_design_ast(
    ast: DesignDocumentAST,
    vision_meta: "Optional[VisionMetadata]" = None,
    existing_ast_json: Optional[str] = None,
) -> DesignDocumentAST:
    """
    Deterministic code auto-healer applied after all LLM repair attempts fail.
    Its output passes ``validate_design_ast(healed, existing_ast_json=...)``.

    Mutations, in order:
    * Replace each container that would hold nodes deeper than MAX_TREE_DEPTH
      by its leaf nodes (they move up; a container left with more than
      MAX_CHILDREN children keeps the first MAX_CHILDREN).
    * Restore the original id of a node that is unchanged apart from a new id
      (edit mode).
    * Deduplicate node IDs and SVG layer IDs with a ``_2``, ``_3``, … suffix,
      shortened to stay within the 40-character id limit. In edit mode the
      node that matches the original keeps an original id; the newcomer is renamed.
    * Strip forbidden SVG attributes.
    * Recolour low-contrast text and gradient stops using the closest legible
      reference shade of the same hue, or the better of black and white
      (in edit mode, only colours the original did not already have there).

    Never changes text content or layout properties.

    Returns a new (re-validated) DesignDocumentAST instance.
    """
    healed = ast.model_copy(deep=True)
    _flatten_too_deep(healed.root, 0)
    if existing_ast_json:
        for node, original_id in _renamed_nodes(healed, existing_ast_json):
            node.id = original_id
    original = _original_ast(healed, existing_ast_json) if existing_ast_json else None
    _dedupe_ids(healed, original)
    _strip_svg_attrs(healed)
    _fix_contrast(healed, vision_meta, _preexisting_low_contrast(healed, existing_ast_json) if existing_ast_json else set())
    # Re-parse so Pydantic re-validates every mutation.
    return type(ast).model_validate(healed.model_dump())


def _leaves(node: DesignNode) -> List[DesignNode]:
    if isinstance(node, ContainerNode):
        return [leaf for child in node.children for leaf in _leaves(child)]
    return [node]


def _flatten_too_deep(container: ContainerNode, depth: int) -> None:
    """``container`` is at ``depth``; a child container at MAX_TREE_DEPTH would put its children past it."""
    children: List[DesignNode] = []
    for child in container.children:
        if isinstance(child, ContainerNode):
            if depth + 1 >= MAX_TREE_DEPTH:
                children.extend(leaf for leaf in _leaves(child) if getattr(leaf, "orientation", None) != "vertical")
                continue
            _flatten_too_deep(child, depth + 1)
        children.append(child)
    container.children = children[:MAX_CHILDREN]
    if getattr(container.children[-1], "orientation", None) == "vertical":
        container.children.pop()


def _unique_id(base: str, taken: Set[str]) -> str:
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = base[: MAX_ID_LENGTH - len(suffix)] + suffix
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        n += 1


def _placed(node: DesignNode, parent: Optional[str] = None, index: int = 0) -> Iterator[tuple[DesignNode, Optional[str], int]]:
    """(node, parent id, index among its siblings) in tree order."""
    yield node, parent, index
    if isinstance(node, ContainerNode):
        for i, child in enumerate(node.children):
            yield from _placed(child, node.id, i)


def _child_ids(node: DesignNode) -> Set[str]:
    return {child.id for child in node.children} if isinstance(node, ContainerNode) else set()


def _dedupe_ids(ast: DesignDocumentAST, original: Optional[DesignDocumentAST] = None) -> None:
    """
    Later nodes (layers) with a used id get a fresh id that no node (layer) uses. Which one keeps the id: in edit mode,
    for an id the original had, the occurrence most like the original node (content, children, parent, position);
    otherwise the first. Audit 2026-11-20: a new button inserted before `btn_order_now` with the same id kept it, and
    the user's own button became `btn_order_now_2` (the editor lost the selection on it).
    """
    placed = list(_placed(ast.root))
    nodes = [n for n, _, _ in placed]
    layers = [layer for n in nodes if isinstance(n, VectorIllustrationNode) for layer in n.layers]

    keeper: Dict[str, DesignNode] = {}
    if original is not None:
        before = {n.id: (n, parent, index) for n, parent, index in _placed(original.root)}
        occurrences: Dict[str, List[tuple]] = {}
        for n, parent, index in placed:
            occurrences.setdefault(n.id, []).append((n, parent, index))
        for nid, found in occurrences.items():
            if len(found) < 2 or nid not in before:
                continue
            old, old_parent, old_index = before[nid]
            signature, old_children = _node_signature(old), _child_ids(old)
            keeper[nid] = max(found, key=lambda f: (
                _node_signature(f[0]) == signature, len(_child_ids(f[0]) & old_children),
                f[1] == old_parent, -abs(f[2] - old_index)))[0]

    taken, seen = {n.id for n in nodes}, set()
    for n in nodes:
        if n.id in keeper:
            if n is not keeper[n.id]:
                n.id = _unique_id(n.id, taken)
            continue
        if n.id in seen:
            n.id = _unique_id(n.id, taken)
        seen.add(n.id)

    taken, seen = {layer.layer_id for layer in layers}, set()
    for layer in layers:
        if layer.layer_id in seen:
            layer.layer_id = _unique_id(layer.layer_id, taken)
        seen.add(layer.layer_id)


def _strip_svg_attrs(ast: DesignDocumentAST) -> None:
    """Strip forbidden SVG attributes from all VectorIllustrationNode layers."""
    for node, _, _ in _collect_all_nodes(ast.root):
        if not isinstance(node, VectorIllustrationNode):
            continue
        for layer in node.layers:
            layer.attributes = {
                k: v for k, v in layer.attributes.items()
                if k.lower() in _SVG_ALLOWED_ATTRS and not _SVG_BLOCKED_PATTERN.search(str(v))
            }


def _fix_contrast(ast: DesignDocumentAST, vision_meta: Optional[VisionMetadata] = None, kept: Set[tuple] = frozenset()) -> None:
    """``kept``: the original's own low-contrast colours (_preexisting_low_contrast), left as the user set them."""
    roles = vision_meta.roles if vision_meta else None
    palette = {color: lab(hex_rgb(color)) for color in (*roles.text, *roles.accents)} if roles else {}
    for node, _, backgrounds in _collect_all_nodes(ast.root):
        if not isinstance(node, TextNode):
            continue
        for i, stop, _, _ in list(_low_contrast_spans(node, backgrounds)):
            if _contrast_key(node, i, stop, backgrounds) in kept:
                continue
            span = node.spans[i]
            original = span.color if stop is None else span.gradient.stops[stop]
            backdrop = [_relative_luminance(bg) for bg in backgrounds]

            def contrast(color):
                return min(_contrast_ratio(_relative_luminance(color), lum) for lum in backdrop)

            original_lab = lab(hex_rgb(original))
            _, a, b = original_lab
            chroma = hypot(a, b)
            candidates = []
            for color, (_, ca, cb) in palette.items():
                other_chroma = hypot(ca, cb)
                similar = (other_chroma < 8 if chroma < 5 else
                           other_chroma >= 5 and a * ca + b * cb >= 0.85 * chroma * other_chroma)
                if similar and contrast(color) >= MIN_TEXT_CONTRAST:
                    candidates.append(color)
            replacement = (min(candidates, key=lambda color: dist(original_lab, palette[color])) if candidates
                           else max(("#000000", "#FFFFFF"), key=contrast))
            if stop is None:
                span.color = replacement
            else:
                span.gradient.stops[stop] = replacement
