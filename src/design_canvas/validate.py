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
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import ValidationError

from src.design_canvas.schema import (
    ContainerNode,
    DesignDocumentAST,
    DesignNode,
    TextNode,
    VectorIllustrationNode,
)

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

def _linearize(c: int) -> float:
    s = c / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> Optional[float]:
    """Return WCAG 2.1 relative luminance [0,1] for a #RRGGBB hex string, or None on parse error."""
    h = hex_color.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h[:3])
    elif len(h) not in (6, 8):
        return None
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    except ValueError:
        return None
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _contrast_ratio(lum_text: float, lum_bg: float) -> float:
    lighter = max(lum_text, lum_bg)
    darker = min(lum_text, lum_bg)
    return (lighter + 0.05) / (darker + 0.05)


def _low_contrast_spans(node: TextNode, ancestor_bg: Optional[str]) -> Iterator[Tuple[int, float, str]]:
    """(span index, ratio, background) for each span below MIN_TEXT_CONTRAST. No known background: none."""
    bg_hex = node.style.background_color if node.style and node.style.background_color else ancestor_bg
    lum_bg = _relative_luminance(bg_hex) if bg_hex else None
    if lum_bg is None:
        return
    for i, span in enumerate(node.spans):
        lum_text = _relative_luminance(span.color)
        if lum_text is None:
            continue
        ratio = _contrast_ratio(lum_text, lum_bg)
        if ratio < MIN_TEXT_CONTRAST:
            yield i, ratio, bg_hex


# ---------------------------------------------------------------------------
# Tree walker helpers
# ---------------------------------------------------------------------------

def _collect_all_nodes(
    node: DesignNode,
    depth: int = 0,
    bg_stack: Optional[List[Optional[str]]] = None,
) -> List[tuple[DesignNode, int, Optional[str]]]:
    """
    DFS walk over the AST tree.

    Returns a flat list of (node, depth, nearest_ancestor_bg_color) tuples.
    ``nearest_ancestor_bg_color`` is the most recent non-None background_color
    found in any ancestor ContainerNode.
    """
    if bg_stack is None:
        bg_stack = [None]

    results: List[tuple[DesignNode, int, Optional[str]]] = [(node, depth, bg_stack[-1])]

    if isinstance(node, ContainerNode):
        own_bg = (node.style.background_color if node.style else None)
        next_bg = own_bg if own_bg else bg_stack[-1]
        bg_stack.append(next_bg)
        for child in node.children:
            results.extend(_collect_all_nodes(child, depth + 1, bg_stack))
        bg_stack.pop()

    return results


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

    all_nodes = _collect_all_nodes(ast.root)

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
    # Check 3: WCAG 2.1 contrast — text spans vs nearest ancestor background
    # ------------------------------------------------------------------
    for node, _, ancestor_bg in all_nodes:
        if not isinstance(node, TextNode):
            continue
        for i, ratio, bg_hex in _low_contrast_spans(node, ancestor_bg):
            errors.append(
                f"WCAG_LOW_CONTRAST: node `{node.id}` span \"{node.spans[i].text[:20]}\" "
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


def _node_signature(node: DesignNode) -> str:
    """Everything about a node except its id and its children, for spotting a node that only got a new id."""
    return json.dumps(node.model_dump(exclude={"id", "children"}), sort_keys=True)


def _renamed_nodes(ast: DesignDocumentAST, existing_ast_json: str) -> List[Tuple[DesignNode, str]]:
    """
    (node, original id) for each node of ``ast`` that has an id the original AST did not have but is otherwise
    identical to an original node whose id is gone. Paired one to one, in tree order. An original that does
    not parse cannot be compared, so it yields nothing.
    """
    try:
        original = DesignDocumentAST.model_validate_json(existing_ast_json)
    except (ValidationError, ValueError, TypeError):
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
      shortened to stay within the 40-character id limit.
    * Strip forbidden SVG attributes.
    * Recolour each text span below MIN_TEXT_CONTRAST to black or white,
      whichever reads better on its background (always ≥ 4.5:1).

    Never changes text content or layout properties.

    Returns a new (re-validated) DesignDocumentAST instance.
    """
    healed = ast.model_copy(deep=True)
    _flatten_too_deep(healed.root, 0)
    if existing_ast_json:
        for node, original_id in _renamed_nodes(healed, existing_ast_json):
            node.id = original_id
    _dedupe_ids(healed)
    _strip_svg_attrs(healed)
    _fix_contrast(healed)
    # Re-parse so Pydantic re-validates every mutation.
    return DesignDocumentAST.model_validate(healed.model_dump())


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
                children.extend(_leaves(child))
                continue
            _flatten_too_deep(child, depth + 1)
        children.append(child)
    container.children = children[:MAX_CHILDREN]


def _unique_id(base: str, taken: Set[str]) -> str:
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = base[: MAX_ID_LENGTH - len(suffix)] + suffix
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        n += 1


def _dedupe_ids(ast: DesignDocumentAST) -> None:
    """The first node (layer) with an id keeps it; later ones get a fresh id that no node (layer) uses."""
    nodes = [n for n, _, _ in _collect_all_nodes(ast.root)]
    layers = [layer for n in nodes if isinstance(n, VectorIllustrationNode) for layer in n.layers]

    taken, seen = {n.id for n in nodes}, set()
    for n in nodes:
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


def _fix_contrast(ast: DesignDocumentAST) -> None:
    """Black or white, whichever contrasts more with the background: never below 4.5:1 for any background."""
    for node, _, ancestor_bg in _collect_all_nodes(ast.root):
        if not isinstance(node, TextNode):
            continue
        for i, _, bg_hex in list(_low_contrast_spans(node, ancestor_bg)):
            lum_bg = _relative_luminance(bg_hex) or 0.0
            node.spans[i].color = "#000000" if _contrast_ratio(0.0, lum_bg) >= _contrast_ratio(1.0, lum_bg) else "#FFFFFF"
