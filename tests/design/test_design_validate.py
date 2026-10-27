"""
Tests for Mode 8 deterministic validator and auto-healer (validate.py).

Covers:
  - WCAG contrast math (relative luminance + contrast ratio)
  - SVG XSS attribute blocking (event handlers, hrefs, script: URIs)
  - SVG geometry attribute whitelist pass-through
  - Duplicate node ID detection
  - Duplicate SVG layer ID detection
  - Maximum tree depth check
  - Sibling ID preservation in edit mode (renames flagged; deletions and replacements allowed)
  - auto_heal_design_ast: every rule healed, output always validates
"""

from __future__ import annotations

import json

import pytest

from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.validate import (
    _contrast_ratio,
    _relative_luminance,
    auto_heal_design_ast,
    validate_design_ast,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal ASTs quickly
# ---------------------------------------------------------------------------

def _text(nid: str, color: str = "#FFFFFF", bg: str | None = None, text: str = "Sample") -> dict:
    node: dict = {
        "node_type": "text",
        "id": nid,
        "tag": "p",
        "spans": [{"text": text, "color": color}],
        "style": {},
    }
    if bg:
        node["style"]["background_color"] = bg
    return node


def _vector(nid: str, layers: list) -> dict:
    return {
        "node_type": "vector_illustration",
        "id": nid,
        "view_box": "0 0 100 100",
        "layers": layers,
    }


def _layer(lid: str, attrs: dict | None = None) -> dict:
    return {
        "layer_id": lid,
        "label": "Test",
        "svg_element": "rect",
        "attributes": attrs or {"x": 0, "y": 0, "width": 50, "height": 50},
        "fill_color": "#34F5A6",
    }


def _ast_from_root(root: dict) -> DesignDocumentAST:
    return DesignDocumentAST.model_validate({
        "block_type": "eddyter_design_canvas",
        "version": "1.0",
        "design_id": "dsn_test",
        "title": "Test",
        "theme_palette": ["#000000", "#FFFFFF"],
        "root": root,
    })


def _container(nid: str, children: list, bg: str | None = None) -> dict:
    node: dict = {
        "node_type": "container",
        "id": nid,
        "layout": "flex_col",
        "children": children,
        "style": {},
    }
    if bg:
        node["style"]["background_color"] = bg
    return node


# ===========================================================================
# WCAG luminance math
# ===========================================================================

class TestRelativeLuminance:
    def test_black_is_zero(self):
        lum = _relative_luminance("#000000")
        assert lum is not None
        assert abs(lum - 0.0) < 1e-9

    def test_white_is_one(self):
        lum = _relative_luminance("#FFFFFF")
        assert lum is not None
        assert abs(lum - 1.0) < 1e-4

    def test_invalid_hex_returns_none(self):
        assert _relative_luminance("red") is None
        assert _relative_luminance("#GGGGGG") is None

    def test_mid_grey(self):
        # #808080 ≈ 0.215
        lum = _relative_luminance("#808080")
        assert lum is not None
        assert 0.20 < lum < 0.23

    def test_hex8_alpha_ignored(self):
        # 8-char hex (with alpha) should parse the R, G, B channels only
        lum8 = _relative_luminance("#FFFFFFFF")
        lum6 = _relative_luminance("#FFFFFF")
        assert lum8 is not None and lum6 is not None
        assert abs(lum8 - lum6) < 1e-4


class TestContrastRatio:
    def test_black_on_white(self):
        ratio = _contrast_ratio(0.0, 1.0)
        assert abs(ratio - 21.0) < 0.01

    def test_white_on_white(self):
        ratio = _contrast_ratio(1.0, 1.0)
        assert abs(ratio - 1.0) < 0.01

    def test_symmetric(self):
        r1 = _contrast_ratio(0.0, 1.0)
        r2 = _contrast_ratio(1.0, 0.0)
        assert abs(r1 - r2) < 1e-9


# ===========================================================================
# Unique ID checks
# ===========================================================================

class TestUniqueIDs:
    def test_no_errors_on_unique_ids(self):
        ast = _ast_from_root(_container("root", [
            _text("txt01"), _text("txt02"),
        ]))
        assert validate_design_ast(ast) == []

    def test_duplicate_node_id_detected(self):
        ast = _ast_from_root(_container("root", [
            _text("dup"), _text("dup"),
        ]))
        errs = validate_design_ast(ast)
        assert any("DUPLICATE_NODE_ID" in e and "dup" in e for e in errs)

    def test_duplicate_nested_id_detected(self):
        inner = _container("inner", [_text("shared_id")])
        outer = _container("outer", [inner, _text("shared_id")])
        ast = _ast_from_root(outer)
        errs = validate_design_ast(ast)
        assert any("DUPLICATE_NODE_ID" in e and "shared_id" in e for e in errs)


# ===========================================================================
# SVG attribute whitelist and XSS blocking
# ===========================================================================

class TestSVGSanitizer:
    def test_geometry_attrs_pass(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"x": 0, "y": 0, "width": 50, "height": 50})])
        ]))
        errs = validate_design_ast(ast)
        assert not any("SVG" in e for e in errs)

    def test_onload_event_handler_blocked(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"x": 0, "onload": "alert(1)"})])
        ]))
        errs = validate_design_ast(ast)
        assert any("SVG_FORBIDDEN_ATTR" in e or "SVG_XSS" in e for e in errs)

    def test_href_attr_blocked(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"href": "javascript:void(0)"})])
        ]))
        errs = validate_design_ast(ast)
        assert any("SVG_FORBIDDEN_ATTR" in e or "SVG_XSS" in e for e in errs)

    def test_javascript_uri_in_value_blocked(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"x": "javascript:alert(1)"})])
        ]))
        errs = validate_design_ast(ast)
        assert any("SVG_XSS" in e for e in errs)

    def test_data_uri_in_value_blocked(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"x": "data:text/html,<h1>XSS</h1>"})])
        ]))
        errs = validate_design_ast(ast)
        assert any("SVG_XSS" in e for e in errs)

    def test_duplicate_layer_id_detected(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("dup"), _layer("dup")])
        ]))
        errs = validate_design_ast(ast)
        assert any("DUPLICATE_LAYER_ID" in e and "dup" in e for e in errs)

    def test_transform_attr_allowed(self):
        ast = _ast_from_root(_container("root", [
            _vector("vec01", [_layer("l01", {"transform": "translate(10,20)"})])
        ]))
        errs = validate_design_ast(ast)
        assert not any("SVG_FORBIDDEN_ATTR" in e for e in errs)


# ===========================================================================
# WCAG contrast
# ===========================================================================

class TestWCAGContrast:
    def test_white_on_black_passes(self):
        # white (#FFFFFF) on black (#000000) → ratio 21:1
        ast = _ast_from_root(_container("root", [
            _text("txt01", color="#FFFFFF", bg=None),
        ], bg="#000000"))
        errs = validate_design_ast(ast)
        assert not any("WCAG" in e for e in errs)

    def test_black_on_black_fails(self):
        # black text on black background → ratio 1:1
        ast = _ast_from_root(_container("root", [
            _text("txt01", color="#000000", bg=None),
        ], bg="#000000"))
        errs = validate_design_ast(ast)
        assert any("WCAG_LOW_CONTRAST" in e for e in errs)

    def test_no_background_skips_check(self):
        # No ancestor background → check skipped (no false positives)
        ast = _ast_from_root(_container("root", [
            _text("txt01", color="#333333"),
        ]))
        errs = validate_design_ast(ast)
        assert not any("WCAG" in e for e in errs)

    def test_node_own_background_used(self):
        # Text node sets its own background, overriding ancestor
        node = _text("txt01", color="#000000")
        node["style"]["background_color"] = "#000000"
        ast = _ast_from_root(_container("root", [node], bg="#FFFFFF"))
        errs = validate_design_ast(ast)
        assert any("WCAG_LOW_CONTRAST" in e for e in errs)


# ===========================================================================
# Max depth check
# ===========================================================================

class TestMaxDepth:
    def _deep_nest(self, depth: int) -> dict:
        if depth == 0:
            return _text(f"leaf_{depth}")
        return _container(f"c_{depth}", [self._deep_nest(depth - 1)])

    def test_depth_6_passes(self):
        ast = _ast_from_root(self._deep_nest(5))  # root + 5 levels = depth 5
        errs = validate_design_ast(ast)
        assert not any("MAX_DEPTH" in e for e in errs)

    def test_depth_7_fails(self):
        ast = _ast_from_root(self._deep_nest(7))
        errs = validate_design_ast(ast)
        assert any("MAX_DEPTH_EXCEEDED" in e for e in errs)


# ===========================================================================
# Sibling ID preservation (edit mode)
# ===========================================================================

class TestSiblingPreservation:
    """A node that is unchanged apart from its id must keep its id; deleting or replacing nodes is an edit."""

    def _errs(self, orig_root: dict, new_root: dict) -> list[str]:
        orig_ast = _ast_from_root(orig_root)
        return validate_design_ast(_ast_from_root(new_root), existing_ast_json=orig_ast.model_dump_json())

    def test_unchanged_ids_pass(self):
        root = _container("root", [_text("t1", text="A"), _text("t2", text="B")])
        assert self._errs(root, root) == []

    def test_deleting_a_node_is_a_legitimate_edit(self):
        # "remove the third line": the repair loop must not tell the model to put it back
        errs = self._errs(
            _container("root", [_text("t1", text="A"), _text("t2", text="B"), _text("t3", text="C")]),
            _container("root", [_text("t1", text="A"), _text("t2", text="B")]),
        )
        assert errs == []

    def test_replacing_a_node_is_a_legitimate_edit(self):
        # "swap the divider for a button"
        errs = self._errs(
            _container("root", [_text("t1", text="A"), {"node_type": "divider", "id": "div1"}]),
            _container("root", [_text("t1", text="A"), {"node_type": "button", "id": "cta1", "label": "Go"}]),
        )
        assert errs == []

    def test_editing_a_node_in_place_passes(self):
        errs = self._errs(
            _container("root", [_text("t1", text="Hello")]),
            _container("root", [_text("t1", text="Goodbye")]),
        )
        assert errs == []

    def test_unchanged_node_with_new_id_is_flagged(self):
        errs = self._errs(
            _container("root", [_text("t1", text="A"), _text("t2", text="B")]),
            _container("root", [_text("t1", text="A"), _text("subtitle_new", text="B")]),
        )
        assert len(errs) == 1
        assert "RENAMED_NODE_ID" in errs[0] and "`subtitle_new`" in errs[0] and "`t2`" in errs[0]

    def test_regenerated_ids_flag_every_unchanged_sibling(self):
        errs = self._errs(
            _container("root", [_text("title", text="A"), _text("body", text="B"), {"node_type": "divider", "id": "rule"}]),
            _container("root", [_text("n1", text="A"), _text("n2", text="B"), {"node_type": "divider", "id": "n3"}]),
        )
        assert sorted(e.split("`")[1] for e in errs) == ["n1", "n2", "n3"]

    def test_corrupt_existing_json_skips_check(self):
        new_ast = _ast_from_root(_container("root", [_text("t1")]))
        errs = validate_design_ast(new_ast, existing_ast_json="{invalid json}")
        assert errs == []


# ===========================================================================
# auto_heal_design_ast
# ===========================================================================

class TestAutoHeal:
    def test_duplicate_ids_deduplicated(self):
        """Duplicate IDs should be renamed by the healer."""
        raw = {
            "block_type": "eddyter_design_canvas",
            "version": "1.0",
            "design_id": "dsn_heal",
            "title": "Healer Test",
            "theme_palette": ["#000000", "#FFFFFF"],
            "root": {
                "node_type": "container",
                "id": "root",
                "layout": "flex_col",
                "children": [
                    {"node_type": "text", "id": "dup", "tag": "p", "spans": [{"text": "A", "color": "#FFF"}]},
                    {"node_type": "text", "id": "dup", "tag": "p", "spans": [{"text": "B", "color": "#FFF"}]},
                ],
            },
        }
        ast = DesignDocumentAST.model_validate(raw)
        healed = auto_heal_design_ast(ast)
        # After healing, all IDs must be unique
        errs = validate_design_ast(healed)
        assert not any("DUPLICATE_NODE_ID" in e for e in errs)

    def test_forbidden_svg_attrs_stripped(self):
        raw = {
            "block_type": "eddyter_design_canvas",
            "version": "1.0",
            "design_id": "dsn_heal",
            "title": "SVG Heal Test",
            "theme_palette": ["#000000", "#FFFFFF"],
            "root": {
                "node_type": "container",
                "id": "root",
                "layout": "flex_col",
                "children": [
                    {
                        "node_type": "vector_illustration",
                        "id": "vec01",
                        "view_box": "0 0 100 100",
                        "layers": [{
                            "layer_id": "l01",
                            "label": "L",
                            "svg_element": "rect",
                            "attributes": {"x": 0, "onload": "alert(1)"},
                            "fill_color": "#FF0000",
                        }],
                    }
                ],
            },
        }
        ast = DesignDocumentAST.model_validate(raw)
        healed = auto_heal_design_ast(ast)
        # The onload attribute should be gone
        vec = healed.root.children[0]  # type: ignore[index]
        assert "onload" not in vec.layers[0].attributes  # type: ignore[union-attr]

    def test_low_contrast_recoloured(self):
        ast = _ast_from_root(_container("root", [_text("t1", color="#111111")], bg="#000000"))
        healed = auto_heal_design_ast(ast)
        assert validate_design_ast(healed) == []
        assert healed.root.children[0].spans[0].color == "#FFFFFF"  # type: ignore[union-attr]
        assert healed.root.children[0].spans[0].text == "Sample"  # type: ignore[union-attr]

    @pytest.mark.parametrize("bg", ["#000000", "#777777", "#767676", "#7F7F7F", "#FFFFFF", "#34F5A6", "#1C1C1E"])
    def test_recoloured_span_reaches_aa_on_any_background(self, bg):
        ast = _ast_from_root(_container("root", [_text("t1", color=bg)], bg=bg))  # 1:1 contrast
        span = auto_heal_design_ast(ast).root.children[0].spans[0]  # type: ignore[union-attr]
        assert _contrast_ratio(_relative_luminance(span.color), _relative_luminance(bg)) >= 4.5

    def test_too_deep_tree_flattened(self):
        root = _text("leaf", text="Deepest")
        for depth in range(9, 0, -1):
            root = _container(f"c_{depth}", [root])
        ast = _ast_from_root(_container("root", [root]))
        assert any("MAX_DEPTH_EXCEEDED" in e for e in validate_design_ast(ast))

        healed = auto_heal_design_ast(ast)
        assert validate_design_ast(healed) == []
        texts = [n.spans[0].text for n in _all_nodes(healed.root) if n.node_type == "text"]
        assert texts == ["Deepest"]

    def test_max_length_duplicate_ids_stay_within_limit(self):
        long_id = "a" * 40
        ast = _ast_from_root(_container("root", [{"node_type": "divider", "id": long_id},
                                                 {"node_type": "divider", "id": long_id}]))
        healed = auto_heal_design_ast(ast)  # used to raise ValidationError (id over 40 chars)
        ids = [n.id for n in healed.root.children]
        assert len(set(ids)) == 2 and all(len(i) <= 40 for i in ids)
        assert validate_design_ast(healed) == []

    def test_dedupe_does_not_collide_with_an_existing_suffixed_id(self):
        ast = _ast_from_root(_container("root", [_text("dup"), _text("dup"), _text("dup_2")]))
        healed = auto_heal_design_ast(ast)
        assert validate_design_ast(healed) == []

    def test_duplicate_layer_ids_deduplicated(self):
        ast = _ast_from_root(_container("root", [
            _vector("v1", [_layer("shape")]), _vector("v2", [_layer("shape")]),
        ]))
        healed = auto_heal_design_ast(ast)
        assert validate_design_ast(healed) == []

    def test_renamed_ids_restored_in_edit_mode(self):
        orig = _ast_from_root(_container("root", [_text("t1", text="A"), _text("t2", text="B")]))
        edited = _ast_from_root(_container("root", [_text("t1", text="A changed"), _text("t9", text="B")]))
        healed = auto_heal_design_ast(edited, existing_ast_json=orig.model_dump_json())
        assert [n.id for n in healed.root.children] == ["t1", "t2"]
        assert healed.root.children[0].spans[0].text == "A changed"  # type: ignore[union-attr]

    def test_output_always_validates_with_every_problem_at_once(self):
        orig = _ast_from_root(_container("root", [_text("keep", text="Original")], bg="#000000"))
        deep = _text("deep_leaf", color="#FFFFFF")
        for depth in range(8, 0, -1):
            deep = _container(f"d_{depth}", [deep])
        broken = _ast_from_root(_container("root", [
            _text("renamed", text="Original"),                                   # rename of `keep`
            _text("dup", color="#0A0A0A"), _text("dup", color="#FFFFFF"),        # low contrast + duplicate id
            _vector("vec", [_layer("l1", {"x": 0, "onclick": "alert(1)"}), _layer("l1")]),  # XSS + duplicate layer
            deep,                                                                # depth 9
        ], bg="#000000"))
        existing = orig.model_dump_json()
        assert validate_design_ast(broken, existing_ast_json=existing)

        healed = auto_heal_design_ast(broken, existing_ast_json=existing)
        assert validate_design_ast(healed, existing_ast_json=existing) == []


def _all_nodes(node):
    yield node
    for child in getattr(node, "children", []):
        yield from _all_nodes(child)
