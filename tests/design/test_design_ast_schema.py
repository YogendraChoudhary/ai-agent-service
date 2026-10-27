"""
Tests for Mode 8 Pydantic AST schema — serialisation, discriminator routing,
and field-level validation constraints.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.design_canvas.schema import (
    ButtonNode,
    ContainerNode,
    DesignDocumentAST,
    DividerNode,
    IconButtonNode,
    NodeStyle,
    SpacingBox,
    SVGLayer,
    TextNode,
    TextSpan,
    VectorIllustrationNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _text_node(nid: str = "txt_01", tag: str = "p", text: str = "Hello") -> dict:
    return {
        "node_type": "text",
        "id": nid,
        "tag": tag,
        "spans": [{"text": text, "color": "#FFFFFF"}],
    }


def _container(nid: str = "root", children: list | None = None) -> dict:
    return {
        "node_type": "container",
        "id": nid,
        "layout": "flex_col",
        "children": children or [_text_node()],
    }


def _minimal_ast(root: dict | None = None) -> dict:
    return {
        "block_type": "eddyter_design_canvas",
        "version": "1.0",
        "design_id": "dsn_test_01",
        "title": "Test Card",
        "theme_palette": ["#000000", "#FFFFFF"],
        "root": root or _container(),
    }


# ---------------------------------------------------------------------------
# DesignDocumentAST round-trip
# ---------------------------------------------------------------------------

class TestDesignDocumentAST:
    def test_minimal_valid_ast_parses(self):
        ast = DesignDocumentAST.model_validate(_minimal_ast())
        assert ast.design_id == "dsn_test_01"
        assert ast.version == "1.0"
        assert ast.block_type == "eddyter_design_canvas"

    def test_json_roundtrip(self):
        raw = _minimal_ast()
        ast = DesignDocumentAST.model_validate(raw)
        serialised = ast.model_dump_json()
        reparsed = DesignDocumentAST.model_validate_json(serialised)
        assert reparsed.design_id == ast.design_id

    def test_theme_palette_min_length(self):
        raw = _minimal_ast()
        raw["theme_palette"] = ["#000000"]
        with pytest.raises(ValidationError):
            DesignDocumentAST.model_validate(raw)

    def test_theme_palette_max_length(self):
        raw = _minimal_ast()
        raw["theme_palette"] = [f"#0000{i:02X}" for i in range(11)]
        with pytest.raises(ValidationError):
            DesignDocumentAST.model_validate(raw)

    def test_wrong_block_type_rejected(self):
        raw = _minimal_ast()
        raw["block_type"] = "something_else"
        with pytest.raises(ValidationError):
            DesignDocumentAST.model_validate(raw)

    def test_wrong_version_rejected(self):
        raw = _minimal_ast()
        raw["version"] = "2.0"
        with pytest.raises(ValidationError):
            DesignDocumentAST.model_validate(raw)


# ---------------------------------------------------------------------------
# TextNode
# ---------------------------------------------------------------------------

class TestTextNode:
    def test_valid_text_node(self):
        node = TextNode.model_validate(_text_node())
        assert node.id == "txt_01"
        assert node.spans[0].text == "Hello"

    def test_empty_spans_rejected(self):
        raw = _text_node()
        raw["spans"] = []
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)

    def test_font_size_bounds(self):
        raw = _text_node()
        raw["font_size_px"] = 9
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)
        raw["font_size_px"] = 97
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)

    def test_invalid_id_pattern(self):
        raw = _text_node(nid="UPPER_INVALID")
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)

    def test_id_too_short(self):
        raw = _text_node(nid="a")
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)

    def test_invalid_color_pattern(self):
        raw = _text_node()
        raw["spans"] = [{"text": "Hi", "color": "red"}]
        with pytest.raises(ValidationError):
            TextNode.model_validate(raw)

    def test_valid_hex8_color(self):
        node = TextNode.model_validate({**_text_node(), "spans": [{"text": "Hi", "color": "#FFFFFF80"}]})
        assert node.spans[0].color == "#FFFFFF80"


# ---------------------------------------------------------------------------
# TextSpan
# ---------------------------------------------------------------------------

class TestTextSpan:
    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TextSpan.model_validate({"text": "", "color": "#FFFFFF"})

    def test_all_font_weights(self):
        for fw in ("normal", "medium", "semibold", "bold"):
            span = TextSpan.model_validate({"text": "x", "color": "#000000", "font_weight": fw})
            assert span.font_weight == fw

    def test_invalid_font_weight(self):
        with pytest.raises(ValidationError):
            TextSpan.model_validate({"text": "x", "color": "#000000", "font_weight": "extra-bold"})


# ---------------------------------------------------------------------------
# ContainerNode
# ---------------------------------------------------------------------------

class TestContainerNode:
    def test_empty_children_rejected(self):
        raw = _container()
        raw["children"] = []
        with pytest.raises(ValidationError):
            ContainerNode.model_validate(raw)

    def test_max_children(self):
        raw = _container(children=[_text_node(f"t{i:02d}") for i in range(26)])
        with pytest.raises(ValidationError):
            ContainerNode.model_validate(raw)

    def test_nested_container(self):
        inner = _container("inner", [_text_node("inner_txt")])
        outer = _container("outer", [inner])
        node = ContainerNode.model_validate(outer)
        assert node.children[0].id == "inner"  # type: ignore[union-attr]

    def test_discriminator_routing_in_children(self):
        raw = _container(children=[
            _text_node("t1"),
            {"node_type": "divider", "id": "div1"},
        ])
        node = ContainerNode.model_validate(raw)
        assert node.children[1].node_type == "divider"


# ---------------------------------------------------------------------------
# DividerNode
# ---------------------------------------------------------------------------

class TestDividerNode:
    def test_valid_divider(self):
        node = DividerNode.model_validate({"id": "dv01", "node_type": "divider"})
        assert node.thickness_px == 1
        assert node.width_percent == 75

    def test_thickness_bounds(self):
        with pytest.raises(ValidationError):
            DividerNode.model_validate({"id": "dv01", "node_type": "divider", "thickness_px": 9})

    def test_width_percent_bounds(self):
        with pytest.raises(ValidationError):
            DividerNode.model_validate({"id": "dv01", "node_type": "divider", "width_percent": 9})


# ---------------------------------------------------------------------------
# VectorIllustrationNode
# ---------------------------------------------------------------------------

class TestVectorIllustrationNode:
    def _layer(self, lid: str = "l01") -> dict:
        return {
            "layer_id": lid,
            "label": "Test Layer",
            "svg_element": "rect",
            "attributes": {"x": 0, "y": 0, "width": 100, "height": 50},
            "fill_color": "#34F5A6",
        }

    def test_valid_vector_node(self):
        node = VectorIllustrationNode.model_validate({
            "node_type": "vector_illustration",
            "id": "vec01",
            "view_box": "0 0 200 180",
            "layers": [self._layer()],
        })
        assert node.id == "vec01"
        assert len(node.layers) == 1

    def test_empty_layers_rejected(self):
        with pytest.raises(ValidationError):
            VectorIllustrationNode.model_validate({
                "node_type": "vector_illustration",
                "id": "vec01",
                "view_box": "0 0 200 180",
                "layers": [],
            })

    def test_invalid_view_box_format(self):
        with pytest.raises(ValidationError):
            VectorIllustrationNode.model_validate({
                "node_type": "vector_illustration",
                "id": "vec01",
                "view_box": "0 0 abc 180",
                "layers": [self._layer()],
            })

    def test_layer_id_pattern(self):
        layer = {**self._layer(), "layer_id": "INVALID"}
        with pytest.raises(ValidationError):
            SVGLayer.model_validate(layer)

    def test_svg_element_enum(self):
        layer = {**self._layer(), "svg_element": "polygon"}
        with pytest.raises(ValidationError):
            SVGLayer.model_validate(layer)

    def test_opacity_bounds(self):
        layer = {**self._layer(), "opacity": 1.5}
        with pytest.raises(ValidationError):
            SVGLayer.model_validate(layer)


# ---------------------------------------------------------------------------
# IconButtonNode
# ---------------------------------------------------------------------------

class TestIconButtonNode:
    def test_valid_icon_button(self):
        node = IconButtonNode.model_validate({
            "node_type": "icon_button",
            "id": "btn01",
            "icon_name": "arrow_right",
        })
        assert node.icon_name == "arrow_right"

    def test_invalid_icon_name(self):
        with pytest.raises(ValidationError):
            IconButtonNode.model_validate({
                "node_type": "icon_button",
                "id": "btn01",
                "icon_name": "star",
            })

    def test_size_bounds(self):
        with pytest.raises(ValidationError):
            IconButtonNode.model_validate({
                "node_type": "icon_button",
                "id": "btn01",
                "icon_name": "send",
                "size_px": 11,
            })


# ---------------------------------------------------------------------------
# ButtonNode
# ---------------------------------------------------------------------------

class TestButtonNode:
    def test_valid_button(self):
        node = ButtonNode.model_validate({
            "node_type": "button",
            "id": "cta01",
            "label": "Get Started",
        })
        assert node.label == "Get Started"
        assert node.font_weight == "semibold"

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            ButtonNode.model_validate({
                "node_type": "button",
                "id": "cta01",
                "label": "",
            })


# ---------------------------------------------------------------------------
# NodeStyle / SpacingBox
# ---------------------------------------------------------------------------

class TestNodeStyle:
    def test_valid_style(self):
        style = NodeStyle.model_validate({
            "background_color": "#000000",
            "border_radius": 16,
            "padding": {"top": 24, "right": 24, "bottom": 24, "left": 24},
        })
        assert style.border_radius == 16

    def test_invalid_background_color(self):
        with pytest.raises(ValidationError):
            NodeStyle.model_validate({"background_color": "black"})

    def test_max_width_bounds(self):
        with pytest.raises(ValidationError):
            NodeStyle.model_validate({"max_width": 30})
        with pytest.raises(ValidationError):
            NodeStyle.model_validate({"max_width": 1441})

    def test_spacing_box_bounds(self):
        with pytest.raises(ValidationError):
            SpacingBox.model_validate({"top": 257})


# ---------------------------------------------------------------------------
# Discriminator correctness
# ---------------------------------------------------------------------------

class TestDiscriminator:
    def test_all_node_types_discriminate(self):
        nodes = [
            {"node_type": "text", "id": "t01", "tag": "p", "spans": [{"text": "Hi", "color": "#FFF"}]},
            {"node_type": "divider", "id": "d01"},
            {"node_type": "vector_illustration", "id": "v01", "view_box": "0 0 100 100", "layers": [
                {"layer_id": "l01", "label": "L", "svg_element": "circle",
                 "attributes": {"cx": 50, "cy": 50, "r": 40}, "fill_color": "#FF0000"}
            ]},
            {"node_type": "icon_button", "id": "ib01", "icon_name": "check"},
            {"node_type": "button", "id": "bt01", "label": "OK"},
        ]
        raw = _container(children=nodes)
        node = ContainerNode.model_validate(raw)
        types = [c.node_type for c in node.children]
        assert types == ["text", "divider", "vector_illustration", "icon_button", "button"]
