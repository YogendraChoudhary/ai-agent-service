"""
Tests for Mode 8 self-healing loop harness (service.py) and the golden
"Ask Anything" reference UI fixture.

Uses an AsyncMock agent runner so tests run offline with no real LLM calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.validate import validate_design_ast


# ---------------------------------------------------------------------------
# Minimal golden AST fixtures
# ---------------------------------------------------------------------------

_VALID_AST_DICT = {
    "block_type": "eddyter_design_canvas",
    "version": "1.0",
    "design_id": "dsn_test_01",
    "title": "Test Card",
    "theme_palette": ["#000000", "#FFFFFF"],
    "root": {
        "node_type": "container",
        "id": "root",
        "layout": "flex_col",
        "children": [
            {
                "node_type": "text",
                "id": "hero_title",
                "tag": "h1",
                "spans": [{"text": "Hello World", "color": "#FFFFFF", "font_weight": "bold"}],
            }
        ],
    },
}


def _make_runner(outputs: list[str]) -> Any:
    """Return a mock agent runner that yields ``outputs`` in sequence."""
    call_count = 0

    async def _run_json_completion(
        messages: List[Dict[str, Any]],
        image_bytes: Optional[bytes] = None,
        temperature: float = 0.2,
    ) -> str:
        nonlocal call_count
        result = outputs[min(call_count, len(outputs) - 1)]
        call_count += 1
        return result

    runner = AsyncMock()
    runner.run_json_completion = _run_json_completion
    return runner


# ===========================================================================
# Harness — clean first-attempt generation
# ===========================================================================

class TestHarnessCleanGeneration:
    @pytest.mark.asyncio
    async def test_returns_ast_on_first_valid_attempt(self):
        from src.design_canvas.service import execute_design_canvas_harness

        runner = _make_runner([json.dumps(_VALID_AST_DICT)])
        result = await execute_design_canvas_harness(runner, "design a hero card", "create_from_prompt")
        assert isinstance(result, DesignDocumentAST)
        assert result.design_id == "dsn_test_01"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        from src.design_canvas.service import execute_design_canvas_harness

        fenced = f"```json\n{json.dumps(_VALID_AST_DICT)}\n```"
        runner = _make_runner([fenced])
        result = await execute_design_canvas_harness(runner, "design a card", "create_from_prompt")
        assert isinstance(result, DesignDocumentAST)

    @pytest.mark.asyncio
    async def test_prompt_mode_sends_no_image_on_retry(self):
        """On attempt > 0, image_bytes must be None even if provided initially."""
        from src.design_canvas.service import execute_design_canvas_harness

        call_images: list[Optional[bytes]] = []

        async def _run(messages, image_bytes=None, temperature=0.2):
            call_images.append(image_bytes)
            # First call: return invalid JSON; second call: return valid AST
            if len(call_images) == 1:
                return "{invalid}"
            return json.dumps(_VALID_AST_DICT)

        runner = AsyncMock()
        runner.run_json_completion = _run

        fake_bytes = b"FAKE_IMAGE"
        result = await execute_design_canvas_harness(
            runner, "recreate this", "create_from_vision",
            reference_image_bytes=fake_bytes,
            max_repair_attempts=2,
        )
        assert isinstance(result, DesignDocumentAST)
        # First call got the image; subsequent calls did not
        assert call_images[0] == fake_bytes
        assert call_images[1] is None


# ===========================================================================
# Harness — repair loop
# ===========================================================================

class TestHarnessRepairLoop:
    @pytest.mark.asyncio
    async def test_repairs_invalid_json_then_succeeds(self):
        from src.design_canvas.service import execute_design_canvas_harness

        outputs = ["{bad json}", json.dumps(_VALID_AST_DICT)]
        runner = _make_runner(outputs)
        result = await execute_design_canvas_harness(
            runner, "design a card", "create_from_prompt", max_repair_attempts=2
        )
        assert isinstance(result, DesignDocumentAST)

    @pytest.mark.asyncio
    async def test_raises_after_max_repairs_with_no_parseable_output(self):
        from src.design_canvas.service import execute_design_canvas_harness

        runner = _make_runner(["{invalid json}"] * 10)
        with pytest.raises(ValueError, match="DesignDocumentAST failed schema validation"):
            await execute_design_canvas_harness(
                runner, "design a card", "create_from_prompt", max_repair_attempts=2
            )

    @pytest.mark.asyncio
    async def test_auto_heals_after_max_domain_errors(self):
        """If domain errors persist, auto_heal_design_ast should be called."""
        from src.design_canvas.service import execute_design_canvas_harness

        # Build an AST with duplicate IDs (will fail domain check)
        bad_ast = json.loads(json.dumps(_VALID_AST_DICT))
        bad_ast["root"]["children"].append({
            "node_type": "text",
            "id": "hero_title",   # duplicate!
            "tag": "p",
            "spans": [{"text": "Duplicate", "color": "#FFFFFF"}],
        })

        runner = _make_runner([json.dumps(bad_ast)] * 5)
        # Should not raise — auto-healer should fix the duplicate ID
        result = await execute_design_canvas_harness(
            runner, "design a card", "create_from_prompt", max_repair_attempts=2
        )
        assert isinstance(result, DesignDocumentAST)
        assert validate_design_ast(result) == []

    @pytest.mark.asyncio
    async def test_auto_healed_result_validates_when_contrast_never_gets_fixed(self):
        from src.design_canvas.service import execute_design_canvas_harness

        bad_ast = json.loads(json.dumps(_VALID_AST_DICT))
        bad_ast["root"]["style"] = {"background_color": "#000000"}
        bad_ast["root"]["children"][0]["spans"][0]["color"] = "#111111"  # 1.1:1, and the model never fixes it

        runner = _make_runner([json.dumps(bad_ast)] * 5)
        result = await execute_design_canvas_harness(
            runner, "design a card", "create_from_prompt", max_repair_attempts=2
        )
        assert validate_design_ast(result) == []
        assert result.root.children[0].spans[0].text == "Hello World"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_edit_that_deletes_a_node_is_accepted_without_repair(self):
        """'remove the subtitle': the validator must not send the model back to restore it."""
        from src.design_canvas.service import execute_design_canvas_harness

        original = json.loads(json.dumps(_VALID_AST_DICT))
        original["root"]["children"].append(
            {"node_type": "text", "id": "subtitle", "tag": "p", "spans": [{"text": "Sub", "color": "#FFFFFF"}]})

        calls: list[int] = []

        async def _run(messages, image_bytes=None, temperature=0.2):
            calls.append(1)
            return json.dumps(_VALID_AST_DICT)  # the subtitle is gone

        runner = AsyncMock()
        runner.run_json_completion = _run
        result = await execute_design_canvas_harness(
            runner, "remove the subtitle", "edit_existing_ast",
            existing_ast_json=json.dumps(original), max_repair_attempts=2,
        )
        assert len(calls) == 1
        assert [n.id for n in result.root.children] == ["hero_title"]

    @pytest.mark.asyncio
    async def test_temperature_reduced_on_repair(self):
        """Repair attempts must use temperature=0.1, not 0.2."""
        from src.design_canvas.service import execute_design_canvas_harness

        temps: list[float] = []

        async def _run(messages, image_bytes=None, temperature=0.2):
            temps.append(temperature)
            if len(temps) == 1:
                return "{bad}"
            return json.dumps(_VALID_AST_DICT)

        runner = AsyncMock()
        runner.run_json_completion = _run
        await execute_design_canvas_harness(
            runner, "design a card", "create_from_prompt", max_repair_attempts=2
        )
        assert temps[0] == pytest.approx(0.2)
        assert temps[1] == pytest.approx(0.1)


# ===========================================================================
# Golden "Ask Anything" reference UI fixture
# ===========================================================================

class TestAskAnythingGoldenFixture:
    def test_ask_anything_reference_ui_ast_validates_cleanly(self):
        """
        Golden fixture: the Ask Anything onboarding screen from the plan spec
        must parse and validate with zero errors.
        """
        golden_payload = {
            "block_type": "eddyter_design_canvas",
            "version": "1.0",
            "design_id": "dsn_ask_anything_01",
            "title": "Ask Anything Onboarding Screen",
            "theme_palette": ["#000000", "#34F5A6", "#E2FFF3", "#228B60", "#FFFFFF", "#8E8E93"],
            "root": {
                "node_type": "container",
                "id": "root_canvas",
                "layout": "flex_col",
                "align_items": "center",
                "justify_content": "center",
                "gap_px": 28,
                "style": {
                    "background_color": "#000000",
                    "padding": {"top": 64, "right": 48, "bottom": 64, "left": 48},
                    "border_radius": 16,
                },
                "children": [
                    {
                        "node_type": "container",
                        "id": "header_section",
                        "layout": "flex_col",
                        "align_items": "center",
                        "gap_px": 12,
                        "children": [
                            {
                                "node_type": "text",
                                "id": "hero_title",
                                "tag": "h1",
                                "font_size_px": 38,
                                "spans": [
                                    {"text": "As", "color": "#8E8E93", "font_weight": "bold"},
                                    {"text": "k Anything", "color": "#FFFFFF", "font_weight": "bold"},
                                ],
                            },
                            {
                                "node_type": "text",
                                "id": "hero_subtitle",
                                "tag": "p",
                                "font_size_px": 16,
                                "style": {"max_width": 520},
                                "spans": [
                                    {
                                        "text": (
                                            "Ask your questions – short, long or anything in between. "
                                            "The more precise you ask, the better the answer."
                                        ),
                                        "color": "#D1D1D6",
                                    }
                                ],
                            },
                            {
                                "node_type": "divider",
                                "id": "header_divider",
                                "color": "#1C1C1E",
                                "thickness_px": 1,
                                "width_percent": 75,
                                "margin_y_px": 16,
                            },
                        ],
                    },
                    {
                        "node_type": "container",
                        "id": "carousel_middle_row",
                        "layout": "flex_row",
                        "align_items": "center",
                        "justify_content": "space_between",
                        "gap_px": 48,
                        "style": {"max_width": 720, "width_percent": 100},
                        "children": [
                            {
                                "node_type": "icon_button",
                                "id": "nav_prev_btn",
                                "icon_name": "arrow_left",
                                "color": "#636366",
                                "size_px": 24,
                                "aria_label": "Previous Slide",
                            },
                            {
                                "node_type": "vector_illustration",
                                "id": "dual_chat_bubbles",
                                "view_box": "0 0 200 180",
                                "width_px": 180,
                                "height_px": 160,
                                "layers": [
                                    {
                                        "layer_id": "bubble_back_right",
                                        "label": "Secondary Pale Mint Bubble",
                                        "svg_element": "path",
                                        "attributes": {
                                            "d": "M95 55 a52 52 0 1 1 -15 95 l25 15 l-5 -25 a52 52 0 0 1 -5 -85 Z"
                                        },
                                        "fill_color": "#E2FFF3",
                                    },
                                    {
                                        "layer_id": "bubble_front_left",
                                        "label": "Primary Mint Green Bubble",
                                        "svg_element": "path",
                                        "attributes": {
                                            "d": "M35 25 a52 52 0 1 1 0 92 l-25 12 l8 -24 a52 52 0 0 1 17 -80 Z"
                                        },
                                        "fill_color": "#34F5A6",
                                    },
                                    {
                                        "layer_id": "chat_line_top",
                                        "label": "Top Message Line",
                                        "svg_element": "rect",
                                        "attributes": {"x": 52, "y": 48, "width": 32, "height": 8, "rx": 4},
                                        "fill_color": "#228B60",
                                    },
                                    {
                                        "layer_id": "chat_line_mid",
                                        "label": "Middle Message Line",
                                        "svg_element": "rect",
                                        "attributes": {"x": 52, "y": 64, "width": 54, "height": 8, "rx": 4},
                                        "fill_color": "#228B60",
                                    },
                                    {
                                        "layer_id": "chat_line_bottom",
                                        "label": "Bottom Message Line",
                                        "svg_element": "rect",
                                        "attributes": {"x": 52, "y": 80, "width": 54, "height": 8, "rx": 4},
                                        "fill_color": "#228B60",
                                    },
                                ],
                            },
                            {
                                "node_type": "icon_button",
                                "id": "nav_next_btn",
                                "icon_name": "arrow_right",
                                "color": "#FFFFFF",
                                "size_px": 24,
                                "aria_label": "Next Slide",
                            },
                        ],
                    },
                    {
                        "node_type": "container",
                        "id": "bottom_copy_section",
                        "layout": "flex_col",
                        "align_items": "center",
                        "gap_px": 14,
                        "style": {"max_width": 560},
                        "children": [
                            {
                                "node_type": "text",
                                "id": "feature_heading",
                                "tag": "h2",
                                "font_size_px": 28,
                                "spans": [
                                    {
                                        "text": "Open-ended questions",
                                        "color": "#FFFFFF",
                                        "font_weight": "bold",
                                    }
                                ],
                            },
                            {
                                "node_type": "text",
                                "id": "feature_body",
                                "tag": "p",
                                "font_size_px": 15,
                                "line_height": 1.6,
                                "spans": [
                                    {
                                        "text": (
                                            "Nova is here for you 24/7. This AI Assistant can generate humanlike "
                                            "responses, making you feel like you are chatting with an old friend. "
                                            "It can even recommend a book to read or a movie to watch!"
                                        ),
                                        "color": "#CCCCCC",
                                    }
                                ],
                            },
                        ],
                    },
                ],
            },
        }

        ast_obj = DesignDocumentAST.model_validate(golden_payload)
        validation_errors = validate_design_ast(ast_obj)
        assert validation_errors == [], f"Golden fixture has errors: {validation_errors}"

    def test_ask_anything_all_node_types_exposed(self):
        """Every text span, navigation arrow, and speech bubble layer is addressable."""
        golden = DesignDocumentAST.model_validate({
            "block_type": "eddyter_design_canvas",
            "version": "1.0",
            "design_id": "dsn_check",
            "title": "Node coverage check",
            "theme_palette": ["#000000", "#FFFFFF"],
            "root": {
                "node_type": "container",
                "id": "root",
                "layout": "flex_col",
                "children": [
                    {"node_type": "text", "id": "t1", "tag": "h1",
                     "spans": [{"text": "X", "color": "#FFF"}]},
                    {"node_type": "icon_button", "id": "ib1", "icon_name": "arrow_left"},
                    {"node_type": "icon_button", "id": "ib2", "icon_name": "arrow_right"},
                    {"node_type": "divider", "id": "dv1"},
                    {
                        "node_type": "vector_illustration",
                        "id": "vi1",
                        "view_box": "0 0 100 100",
                        "layers": [
                            {"layer_id": "layer_a", "label": "A", "svg_element": "circle",
                             "attributes": {"cx": 50, "cy": 50, "r": 40}, "fill_color": "#34F5A6"},
                            {"layer_id": "layer_b", "label": "B", "svg_element": "rect",
                             "attributes": {"x": 10, "y": 10, "width": 30, "height": 30}, "fill_color": "#E2FFF3"},
                        ],
                    },
                ],
            },
        })
        assert golden.root.id == "root"
        # All layers addressable
        vi = golden.root.children[4]  # type: ignore[index]
        layer_ids = {layer.layer_id for layer in vi.layers}  # type: ignore[union-attr]
        assert "layer_a" in layer_ids
        assert "layer_b" in layer_ids
