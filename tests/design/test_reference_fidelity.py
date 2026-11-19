"""Regression checks for the reported promo and the 1.1 wire contract."""

import copy
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from config import settings
from scripts.design_reference_render import render_html
from src.design_canvas.colors import delta_e
from src.design_canvas.prompts import DESIGN_SYSTEM_PROMPT, DESIGN_SYSTEM_PROMPT_V11, REFERENCE_SYSTEM_PROMPT
from src.design_canvas.reference import ReferenceSpec, check_fidelity, heal_reference_details, tracking_range
from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.schema_v11 import DesignDocumentASTV11, downgrade_to_v10
from src.design_canvas.service import execute_design_canvas_harness
from src.design_canvas.validate import auto_heal_design_ast, validate_design_ast
from src.design_canvas.vision_extractor import PaletteRoles, VisionMetadata, extract_vision_metadata
from src.text_processing_agent import _design_reference_image

FIXTURES = Path(__file__).parent / "fixtures/reference"
LAUNCH_IMAGE = (FIXTURES / "promo_blackfriday.webp").read_bytes()


@pytest.fixture
def promo_row():
    return {
        "version": "1.1", "design_id": "promo", "title": "Promo", "theme_palette": ["#1B1923", "#BD9CF4"],
        "fonts": {"display": {"family": "Inter Tight"}, "mono": {"family": "IBM Plex Mono", "fallback": "monospace"}},
        "root": {
            "node_type": "container", "id": "offer", "layout": "flex_row", "align_items": "center",
            "style": {"background_color": "#1B1923", "max_width": 1200}, "gap_px": 24,
            "children": [
                {"node_type": "text", "id": "left_panel", "tag": "h2", "font_role": "display",
                 "letter_spacing_em": -0.02, "font_size_px": 48, "style": {"flex_grow": 58},
                 "spans": [{"text": "More room to build.", "color": "#F4F1EB"}]},
                {"node_type": "divider", "id": "vertical_rule", "orientation": "vertical", "color": "#4B3E60"},
                {"node_type": "text", "id": "right_panel", "tag": "h2", "style": {"flex_grow": 42},
                 "font_size_px": 160, "font_role": "display", "letter_spacing_em": -0.03,
                 "spans": [{"text": "50", "color": "#BD9CF4", "font_weight": "light"},
                           {"text": "%", "color": "#BD9CF4", "baseline": "super", "size_scale": 0.45}]},
            ],
        },
    }


@pytest.fixture
def promo_spec():
    expected = json.loads((FIXTURES / "promo_blackfriday.expect.json").read_text())
    spec = expected["spec"]
    spec["font_guess"] = {"display": "Inter Tight"}
    spec["regions"] = [region for region in spec["regions"] if region["id"].startswith("offer")]
    spec["regions"][1]["text_ids"] = ["t8"]
    spec["regions"][2]["text_ids"] = ["t13"]
    spec["texts"] = [text for text in spec["texts"] if text["id"] in ("t8", "t13")]
    return ReferenceSpec.model_validate(spec)


def test_legacy_design_prompt_bytes():
    assert sha256(DESIGN_SYSTEM_PROMPT.encode()).hexdigest() == (
        "d02d305428d183b625d918f7f73207c4f5598b8b33b448746bfa092df78f49cd"
    )


def test_promo_palette_keeps_lavender_and_mint_after_provider_resize():
    metadata = extract_vision_metadata(_design_reference_image(LAUNCH_IMAGE))
    assert delta_e(metadata.dominant_background, "#1B1923") <= 5
    assert len(metadata.quantized_hex_colors) <= 64
    for color in ("#BD9CF4", "#BED5D4"):
        assert min(delta_e(color, swatch) for swatch in metadata.roles.accents) <= 8


@pytest.mark.parametrize("change", [{"layout": "grid", "columns": 2, "column_ratios": [58, 42]}, {"wrap": True}])
def test_vertical_divider_requires_a_non_wrapping_flex_row(promo_row, change):
    promo_row["root"].update(change)
    with pytest.raises(ValidationError, match="vertical divider"):
        DesignDocumentASTV11.model_validate(promo_row)


def test_v11_downgrade_keeps_visible_text_and_legacy_wire_shape(promo_row):
    promo_row["root"]["children"][0]["text_transform"] = "uppercase"
    promo_row["root"]["children"][0]["spans"][0]["gradient"] = {"stops": ["#BD9CF4", "#BED5D4"]}
    promo_row["root"]["children"].append({"node_type": "icon_button", "id": "copy_code", "icon_name": "copy"})
    rich = DesignDocumentASTV11.model_validate(promo_row)
    legacy = downgrade_to_v10(rich).model_dump()
    assert DesignDocumentAST.model_validate(legacy).model_dump() == legacy
    left, right, icon = legacy["root"]["children"]
    assert legacy["version"] == "1.0" and "fonts" not in legacy
    assert left["spans"][0] == dict(text="MORE ROOM TO BUILD.", color="#BD9CF4", font_weight="normal",
                                     italic=False, underline=False)
    assert right["font_size_px"] == 96 and right["spans"][0]["font_weight"] == "normal"
    assert icon["node_type"] == "text" and icon["id"] == "copy_code"


def test_ratio_grid_and_font_tokens_are_validated(promo_row):
    row = promo_row["root"]
    row["children"].pop(1)
    row.update(layout="grid", columns=2, column_ratios=[58, 42])
    assert DesignDocumentASTV11.model_validate(promo_row).root.column_ratios == [58, 42]
    row["column_ratios"] = [38, 32, 30]
    with pytest.raises(ValidationError, match="one ratio for each column"):
        DesignDocumentASTV11.model_validate(promo_row)
    row["column_ratios"] = [58, 42]
    promo_row["fonts"]["display"]["family"] = "Unknown Commercial Font"
    with pytest.raises(ValidationError, match="fonts.display.family"):
        DesignDocumentASTV11.model_validate(promo_row)


def test_heals_low_gradient_stop_without_changing_the_legible_fallback(promo_row):
    span = promo_row["root"]["children"][0]["spans"][0]
    span.update(color="#E5D4F2", gradient={"angle_deg": 120, "stops": ["#422767", "#BED5D4"]})
    ast = DesignDocumentASTV11.model_validate(promo_row)
    meta = VisionMetadata(1500, 872, ["#1B1923"], True, "#1B1923",
                          PaletteRoles(["#1B1923"], [], [], ["#BD9CF4", "#BED5D4"]))
    assert any("gradient stop 1" in error for error in validate_design_ast(ast))
    healed = auto_heal_design_ast(ast, vision_meta=meta)
    after = healed.root.children[0].spans[0]
    assert after.color == "#E5D4F2"
    assert after.gradient.angle_deg == 120 and after.gradient.stops == ["#BD9CF4", "#BED5D4"]
    assert validate_design_ast(healed) == []


def test_flat_reference_rejects_cycles_and_drops_unknown_text(promo_spec):
    raw = promo_spec.model_dump()
    region_schema = ReferenceSpec.model_json_schema()["$defs"]["SpecRegion"]
    assert "$ref" not in json.dumps(region_schema)
    raw["regions"][0]["parent_id"] = raw["regions"][2]["id"]
    with pytest.raises(ValidationError, match="cannot form cycles"):
        ReferenceSpec.model_validate(raw)
    raw["regions"][0]["parent_id"] = None
    original = raw["regions"][2]["text_ids"]
    raw["regions"][2]["text_ids"] = ["unknown"]
    spec = ReferenceSpec.model_validate(raw)                       # placement slips are normalised, not rejected
    by_id = {r.id: r.text_ids for r in spec.regions}
    assert by_id[raw["regions"][2]["id"]] == [] and by_id["unplaced"] == original
    raw["regions"].append(dict(raw["regions"][0]))                 # a duplicate region id is still structural
    with pytest.raises(ValidationError, match="must each be unique"):
        ReferenceSpec.model_validate(raw)


def test_fidelity_sees_unmerged_shades_and_checks_panel_geometry(promo_row, promo_spec):
    palette = ("#1B1923", "#BD9CF4", "#F4F1EB", "#4B3E60", "#CDFF23")
    promo_row["root"]["children"][0]["spans"][0]["color"] = "#CDFF23"
    ast = DesignDocumentASTV11.model_validate(promo_row)
    scores, findings = check_fidelity(ast.model_dump(), promo_spec, palette)
    assert scores["text_recall"] == 1 and scores["layout_matches"] == 1
    assert scores["font_class_matches"] == 2
    assert not findings
    _, without_shade = check_fidelity(ast.model_dump(), promo_spec, palette[:-1])
    assert any(f.startswith("FIDELITY_COLOUR_OFF_PALETTE") for f in without_shade)
    ast.root.children = [child for child in ast.root.children if child.node_type != "divider"]
    ast.root.layout = "flex_col"
    scores, findings = check_fidelity(ast.model_dump(), promo_spec, palette)
    assert scores["layout_matches"] == 0
    assert any(f.startswith("FIDELITY_LAYOUT_STACKED") for f in findings)


def test_symbol_glyphs_drawn_as_icons_still_match(promo_row, promo_spec):
    # Live run 2026-11-19: the reference description said "Find my plan ↗" while the design drew the arrow as a
    # trailing_icon, so the copy counted as missing and its panel as stacked, costing a needless repair call.
    promo_spec.texts[0].text = "More room to build. ↗"
    promo_spec.texts[1].text = "50% ⧉"
    ast = DesignDocumentASTV11.model_validate(promo_row).model_dump()
    scores, findings = check_fidelity(ast, promo_spec)
    assert scores["text_recall"] == 1 and scores["layout_matches"] == 1
    assert not [f for f in findings if f.startswith(("FIDELITY_TEXT_MISSING", "FIDELITY_LAYOUT_STACKED"))]
    promo_spec.texts[0].text = "More room to grow. ↗"
    scores, findings = check_fidelity(ast, promo_spec)
    assert scores["text_recall"] == 0.5
    assert any(f.startswith("FIDELITY_TEXT_MISSING") for f in findings)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["fixed", "invalid", "worse"])
async def test_two_pass_images_and_bounded_repair(monkeypatch, promo_row, promo_spec, repair):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    first = copy.deepcopy(promo_row)
    first["root"]["children"].pop(1)
    first["root"]["layout"] = "flex_col"
    repair_succeeds = repair == "fixed"
    if repair == "worse":
        promo_row["root"]["children"][-1]["spans"] = [{"text": "New copy", "color": "#BD9CF4"}]
    response = "broken JSON" if repair == "invalid" else json.dumps(promo_row)
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [promo_spec.model_dump_json(), json.dumps(first), response]
    report = {}
    image = _design_reference_image(LAUNCH_IMAGE)
    result = await execute_design_canvas_harness(runner, "recreate this", "create_from_vision", image, report=report)
    assert result.root.layout == ("flex_row" if repair_succeeds else "flex_col")
    calls = runner.run_json_completion.await_args_list
    assert [call.kwargs["image_bytes"] for call in calls] == [image] * 3
    assert calls[0].kwargs["messages"][0]["content"] == REFERENCE_SYSTEM_PROMPT
    assert calls[1].kwargs["messages"][0]["content"] == DESIGN_SYSTEM_PROMPT_V11
    assert "<reference_spec>" in calls[1].kwargs["messages"][1]["content"]
    assert "<reference_fidelity_findings>" in calls[2].kwargs["messages"][-1]["content"]
    assert report["attempts"][0]["outcome"] == "fidelity_repair"
    assert report["fidelity"]["layout_matches"] == int(repair_succeeds)
    assert bool(report["fidelity"]["findings"]) is not repair_succeeds


def _tracking_off(row):
    row["root"]["children"][0]["letter_spacing_em"] = 0


def _percent_not_raised(row):
    row["root"]["children"][-1]["spans"][1]["baseline"] = "normal"


def _off_palette(row):
    row["root"]["children"][0]["spans"][0]["color"] = "#00FF00"


def _copy_changed(row):
    row["root"]["children"][0]["spans"][0]["text"] = "Room to build."


def _serif_display(row):
    row["fonts"]["display"] = {"family": "Playfair Display", "fallback": "serif"}


def _fonts_removed(row):
    row["fonts"] = None


@pytest.mark.asyncio
@pytest.mark.parametrize("change, finding", [
    (_percent_not_raised, "FIDELITY_BASELINE"),
    (_off_palette, "FIDELITY_COLOUR_OFF_PALETTE"),
])
async def test_cosmetic_findings_are_reported_without_a_repair_call(monkeypatch, promo_row, promo_spec, change, finding):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    change(promo_row)
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [promo_spec.model_dump_json(), json.dumps(promo_row)]
    report = {}
    await execute_design_canvas_harness(runner, "recreate this", "create_from_vision",
                                        _design_reference_image(LAUNCH_IMAGE), report=report)
    assert runner.run_json_completion.await_count == 2
    assert [a["outcome"] for a in report["attempts"]] == ["pass_with_fidelity_warnings"]
    assert any(f.startswith(finding) for f in report["fidelity"]["findings"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change, finding", [
    (_copy_changed, "FIDELITY_TEXT_MISSING"),
    (_serif_display, "FIDELITY_FONT_CLASS"),
    (_fonts_removed, "FIDELITY_FONTS_MISSING"),
])
async def test_text_and_font_findings_trigger_the_repair_call(monkeypatch, promo_row, promo_spec, change, finding):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    first = copy.deepcopy(promo_row)
    change(first)
    _percent_not_raised(first)   # a cosmetic finding rides along with the triggering one
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [promo_spec.model_dump_json(), json.dumps(first), json.dumps(promo_row)]
    report = {}
    await execute_design_canvas_harness(runner, "recreate this", "create_from_vision",
                                        _design_reference_image(LAUNCH_IMAGE), report=report)
    assert runner.run_json_completion.await_count == 3
    assert report["attempts"][0]["outcome"] == "fidelity_repair"
    sent = runner.run_json_completion.await_args_list[2].kwargs["messages"][-1]["content"]
    assert finding in sent and "FIDELITY_BASELINE" in sent
    assert report["fidelity"]["findings"] == []


@pytest.mark.asyncio
async def test_invalid_reference_description_builds_from_image_and_reports_it(monkeypatch, promo_row):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    runner = AsyncMock()
    runner.run_json_completion.side_effect = ["bad spec", "still bad", json.dumps(promo_row)]
    image = _design_reference_image(LAUNCH_IMAGE)
    report = {}
    result = await execute_design_canvas_harness(runner, "recreate", "create_from_vision", image, report=report)
    assert result.version == "1.1"
    assert runner.run_json_completion.await_count == 3            # describe, one repair, build
    assert report["reference_analysis"] == "unavailable"
    assert report["fidelity"]["findings"][0].startswith("FIDELITY_ANALYSIS_UNAVAILABLE")


@pytest.mark.asyncio
async def test_one_slip_in_the_description_is_repaired_not_dropped(monkeypatch, promo_row, promo_spec):
    # Live 2026-11-19: '"aspect":"3:2".length?"16:9":"16:9"' made the description invalid JSON and the design was
    # built with no layout checks and no typography measurement.
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    broken = promo_spec.model_dump_json().replace('"regions":', '"regions".length?:', 1)
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [broken, promo_spec.model_dump_json(), json.dumps(promo_row)]
    report = {}
    await execute_design_canvas_harness(runner, "recreate", "create_from_vision",
                                        _design_reference_image(LAUNCH_IMAGE), report=report)
    assert report["reference_analysis"] == "pass_after_repair"
    repair = runner.run_json_completion.await_args_list[1].kwargs["messages"]
    assert repair[-2] == {"role": "assistant", "content": broken}
    assert "<reference_spec_validation_failure>" in repair[-1]["content"]
    build = runner.run_json_completion.await_args_list[2].kwargs["messages"][1]["content"]
    assert "<reference_spec>" in build


@pytest.mark.asyncio
async def test_request_can_keep_legacy_wire_format_and_v11_edits_need_flag(monkeypatch, promo_row):
    runner = AsyncMock()
    raw = json.dumps(promo_row)
    monkeypatch.setattr(settings, "design_schema_v11_enabled", False)
    with pytest.raises(ValueError, match="requires DESIGN_SCHEMA_V11_ENABLED"):
        await execute_design_canvas_harness(runner, "edit", "edit_existing_ast", existing_ast_json=raw)
    runner.run_json_completion.assert_not_awaited()
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    runner.run_json_completion.return_value = raw
    legacy = await execute_design_canvas_harness(runner, "edit", "edit_existing_ast",
                                                 existing_ast_json=raw, schema_version="1.0")
    assert type(legacy) is DesignDocumentAST and legacy.version == "1.0"
    assert runner.run_json_completion.await_args.kwargs["messages"][0]["content"] == DESIGN_SYSTEM_PROMPT


def test_flex_panels_and_rule_stay_on_one_row_at_narrow_browser_width(promo_row):
    playwright = pytest.importorskip("playwright.sync_api")
    ast = DesignDocumentASTV11.model_validate(promo_row).model_dump()
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Install Playwright Chromium to run the reference render check")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 370, "height": 650})
        page.route("https://**/*", lambda route: route.abort())
        page.set_content(render_html(ast, width=1200), wait_until="load")
        left, rule, right = [page.locator(f'[data-node-id="{node}"]').bounding_box()
                             for node in ("left_panel", "vertical_rule", "right_panel")]
        assert left["x"] + left["width"] < rule["x"] < right["x"]
        assert left["y"] < right["y"] + right["height"] and left["y"] + left["height"] > right["y"]
        assert rule["height"] >= right["height"] and rule["width"] > 0
        assert right["x"] + right["width"] <= 370
        assert left["width"] / (left["width"] + right["width"]) == pytest.approx(.58, abs=.01)
        browser.close()


def _spec_text(tid, text, letter_spacing="normal", font_class="grotesk_sans", **extra):
    return {"id": tid, "text": text, "role": "body", "approx_size_px": 14, "color_hex": "#F4F1EB",
            "font_class": font_class, "weight": "normal", "uppercase": False, "letter_spacing": letter_spacing, **extra}


@pytest.fixture
def perks_row():
    """The promo's inline rows: '✓ … ✓ … | Compare every detail', 'Use code [BLACKFRIDAY]' and the 50% numeral."""
    text = lambda nid, words, **kw: {"node_type": "text", "id": nid, "tag": "p", "font_size_px": kw.pop("size", 14),
                                     "spans": [{"text": words, "color": "#F4F1EB"}], **kw}
    return {
        "version": "1.1", "design_id": "perks", "title": "Perks", "theme_palette": ["#1B1923", "#BD9CF4"],
        "fonts": {"display": {"family": "Inter Tight"}, "body": {"family": "Inter"},
                  "mono": {"family": "JetBrains Mono", "fallback": "monospace"}},
        "root": {"node_type": "container", "id": "root", "layout": "flex_col",
                 "style": {"background_color": "#1B1923"}, "children": [
            text("t3", "Celebrate Thanksgiving & Black Friday with 50% off any paid monthly plan."),
            {"node_type": "container", "id": "perks", "layout": "flex_row", "children": [
                text("t4", "✓ 14-day free trial"), text("t5", "✓ No credit card required"),
                {"node_type": "button", "id": "t6", "label": "Compare every detail", "text_color": "#F4F1EB",
                 "trailing_icon": "chevron_down", "font_size_px": 14}]},
            text("t13", "50%", size=160, font_role="display", letter_spacing_em=-0.05),
            {"node_type": "container", "id": "code_row", "layout": "flex_row", "children": [
                text("t16", "Use code"),
                {"node_type": "button", "id": "t17", "label": "BLACKFRIDAY", "text_color": "#F4F1EB",
                 "font_role": "mono", "font_size_px": 14, "trailing_icon": "copy",
                 "style": {"border_width": 1, "border_color": "#F4F1EB"}}]},
        ]},
    }


@pytest.fixture
def perks_spec():
    texts = [_spec_text("t3", "Celebrate Thanksgiving & Black Friday with 50% off any paid monthly plan."),
             _spec_text("t4", "14-day free trial"), _spec_text("t5", "No credit card required"),
             _spec_text("t6", "Compare every detail", rule_before=True),
             _spec_text("t13", "50%", "tight"), _spec_text("t16", "Use code"),
             _spec_text("t17", "BLACKFRIDAY", "wide", "mono")]
    return ReferenceSpec.model_validate({
        "canvas_background_hex": "#1B1923", "font_guess": {"display": "Inter Tight", "body": "Inter"},
        "regions": [{"id": "hero", "arrangement": "stack", "text_ids": [t["id"] for t in texts]}], "texts": texts})


def test_tracking_ranges_follow_the_promo_measurements():
    # Ink width/height matched against Chromium renders, 2026-11-19: 96 px display -0.03, 46 px heading -0.01,
    # BLACKFRIDAY (mono, bold) +0.10, OFF FOR 3 MONTHS +0.20, eyebrow +0.15.
    # DM Sans (typography match, same day): 96 px display -0.04, 46 px heading -0.019.
    for spacing, size, measured in (("tight", 96, -0.03), ("tight", 46, -0.01), ("tight", 96, -0.04),
                                    ("tight", 46, -0.019), ("wide", 14, 0.10),
                                    ("wide", 15, 0.20), ("wide", 13, 0.15), ("normal", 14, 0.0)):
        low, high = tracking_range(spacing, size)
        assert low <= measured <= high


def test_heal_fixes_tracking_and_inserts_the_inline_rule(perks_row, perks_spec):
    raw = DesignDocumentASTV11.model_validate(perks_row).model_dump()
    changes = heal_reference_details(raw, perks_spec)
    healed = DesignDocumentASTV11.model_validate(raw).model_dump()   # still a valid 1.1 document
    nodes = {node["id"]: node for node in _walk_dump(healed["root"])}
    assert nodes["t13"]["letter_spacing_em"] == -0.045          # was -0.05; nearest edge of the 160 px tight range
    assert nodes["t17"]["letter_spacing_em"] == 0.08            # the button gained wide tracking
    assert nodes["t3"]["letter_spacing_em"] == 0                # "50%" never restyles the sentence containing it
    perks = [child["id"] for child in nodes["perks"]["children"]]
    assert perks == ["t4", "t5", "t6_rule", "t6"]
    rule = nodes["t6_rule"]
    assert rule["orientation"] == "vertical" and rule["color"] == "#F4F1EB59"
    assert len(changes) == 3
    scores, findings = check_fidelity(healed, perks_spec)
    assert scores["separator_matches"] == scores["separator_texts"] == 1
    assert not [f for f in findings if f.startswith(("FIDELITY_TRACKING", "FIDELITY_SEPARATOR_MISSING"))]
    assert heal_reference_details(raw, perks_spec) == []        # idempotent


def test_heal_reuses_an_existing_rule_colour_and_leaves_stacked_items_alone(perks_row, perks_spec):
    perks_row["root"]["children"][3]["children"].insert(1, {"node_type": "divider", "id": "code_rule",
                                                            "orientation": "vertical", "color": "#4B3E60"})
    raw = DesignDocumentASTV11.model_validate(perks_row).model_dump()
    heal_reference_details(raw, perks_spec)
    assert raw["root"]["children"][1]["children"][2]["color"] == "#4B3E60"
    perks_row["root"]["children"][1]["layout"] = "flex_col"
    perks_row["root"]["children"][3]["children"].pop(1)
    raw = DesignDocumentASTV11.model_validate(perks_row).model_dump()
    heal_reference_details(raw, perks_spec)
    assert [c["id"] for c in raw["root"]["children"][1]["children"]] == ["t4", "t5", "t6"]
    _, findings = check_fidelity(raw, perks_spec)
    assert any(f.startswith("FIDELITY_SEPARATOR_MISSING") for f in findings)


@pytest.mark.asyncio
async def test_harness_heals_tracking_and_rule_without_a_model_call(monkeypatch, perks_row, perks_spec):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [perks_spec.model_dump_json(), json.dumps(perks_row)]
    report = {}
    result = await execute_design_canvas_harness(runner, "recreate this", "create_from_vision",
                                                 _design_reference_image(LAUNCH_IMAGE), report=report)
    assert runner.run_json_completion.await_count == 2
    assert len(report["attempts"]) == 1 and len(report["attempts"][0]["reference_heal"]) == 3
    assert not [f for f in report["fidelity"]["findings"]
                if f.startswith(("FIDELITY_TRACKING", "FIDELITY_SEPARATOR_MISSING"))]
    perks = next(child for child in result.root.children if child.id == "perks")
    assert [child.id for child in perks.children] == ["t4", "t5", "t6_rule", "t6"]


def test_button_tracking_and_case_render_and_downgrade(perks_row):
    button = perks_row["root"]["children"][3]["children"][1]
    button.update(label="Blackfriday", text_transform="uppercase", letter_spacing_em=0.1)
    ast = DesignDocumentASTV11.model_validate(perks_row)
    html = render_html(ast.model_dump(), 800)
    assert "letter-spacing:0.1em" in html and "text-transform:uppercase" in html
    legacy = downgrade_to_v10(ast)
    assert legacy.root.children[3].children[1].label == "BLACKFRIDAY"
    assert "letter_spacing_em" not in legacy.model_dump()["root"]["children"][3]["children"][1]


def _walk_dump(node):
    yield node
    for child in node.get("children", ()):
        yield from _walk_dump(child)


def _listing_spec(filters_arrangement="side_by_side", ratio=None, columns=4):
    texts = [_spec_text(f"f{i}", label) for i, label in enumerate(("Filter", "Sort By", "Ratings"))]
    texts += [_spec_text(f"c{i}", name) for i, name in enumerate(("Wow! China", "KFC", "Boba Bhai", "McDonald's", "Burgjoy"))]
    regions = [{"id": "filters", "arrangement": filters_arrangement, "ratio": ratio, "text_ids": ["f0", "f1", "f2"]},
               {"id": "grid", "arrangement": "grid", "columns": columns, "text_ids": []}]
    regions += [{"id": f"card{i}", "parent_id": "grid", "arrangement": "stack", "text_ids": [f"c{i}"]} for i in range(5)]
    return ReferenceSpec.model_validate({"canvas_background_hex": "#FFFFFF", "font_guess": {}, "regions": regions,
                                         "texts": texts})


def _listing_ast(grid_columns=4, filters_layout="flex_row"):
    label = lambda nid, words: {"node_type": "text", "id": nid, "tag": "p",
                                "spans": [{"text": words, "color": "#1B1B1F"}]}
    chips = {"node_type": "container", "id": "filters", "layout": filters_layout, "children": [
        {"node_type": "button", "id": f"chip{i}", "label": words, "text_color": "#1B1B1F"}
        for i, words in enumerate(("Filter", "Sort By", "Ratings"))]}
    cards = [{"node_type": "container", "id": f"card{i}", "children": [label(f"name{i}", name)]}
             for i, name in enumerate(("Wow! China", "KFC", "Boba Bhai", "McDonald's", "Burgjoy"))]
    grid = {"node_type": "container", "id": "grid", "layout": "grid", "columns": grid_columns, "children": cards}
    if grid_columns is None:
        grid = {**grid, "layout": "flex_col", "columns": None}
    return DesignDocumentASTV11.model_validate({
        "version": "1.1", "design_id": "listing", "title": "Listing", "theme_palette": ["#FFFFFF", "#1B1B1F"],
        "root": {"node_type": "container", "id": "root", "style": {"background_color": "#FFFFFF"},
                 "children": [chips, grid]}}).model_dump()


def test_chip_row_sent_as_side_by_side_without_panels_no_longer_rejects_the_description():
    # Live 2026-11-19 (burger listing): this one hint made the whole description fail validation.
    spec = _listing_spec(ratio=[50, 50])
    filters = next(r for r in spec.regions if r.id == "filters")
    assert filters.arrangement == "row" and filters.ratio is None
    with pytest.raises(ValidationError, match="cannot form cycles"):    # real structural errors still fail
        ReferenceSpec.model_validate({**json.loads(spec.model_dump_json()), "regions": [
            {"id": "a", "parent_id": "b", "arrangement": "stack", "text_ids": ["f0"]},
            {"id": "b", "parent_id": "a", "arrangement": "stack", "text_ids": []}]})


def test_row_and_card_grid_are_checked():
    spec = _listing_spec(filters_arrangement="row")
    scores, findings = check_fidelity(_listing_ast(), spec)
    assert scores["layout_matches"] == scores["layout_regions"] == 2
    assert not [f for f in findings if f.startswith("FIDELITY_LAYOUT")]
    for ast, broken in ((_listing_ast(grid_columns=3), "grid"), (_listing_ast(grid_columns=None), "grid"),
                        (_listing_ast(filters_layout="flex_col"), "filters")):
        scores, findings = check_fidelity(ast, spec)
        assert scores["layout_matches"] == 1
        assert any(f.startswith("FIDELITY_LAYOUT_STACKED") and repr(broken) in f for f in findings)
