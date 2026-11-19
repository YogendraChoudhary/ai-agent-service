"""Measured typography: family, exact weight and letter spacing of reference lines, from bundled OFL fonts."""

import io
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from pydantic import ValidationError

from config import settings
from scripts.design_reference_render import font_query, render_html
from src.design_canvas import service
from src.design_canvas.reference import ReferenceSpec, check_fidelity, heal_reference_details
from src.design_canvas.schema_v11 import ALLOWED_FONT_FAMILIES, DesignDocumentASTV11, downgrade_to_v10
from src.design_canvas.service import execute_design_canvas_harness
from src.design_canvas.typography_match import (
    FONT_DIR, FONT_FILES, SPACING_SLACK, MeasuredText, TypographyMatch, match_typography, measure_line, named_weight,
    reference_line, render_line, tracking_range,
)
from src.text_processing_agent import _design_measure_image, _design_reference_image

FIXTURE = Path(__file__).parent / "fixtures/reference/promo_blackfriday.webp"
# Boxes Claude returned for the promo in a live describe call (2026-11-19), in the 1500 x 871 image it was sent.
PROMO_BOXES = {"t1": [378, 14, 1120, 108], "t2": [316, 115, 1178, 212], "t8": [88, 473, 475, 516]}


def _spec_text(tid, text, role="display", box=None, **extra):
    return {"id": tid, "text": text, "role": role, "approx_size_px": 60, "color_hex": "#F4F1EB",
            "font_class": "grotesk_sans", "weight": "semibold", "uppercase": False, "letter_spacing": "tight",
            "box": box, **extra}


def _spec(texts, font_guess=None):
    return ReferenceSpec.model_validate({
        "canvas_background_hex": "#1B1923", "font_guess": font_guess or {"display": "Inter Tight"},
        # A full-page box: these stand for complete descriptions (description_coverage would otherwise ask for more).
        "regions": [{"id": "page", "arrangement": "stack", "text_ids": [t["id"] for t in texts],
                     "box": [0, 0, 20000, 20000]}], "texts": texts})


def _screenshot(lines, background=(27, 25, 35), ink=(244, 241, 235), gradient=None):
    """Draw lines (family, weight, size, spacing, text) one under another; return PNG bytes and each line's box."""
    rendered = [render_line(f, w, size, ls, text) for f, w, size, ls, text in lines]
    width = max(r.width for r in rendered) + 80
    height = sum(r.height for r in rendered) + 30 * len(rendered) + 40
    canvas = Image.new("RGB", (width, height), background)
    boxes, y = [], 30
    for mask in rendered:
        fill = Image.new("RGB", mask.size, ink)
        if gradient:
            for x in range(mask.width):
                t = x / max(mask.width - 1, 1)
                colour = tuple(round(a + (b - a) * t) for a, b in zip(*gradient))
                fill.paste(colour, (x, 0, x + 1, mask.height))
        canvas.paste(fill, (40, y), mask)
        boxes.append([40, y, 40 + mask.width, y + mask.height])
        y += mask.height + 30
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue(), boxes


def test_every_allowed_family_is_bundled_with_its_licence():
    assert set(FONT_FILES) == set(ALLOWED_FONT_FAMILIES)
    for family, (directory, files) in FONT_FILES.items():
        assert (FONT_DIR / directory / "OFL.txt").is_file(), family
        for name in (files.values() if isinstance(files, dict) else [files]):
            assert (FONT_DIR / directory / name).is_file(), name


@pytest.mark.parametrize("family, weight, spacing, background, ink", [
    ("Urbanist", 600, -0.02, (27, 25, 35), (244, 241, 235)),
    ("Inter Tight", 500, -0.03, (250, 249, 246), (20, 20, 24)),       # dark on light
    ("Poppins", 700, 0.0, (27, 25, 35), (244, 241, 235)),             # static weights
])
def test_recovers_known_family_weight_and_spacing(family, weight, spacing, background, ink):
    png, boxes = _screenshot([(family, weight, 64, spacing, "Your next big idea."),
                              (family, weight, 64, spacing, "Half the price to start.")], background, ink)
    spec = _spec([_spec_text("a", "Your next big idea.", box=boxes[0]),
                  _spec_text("b", "Half the price to start.", box=boxes[1])])
    result = match_typography(png, spec)
    assert result.fonts == {"display": family}
    for measured in result.texts.values():
        assert abs(measured.weight - weight) <= 50
        assert abs(measured.letter_spacing_em - spacing) <= 0.012


def test_gradient_text_and_a_neighbouring_line_in_the_box_do_not_skew_the_measurement():
    clean_png, clean_boxes = _screenshot([("DM Sans", 550, 64, -0.035, "Half the price to start.")])
    clean = match_typography(clean_png, _spec([_spec_text("a", "Half the price to start.", box=clean_boxes[0])]))
    png, boxes = _screenshot([("DM Sans", 550, 64, -0.035, "Your next big idea."),
                              ("DM Sans", 550, 64, -0.035, "Half the price to start.")],
                             gradient=((201, 176, 238), (190, 213, 212)))
    x0, y0, x1, y1 = boxes[1]
    loose = [x0, y0 - 24, x1, y1]                     # reaches up into the "g" and "y" of the line above
    result = match_typography(png, _spec([_spec_text("a", "Half the price to start.", box=loose)]))
    assert result.fonts == {"display": "DM Sans"}
    assert abs(result.texts["a"].weight - clean.texts["a"].weight) <= 25
    assert abs(result.texts["a"].letter_spacing_em - clean.texts["a"].letter_spacing_em) <= 0.008


def test_promo_headlines_measure_as_dm_sans_about_550_from_the_original_upload():
    original = _design_measure_image(FIXTURE.read_bytes())
    texts = [_spec_text("t1", "Your next big idea.", box=PROMO_BOXES["t1"]),
             _spec_text("t2", "Half the price to start.", box=PROMO_BOXES["t2"]),
             _spec_text("t8", "More room to build.\nA little less to spend.", role="heading", box=PROMO_BOXES["t8"])]
    started = time.perf_counter()
    result = match_typography(original, _spec(texts), supplied_width=1500)
    assert time.perf_counter() - started < 5
    assert result.fonts == {"display": "DM Sans", "heading": "DM Sans"}
    assert all(500 <= m.weight <= 600 for m in result.texts.values())
    assert all(-0.05 <= m.letter_spacing_em <= -0.03 for m in (result.texts["t1"], result.texts["t2"]))
    assert result.ranking["display"][0][0] == "DM Sans"
    inter_tight = dict(result.ranking["display"])["Inter Tight"]
    assert dict(result.ranking["display"])["DM Sans"] > inter_tight + 0.05


def test_unmeasurable_texts_are_skipped_with_a_reason():
    png, boxes = _screenshot([("Inter", 500, 64, 0.0, "Plan")])
    spec = _spec([_spec_text("a", "50%", role="numeral", box=boxes[0], raised_suffix="%"),
                  _spec_text("b", "Plan", box=None), _spec_text("c", "Plan", box=[0, 0, 5, 5]),
                  _spec_text("d", "body copy", role="body", box=boxes[0])])
    result = match_typography(png, spec)
    assert result.skipped == {"a": "raised suffix", "b": "no box", "c": "no measurable line in box"}
    assert result.fonts == {} and result.texts == {}


def test_box_must_be_ordered():
    with pytest.raises(ValidationError):
        _spec([_spec_text("a", "Plan", box=[10, 10, 5, 20])])


def _typography():
    return TypographyMatch(fonts={"display": "DM Sans"}, texts={
        "t1": MeasuredText("t1", "display", "DM Sans", 550, -0.04, 0.63)})


@pytest.fixture
def headline_ast():
    return DesignDocumentASTV11.model_validate({
        "version": "1.1", "design_id": "promo", "title": "Promo", "theme_palette": ["#1B1923", "#F4F1EB"],
        "fonts": {"display": {"family": "Inter Tight"}},
        "root": {"node_type": "container", "id": "root", "style": {"background_color": "#1B1923"}, "children": [
            {"node_type": "text", "id": "headline", "tag": "h1", "font_size_px": 92, "font_role": "heading",
             "letter_spacing_em": -0.05, "spans": [{"text": "Your next big idea.", "color": "#F4F1EB",
                                                     "font_weight": "semibold"}]},
            {"node_type": "text", "id": "tagline", "tag": "p", "font_size_px": 92, "letter_spacing_em": -0.05,
             "spans": [{"text": "Half the price to start.", "color": "#F4F1EB"}]}]}})


def test_heal_applies_measured_typography_and_leaves_ranges_to_unmeasured_text(headline_ast):
    spec = _spec([_spec_text("t1", "Your next big idea.", box=[1, 1, 9, 9]), _spec_text("t2", "Half the price to start.")])
    raw = headline_ast.model_dump()
    changes = heal_reference_details(raw, spec, _typography())
    healed = DesignDocumentASTV11.model_validate(raw).model_dump()
    headline, tagline = healed["root"]["children"]
    assert healed["fonts"]["display"] == {"family": "DM Sans", "fallback": "sans-serif"}
    assert headline["font_role"] == "display" and headline["letter_spacing_em"] == -0.04
    assert headline["spans"][0]["font_weight_value"] == 550 and headline["spans"][0]["font_weight"] == "semibold"
    assert tagline["letter_spacing_em"] == -0.045                 # unmeasured: tight range for 92 px
    assert "font_weight_value" not in tagline["spans"][0] or tagline["spans"][0]["font_weight_value"] is None
    assert any(c.startswith("font display: DM Sans") for c in changes)
    scores, findings = check_fidelity(healed, spec, typography=_typography())
    assert scores["typography_matches"] == scores["typography_texts"] == 1
    assert not [f for f in findings if f.startswith(("FIDELITY_TYPOGRAPHY", "FIDELITY_TRACKING"))]
    assert heal_reference_details(raw, spec, _typography()) == []
    _, unhealed = check_fidelity(headline_ast.model_dump(), spec, typography=_typography())
    assert any(f.startswith("FIDELITY_TYPOGRAPHY") for f in unhealed)


def test_exact_weight_renders_downgrades_and_requests_font_axes(headline_ast):
    raw = headline_ast.model_dump()
    raw["root"]["children"][0]["spans"][0]["font_weight_value"] = 550
    raw["fonts"]["display"] = {"family": "DM Sans", "fallback": "sans-serif"}
    raw["root"]["children"][0]["font_role"] = "display"
    html = render_html(raw, 1200)
    assert "font-weight:550" in html and "DM+Sans%3Aopsz%2Cwght%409..40%2C100..1000" in html
    assert font_query("Poppins") == "Poppins:wght@300;400;500;600;700;800"
    legacy = downgrade_to_v10(DesignDocumentASTV11.model_validate(raw)).model_dump()
    span = legacy["root"]["children"][0]["spans"][0]
    assert span["font_weight"] == "semibold" and "font_weight_value" not in span
    assert [named_weight(w) for w in (300, 525, 550, 575, 650, 800)] == [
        "light", "medium", "semibold", "semibold", "bold", "extrabold"]


def test_measure_image_is_capped_png():
    measured = _design_measure_image(FIXTURE.read_bytes())
    with Image.open(io.BytesIO(measured)) as img:
        assert img.format == "PNG" and max(img.size) <= 2600 and img.size == (2512, 1460)
    assert _design_measure_image(b"not an image") is None


def _harness_runner(spec_texts, ast):
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [_spec(spec_texts).model_dump_json(), json.dumps(ast)]
    return runner


@pytest.mark.asyncio
async def test_harness_measures_prompts_and_applies_typography(monkeypatch, headline_ast):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    texts = [_spec_text("t1", "Your next big idea.", box=PROMO_BOXES["t1"]),
             _spec_text("t2", "Half the price to start.", box=PROMO_BOXES["t2"])]
    runner = _harness_runner(texts, headline_ast.model_dump())
    report = {}
    result = await execute_design_canvas_harness(
        runner, "recreate", "create_from_vision", _design_reference_image(FIXTURE.read_bytes()), report=report,
        measure_image_bytes=_design_measure_image(FIXTURE.read_bytes()))
    assert runner.run_json_completion.await_count == 2
    assert report["typography"]["fonts"] == {"display": "DM Sans"}
    build_prompt = runner.run_json_completion.await_args_list[1].kwargs["messages"][1]["content"]
    assert "<measured_typography>" in build_prompt and '"DM Sans"' in build_prompt
    assert result.fonts.display.family == "DM Sans"
    headline = result.root.children[0]
    assert 500 <= headline.spans[0].font_weight_value <= 600 and headline.font_role == "display"
    assert report["fidelity"]["typography_matches"] == report["fidelity"]["typography_texts"] == 2


@pytest.mark.asyncio
async def test_harness_falls_back_when_measurement_fails_or_images_differ(monkeypatch, headline_ast):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    texts = [_spec_text("t1", "Your next big idea.", box=PROMO_BOXES["t1"])]
    image = _design_reference_image(FIXTURE.read_bytes())
    seen = {}
    real = service.match_typography

    def spy(measure, spec, supplied_width):
        seen["measure"] = measure
        return real(measure, spec, supplied_width)

    monkeypatch.setattr(service, "match_typography", spy)
    portrait = io.BytesIO()
    Image.new("RGB", (800, 1600)).save(portrait, "PNG")          # not the same pixels: boxes would not line up
    report = {}
    await execute_design_canvas_harness(_harness_runner(texts, headline_ast.model_dump()), "recreate",
                                        "create_from_vision", image, report=report,
                                        measure_image_bytes=portrait.getvalue())
    assert seen["measure"] is image and report["typography"]["fonts"] == {"display": "DM Sans"}

    def broken(*_):
        raise RuntimeError("font file unreadable")

    monkeypatch.setattr(service, "match_typography", broken)
    report = {}
    result = await execute_design_canvas_harness(_harness_runner(texts, headline_ast.model_dump()), "recreate",
                                                 "create_from_vision", image, report=report)
    assert report["typography"] == {"error": "RuntimeError"}
    assert result.fonts.display.family == "Inter Tight"          # the model's design, untouched by measurement


def test_spacing_cannot_stand_in_for_a_wider_typeface():
    # Live 2026-11-19 (burger listing): "Burger", described as tight, measured as Inter 800 at +0.056 em: visibly
    # spread letters. The described tracking class now bounds the solved spacing (with SPACING_SLACK).
    png, boxes = _screenshot([("Sora", 700, 64, 0.0, "Restaurants to explore")])      # wider than Inter
    with Image.open(io.BytesIO(png)) as img:
        reference = reference_line(img, boxes[0])
    ink = reference.info["source_height"]
    tight = measure_line(reference, "Restaurants to explore", "Inter", "tight", ink)
    free = measure_line(reference, "Restaurants to explore", "Inter")
    assert tight[2] <= tracking_range("tight", 64)[1] + SPACING_SLACK
    assert free[2] > 0.03                          # unbounded, Inter would spread its letters to fill Sora's width
    assert measure_line(reference, "Restaurants to explore", "Inter", "wide", ink)[2] > 0.03


def test_font_size_comes_from_the_box_not_the_size_guess():
    png, boxes = _screenshot([("DM Sans", 550, 100, -0.04, "Your next big idea.")])
    for guess in (40, 100, 200):                    # the describe call's approx_size_px no longer matters
        spec = _spec([_spec_text("a", "Your next big idea.", box=boxes[0], approx_size_px=guess)])
        measured = match_typography(png, spec).texts["a"]
        assert measured.family == "DM Sans" and abs(measured.letter_spacing_em + 0.04) <= 0.01
