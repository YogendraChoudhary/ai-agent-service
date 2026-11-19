"""A screenshot used as a template for new content ("like this, but for pizzas") vs a copy of it."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from config import settings
from src.design_canvas.prompts import build_initial_design_prompt
from src.design_canvas.reference import ReferenceSpec, check_fidelity
from src.design_canvas.schema_v11 import DesignDocumentASTV11
from src.design_canvas.service import execute_design_canvas_harness
from src.design_intent import detect_design_intent, wants_new_content
from src.text_processing_agent import _design_reference_image

SHOT = (Path(__file__).parent / "fixtures/reference/promo_blackfriday.webp").read_bytes()


@pytest.mark.parametrize("query", [
    "Design a page like this for pizzas and Asian fusion dishes",          # was answered as chat before
    "Make a design like this for my bakery",
    "Build a landing page in this style for a yoga studio",
    "Create a menu page based on this for a sushi bar",
])
def test_screenshot_as_template_routes_to_design(query):
    decision = detect_design_intent(query, "general", True, False)
    assert decision.sub_mode == "create_from_vision"


@pytest.mark.parametrize("query", ["make a picture like this", "make it look like this", "what is in this image?",
                                   "Design a page like this"])
def test_template_rule_needs_a_screenshot_a_design_verb_and_a_page_noun(query):
    with_image = detect_design_intent(query, "general", True, False)
    without_image = detect_design_intent(query, "general", False, False)
    if query == "Design a page like this":
        assert with_image.sub_mode == "create_from_vision" and not without_image.is_design_request
    else:
        assert not with_image.is_design_request


@pytest.mark.parametrize("query, new", [
    ("Create a design like this screenshot, but for a page of pizzas and Asian-American fusion dishes instead of "
     "burgers.", True),
    ("Design a page like this for pizzas and Asian fusion dishes", True),
    ("clone this design for our coffee shop", True),
    ("same layout but with sneakers", True),
    ("replace the dishes with desserts", True),
    ("replicate this design 100%", False),
    ("Recreate this screenshot faithfully as an editable design.", False),
    ("Create a design like this screenshot", False),
    ("Recreate this but in dark mode", False),
])
def test_new_content_wording(query, new):
    assert wants_new_content(query) is new


def test_adapt_prompt_asks_for_new_content_and_copy_prompt_for_verbatim_text():
    copy = build_initial_design_prompt("replicate this", "create_from_vision", schema_version="1.1")
    adapt = build_initial_design_prompt("like this for pizzas", "create_from_vision", schema_version="1.1",
                                        reference_use="adapt")
    assert "Copy visible text verbatim" in copy and "Copy visible text verbatim" not in adapt
    assert "layout and style template" in adapt.replace("\n", " ") and "photo of the NEW content" in adapt
    assert "examples of the content, not copy to keep" in adapt
    assert "prompt = its description" in copy and "prompt = its description" not in adapt


@pytest.fixture
def listing_spec():
    texts = [{"id": f"n{i}", "text": name, "role": "subheading", "approx_size_px": 17, "color_hex": "#1B1B1F",
              "font_class": "geometric_sans", "weight": "semibold", "uppercase": False, "letter_spacing": "normal"}
             for i, name in enumerate(("KFC", "Burger King", "McDonald's", "Burgjoy"))]
    regions = [{"id": "grid", "arrangement": "grid", "columns": 2, "text_ids": [], "box": [0, 0, 20000, 20000]}]
    regions += [{"id": f"c{i}", "parent_id": "grid", "arrangement": "stack", "text_ids": [f"n{i}"]} for i in range(4)]
    images = [{"id": f"img{i}", "region_id": f"c{i}", "description": "a burger", "box": [0, 0, 280, 186]}
              for i in range(4)]
    return ReferenceSpec.model_validate({"canvas_background_hex": "#FFFFFF", "font_guess": {"heading": "Outfit"},
                                         "regions": regions, "texts": texts, "images": images})


def _pizza_page(columns=2, photos=4):
    cards = []
    for i, name in enumerate(("Slice Society", "Seoul Pie Co.", "Napoli Express", "Crust Joy")):
        kids = [{"node_type": "text", "id": f"name{i}", "tag": "p", "font_role": "heading",
                 "spans": [{"text": name, "color": "#1B1B1F"}]}]
        if i < photos:
            kids.insert(0, {"node_type": "image", "id": f"img{i}", "alt": "Pizza", "prompt": "a pepperoni pizza",
                            "aspect_ratio": "140:93"})
        cards.append({"node_type": "container", "id": f"card{i}", "children": kids})
    return {"version": "1.1", "design_id": "pizza", "title": "Pizza", "theme_palette": ["#FFFFFF", "#1B1B1F"],
            "fonts": {"heading": {"family": "Outfit"}},
            "root": {"node_type": "container", "id": "root", "layout": "grid", "columns": columns,
                     "style": {"background_color": "#FFFFFF"}, "children": cards}}


def test_new_content_is_checked_for_structure_fonts_and_photos_not_words(listing_spec):
    ast = DesignDocumentASTV11.model_validate(_pizza_page()).model_dump()
    _, as_copy = check_fidelity(ast, listing_spec)
    assert any(f.startswith("FIDELITY_TEXT_MISSING") for f in as_copy)
    scores, as_template = check_fidelity(ast, listing_spec, content="new")
    assert as_template == [] and scores["layout_matches"] == scores["layout_regions"] == 1
    assert scores["images_matched"] == 4 and scores["fonts_set"]
    wrong = DesignDocumentASTV11.model_validate(_pizza_page(columns=3, photos=2)).model_dump()
    _, findings = check_fidelity(wrong, listing_spec, content="new")
    assert any(f.startswith("FIDELITY_LAYOUT_STACKED") for f in findings)
    assert any(f.startswith("FIDELITY_IMAGE_MISSING") for f in findings)
    assert not any(f.startswith("FIDELITY_TEXT_MISSING") for f in findings)


@pytest.mark.asyncio
@pytest.mark.parametrize("query, calls, use", [
    ("Design a page like this for pizzas and Asian fusion dishes", 2, "adapt"),     # no repair for burger words
    ("Recreate this screenshot faithfully", 3, "replicate"),                       # copy: missing words are repaired
])
async def test_template_requests_skip_the_word_repair(monkeypatch, listing_spec, query, calls, use):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    page = json.dumps(_pizza_page())
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [listing_spec.model_dump_json(), page, page]
    report = {}
    await execute_design_canvas_harness(runner, query, "create_from_vision", _design_reference_image(SHOT),
                                        report=report)
    assert runner.run_json_completion.await_count == calls
    assert report["reference_use"] == use
    build = runner.run_json_completion.await_args_list[1].kwargs["messages"][1]["content"]
    assert ("layout and style template" in build.replace("\n", " ")) is (use == "adapt")
    if use == "adapt":
        assert not [f for f in report["fidelity"]["findings"] if f.startswith("FIDELITY_TEXT_MISSING")]


def test_text_placement_slips_are_normalised_not_sent_back(listing_spec):
    # Live 2026-11-19: one region listed "c9l", a text id that did not exist, and the whole description took a
    # 38 s repair call. Unknown ids are dropped, duplicates keep their first region, unplaced texts get a region.
    data = json.loads(listing_spec.model_dump_json())
    data["regions"][1]["text_ids"] += ["c9l", "n2"]                  # unknown id, and n2 listed twice
    data["regions"][4]["text_ids"] = []                              # n3 no longer in any region
    spec = ReferenceSpec.model_validate(data)
    by_id = {r.id: r.text_ids for r in spec.regions}
    assert by_id["c0"] == ["n0", "n2"] and by_id["c2"] == [] and by_id["unplaced"] == ["n3"]


# ------------------------------------------------------------------------------------------- description coverage

def _page_png(rows_to=950):
    """A white 1000x1000 page with text-like bars from the top down to ``rows_to``."""
    import io

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, rows_to, 70):
        draw.rectangle([40, y, 700, y + 22], fill="#222222")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _described(bottom, region_box=True, texts=3):
    entries = [{"id": f"t{i}", "text": f"Line {i}", "role": "heading", "approx_size_px": 22, "color_hex": "#222222",
                "font_class": "grotesk_sans", "weight": "bold", "uppercase": False, "letter_spacing": "normal",
                "box": [40, 40 + 70 * i, 700, 62 + 70 * i]} for i in range(texts)]
    region = {"id": "page", "arrangement": "stack", "text_ids": [e["id"] for e in entries]}
    if region_box:
        region["box"] = [0, 0, 1000, bottom]
    return ReferenceSpec.model_validate({"canvas_background_hex": "#FFFFFF", "font_guess": {}, "regions": [region],
                                         "texts": entries})


def test_coverage_flags_a_description_that_stops_after_the_header():
    from src.design_canvas.reference import description_coverage
    page = _page_png()
    short = description_coverage(_described(200), page)
    assert not short["complete"] and short["uncovered"] > 0.5 and short["basis"] == "regions"
    assert description_coverage(_described(980), page)["complete"]
    # Only large texts are boxed: unboxed lower texts are not evidence of a short description, so the bar is higher.
    texts_only = description_coverage(_described(0, region_box=False, texts=8), page)
    assert texts_only["basis"] == "texts" and texts_only["complete"]
    assert not description_coverage(_described(0, region_box=False, texts=2), page)["complete"]
    blank = _page_png(rows_to=0)
    assert description_coverage(_described(200), blank) is None
    no_boxes = ReferenceSpec.model_validate({**json.loads(_described(200).model_dump_json()), "regions": [
        {"id": "page", "arrangement": "stack", "text_ids": ["t0", "t1", "t2"]}], "texts": [
        {**t, "box": None} for t in json.loads(_described(200).model_dump_json())["texts"]]})
    assert description_coverage(no_boxes, _page_png()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("second, analysis, calls", [
    ("complete", "pass_after_repair", 3),     # asked once for the whole page, got it
    ("short", "partial", 3),                  # still short: used, and the result says what went unchecked
    ("invalid", "partial", 3),                # the retry broke: the short description is still used
])
async def test_short_description_is_asked_for_once_then_reported(monkeypatch, second, analysis, calls):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_vision_two_pass", True)
    short, full = _described(200).model_dump_json(), _described(980).model_dump_json()
    answer = {"complete": full, "short": short, "invalid": "{not json"}[second]
    page = json.dumps({"version": "1.1", "design_id": "p", "title": "P", "theme_palette": ["#FFFFFF", "#222222"],
                       "fonts": {"body": {"family": "Inter"}},
                       "root": {"node_type": "container", "id": "root", "style": {"background_color": "#FFFFFF"},
                                "children": [{"node_type": "text", "id": f"l{i}", "tag": "p",
                                              "spans": [{"text": f"Line {i}", "color": "#222222"}]} for i in range(3)]}})
    runner = AsyncMock()
    runner.run_json_completion.side_effect = [short, answer, page]
    report = {}
    await execute_design_canvas_harness(runner, "Recreate this screenshot", "create_from_vision", _page_png(),
                                        report=report)
    assert runner.run_json_completion.await_count == calls
    retry = runner.run_json_completion.await_args_list[1].kwargs["messages"][-1]["content"]
    assert "<reference_spec_incomplete>" in retry and "reaches y=202 px" in retry   # lowest box: text t2
    assert report["reference_analysis"] == analysis
    partial = [f for f in report["fidelity"]["findings"] if f.startswith("FIDELITY_ANALYSIS_PARTIAL")]
    assert bool(partial) is (analysis == "partial")
