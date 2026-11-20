"""Photo slots (ImageNode) and photo fill through the existing image flow (docs/design_image_fill_plan.md)."""

import json
import unittest
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
import src.text_processing_agent as tpa
from config import settings
from scripts.design_reference_render import render_html
from src.credit_manager import design_credits_scaled
from src.design_canvas import images as design_images
from src.design_canvas.reference import ReferenceSpec, check_fidelity, heal_reference_details
from src.design_canvas.schema_v11 import DesignDocumentASTV11, ImageNode, downgrade_to_v10
from src.design_intent import detect_design_intent
from src.request_plan import plan_design_request

CDN = settings.cdn_url.rstrip("/")


def _card(i, name, image=None):
    children = [image] if image else []
    children.append({"node_type": "text", "id": f"name{i}", "tag": "p", "spans": [{"text": name, "color": "#1B1B1F"}]})
    return {"node_type": "container", "id": f"card{i}", "children": children}


def _image(i, src=None, aspect="4:3"):
    return {"node_type": "image", "id": f"img{i}", "alt": f"Photo {i}", "prompt": f"a burger number {i}",
            "src": src, "aspect_ratio": aspect}


def _listing(cards):
    return {"version": "1.1", "design_id": "listing", "title": "Listing", "theme_palette": ["#FFFFFF", "#1B1B1F"],
            "root": {"node_type": "container", "id": "root", "layout": "grid", "columns": 2,
                     "style": {"background_color": "#FFFFFF"}, "children": cards}}


# ---------------------------------------------------------------------------------------------------- the schema


def test_image_src_must_be_on_the_cdn_and_placeholder_ids_are_uuids():
    ok = ImageNode(id="img1", alt="Burger", prompt="a burger", src=f"{CDN}/u/ai-images/a.png")
    assert ok.src.startswith(CDN)
    for bad in ("https://evil.example/a.png", "http://" + CDN.split("://")[1] + "/a.png", "javascript:alert(1)",
                f"{CDN}/a b.png"):
        with pytest.raises(ValidationError):
            ImageNode(id="img1", alt="Burger", prompt="a burger", src=bad)
    with pytest.raises(ValidationError):
        ImageNode(id="img1", alt="Burger", prompt="a burger", placeholder_id="not-a-uuid")
    assert ImageNode(id="img1", alt="B", prompt="a burger", src="").src is None


def test_slot_keeps_its_shape_and_generation_snaps_to_a_supported_aspect():
    from src.design_canvas.schema_v11 import generation_aspect
    assert ImageNode(id="img1", alt="B", prompt="a burger", aspect_ratio="140:93").aspect_ratio == "140:93"
    assert [generation_aspect(a) for a in ("140:93", "16:10", "1:1", "2:3", "21:9")] == [
        "4:3", "16:9", "1:1", "3:4", "16:9"]
    for bad in ("0:3", "3:2:1", "abc", "10:1"):
        with pytest.raises(ValidationError):
            ImageNode(id="img1", alt="B", prompt="a burger", aspect_ratio=bad)


def test_image_downgrades_to_a_placeholder_rectangle_of_the_same_shape():
    ast = DesignDocumentASTV11.model_validate(_listing([_card(1, "KFC", _image(1, aspect="16:9"))]))
    legacy = downgrade_to_v10(ast).model_dump()
    slot = legacy["root"]["children"][0]["children"][0]
    assert slot["node_type"] == "vector_illustration" and slot["id"] == "img1"
    assert (slot["width_px"], slot["height_px"]) == (320, 180)


# ---------------------------------------------------------------------------------------------------- routing


@pytest.mark.parametrize("query", ["add real photos", "please fill the images", "use real pictures in the cards",
                                   "generate images for each card", "replace the photos with new ones",
                                   "can you add photos"])
def test_photo_requests_on_a_selected_design_route_to_fill(query):
    assert detect_design_intent(query, "general", False, True).sub_mode == "fill_design_images"
    assert detect_design_intent(query, "edit_design", False, True).sub_mode == "fill_design_images"


@pytest.mark.parametrize("query, has_ast", [("make the button green", True), ("add a pricing section", True),
                                           ("what photos are these?", True), ("add real photos", False)])
def test_other_requests_do_not_fill(query, has_ast):
    assert detect_design_intent(query, "general", False, has_ast).sub_mode != "fill_design_images"


def test_fill_is_an_ordinary_edit_while_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "design_canvas_enabled", True)
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_image_fill_enabled", False)
    assert plan_design_request("add real photos", "general", False, True).design_sub_mode == "edit_existing_ast"
    monkeypatch.setattr(settings, "design_image_fill_enabled", True)
    assert plan_design_request("add real photos", "general", False, True).design_sub_mode == "fill_design_images"


def test_fill_price_is_images_only_unless_a_conversion_edit_runs():
    assert design_credits_scaled("fill_design_images", has_image_slots=True) == 0
    assert design_credits_scaled("fill_design_images", has_image_slots=False) == design_credits_scaled("edit_existing_ast")
    assert design_images.has_image_slots(json.dumps(_listing([_card(1, "KFC", _image(1))])))
    assert not design_images.has_image_slots(_listing([_card(1, "KFC")]))
    assert not design_images.has_image_slots("not json")
    # An image node the schema drops (here under a text node) must not make the conversion edit free.
    hidden = _listing([_card(1, "KFC")])
    text = next(n for n, _ in design_images.walk(hidden["root"]) if n["node_type"] == "text")
    text["children"] = [_image(9)]
    assert not design_images.has_image_slots(json.dumps(hidden))


# ---------------------------------------------------------------------------------------------------- fill targets


def test_fill_targets_empty_slots_replace_named_card_and_cap():
    ast = _listing([_card(1, "KFC", _image(1)), _card(2, "Burger King", _image(2, src=f"{CDN}/u/ai-images/b.png")),
                    _card(3, "McDonald's", _image(3))])
    targets, summary = design_images.fill_targets(ast, "add real photos", cap=12)
    assert [n["id"] for n in targets] == ["img1", "img3"] and summary["slots"] == 3 and not summary["replace"]
    targets, summary = design_images.fill_targets(ast, "replace the photos", cap=12)
    assert [n["id"] for n in targets] == ["img1", "img2", "img3"] and summary["replace"]
    targets, summary = design_images.fill_targets(ast, "add a real photo only for the KFC card", cap=12)
    assert [n["id"] for n in targets] == ["img1"] and summary["named"]
    targets, summary = design_images.fill_targets(ast, "regenerate the burger king picture", cap=12)
    assert [n["id"] for n in targets] == ["img2"]
    targets, summary = design_images.fill_targets(ast, "replace all images", cap=2)
    assert len(targets) == 2 and summary["over_cap"] == 1


def test_prompt_placeholder_parsing_and_applying_finished_photos():
    assert design_images.photo_prompt({"prompt": "a crispy burger."}).startswith("a crispy burger. Photograph.")
    pid = str(uuid.uuid4())
    assert design_images.placeholder_id(f"Image URL: placeholder::{pid}") == pid
    assert design_images.placeholder_id("Image generation skipped: insufficient credits") is None
    done, failed = str(uuid.uuid4()), str(uuid.uuid4())
    ast = _listing([_card(1, "A", {**_image(1), "placeholder_id": done}),
                    _card(2, "B", {**_image(2, src=f"{CDN}/u/old.png"), "placeholder_id": failed})])
    design_images.apply_finished_images(ast, {done: {"status": "done", "cdn_url": f"{CDN}/u/new.png"},
                                              failed: {"status": "failed", "error": "x"}})
    first, second = (c["children"][0] for c in ast["root"]["children"])
    assert first["src"] == f"{CDN}/u/new.png" and first["placeholder_id"] is None
    assert second["src"] == f"{CDN}/u/old.png" and second["placeholder_id"] is None
    DesignDocumentASTV11.model_validate(ast)


# ---------------------------------------------------------------------------------------------------- recreation


def _spec_with_images(images):
    texts = [{"id": "n1", "text": "KFC", "role": "subheading", "approx_size_px": 17, "color_hex": "#1B1B1F",
              "font_class": "geometric_sans", "weight": "semibold", "uppercase": False, "letter_spacing": "normal"}]
    regions = [{"id": "grid", "arrangement": "grid", "columns": 2, "text_ids": []},
               {"id": "c1", "parent_id": "grid", "arrangement": "stack", "text_ids": ["n1"]}]
    return ReferenceSpec.model_validate({"canvas_background_hex": "#FFFFFF", "font_guess": {}, "regions": regions,
                                         "texts": texts, "images": images})


def test_spec_images_normalise_aspect_and_drop_unknown_regions():
    spec = _spec_with_images([{"id": "img1", "region_id": "c1", "description": "a burger", "aspect": "280:186"},
                              {"id": "img2", "region_id": "nowhere", "description": "fries", "aspect": "junk"}])
    assert [(i.aspect, i.region_id) for i in spec.images] == [("140:93", "c1"), ("4:3", None)]   # no box: kept / 4:3
    boxed = _spec_with_images([{"id": "img1", "description": "a burger", "aspect": "16:9", "box": [0, 0, 280, 186]}])
    assert boxed.images[0].aspect == "140:93"                      # the box wins over the model's guess
    with pytest.raises(ValidationError):
        _spec_with_images([{"id": "img1", "description": "a"}, {"id": "img1", "description": "b c d"}])


def test_drawn_photo_in_its_card_becomes_a_photo_slot_with_no_model_call():
    stand_in = {"node_type": "vector_illustration", "id": "kfc_art", "layers": [
        {"layer_id": "bun", "label": "Bun", "svg_element": "ellipse",
         "attributes": {"cx": 100, "cy": 90, "rx": 60, "ry": 40}, "fill_color": "#D9A066"}]}
    ast = DesignDocumentASTV11.model_validate(_listing([_card(1, "KFC", stand_in), _card(2, "Other")])).model_dump()
    spec = _spec_with_images([{"id": "img1", "region_id": "c1", "box": [10, 10, 290, 196],
                               "description": "crispy chicken burger on a red background, studio shot",
                               "aspect": "3:2"}])
    _, before = check_fidelity(ast, spec)
    assert any(f.startswith("FIDELITY_IMAGE_MISSING") for f in before)
    changes = heal_reference_details(ast, spec)
    slot = DesignDocumentASTV11.model_validate(ast).root.children[0].children[0]
    assert slot.node_type == "image" and slot.id == "kfc_art" and slot.aspect_ratio == "140:93"   # 280x186 box
    assert slot.prompt.startswith("crispy chicken burger") and slot.alt == "crispy chicken burger on a red background"
    assert slot.source_box == [10, 10, 290, 196] and slot.src is None and changes
    scores, after = check_fidelity(ast, spec)
    assert scores["images_matched"] == scores["images_expected"] == 1
    assert not any(f.startswith("FIDELITY_IMAGE_MISSING") for f in after)


def test_renderer_draws_photos_and_labelled_placeholders():
    ast = _listing([_card(1, "A", _image(1, src=f"{CDN}/u/ai-images/a.png")),
                    _card(2, "B", {**_image(2), "placeholder_id": str(uuid.uuid4())})])
    html = render_html(DesignDocumentASTV11.model_validate(ast).model_dump(), 800)
    assert f'src="{CDN}/u/ai-images/a.png"' in html and "aspect-ratio:4 / 3" in html and "object-fit:cover" in html
    assert 'data-generating="true"' in html and ">Photo 2</div>" in html


# ---------------------------------------------------------------------------------------------------- end to end


def _fake_image_tool_builder(record):
    """Stands in for _build_image_tool: same arguments, same placeholder answer, image finished at once."""
    def build(model_provider, api_key, pending_images, request_id, user_id, credit_manager, is_byok, image_ctx, budget):
        def tool(prompt):
            if not budget.take():
                return "ERROR: image limit reached."
            pid = str(uuid.uuid4())
            record.append({"prompt": prompt, "aspect": image_ctx["explicit_params"]["aspect_ratio"],
                           "user_query": image_ctx["user_query"]})
            pending_images[pid] = {"status": "done", "cdn_url": f"{CDN}/u/ai-images/{pid}.png", "error": None,
                                   "prompt": prompt, "width": 1024, "height": 768, "format": "png",
                                   "provider": "gemini", "model": "gemini-3.1-flash-lite-image",
                                   "credits_scaled": 50, "reserved_scaled": 0}
            callback = pending_images.get("__completion_callback__")
            if callback:
                callback()
            return f"Image URL: placeholder::{pid}"
        return tool
    return build


def _sse(text):
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except ValueError:
                data = line[6:]
            events.append((name or (data.get("type") if isinstance(data, dict) else None), data))
            name = None
    return events


class TestPhotoFillEndToEnd(unittest.TestCase):
    def setUp(self):
        self.record = []
        self.converted = DesignDocumentASTV11.model_validate(_listing([_card(1, "KFC", _image(1)), _card(2, "BK")]))
        self.harness = AsyncMock(return_value=self.converted)
        meta = {"user_id": "user_123", "requires_user_api_keys": False, "user_keys": {}}
        self.patches = [
            patch("src.design_canvas.service.execute_design_canvas_harness", self.harness),
            patch.object(tpa, "_build_image_tool", _fake_image_tool_builder(self.record)),
            patch.object(settings, "design_canvas_enabled", True),
            patch.object(settings, "design_schema_v11_enabled", True),
            patch.object(settings, "design_image_fill_enabled", True),
            patch.object(settings, "task_routing_enabled", False),
            patch.object(settings, "provider_fallback_enabled", False),
            patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("fake_key", meta))),
            patch.object(api, "credit_manager", None),
            patch.object(api, "CREDIT_PRECHECK_ENABLED", False),
            patch.object(api, "fire_credit_deduction"),
            patch.object(api, "fire_prompt_history_save"),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _stream(self, ast, query="add real photos"):
        response = self.client.post("/agent/stream", json={
            "type": "general", "query": query, "model": "claude", "api_key": "lic_test_123",
            "existing_design_ast": ast})
        self.assertEqual(response.status_code, 200)
        return _sse(response.text)

    def test_stream_returns_placeholders_then_one_image_ready_per_photo(self):
        ast = _listing([_card(1, "KFC", _image(1, aspect="16:9")), _card(2, "BK", _image(2, aspect="140:93")),
                        _card(3, "Done", _image(3, src=f"{CDN}/u/ai-images/kept.png"))])
        events = self._stream(ast)
        names = [n for n, _ in events]
        self.assertIn("design_ready", names)
        design = next(d for n, d in events if n == "design_ready")
        complete = next(d for n, d in events if n == "complete")
        ready = [d for n, d in events if n == "image_ready"]
        self.harness.assert_not_called()                                  # no layout model call
        slots = {c["children"][0]["id"]: c["children"][0] for c in design["root"]["children"]}
        self.assertTrue(slots["img1"]["placeholder_id"] and slots["img2"]["placeholder_id"])
        self.assertIsNone(slots["img3"]["placeholder_id"])                 # already had a photo
        self.assertEqual(complete["sub_mode"], "fill_design_images")
        self.assertEqual(complete["images_started"], 2)
        self.assertEqual({d["placeholder_id"] for d in ready},
                         {slots["img1"]["placeholder_id"], slots["img2"]["placeholder_id"]})
        self.assertTrue(all(d["url"].startswith(CDN) for d in ready))
        self.assertLess(names.index("design_ready"), names.index("image_ready"))
        self.assertEqual([r["aspect"] for r in self.record], ["16:9", "4:3"])      # each slot's own shape
        self.assertTrue(self.record[0]["prompt"].startswith("a burger number 1. Photograph."))
        self.assertEqual(self.record[0]["user_query"], "add real photos")

    def test_design_without_photo_slots_is_converted_once_then_filled(self):
        drawn = _listing([_card(1, "KFC", {"node_type": "vector_illustration", "id": "img1", "layers": [
            {"layer_id": "bun", "label": "Bun", "svg_element": "circle", "attributes": {"cx": 10, "cy": 10, "r": 5},
             "fill_color": "#D9A066"}]}), _card(2, "BK")])
        events = self._stream(drawn)
        self.harness.assert_awaited_once()
        kwargs = self.harness.await_args.kwargs
        self.assertEqual(kwargs["sub_mode"], "edit_existing_ast")
        self.assertEqual(kwargs["user_query"], design_images.CONVERT_INSTRUCTION)
        complete = next(d for n, d in events if n == "complete")
        self.assertTrue(complete["converted"])
        self.assertEqual(len([1 for n, _ in events if n == "image_ready"]), 1)

    def test_nothing_to_fill_is_reported_not_silent(self):
        full = _listing([_card(1, "KFC", _image(1, src=f"{CDN}/u/ai-images/a.png"))])
        events = self._stream(full)
        complete = next(d for n, d in events if n == "complete")
        self.assertEqual(complete["images_started"], 0)
        self.assertEqual(complete["warnings"][0]["kind"], "no_image_slots")
        self.assertFalse([1 for n, _ in events if n == "image_ready"])

    def test_process_endpoint_returns_the_design_with_photos_in_place(self):
        ast = _listing([_card(1, "KFC", _image(1))])
        response = self.client.post("/agent/process", json={
            "type": "general", "query": "add real photos", "model": "claude", "api_key": "lic_test_123",
            "existing_design_ast": ast})
        self.assertEqual(response.status_code, 200, response.text)
        slot = response.json()["design"]["root"]["children"][0]["children"][0]
        self.assertTrue(slot["src"].startswith(CDN) and slot["placeholder_id"] is None)


def test_near_miss_text_and_button_sizes_are_clamped_not_repaired():
    # Live 2026-11-19: a 9 px "NEW" badge failed the >= 10 px rule and cost a 44 s repair call.
    ast = _listing([_card(1, "KFC", _image(1))])
    ast["root"]["children"][0]["children"][1]["font_size_px"] = 9
    ast["root"]["children"].append({"node_type": "button", "id": "chip", "label": "NEW", "text_color": "#FFFFFF",
                                    "font_size_px": 40})
    parsed = DesignDocumentASTV11.model_validate(ast)
    assert parsed.root.children[0].children[1].font_size_px == 10 and parsed.root.children[1].font_size_px == 32
    with pytest.raises(ValidationError):
        DesignDocumentASTV11.model_validate({**ast, "root": {**ast["root"], "children": [
            {"node_type": "text", "id": "t", "tag": "p", "font_size_px": "big", "spans": [{"text": "x"}]}]}})
