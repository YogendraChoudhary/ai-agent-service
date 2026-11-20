"""
Editing a selected photo of a design: the image flow redraws only the selected photos, from a description rewritten
from the user's words ("replace this image with a pizza"), and size/crop/position wording stays a layout edit.
Before 2026-11-20 a selected photo's request redrew every photo of the design from its old prompt.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.design_canvas import images as design_images
from src.design_canvas.schema_v11 import DesignDocumentASTV11
from src.design_intent import detect_design_intent
from src.request_plan import plan_design_request
from tests.design.test_design_images import CDN, _card, _fake_image_tool_builder, _image, _listing, _sse

SELECT = "\n\n(Apply this to the selected section, id {}. Keep everything else unchanged.)"


def _design():
    """Three cards with finished photos and a header without one."""
    header = {"node_type": "container", "id": "hdr", "children": [
        {"node_type": "text", "id": "title", "tag": "h1", "spans": [{"text": "Burgers", "color": "#1B1B1F"}]}]}
    cards = [_card(i, name, _image(i, src=f"{CDN}/u/ai-images/old{i}.png")) for i, name in
             ((1, "KFC"), (2, "BK"), (3, "Shack"))]
    return _listing([header] + cards)


def _sub_mode(query, ast=None):
    return detect_design_intent(query, operation_type="edit_design", has_existing_design_ast=True,
                                existing_design_ast=_design() if ast is None else ast).sub_mode


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls, self.used = reply, [], None

    def __call__(self, system, messages, spec):
        self.calls.append(json.loads(messages[0]["content"].split("Slots: ", 1)[1]))
        self.used = ("claude", "claude-opus-5-5")
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


class TestRouting:
    @pytest.mark.parametrize("query, target", [
        ("replace this image with a pizza", "img1"), ("make it a pizza", "img1"), ("brighter please", "img1"),
        ("change this image to a pizza", "card1"), ("make the photo brighter", "card1"),
        ("use a different picture here", "card1"),
    ])
    def test_selected_photo_requests_go_to_the_image_flow(self, query, target):
        assert _sub_mode(query + SELECT.format(target)) == "fill_design_images"

    @pytest.mark.parametrize("query, target", [
        ("make the image bigger", "img1"), ("crop this image", "img1"), ("round the corners of this photo", "img1"),
        ("move the photo to the right", "card1"), ("remove this image", "img1"),
        ("add a photo here", "hdr"),                                         # a section with no photo slot: layout edit
    ])
    def test_layout_wording_and_photo_less_sections_stay_layout_edits(self, query, target):
        assert _sub_mode(query + SELECT.format(target)) == "edit_existing_ast"

    def test_without_a_selection_the_old_rules_hold(self):
        assert _sub_mode("add real photos") == "fill_design_images"
        assert _sub_mode("change the images to pizza ones") == "fill_design_images"
        assert _sub_mode("what photos are these?") != "fill_design_images"
        # callers that only know a design is attached (no design given) are unchanged
        assert detect_design_intent("add real photos", "edit_design", False, True).sub_mode == "fill_design_images"

    def test_the_price_follows_the_same_decision(self):
        plan = plan_design_request("make the image bigger" + SELECT.format("img1"), "edit_design", False, True, _design())
        assert plan.design_sub_mode == "edit_existing_ast"
        plan = plan_design_request("make it a pizza" + SELECT.format("img1"), "edit_design", False, True, _design())
        assert plan.design_sub_mode == "fill_design_images"


class TestTargets:
    def test_only_the_selected_photo_even_when_it_has_one(self):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        targets, summary = design_images.fill_targets(ast, "make it a pizza" + SELECT.format("img2"), cap=12)
        assert [t["id"] for t in targets] == ["img2"] and summary["selected"] == "img2"

    def test_a_selected_card_means_its_photos(self):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        targets, _ = design_images.fill_targets(ast, "brighter" + SELECT.format("card3"), cap=12)
        assert [t["id"] for t in targets] == ["img3"]

    def test_without_a_selection_nothing_changes(self):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        assert [t["id"] for t in design_images.fill_targets(ast, "replace the photos", cap=12)[0]] == ["img1", "img2", "img3"]
        assert design_images.fill_targets(ast, "add real photos", cap=12)[0] == []


class TestPromptRewrite:
    @pytest.mark.parametrize("query, rewrite", [
        ("add real photos", False), ("fill the photo slots", False), ("generate images", False),
        ("replace this image with a pizza", True), ("make the photos brighter", True), ("regenerate this photo", True),
        ("add photos of pizzas", True),
    ])
    def test_only_a_plain_fill_keeps_the_old_descriptions(self, query, rewrite):
        assert design_images.needs_prompt_rewrite(query + SELECT.format("img1")) is rewrite

    def test_prompts_come_from_the_request_with_the_card_as_context(self):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        targets, _ = design_images.fill_targets(ast, "x" + SELECT.format("img1"), cap=12)
        llm = FakeLLM({"photos": {"img1": {"prompt": "A pepperoni pizza on a wooden board", "alt": "Pizza"}}})
        out = design_images.rewrite_photo_prompts(ast, targets, "replace this image with a pizza" + SELECT.format("img1"), llm)
        assert out == {"img1": {"prompt": "A pepperoni pizza on a wooden board", "alt": "Pizza"}}
        assert llm.calls[0] == [{"id": "img1", "alt": "Photo 1", "prompt": "a burger number 1", "context": ["KFC"]}]

    @pytest.mark.parametrize("reply", [{"not_photo_edit": True}, "nope", {"photos": {}},
                                       {"photos": {"img1": {"prompt": "x"}}}, {"photos": {"zz": {"prompt": "a pizza"}}}])
    def test_an_answer_that_does_not_hold_is_none(self, reply):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        targets, _ = design_images.fill_targets(ast, "x" + SELECT.format("img1"), cap=12)
        assert design_images.rewrite_photo_prompts(ast, targets, "make it a pizza", FakeLLM(reply)) is None


class TestSelectedPhotoEndToEnd(unittest.TestCase):
    """/agent/stream as the editor calls it: type edit_design, the design attached, a photo selected."""

    def setUp(self):
        self.record = []
        self.harness = AsyncMock(side_effect=AssertionError("no layout model call"))
        self.llm = FakeLLM({"photos": {"img2": {"prompt": "A pepperoni pizza on a wooden board", "alt": "Pizza"}}})
        meta = {"user_id": "user_123", "requires_user_api_keys": False, "user_keys": {}}
        self.patches = [
            patch("src.design_canvas.service.execute_design_canvas_harness", self.harness),
            patch.object(tpa, "_build_image_tool", _fake_image_tool_builder(self.record)),
            patch.object(tpa, "_design_text_llm", lambda *a, **k: self.llm),
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

    def _stream(self, query):
        response = self.client.post("/agent/stream", json={
            "type": "edit_design", "query": query, "model": "claude", "api_key": "lic_test_123",
            "existing_design_ast": _design()})
        self.assertEqual(response.status_code, 200)
        events = _sse(response.text)
        return {n: d for n, d in events if n}, events

    def test_one_photo_redrawn_with_the_new_subject(self):
        by_name, _ = self._stream("replace this image with a pizza" + SELECT.format("img2"))
        design = DesignDocumentASTV11.model_validate(by_name["design_ready"]).model_dump()
        slots = {n["id"]: n for n, _ in design_images.walk(design["root"]) if n["node_type"] == "image"}
        self.harness.assert_not_called()
        self.assertEqual(len(self.record), 1)                               # one photo, not all three
        self.assertTrue(self.record[0]["prompt"].startswith("A pepperoni pizza on a wooden board. Photograph."))
        self.assertEqual(slots["img2"]["prompt"], "A pepperoni pizza on a wooden board")
        self.assertEqual(slots["img2"]["alt"], "Pizza")
        self.assertTrue(slots["img2"]["placeholder_id"])
        for other in ("img1", "img3"):                                      # the other photos are untouched
            self.assertEqual(slots[other]["src"], f"{CDN}/u/ai-images/old{other[-1]}.png")
            self.assertIsNone(slots[other]["placeholder_id"])
        self.assertEqual(by_name["complete"]["images_started"], 1)

    def test_no_photo_is_made_when_the_description_cannot_be_written(self):
        self.llm.reply = {"not_photo_edit": True}
        by_name, _ = self._stream("replace this image with a pizza" + SELECT.format("img2"))
        self.assertEqual(self.record, [])
        self.assertEqual([w["kind"] for w in by_name["complete"]["warnings"]], ["photo_prompt_failed"])


# 2026-11-20 audit: a photo word anywhere sent text and layout edits to photo fill (nothing changed, "no photo was made").
class TestPhotoWordIsNotAPhotoRequest:
    @pytest.mark.parametrize("query, target", [
        ("Change the button text to 'Upload images'", None), ("Change the button text to Upload images", None),
        ("Change the subtitle to say we use real photos of our food", None),
        ("Use a different font for the image captions", None), ("Change the color of the text over the image to white", None),
        ("Add a 'View all photos' button", None), ("Add a gallery section with 6 images", None),
        ("add alt text to the images", None), ("add a photo gallery", None),
        ("add a caption below this", "img1"), ("make it a bit bigger", "img1"),
    ])
    def test_text_and_layout_edits_stay_edits(self, query, target):
        assert _sub_mode(query + (SELECT.format(target) if target else "")) == "edit_existing_ast"

    @pytest.mark.parametrize("query", [
        "Replace this with a burger on the left side of a wooden table", "Show a chef holding a burger in his right hand",
        "A close-up of a burger, centered on the plate", "Make it a cheeseburger with a shadow on the table",
        "Use a photo of a burger with fries on the right",
    ])
    def test_position_words_inside_a_description_still_mean_a_new_photo(self, query):
        assert _sub_mode(query + SELECT.format("img1")) == "fill_design_images"

    @pytest.mark.parametrize("query", ["make the photos brighter", "make the image a pizza", "Make the photo show a pizza",
                                       "Add photos of pizzas"])
    def test_changing_finished_photos_without_a_selection_redraws_them(self, query):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        assert _sub_mode(query) == "fill_design_images"
        assert [t["id"] for t in design_images.fill_targets(ast, query, cap=12)[0]] == ["img1", "img2", "img3"]

    def test_a_plain_fill_on_a_finished_design_still_redraws_nothing(self):
        ast = DesignDocumentASTV11.model_validate(_design()).model_dump()
        assert design_images.fill_targets(ast, "add real photos", cap=12)[0] == []


def test_selected_photo_scope_reads_large_or_malformed_raw_designs_cheaply():
    """It runs on every request before validation: no ancestor copies per node, no crash on odd input."""
    import tracemalloc
    from src.design_canvas.images import selected_photo_scope

    q = "make it a pizza (apply this to the selected section, id: sel)"
    leaves = ",".join('{"id":"l%d","node_type":"text"}' % i for i in range(20000))
    depth = 800
    body = json.loads('{"root":' + '{"id":"x","node_type":"frame","children":[' * depth + leaves + ']}' * depth + '}')
    tracemalloc.start()
    assert selected_photo_scope(body, q) == ("sel", None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20_000_000      # was ~150 MB with one ancestor list per node

    deep = '{"root":' + '{"children":[' * 5000 + ']}' * 5000 + '}'
    assert selected_photo_scope(deep, q) == ("sel", None)           # RecursionError in json.loads
    odd = {"root": {"id": "a", "node_type": "frame", "children": ["text", 3, None,
                                                                  {"id": "sel", "node_type": "image"}]}}
    assert selected_photo_scope(odd, q) == ("sel", "image")


def test_other_raw_design_readers_do_not_throw_on_deeply_nested_bodies():
    """Same rule as above for the chat reference and the wording fast lane (both read the raw design; RecursionError seen)."""
    import src.text_processing_agent as tpa
    from src.design_canvas.text_edit import try_text_edit
    from src.design_canvas.links import check_links

    deep = '{"root":' + '{"children":[' * 5000 + ']}' * 5000 + '}'
    q = "make this shorter\n\n(Apply this to the selected section, id header. Keep everything else unchanged.)"
    assert tpa._design_chat_reference(deep, "What font is this?") is None
    assert try_text_edit(deep, q, lambda *a: pytest.fail("no model call for an unreadable design")) is None
    nested = '{"root":' + '{"id":"x","children":[' * 900 + ']}' * 900 + '}'
    ast = DesignDocumentASTV11.model_validate(_design())
    assert check_links(ast, "make it blue", nested) == []
