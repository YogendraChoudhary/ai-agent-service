"""
Wording-only design edits (src/design_canvas/text_edit.py): a selected section's text is changed from one small call and
patched into the design in code, instead of the model rewriting the whole design. Live 2026-11-20 a one-word rename on a
43 KB design ran for minutes on the full path.
"""

from __future__ import annotations

import json

import pytest

import src.text_processing_agent as tpa
from config import settings
from src.design_canvas.patch import compact_design_json
from src.design_canvas.schema_v11 import DesignDocumentASTV11
from src.design_canvas.text_edit import split_selection, try_text_edit, wants_text_lane
from tests.design.test_design_routing import _events, _stream, edges  # noqa: F401  (edges is a fixture)

LIVE_QUERY = "change the name to Pizza here\n\n(Apply this to the selected section, id header. Keep everything else unchanged.)"

DESIGN = {
    "block_type": "eddyter_design_canvas",
    "version": "1.1",
    "design_id": "dsn_burger_listing",
    "title": "Burger Restaurants",
    "theme_palette": ["#FFFFFF", "#02060C", "#1E8B36"],
    "fonts": {"display": {"family": "Geist"}, "body": {"family": "Outfit"}, "heading": None, "mono": None},
    "root": {
        "node_type": "container", "id": "root", "layout": "flex_col", "align_items": "stretch",
        "style": {"background_color": "#FFFFFF"},
        "children": [
            {"node_type": "container", "id": "header", "children": [
                {"node_type": "text", "id": "t1", "tag": "h1", "font_size_px": 40, "font_role": "display",
                 "spans": [{"text": "Burger", "color": "#02060C", "font_weight": "bold"}]},
                {"node_type": "text", "id": "t2", "tag": "p",
                 "spans": [{"text": "Satisfy your cravings with these ", "color": "#3D4046"},
                           {"text": "fresh and flavoursome", "color": "#1E8B36", "font_weight": "semibold"},
                           {"text": " burgers.", "color": "#3D4046"}]},
            ]},
            {"node_type": "container", "id": "cards", "layout": "grid", "columns": 2, "children": [
                {"node_type": "text", "id": "c1", "tag": "h3", "spans": [{"text": "Burger King", "color": "#02060C"}]},
                {"node_type": "button", "id": "b1", "label": "Order now", "text_color": "#FFFFFF",
                 "href": "https://example.com/order", "style": {"background_color": "#1E8B36"}},
            ]},
        ],
    },
}


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.used = None

    def __call__(self, system, messages, spec):
        self.calls.append(json.loads(messages[0]["content"].split("Items: ", 1)[1]))
        self.used = ("openai", "gpt-6.1-terra")
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def _design_json():
    return DesignDocumentASTV11.model_validate(DESIGN).model_dump_json()


def _node(ast, node_id):
    def walk(node):
        if node["id"] == node_id:
            return node
        for child in node.get("children") or []:
            found = walk(child)
            if found:
                return found
    return walk(ast.model_dump()["root"])


class TestSelectionAndWording:
    def test_the_editors_selection_sentence_is_split_off(self):
        assert split_selection(LIVE_QUERY) == ("change the name to Pizza here", "header")
        assert split_selection("make it shorter") == ("make it shorter", None)

    @pytest.mark.parametrize("instruction", [
        "make this shorter", "change the name to Pizza here", "remove the word fresh", "translate to French",
        "say 'Bold red flavours' instead", "more playful",
    ])
    def test_wording_requests_on_a_selection_take_the_text_lane(self, instruction):
        assert wants_text_lane(instruction, has_selection=True)

    @pytest.mark.parametrize("instruction", [
        "make it red", "greener please", "bigger font", "add a button below", "remove the second card",
        "center this", "use a dark background", "add a photo", "more padding",
    ])
    def test_style_and_structure_requests_do_not(self, instruction):
        assert not wants_text_lane(instruction, has_selection=True)

    def test_without_a_selection_the_wording_must_name_text(self):
        assert wants_text_lane("change the title to Pizza Place", has_selection=False)
        assert not wants_text_lane("more modern please", has_selection=False)


class TestTryTextEdit:
    def test_rename_in_the_selected_section_only(self):
        llm = FakeLLM({"texts": {"t1": ["Pizza"]}})
        result = try_text_edit(_design_json(), LIVE_QUERY, llm)
        assert result is not None and result.changed_ids == ["t1"] and result.scope_id == "header"
        assert [item["id"] for item in llm.calls[0]] == ["t1", "t2"]          # only the selected section is sent
        t1 = _node(result.ast, "t1")
        assert t1["spans"][0] == {**_node(DesignDocumentASTV11.model_validate(DESIGN), "t1")["spans"][0], "text": "Pizza"}
        assert _node(result.ast, "c1")["spans"][0]["text"] == "Burger King"   # outside the selection: untouched
        before = DesignDocumentASTV11.model_validate(DESIGN).model_dump()
        after = result.ast.model_dump()
        after["root"]["children"][0]["children"][0]["spans"][0]["text"] = "Burger"
        assert after == before                                                # nothing else changed

    def test_runs_keep_their_colours_and_extra_runs_join_the_last(self):
        query = "make this shorter\n\n(Apply this to the selected section, id t2. Keep everything else unchanged.)"
        result = try_text_edit(_design_json(), query, FakeLLM({"texts": {"t2": ["Try our ", "fresh", " burgers", "."]}}))
        spans = _node(result.ast, "t2")["spans"]
        assert [s["text"] for s in spans] == ["Try our ", "fresh", " burgers."]
        assert [s["color"] for s in spans] == ["#3D4046", "#1E8B36", "#3D4046"]

    def test_button_label_changes_and_its_link_stays(self):
        query = "say Order pizza\n\n(Apply this to the selected section, id cards. Keep everything else unchanged.)"
        result = try_text_edit(_design_json(), query, FakeLLM({"texts": {"b1": ["Order pizza"]}}))
        button = _node(result.ast, "b1")
        assert button["label"] == "Order pizza" and button["href"] == "https://example.com/order"

    @pytest.mark.parametrize("reply", [
        {"not_text_edit": True}, "not json", {"texts": {"zz": ["x"]}}, {"texts": {"t1": [""]}},
        {"texts": {"t1": ["Burger"]}}, {"texts": {"t1": ["x" * 5000]}}, {"texts": []},
    ])
    def test_any_doubt_means_the_full_path(self, reply):
        assert try_text_edit(_design_json(), LIVE_QUERY, FakeLLM(reply)) is None

    def test_style_wording_makes_no_call(self):
        llm = FakeLLM({"texts": {}})
        query = "make it red\n\n(Apply this to the selected section, id header. Keep everything else unchanged.)"
        assert try_text_edit(_design_json(), query, llm) is None and llm.calls == []

    def test_unknown_selection_makes_no_call(self):
        llm = FakeLLM({"texts": {}})
        query = "make this shorter\n\n(Apply this to the selected section, id nope. Keep everything else unchanged.)"
        assert try_text_edit(_design_json(), query, llm) is None and llm.calls == []


class TestCompactDesign:
    def test_lossless_and_much_smaller(self):
        full = _design_json()
        compact = compact_design_json(full)
        assert len(compact) < len(full) / 2
        assert DesignDocumentASTV11.model_validate_json(compact) == DesignDocumentASTV11.model_validate_json(full)
        assert json.loads(compact)["root"]["children"][0]["node_type"] == "container"
        assert "null" not in compact

    def test_unparseable_input_is_returned_unchanged(self):
        assert compact_design_json("{not json") == "{not json"


class TestOnTheRoutingPath:
    @pytest.fixture(autouse=True)
    def _v11(self, monkeypatch):
        monkeypatch.setattr(settings, "design_schema_v11_enabled", True)

    def test_rename_never_calls_the_full_design_model(self, edges, monkeypatch):
        llm = FakeLLM({"texts": {"t1": ["Pizza"]}})
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: llm)
        chunks = _stream(LIVE_QUERY, operation_type="edit_design", existing_design_ast=DESIGN)
        events = _events(chunks)
        assert edges["providers"] == []                                       # no full design rewrite
        assert _node(DesignDocumentASTV11.model_validate(events["design_ready"]), "t1")["spans"][0]["text"] == "Pizza"
        assert events["complete"]["edit_path"] == "text_only"

    def test_a_declined_edit_runs_the_full_path_once(self, edges, monkeypatch):
        monkeypatch.setattr(settings, "design_section_edit_enabled", False)
        llm = FakeLLM({"not_text_edit": True})
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: llm)
        monkeypatch.setattr(tpa, "_run_design_json_completion", lambda *a, **k: (edges["providers"].append("x"), _design_json())[1])
        chunks = _stream("make it pop\n\n(Apply this to the selected section, id header. Keep everything else unchanged.)",
                         operation_type="edit_design", existing_design_ast=DESIGN)
        events = _events(chunks)
        assert len(llm.calls) == 1 and edges["providers"] == ["x"]
        assert events["complete"]["edit_path"] == "full"

    def test_text_lane_error_falls_back_to_the_full_path(self, edges, monkeypatch):
        monkeypatch.setattr(settings, "design_section_edit_enabled", False)
        def broken(*a, **k):
            raise RuntimeError("provider down")
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: broken)
        monkeypatch.setattr(tpa, "_run_design_json_completion", lambda *a, **k: (edges["providers"].append("x"), _design_json())[1])
        events = _events(_stream(LIVE_QUERY, operation_type="edit_design", existing_design_ast=DESIGN))
        assert edges["providers"] == ["x"] and "design_ready" in events

    def test_switched_off_means_the_full_path(self, edges, monkeypatch):
        monkeypatch.setattr(settings, "design_text_edit_enabled", False)
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: pytest.fail("text lane used"))
        monkeypatch.setattr(tpa, "_run_design_json_completion", lambda *a, **k: (edges["providers"].append("x"), _design_json())[1])
        _stream(LIVE_QUERY, operation_type="edit_design", existing_design_ast=DESIGN)
        assert edges["providers"] == ["x"]

    def test_cut_off_output_is_not_retried_on_every_provider(self, edges, monkeypatch):
        monkeypatch.setattr(settings, "design_text_edit_enabled", False)

        def truncated(messages, image_bytes, temperature, provider, key, request_id=""):
            edges["providers"].append(provider)
            raise ValueError("Design output truncated by model at max_tokens limit (8192)")
        monkeypatch.setattr(tpa, "_run_design_json_completion", truncated)
        with pytest.raises(ValueError, match="truncated"):
            _stream(LIVE_QUERY, operation_type="edit_design", existing_design_ast=DESIGN)
        assert len(edges["providers"]) == 1
