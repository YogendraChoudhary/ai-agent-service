"""
Section-only design edits (src/design_canvas/section_edit.py): a style or layout change of the selected section sends
the model that section alone and splices the answer back, instead of a rewrite of the whole design (~45 s live).
"""

from __future__ import annotations

import copy
import json
import re

import pytest

import src.text_processing_agent as tpa
from config import settings
from src.design_canvas.schema_v11 import DesignDocumentASTV11
from src.design_canvas.section_edit import prepare_section_edit, section_query, splice_section
from tests.design.test_design_routing import _events, _stream, edges  # noqa: F401  (edges is a fixture)
from tests.design.test_design_text_edit import DESIGN as SMALL_DESIGN, FakeLLM, _node

# A page whose header is a small part of it, as in the live burger design (header 1.7 KB of 43 KB).
DESIGN = copy.deepcopy(SMALL_DESIGN)
DESIGN["root"]["children"][1]["children"] += [
    {"node_type": "text", "id": f"c{i}", "tag": "h3", "spans": [{"text": f"Burger place number {i} on the high street",
                                                                "color": "#02060C"}]}
    for i in range(2, 20)]


def _design_json():
    return DesignDocumentASTV11.model_validate(DESIGN).model_dump_json()


SELECT = "\n\n(Apply this to the selected section, id {}. Keep everything else unchanged.)"


def _section_answer(scope_json, edit):
    section = json.loads(scope_json)
    edit(section["root"]["children"][0])
    return DesignDocumentASTV11.model_validate(section)


def _make_heading_green(header):
    header["children"][0]["spans"][0]["color"] = "#1E8B36"


class TestPrepare:
    def test_the_section_stands_alone_on_its_real_background(self):
        scope = prepare_section_edit(_design_json(), "make the heading green" + SELECT.format("header"))
        section = json.loads(scope.section_json)
        wrapper = section["root"]
        assert wrapper["id"] == scope.wrapper_id == "section_context"
        assert wrapper["style"]["background_color"] == "#FFFFFF"            # the root's, behind the header
        assert [child["id"] for child in wrapper["children"]] == ["header"]
        assert "cards" not in scope.section_json and "Burger King" not in scope.section_json
        assert scope.instruction == "make the heading green"

    @pytest.mark.parametrize("query", [
        "make it green",                                                     # no selection
        "make it green" + SELECT.format("root"),                             # the whole design
        "make it green" + SELECT.format("nope"),                             # not in the design
    ])
    def test_no_usable_selection(self, query):
        assert prepare_section_edit(_design_json(), query) is None

    def test_a_section_that_is_most_of_the_design_uses_the_full_path(self):
        assert prepare_section_edit(_design_json(), "make it green" + SELECT.format("header"), max_fraction=0.1) is None

    def test_the_stand_in_id_never_clashes(self):
        design = copy.deepcopy(DESIGN)
        design["root"]["children"][1]["id"] = "section_context"
        raw = DesignDocumentASTV11.model_validate(design).model_dump_json()
        assert prepare_section_edit(raw, "x" + SELECT.format("header")).wrapper_id == "section_context_2"

    def test_the_query_names_the_section(self):
        scope = prepare_section_edit(_design_json(), "more padding" + SELECT.format("header"))
        assert "`header`" in section_query(scope) and "`section_context`" in section_query(scope)


class TestSplice:
    def test_only_the_section_changes(self):
        scope = prepare_section_edit(_design_json(), "make the heading green" + SELECT.format("header"))
        result = splice_section(scope, _section_answer(scope.section_json, _make_heading_green))
        assert _node(result, "t1")["spans"][0]["color"] == "#1E8B36"
        before, after = DesignDocumentASTV11.model_validate(DESIGN).model_dump(), result.model_dump()
        after["root"]["children"][0]["children"][0]["spans"][0]["color"] = before["root"]["children"][0]["children"][0]["spans"][0]["color"]
        assert after == before

    def test_new_elements_next_to_the_section_are_kept(self):
        scope = prepare_section_edit(_design_json(), "add a divider below" + SELECT.format("header"))
        section = json.loads(scope.section_json)
        section["root"]["children"].append({"node_type": "divider", "id": "hdr_rule"})
        result = splice_section(scope, DesignDocumentASTV11.model_validate(section))
        assert [c["id"] for c in result.model_dump()["root"]["children"]] == ["header", "hdr_rule", "cards"]

    def test_a_new_id_that_exists_elsewhere_is_made_unique(self):
        scope = prepare_section_edit(_design_json(), "add a line" + SELECT.format("header"))
        section = json.loads(scope.section_json)
        section["root"]["children"][0]["children"].append(
            {"node_type": "text", "id": "c1", "tag": "p", "spans": [{"text": "New", "color": "#02060C"}]})
        result = splice_section(scope, DesignDocumentASTV11.model_validate(section))
        ids = re.findall(r'"id":"([a-z0-9_]+)"', result.model_dump_json())
        assert len(ids) == len(set(ids)) and _node(result, "c1")["spans"][0]["text"] == "Burger King"


class TestOnTheRoutingPath:
    @pytest.fixture(autouse=True)
    def _v11(self, monkeypatch):
        monkeypatch.setattr(settings, "design_schema_v11_enabled", True)

    def test_style_edit_sends_only_the_section(self, edges, monkeypatch):
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: pytest.fail("style wording needs no text call"))
        prompts = []

        def provider(messages, image_bytes, temperature, provider, key, request_id=""):
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            existing = prompt.split("<existing_design_ast>\n", 1)[1].split("\n</existing_design_ast>", 1)[0]
            return _section_answer(existing, _make_heading_green).model_dump_json()
        monkeypatch.setattr(tpa, "_run_design_json_completion", provider)
        events = _events(_stream("make the heading green" + SELECT.format("header"), operation_type="edit_design",
                                 existing_design_ast=DESIGN))
        assert len(prompts) == 1 and "Burger King" not in prompts[0]
        assert _node(DesignDocumentASTV11.model_validate(events["design_ready"]), "t1")["spans"][0]["color"] == "#1E8B36"
        assert _node(DesignDocumentASTV11.model_validate(events["design_ready"]), "c1")["spans"][0]["text"] == "Burger King"
        assert events["complete"]["edit_path"] == "section"

    def test_a_section_answer_that_fails_runs_the_full_edit(self, edges, monkeypatch):
        prompts = []

        def provider(messages, image_bytes, temperature, provider, key, request_id=""):
            prompts.append(messages[-1]["content"])
            if len(prompts) <= 3:                                            # the section attempt and its repairs
                return "{not json"
            return _design_json()
        monkeypatch.setattr(tpa, "_run_design_json_completion", provider)
        events = _events(_stream("make the heading green" + SELECT.format("header"), operation_type="edit_design",
                                 existing_design_ast=DESIGN))
        assert "Burger King" in prompts[-1] and events["complete"]["edit_path"] == "full"

    def test_switched_off_means_the_full_edit(self, edges, monkeypatch):
        monkeypatch.setattr(settings, "design_section_edit_enabled", False)
        prompts = []
        monkeypatch.setattr(tpa, "_run_design_json_completion",
                            lambda messages, *a, **k: (prompts.append(messages[-1]["content"]), _design_json())[1])
        events = _events(_stream("make the heading green" + SELECT.format("header"), operation_type="edit_design",
                                 existing_design_ast=DESIGN))
        assert len(prompts) == 1 and "Burger King" in prompts[0] and events["complete"]["edit_path"] == "full"

    def test_wording_still_takes_the_text_lane_first(self, edges, monkeypatch):
        monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: FakeLLM({"texts": {"t1": ["Pizza"]}}))
        events = _events(_stream("change the name to Pizza here" + SELECT.format("header"), operation_type="edit_design",
                                 existing_design_ast=DESIGN))
        assert edges["providers"] == [] and events["complete"]["edit_path"] == "text_only"
