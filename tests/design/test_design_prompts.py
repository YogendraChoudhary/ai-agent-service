"""
The design system prompt carries the real DesignDocumentAST schema and a valid example, and stays byte-stable.

Without the schema the model had to guess every field name (`node_type`, `spans`, `icon_name`, `layers`, ...), so
first drafts failed Pydantic parsing and burned the repair attempts.
"""

from __future__ import annotations

import importlib
import json
import re

import pytest

import src.design_canvas.prompts as prompts
from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.validate import validate_design_ast


def _block(tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompts.DESIGN_SYSTEM_PROMPT, re.S)
    assert match, f"<{tag}> block missing from the system prompt"
    return match.group(1)


def _walk(node):
    yield node
    for child in getattr(node, "children", []):
        yield from _walk(child)


class TestSchemaInPrompt:
    def test_embedded_schema_is_the_models_schema(self):
        embedded = json.loads(_block("design_document_ast_json_schema"))
        assert embedded == prompts._without_titles(DesignDocumentAST.model_json_schema())

    def test_every_node_type_and_field_is_spelled_out(self):
        defs = json.loads(_block("design_document_ast_json_schema"))["$defs"]
        node_types = {d["properties"]["node_type"]["const"] for d in defs.values() if "node_type" in d.get("properties", {})}
        assert node_types == {"container", "text", "divider", "vector_illustration", "icon_button", "button"}
        assert {"spans", "tag", "align", "font_size_px"} <= set(defs["TextNode"]["properties"])
        assert {"layers", "view_box"} <= set(defs["VectorIllustrationNode"]["properties"])
        assert {"layer_id", "svg_element", "attributes", "fill_color"} <= set(defs["SVGLayer"]["properties"])
        assert "arrow_left" in defs["IconButtonNode"]["properties"]["icon_name"]["enum"]

    def test_generated_titles_are_dropped_but_the_title_field_is_kept(self):
        embedded = json.loads(_block("design_document_ast_json_schema"))
        assert "title" in embedded["properties"]                          # DesignDocumentAST.title is a real field
        assert '"title":"' not in prompts.DESIGN_AST_JSON_SCHEMA          # Pydantic's "title": "Font Size Px" noise

    def test_prompt_is_byte_stable_across_imports(self):
        before = prompts.DESIGN_SYSTEM_PROMPT
        assert importlib.reload(prompts).DESIGN_SYSTEM_PROMPT == before


class TestExampleInPrompt:
    def test_example_parses_and_passes_every_domain_check(self):
        example = DesignDocumentAST.model_validate_json(_block("example_output"))
        assert validate_design_ast(example) == []

    def test_example_uses_every_node_type(self):
        example = DesignDocumentAST.model_validate_json(_block("example_output"))
        assert {n.node_type for n in _walk(example.root)} == {
            "container", "text", "divider", "vector_illustration", "icon_button", "button"}


@pytest.mark.asyncio
async def test_harness_sends_the_schema_bearing_prompt_as_the_system_message():
    from src.design_canvas.service import execute_design_canvas_harness

    seen = []

    async def _run(messages, image_bytes=None, temperature=0.2):
        seen.append(messages[0])
        return prompts.DESIGN_EXAMPLE_JSON

    runner = type("Runner", (), {"run_json_completion": staticmethod(_run)})()
    await execute_design_canvas_harness(runner, "design a card", "create_from_prompt")
    assert seen[0] == {"role": "system", "content": prompts.DESIGN_SYSTEM_PROMPT}
