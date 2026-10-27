"""
Prompt factory for Mode 8: Editable UI Design Engine.

Provides:
    DESIGN_SYSTEM_PROMPT          – byte-stable system prompt injected once: rules, the
                                    DesignDocumentAST JSON Schema and one valid example.
    build_initial_design_prompt() – user turn for first-attempt generation.
    build_design_repair_prompt()  – XML-fenced repair turn appended on validation failure.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from src.design_canvas.schema import DesignDocumentAST

if TYPE_CHECKING:
    from src.design_canvas.vision_extractor import VisionMetadata

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _without_titles(node: Any, is_properties: bool = False) -> Any:
    """Drop Pydantic's generated ``title`` annotations (noise for the model); keep a *field* named ``title``."""
    if isinstance(node, dict):
        return {k: _without_titles(v, k == "properties") for k, v in node.items()
                if is_properties or not (k == "title" and isinstance(v, str))}
    if isinstance(node, list):
        return [_without_titles(v) for v in node]
    return node


# Generated from the Pydantic model, so the prompt can never drift from what the harness parses. Sorted keys and
# fixed separators keep it byte-stable (prompt-cache friendly) for a given Pydantic version.
DESIGN_AST_JSON_SCHEMA = json.dumps(
    _without_titles(DesignDocumentAST.model_json_schema()), sort_keys=True, separators=(",", ":"))

# One small design that uses every node type; tests/design/test_design_prompts.py keeps it valid.
DESIGN_EXAMPLE_JSON = json.dumps({
    "block_type": "eddyter_design_canvas",
    "version": "1.0",
    "design_id": "dsn_example_card",
    "title": "Example Card",
    "theme_palette": ["#0B0B0F", "#34F5A6", "#FFFFFF", "#8E8E93"],
    "root": {
        "node_type": "container", "id": "root_card", "layout": "flex_col", "align_items": "center",
        "justify_content": "center", "gap_px": 20,
        "style": {"background_color": "#0B0B0F", "padding": {"top": 40, "right": 32, "bottom": 40, "left": 32},
                  "border_radius": 16},
        "children": [
            {"node_type": "text", "id": "title", "tag": "h1", "font_size_px": 32, "spans": [
                {"text": "Ask ", "color": "#8E8E93", "font_weight": "bold"},
                {"text": "Anything", "color": "#FFFFFF", "font_weight": "bold"}]},
            {"node_type": "container", "id": "media_row", "layout": "flex_row", "justify_content": "space_between",
             "gap_px": 24, "children": [
                {"node_type": "icon_button", "id": "prev_btn", "icon_name": "arrow_left", "color": "#8E8E93",
                 "aria_label": "Previous"},
                {"node_type": "vector_illustration", "id": "bubble_art", "view_box": "0 0 120 100",
                 "width_px": 120, "height_px": 100, "layers": [
                    {"layer_id": "bubble", "label": "Speech bubble", "svg_element": "rect",
                     "attributes": {"x": 10, "y": 10, "width": 100, "height": 70, "rx": 20}, "fill_color": "#34F5A6"},
                    {"layer_id": "dot", "label": "Dot", "svg_element": "circle",
                     "attributes": {"cx": 60, "cy": 45, "r": 8}, "fill_color": "#0B0B0F"}]},
                {"node_type": "icon_button", "id": "next_btn", "icon_name": "arrow_right", "color": "#FFFFFF",
                 "aria_label": "Next"}]},
            {"node_type": "divider", "id": "rule", "color": "#2C2C2E", "width_percent": 60},
            {"node_type": "text", "id": "body", "tag": "p", "font_size_px": 15, "spans": [
                {"text": "Short, long or anything in between.", "color": "#D1D1D6"}]},
            {"node_type": "button", "id": "cta", "label": "Get started", "text_color": "#0B0B0F",
             "style": {"background_color": "#34F5A6", "border_radius": 999,
                       "padding": {"top": 12, "right": 24, "bottom": 12, "left": 24}}},
        ],
    },
}, separators=(",", ":"))

DESIGN_SYSTEM_PROMPT = """You are an expert UI/UX design engineer embedded inside the Eddyter AI editor.
Your sole job is to output a **single valid JSON object** that conforms exactly to
the JSON Schema in <design_document_ast_json_schema> — never Markdown, never prose,
never code fences.

Schema rules (read carefully):
- Every node is an object with a `node_type` of "container", "text", "divider",
  "vector_illustration", "icon_button" or "button"; use only the fields the schema
  defines for that node type.
- Every node MUST have a unique `id` matching ^[a-z0-9_]{2,40}$.
- Every SVGLayer MUST have a unique `layer_id` matching ^[a-z0-9_]{2,40}$.
- `theme_palette` must contain 2–10 hex colour strings (#RRGGBB or #RRGGBBAA).
- `ContainerNode.children` must be non-empty (1–25 items).
- `VectorIllustrationNode.layers` must be non-empty (1–24 items).
- Colour values must match ^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$.
- Maximum AST tree depth is 6.
- Text spans on dark backgrounds MUST have sufficient WCAG 2.1 contrast.
- SVG attributes must only contain geometry values (d, cx, cy, r, x, y, width,
  height, rx, ry, transform); never event handlers, hrefs, or script references.

Return ONLY the raw JSON object with zero markdown wrappers or explanations.

<design_document_ast_json_schema>
""" + DESIGN_AST_JSON_SCHEMA + """
</design_document_ast_json_schema>

<example_output>
""" + DESIGN_EXAMPLE_JSON + """
</example_output>\
"""

# ---------------------------------------------------------------------------
# Initial design prompt
# ---------------------------------------------------------------------------

def build_initial_design_prompt(
    user_query: str,
    sub_mode: str,
    vision_meta: "Optional[VisionMetadata]" = None,
    existing_ast_json: Optional[str] = None,
) -> str:
    """
    Build the first-turn user message for the design harness.

    Args:
        user_query:       The user's raw request.
        sub_mode:         "create_from_prompt" | "create_from_vision" | "edit_existing_ast".
        vision_meta:      Extracted palette/dimension data (create_from_vision only).
        existing_ast_json: Serialised DesignDocumentAST to mutate (edit_existing_ast only).

    Returns:
        A multi-section prompt string ready to send as the ``user`` role message.
    """
    parts: list[str] = []

    # Vision metadata block
    if vision_meta is not None:
        palette_csv = ", ".join(vision_meta.top_hex_colors)
        theme_word = "dark" if vision_meta.is_dark_theme else "light"
        parts.append(
            f"<deterministic_vision_metadata>\n"
            f"  image_dimensions: {vision_meta.width}×{vision_meta.height}px\n"
            f"  theme_polarity: {theme_word}\n"
            f"  dominant_background: {vision_meta.dominant_background}\n"
            f"  extracted_palette: [{palette_csv}]\n"
            f"</deterministic_vision_metadata>\n"
        )

    # Existing AST block (edit mode)
    if existing_ast_json:
        parts.append(
            f"<existing_design_ast>\n{existing_ast_json}\n</existing_design_ast>\n"
        )

    # Task description
    if sub_mode == "edit_existing_ast":
        task = (
            "TASK: Edit the existing DesignDocumentAST above according to the user request.\n"
            "Preserve all sibling node `id` values that are not being changed.\n"
        )
    elif sub_mode == "create_from_vision":
        task = (
            "TASK: Recreate the attached reference UI image as a fully editable "
            "DesignDocumentAST.\n"
            "Use the extracted palette colours exactly — do NOT invent new colours.\n"
        )
    else:
        task = (
            "TASK: Generate a new editable UI design as a DesignDocumentAST "
            "that matches the user's description.\n"
        )

    parts.append(task)
    parts.append(f"USER REQUEST:\n{user_query}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Repair prompt (appended to the conversation on validation failure)
# ---------------------------------------------------------------------------

def build_design_repair_prompt(errors: list[str]) -> str:
    """
    Build an XML-fenced repair instruction appended to the conversation
    when ``validate_design_ast()`` or Pydantic returns errors.

    Args:
        errors: List of human-readable error strings from the validator.

    Returns:
        A structured string to send as the next ``user`` role message.
    """
    formatted_errors = "\n".join(f"  - {err}" for err in errors)
    return (
        "<harness_validation_failure>\n"
        "Your previous JSON output failed deterministic verification with the following exact errors:\n"
        f"{formatted_errors}\n"
        "</harness_validation_failure>\n\n"
        "REPAIR INSTRUCTIONS:\n"
        "1. Modify ONLY the failing nodes, IDs, colors, or attributes listed inside <harness_validation_failure>.\n"
        "2. Do NOT alter valid sibling nodes, text content, or layout hierarchy.\n"
        "3. Return ONLY the complete, valid raw JSON object conforming to DesignDocumentAST with zero markdown wrappers."
    )
