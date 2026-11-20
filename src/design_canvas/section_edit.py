"""
Section-only edits of a design: a style or layout change of the selected section ("make the heading green", "more
padding here") sends the model that section alone, then splices the answer back into the design.

The full edit path sends the whole design and asks for the whole design back: about 45 s for a 43 KB design even after
compaction, for a change to a 1.7 KB header. Here the model edits a small design made of the selected node inside a
stand-in container, which carries the background the section sits on (so text contrast is judged against what is really
behind it). The answer goes through the normal harness (schema checks, repairs); the spliced design is validated again as
a whole (ids unique across the page, depth, contrast) and auto-healed the usual way. If that does not hold, the caller
runs the full edit path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from src.design_canvas.patch import compact_design_json
from src.design_canvas.text_edit import split_selection

logger = logging.getLogger(__name__)

_WRAPPER_ID = "section_context"
_BACKGROUND_FIELDS = ("background_color", "background_gradient")


@dataclass
class SectionScope:
    selected_id: str
    instruction: str
    wrapper_id: str
    section_json: str          # the small design the model edits
    original: object           # the whole design, validated
    original_json: str


def _path_to(node: dict, node_id: str, path: List[dict]) -> Optional[List[dict]]:
    """[root, ..., parent, node] for the node with this id, or None."""
    path = path + [node]
    if node.get("id") == node_id:
        return path
    for child in node.get("children") or []:
        found = _path_to(child, node_id, path)
        if found:
            return found
    return None


def _ids(node: dict, out: set) -> set:
    out.add(node.get("id"))
    for child in node.get("children") or []:
        _ids(child, out)
    return out


def _rename_clashes(node: dict, taken: set) -> None:
    node_id = node.get("id")
    if node_id in taken:
        n = 2
        while True:
            suffix = f"_{n}"
            candidate = node_id[:40 - len(suffix)] + suffix
            if candidate not in taken:
                break
            n += 1
        node["id"] = candidate
    taken.add(node.get("id"))
    for child in node.get("children") or []:
        _rename_clashes(child, taken)


def prepare_section_edit(existing_ast_json: str, query: str, *, schema_version: Optional[str] = None,
                         max_fraction: float = 0.6) -> Optional[SectionScope]:
    """The section to edit on its own, or None when there is no usable selection or the section is most of the design."""
    from src.design_canvas.schema import DesignDocumentAST
    from src.design_canvas.schema_v11 import DesignDocumentASTV11

    instruction, selected_id = split_selection(query)
    if not selected_id or not instruction:
        return None
    try:
        raw = json.loads(existing_ast_json)
        version = raw.get("version", "1.0")
        if version == "1.1" and schema_version == "1.0":
            return None                           # the full path downgrades the design; keep that behaviour
        model = DesignDocumentASTV11 if version == "1.1" else DesignDocumentAST
        original = model.model_validate(raw)
    except Exception:
        return None
    full = original.model_dump(mode="json")
    path = _path_to(full["root"], selected_id, [])
    if not path or len(path) < 2:                 # not found, or the root itself: that is the whole design
        return None
    node, ancestors = path[-1], path[:-1]

    wrapper_id = _WRAPPER_ID
    taken = _ids(full["root"], set())
    n = 2
    while wrapper_id in taken:
        wrapper_id = f"{_WRAPPER_ID}_{n}"
        n += 1
    backdrop = {}
    for ancestor in reversed(ancestors):
        style = ancestor.get("style") or {}
        if any(style.get(name) for name in _BACKGROUND_FIELDS):
            backdrop = {name: style[name] for name in _BACKGROUND_FIELDS if style.get(name)}
            break
    wrapper = {"node_type": "container", "id": wrapper_id, "layout": "flex_col",
               "align_items": ancestors[-1].get("align_items", "stretch"), "style": backdrop, "children": [node]}
    section = {**{k: v for k, v in full.items() if k != "root"}, "root": wrapper}
    try:
        section_json = compact_design_json(model.model_validate(section).model_dump_json())
    except Exception as exc:
        logger.info("design section edit: section %s does not stand alone (%s)", selected_id, type(exc).__name__)
        return None
    if len(section_json) > max_fraction * len(compact_design_json(original.model_dump_json())):
        return None                               # most of the design: the full path costs about the same
    return SectionScope(selected_id, instruction, wrapper_id, section_json, original, existing_ast_json)


def section_query(scope: SectionScope) -> str:
    """The request the model gets with the section design."""
    return (f"{scope.instruction}\n\n"
            f"The design above is only the part of a larger page the user selected: the node `{scope.selected_id}`. "
            f"The container `{scope.wrapper_id}` stands in for the rest of the page (its background is what the section "
            f"sits on): keep its id and style as they are. Apply the request to `{scope.selected_id}`; you may add new "
            f"elements inside it or next to it inside `{scope.wrapper_id}`. Keep everything else exactly as given.")


def splice_section(scope: SectionScope, edited_section) -> Optional[object]:
    """The whole design with the edited section in place of the selected node, validated; None if it does not hold."""
    from src.design_canvas.validate import auto_heal_design_ast, validate_design_ast

    model = type(scope.original)
    section = edited_section.model_dump(mode="json")
    replacement = section["root"].get("children") or []
    if not replacement:
        return None
    full = scope.original.model_dump(mode="json")
    path = _path_to(full["root"], scope.selected_id, [])
    parent = path[-2]
    # Ids used outside the section belong to nodes the user did not touch: a new node that reuses one is renamed, never
    # the existing node (auto-heal keeps the first occurrence, which may be the new one).
    taken = _ids(full["root"], set()) - _ids(path[-1], set())
    for node in replacement:
        _rename_clashes(node, taken)
    index = next(i for i, child in enumerate(parent["children"]) if child.get("id") == scope.selected_id)
    parent["children"][index:index + 1] = replacement
    if len(parent["children"]) > 25:
        return None
    if section.get("fonts") is not None and "fonts" in full:
        full["fonts"] = section["fonts"]          # a font family change of a role, as the full path would allow
    try:
        candidate = model.model_validate(full)
    except Exception as exc:
        logger.info("design section edit: spliced design did not validate (%s)", type(exc).__name__)
        return None
    before = set(validate_design_ast(scope.original, existing_ast_json=scope.original_json))
    errors = set(validate_design_ast(candidate, existing_ast_json=scope.original_json))
    if errors <= before:
        return candidate
    try:
        healed = auto_heal_design_ast(candidate, existing_ast_json=scope.original_json)
    except Exception:
        return None
    if set(validate_design_ast(healed, existing_ast_json=scope.original_json)) <= before:
        return healed
    logger.info("design section edit: spliced design keeps errors %s", sorted(errors - before)[:3])
    return None
