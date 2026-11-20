"""
AST subtree freezer and splicer for Mode 8 iterative edits.

``freeze_subtree()``  – marks a subtree's ``editable`` flag False so the
                        harness and the frontend know not to mutate it.

``splice_subtree()``  – replaces a node identified by ``target_id`` anywhere
                        in the tree with a new node dict, returning a deep-
                        copied mutated version of the root.

These are used for iterative edit workflows where the user asks to "change
only the header section" — the rest of the tree is frozen before the model
sees it, preventing unintended drift.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

from src.design_canvas.schema import ContainerNode, DesignDocumentAST


def freeze_subtree(node: dict, freeze: bool = True) -> dict:
    """
    Return a deep copy of ``node`` with every ``editable`` field set to
    ``not freeze`` (i.e., False when freeze=True).

    Args:
        node:   Raw dict representation of any DesignNode.
        freeze: True = mark uneditable; False = restore editable.

    Returns:
        Deep-copied and mutated dict.
    """
    node = copy.deepcopy(node)
    _set_editable(node, not freeze)
    return node


def _set_editable(node: Any, value: bool) -> None:
    if not isinstance(node, dict):
        return
    if "editable" in node:
        node["editable"] = value
    for child in node.get("children", []):
        _set_editable(child, value)
    for layer in node.get("layers", []):
        _set_editable(layer, value)


def splice_subtree(
    ast: DesignDocumentAST,
    target_id: str,
    replacement_node: dict,
) -> Optional[DesignDocumentAST]:
    """
    Replace the node with ``id == target_id`` anywhere in the AST tree with
    ``replacement_node``.

    Args:
        ast:              The current DesignDocumentAST.
        target_id:        The ``id`` of the node to replace.
        replacement_node: Raw dict for the replacement node (must be valid).

    Returns:
        A new DesignDocumentAST with the splice applied, or None if
        ``target_id`` was not found.
    """
    raw = json.loads(ast.model_dump_json())
    found = _splice_in_dict(raw.get("root", {}), target_id, replacement_node)
    if not found:
        return None
    return type(ast).model_validate(raw)


def _splice_in_dict(node: Any, target_id: str, replacement: dict) -> bool:
    """Return True if splice was performed."""
    if not isinstance(node, dict):
        return False
    children = node.get("children")
    if isinstance(children, list):
        for i, child in enumerate(children):
            if isinstance(child, dict) and child.get("id") == target_id:
                children[i] = replacement
                return True
            if _splice_in_dict(child, target_id, replacement):
                return True
    return False


def compact_design_json(design_json: str) -> str:
    """
    The design without the fields that only repeat their default value, for the edit prompt. The editor sends every field
    back (a saved 43 KB burger design held 631 nulls); the model was asked to copy all of them, which pushed a full edit
    past the 8,192-token output cap. Unset fields mean their defaults, so the compact form validates to the same design;
    that is checked, and the input is returned unchanged when it does not hold (or does not parse).
    """
    from src.design_canvas.schema_v11 import DesignDocumentASTV11

    try:
        raw = json.loads(design_json)
        model = DesignDocumentASTV11 if isinstance(raw, dict) and raw.get("version") == "1.1" else DesignDocumentAST
        design = model.model_validate(raw)
        full = design.model_dump(mode="json")
        compact = design.model_dump(mode="json", exclude_defaults=True)
        compact = {"block_type": full["block_type"], "version": full["version"], **compact}
        _keep_node_types(full["root"], compact["root"])
        if model.model_validate(compact) != design:
            return design_json
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return design_json


def _keep_node_types(full: Any, compact: Any) -> None:
    """node_type has a default, but it is the union's discriminator: every node keeps it."""
    if isinstance(full, dict) and isinstance(compact, dict):
        if "node_type" in full and "node_type" not in compact:
            items = list(compact.items())
            compact.clear()
            compact.update([("node_type", full["node_type"])] + items)
        for key, value in compact.items():
            if key in full:
                _keep_node_types(full[key], value)
    elif isinstance(full, list) and isinstance(compact, list):
        for full_item, compact_item in zip(full, compact):
            _keep_node_types(full_item, compact_item)
