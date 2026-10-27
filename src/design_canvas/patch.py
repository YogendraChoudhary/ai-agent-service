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
    return DesignDocumentAST.model_validate(raw)


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
