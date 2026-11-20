"""
Fast lane for wording-only edits of a design ("make this shorter", "change the name to Pizza").

The full edit path sends the whole design to the model and asks for the whole design back. A 40 KB design is about
13,000 output tokens, more than the 8,192 the design call may write, so a one-word rename ran for minutes and could
still fail. Here the model sees only the text items of the selected section (or of the design, for wording that names
text) and answers with the items it changed. The new words are written into the design in code: ids, colours, layout,
links and photos cannot change, and the result is validated like any other design.

The model may answer that the request is not a wording change ("make it red", "add a button"); the caller then runs
the full edit path, as before. Any invalid answer does the same.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.json_reply import strip_json_fence

logger = logging.getLogger(__name__)

# The editor tells the model which section is selected by appending this sentence to the user's words
# (live 2026-11-20: "...\n\n(Apply this to the selected section, id header. Keep everything else unchanged.)").
_SELECTION_RE = re.compile(
    r"\s*\(\s*apply this to the selected [a-z ]{2,24}?,?\s+id[\s:=]+[\"'`]?([a-z0-9_]{2,40})[\"'`]?\s*\.?[^()]*\)\s*$",
    re.IGNORECASE,
)

# Wording that needs more than new words: colours, type, spacing, layout, images, links, theme.
_COLOURS = (r"red|blue|green|yellow|orange|purple|violet|pink|black|white|gr[ae]y|teal|navy|gold|golden|silver|brown"
            r"|beige|cyan|magenta|maroon|olive|lime|indigo|cream|pastel|neon")
_STYLE_RE = re.compile(
    rf"\b(?:(?:{_COLOURS})(?:er|ish|s)?|colou?r(?:s|ed|ful)?|dark(?:er)?|light(?:er)?\s+(?:theme|mode|background)"
    r"|bright(?:er)?|theme|background|bg|gradient|shadow|border|outline|rounded|radius|corners?|padding|margins?"
    r"|spacing|gap|align(?:ed|ment)?|cent(?:er|re)(?:ed|d)?|layout|columns?|grid|side by side|stack(?:ed)?"
    r"|font|typeface|bold(?:er)?|italic|underline(?:d)?|uppercase|lowercase|all caps|size|bigger|smaller|larger"
    r"|wider|narrower|taller|width|height|images?|photos?|pictures?|icons?|logo|illustration|svg|links?|href|urls?"
    r"|style|opacity|transparent|move|swap|reorder|rearrange|duplicate)\b",
    re.IGNORECASE,
)
# Adding or removing an element ("add a button", "remove the second card"); adding or removing words is a text edit.
_STRUCTURE_RE = re.compile(
    r"\b(?:add|insert|remove|delete|hide|drop)\b\s+(?:(?:a|an|the|another|one|this|that|these|those|new|more|\d+)\s+)?"
    r"(?:\w+\s+)?(?:buttons?|sections?|cards?|divider|rows?|items?|elements?|blocks?|badges?|chips?|fields?|nodes?"
    r"|headers?|footers?|banners?|tiles?)\b",
    re.IGNORECASE,
)
# Wording that names text, for a request with no section selected.
_TEXT_CUE_RE = re.compile(
    r"\b(?:short(?:er|en)|long(?:er)|lengthen|concise|brief(?:er)?|condense|trim|rename|renamed|rewrite|rewritten|reword"
    r"|rephrase|paraphrase|translate|translation|tone|formal|casual|friendly|playful|professional|punchier|catchier"
    r"|typos?|grammar|spelling|proofread|simplify|wording|text|copy(?:writing)?|title|heading|headline|subtitle"
    r"|tagline|name|label|caption|description|slogan|says?|replace|instead of)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^']{2,}'|‘[^’]*’")

MAX_ITEMS = 120
MAX_SOURCE_CHARS = 12_000

SYSTEM_PROMPT = """You change the wording inside a UI design.
You get the user's request and a JSON list of the text items in the part of the design they selected. Each item has an
`id`, a `kind` (h1, h2, h3, p, span, caption or button) and `runs`: the item's text split where its colour or weight
changes (a two-colour headline has two runs).

Decide whether the request can be done by changing only the words of these items.

If it can, answer {"texts": {"<id>": ["run", ...]}} with ONLY the items you changed:
- Change only what the request asks for. Leave every other item out of the answer.
- Keep the same number of runs where you can, and keep the words meant to stand out in their own run.
- Keep spaces between runs as they are (a run that ends in a space keeps it).
- "Shorter" means clearly shorter (about half, unless a length is given); "longer" means clearly longer.
- Keep the language unless asked to translate. Never add Markdown, HTML, quotes around the text, or emoji the
  request did not ask for.
- A name or word to replace: replace it wherever it appears in these items, matching plural and capitalisation.

If it needs anything else (colours, fonts, sizes, spacing, layout, images, links, or adding, removing or moving
elements), answer {"not_text_edit": true}.

Answer with the JSON object only."""


@dataclass
class TextEditResult:
    ast: object                                   # the validated design, same model class as the input
    changed_ids: List[str] = field(default_factory=list)
    scope_id: Optional[str] = None                # the selected node, or None for the whole design


def split_selection(query: str) -> tuple[str, Optional[str]]:
    """(the user's own words, the selected node id or None), from the editor's appended selection sentence."""
    text = query or ""
    match = _SELECTION_RE.search(text)
    if not match:
        return text.strip(), None
    return text[:match.start()].strip(), match.group(1).lower()


def wants_text_lane(instruction: str, has_selection: bool) -> bool:
    """Zero-LLM pre-check: skip the extra call for wording that clearly changes more than words."""
    words = _QUOTED_RE.sub(" ", instruction or "")        # quoted new text ("say 'Bold flavours'") is not a style cue
    if not words.strip() or _STYLE_RE.search(words) or _STRUCTURE_RE.search(words):
        return False
    return has_selection or bool(_TEXT_CUE_RE.search(words))


def _find(node: dict, node_id: str) -> Optional[dict]:
    if node.get("id") == node_id:
        return node
    for child in node.get("children") or []:
        found = _find(child, node_id)
        if found is not None:
            return found
    return None


def _text_nodes(node: dict, out: List[dict]) -> List[dict]:
    if node.get("node_type") in ("text", "button"):
        out.append(node)
    for child in node.get("children") or []:
        _text_nodes(child, out)
    return out


def _item(node: dict) -> dict:
    if node["node_type"] == "button":
        return {"id": node["id"], "kind": "button", "runs": [node["label"]]}
    return {"id": node["id"], "kind": node.get("tag") or "p", "runs": [span["text"] for span in node["spans"]]}


def _runs(value) -> Optional[List[str]]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(run, str) for run in value):
        return None
    runs = [run for run in value if run.strip()]
    return runs or None


def _apply(node: dict, runs: List[str]) -> bool:
    """Write new runs into a text or button node; False when nothing changed."""
    if node["node_type"] == "button":
        label = "".join(runs).strip()
        if label == node["label"]:
            return False
        node["label"] = label
        return True
    spans = node["spans"]
    if [span["text"] for span in spans] == runs:
        return False
    if len(runs) > len(spans):                    # extra runs join the last span, in its style
        runs = runs[:len(spans) - 1] + ["".join(runs[len(spans) - 1:])]
    node["spans"] = [dict(spans[i], text=run) for i, run in enumerate(runs)]
    return True


def try_text_edit(existing_ast_json: str, query: str, llm: Callable[[str, List[dict], object], str], *,
                  schema_version: Optional[str] = None) -> Optional[TextEditResult]:
    """
    The edited design when the request is a wording change of the selected section (or of named text), else None.
    None means "use the full edit path": not a wording request, nothing to edit, or an answer that did not hold up.
    Provider errors and cancellation propagate; the caller decides whether to fall back.
    """
    from src.design_canvas.schema import DesignDocumentAST
    from src.design_canvas.schema_v11 import DesignDocumentASTV11
    from src.design_canvas.validate import validate_design_ast
    from src.longform.llm import CallSpec

    instruction, selected_id = split_selection(query)
    if not wants_text_lane(instruction, selected_id is not None):
        return None
    try:
        raw = json.loads(existing_ast_json)
    except ValueError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("root"), dict):
        return None
    version = raw.get("version", "1.0")
    if version == "1.1" and schema_version == "1.0":
        return None                               # the full path downgrades the design; keep that behaviour
    model = DesignDocumentASTV11 if version == "1.1" else DesignDocumentAST
    try:
        original = model.model_validate(raw)
    except Exception:
        return None
    working = original.model_dump()
    scope = _find(working["root"], selected_id) if selected_id else working["root"]
    if scope is None:
        logger.info("design text edit: selected node %s not in the design", selected_id)
        return None
    nodes = _text_nodes(scope, [])
    items = [_item(node) for node in nodes]
    source_chars = sum(len(run) for item in items for run in item["runs"])
    if not items or len(items) > MAX_ITEMS or source_chars > MAX_SOURCE_CHARS:
        return None

    user = (f"Design: {json.dumps(working.get('title') or '', ensure_ascii=False)}\n"
            f"Request: {json.dumps(instruction, ensure_ascii=False)}\n"
            f"Items: {json.dumps(items, ensure_ascii=False)}")
    spec = CallSpec(stage="design_text_edit", temperature=0.2,
                                    max_tokens=min(4000, 400 + source_chars))
    reply = llm(SYSTEM_PROMPT, [{"role": "user", "content": user}], spec)
    try:
        answer = json.loads(strip_json_fence(reply or "").strip())
    except ValueError:
        logger.info("design text edit: answer was not JSON")
        return None
    if not isinstance(answer, dict) or answer.get("not_text_edit"):
        return None
    texts = answer.get("texts")
    if not isinstance(texts, dict):
        return None

    by_id = {node["id"]: node for node in nodes}
    originals = {item["id"]: item["runs"] for item in items}
    changed: List[str] = []
    for node_id, value in texts.items():
        node = by_id.get(node_id)
        runs = _runs(value)
        if node is None or runs is None:
            logger.info("design text edit: answer item %r ignored", node_id)
            continue
        limit = max(400, 4 * sum(len(run) for run in originals[node_id]) + 200)
        if sum(len(run) for run in runs) > limit:
            return None
        if _apply(node, runs):
            changed.append(node_id)
    if not changed:
        return None

    try:
        edited = model.model_validate(working)
    except Exception as exc:
        logger.info("design text edit: edited design did not validate (%s)", type(exc).__name__)
        return None
    # New words cannot fix or break a design check except by accident: accept no error the original did not have.
    before = set(validate_design_ast(original, existing_ast_json=existing_ast_json))
    after = set(validate_design_ast(edited, existing_ast_json=existing_ast_json))
    if not after <= before:
        logger.info("design text edit: edit added design errors %s", sorted(after - before)[:3])
        return None
    return TextEditResult(ast=edited, changed_ids=changed, scope_id=selected_id)
