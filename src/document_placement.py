"""
Where a Draft goes when the task names a place ("below the Pricing heading", "at the end of the Risks section",
"after the second paragraph"). Resolved in code against the same top-level blocks Edit uses, with no model call:
the editor gets the anchor block's exact text and UTF-16 offsets plus "before"/"after", or a clarification.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .document_edit.blocks import DocumentBlock, document_blocks, utf16_length


class DraftPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: Literal["after", "before"]
    text: str = Field(min_length=1)          # exact text of one top-level block, as Edit's `before`
    start_utf16: int = Field(ge=0)
    end_utf16: int = Field(ge=0)


_AFTER = r"below|under|beneath|underneath|after|following"
_BEFORE = r"above|before|ahead\s+of|preceding"
_DOCUMENT = r"(?:document|doc|page|article|text|file|post|draft|note|piece)"
_STRUCTURE = r"(?:heading|title|section|chapter)"
_KINDS = {"paragraph": ("paragraph",), "list": ("list",), "table": ("table",), "quote": ("quote", "blockquote"),
          "code block": ("fence", "code"), "image": ("image",), "block": None}
_KIND = r"(?:paragraph|list|table|quote|code\s+block|image|block)"
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
             "ninth": 9, "tenth": 10, "last": -1, "final": -1}
_QUOTE = r"[\"“”'‘’]"
_CLAUSE_END = re.compile(r"\s+(?:and|then|with|that|which|so|to\s+(?:explain|cover|describe|summari[sz]e))\s.*$", re.I)


# Words after a heading name that are not part of it: "between Pricing and Conclusion that explains refunds".
_CLOSING_WORDS = re.compile(r"\s+(?:that|which|to|about|explaining|describing|covering|with|for)\b.*$", re.I)


class _Unresolved(Exception):
    """A place was named but cannot be identified safely."""


def _norm(value: str) -> str:
    value = re.sub(r"[*_`#]+", "", value)
    return re.sub(r"\s+", " ", value.strip(" \t\"“”'‘’:.,")).casefold()


def _kind_name(word: str) -> str:
    return re.sub(r"\s+", " ", word.lower())


class _Document:
    def __init__(self, text: str, focus: Optional[int]):
        self.text = text
        self.blocks = [b for b in document_blocks(text) if b.kind != "blank" and b.text.strip()]
        self.focus = focus

    def placement(self, block: DocumentBlock, position: str) -> DraftPlacement:
        return DraftPlacement(position=position, text=block.text, start_utf16=utf16_length(self.text[:block.start]),
                              end_utf16=utf16_length(self.text[:block.end]))

    def headings(self, name: str) -> list[DocumentBlock]:
        wanted = _norm(name)
        if not wanted:
            return []
        named = [b for b in self.blocks if b.kind == "heading" and b.section]
        exact = [b for b in named if _norm(b.section[-1]) == wanted]
        return exact or [b for b in named if wanted in _norm(b.section[-1])]

    def heading(self, name: str) -> DocumentBlock:
        found = self.headings(name)
        if not found:
            raise _Unresolved(f"There is no heading named '{name.strip()}'. Name an existing heading, or place the "
                              "cursor where the draft should go.")
        if len(found) > 1:
            raise _Unresolved(f"More than one heading matches '{name.strip()}'. Use the full heading text, or place "
                              "the cursor where the draft should go.")
        return found[0]

    def section_end(self, heading: DocumentBlock) -> DocumentBlock:
        path = heading.section
        inside = [b for b in self.blocks if b.start >= heading.start and b.section[:len(path)] == path]
        return inside[-1]

    def of_kind(self, kind: str, within: Optional[DocumentBlock] = None) -> list[DocumentBlock]:
        kinds = _KINDS[_kind_name(kind)]
        blocks = [b for b in self.blocks if b.kind != "heading" and (kinds is None or b.kind in kinds)]
        if within is not None:
            blocks = [b for b in blocks if b.start > within.start and b.section[:len(within.section)] == within.section]
        return blocks

    def at_focus(self) -> DocumentBlock:
        if self.focus is None:
            raise _Unresolved("Place the cursor in the paragraph you mean, or name its heading.")
        before = [b for b in self.blocks if b.start <= self.focus]
        if not before:
            raise _Unresolved("Place the cursor in the paragraph you mean, or name its heading.")
        return before[-1]


def _target(doc: _Document, target: str, after: bool, word: str = "") -> Optional[DraftPlacement]:
    """Placement for the words after a position word, None when they do not name part of the document."""
    position = "after" if after else "before"
    t = _CLAUSE_END.sub("", target.strip())
    t = re.sub(r"^(?:the|this|that|my|our)\s+", lambda m: m.group(0) if m.group(0).lower().startswith("this") else "", t, flags=re.I)

    # "this paragraph", "this section", "it": the block at the cursor.
    match = re.match(r"this\s+(" + _KIND + r"|section|heading)\b", t, re.I)
    if match:
        block = doc.at_focus()
        if match.group(1).lower() == "section":
            heading = next((b for b in reversed(doc.blocks) if b.kind == "heading" and b.start <= block.start), None)
            if heading is None:
                raise _Unresolved("The cursor is not inside a section. Name the heading instead.")
            return doc.placement(doc.section_end(heading) if after else heading, position)
        return doc.placement(block, position)

    # A quoted passage: "after 'Our refund policy…'", or a quoted heading: 'below the "Pricing" heading'.
    match = re.match(_QUOTE + r"(.+?)" + _QUOTE + r"\s*(" + _STRUCTURE + r")?", t, re.I)
    if match:
        if match.group(2):
            return _heading_target(doc, match.group(1), match.group(2), after)
        needle = match.group(1).strip()
        found = [b for b in doc.blocks if needle.casefold() in b.text.casefold()]
        if len(found) != 1:
            raise _Unresolved(f"The quoted text '{needle}' {'was not found' if not found else 'appears more than once'}. "
                              "Quote words from one paragraph, or place the cursor there.")
        return doc.placement(found[0], position)

    # "heading X", "section X", "the X heading", "the X section".
    match = re.match(r"(" + _STRUCTURE + r")\s+(?:called\s+|named\s+|titled\s+)?(.+)$", t, re.I)
    if match and not re.match(_DOCUMENT + r"\b", match.group(2), re.I):
        return _heading_target(doc, match.group(2), match.group(1), after)
    match = re.match(r"(.+?)\s+(" + _STRUCTURE + r")\b", t, re.I)
    if match and not re.fullmatch(r"(?:\w+\s+)?" + _DOCUMENT, match.group(1), re.I):
        return _heading_target(doc, match.group(1), match.group(2), after)

    # "the second paragraph", "the last paragraph in Pricing", "paragraph 3", "the 2nd table under Results".
    match = re.match(r"(" + "|".join(_ORDINALS) + r"|\d+(?:st|nd|rd|th)?)\s+(" + _KIND + r")"
                     r"(?:\s+(?:in|under|of|within)\s+(?:the\s+)?(.+?)(?:\s+" + _STRUCTURE + r")?)?\s*$", t, re.I)
    number = None
    if match:
        word, kind, scope = match.group(1).lower(), match.group(2), match.group(3)
        number = _ORDINALS.get(word) or int(re.sub(r"\D", "", word))
    else:
        match = re.match(r"(" + _KIND + r")\s+(\d+)\s*$", t, re.I)
        if match:
            kind, scope, number = match.group(1), None, int(match.group(2))
    if number is not None:
        within = doc.heading(scope) if scope and not re.fullmatch(_DOCUMENT, scope, re.I) else None
        candidates = doc.of_kind(kind, within)
        if not candidates or number > len(candidates) or number == 0:
            raise _Unresolved(f"There is no {match.group(0).strip()}. Name the heading, or place the cursor there.")
        return doc.placement(candidates[number - 1] if number > 0 else candidates[-1], position)

    # "the paragraph about refunds", "the table on pricing", "the list starting with 'Setup'".
    match = re.match(r"(" + _KIND + r")\s+(?:about|on|mentioning|that\s+mentions|that\s+talks\s+about|"
                     r"starting\s+with|beginning\s+with|that\s+starts\s+with|containing|with)\s+(.+)$", t, re.I)
    if match:
        needle = _norm(re.sub(r"^(?:the|a|an)\s+", "", match.group(2), flags=re.I))
        found = [b for b in doc.of_kind(match.group(1)) if needle and needle in _norm(b.text)]
        if len(found) != 1:
            raise _Unresolved(f"{'No' if not found else 'More than one'} {_kind_name(match.group(1))} mentions "
                              f"'{needle}'. Quote its opening words, or place the cursor there.")
        return doc.placement(found[0], position)

    # "the table", "the list": only when there is exactly one.
    match = re.fullmatch(r"(" + _KIND + r")", t, re.I)
    if match and _kind_name(match.group(1)) != "block":
        found = doc.of_kind(match.group(1))
        if len(found) != 1:
            raise _Unresolved(f"The document has {'no' if not found else 'more than one'} {_kind_name(match.group(1))}. "
                              "Name its heading, or place the cursor there.")
        return doc.placement(found[0], position)

    # A bare heading name: "below Pricing" goes straight after the heading; "after the Conclusion" (or "following")
    # goes after the whole section, as "after the Conclusion section" does.
    if doc.headings(t) and len(doc.headings(t)) == 1 and _norm(doc.headings(t)[0].section[-1]) == _norm(t):
        heading = doc.headings(t)[0]
        if after and word.lower() in ("after", "following"):
            return doc.placement(doc.section_end(heading), "after")
        return doc.placement(heading, position)
    return None


def _heading_target(doc: _Document, name: str, noun: str, after: bool) -> DraftPlacement:
    heading = doc.heading(name)
    if not after:
        return doc.placement(heading, "before")
    # "after the X section" goes after the whole section; "below the X heading" goes straight after the heading.
    whole_section = noun.lower() in ("section", "chapter") and len(doc.section_end(heading).text) > 0
    return doc.placement(doc.section_end(heading) if whole_section else heading, "after")


def resolve_placement(text: str, task: str, focus: Optional[int] = None) -> tuple[Optional[DraftPlacement], Optional[str]]:
    """
    (placement, None) when the task names a place that resolves to exactly one block; (None, question) when it names a
    place that cannot be identified; (None, None) when it names no place (the editor inserts at the cursor).
    """
    if not text.strip():
        return None, None
    try:
        doc = _Document(text, focus)
    except ValueError:
        return None, None
    if not doc.blocks:
        return None, None
    try:
        # "at the end of the Risks section", "at the top of the document", "at the end".
        edge = re.search(r"\b(?:at|to)\s+the\s+(?:very\s+)?(end|bottom|start|beginning|top)\b"
                         r"(?:\s+of\s+(?:the\s+|this\s+|my\s+|our\s+)?([^.;!?\n]+))?", task, re.I)
        if edge:
            at_end = edge.group(1).lower() in ("end", "bottom")
            scope = _CLAUSE_END.sub("", (edge.group(2) or "").strip())
            scope = re.sub(r"\s+" + _STRUCTURE + r"$", "", scope, flags=re.I)
            if not scope or re.fullmatch(r"(?:\w+\s+)?" + _DOCUMENT, scope, re.I):
                return (doc.placement(doc.blocks[-1], "after") if at_end else doc.placement(doc.blocks[0], "before")), None
            if doc.headings(scope) or re.search(_STRUCTURE, edge.group(2) or "", re.I):
                heading = doc.heading(scope)
                return doc.placement(doc.section_end(heading) if at_end else heading, "after"), None
        # "between Pricing and Conclusion": before the second heading, or after the first section.
        between = re.search(r"\bbetween\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?([^.;!?\n,]+)", task, re.I)
        if between:
            first, second = (re.sub(r"\s+(?:sections?|headings?|chapters?)$", "", g.strip(), flags=re.I)
                             for g in (between.group(1), between.group(2)))
            second = _CLOSING_WORDS.sub("", second).strip()
            if len(doc.headings(second)) == 1:
                return doc.placement(doc.headings(second)[0], "before"), None
            if len(doc.headings(first)) == 1:
                return doc.placement(doc.section_end(doc.headings(first)[0]), "after"), None
            if re.search(r"\b(?:sections?|headings?|chapters?)\b", between.group(0), re.I):
                raise _Unresolved(f"I couldn't find the sections '{first}' and '{second}'. Name existing headings, or "
                                  "place the cursor where the draft should go.")
        for match in re.finditer(r"\b(?:(" + _AFTER + r")|(" + _BEFORE + r"))\s+([^.;!?\n]{1,160})", task, re.I):
            placement = _target(doc, match.group(3), after=bool(match.group(1)), word=match.group(1) or match.group(2))
            if placement is not None:
                return placement, None
    except _Unresolved as unclear:
        return None, str(unclear)
    return None, None
