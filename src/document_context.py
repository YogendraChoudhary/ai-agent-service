"""Request-scoped document references; exact source text never becomes an edit target implicitly."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .document_edit.blocks import fingerprint, utf16_length
from .longform.markdown_ast import looks_like_html, parse
from .document_literal import literal_draft_text
from .document_placement import DraftPlacement, resolve_placement

DocumentAction = Literal["ask", "draft", "edit"]
# What a request may ask for: "auto" lets the agent pick (src/document_action.py) and is always resolved to one of the above, or to Write.
DocumentActionRequest = Literal["ask", "draft", "edit", "auto"]
DocumentScope = Literal["document", "selection", "focused"]
# What a request may ask for: "auto" is Ask over the whole document when it fits, else over the passages that match the question (never for a
# question that needs the whole document). It is always resolved to "document" or "focused"; the metadata reports which.
DocumentScopeRequest = Literal["document", "selection", "focused", "auto"]
AnalysisMode = Literal["summary", "contradictions"]
if TYPE_CHECKING:
    from .document_analysis import AnalysisPlan
CONTEXT_MAX_BYTES = 24_000
CONTEXT_TOKEN_BUDGET = 6_000
OUTPUT_TOKENS = {"ask": 2_000, "draft": 1_000}
CONTEXT_ERROR = "The document request could not be completed. Please retry."


def source_index(text: str, offset: int) -> int:
    encoded = text.encode("utf-16-le")
    if offset * 2 > len(encoded):
        raise ValueError("Focus position is outside the document")
    try:
        return len(encoded[:offset * 2].decode("utf-16-le"))
    except UnicodeDecodeError:
        raise ValueError("Focus position splits a Unicode character") from None


class ContextRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_utf16: int = Field(ge=0, strict=True)
    end_utf16: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_utf16 < self.start_utf16:
            raise ValueError("Selection end must not precede its start")
        return self


class DocumentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=100_000)
    selection: ContextRange | None = None
    cursor_utf16: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def valid_source(self):
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Document context must contain valid Unicode") from None
        if "\x00" in self.text or looks_like_html(self.text):
            raise ValueError("Supply document context as plain text or Markdown, without NUL characters")
        positions = ([self.selection.start_utf16, self.selection.end_utf16] if self.selection else [])
        if self.cursor_utf16 is not None:
            positions.append(self.cursor_utf16)
        for offset in positions:
            source_index(self.text, offset)
        return self


class DocumentReference(ContextRange):
    id: str = Field(pattern=r'^[re][0-9]+$')
    quote: str = Field(min_length=1, max_length=4000)
    section: list[str] = Field(default_factory=list, max_length=6)


class ContextMetadata(BaseModel):
    action: Literal["ask", "draft"]
    response_kind: Literal["chat_answer", "draft"]
    status: Literal["ready", "needs_context", "needs_clarification"]
    source_sha256: str
    coverage: Literal["full", "partial", "none", "pending"]
    strategy: Literal["full", "focused", "none", "retrieval", "selection", "analysis"]
    scope: DocumentScope = 'document'
    analysis_mode: AnalysisMode | None = None
    source_chars: int
    included_source_chars: int = 0
    context_chars: int = 0
    included_ranges: list[ContextRange] = Field(default_factory=list)
    estimated_input_tokens: int = 0
    billable_input_chars: int = 0
    billable_output_chars: int = 0
    reserved_input_chars: int = 0
    reserved_output_chars: int = 0
    max_stage_calls: int = 0
    references: list[DocumentReference] = Field(default_factory=list)
    # Draft only: where the editor inserts the draft when the task named a place; absent means "at the cursor".
    placement: Optional["DraftPlacement"] = None
    # The reason when status is not "ready" (also streamed as text for needs_context, as before).
    message: Optional[str] = None
    # Draft only: "literal" when the task supplied the exact text ("add Acme Inc. below the Contact heading") and it was
    # returned as written without a model call; "model" when a model writes the draft. null for Ask and whenever no draft is produced
    # (needs_context, needs_clarification).
    source: Optional[Literal["model", "literal"]] = None


@dataclass(frozen=True)
class PreparedContext:
    prompt: str = field(repr=False)
    metadata: ContextMetadata
    message: str | None = None
    analysis: AnalysisPlan | None = field(default=None, repr=False)

    @property
    def warnings(self):
        return ([{"kind": "partial_document_context", "detail":
                  "Only the supplied source passages were included; this is not a whole-document review."}]
                if self.metadata.coverage == "partial" else [])


def reference(text, start, end, identifier, section=()):
    return DocumentReference(id=identifier, start_utf16=utf16_length(text[:start]),
        end_utf16=utf16_length(text[:end]), quote=text[start:end], section=list(section))


def unavailable_context(context, action, scope, message, analysis_mode=None):
    return PreparedContext('', ContextMetadata(action=action, response_kind='draft' if action == 'draft' else 'chat_answer',
        status='needs_context', source_sha256=fingerprint(context.text), source_chars=len(context.text),
        scope=scope, coverage='none', strategy='none', analysis_mode=analysis_mode, message=message), message)


def estimated_tokens(text: str) -> int:
    # shortcut: one UTF-8 byte per token overestimates ordinary prose; calibrate per provider before increasing the budget.
    return len(text.encode("utf-8"))


WHOLE_DOCUMENT_MESSAGE = ("This question needs the whole document, which is too large to read in one request. Ask about a specific part (name a section or "
                          "select the text), or use the document summary or contradiction analysis, which reads every part; no document analysis was run.")


def prepare_context(context: DocumentContext, query: str, action: str = "ask", *,
                    scope: DocumentScopeRequest = 'document', analysis: AnalysisMode | None = None) -> PreparedContext:
    if action not in OUTPUT_TOKENS:
        raise ValueError("Only Ask and Draft use read-only context preparation")
    if scope not in ('document', 'selection', 'focused', 'auto') or (scope != 'document' and action != 'ask'):
        raise ValueError('Scoped retrieval is only available for Ask')
    if analysis is not None:
        if analysis not in ('summary', 'contradictions') or scope != 'document' or action != 'ask':
            raise ValueError('Analysis requires Ask with document scope')
        from .document_analysis import prepare_analysis
        return prepare_analysis(context, query, analysis)
    if scope in ('selection', 'focused'):
        from .document_retrieval import retrieve_context
        return retrieve_context(context, query, scope)
    # Whole document if it fits. If it does not: "auto" (or the server-wide DOCUMENT_LARGE_ASK=auto on an Ask with no scope) answers a question about a
    # part from the matching passages, labelled partial; a question that needs the whole document is refused, as it always was.
    from config import settings
    large_ask_auto = action == 'ask' and (scope == 'auto' or settings.document_large_ask == 'auto')
    text = context.text
    selection = context.selection
    if selection and selection.start_utf16 != selection.end_utf16:
        focus_start = source_index(text, selection.start_utf16)
        focus_end = source_index(text, selection.end_utf16)
    else:
        focus_start = focus_end = source_index(text, context.cursor_utf16) if context.cursor_utf16 is not None else len(text)
    metadata = ContextMetadata(action=action, response_kind="draft" if action == "draft" else "chat_answer",
        status="ready", source_sha256=fingerprint(text), coverage="full", strategy="full", source_chars=len(text),
        source="model" if action == "draft" else None)
    if action == "draft":
        cursor = focus_start if (context.cursor_utf16 is not None or (selection and selection.start_utf16 != selection.end_utf16)) else None
        placement, question = resolve_placement(text, query, cursor)
        if question is not None:
            # A named place that cannot be found: no model call, nothing to insert.
            return PreparedContext("", metadata.model_copy(update={"status": "needs_clarification", "coverage": "none",
                                                                     "strategy": "none", "message": question, "source": None}), question)
        if placement is not None:
            # The draft is written for the anchor, not the cursor: the context is centred there.
            anchor = source_index(text, placement.end_utf16 if placement.position == "after" else placement.start_utf16)
            focus_start = focus_end = anchor
            metadata = metadata.model_copy(update={"placement": placement})
        literal = literal_draft_text(query, re.findall(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", text, re.M))
        if literal is not None:
            # The task gave the exact words: insert them as written. No model call, no document context, nothing to bill.
            return PreparedContext("", metadata.model_copy(update={"coverage": "none", "strategy": "none", "source": "literal"}), literal)

    def envelope(start, end, *, outline=(), coverage="full"):
        cuts = sorted({start, end, max(start, min(end, focus_start)), max(start, min(end, focus_end))})
        excerpts = []
        for left, right in zip(cuts, cuts[1:]):
            excerpts.append({"start_utf16": utf16_length(text[:left]), "end_utf16": utf16_length(text[:right]),
                "relation": "before_focus" if right <= focus_start else "after_focus" if left >= focus_end else "selection",
                "text": text[left:right], "is_excerpt": start != 0 or end != len(text)})
        return json.dumps({"action": action, "source_sha256": metadata.source_sha256, "coverage": coverage,
            "focus": {"start_utf16": utf16_length(text[:focus_start]), "end_utf16": utf16_length(text[:focus_end])},
            "outline": outline, "excerpts": excerpts}, ensure_ascii=False, separators=(",", ":"))

    def fits(data):
        return len(data.encode("utf-8")) <= CONTEXT_MAX_BYTES and estimated_tokens(data) <= CONTEXT_TOKEN_BUDGET

    def unavailable(message):
        return PreparedContext("", metadata.model_copy(update={"status": "needs_context", "coverage": "none", "strategy": "none",
                                                                 "placement": None, "message": message, "source": None}), message)

    if action == "ask" and not text.strip():
        return unavailable("There is no document content to answer questions about. Add some text first.")
    start, end, outline = 0, len(text), []
    data = envelope(start, end)
    if not fits(data):
        if action == "ask":
            if large_ask_auto:
                from .document_question import asks_whole_document
                if asks_whole_document(query):
                    return unavailable(WHOLE_DOCUMENT_MESSAGE)
                from .document_retrieval import retrieve_context
                return retrieve_context(context, query, 'focused', prefer_cursor=False)
            return unavailable("This document is too large to review in full in one request. Send a smaller document or section; no document analysis was run.")
        blocks, at = [], 1 if text.startswith('\ufeff') else 0
        for block in parse(text[at:]):
            blocks.append((at, at + len(block.text), block))
            at += len(block.text)
        headings = [(left, block) for left, _, block in blocks if block.kind == "heading"]
        active = next(((left, block) for left, block in reversed(headings) if left <= focus_start), None)
        section_start = active[0] if active else 0
        section_end = next((left for left, block in headings if left > focus_start and active and block.level <= active[1].level), len(text))
        outline = []
        # Include the active heading even when the beginning of the outline exhausts its allowance.
        for left, block in ([active] if active else []) + [h for h in headings if h != active]:
            item = {"start_utf16": utf16_length(text[:left]), "end_utf16": utf16_length(text[:left + min(80, len(block.text))]), "level": block.level,
                    "text": block.text[:80], "is_excerpt": len(block.text) > 80}
            if len(json.dumps([*outline, item], ensure_ascii=False).encode("utf-8")) > 1_000:
                break
            outline.append(item)
        outline.sort(key=lambda item: item["start_utf16"])
        start, end = section_start, section_end
        data = envelope(start, end, outline=outline, coverage="partial")
        if not fits(data):
            low, high = 0, len(text)
            while low < high:
                radius = (low + high + 1) // 2
                left, right = max(section_start, focus_start - 3 * radius), min(section_end, focus_start + radius)
                if fits(envelope(left, right, outline=outline, coverage="partial")):
                    low = radius
                else:
                    high = radius - 1
            start, end = max(section_start, focus_start - 3 * low), min(section_end, focus_start + low)
            # Prefer complete paragraphs when their boundaries still leave text around the focus.
            start = next((left for left, _, _ in blocks if start <= left < focus_start), start)
            end = next((right for _, right, _ in reversed(blocks) if focus_start < right <= end), end)
            data = envelope(start, end, outline=outline, coverage="partial")
        if not fits(data) or start == end:
            return unavailable("The focused document context could not fit. Send a smaller section.")
        metadata = metadata.model_copy(update={"coverage": "partial", "strategy": "focused"})
    prompt = data + "\n\nUSER_TASK:\n" + query
    ranges = [(start, end)] if end > start else []
    ranges.extend((source_index(text, h['start_utf16']), source_index(text, h['end_utf16'])) for h in outline)
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    metadata = metadata.model_copy(update={"included_source_chars": sum(right - left for left, right in merged), "context_chars": len(data),
        "included_ranges": [ContextRange(start_utf16=utf16_length(text[:left]), end_utf16=utf16_length(text[:right])) for left, right in merged],
        "estimated_input_tokens": estimated_tokens(prompt), "billable_input_chars": len(data) + len(query)})
    return PreparedContext(prompt, metadata)


ContextMetadata.model_rebuild()
