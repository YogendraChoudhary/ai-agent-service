"""Exact source blocks. Python owns boundaries and preserves every non-target byte."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

from markdown_it import MarkdownIt

from src.longform.markdown_ast import looks_like_html, parse


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


# The most a model rewrites in one piece (DocumentPatch.before has the same limit).
MAX_BLOCK_CHARS = 12_000
# Blocks Edit accepts, blank lines not counted (audit 2026-11-20: a 2.6 KB handbook of 41 short sections had 402 blocks, about
# half of them blank lines, and Edit refused it). The cap bounds the planner's catalog and the patch checks, which never see blanks.
MAX_DOCUMENT_BLOCKS = 400
# A list over MAX_BLOCK_CHARS is rewritten as runs of whole items of about this size (edit_units), each its own model call.
LIST_RUN_CHARS = 4_000


@dataclass(frozen=True)
class DocumentBlock:
    id: str
    kind: str
    section: tuple[str, ...]
    start: int
    end: int
    text: str
    parent: str | None = None         # edit_units(): the id of the long list this run of items belongs to

    @property
    def editable(self) -> bool:
        """Prose: a paragraph or list that any request about its section may rewrite."""
        return self.kind in ("paragraph", "list") and len(self.text) <= MAX_BLOCK_CHARS

    @property
    def span_editable(self) -> bool:
        """
        A paragraph or list of any length: a patch may replace an exact range inside it (at most MAX_BLOCK_CHARS of `before`), so
        "Change X to Y" and selection edits work inside a 13.8 KB list that no model may rewrite whole.
        """
        return self.kind in ("paragraph", "list")

    @property
    def rewritable(self) -> bool:
        """
        Whether a patch may replace this block. Prose, plus headings and plain-text quotes, which allowed_targets() offers only when
        the instruction is about them (a title, a heading, a quote), so "make the intro shorter" never touches the heading above it.
        """
        if self.editable:
            return True
        if len(self.text) > MAX_BLOCK_CHARS:
            return False
        if self.kind == "heading":
            return True
        return self.kind == "blockquote" and quote_is_plain(self.text)


_QUOTE_PREFIX = re.compile(r"^( {0,3}>[ \t]?)")


def unquote(text: str) -> tuple[str, list[str]]:
    """A quote block as plain prose: its text without the leading '>' of each line, and those prefixes in order."""
    inner, prefixes = [], []
    for line in text.splitlines(keepends=True):
        match = _QUOTE_PREFIX.match(line)
        prefixes.append(match.group(1) if match else "")
        inner.append(line[match.end():] if match else line)
    return "".join(inner), prefixes


def requote(inner: str, prefixes: list[str]) -> str | None:
    """The quote block again, each line with the prefix it had; None when the line count changed (a model merged or split lines)."""
    lines = inner.splitlines(keepends=True)
    if len(lines) != len(prefixes):
        return None
    return "".join(prefix + line for prefix, line in zip(prefixes, lines))


def quote_is_plain(text: str) -> bool:
    """A quote of paragraphs and lists only: no nested quote, code, table, image, raw HTML or heading inside it."""
    inner, prefixes = unquote(text)
    if not inner.strip() or not all(prefixes[:1]) or ">" in "".join(line.lstrip()[:1] for line in inner.splitlines()):
        return False
    return all(b.kind in ("paragraph", "list", "blank") for b in parse(inner))


def document_blocks(text: str, *, max_blocks: int | None = MAX_DOCUMENT_BLOCKS) -> list[DocumentBlock]:
    """The document as exact top-level blocks. `max_blocks` None: no cap (read-only placement, which makes no model call per block)."""
    if not text.strip() or len(text) > 100_000 or looks_like_html(text) or "\x00" in text:
        raise ValueError("Supply a non-empty plain-text or Markdown document of at most 100,000 characters")
    result, headings, start = [], [], 0
    inline_parser = MarkdownIt("commonmark")
    if text.startswith("\ufeff"):
        result.append(DocumentBlock(f"b0_{fingerprint(text[0])[:12]}", "other", (), 0, 1, text[0]))
        start = 1
    for i, block in enumerate(parse(text[start:]), len(result)):
        if block.kind == "heading":
            heading = re.sub(r"^#{1,6}\s*|\s*#+\s*$", "", block.text.splitlines()[0]).strip()
            children = inline_parser.parseInline(heading)[0].children or []
            heading = "".join(t.content for t in children if t.type in ("text", "code_inline", "image"))
            headings = [(level, value) for level, value in headings if level < block.level]
            headings.append((block.level, heading))
        end = start + len(block.text)
        result.append(DocumentBlock(f"b{i}_{fingerprint(block.text)[:12]}", block.kind,
                                    tuple(value for _, value in headings), start, end, block.text))
        start = end
    if max_blocks is not None and sum(b.kind != "blank" for b in result) > max_blocks:
        raise ValueError(f"This document has too many blocks; use a smaller document (at most {max_blocks} paragraphs, headings, "
                         "lists and other blocks; blank lines do not count)")
    assert "".join(b.text for b in result) == text
    return result


_LINES = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")         # markdown-it's lines (not str.splitlines, which splits on more)


@lru_cache(maxsize=8)
def _item_starts(text: str) -> tuple[int, ...] | None:
    """
    Where each top-level item of a list block starts (character offsets), or None when the block does not open with an item.
    Cached: applying 24 patches to one 91 KB list parsed it 24 times (0.6 s).
    """
    lines = _LINES.findall(text)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = MarkdownIt("commonmark").parse(text)
    starts = [offsets[t.map[0]] for t in tokens if t.type == "list_item_open" and t.level == 1 and t.map]
    return tuple(starts) if starts and starts[0] == 0 else None


def edit_units(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """
    The blocks a model rewrite may target: every block, except that a list over MAX_BLOCK_CHARS becomes runs of whole top-level items
    of about LIST_RUN_CHARS each (audit 2026-11-20: a 13.8 KB meeting-notes list of 160 items could not be proofread at all). A run
    stops at its last non-blank line, so the blank lines between runs are never sent and never change; a patch for a run replaces
    that range of the parent list (`parent`). A single item over MAX_BLOCK_CHARS stays too long to rewrite.
    """
    units = []
    for block in blocks:
        starts = _item_starts(block.text) if block.kind == "list" and len(block.text) > MAX_BLOCK_CHARS else None
        if starts is None:
            units.append(block)
            continue
        runs, first = [], 0
        for item_start, item_end in zip(starts, starts[1:] + (len(block.text),)):
            # Close the run before an item that would take it past LIST_RUN_CHARS; one long item is a run of its own.
            if item_end - first > LIST_RUN_CHARS and item_start > first:
                runs.append((first, item_start))
                first = item_start
        runs.append((first, len(block.text)))
        index = block.id.split("_", 1)[0]
        for first, last in runs:
            lines = _LINES.findall(block.text[first:last])
            while len(lines) > 1 and not lines[-1].strip():
                lines.pop()
            text = "".join(lines)
            start = block.start + first
            units.append(DocumentBlock(f"{index}_{fingerprint(f'{start}:{text}')[:12]}", "list", block.section, start,
                                       start + len(text), text, parent=block.id))
    return units


def item_window(block: DocumentBlock, start: int, end: int) -> DocumentBlock:
    """
    The whole top-level items of a list over MAX_BLOCK_CHARS that hold the range [start, end) of its text, as a block of their own:
    what a patch of that range is checked against. Items are independent Markdown (nothing inline runs from one into the next), so
    checking them is checking the patch, and 24 patches into a 91 KB list no longer re-check the whole list each (4.0 s, measured).
    Any other block is its own window.
    """
    starts = _item_starts(block.text) if block.kind == "list" and len(block.text) > MAX_BLOCK_CHARS else None
    if starts is None:
        return block
    low = max(s for s in starts if s <= start)
    high = min((s for s in starts if s >= end), default=len(block.text))
    return DocumentBlock(block.id, block.kind, block.section, block.start + low, block.start + high, block.text[low:high])


# What the instruction says about headings and quotes. They are rewritten only when asked for, never as part of "make it shorter".
_HEADING_WORDS = re.compile(r"\b(?:headings?|headlines?|titles?|subtitles?|sub-?headings?|(?:section|chapter)\s+(?:names?|titles?)|"
                            r"re-?titles?|rename)\b", re.I)
_TITLE_ONLY = re.compile(r"\b(?:the\s+(?:document|page|main|report|article|post)\s+title|the\s+title|title\s+of\s+the\s+"
                         r"(?:document|page|report|article|post)|(?:re-?title|rename)\s+the\s+(?:document|page|report|article|post))\b", re.I)
_ALL_HEADINGS = re.compile(r"\b(?:headings|headlines|subtitles|sub-?headings|titles|section\s+(?:names|titles)|"
                           r"(?:all|every|each|both)\s+(?:the\s+)?(?:headings?|headlines?|titles?))\b", re.I)
_QUOTE_WORDS = re.compile(r"\b(?:quotes?|quotations?|block-?quotes?|pull-?quotes?)\b", re.I)
# Clauses that say what NOT to touch: their heading/quote words do not ask for anything. A clause runs to the next "but/then", to an
# "and"/comma that starts a new instruction (an action verb), or to the end of the sentence, so "leave the title and the quote alone" keeps
# both, while "keep the quote but make the title shorter" still asks about the title.
_ACTION_VERB = (r"(?:make|shorten|rewrite|rephrase|reword|fix|improve|rename|retitle|change|edit|tighten|expand|translate|simplify|add|"
                r"remove|delete|polish|proofread|correct|update|revise|lengthen|condense|clarify|summari[sz]e)")
_EXCEPTIONS = re.compile(
    r"\b(?:without|except|excluding|apart\s+from|other\s+than|leave|leaving|keep|keeping|preserve|preserving|skip|skipping|ignore|"
    r"ignoring|do\s+not|don['’]t|never|must\s+not|should\s+not|shouldn['’]t|avoid|avoiding|not)\b(?:\s+[^\s.;\n]+)+?"
    r"(?=\s+(?:but|then)\b|\s+(?:and\s+)?(?:please\s+)?" + _ACTION_VERB + r"\b|\s*[.;\n]|\s*$)", re.I)
_TITLE_KEPT = re.compile(r"\b(?:headline|title)\b", re.I)


def _level(block: DocumentBlock) -> int:
    atx = re.match(r"\s{0,3}(#{1,6})(?:\s|$)", block.text)
    if atx:
        return len(atx.group(1))
    lines = block.text.split("\n")
    return 1 if len(lines) > 1 and lines[1].strip().startswith("=") else 2


def allowed_targets(instruction: str, blocks: list[DocumentBlock]) -> tuple[set[str], str | None]:
    """Enforce named/protected headings independently of the model's proposed targets."""
    headings = {heading for b in blocks for heading in b.section if len(heading) >= 3}

    def mentioned(heading, in_text):
        return re.search(r"(?<!\w)" + re.escape(heading) + r"(?!\w)", in_text, re.I) is not None

    protected_clauses = [m.group(0) for m in re.finditer(
        r"\b(?:leave|leaving|keep|keeping)\b[^.;\n]*?\b(?:unchanged|intact|as is)\b|"
        r"\b(?:do not|don't|never)\s+(?:change|edit|touch|modify)\b[^.;\n]*", instruction, re.I)]
    protected = {h for h in headings if any(mentioned(h, c) for c in protected_clauses)}
    positive = instruction
    for clause in protected_clauses:
        positive = positive.replace(clause, "")
    named = {h for h in headings if mentioned(h, positive)}
    selected_paths = []
    for heading in named:
        paths = [b.section for b in blocks if b.kind == "heading" and b.section[-1:] == (heading,)]
        if len(paths) > 1 and not re.search(r"\b(?:all|both|every)\b", positive, re.I):
            # A parent heading in the same request disambiguates nested sections.
            matching = [p for p in paths if any(mentioned(parent, positive) for parent in p[:-1])]
            if len(matching) != 1:
                return set(), f"There is more than one '{heading}' section. Name its parent section or ask for all occurrences."
            paths = matching
        selected_paths.extend(paths)
    qualifiers = {h for h in named if re.search(r"\b(?:under|within|inside|in)\s+(?:the\s+)?" + re.escape(h) + r"(?!\w)", positive, re.I)}
    selected_paths = [path for path in selected_paths if not (path[-1] in qualifiers and any(
        len(other) > len(path) and other[:len(path)] == path for other in selected_paths))]
    broad = re.search(r"\b(?:throughout|entire|whole|other|all|every|wherever)\b", positive, re.I)

    def in_scope(b):
        return not (set(b.section) & protected) and (not named or broad or any(b.section[:len(path)] == path for path in selected_paths))

    # Headings and quotes only when the instruction is about them (words inside "leave/without/except …" clauses do not count).
    # (read from the whole instruction, not from `positive`: the older protected-clause rule above runs to the end of the sentence and
    # would swallow "… , rename the Office heading" after "Don't touch the quote")
    asked = _EXCEPTIONS.sub(" ", instruction)
    title_kept = any(_TITLE_KEPT.search(m.group(0)) for m in _EXCEPTIONS.finditer(instruction))
    kinds = set()
    if _HEADING_WORDS.search(asked):
        kinds.add("heading")
    if _QUOTE_WORDS.search(asked):
        kinds.add("blockquote")
    top = [b for b in blocks if b.kind == "heading" and _level(b) == 1] or [b for b in blocks if b.kind == "heading"][:1]
    title_only = "heading" in kinds and _TITLE_ONLY.search(asked) and not _ALL_HEADINGS.search(asked)

    def asked_for(b):
        if b.kind == "heading":
            # "the title" is the document's top-level heading; naming a heading ("rename Contact") scopes to it by in_scope().
            return "heading" in kinds and b.rewritable and (not title_only or b in top[:1]) and not (title_kept and b in top[:1])
        return b.kind in kinds and b.rewritable

    allowed = {b.id for b in blocks if (b.editable or asked_for(b)) and in_scope(b)}
    return allowed, None
