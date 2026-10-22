"""
Top-level Markdown blocks with exact line ranges.

markdown-it (CommonMark plus GFM tables) says where every top-level block starts and ends. We slice the ORIGINAL
text by those lines and never re-render it, so join(parse(text)) == text for any input. That is what lets code,
not the model, own structure: tables, quotes and images are found, counted, moved and checked byte for byte.

Block kinds:
  heading  paragraph  image (a figure: one image, optionally with a short caption line)  table  blockquote
  list  fence (fenced or indented code)  html  hr  blank  other (e.g. link reference definitions)
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from markdown_it import MarkdownIt

KINDS = ("heading", "paragraph", "image", "table", "blockquote", "list", "fence", "html", "hr", "blank", "other")
PROSE_KINDS = ("paragraph", "list", "blockquote")
CAPTION_MAX_WORDS = 40

_KIND_OF_TOKEN = {
    "heading_open": "heading", "paragraph_open": "paragraph", "table_open": "table",
    "blockquote_open": "blockquote", "bullet_list_open": "list", "ordered_list_open": "list",
    "fence": "fence", "code_block": "fence", "html_block": "html", "hr": "hr",
}
_CAPTION_TOKENS = {"softbreak", "hardbreak", "text", "em_open", "em_close", "strong_open", "strong_close"}
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")
_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
_PIPE_RE = re.compile(r"(?<!\\)\|")

_local = threading.local()


def _parser() -> MarkdownIt:
    # One parser per thread: blog units are linted from worker threads at the same time.
    md = getattr(_local, "md", None)
    if md is None:
        md = _local.md = MarkdownIt("commonmark").enable("table")
    return md


@dataclass(frozen=True)
class Block:
    kind: str
    start: int                      # first line, 0-based
    end: int                        # one past the last line
    text: str                       # the source lines [start, end) exactly as written
    level: int = 0                  # heading level
    words: int = 0                  # prose words: inline text and code spans (not URLs, image alt text or code blocks)
    images: int = 0                 # images inside the block
    items: int = 0                  # list: top-level items
    ordered: bool = False           # list: numbered
    depths: Tuple[int, ...] = ()    # list: nesting depth of every item, in order


@dataclass(frozen=True)
class TableShape:
    header: int                     # cells in the header row
    delimiter: int                  # cells in the |---|---| row
    rows: Tuple[int, ...]           # cells in every body row

    @property
    def consistent(self) -> bool:
        return all(n == self.header for n in (self.delimiter, *self.rows))


def word_count(text: Optional[str]) -> int:
    """Words in plain text; a hyphenated or apostrophised word counts once."""
    return len(_WORD_RE.findall(text or ""))


def _lines(text: str) -> List[str]:
    return _LINE_RE.findall(text)


def _inline_children(group) -> list:
    return [c for t in group if t.type == "inline" for c in (t.children or ())]


def _is_figure(group) -> bool:
    inline = next((t for t in group if t.type == "inline"), None)
    kids = list(inline.children or ()) if inline is not None else []
    if not kids or kids[0].type != "image" or sum(c.type == "image" for c in kids) != 1:
        return False
    rest = kids[1:]
    return all(c.type in _CAPTION_TOKENS for c in rest) and \
        sum(word_count(c.content) for c in rest if c.type == "text") <= CAPTION_MAX_WORDS


def _make_block(first, group, lines: List[str], start: int, end: int) -> Block:
    kind = _KIND_OF_TOKEN[first.type]
    if kind == "paragraph" and _is_figure(group):
        kind = "image"
    kids = _inline_children(group)
    words = sum(word_count(c.content) for c in kids if c.type in ("text", "code_inline"))
    images = sum(c.type == "image" for c in kids)
    level = int(first.tag[1]) if kind == "heading" else 0
    items, ordered, depths = 0, False, ()
    if kind == "list":
        ordered = first.type == "ordered_list_open"
        opened = [t for t in group if t.type == "list_item_open"]
        items = sum(t.level == 1 for t in opened)
        depths = tuple((t.level - 1) // 2 for t in opened)
    return Block(kind, start, end, "".join(lines[start:end]), level=level, words=words, images=images,
                 items=items, ordered=ordered, depths=depths)


def parse(text: str) -> List[Block]:
    """Every line of `text` belongs to exactly one block, in order, so join(parse(text)) == text."""
    lines = _lines(text or "")
    if not lines:
        return []
    tokens = _parser().parse(text)
    spans: list[Block] = []
    i = 0
    while i < len(tokens):
        first = tokens[i]
        if first.nesting == 1:
            j = i + 1
            while not (tokens[j].nesting == -1 and tokens[j].level == first.level):
                j += 1
            group, i = tokens[i:j + 1], j + 1
        else:
            group, i = [first], i + 1
        if first.map is not None and first.type in _KIND_OF_TOKEN:
            spans.append(_make_block(first, group, lines, first.map[0], min(first.map[1], len(lines))))

    blocks: list[Block] = []
    cursor = 0

    def fill(upto: int) -> None:
        if upto > cursor:
            gap = "".join(lines[cursor:upto])
            blocks.append(Block("blank" if not gap.strip() else "other", cursor, upto, gap))

    for b in spans:
        if b.start < cursor or b.end <= b.start:
            continue                                  # defensive: a block that overlaps the previous one
        fill(b.start)
        blocks.append(b)
        cursor = b.end
    fill(len(lines))
    if join(blocks) != text:                          # never hand back something that is not the input
        return [Block("other", 0, len(lines), text)]
    return blocks


def join(blocks: Sequence[Block]) -> str:
    return "".join(b.text for b in blocks)


def shape(blocks: Sequence[Block]) -> Tuple[str, ...]:
    """The sequence of block kinds, blank lines left out."""
    return tuple(b.kind for b in blocks if b.kind != "blank")


def heading_levels(blocks: Sequence[Block]) -> Tuple[int, ...]:
    return tuple(b.level for b in blocks if b.kind == "heading")


def list_shape(blocks: Sequence[Block]) -> Tuple[Tuple[int, ...], ...]:
    """For every list, the nesting depth of each of its items."""
    return tuple(b.depths for b in blocks if b.kind == "list")


def prose_words(blocks: Sequence[Block], kinds: Sequence[str] = PROSE_KINDS) -> int:
    """Words a reader reads as text. Tables, code, HTML, headings and image captions are not counted."""
    return sum(b.words for b in blocks if b.kind in kinds)


def table_cells(line: str) -> List[str]:
    """The cells of one table line, without the outer pipes (an escaped \\| does not split)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in _PIPE_RE.split(s)]


def table_shape(block: Block) -> Optional[TableShape]:
    """Cell counts of a table block taken from its source lines; None when it is not a table."""
    if block.kind != "table":
        return None
    lines = [ln for ln in block.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    return TableShape(header=len(table_cells(lines[0])), delimiter=len(table_cells(lines[1])),
                      rows=tuple(len(table_cells(ln)) for ln in lines[2:]))


# ---------------------------------------------------------------------------
# Normalise: what the pipeline works on, and how the answer is put back
# ---------------------------------------------------------------------------

_HTML_LINE = re.compile(r"^\s*<(?:p|div|h[1-6]|table|ul|ol|li|blockquote|img|section|article|figure|pre|span|br|hr)\b[^>]*>",
                        re.IGNORECASE)


@dataclass(frozen=True)
class Style:
    """What normalise() took off a document, so the answer can be returned the way the document came."""
    bom: bool = False
    crlf: bool = False


def normalise(text: Optional[str]) -> Tuple[str, Style]:
    """LF line endings and no byte-order mark for the pipeline; restyle() puts both back on the way out."""
    text = text or ""
    bom = text.startswith("\ufeff")
    if bom:
        text = text[1:]
    crlf = text.count("\r\n") * 2 > text.count("\n")                 # most of its line breaks are CRLF
    return text.replace("\r\n", "\n").replace("\r", "\n"), Style(bom=bom, crlf=crlf)


def restyle(text: str, style: Style, *, first: bool = False) -> str:
    """`text` with the document's own line endings back; `first` marks the chunk that carries the byte-order mark."""
    if style.crlf:
        text = text.replace("\n", "\r\n")
    return "\ufeff" + text if style.bom and first else text


def looks_like_html(text: Optional[str]) -> bool:
    """True when the document is HTML rather than Markdown: most of its non-blank lines open with a tag (open item O1)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    if re.match(r"\s*<(?:!doctype|html|body)\b", text, re.IGNORECASE):
        return True
    return sum(bool(_HTML_LINE.match(ln)) for ln in lines) > 0.6 * len(lines)

