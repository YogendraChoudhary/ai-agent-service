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
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from markdown_it import MarkdownIt
from yaml import YAMLError, parse as parse_yaml
from yaml.events import DocumentEndEvent, MappingStartEvent, StreamEndEvent

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

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


def _block_parser() -> MarkdownIt:
    # _parser() without the inline pass (see heading_lines).
    md = getattr(_local, "block_md", None)
    if md is None:
        md = _local.block_md = MarkdownIt("commonmark").enable("table").disable("inline")
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
    flavor: str = ""                # kind "other" only: what a dialect block is ("front_matter", "math", "footnote"), see _dialect_regions


# ---------------------------------------------------------------------------
# Dialects beyond CommonMark + tables
#
# YAML front matter, display math and footnote definitions are not CommonMark: markdown-it reads front matter as a thematic break
# plus a setext heading, and math and footnote text as ordinary paragraphs. Left alone, a model could "fix" the grammar of a
# YAML title or an equation, and the upload would list "title: My Post" as a heading. So they are found here, in code, before
# markdown-it runs, and come back as kind "other": the kind freeze already makes opaque (link reference definitions use it) and
# Edit never touches. Their lines are blanked for markdown-it so every line number stays the same.
#
# The inline constructs (math, wikilinks, footnote references, ~~strikethrough~~, :emoji: codes, task-list boxes) are matched by
# the patterns below; freeze.py swaps each for an opaque token. Every repetition is bounded so a crafted line cannot make a
# scan quadratic.
# ---------------------------------------------------------------------------

FRONT_MATTER_MAX_LINES = 300
MATH_BLOCK_MAX_LINES = 200

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FM_TOML_LINE = re.compile(r"""^(?:[\w."'\-]{1,80}\s*=|\[{1,2}[^\]\n]{1,80}\]{1,2}\s*$)""")
_FOOTNOTE_DEF = re.compile(r"^ {0,3}\[\^[^\]\s]{1,60}\]:")

INLINE_DIALECT_PATTERNS = {
    "math": r"\$\$[^$\n]{1,200}?\$\$|(?<![\\$\w])\$(?![\s$])[^$\n]{1,200}?(?<![\s\\])\$(?![\w$])",
    "wikilink": r"!?\[\[(?!(?:TABLE|QUOTE|IMAGE|CODE|HTML|URL)_\d+\]\])[^\[\]\n]{1,200}\]\]",
    "footnote_ref": r"\[\^[^\]\s]{1,60}\](?!:)",
    "strikethrough": r"(?<![~\\])~~(?![\s~])[^~\n]{1,300}?(?<![\s~])~~(?!~)",
    "emoji_code": r"(?<![\w:]):[a-z][a-z0-9_+\-]{1,40}:(?![\w:])",
    "attributes": r"\{(?:#[\w][\w\-:.]{0,59}|\.[A-Za-z_][\w\-:.]{0,59})(?:[ \t]+(?:#|\.)?[A-Za-z_][\w\-:.]{0,59}(?:=[^\s}]{1,60})?){0,8}\}",      # {#id .class key=value}
}
# "> [!NOTE]" (a GitHub alert): the keyword is a code word, not wording, so it must not be translated or reworded.
ALERT_MARKER = re.compile(r"^((?:>[ \t]*)+)(\[![A-Za-z]{2,20}\])", re.M)
# "[^1]:" at the start of a footnote definition: the label is a code word, the text after it is prose (a translation translates it).
FOOTNOTE_LABEL = re.compile(r"^(\[\^[^\]\s]{1,60}\]:)", re.M)
TASK_MARKER = re.compile(r"^([ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+)(\[[ xX]\])(?=[ \t]|$)", re.M)
_TASK_CONTENT = re.compile(r"^\[[ xX]\](?=[ \t]|$)")
_INLINE_FEATURE = {"math": "math", "wikilink": "wikilinks", "footnote_ref": "footnotes", "strikethrough": "strikethrough",
                   "emoji_code": "emoji_codes", "attributes": "attributes"}
_INLINE_RE = {name: re.compile(p) for name, p in INLINE_DIALECT_PATTERNS.items()}
FEATURE_ORDER = ("front_matter", "math", "footnotes", "task_list", "wikilinks", "strikethrough", "attributes", "emoji_codes")
_FLAVOR_FEATURE = {"front_matter": "front_matter", "math": "math", "footnote": "footnotes"}


def _front_matter_end(lines: Sequence[str]) -> Optional[int]:
    """One past the closing delimiter when the document opens with YAML (---) or TOML (+++) front matter, else None."""
    if not lines:
        return None
    opener = lines[0].removeprefix("\ufeff").rstrip()
    if opener == "---":
        for i in range(1, min(len(lines), FRONT_MATTER_MAX_LINES)):
            if lines[i].rstrip() in ("---", "..."):
                # Read parser events only, so tags and aliases are never resolved and the original text stays intact.
                try:
                    events = parse_yaml("".join(lines[1:i]), Loader=SafeLoader)
                    next(events, None)                     # stream start
                    next(events, None)                     # document start, if nonempty
                    if isinstance(next(events, None), MappingStartEvent):
                        for event in events:
                            if isinstance(event, DocumentEndEvent):
                                return i + 1 if isinstance(next(events, None), StreamEndEvent) else None
                except YAMLError:
                    pass
                return None
        return None
    if opener != "+++":
        return None
    seen_entry = False
    for i in range(1, min(len(lines), FRONT_MATTER_MAX_LINES)):
        line = lines[i].rstrip("\r\n")
        if line.rstrip() == "+++":
            return i + 1 if seen_entry else None
        if not line.strip() or line[0] in " \t" or line.lstrip().startswith("#"):
            continue                                        # blank, nested/continued, or a comment
        if _FM_TOML_LINE.match(line):
            seen_entry = True
            continue
        return None
    return None


def is_front_matter(block_text: str) -> bool:
    """True when `block_text` is exactly a front matter block (what parse() returns as kind "other", flavor "front_matter")."""
    lines = _lines(block_text or "")
    return bool(lines) and _front_matter_end(lines) == len(lines)


def _math_block_end(lines: Sequence[str], i: int) -> Optional[int]:
    """One past the last line of the display-math block that opens on lines[i] (`$$ ... $$` or `\\[ ... \\]`), else None."""
    first = lines[i].strip()
    if first.startswith("$$"):
        rest = first[2:]
        if rest.endswith("$$"):
            return i + 1 if rest.count("$$") == 1 else None  # one line: $$ E = mc^2 $$
        if "$$" in rest:
            return None
        for j in range(i + 1, min(len(lines), i + 1 + MATH_BLOCK_MAX_LINES)):
            if lines[j].rstrip().endswith("$$"):
                return j + 1
        return None
    if first == "\\[":
        for j in range(i + 1, min(len(lines), i + 1 + MATH_BLOCK_MAX_LINES)):
            if lines[j].strip() == "\\]":
                return j + 1
    return None


def _footnote_end(lines: Sequence[str], i: int) -> int:
    """One past the last line of the footnote definition on lines[i]: its indented continuation lines, blank ones included."""
    end, j = i + 1, i + 1
    while j < len(lines):
        line = lines[j]
        if line.strip() and (line.startswith("  ") or line.startswith("\t")):
            end = j = j + 1
        elif not line.strip():
            j += 1                                          # a blank line belongs to the footnote only if an indented line follows
        else:
            break
    return end


def _dialect_regions(lines: Sequence[str]) -> List[Tuple[int, int, str]]:
    """(first line, one past the last, flavor) of each top-level front matter, display math block and footnote definition."""
    regions: List[Tuple[int, int, str]] = []
    i, n = 0, len(lines)
    fm_end = _front_matter_end(lines)
    if fm_end:
        regions.append((0, fm_end, "front_matter"))
        i = fm_end
    fence: Optional[Tuple[str, int]] = None
    while i < n:
        line = lines[i]
        if fence is not None:
            m = _FENCE_OPEN.match(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not line.strip(" \t\r\n`~"):
                fence = None
            i += 1
            continue
        m = _FENCE_OPEN.match(line)
        if m:
            fence = (m.group(1)[0], len(m.group(1)))
            i += 1
            continue
        if line.strip() and not line.startswith("\t") and len(line) - len(line.lstrip(" ")) < 4:
            end = _math_block_end(lines, i) if line.lstrip().startswith(("$$", "\\[")) else None
            flavor = "math"
            if end is None and _FOOTNOTE_DEF.match(line):
                end, flavor = _footnote_end(lines, i), "footnote"
            if end is not None:
                regions.append((i, end, flavor))
                i = end
                continue
        i += 1
    return regions


def _clip(span: Block, regions: Sequence[Tuple[int, int, str]], starts: Sequence[int], lines: Sequence[str],
          masked: "set[int]") -> Block:
    """
    markdown-it ends a list (or any block that runs to the end of the document) after the last blank line, and a masked dialect
    block looks blank to it, so the span can swallow a region that follows it. Cut such a span back to where its real content ends.
    A region with real content after it inside the same span (display math between two list items) is left to the overlap rule.
    """
    i = bisect_right(starts, span.start)
    while i < len(regions) and regions[i][0] < span.end:
        first = regions[i][0]
        if all(k in masked or not lines[k].strip() for k in range(first, span.end)):
            return replace(span, end=first, text="".join(lines[span.start:first]))
        i += 1
    return span


def _masked(lines: Sequence[str], regions: Sequence[Tuple[int, int, str]]) -> str:
    """The document with every dialect line blank, for markdown-it: line numbers do not move."""
    out = list(lines)
    for first, end, _ in regions:
        for k in range(first, end):
            out[k] = "\n" if lines[k].endswith(("\n", "\r")) else ""
    return "".join(out)


@dataclass(frozen=True)
class TableShape:
    header: int                     # cells in the header row
    delimiter: int                  # cells in the |---|---| row
    rows: Tuple[int, ...]           # cells in every body row

    @property
    def consistent(self) -> bool:
        return all(n == self.header for n in (self.delimiter, *self.rows))


_CJK_CHAR = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")       # Hiragana, Katakana, Han
_THAI_CHAR = re.compile("[\u0e00-\u0e7f]")


def script_share(text: Optional[str]) -> float:
    """The share of letters written in Han, Kana or Thai (0.0 for most text): a translation that crosses it changes length a lot."""
    text = text or ""
    cjk_or_thai = len(_CJK_CHAR.findall(text)) + len(_THAI_CHAR.findall(text))
    return cjk_or_thai / max(1, len(re.findall(r"[^\W\d_]", text))) if cjk_or_thai else 0.0


def word_count(text: Optional[str]) -> int:
    """
    Words in plain text; a hyphenated or apostrophised word counts once. Chinese, Japanese and Thai are written without spaces, so a
    whole sentence would be one "word": there a Han, Kana or Thai character counts as about 0.55 (Chinese, Japanese) or a quarter
    (Thai) of a word. That keeps chunk sizes and length windows meaningful when a document is translated into or out of those
    languages. Text with none of those characters is counted exactly as before.
    """
    text = text or ""
    found = _WORD_RE.findall(text)
    if not (_CJK_CHAR.search(text) or _THAI_CHAR.search(text)):
        return len(found)
    total = 0
    for word in found:
        cjk, thai = len(_CJK_CHAR.findall(word)), len(_THAI_CHAR.findall(word))
        if not cjk and not thai:
            total += 1
        else:
            total += max(1, (cjk * 5 + 8) // 9 + (thai + 3) // 4 + (1 if len(word) > cjk + thai else 0))
    return total


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
    regions = _dialect_regions(lines)
    tokens = _parser().parse(_masked(lines, regions) if regions else text)
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
    if regions:
        starts = [r[0] for r in regions]
        masked = {k for a, b, _ in regions for k in range(a, b)}
        spans = [_clip(s, regions, starts, lines, masked) for s in spans]
        spans.extend(Block("other", a, b, "".join(lines[a:b]), flavor=flavor) for a, b, flavor in regions)
        spans.sort(key=lambda s: s.start)                 # a dialect block inside a list is dropped by the overlap rule below

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


def heading_lines(text: str) -> List[Tuple[int, str]]:
    """
    (level, first source line) of every top-level heading: the same as
    [(b.level, b.text.splitlines()[0]) for b in parse(text) if b.kind == "heading"], found from block structure alone.

    parse() also runs markdown-it's inline pass, which only feeds word and image counts and is most of its cost
    (about 0.8 s for a crafted 100,000-character document of link brackets, against a few ms without it).
    """
    return scan(text).headings


@dataclass(frozen=True)
class Scan:
    headings: List[Tuple[int, str]]         # (level, first source line) of every top-level heading
    features: List[str]                     # the dialect features the document uses, in FEATURE_ORDER


def scan(text: str) -> Scan:
    """
    Headings and dialect features in one block-structure pass (no inline pass, see heading_lines). The features are what
    CommonMark alone does not know: front matter, math, footnotes, task lists, wikilinks, strikethrough and :emoji: codes. Code
    fences are never searched, so `$x$` in a code sample is not math.
    """
    lines = _lines(text or "")
    headings: List[Tuple[int, str]] = []
    if not lines:
        return Scan(headings, [])
    regions = _dialect_regions(lines)
    found = {_FLAVOR_FEATURE[flavor] for _, _, flavor in regions}
    list_depth = 0
    for tok in _block_parser().parse(_masked(lines, regions) if regions else text):
        if tok.type == "heading_open" and tok.level == 0 and tok.map is not None:
            start, end = tok.map[0], min(tok.map[1], len(lines))
            if end > start:
                headings.append((int(tok.tag[1]), "".join(lines[start:end]).splitlines()[0]))
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif tok.type == "inline" and tok.content:
            for name, rx in _INLINE_RE.items():
                if _INLINE_FEATURE[name] not in found and rx.search(tok.content):
                    found.add(_INLINE_FEATURE[name])
            if list_depth and "task_list" not in found and _TASK_CONTENT.match(tok.content):
                found.add("task_list")
    return Scan(headings, [f for f in FEATURE_ORDER if f in found])


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

