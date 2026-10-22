"""
Stage 3: the gate between a drafted part and the editor.

sanitize()  deterministic fixes, no model involved: stray headings, image markup, URLs of pages nobody read, quote
            attributions that have no page behind them (rule D13).
lint()      what can still be wrong afterwards and is worth one repair call: the assigned table or quote is missing,
            too much prose without a visual break, the part is far from its target length.
article_warnings()  whole-post checks reported to the client, never repaired.

Findings that are only matters of taste (a long paragraph, a second list) are returned as `notes` for the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from src.longform.markdown_ast import Block, parse, prose_words, table_cells, table_shape

from .schema import Issue
from .sources import WebSources

LENGTH_TOLERANCE = 0.35
MAX_SENTENCES_PER_PARAGRAPH = 4
MAX_LIST_ITEMS = 7

_WRAP_FENCE_RE = re.compile(r"^\s*```[A-Za-z]*[ \t]*\n(.*)\n[ \t]*```\s*$", re.S)
_IMAGE_PROMPT_TAG_RE = re.compile(r"<ImagePrompt\b[^>]*?(?:/>|>.*?</ImagePrompt>)", re.S | re.I)
_IMAGE_MARKUP_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_PLACEHOLDER_RE = re.compile(r"placeholder::[0-9A-Za-z-]+")
_LINK_OR_URL_RE = re.compile(
    r"\[(?P<text>[^\]]*)\]\((?P<href>[^)\s]*)(?:\s+\"[^\"]*\")?\)|(?P<bare>(?:https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"])", re.I)
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)|\[\s*\]")
_QUOTE_PREFIX_RE = re.compile(r"^\s{0,3}>+\s?")
_DASH_LINE_RE = re.compile(r"^\s*(?:[—–]|--+)\s*(?P<who>\S.*)$")
_DASH_TAIL_RE = re.compile(r"^(?P<quote>.*?\S)\s+(?:[—–]|--+)\s+(?P<who>[^—–\n]{2,120})$")
_HYPHEN_TAIL_RE = re.compile(r"^(?P<quote>.*?\S)\s-\s+(?P<who>[A-Z@][^\n]{1,120})$")
_DELIMITER_CELL_RE = re.compile(r"^\s*:?-+:?\s*$")
_ABBREV_RE = re.compile(r"\b(?:e\.g|i\.e|vs|etc|dr|mr|mrs|ms|prof|approx|inc|ltd|no|fig)\.", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*\s+(?=[A-Z0-9\"'“(\[])")


@dataclass
class Sanitized:
    text: str
    fixes: List[str] = field(default_factory=list)


@dataclass
class Lint:
    errors: List[Issue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    words: int = 0
    longest_run: int = 0


# ---------------------------------------------------------------------------
# Sanitize
# ---------------------------------------------------------------------------

def _scrub_links(text: str, sources: Optional[WebSources], fixes: List[str]) -> str:
    """Markdown links keep only if the page was read; bare URLs the same. Otherwise the link text stays."""
    def repl(m: re.Match) -> str:
        url = m.group("href") if m.group("href") is not None else m.group("bare")
        if sources is not None and sources.allows(url):
            return m.group(0)
        fixes.append("url")
        return m.group("text") if m.group("href") is not None else ""
    text = _LINK_OR_URL_RE.sub(repl, text)
    return _EMPTY_PARENS_RE.sub("", text)


_NAME_CONNECTORS = {"of", "the", "and", "at", "for", "in", "on", "de", "da", "del", "van", "von", "la", "le", "&"}
_NAME_WORD_RE = re.compile(r"[\w@.'’&-]+")


def _looks_like_attribution(who: str) -> bool:
    """
    A name or a source ("Jane Doe, CEO of Acme", "The Economist", "@handle"), not the tail of a sentence. Names are
    mostly capitalised words; a clause after a dash ("Then do it well.") is mostly lowercase.
    """
    who = who.strip().rstrip(".")
    words = [w for w in _NAME_WORD_RE.findall(who) if w.lower() not in _NAME_CONNECTORS]
    if not words or len(who.split()) > 12 or who.endswith(("!", "?")) or ". " in who:
        return False
    return sum(w[0].isupper() or w[0] in "@0123456789" for w in words) / len(words) >= 0.6


def _strip_attribution(block: Block, fixes: List[str]) -> Optional[str]:
    """
    The block quote without its attribution, or None when nothing is left of it. An attribution that links to a
    page that was read is kept (the link survived _scrub_links); any other attribution has no page behind it.
    A dash inside the sentence itself ("not more meetings -- better ones") is punctuation, not attribution.
    """
    lines = block.text.rstrip("\n").splitlines()
    last = _QUOTE_PREFIX_RE.sub("", lines[-1]) if lines else ""
    if "](" in last:
        return block.text.rstrip("\n")
    own_line = _DASH_LINE_RE.match(last)
    if own_line and _looks_like_attribution(own_line.group("who")):
        fixes.append("attribution")
        lines = lines[:-1]
    else:
        tail = next((m for m in (_DASH_TAIL_RE.match(last), _HYPHEN_TAIL_RE.match(last))
                     if m and _looks_like_attribution(m.group("who"))), None)
        if tail:
            fixes.append("attribution")
            lines[-1] = "> " + tail.group("quote").rstrip(" ,;:")
    return "\n".join(lines) if any(_QUOTE_PREFIX_RE.sub("", ln).strip() for ln in lines) else None


def _fix_table_delimiter(block: Block) -> Optional[str]:
    """
    A GFM table needs a delimiter row with exactly as many cells as the header. A model that miscounts (4 columns,
    `|---|---|---|`) leaves a paragraph of pipe characters in the editor. Returns the block with the delimiter rebuilt,
    or None when the block is not a pipe table with a miscounted delimiter.
    """
    lines = block.text.rstrip("\n").splitlines()
    if len(lines) < 2 or not all(ln.lstrip().startswith("|") for ln in lines[:2]):
        return None
    header, delimiter = table_cells(lines[0]), table_cells(lines[1])
    if len(delimiter) == len(header) or not all(_DELIMITER_CELL_RE.match(c) for c in delimiter):
        return None
    return "\n".join([lines[0], "|" + "|".join(["---"] * len(header)) + "|", *lines[2:]])


def sanitize(reply: str, sources: Optional[WebSources] = None) -> Sanitized:
    """
    Make a model's part safe to place in the post. Removes what code owns (headings, image markup), URLs of pages
    that were not read and quote attributions that have no read page behind them, repairs a miscounted table
    delimiter, and normalises blank lines.
    """
    fixes: List[str] = []
    text = (reply or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    wrapped = _WRAP_FENCE_RE.match(text)
    if wrapped:
        text = wrapped.group(1)
        fixes.append("fence")
    for rx, name in ((_IMAGE_PROMPT_TAG_RE, "image_markup"), (_IMAGE_MARKUP_RE, "image_markup"),
                     (_PLACEHOLDER_RE, "image_markup")):
        text, n = rx.subn("", text)
        if n:
            fixes.append(name)
    text = _scrub_links(text, sources, fixes)

    parts: List[str] = []
    for block in parse(text):
        if block.kind == "blank":
            continue
        if block.kind == "heading":
            fixes.append("heading")
            continue
        chunk = block.text.rstrip("\n")
        if block.kind == "paragraph":
            fixed = _fix_table_delimiter(block)
            if fixed is not None:
                fixes.append("table_delimiter")
                chunk = fixed
        if block.kind == "blockquote":
            chunk = _strip_attribution(block, fixes)
            if chunk is None:
                continue
        if chunk.strip():
            parts.append(chunk)
    return Sanitized("\n\n".join(parts), fixes)


def trim_extra_quotes(text: str, keep: int, errors_of: Callable[[str], set]) -> Tuple[str, int]:
    """
    Drop block quotes beyond the `keep` the part was assigned: a pull quote only echoes the prose, and the post's rhythm
    is code's to set. But never at the price of a rule the part otherwise meets. A quote is also a visual break and part
    of the word count, so an extra one stays when removing it would add a lint error (a long run of prose, a part that
    is now too short): that would cost a repair call to undo a harmless quote. Tables are never removed: they carry content.
    `errors_of(text)` returns the set of lint error kinds. Returns the text and how many quotes were dropped.
    """
    removed, baseline = 0, errors_of(text)
    while True:
        blocks = parse(text)
        quotes = [b for b in blocks if b.kind == "blockquote"]
        for extra in quotes[keep:]:
            candidate = "\n\n".join(b.text.rstrip("\n") for b in blocks if b.kind != "blank" and b is not extra) + "\n\n"
            if errors_of(candidate) <= baseline:
                text, removed = candidate, removed + 1
                break
        else:
            return text, removed


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

def sentence_count(paragraph: str) -> int:
    text = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", ""), re.sub(r"\s+", " ", paragraph).strip())
    return len(_SENTENCE_END_RE.split(text)) if text else 0


def longest_prose_run(blocks: Sequence[Block]) -> int:
    """Most words of paragraph text in a row. Headings, figures, tables, quotes, lists and code all break a run."""
    run = best = 0
    for b in blocks:
        if b.kind == "paragraph":
            run += b.words
            best = max(best, run)
        elif b.kind != "blank":
            run = 0
    return best


def lint(text: str, *, unit: Optional[str] = None, expect_table: bool = False, expect_quotes: int = 0,
         target_words: int = 0, max_prose_run: int = 220) -> Lint:
    """Check an assembled part (heading and image already in place, as the reader will see it)."""
    blocks = parse(text)
    out = Lint(words=prose_words(blocks), longest_run=longest_prose_run(blocks))
    if out.words == 0:
        out.errors.append(Issue("empty", "The part has no text. Write it.", unit))
        return out

    if expect_table:
        shapes = [table_shape(b) for b in blocks if b.kind == "table"]
        if not shapes:
            out.errors.append(Issue("table_missing", "The assigned table is missing. Write one GitHub-flavoured Markdown "
                                                     "table: a header row, the |---|---| delimiter row, then data rows.", unit))
        elif not any(s and s.consistent and s.header >= 2 and len(s.rows) >= 2 for s in shapes):
            out.errors.append(Issue("table_shape", "The table is malformed: it needs at least 2 columns and 2 data rows, "
                                                   "and every row must have the same number of cells as the header.", unit))
    if sum(b.kind == "blockquote" for b in blocks) < expect_quotes:
        out.errors.append(Issue("quote_missing", "The assigned pull quote is missing. Add one block quote: a single "
                                                 "line starting with '> '.", unit))
    if out.longest_run > max_prose_run:
        out.errors.append(Issue("prose_run", f"{out.longest_run} words of prose in a row; the limit is {max_prose_run}. "
                                             "Break it up with a short list, the assigned table or quote, or by tightening "
                                             "the text. Do not add headings.", unit))
    if target_words:
        low, high = round(target_words * (1 - LENGTH_TOLERANCE)), round(target_words * (1 + LENGTH_TOLERANCE))
        if not low <= out.words <= high:
            out.errors.append(Issue("length", f"{out.words} words; this part should be about {target_words} "
                                              f"(between {low} and {high}).", unit))

    if not expect_table and any(b.kind == "table" for b in blocks):
        out.notes.append("unassigned_table")
    if sum(b.kind == "blockquote" for b in blocks) > expect_quotes:
        out.notes.append("unassigned_quote")
    lists = [b for b in blocks if b.kind == "list"]
    if len(lists) > 1 or any(b.items > MAX_LIST_ITEMS for b in lists):
        out.notes.append("lists")
    if any(b.kind == "paragraph" and sentence_count(b.text) > MAX_SENTENCES_PER_PARAGRAPH for b in blocks):
        out.notes.append("long_paragraph")
    return out


# ---------------------------------------------------------------------------
# Whole post
# ---------------------------------------------------------------------------

def article_warnings(markdown: str, *, tables: int, quotes: int, headings: Sequence[str]) -> List[dict]:
    """Whole-post findings for the client. `headings` are the section headings the outline promised, in order."""
    blocks = parse(markdown)
    found_tables = sum(1 for b in blocks if b.kind == "table")
    found_quotes = sum(1 for b in blocks if b.kind == "blockquote")
    warnings = []
    if found_tables < tables:
        warnings.append({"kind": "quota_tables", "detail": f"The post has {found_tables} table(s); {tables} were planned."})
    if found_quotes < quotes:
        warnings.append({"kind": "quota_quotes", "detail": f"The post has {found_quotes} quote(s); {quotes} were planned."})
    got = [re.sub(r"^#+\s*|\s+#+\s*$|\s+$", "", b.text).strip().lower() for b in blocks if b.kind == "heading" and b.level == 2]
    if got != [h.strip().lower() for h in headings]:
        warnings.append({"kind": "heading_mismatch", "detail": "The section headings differ from the outline."})
    return warnings
