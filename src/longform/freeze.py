"""
Freeze and thaw: the model polishes prose, code keeps everything else.

Before a model sees a passage, the parts of a Markdown document that must not change are swapped for short tokens, and
after the model has answered the tokens are swapped back for the original bytes. Preservation is a property of the
pipeline, not a request in the prompt (decision D2 of docs/rich_content_engine_plan.md).

  block tokens     [[TABLE_n]] [[QUOTE_n]] [[IMAGE_n]] [[CODE_n]] [[HTML_n]]   replace the whole block, on its own line
  inline tokens    [[IMAGE_n]] (an image inside prose)  [[URL_n]] (a bare URL, an autolink or an email address)
                   [[HTML_n]] (an inline tag such as <kbd> or <br>, or a comment)
  link references  [anchor text](link://n)                                       the model may reword the anchor only

Headings and lists stay in the text (their wording is the job); their count, level and nesting are checked by verify.py.
`freeze(text)` then `thaw(frozen, vault)` is the identity on every input, which is the property the tests pin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .markdown_ast import ALERT_MARKER, FOOTNOTE_LABEL, INLINE_DIALECT_PATTERNS, TASK_MARKER, Block, parse, word_count

# What each kind of block becomes. A horizontal rule and anything markdown-it leaves unclassified (link reference
# definitions) are opaque to the model like raw HTML: not its to touch.
BLOCK_TOKEN = {"table": "TABLE", "blockquote": "QUOTE", "image": "IMAGE", "fence": "CODE", "html": "HTML", "hr": "HTML",
               "other": "HTML"}

TOKEN_RE = re.compile(r"\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML|URL)_\d+\]\]")
LINK_REF_RE = re.compile(r"\]\(link://(\d+)\)")
_ANY_TOKEN_RE = re.compile(TOKEN_RE.pattern + "|" + LINK_REF_RE.pattern)
BLOCK_TOKEN_LINE = re.compile(r"^\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML)_\d+\]\]$")        # a line that is only a block token

# Inline Markdown, scanned left to right in one pass. Atomic groups and possessive quantifiers keep the scan linear.
_LABEL = r"(?>[^\[\]\\]|\\.|\[[^\[\]]*\]){0,1000}+"      # bounded: unbounded, "\\[\nx\n" repeated made every "[" scan to the end of the text (23 s per 100k chars)
_DEST = r"(?>(?><[^<>\n]*>)|(?>(?>[^()\s\\]|\\.|\([^()\s]*\))+))"
_TITLE = r"""(?>\s+(?>"(?>[^"\\]|\\.)*"|'(?>[^'\\]|\\.)*'|\((?>[^()\\]|\\.)*\)))?"""
_EMAIL = r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
_TAG_ATTR = r"""\s+[A-Za-z_:][\w.:-]*+(?:\s*+=\s*+(?:[^\s"'=<>`]++|'[^']*+'|"[^"]*+"))?+"""
_DIALECT_GROUPS = tuple(INLINE_DIALECT_PATTERNS)            # math, wikilinks, footnote references, ~~strikethrough~~, :emoji: codes
_DIALECT = "|".join(f"(?P<{name}>{pattern})" for name, pattern in INLINE_DIALECT_PATTERNS.items())
_INLINE = re.compile(
    _DIALECT + "|"                                           # first: at one position a wikilink beats an image, $x$ beats nothing
    rf"(?P<image>!\[(?P<ilabel>{_LABEL})\](?P<itail>\(\s*{_DEST}{_TITLE}\s*\)))"
    rf"|(?P<imageref>!\[{_LABEL}\]\[[^\[\]]*\])"
    rf"|(?P<link>\[(?P<label>{_LABEL})\](?P<tail>\(\s*{_DEST}{_TITLE}\s*\)|\[[^\[\]]*\]))"
    rf"|(?P<auto><(?:(?:https?://|mailto:)[^<>\s]+|{_EMAIL})>)"
    rf"|(?P<tag><[A-Za-z][A-Za-z0-9-]*+(?:{_TAG_ATTR})*+\s*+/?+>|</[A-Za-z][A-Za-z0-9-]*+\s*+>|<!--(?:(?!-->).)*+-->)"
    r"|(?P<url>(?<![\w/@.])(?:https?://|www\.)[^\s<>\[\]()\"']*[^\s<>\[\]()\"'.,;:!?*_~`])"
    rf"|(?P<email>(?<![\w.+@-]){_EMAIL}(?![\w@-]))"
)
_FENCE_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<mark>`{3,}|~{3,})")


@dataclass
class Vault:
    """Everything that was taken out: token -> original bytes (without the block's final line break)."""
    blocks: Dict[str, str] = field(default_factory=dict)
    links: Dict[int, Tuple[str, str]] = field(default_factory=dict)       # n -> (opening bracket, what was inside)
    counts: Dict[str, int] = field(default_factory=dict)

    def block(self, kind: str, original: str) -> str:
        n = self.counts[kind] = self.counts.get(kind, 0) + 1
        token = f"[[{kind}_{n}]]"
        self.blocks[token] = original
        return token

    def link(self, opener: str, inside: str) -> str:
        n = len(self.links) + 1
        self.links[n] = (opener, inside)
        return f"(link://{n})"

    def summary(self) -> Dict[str, int]:
        """How many of each kind were frozen (the trace records it)."""
        return {**self.counts, **({"LINK": len(self.links)} if self.links else {})}


@dataclass(frozen=True)
class FrozenBlock:
    block: Block                    # the original block, exactly as written
    text: str                       # what the model sees for it
    words: int                      # prose words in `text` (tokens are not words)


# ---------------------------------------------------------------------------
# Reading tokens
# ---------------------------------------------------------------------------

def token_list(text: str) -> List[str]:
    """Every token in `text`, in order: "[[TABLE_1]]", or "link://3" for a link reference "](link://3)"."""
    return [m.group(0) if m.group(0).startswith("[[") else f"link://{m.group(1)}" for m in _ANY_TOKEN_RE.finditer(text)]


def own_line_tokens(text: str) -> List[str]:
    """The block tokens that sit alone on their line."""
    return [ln.strip() for ln in text.splitlines() if BLOCK_TOKEN_LINE.match(ln.strip())]


def strip_tokens(text: str) -> str:
    """`text` with every token and link reference blanked, for counting the words that are really there."""
    return LINK_REF_RE.sub("]", TOKEN_RE.sub(" ", text))


def words_of(text: str) -> int:
    return word_count(strip_tokens(text))


def stray_link_marks(text: str) -> int:
    """`link://` occurrences that are not a well-formed reference: the model broke one."""
    return text.count("link://") - len(LINK_REF_RE.findall(text))


def collides(text: str) -> bool:
    """True when the document already contains something that looks like a token: the chunked path must not run on it."""
    return bool(TOKEN_RE.search(text)) or "link://" in text


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def _inline(text: str, vault: Vault, alt_images: bool = False) -> str:
    """
    Images, links and URLs inside prose. A link's anchor is recursed into, so a linked image is frozen too. With `alt_images` an image
    is written like a link, ![alt text](link://n): the alt text stays in the passage (a translation must translate it) and the
    address is locked, exactly as a link's.
    """
    def swap(m: "re.Match[str]") -> str:
        if any(m.group(name) is not None for name in _DIALECT_GROUPS):
            return vault.block("HTML", m.group(0))                # opaque, like an inline tag: the model never sees or edits it
        if m.group("image") and alt_images:
            tail = m.group("itail")
            return f"![{_inline(m.group('ilabel'), vault, True)}]{vault.link(tail[0], tail[1:-1])}"
        if m.group("image") or m.group("imageref"):
            return vault.block("IMAGE", m.group(0))
        if m.group("link") is not None:
            tail = m.group("tail")
            opener, inside = tail[0], tail[1:-1]
            return f"[{_inline(m.group('label'), vault, alt_images)}]{vault.link(opener, inside)}"
        if m.group("tag") is not None:
            return vault.block("HTML", m.group(0))
        return vault.block("URL", m.group(0))
    # A task-list box ("- [ ] item") or a quote's alert keyword ("> [!NOTE]") is markup, not wording: swapped first, on the line it opens.
    text = FOOTNOTE_LABEL.sub(lambda m: vault.block("HTML", m.group(1)), text)               # "[^1]:" of a footnote that is exposed
    text = ALERT_MARKER.sub(lambda m: m.group(1) + vault.block("HTML", m.group(2)), text)
    return _INLINE.sub(swap, TASK_MARKER.sub(lambda m: m.group(1) + vault.block("HTML", m.group(2)), text))


def _fenced_regions(lines: Sequence[str]) -> List[Tuple[int, int]]:
    """(first, last) line numbers of every fenced code region inside a block, fences included."""
    regions, start, mark = [], None, ""
    for i, line in enumerate(lines):
        m = _FENCE_LINE.match(line)
        if start is None:
            if m:
                start, mark = i, m.group("mark")
        elif m and m.group("mark")[0] == mark[0] and len(m.group("mark")) >= len(mark) and not line.strip(" \t\r\n`~"):
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(lines) - 1))            # an unclosed fence runs to the end of the block
    return regions


def _prose_block(text: str, vault: Vault, allow_code: bool, alt_images: bool = False) -> str:
    """A heading, paragraph or list: code fenced inside a list item is frozen like any code, the rest by _inline."""
    lines = text.splitlines(keepends=True)
    regions = _fenced_regions(lines) if allow_code else []
    if not regions:
        return _inline(text, vault, alt_images)
    out, cursor = [], 0
    for first, last in regions:
        out.append(_inline("".join(lines[cursor:first]), vault, alt_images))
        body = "".join(lines[first:last + 1])
        eol = "\n" if body.endswith("\n") else ""
        indent = lines[first][:len(lines[first]) - len(lines[first].lstrip(" \t"))]
        original = body[len(indent):-1] if eol else body[len(indent):]
        out.append(indent + vault.block("CODE", original) + eol)
        cursor = last + 1
    out.append(_inline("".join(lines[cursor:]), vault, alt_images))
    return "".join(out)


_QUOTED_FENCE = re.compile(r"^[ \t]*(?:>[ \t]*)*(?:`{3,}|~{3,})")
_QUOTED_TABLE = re.compile(r"^[ \t]*(?:>[ \t]*)+\|")


def _exposable(block: Block) -> bool:
    """
    A table or a figure can always be worked on in place. A quote can too, unless it holds code or a table of its own: those are
    not verified inside a quote, so such a quote stays frozen.
    """
    if block.kind != "blockquote":
        return True
    return not any(_QUOTED_FENCE.match(line) or _QUOTED_TABLE.match(line) for line in block.text.splitlines())


def freeze_blocks(blocks: Sequence[Block], *, freeze_quotes: bool = True,
                  expose: Sequence[str] = ()) -> Tuple[List[FrozenBlock], Vault]:
    """
    Freeze document-ordered blocks. Token numbers run through the whole document, so a chunk's tokens are a subsequence.
    `expose` names block kinds ("table", "blockquote", "image", "footnote") the model works on in place instead of receiving a token: a
    translation must translate a table's cells, a quote, a figure's caption and alt text and a footnote's text. Their structure is then
    checked in code (verify.py).
    """
    vault, frozen = Vault(), []
    exposed = set(expose) | (set() if freeze_quotes else {"blockquote"})
    alt_images = "image" in exposed
    for b in blocks:
        kind = "footnote" if b.kind == "other" and b.flavor == "footnote" else b.kind
        if b.kind in BLOCK_TOKEN and not (kind in exposed and _exposable(b)):
            eol = "\n" if b.text.endswith("\n") else ""
            text = vault.block(BLOCK_TOKEN[b.kind], b.text[:-1] if eol else b.text) + eol
        elif b.kind == "blank":
            text = b.text
        else:
            text = _prose_block(b.text, vault, allow_code=b.kind == "list", alt_images=alt_images)
        frozen.append(FrozenBlock(b, text, words_of(text)))
    return frozen, vault


def freeze(text: str, *, freeze_quotes: bool = True, expose: Sequence[str] = ()) -> Tuple[str, Vault]:
    """(the text the model may see, the vault that restores it). `text` is the pipeline's LF-normalised document."""
    frozen, vault = freeze_blocks(parse(text), freeze_quotes=freeze_quotes, expose=expose)
    return "".join(f.text for f in frozen), vault


# ---------------------------------------------------------------------------
# Thaw
# ---------------------------------------------------------------------------

def thaw(text: str, vault: Vault) -> str:
    """Tokens back to the original bytes. A token the vault does not know is left as written (verify.py rejects those first)."""
    text = TOKEN_RE.sub(lambda m: vault.blocks.get(m.group(0), m.group(0)), text)

    def link(m: "re.Match[str]") -> str:
        known = vault.links.get(int(m.group(1)))
        if known is None:
            return m.group(0)
        opener, inside = known
        return "]" + opener + inside + (")" if opener == "(" else "]")
    return LINK_REF_RE.sub(link, text)
