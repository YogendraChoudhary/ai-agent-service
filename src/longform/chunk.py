"""
Chunks: a long document cut into passages a model can refine, with what each passage must still look like afterwards.

A chunk is a run of whole top-level blocks, so a list, a table or a quote is never split. It aims at LONGFORM_CHUNK_WORDS
words (1,000) inside a window of 0.6x to 1.4x, and prefers to end before a heading, so sections stay together and the tone
stays consistent inside them (decision D4 of docs/rich_content_engine_plan.md). The first chunk may be smaller
(LONGFORM_FIRST_CHUNK_WORDS): a chunk reaches the reader only when it is verified, so a short first one gets text on screen
in a few seconds instead of after a whole thousand words.

Each chunk carries two texts and one description:
  original   exactly what these blocks say in the document (the thing to restore if the model fails)
  frozen     the same blocks with tables, quotes, images, code and URLs swapped for tokens (what the model sees)
  structure  what the answer must still show: headings, list shape, paragraph count, token order, word count
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .freeze import BLOCK_TOKEN_LINE, FrozenBlock, own_line_tokens, strip_tokens, token_list, words_of
from .markdown_ast import heading_levels, list_shape, parse

LOW, HIGH = 0.6, 1.4                # the chunk window around the target size, as a share of it
TINY = 0.15                         # a last chunk under this share of the target joins the one before it if it fits
CONTEXT_WORDS = 60                  # how much of the previous chunk the next one is shown, for continuity only
WATCHED = ("table", "blockquote", "fence", "html", "image", "hr")    # block kinds the answer must neither add nor drop


@dataclass(frozen=True)
class Structure:
    """What can be counted in a passage without reading it: the same measure is taken of the input and of the answer."""
    headings: Tuple[int, ...]
    lists: Tuple[Tuple[int, ...], ...]
    paragraphs: int
    kinds: Tuple[Tuple[str, int], ...]
    words: int
    sequence: Tuple[str, ...] = ()          # the block kinds in order (a locked block by its token), a run of paragraphs counting once


def collapse(kinds: Sequence[str]) -> Tuple[str, ...]:
    """Block kinds in order with every run of paragraphs counted once: merging or splitting paragraphs moves nothing."""
    out: List[str] = []
    for kind in kinds:
        if not (kind == "paragraph" and out and out[-1] == "paragraph"):
            out.append(kind)
    return tuple(out)


def _kind(block) -> str:
    """The block's kind; a paragraph that is only a block token is that token's kind, so a table moved past a paragraph shows."""
    if block.kind == "paragraph":
        token = block.text.strip()
        if BLOCK_TOKEN_LINE.match(token):
            return token[2:token.index("_")].lower()
    return block.kind


def analyse(text: str) -> Structure:
    blocks = parse(text)
    counts = {k: sum(b.kind == k for b in blocks) for k in WATCHED}
    return Structure(headings=heading_levels(blocks), lists=list_shape(blocks),
                     paragraphs=sum(b.kind == "paragraph" for b in blocks),
                     kinds=tuple((k, n) for k, n in counts.items() if n), words=words_of(text),
                     sequence=collapse([_kind(b) for b in blocks if b.kind != "blank"]))


@dataclass(frozen=True)
class Chunk:
    index: int
    original: str
    frozen: str
    tokens: Tuple[str, ...]                 # every token the answer must contain, in order
    own_line: Tuple[str, ...]               # the block tokens that must stay alone on their line
    structure: Structure
    before: str = ""                        # the end of the previous chunk: context, never to be output
    after: Optional[str] = None             # the first heading of the next chunk: context, never to be output

    @property
    def refinable(self) -> bool:
        """False for a passage with nothing to polish (only tokens and blank lines): it is passed through as written."""
        return self.structure.words > 0


def _heading_near(frozen: Sequence[FrozenBlock], start: int, size: int, high: int) -> bool:
    """Does a heading arrive from `start` on before the chunk, now `size` words, would pass `high`?"""
    for fb in frozen[start:]:
        if fb.block.kind == "heading":
            return True
        size += fb.words
        if size > high:
            return False
    return False


def _groups(frozen: Sequence[FrozenBlock], target: int, first: int = 0) -> List[List[FrozenBlock]]:
    """Runs of whole blocks of about `target` words; the first run aims at `first` words instead when that is smaller."""
    high_end = round(target * HIGH)
    groups: List[List[FrozenBlock]] = []
    i, n = 0, len(frozen)
    while i < n:
        goal = first if first and not groups and first < target else target
        low, high = round(goal * LOW), round(goal * HIGH)
        group: List[FrozenBlock] = []
        size = 0
        while i < n:
            fb = frozen[i]
            heading = fb.block.kind == "heading"
            if group and fb.words and ((heading and fb.block.level in (2, 3) and size >= low)
                                       or (heading and size >= goal) or size + fb.words > high):
                break
            group.append(fb)
            size += fb.words
            i += 1
            if size >= goal and fb.words and not _heading_near(frozen, i, size, high):
                while i < n and frozen[i].block.kind == "blank":        # the chunk keeps the blank lines that follow it
                    group.append(frozen[i])
                    i += 1
                break
        groups.append(group)
    if len(groups) > 1:                     # a sliver at the end is not worth a call of its own
        last, before = sum(f.words for f in groups[-1]), sum(f.words for f in groups[-2])
        if last < TINY * target and last + before <= high_end:
            groups[-2].extend(groups.pop())
    return groups


def _tail(chunk_frozen: str) -> str:
    words = strip_tokens(chunk_frozen).split()
    return " ".join(words[-CONTEXT_WORDS:])


def _first_heading(group: Sequence[FrozenBlock]) -> Optional[str]:
    first = next((f for f in group if f.block.kind != "blank"), None)
    return first.block.text.strip() if first is not None and first.block.kind == "heading" else None


def chunk_blocks(frozen: Sequence[FrozenBlock], target: int = 1000, first: int = 0) -> List[Chunk]:
    """
    Cut frozen blocks into chunks of about `target` words, never inside a block. `first` > 0 makes the FIRST chunk smaller
    (about that many words) so its text reaches the reader sooner; the others are unchanged. 0 or more than `target`: off.
    """
    groups = _groups(frozen, max(50, int(target)), max(50, int(first)) if first else 0)
    texts = ["".join(f.text for f in g) for g in groups]
    return [Chunk(index=i, original="".join(f.block.text for f in group), frozen=texts[i], tokens=tuple(token_list(texts[i])),
                  own_line=tuple(own_line_tokens(texts[i])), structure=analyse(texts[i]),
                  before=_tail(texts[i - 1]) if i else "", after=_first_heading(groups[i + 1]) if i + 1 < len(groups) else None)
            for i, group in enumerate(groups)]
