"""
Verify: code checks, the model does not get to say it was careful.

A refined chunk is compared with the structure its input had (chunk.Structure). Anything that moved is an Issue, the issues go back
to the model once as a numbered list, and a chunk that is still wrong is restored as written by refine.py (decision D6 of
docs/rich_content_engine_plan.md). Nothing here calls a model.

  tokens      the same tokens, in the same order, none invented; block tokens still alone on their line (snap_blocks puts
              back the exact line and the blank lines around it); link references well formed
  headings    the same count and levels
  lists       the same shape (items per list, nesting)
  paragraphs  `grammar`: the same count; the other operations: within 25%
  order       the headings, lists and runs of paragraphs come in the same sequence (no lead-in paragraph under a heading)
  structure   no table, quote, image, code or HTML appeared or went missing
  length      words within the operation's ratio window
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .chunk import Chunk, analyse
from .freeze import BLOCK_TOKEN_LINE, stray_link_marks, token_list

PARAGRAPH_TOLERANCE = 0.25
MIN_WORD_SLACK = 5                       # a short passage may move a few words whatever the ratio says
_PREFACE = re.compile(r"^(?:here(?:'s| is| are)\b|sure\b|certainly\b|of course\b|below is\b)[^\n]{0,160}:\s*$", re.IGNORECASE)
_FENCE_OPEN = re.compile(r"^```[\w+-]*[ \t]*$")
_MARKERS = {"<<<PASSAGE", "PASSAGE>>>"}


@dataclass(frozen=True)
class Issue:
    kind: str
    message: str


@dataclass
class Verdict:
    text: str                                       # the answer as it will be used: unwrapped, with the chunk's own edges
    issues: List[Issue] = field(default_factory=list)
    words: int = 0
    ratio: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.issues


def unwrap(output: str, original: str = "") -> str:
    """The answer without what models wrap around it: a "Here is..." line, echoed markers, a code fence around the lot."""
    lines = [ln for ln in (output or "").replace("\r\n", "\n").strip().split("\n")]
    while lines and lines[0].strip() in _MARKERS:
        lines.pop(0)
    while lines and lines[-1].strip() in _MARKERS:
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _PREFACE.match(lines[0].strip()) and lines[0].strip() not in original:
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) >= 2 and _FENCE_OPEN.match(lines[0].strip()) and lines[-1].strip() == "```" and not original.lstrip().startswith("```"):
        lines = lines[1:-1]
    return "\n".join(lines).strip("\n")


def _edges(frozen: str) -> Tuple[str, str]:
    core = frozen.strip()
    if not core:
        return frozen, ""
    return frozen[:frozen.index(core[0])], frozen[frozen.rindex(core[-1]) + 1:]


def with_edges(core: str, chunk: Chunk) -> str:
    """`core` between the leading and trailing whitespace the chunk had, so chunks join exactly as the document was laid out."""
    lead, trail = _edges(chunk.frozen)
    return lead + core.strip() + trail


def _layout(frozen: str) -> Dict[str, Tuple[str, bool, bool]]:
    """For each block token alone on its line in the passage: the line as written, and whether blank lines (or an edge) were around it."""
    lines = frozen.split("\n")
    found: Dict[str, Tuple[str, bool, bool]] = {}
    for i, line in enumerate(lines):
        token = line.strip()
        if BLOCK_TOKEN_LINE.match(token) and token not in found:
            found[token] = (line, i == 0 or not lines[i - 1].strip(), i == len(lines) - 1 or not lines[i + 1].strip())
    return found


def snap_blocks(core: str, chunk: Chunk) -> str:
    """
    Every block token goes back on the line it had: the same indent and nothing after it, and with the blank lines it had around
    it. A model that adds two trailing spaces (a Markdown hard break) would otherwise change the bytes of the table that comes
    back in its place, and one that drops the blank line below a quote would glue the next paragraph to it (lazy continuation).
    Lossless: only whitespace around a token moves, and a token that was not alone on its line is left for verify() to judge.
    """
    layout = _layout(chunk.frozen)
    if not layout:
        return core
    lines, out = core.split("\n"), []
    for i, line in enumerate(lines):
        token = line.strip()
        if token in layout and BLOCK_TOKEN_LINE.match(token):
            exact, blank_before, blank_after = layout[token]
            if blank_before and out and out[-1].strip():
                out.append("")
            out.append(exact)
            if blank_after and i + 1 < len(lines) and lines[i + 1].strip():
                out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _diff(want: Sequence[str], got: Sequence[str]) -> Tuple[List[str], List[str]]:
    missing = list((Counter(want) - Counter(got)).elements())
    extra = list((Counter(got) - Counter(want)).elements())
    return missing, extra


def _reading(sequence: Sequence[str]) -> str:
    return " > ".join(sequence[:14]) + (f" > ... ({len(sequence)} in all)" if len(sequence) > 14 else "")


def _names(tokens: Sequence[str]) -> str:
    return ", ".join(tokens[:6]) + (f" and {len(tokens) - 6} more" if len(tokens) > 6 else "")


def verify(chunk: Chunk, output: str, *, ratio: Tuple[float, float] = (0.85, 1.25), paragraphs: str = "within") -> Verdict:
    """
    Check `output` (the model's answer for `chunk`) against the chunk. `paragraphs` is "equal" (grammar: the same count) or
    "within" (the others: within 25%). The returned text is the answer with wrappers removed and the chunk's own edges.
    """
    text = with_edges(snap_blocks(unwrap(output, chunk.frozen), chunk), chunk)
    want, issues = chunk.structure, []

    got_tokens = token_list(text)
    if got_tokens != list(chunk.tokens):
        missing, extra = _diff(chunk.tokens, got_tokens)
        if missing:
            issues.append(Issue("tokens_missing", f"{_names(missing)} {'is' if len(missing) == 1 else 'are'} missing: every locked "
                                                  "token must appear exactly once."))
        if extra:
            issues.append(Issue("tokens_extra", f"{_names(extra)} {'was' if len(extra) == 1 else 'were'} not in the passage: do "
                                                "not add, repeat or invent tokens."))
        if not missing and not extra:
            issues.append(Issue("tokens_order", "The locked tokens are in a different order: keep them in the order given."))
    if stray_link_marks(text):
        issues.append(Issue("link_form", "A link reference was changed: keep each one exactly as written, [anchor](link://n)."))
    alone = {ln.strip() for ln in text.splitlines()}
    lost = [t for t in dict.fromkeys(chunk.own_line) if t in got_tokens and t not in alone]
    if lost:
        issues.append(Issue("token_inline", f"{_names(lost)} must stay alone on its own line, as in the passage."))

    got = analyse(text)
    if got.headings != want.headings:
        issues.append(Issue("headings", f"The headings changed: the passage has {len(want.headings)} (levels "
                                        f"{list(want.headings)}), the answer {len(got.headings)} (levels {list(got.headings)}). "
                                        "Keep every heading at its level."))
    if got.lists != want.lists:
        issues.append(Issue("lists", f"The lists changed: the passage has {len(want.lists)} list(s) with "
                                     f"{[len(d) for d in want.lists]} items, the answer {len(got.lists)} with "
                                     f"{[len(d) for d in got.lists]}. Keep every list item and its nesting."))
    if paragraphs == "equal":
        if got.paragraphs != want.paragraphs:
            issues.append(Issue("paragraphs", f"The passage has {want.paragraphs} paragraphs; the answer has {got.paragraphs}. "
                                              "Keep the same number: do not merge or split them."))
    else:
        slack = max(1, round(want.paragraphs * PARAGRAPH_TOLERANCE))
        if abs(got.paragraphs - want.paragraphs) > slack:
            issues.append(Issue("paragraphs", f"The passage has {want.paragraphs} paragraphs; the answer has {got.paragraphs}. "
                                              "Keep about the same number: do not merge or split them."))
    if got.sequence != want.sequence:
        issues.append(Issue("order", f"The blocks are in a different order or kind: the passage reads {_reading(want.sequence)}, the "
                                     f"answer {_reading(got.sequence)}. Do not add, remove or move a heading, list or paragraph "
                                     "(for example a lead-in paragraph under a heading)."))
    if got.kinds != want.kinds:
        was, now = dict(want.kinds), dict(got.kinds)
        changes = [f"{kind}: {was.get(kind, 0)} -> {now.get(kind, 0)}" for kind in sorted(set(was) | set(now))
                   if was.get(kind, 0) != now.get(kind, 0)]
        issues.append(Issue("structure", "Blocks appeared or went missing (" + "; ".join(changes) + "). Do not create a table, "
                                         "quote, image or code block, and do not remove one."))

    low, high = ratio
    floor, ceiling = min(round(low * want.words), want.words - MIN_WORD_SLACK), max(round(high * want.words), want.words + MIN_WORD_SLACK)
    if want.words and not floor <= got.words <= ceiling:
        issues.append(Issue("length", f"The answer has {got.words} words; the passage has {want.words} and the answer must have "
                                      f"between {floor} and {ceiling}. Return the whole passage, not a summary or an expansion."))
    return Verdict(text=text, issues=issues, words=got.words, ratio=got.words / want.words if want.words else 1.0)


def repair_message(issues: Sequence[Issue]) -> str:
    """The numbered list that goes back to the model with its own answer (temperature 0)."""
    numbered = "\n".join(f"{i}. [{e.kind}] {e.message}" for i, e in enumerate(issues, 1))
    return ("Your answer broke these rules:\n" + numbered + "\nReturn the whole passage again with every rule kept. "
            "Output only the passage.")
