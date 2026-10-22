"""
Stage 2: one part of the post (intro, a section, or the conclusion).

The model writes the BODY. Code writes the `## Heading` line, inserts the image block at the position it chose, and
runs the lint; a part that fails the lint gets one repair. A part that still fails is accepted with a warning (D14),
except an empty one: a post with a hole in it is not a post.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import settings
from src.longform.markdown_ast import parse

from . import schema as s
from .llm import LLM, CallSpec
from .media import Placed
from .plan import BlogPlan, BlogRequest
from .prompts import repair_unit_message, section_system_prompt, unit_user_message
from .sources import WebSources
from .text import clean_alt, clean_caption
from .validate import Lint, lint, sanitize, trim_extra_quotes

logger = logging.getLogger(__name__)

UNIT_TEMPERATURE = 0.7
# gpt-5.2 writes about 4% more than the length it is asked for (7 live posts planned at 1,450 words came to 1,512 on average: the
# sections ran 3.6% over, the intro 12%), so a part is asked for a little less and the post lands on its plan. The lint still
# judges the part against its real target.
WRITER_LENGTH_FACTOR = 0.96
CLOSING_ANGLE = "Close the post: restate its one idea and send the reader off with a concrete next step."


@dataclass(frozen=True)
class Unit:
    id: str                                   # intro | s1.. | conclusion
    kind: str                                 # intro | section | conclusion
    heading: Optional[str]                    # written by code as "## heading"; the intro has none
    angle: str
    target_words: int
    table: Optional[s.TableDirective] = None
    quote: bool = False
    image: Optional[s.ImageDirective] = None  # only when its picture was actually started
    placeholder: Optional[str] = None
    takeaways: int = 0


@dataclass
class UnitResult:
    unit: Unit
    text: str                                 # the part as the reader sees it, ending in a blank line
    words: int = 0
    repaired: bool = False
    repair_kinds: List[str] = field(default_factory=list)        # the lint errors that sent the part back to the model
    fixes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[dict] = field(default_factory=list)


def units_from_outline(outline: s.Outline, placed: Dict[str, Placed]) -> List[Unit]:
    units = [Unit("intro", "intro", None, outline.intro.hook, outline.intro.target_words)]
    for sec in outline.sections:
        picture = placed.get(sec.id)
        images, tables = sec.of_kind("image"), sec.of_kind("table")
        started = bool(images and picture and picture.placeholder_id)
        units.append(Unit(
            sec.id, "section", sec.heading, sec.angle, sec.target_words,
            table=tables[0] if tables else None, quote=bool(sec.of_kind("quote")),
            image=images[0] if started else None, placeholder=picture.placeholder_id if started else None,
        ))
    c = outline.conclusion
    units.append(Unit("conclusion", "conclusion", c.heading, CLOSING_ANGLE, c.target_words, takeaways=c.takeaways))
    return units


def unit_max_tokens(unit: Unit) -> int:
    return max(512, 3 * unit.target_words)


def _compact(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def unit_request(request: BlogRequest, plan: BlogPlan, outline: s.Outline, unit: Unit,
                 previous_angle: Optional[str]) -> dict:
    """The part's user data: the whole outline for coherence, this part's job, and what it must contain."""
    parts = [{"id": "intro", "angle": outline.intro.hook}]
    parts += [{"id": sec.id, "heading": sec.heading, "angle": sec.angle} for sec in outline.sections]
    parts.append({"id": "conclusion", "heading": outline.conclusion.heading})
    return _compact({
        "post": _compact({"title": outline.title, "subtitle": outline.subtitle, "parts": parts}),
        "write": _compact({"part": unit.kind, "id": unit.id, "heading": unit.heading, "angle": unit.angle,
                           "target_words": round(unit.target_words * WRITER_LENGTH_FACTOR)}),
        "assignments": _compact({
            "table": _compact({"title": unit.table.title, "columns": unit.table.columns,
                               "rows_hint": unit.table.rows_hint}) if unit.table else None,
            "quote": True if unit.quote else None,
            "takeaways": unit.takeaways or None,
        }),
        # A general "only when assigned" rule is ignored about half the time; a concrete list of what this part must not
        # contain is followed. The editor controls how many tables and quotes the post has.
        "not_assigned": [name for name, assigned in (("table", unit.table), ("quote", unit.quote)) if not assigned],
        "already_covered": previous_angle,
        "image_follows_first_paragraph": True if unit.image else None,
        "brief": request.brief,
        "style_notes": request.style_notes,
        "audience": request.audience,
        "tone": request.tone,
        "limits": {"max_words_without_visual_break": plan.max_prose_run},
        "web": plan.use_web,
    })


def image_block(unit: Unit) -> str:
    """The picture as the editor receives it: the placeholder it swaps on `image_ready`, and an italic caption line."""
    line = f"![{clean_alt(unit.image.alt)}](placeholder::{unit.placeholder})"
    caption = clean_caption(unit.image.caption)
    return f"{line}\n*{caption}*" if caption else line


def assemble(unit: Unit, body: str) -> str:
    """Heading, then the body with the picture after its first paragraph. Blocks are separated by one blank line."""
    parts = [f"## {unit.heading}"] if unit.heading else []
    blocks = [b for b in parse(body) if b.kind != "blank"]
    texts = [b.text.rstrip("\n") for b in blocks]
    if unit.image and unit.placeholder:
        texts.insert(next((i + 1 for i, b in enumerate(blocks) if b.kind == "paragraph"), 0), image_block(unit))
    return "\n\n".join(parts + texts) + "\n\n"


def _lint(text: str, unit: Unit, plan: BlogPlan) -> Lint:
    return lint(text, unit=unit.id, expect_table=unit.table is not None, expect_quotes=1 if unit.quote else 0,
                target_words=unit.target_words, max_prose_run=plan.max_prose_run)


def _finish(unit: Unit, plan: BlogPlan, reply: str, sources: Optional[WebSources]) -> Tuple[str, Lint, List[str]]:
    """A model reply as the reader would see it: sanitised, assembled, extra quotes dropped where that breaks nothing."""
    clean = sanitize(reply, sources)
    text = assemble(unit, clean.text)
    text, dropped = trim_extra_quotes(
        text, 1 if unit.quote else 0, lambda t: {e.kind for e in _lint(t, unit, plan).errors})
    return text, _lint(text, unit, plan), clean.fixes + ["extra_quote"] * dropped


def write_unit(llm: LLM, request: BlogRequest, plan: BlogPlan, outline: s.Outline, unit: Unit,
               previous_angle: Optional[str], sources: Optional[WebSources] = None,
               request_id: Optional[str] = None) -> UnitResult:
    """Draft, sanitise, lint, and repair once. Raises BlogError when the part comes back empty."""
    system = section_system_prompt()
    messages = [{"role": "user", "content": unit_user_message(unit_request(request, plan, outline, unit, previous_angle))}]
    tokens = unit_max_tokens(unit)
    reply = llm(system, messages, CallSpec("unit", UNIT_TEMPERATURE, tokens, use_web=plan.use_web, unit=unit.id))
    text, found, fixes = _finish(unit, plan, reply, sources)
    repaired, repair_kinds = False, []

    if found.errors and settings.blog_repair_enabled:
        repair_kinds = sorted({e.kind for e in found.errors})
        logger.info(f"[{request_id}] {unit.id} needs repair: {repair_kinds}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": repair_unit_message(s.format_errors(found.errors))}]
        # No web tools on the repair: it fixes what was written, it does not research again.
        second_text, second_found, second_fixes = _finish(
            unit, plan, llm(system, messages, CallSpec("unit_repair", UNIT_TEMPERATURE, tokens, unit=unit.id)), sources)
        repaired, fixes = True, fixes + second_fixes
        was_empty = any(e.kind == "empty" for e in found.errors)
        if was_empty or len(second_found.errors) <= len(found.errors):    # never trade a flawed part for a worse one
            text, found = second_text, second_found

    if any(e.kind == "empty" for e in found.errors):
        raise s.BlogError(f"The model returned no text for part '{unit.id}'", found.errors)
    warnings = []
    if found.errors:
        warnings.append({"kind": "unit_lint", "unit": unit.id,
                         "detail": "; ".join(f"{e.kind}: {e.message}" for e in found.errors)})
    return UnitResult(unit=unit, text=text, words=found.words, repaired=repaired, repair_kinds=repair_kinds, fixes=fixes,
                      notes=found.notes, warnings=warnings)
