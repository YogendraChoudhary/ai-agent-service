"""
The Mode 2 pipeline: outline -> pictures started -> parts drafted in parallel, emitted in order -> whole-post check.

generate_blog_stream() is a generator. It yields what the stream endpoint understands:
  {"__event__": {...}}        an additive SSE event (`outline`, sent as soon as the outline validates)
  "markdown"                  content, in document order, every part already linted
  {"__warnings__": [...]}     once at the end, only if there is something to say; the API puts them in `done`
  {"__pictures_started__": True}   for runtime.py only: pictures exist now, hand their record over before waiting again
It takes the model call and the picture source as arguments, so it runs without any provider key (see runtime.py
for the binding to this app). Closing the generator cancels parts that have not started.

The outline is streamed. The title and subtitle open the post (BLOG_STREAM_TITLE) as soon as the model has written them,
well before the rest of the outline exists; the `outline` event, the pictures and the parts follow as before.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter
from functools import partial
from typing import Iterator, Optional, Union

from config import settings
from src.longform.markdown_ast import parse
from src.longform.ordered_stream import run_ordered
from src.observability import trace_mark, trace_set

from . import schema as s
from .draft import units_from_outline, write_unit
from .llm import LLM
from .media import ImageSource, start_images
from .outline import build_outline
from .plan import BlogPlan, BlogRequest
from .sources import WebSources
from .validate import article_warnings

logger = logging.getLogger(__name__)


def title_text(title: str, subtitle: Optional[str]) -> str:
    return f"# {title}\n\n" + (f"*{subtitle}*\n\n" if subtitle else "")


def title_block(outline: s.Outline) -> str:
    return title_text(outline.title, outline.subtitle)


PICTURES_STARTED = {"__pictures_started__": True}


def _outline_events(llm: LLM, request: BlogRequest, plan: BlogPlan, request_id: Optional[str],
                    early_title: bool) -> Iterator[tuple]:
    """
    ("title", (title, subtitle)) once, as soon as the model has written both, then ("outline", OutlineResult). The model call
    is most of the time to first text, so with `early_title` it runs in a thread and this generator opens the post meanwhile.
    A failure is raised here, where the stream can report it; if the consumer leaves, the call finishes unobserved.
    """
    if not early_title:
        yield "outline", build_outline(llm, request, plan, request_id)
        return
    events: "queue.Queue[tuple]" = queue.Queue()

    def work() -> None:
        try:
            events.put(("outline", build_outline(llm, request, plan, request_id,
                                                 on_title=lambda title, subtitle: events.put(("title", (title, subtitle))))))
        except BaseException as error:                  # whatever it was, the consumer must wake up and raise it
            events.put(("error", error))

    threading.Thread(target=work, name="blog_outline", daemon=True).start()
    while True:
        kind, payload = events.get()
        if kind == "error":
            raise payload
        yield kind, payload
        if kind == "outline":
            return


def outline_event(outline: s.Outline) -> dict:
    return {"type": "outline", "title": outline.title,
            "sections": [{"id": sec.id, "heading": sec.heading} for sec in outline.sections]
            + [{"id": "conclusion", "heading": outline.conclusion.heading}]}


def generate_blog_stream(
    request: BlogRequest,
    plan: BlogPlan,
    llm: LLM,
    media: Optional[ImageSource] = None,
    *,
    sources: Optional[WebSources] = None,
    request_id: Optional[str] = None,
    parallel: Optional[int] = None,
) -> Iterator[Union[str, dict]]:
    started = time.perf_counter()
    trace_set(request_id, mode="blog", blog_plan={
        "words": plan.target_words, "sections": plan.sections, "tables": plan.tables, "quotes": plan.quotes,
        "images": plan.images, "web": plan.use_web})

    opening, title_seconds, result = None, None, None      # opening: the title block, once it has been streamed
    for kind, payload in _outline_events(llm, request, plan, request_id, settings.blog_stream_title):
        if kind == "title":
            opening, title_seconds = title_text(*payload), round(time.perf_counter() - started, 2)
            trace_mark(request_id, "title_streamed")
            yield opening
        else:
            result = payload
    outline, warnings = result.outline, list(result.warnings)
    outline_seconds = round(time.perf_counter() - started, 2)
    trace_mark(request_id, "outline_done")
    yield {"__event__": outline_event(outline)}

    # Pictures start now, in the background, so they are usually ready before the text that carries them.
    placed = start_images(media, outline)
    for sec_id, picture in placed.items():
        if picture.placeholder_id is None:
            warnings.append({"kind": "image_skipped", "unit": sec_id, "detail": f"No picture for {sec_id}: {picture.reason}"})
    if plan.images_asked is not None and plan.images_asked > plan.images:
        warnings.append({"kind": "images_limited", "detail":
                         f"{plan.images_asked} images were asked for; this post gets {plan.images} ({plan.reasons.get('images')})."})

    if opening is None:
        opening = title_block(outline)
        yield opening
    elif any(p.placeholder_id for p in placed.values()):
        # The title went out before any picture existed, so the next chunk is a whole part away: say now that pictures are
        # started (they are billed work), or a client that leaves in between would never have them released.
        yield PICTURES_STARTED

    units = units_from_outline(outline, placed)
    tasks = [partial(write_unit, llm, request, plan, outline, unit, units[i - 1].angle if i else None, sources, request_id)
             for i, unit in enumerate(units)]
    drafted, texts = [], [opening]
    for i, part in enumerate(run_ordered(tasks, parallel or settings.blog_section_parallel, thread_name_prefix="blog_unit")):
        drafted.append(part)
        warnings.extend(part.warnings)
        if i == 0:
            trace_mark(request_id, "first_unit")
        text = part.text.rstrip() + "\n" if i == len(units) - 1 else part.text
        texts.append(text)
        yield text
    trace_mark(request_id, "draft_done")

    document = "".join(texts)
    warnings.extend(article_warnings(
        document, tables=plan.tables, quotes=plan.quotes,
        headings=[u.heading for u in units if u.heading]))
    blocks = parse(document)
    fixes = Counter(f for part in drafted for f in part.fixes)
    trace_set(request_id, blog={
        "sections": len(outline.sections),
        "tables": sum(b.kind == "table" for b in blocks),
        "quotes": sum(b.kind == "blockquote" for b in blocks),
        "images": sum(1 for p in placed.values() if p.placeholder_id),
        "repairs": int(result.repaired) + sum(p.repaired for p in drafted),
        "repair_reasons": {p.unit.id: p.repair_kinds for p in drafted if p.repaired},
        "outline_filled": result.filled,
        "fixes": dict(fixes),
        "notes": dict(Counter(n for p in drafted for n in p.notes)),
        "warnings": len(warnings),
        "title_seconds": title_seconds,
        "outline_seconds": outline_seconds,
        "draft_seconds": round(time.perf_counter() - started - outline_seconds, 2),
        "words": sum(p.words for p in drafted),
    })
    if warnings:
        yield {"__warnings__": warnings}
