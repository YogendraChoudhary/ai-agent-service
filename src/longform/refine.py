"""
Mode 1: refine a long document passage by passage, and prove each passage kept its shape.

  normalise -> parse -> freeze (tables, quotes, images, code, URLs become tokens) -> chunk (~1,000 words, whole blocks)
    -> refine chunks in parallel (the model sees the frozen text) -> verify each chunk in code -> one repair
    -> still wrong? the chunk is restored as written -> thaw (tokens back to the original bytes) -> emit in document order

The document is never summarised and never loses a block: a chunk the model cannot refine without moving something is
returned exactly as it came in, with a warning (decision D6 of docs/rich_content_engine_plan.md). This module takes the model
call as an argument (`LLM`), so it runs without any provider key; runtime.py binds it to this app.

refine_stream() yields what the stream endpoint understands:
  "markdown"                                         a verified, thawed chunk, in document order (never retracted)
  {"__event__": {"type": "refine_progress", ...}}    after each chunk: {"chunk": i, "of": n}
  {"__warnings__": [...]}                            once at the end, only if a chunk had to be restored
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Iterator, List, Optional, Union

from src.observability import trace_mark, trace_set

from .chunk import Chunk, chunk_blocks
from .freeze import collides, freeze_blocks, thaw
from .llm import LLM, CallSpec
from .markdown_ast import looks_like_html, normalise, parse, restyle, word_count
from .ordered_stream import run_ordered
from .prompts import Operation, ratio_window, system_prompt, user_message
from .verify import Issue, repair_message, verify

logger = logging.getLogger(__name__)

STRUCTURE_KINDS = ("table", "blockquote", "image", "fence", "html")      # blocks worth protecting even in a short text


@dataclass(frozen=True)
class RefineConfig:
    chunk_words: int = 1000
    first_chunk_words: int = 0                      # LONGFORM_FIRST_CHUNK_WORDS: a smaller first chunk, so text appears sooner (0 = off)
    parallel: int = 5
    freeze_quotes: bool = True
    repair: bool = True
    min_words: int = 600
    word_ratio: Optional[str] = None                # LONGFORM_WORD_RATIO: JSON override of the ratio windows


@dataclass
class ChunkResult:
    index: int
    text: str                                       # the chunk as it goes on: refined (still frozen) or the original
    words_in: int
    words_out: int
    repaired: bool = False
    restored: bool = False
    passthrough: bool = False                       # nothing to refine: no model call was made
    issues: List[Issue] = field(default_factory=list)       # why it was restored
    first_issues: List[str] = field(default_factory=list)   # what the first answer got wrong, for tuning
    seconds: float = 0.0


def qualifies(text: Optional[str], min_words: int = 600) -> bool:
    """
    D7: the chunked path runs for a document of at least `min_words` words, or one with structure worth protecting. Never for
    HTML (open item O1: the single call keeps handling it) and never for text that already looks like our tokens.
    """
    body, _ = normalise(text)
    if not body.strip() or looks_like_html(body) or collides(body):
        return False
    if word_count(body) >= min_words:
        return True
    return any(b.kind in STRUCTURE_KINDS or b.images for b in parse(body))


def chunk_max_tokens(chunk: Chunk) -> int:
    """D15: the answer is the passage again, so 1.6 x its estimated tokens (never below 1,024)."""
    return max(1024, int(1.6 * len(chunk.frozen) / 3.5))


def refine_chunk(llm: LLM, operation: Operation, instruction: Optional[str], chunk: Chunk, config: RefineConfig,
                 request_id: Optional[str] = None) -> ChunkResult:
    """One chunk: ask, verify, repair once, restore if it is still wrong. Provider errors are raised, never swallowed."""
    started = time.perf_counter()
    words_in = chunk.structure.words
    if not chunk.refinable:
        return ChunkResult(chunk.index, chunk.frozen, words_in, words_in, passthrough=True)

    system = system_prompt(operation.name)
    messages = [{"role": "user", "content": user_message(operation, instruction, chunk)}]
    unit, tokens = f"c{chunk.index + 1}", chunk_max_tokens(chunk)
    ratio = ratio_window(operation, config.word_ratio)
    reply = llm(system, messages, CallSpec("chunk", operation.temperature, tokens, unit=unit))
    verdict = verify(chunk, reply, ratio=ratio, paragraphs=operation.paragraphs)
    first_issues = [i.kind for i in verdict.issues]

    repaired = False
    if verdict.issues and config.repair:
        logger.info(f"[{request_id}] {unit} needs repair: {first_issues}")
        logger.debug(f"[{request_id}] {unit} first answer ({len(reply.split())} words, the passage has {words_in}): {reply[:300]!r}")
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": repair_message(verdict.issues)}]
        verdict = verify(chunk, llm(system, messages, CallSpec("chunk_repair", 0.0, tokens, unit=unit)), ratio=ratio,
                         paragraphs=operation.paragraphs)
        repaired = True
    seconds = round(time.perf_counter() - started, 2)
    if verdict.issues:
        logger.warning(f"[{request_id}] {unit} restored as written: {[i.kind for i in verdict.issues]}")
        return ChunkResult(chunk.index, chunk.original, words_in, words_in, repaired=repaired, restored=True,
                           issues=list(verdict.issues), first_issues=first_issues, seconds=seconds)
    return ChunkResult(chunk.index, verdict.text, words_in, verdict.words, repaired=repaired, first_issues=first_issues,
                       seconds=seconds)


def _warning(result: ChunkResult, total: int) -> dict:
    why = "; ".join(f"{i.kind}: {i.message}" for i in result.issues[:3])
    return {"kind": "chunk_restored", "chunk": result.index + 1,
            "detail": f"Passage {result.index + 1} of {total} was kept as written because the AI changed its structure ({why})"}


def refine_stream(text: str, instruction: Optional[str], operation: Operation, llm: LLM, config: Optional[RefineConfig] = None,
                  *, request_id: Optional[str] = None) -> Iterator[Union[str, dict]]:
    """Refine `text` (a document that qualifies) and stream it back in verified chunks, in document order."""
    config = config or RefineConfig()
    started = time.perf_counter()
    body, style = normalise(text)
    frozen, vault = freeze_blocks(parse(body), freeze_quotes=config.freeze_quotes)
    chunks = chunk_blocks(frozen, config.chunk_words, config.first_chunk_words)
    trace_set(request_id, mode="longform")
    trace_mark(request_id, "refine_planned")

    tasks = [partial(refine_chunk, llm, operation, instruction, c, config, request_id) for c in chunks]
    results: List[ChunkResult] = []
    for item in run_ordered(tasks, config.parallel, progress=True, thread_name_prefix="longform"):
        if isinstance(item, dict):                                       # the progress sentinel
            yield {"__event__": {"type": "refine_progress", **item["__progress__"]}}
            continue
        results.append(item)
        if len(results) == 1:
            trace_mark(request_id, "first_chunk")
        yield restyle(thaw(item.text, vault), style, first=len(results) == 1)

    total_in, total_out = sum(r.words_in for r in results), sum(r.words_out for r in results)
    reasons: Dict[str, List[str]] = {f"c{r.index + 1}": r.first_issues for r in results if r.first_issues}
    trace_set(request_id, longform={
        "operation": operation.name, "chunks": len(results), "parallel": config.parallel,
        "repaired": sum(r.repaired for r in results), "restored": sum(r.restored for r in results),
        "passthrough": sum(r.passthrough for r in results), "words_in": total_in, "words_out": total_out,
        "word_ratio": round(total_out / total_in, 3) if total_in else 1.0, "frozen": vault.summary(),
        "repair_reasons": reasons, "seconds": round(time.perf_counter() - started, 2)})
    warnings = [_warning(r, len(results)) for r in results if r.restored]
    if warnings:
        yield {"__warnings__": warnings}
