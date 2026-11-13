"""Exact source chunks and request-local FTS5 retrieval; no stored documents or embedding calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import closing
import json
import re
import sqlite3

from config import settings
from .document_context import (ContextMetadata, ContextRange, PreparedContext,
                               reference, source_index, unavailable_context)
from .document_edit.blocks import fingerprint
from .longform.markdown_ast import parse


@dataclass(frozen=True)
class SourceChunk:
    id: str
    start: int
    end: int
    text: str = field(repr=False)
    section: tuple[str, ...] = ()


def chunk_source(text: str, max_bytes: int = 4000, *, split_sections: bool = True) -> list[SourceChunk]:
    if max_bytes < 4:
        raise ValueError('Chunk allowance must hold a Unicode character')
    result, headings = [], []
    at = 1 if text.startswith('\ufeff') else 0
    start, end = 0, 0
    path = ()

    def flush():
        nonlocal start
        if end > start:
            result.append(SourceChunk(f'c{len(result)}', start, end, text[start:end], path if split_sections else ()))
        start = end

    for block in parse(text[at:]):
        block_end = at + len(block.text)
        if block.kind == 'heading':
            if split_sections:
                flush()
            title = block.text.splitlines()[0].encode('utf-8')[:120].decode('utf-8', errors='ignore')
            headings = [(level, value) for level, value in headings if level < block.level]
            headings.append((block.level, title))
            path = tuple(value for _, value in headings)
        if len(text[start:block_end].encode('utf-8')) > max_bytes:
            flush()
        while len(text[start:block_end].encode('utf-8')) > max_bytes:
            prefix = text[start:block_end].encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
            newline = prefix.rfind('\n')
            if newline >= len(prefix) // 2:
                prefix = prefix[:newline + 1]
            end = start + len(prefix)
            flush()
        end, at = block_end, block_end
    if not result and not text[at:]:
        end = len(text)
    flush()
    assert ''.join(c.text for c in result) == text
    return result


_STOP_WORDS = frozenset('a an the is are was were what where when how why which does do did please tell me about this that of in on for to and or'.split())


def search_terms(text):
    words = re.findall(r'[^\W_]+', text.casefold())
    terms = [w for w in words if w not in _STOP_WORDS]
    for word in words:
        for run in re.findall(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+', word):
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return list(dict.fromkeys(terms))


def _section_range(text, target):
    name, number = target
    headings = []
    start = int(text.startswith('\ufeff'))
    for block in parse(text[start:]):
        if block.kind == 'heading':
            title = re.sub(r'^ {0,3}#{1,6}[ \t]+|[ \t]+#+[ \t]*$', '', block.text.splitlines()[0])
            title = re.sub(r'\s*\{[#.][^}\n]{0,100}\}\s*$', '', title)
            headings.append((start, block.level, ' '.join(title.split()).casefold()))
        start += len(block.text)
    name = name.casefold()
    if number is None:
        matches = [i for i, (_, _, title) in enumerate(headings) if title == name]
    else:
        number = number.casefold()
        prefix = re.compile(rf'^(?:{re.escape(name)}|{re.escape(number)})(?=$|[\s:.)–—-])(?![.\-]\d)')
        matches = [i for i, (_, _, title) in enumerate(headings) if prefix.match(title)]
    if len(matches) != 1:
        message = 'More than one heading matches' if matches else 'No heading matches'
        raise ValueError(f'{message} “{target[0]}”. Quote the full heading title or select the relevant text.')
    index, = matches
    start, level, _ = headings[index]
    end = next((at for at, depth, _ in headings[index + 1:] if depth <= level), len(text))
    return start, end


def retrieve_context(context, query, scope, *, prefer_cursor=True, target=None):
    """
    Scoped Ask: the passages of the document that best match the question, as exact quoted excerpts with references.

    Request-local FTS5 index, BM25 ranking, Porter stemming ("selling" finds "sell", "layers" finds "layer"). `prefer_cursor` False (document_scope "auto")
    puts a passage first only for an explicit selection, not for a bare cursor, which says nothing about what the question is about.
    """
    text = context.text
    selection = context.selection
    has_selection = selection is not None and selection.end_utf16 > selection.start_utf16
    if scope == 'selection' and not has_selection:
        raise ValueError('Selection scope requires a nonempty selection')
    base, end = (source_index(text, selection.start_utf16), source_index(text, selection.end_utf16)) if scope == 'selection' else (0, len(text))
    if target is not None:
        try:
            section_start, section_end = _section_range(text, target)
        except ValueError as error:
            return unavailable_context(context, 'ask', scope, str(error))
        base, end = max(base, section_start), min(end, section_end)
        if base >= end:
            return unavailable_context(context, 'ask', scope, 'The selection is outside the requested section. Select text in that section or clear the selection scope.')
    scoped_text = text[base:end]
    if not scoped_text.strip():
        return unavailable_context(context, 'ask', scope, 'There is no source content in the requested scope.')
    chunks = [SourceChunk(c.id, c.start + base, c.end + base, c.text, c.section) for c in chunk_source(scoped_text, 1600)]
    ranked = []
    if (scope == 'selection' or target is not None) and len(scoped_text.encode('utf-8')) < 3500:
        ranked = list(range(len(chunks)))
    else:
        terms = search_terms(query)[:32]
        try:
            with closing(sqlite3.connect(':memory:')) as connection:
                connection.execute("CREATE VIRTUAL TABLE passages USING fts5(heading, body, tokenize='porter unicode61')")
                connection.executemany('INSERT INTO passages(rowid, heading, body) VALUES (?, ?, ?)',
                    [(i + 1, ' '.join(search_terms(' '.join(c.section))), ' '.join(search_terms(c.text))) for i, c in enumerate(chunks)])
                if terms:
                    expression = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
                    ranked = [row[0] - 1 for row in connection.execute(
                        'SELECT rowid FROM passages WHERE passages MATCH ? ORDER BY bm25(passages, 4.0, 1.0), rowid', (expression,))]
        except sqlite3.OperationalError:
            return unavailable_context(context, 'ask', scope, 'Document retrieval is unavailable on this server. Send a smaller source or selection.')
        if target is not None:
            # A heading-only query should still cover descendants and chunks that do not repeat the heading terms.
            matched = set(ranked)
            ranked.extend(i for i in range(len(chunks)) if i not in matched)
        if has_selection or (prefer_cursor and context.cursor_utf16 is not None):
            focus = source_index(text, selection.start_utf16) if has_selection else source_index(text, context.cursor_utf16)
            focused = next((i for i, c in enumerate(chunks) if c.start <= focus < c.end or focus == c.end == len(text)), None)
            if focused is not None:
                ranked = [focused, *[i for i in ranked if i != focused]]
    if not ranked:
        return unavailable_context(context, 'ask', scope, 'No matching source passages were found. Name a section or select the relevant text.')
    picked = []

    def envelope(indices):
        ordered = sorted(indices, key=lambda i: chunks[i].start)
        refs = [reference(text, chunks[i].start, chunks[i].end, f'r{j}', chunks[i].section) for j, i in enumerate(ordered)]
        data = json.dumps({'action': 'ask', 'scope': scope, 'coverage': 'partial',
            'source_sha256': fingerprint(text), 'excerpts': [r.model_dump(mode='json') for r in refs]}, ensure_ascii=False, separators=(',', ':'))
        return data, refs

    budget = settings.document_retrieval_budget_bytes
    for i in ranked:
        data, _ = envelope([*picked, i])
        if len(data.encode('utf-8')) <= budget:
            picked.append(i)
    if not picked:
        return unavailable_context(context, 'ask', scope, 'The matching source could not fit. Select a smaller passage.')
    data, refs = envelope(picked)
    coverage = 'full' if sum(chunks[i].end - chunks[i].start for i in picked) == len(text) else 'partial'
    # The envelope remains conservative about scope even when all text happened to fit.
    prompt = data + '\n\nUSER_TASK:\n' + query
    metadata = ContextMetadata(action='ask', response_kind='chat_answer', status='ready', source_sha256=fingerprint(text),
        source_chars=len(text), coverage=coverage, strategy='selection' if scope == 'selection' else 'retrieval', scope=scope,
        included_source_chars=sum(len(r.quote) for r in refs), context_chars=len(data), references=refs,
        included_ranges=[ContextRange(start_utf16=r.start_utf16, end_utf16=r.end_utf16) for r in refs],
        estimated_input_tokens=len(prompt.encode('utf-8')), billable_input_chars=len(data) + len(query))
    return PreparedContext(prompt, metadata)
