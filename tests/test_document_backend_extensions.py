"""Scoped evidence, complete-source coverage, budget failures and advisory routing."""
import json
import threading
import time

import pytest

from src.document_context import DocumentContext, prepare_context, source_index
from src.document_retrieval import chunk_source
from src.document_intent import suggest_document_intent
from src.document_analysis import (AnalysisBudget, MAP_OUTPUT_BYTES, MAX_INPUT_BYTES, analyze_document)

FIRST = 'The approved cohort has 36 participants.'
LAST = 'The approved cohort has 48 participants.'
SOURCE = '# First claim\n\n' + FIRST + '\n\n' + ('Background material is retained exactly for the complete document review.\n\n' * 120) + '## Final claim\n\n' + LAST + '\n'


def event_data(chunks, name):
    return next(c['__event__']['data'] for c in chunks if isinstance(c, dict) and c.get('__event__', {}).get('event') == name)


class ScriptedAnalysis:
    def __init__(self, *, verdict='supported', bad_quote=False, bad_id=False, fail=False):
        self.calls = []
        self.verdict, self.bad_quote, self.bad_id, self.fail = verdict, bad_quote, bad_id, fail

    def __call__(self, system, messages, spec):
        data = json.loads(messages[0]['content'])
        self.calls.append((spec.stage, data))
        if spec.stage == 'document_analysis_extract':
            if self.fail:
                raise RuntimeError('Synthetic provider failure')
            quotes = [q for q in (FIRST, LAST) if q in data['source']]
            if self.bad_quote:
                quotes = ['An invented source passage.']
            return json.dumps({'claims': [{'statement': q, 'quote': q} for q in quotes]})
        if spec.stage == 'document_analysis_reduce':
            ids = [e['id'] for e in data['evidence']]
            return json.dumps({'findings': [{'text': 'The draft gives two different approved cohort sizes: 36 and 48.',
                'evidence_ids': ['e999'] if self.bad_id else ids}]})
        assert spec.stage == 'document_analysis_verify'
        assert FIRST in json.dumps(data) and LAST in json.dumps(data)
        return json.dumps({'decisions': [{'finding_id': f['finding_id'], 'verdict': self.verdict} for f in data['findings']]})


@pytest.mark.parametrize('split_sections', [True, False])
@pytest.mark.parametrize('source', ['', '\ufeff', '\ufeff# A\r\n\r\n😀 a\r\n' * 50, 'Heading\n-------\n\n' + '中文 👩‍💻 prose. ' * 500])
def test_chunks_preserve_every_source_character_and_bounded_utf8(source, split_sections):
    chunks = chunk_source(source, 200, split_sections=split_sections)
    assert ''.join(c.text for c in chunks) == source
    assert all(len(c.text.encode('utf-8')) <= 200 for c in chunks)
    assert all(c.text == source[c.start:c.end] for c in chunks)
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))


@pytest.mark.parametrize('position', [0, 50, 100])
def test_retrieval_finds_unique_evidence_at_any_document_position(position):
    sections = [f'## Section {i}\n\nOrdinary background notes for this section.\n\n' for i in range(101)]
    sections[position] += 'The cobalt inventory is 91 units.\n\n'
    source = ''.join(sections)
    prepared = prepare_context(DocumentContext(text=source), 'What is the cobalt inventory?', scope='focused')
    assert prepared.metadata.strategy == 'retrieval'
    assert any('cobalt inventory is 91' in ref.quote for ref in prepared.metadata.references)
    assert len(prepared.prompt.encode('utf-8')) < 6500
    for ref in prepared.metadata.references:
        assert source[source_index(source, ref.start_utf16):source_index(source, ref.end_utf16)] == ref.quote


def test_selection_scope_never_retrieves_outside_the_captured_occurrence():
    source = '😀 First cobalt inventory is 1.\n\nSecond cobalt inventory is 91.'
    start = source.index('Second')
    selection = {'start_utf16': len(source[:start].encode('utf-16-le')) // 2, 'end_utf16': len(source.encode('utf-16-le')) // 2}
    prepared = prepare_context(DocumentContext(text=source, selection=selection), 'What is the cobalt inventory?', scope='selection')
    assert prepared.metadata.coverage == 'partial'
    assert all(ref.start_utf16 >= selection['start_utf16'] for ref in prepared.metadata.references)
    assert 'First cobalt' not in prepared.prompt


@pytest.mark.parametrize('query, expected', [('¿Cuál es la población de Córdoba?', 'Córdoba'), ('北京的人口是多少', '北京')])
def test_unicode_retrieval_and_duplicate_heading_paths(query, expected):
    source = '# Spain\n\n## Facts\n\nCórdoba tiene una población ficticia de 37 habitantes.\n\n# China\n\n## Facts\n\n北京的人口是虚构的91人。\n'
    prepared = prepare_context(DocumentContext(text=source), query, scope='focused')
    assert expected in prepared.prompt
    assert prepared.metadata.references
    assert any('# Spain' in r.section or '# China' in r.section for r in prepared.metadata.references)


def test_no_match_and_missing_fts5_do_not_create_a_model_prompt(monkeypatch):
    source = DocumentContext(text='## Inventory\n\nCobalt stocks are stable.')
    prepared = prepare_context(source, 'zebra spacetime unicorn', scope='focused')
    assert prepared.message and prepared.metadata.billable_input_chars == 0
    import sqlite3
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError('module unavailable')
    monkeypatch.setattr('src.document_retrieval.sqlite3.connect', unavailable)
    assert prepare_context(source, 'cobalt', scope='focused').metadata.status == 'needs_context'


def test_analysis_reads_all_chunks_and_verifies_distant_original_passages():
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Check cohort sizes', analysis='contradictions')
    assert prepared.metadata.coverage == 'pending'
    llm = ScriptedAnalysis()
    chunks = list(analyze_document(prepared, llm))
    result = event_data(chunks, 'document_analysis_ready')
    assert result['context']['coverage'] == 'full'
    assert result['context']['included_source_chars'] == len(SOURCE)
    maps = [data['source'] for data in sorted((data for stage, data in llm.calls if stage == 'document_analysis_extract'),
                                             key=lambda data: int(data['chunk_id'][1:]))]
    assert ''.join(maps) == SOURCE
    assert len(maps) == len(prepared.analysis.chunks)
    assert result['findings'][0]['evidence_ids'] == ['e0', 'e1']
    for ref in result['context']['references']:
        assert SOURCE[source_index(SOURCE, ref['start_utf16']):source_index(SOURCE, ref['end_utf16'])] == ref['quote']
    assert result['billing']['input_chars'] <= prepared.metadata.reserved_input_chars
    assert result['billing']['output_chars'] <= prepared.metadata.reserved_output_chars


@pytest.mark.parametrize('mode', ['summary', 'contradictions'])
def test_unsupported_or_different_time_period_findings_are_not_reported(mode):
    prepared = prepare_context(DocumentContext(text=SOURCE.replace('First claim', '2024 results').replace('Final claim', '2025 results')),
        'Review cohort sizes', analysis=mode)
    llm = ScriptedAnalysis(verdict='unsupported')
    chunks = list(analyze_document(prepared, llm))
    result = event_data(chunks, 'document_analysis_ready')
    assert result['findings'] == [] and result['rejected_findings'] == 1
    assert result['status'] == 'no_findings'
    assert not any('two different approved' in c for c in chunks if isinstance(c, str))


@pytest.mark.parametrize('options', [{'bad_quote': True}, {'bad_id': True}, {'fail': True}])
def test_analysis_failure_cannot_emit_a_completed_full_coverage_result(options):
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Check cohort sizes', analysis='contradictions')
    output = []
    with pytest.raises((ValueError, RuntimeError)):
        for chunk in analyze_document(prepared, ScriptedAnalysis(**options)):
            output.append(chunk)
    assert not any(isinstance(c, str) or c.get('__event__', {}).get('event') == 'document_analysis_ready' for c in output)


def test_analysis_cancellation_deadline_and_provider_attempt_limits_prevent_new_calls():
    from src.document_edit.service import EditCancelled
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Check cohort sizes', analysis='summary')
    stop = threading.Event()
    stop.set()
    llm = ScriptedAnalysis()
    with pytest.raises(EditCancelled):
        list(analyze_document(prepared, llm, cancel_event=stop))
    assert not llm.calls
    budget = AnalysisBudget(prepared.analysis)
    budget.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        budget.call(llm, '', {}, 'document_analysis_extract', MAP_OUTPUT_BYTES)
    assert not llm.calls
    budget = AnalysisBudget(prepared.analysis)
    for _ in range(2 * prepared.analysis.max_stage_calls):
        budget.attempt()
    with pytest.raises(ValueError):
        budget.attempt()


def test_analysis_chunk_and_instruction_caps_are_checked_before_model_work():
    prepared = prepare_context(DocumentContext(text='👩' * 50000), 'Summarize', analysis='summary')
    assert prepared.metadata.status == 'needs_context' and prepared.message
    assert prepared.analysis is None
    with pytest.raises(ValueError):
        prepare_context(DocumentContext(text=SOURCE), 'x' * 2001, analysis='summary')


@pytest.mark.parametrize('decisions', [[], [{'finding_id': 'f1', 'verdict': 'supported'}],
    [{'finding_id': 'f0', 'verdict': 'supported'}] * 2])
def test_missing_unknown_or_duplicate_verification_decisions_fail_closed(decisions):
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Check cohort sizes', analysis='contradictions')
    scripted = ScriptedAnalysis()
    def llm(system, messages, spec):
        return json.dumps({'decisions': decisions}) if spec.stage == 'document_analysis_verify' else scripted(system, messages, spec)
    output = []
    with pytest.raises(ValueError, match='exactly one verification'):
        output.extend(analyze_document(prepared, llm))
    assert not any(c.get('__event__', {}).get('event') == 'document_analysis_ready' for c in output if isinstance(c, dict))


def test_analysis_cancel_after_extraction_stops_reduction():
    from src.document_edit.service import EditCancelled
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Check cohort sizes', analysis='summary')
    cancel = threading.Event()
    scripted = ScriptedAnalysis()
    stream = analyze_document(prepared, scripted, cancel_event=cancel)
    next(stream)
    cancel.set()
    with pytest.raises(EditCancelled):
        list(stream)
    assert all(stage == 'document_analysis_extract' for stage, _ in scripted.calls)


def test_analysis_stage_input_output_and_call_limits_are_enforced():
    prepared = prepare_context(DocumentContext(text=SOURCE), 'Summarize', analysis='summary')
    budget = AnalysisBudget(prepared.analysis)
    llm = ScriptedAnalysis()
    with pytest.raises(ValueError, match='input allowance'):
        budget.call(llm, '', {'source': 'x' * MAX_INPUT_BYTES}, 'document_analysis_extract', MAP_OUTPUT_BYTES)
    assert not llm.calls
    with pytest.raises(ValueError, match='output allowance'):
        budget.call(lambda *args: 'x' * (MAP_OUTPUT_BYTES + 1), '', {}, 'document_analysis_extract', MAP_OUTPUT_BYTES)
    budget = AnalysisBudget(prepared.analysis)
    budget.stage_calls = prepared.analysis.max_stage_calls
    with pytest.raises(ValueError, match='call or input allowance'):
        budget.call(llm, '', {}, 'document_analysis_extract', MAP_OUTPUT_BYTES)
    assert not llm.calls


@pytest.mark.parametrize('query, expected, ambiguous', [
    ('What would you rewrite?', 'ask', False), ('Rewrite the conclusion', 'edit', False),
    ('Please write the next paragraph', 'draft', False), ('Can you rewrite the conclusion?', 'edit', False),
    ('Suggest better headings', 'ask', False), ('Summarize this draft', 'ask', False),
    ('Rewrite the conclusion and draft a new section', 'ask', True), ('Help', 'ask', True),
    ('Reescribe la conclusión', 'edit', False), ('Continúa escribiendo', 'draft', False),
    ('अगला अनुच्छेद लिखो', 'draft', False),
])
def test_intent_suggestions_are_advisory_and_explicit_action_always_wins(query, expected, ambiguous):
    suggestion = suggest_document_intent(query, requested_action='ask')
    assert suggestion.suggested_action == expected
    assert suggestion.ambiguous == ambiguous
    assert suggestion.advisory is True and suggestion.effective_action == 'ask'
