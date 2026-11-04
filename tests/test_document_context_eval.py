"""Live evaluation grading must catch incorrect facts, mutations and incomplete SSE streams."""
import pytest

from scripts.evaluate_document_context import assess, cases, sse_events
from src.document_edit.blocks import fingerprint


def response(case, text):
    return [{'event': 'document_context_ready', 'data': {'status': 'ready', 'source_sha256': fingerprint(case['source']),
             'coverage': case.get('coverage', 'full')}}, {'event': 'content', 'data': {'content': text}}, {'event': 'done', 'data': {}}]


def test_sse_parser_handles_named_events_comments_multiline_and_unterminated_final_event():
    lines = [': heartbeat', 'event: document_context_ready', 'data: {"status":', 'data: "ready"}', '',
             'data: {"type":"content","content":"Hi"}', '', 'data: {"type":"done"}']
    assert list(sse_events(lines)) == [{'event': 'document_context_ready', 'data': {'status': 'ready'}},
        {'event': 'content', 'data': {'type': 'content', 'content': 'Hi'}}, {'event': 'done', 'data': {'type': 'done'}}]


def test_accuracy_rejects_wrong_numeric_facts_and_incomplete_streams():
    case = cases()[0]
    good = response(case, 'Juniper Lantern is led by Mira Vale with 36 participants for 21 days.')
    assert all(assess(case, good)[0].values())
    bad = response(case, 'Juniper Lantern is led by Mira Vale with 136 participants for 21 days.')
    assert assess(case, bad)[0]['required_facts'] is False
    assert assess(case, good[:-1])[0]['completed'] is False


@pytest.mark.parametrize('text, expected', [('The renewal price has not been decided.', True),
    ('No renewal price has been decided.', True), ('The renewal price is USD 27.', False)])
def test_missing_information_is_not_confused_with_the_known_pilot_price(text, expected):
    case = next(c for c in cases() if c['id'] == 'ask_absent')
    assert assess(case, response(case, text))[0]['no_invented_renewal_price'] is expected


def test_draft_checks_detect_suffix_repetition_and_unrequested_numbers():
    case = next(c for c in cases() if c['id'] == 'draft_full')
    text = 'Mira Vale will lead Juniper Lantern carefully. Gather feedback before expanding.'
    assert all(assess(case, response(case, text))[0].values())
    assert assess(case, response(case, text + ' ' + case['suffix']))[0]['no_suffix_duplication'] is False
    assert assess(case, response(case, text + ' Add 99 participants.'))[0]['no_new_figures'] is False


def test_edit_evaluation_requires_real_proposed_patches():
    case = next(c for c in cases() if c['id'] == 'edit_explicit')
    checks, _, _ = assess(case, [{'event': 'document_edit_ready', 'data': {'status': 'failed', 'patches': []}},
                               {'event': 'content', 'data': {'content': case['source']}}, {'event': 'done', 'data': {}}])
    assert checks['proposed'] is False
    assert checks['exact_target'] is False
    assert checks['verified_replay'] is False


def test_retrieval_grading_rejects_forged_ranges_quotes_and_citation_ids():
    from src.document_context import reference
    case = next(c for c in cases() if c['id'] == 'ask_retrieval')
    source = case['source']
    start = source.index('The cobalt inventory')
    ref = reference(source, start, start + len('The cobalt inventory is 91 units.'), 'r0').model_dump()
    events = response(case, 'The cobalt inventory is 91 units. [r0]')
    events[0]['data'].update(strategy='retrieval', references=[ref])
    assert all(assess(case, events)[0].values())
    ref['quote'] = 'The cobalt inventory is 92 units.'
    assert assess(case, events)[0]['exact_source_references'] is False
    ref['quote'] = 'The cobalt inventory is 91 units.'
    ref['start_utf16'] += 1
    assert assess(case, events)[0]['exact_source_references'] is False
    events[1]['data']['content'] = 'The cobalt inventory is 91 units. [r999]'
    assert assess(case, events)[0]['answer_cites_sources'] is False


def test_contradiction_grading_accepts_a_direct_comparison_without_requiring_the_word_conflict():
    case = next(c for c in cases() if c['id'] == 'ask_conflict')
    answer = ('The approved cohort size for the 21-day pilot period is stated as 36 participants in one source, '
              'but reported as 48 participants for the same pilot period in another source.')
    assert assess(case, response(case, answer))[0]['recognizes_conflict'] is True
    answer = 'Approved cohort size for the same 21-day pilot is given as both 36 participants and 48 participants.'
    assert assess(case, response(case, answer))[0]['recognizes_conflict'] is True
    assert assess(case, response(case, 'The cohort has 36 participants and the updated register has 48.'))[0]['recognizes_conflict'] is False


def test_analysis_negative_control_requires_completed_no_findings():
    case = next(c for c in cases() if c['id'] == 'analysis_separate_periods')
    events = response(case, 'No verified contradictions were identified.')
    analysis = {'status': 'no_findings', 'context': events[0]['data'], 'findings': []}
    events.insert(1, {'event': 'document_analysis_ready', 'data': analysis})
    assert all(assess(case, events)[0].values())
    analysis['findings'] = [{'text': 'A false contradiction.', 'evidence_ids': ['e0', 'e1']}]
    assert assess(case, events)[0]['no_false_contradiction'] is False
    analysis['status'] = 'ready'
    assert assess(case, events)[0]['analysis_complete'] is False


@pytest.mark.parametrize('shorter', [True, False])
def test_style_case_requires_a_shorter_target_and_preserved_source(shorter):
    import json
    from src.document_edit.blocks import document_blocks
    from src.document_edit.service import edit_document
    case = next(c for c in cases() if c['id'] == 'edit_style')
    target = next(b for b in document_blocks(case['source']) if b.editable and b.section[-1:] == ('Introduction',))
    def llm(system, messages, spec):
        if spec.stage == 'document_plan':
            return json.dumps({'status': 'ready', 'message': 'Edit Introduction only.', 'targets': [{
                'block_id': target.id, 'request_quote': 'Rewrite the Introduction paragraph', 'reason': 'Match the guide style.'}]})
        return 'Prepare carefully for the session.' if shorter else target.text
    text, proposal = edit_document(case['source'], case['query'], llm, match_style=True)
    events = [{'event': 'document_edit_ready', 'data': proposal.model_dump()},
        {'event': 'content', 'data': {'content': text}}, {'event': 'done', 'data': {}}]
    checks, _, _ = assess(case, events)
    assert checks['style_shorter_target'] is shorter
    if shorter:
        assert all(checks.values())
