import json
from types import SimpleNamespace

import pytest

import src.text_processing_agent as tpa
from config import settings
from src.document_context import ContextRange
from src.document_edit.blocks import utf16_length
from src.document_edit.service import apply_document_patches, edit_document
from tests.design.test_design_api import parse_sse_events
from tests.document_edit.test_api import body, runtime  # noqa: F401


SOURCE = '🚚 Acme operates the warehouse. Acme handles returns.\r\n\r\nKeep this paragraph.\r\n'
PROOFREAD = 'Correct the spelling in this sentence.'
SELECTED = 'Teh warehouse ships orders to local stores.'
CORRECTED = 'The warehouse ships orders to local stores.'
PROSE = f'🚚 Dispatch update. {SELECTED} End of update.\r\n\r\n{SELECTED}\r\n'


def select(source, phrase, *, last=False):
    start = (source.rindex if last else source.index)(phrase)
    return ContextRange(start_utf16=utf16_length(source[:start]), end_utf16=utf16_length(source[:start + len(phrase)]))


def never_called(*args):
    pytest.fail('A literal or ambiguous edit must not call the model')


@pytest.mark.parametrize('query, expected', [
    ('Replace "Acme" with "Acme Inc."', 'Acme Inc.'),
    ('Change Acme to Acme Inc.', 'Acme Inc.'),
    ('add Inc. after Acme', 'Acme Inc.'),
    ('Add "," after "Acme"', 'Acme,'),
    ('Replace this with "Acme Inc."', 'Acme Inc.'),
    ('Write “Acme Inc.” in place', 'Acme Inc.'),
    ('Write Acme Inc. in place', 'Acme Inc.'),
])
def test_literal_edit_changes_only_the_selected_occurrence(query, expected):
    selection = select(SOURCE, 'Acme', last=True)
    result, metadata = edit_document(SOURCE, query, never_called, selection=selection)
    assert metadata.status == 'proposed'
    patch, = metadata.patches
    assert patch.before == 'Acme' and patch.after == expected
    assert (patch.start_utf16, patch.end_utf16) == (selection.start_utf16, selection.end_utf16)
    assert result == SOURCE.replace('Acme handles', f'{expected} handles')
    assert apply_document_patches(SOURCE, metadata.base_sha256, [patch]) == result


@pytest.mark.parametrize('query, selection, message', [
    ('Replace "Acme" with "Acme Inc."', None, 'more than once'),
    ('Replace "Acme" with "Acme Inc."', select(SOURCE, 'warehouse'), 'exact phrase'),
    ('Replace "Missing" with "Acme Inc."', None, 'exact phrase'),
    ('Replace "Acm" with "Acme Inc."', None, 'exact phrase'),
    ('Replace "Acme" with "Acme Inc." and shorten it', select(SOURCE, 'Acme'), 'exact change'),
    ('Replace Acme with Acme Inc. Keep pricing.', select(SOURCE, 'Acme'), 'exact change'),
    ('Write it in place', select(SOURCE, 'Acme'), 'exact change'),
    ('Write "Acme Inc." in place', None, 'Which text'),
    ('Correct the company name', None, 'Which text'),
    (PROOFREAD, None, 'Which text'),
    (PROOFREAD, select(SOURCE, 'returns.\r\n\r\nKeep'), 'single paragraph'),
])
def test_unresolved_target_asks_without_editing(query, selection, message):
    result, metadata = edit_document(SOURCE, query, never_called, selection=selection)
    assert metadata.status == 'needs_clarification' and message in metadata.message
    assert not metadata.patches and metadata.result_sha256 == metadata.base_sha256 and result == SOURCE


def test_a_unique_exact_phrase_can_be_used_without_a_selection():
    result, metadata = edit_document(SOURCE, 'Replace "handles returns" with "processes returns"', never_called)
    assert result == SOURCE.replace('handles returns', 'processes returns')
    assert [(patch.before, patch.after) for patch in metadata.patches] == [('handles returns', 'processes returns')]


def test_exact_phrase_inside_a_protected_link_leaves_the_document_unchanged():
    source = 'Read [the guide](https://old.example) now.\n'
    result, metadata = edit_document(source, 'Replace "old.example" with "new.example"', never_called)
    assert metadata.status == 'failed' and not metadata.patches and result == source


def test_generated_correction_only_replaces_the_selected_sentence():
    def llm(system, messages, spec):
        assert spec.stage == 'document_in_place'
        assert json.loads(messages[0]['content'])['TARGET'] == SELECTED
        return json.dumps({'replacement': CORRECTED})

    result, metadata = edit_document(PROSE, PROOFREAD, llm, selection=select(PROSE, SELECTED))
    assert result == PROSE.replace(SELECTED, CORRECTED, 1)
    patch, = metadata.patches
    assert patch.before == SELECTED and patch.after == CORRECTED
    assert apply_document_patches(PROSE, metadata.base_sha256, [patch]) == result


def test_the_model_can_correct_a_unique_quoted_typo_in_place():
    source = 'We recieved the delivery.\n'
    result, metadata = edit_document(source, 'Fix "recieved"', lambda *args: '{"replacement":"received"}')
    assert result == 'We received the delivery.\n'
    patch, = metadata.patches
    assert (patch.before, patch.after) == ('recieved', 'received')


@pytest.mark.parametrize('reply', [
    {'replacement': CORRECTED + ' It also handles returns.'},
    {'replacement': CORRECTED + ' It also handles returns'},
    {'replacement': CORRECTED.replace('ships', 'also ships')},
    {'replacement': CORRECTED.replace('stores.', 'stores!!!')},
    {'replacement': CORRECTED.replace('local stores', 'international stores')},
    {'replacement': f'Dispatch update. {CORRECTED} End of update.'},
    {'replacement': CORRECTED, 'extra': 'It also handles returns.'},
    'Here is the correction: ' + json.dumps({'replacement': CORRECTED}),
])
def test_unrequested_model_output_is_rejected_atomically(reply):
    calls = []

    def llm(system, messages, spec):
        calls.append(spec.stage)
        return reply if isinstance(reply, str) else json.dumps(reply)

    result, metadata = edit_document(PROSE, PROOFREAD, llm, selection=select(PROSE, SELECTED))
    assert calls == ['document_in_place', 'document_repair']
    assert metadata.status == 'failed' and not metadata.patches and result == PROSE


@pytest.mark.parametrize('endpoint', ['/agent/process', '/agent/stream'])
def test_selection_reaches_the_editor_and_apply(endpoint, runtime, monkeypatch):
    client, state = runtime
    monkeypatch.setattr(settings, 'document_context_enabled', True)
    selection = select(SOURCE, 'Acme', last=True)
    request = body(query='Replace "Acme" with "Acme Inc."', document_action='auto',
                   document_context={'text': SOURCE, 'selection': selection.model_dump()})
    del request['document_text']
    response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    if endpoint.endswith('process'):
        payload = response.json()
        assert payload['document_action_resolved']['action'] == 'edit'
        metadata, output = payload['document_edit'], payload['markdown_response']
    else:
        events = parse_sse_events(response.text)
        assert next(e['data'] for e in events if e['event'] == 'document_action_resolved')['action'] == 'edit'
        metadata = next(e['data'] for e in events if e['event'] == 'document_edit_ready')
        output = next(e['data']['content'] for e in events if e['data'].get('type') == 'content')
    assert metadata['status'] == 'proposed' and not state['calls']
    assert output == SOURCE.replace('Acme handles', 'Acme Inc. handles')
    payload = {'document_text': SOURCE, 'base_sha256': metadata['base_sha256'], 'patches': metadata['patches']}
    replay = client.post('/agent/documents/apply', json=payload)
    assert replay.status_code == 200 and replay.json()['document_text'] == output
    assert client.post('/agent/documents/apply', json={**payload, 'document_text': SOURCE + 'A new edit.'}).status_code == 409
    payload['patches'][0]['start_utf16'] = 1  # inside the opening emoji's UTF-16 surrogate pair
    assert client.post('/agent/documents/apply', json=payload).status_code == 400


@pytest.mark.parametrize('endpoint', ['/agent/process', '/agent/stream'])
def test_the_endpoint_keeps_original_text_when_the_model_adds_content(endpoint, runtime, monkeypatch):
    client, _ = runtime
    monkeypatch.setattr(settings, 'document_context_enabled', True)
    calls = []

    def make_agent(*args, **kwargs):
        def run(history, **kwargs):
            calls.append(json.loads(history[0].content)['TARGET'])
            return SimpleNamespace(content=json.dumps({'replacement': CORRECTED + ' It also handles returns.'}), metrics=None)

        return SimpleNamespace(run=run)

    monkeypatch.setattr(tpa, 'create_agent', make_agent)
    request = body(query=PROOFREAD, document_action='edit',
                   document_context={'text': PROSE, 'selection': select(PROSE, SELECTED).model_dump()})
    del request['document_text']
    response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    if endpoint.endswith('process'):
        payload = response.json()
        metadata, output = payload['document_edit'], payload['markdown_response']
    else:
        events = parse_sse_events(response.text)
        metadata = next(e['data'] for e in events if e['event'] == 'document_edit_ready')
        output = next(e['data']['content'] for e in events if e['data'].get('type') == 'content')
    assert calls == [SELECTED, SELECTED]
    assert metadata['status'] == 'failed' and not metadata['patches'] and output == PROSE
