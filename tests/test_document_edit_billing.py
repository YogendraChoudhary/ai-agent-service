"""Canceled edits must not charge for a model call that never started."""
import threading
from types import SimpleNamespace

import pytest

import api
import src.text_processing_agent as tpa
from config import settings
from tests.document_edit.test_api import runtime as editor_runtime, body as edit_body
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import paid


@pytest.fixture
def paid_editor(monkeypatch):
    client, state = editor_runtime.__wrapped__(monkeypatch)
    state.update(charges=[], reserves=[], releases=[])
    paid((client, state), monkeypatch)
    monkeypatch.setattr(api, 'fire_credit_deduction_and_release', lambda **kw: state['charges'].append(kw))
    monkeypatch.setattr(api, 'fire_credit_deduction', lambda *a, **kw: state['charges'].append((a, kw)))
    monkeypatch.setattr(settings, 'document_context_enabled', True)
    return client, state


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('cancel_during_factory', [False, True])
def test_zero_call_canceled_edit_refunds_credit_and_reports_abort(paid_editor, monkeypatch, explicit, cancel_during_factory):
    client, state = paid_editor
    cancel = threading.Event()
    if cancel_during_factory:
        def factory(*args, **kwargs):
            cancel.set()
            return SimpleNamespace(run=lambda *a, **kw: pytest.fail('Provider call after cancellation'))
        monkeypatch.setattr(tpa, 'create_agent', factory)
    else:
        cancel.set()
    monkeypatch.setattr(api, 'register_cancellation_event', lambda rid: cancel)
    request = edit_body()
    if explicit:
        request['document_context'] = {'text': request.pop('document_text')}
        request['document_action'] = 'edit'
    events = parse_sse_events(client.post('/agent/stream', json=request).text)
    assert events[-1]['data']['type'] == 'done' and events[-1]['data']['aborted'] is True
    assert not state['calls'] and not state['charges']
    assert len(state['releases']) == 1
    assert state['releases'][0][0][1] == sum(state['reserves'])
    assert not any(e['event'] == 'document_edit_ready' for e in events)


@pytest.mark.parametrize('explicit', [False, True])
def test_completed_edit_remains_billable_and_preserves_the_existing_patch_flow(paid_editor, explicit):
    client, state = paid_editor
    request = edit_body()
    if explicit:
        request['document_context'] = {'text': request.pop('document_text')}
        request['document_action'] = 'edit'
    events = parse_sse_events(client.post('/agent/stream', json=request).text)
    assert any(e['event'] == 'document_edit_ready' for e in events)
    assert events[-1]['data'].get('aborted') is not True
    assert len(state['calls']) == 3 and len(state['charges']) == 1
    assert state['charges'][0]['success'] is True
    assert state['charges'][0]['reserved_scaled'] == sum(state['reserves'])
    assert not state['releases']


def test_unavailable_credit_service_blocks_paid_provider_work(paid_editor, monkeypatch):
    client, state = paid_editor
    monkeypatch.setattr(api, 'credit_manager', None)
    response = client.post('/agent/stream', json=edit_body())
    assert response.status_code == 503 and not state['calls'] and not state['charges']
