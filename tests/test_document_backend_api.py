"""Production API/runner/factory integration for backend phases 5–7; only account/provider I/O is replaced."""
import json
import logging
import threading
from types import SimpleNamespace

import pytest

import api
import src.text_processing_agent as tpa
from config import settings
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import body, paid, runtime as existing_runtime
from tests.test_document_backend_extensions import SOURCE, FIRST, LAST, ScriptedAnalysis


@pytest.fixture
def analysis_runtime(monkeypatch):
    client, state = existing_runtime.__wrapped__(monkeypatch)
    scripted = ScriptedAnalysis()
    state['analysis'] = scripted
    def run(agent, history, **kwargs):
        assert kwargs.get('stream') is False
        assert not agent.tools
        assert agent.markdown is False
        data = json.loads(history[0].content)
        stage = 'document_analysis_extract' if 'chunk_id' in data else 'document_analysis_reduce' if 'evidence' in data else 'document_analysis_verify'
        state['calls'].append((stage, data))
        reply = scripted(agent.instructions, [{'role': 'user', 'content': history[0].content}], SimpleNamespace(stage=stage))
        return SimpleNamespace(content=reply, metrics={'input_tokens': 100, 'output_tokens': 20})
    monkeypatch.setattr(tpa.Agent, 'run', run)
    return client, state


def request(**changes):
    return body(document_context={'text': SOURCE}, document_analysis='contradictions', query='Check cohort sizes', **changes)


@pytest.mark.parametrize('mode', ['summary', 'contradictions'])
def test_process_and_stream_publish_completed_analysis_context_and_grounded_references(analysis_runtime, mode):
    client, state = analysis_runtime
    data = request()
    data['document_analysis'] = mode
    response = client.post('/agent/process', json=data)
    assert response.status_code == 200, response.text
    process = response.json()
    assert process['document_analysis']['status'] == 'ready'
    assert process['document_context']['coverage'] == 'full'
    assert process['document_context']['analysis_mode'] == mode
    assert process['document_context']['references']
    assert process['document_edit'] is None and process['design'] is None
    events = parse_sse_events(client.post('/agent/stream', json=data).text)
    names = [e['event'] or e['data'].get('type') for e in events]
    assert names[0:2] == ['start', 'document_context_ready']
    assert events[1]['data']['coverage'] == 'pending'
    assert 'document_analysis_progress' in names
    final = next(e['data'] for e in events if e['event'] == 'document_analysis_ready')
    assert final == process['document_analysis']
    assert names[-2:] == ['content', 'done']
    assert not state['charges']


@pytest.mark.parametrize('path', ['/agent/process', '/agent/stream'])
@pytest.mark.parametrize('changed', [
    {'document_scope': 'unknown'}, {'document_scope': 'selection'},
    {'document_scope': 'focused', 'document_action': 'draft'},
    {'document_scope': 'focused', 'document_analysis': 'summary'},
    {'document_action': 'edit', 'document_analysis': 'summary'},
    {'document_analysis': 'unknown'}, {'document_analysis': 'summary', 'query': 'x' * 2001},
    {'suggest_document_intent': 'true'},
])
def test_invalid_scope_analysis_and_suggestion_inputs_fail_before_provider_work(monkeypatch, path, changed):
    client, state = existing_runtime.__wrapped__(monkeypatch)
    response = client.post(path, json=body(**changed))
    assert response.status_code == 400, response.text
    assert not state['calls']


@pytest.mark.parametrize('option', [{'document_scope': 'focused'}, {'document_analysis': 'summary'}, {'suggest_document_intent': True}])
def test_new_options_cannot_be_silently_used_without_a_document(monkeypatch, option):
    client, _ = existing_runtime.__wrapped__(monkeypatch)
    data = body(**option)
    data.pop('document_context')
    assert client.post('/agent/process', json=data).status_code == 400


def test_scoped_retrieval_reaches_the_existing_tool_free_runner_and_keeps_evidence_private(monkeypatch, caplog):
    client, state = existing_runtime.__wrapped__(monkeypatch)
    monkeypatch.setattr(settings, 'log_request_bodies', True)
    with caplog.at_level(logging.INFO):
        response = client.post('/agent/process', json=body(document_context={'text': SOURCE},
            document_scope='focused', query='What is the approved cohort?'))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['document_context']['strategy'] == 'retrieval'
    assert result['document_context']['references']
    assert not state['calls'][0]['tools']
    assert FIRST not in caplog.text and LAST not in caplog.text


def test_intent_suggestions_are_optional_advisory_and_do_not_invoke_the_edit_engine(monkeypatch):
    client, state = existing_runtime.__wrapped__(monkeypatch)
    response = client.post('/agent/process', json=body(query='Rewrite the conclusion', suggest_document_intent=True))
    result = response.json()
    assert result['document_intent']['suggested_action'] == 'edit'
    assert result['document_intent']['effective_action'] == 'ask'
    assert result['document_intent']['advisory'] is True
    assert result['document_context']['action'] == 'ask' and result['document_edit'] is None
    events = parse_sse_events(client.post('/agent/stream', json=body(query='Write the next paragraph',
        document_action='ask', suggest_document_intent=True)).text)
    assert events[1]['event'] == 'document_intent_suggested'
    assert events[1]['data']['suggested_action'] == 'draft'
    assert events[1]['data']['effective_action'] == 'ask'
    assert len(state['calls']) == 2


def test_analysis_reserves_all_stages_and_settles_actual_intermediate_work_once(analysis_runtime, monkeypatch):
    client, _ = analysis_runtime
    state = paid(analysis_runtime, monkeypatch)
    events = parse_sse_events(client.post('/agent/stream', json=request()).text)
    prepared = events[1]['data']
    result = next(e['data'] for e in events if e['event'] == 'document_analysis_ready')
    assert state['reserves'] == [api.text_credits_scaled(prepared['reserved_input_chars'], prepared['reserved_output_chars'])]
    charge, = state['charges']
    assert charge['input_chars'] == result['billing']['input_chars'] == result['context']['billable_input_chars']
    assert charge['output_chars'] == result['billing']['output_chars'] == result['context']['billable_output_chars']
    assert charge['output_chars'] > len(next(e['data']['content'] for e in events if e['data'].get('type') == 'content'))
    assert charge['reserved_scaled'] == sum(state['reserves'])
    assert result['billing']['provider_attempts'] == result['billing']['stage_calls'] == len(state['calls'])


def test_analysis_credit_limit_rejects_the_whole_plan_before_model_work(analysis_runtime, monkeypatch):
    client, _ = analysis_runtime
    state = paid(analysis_runtime, monkeypatch)
    monkeypatch.setattr(api, 'MAX_RESERVATION_PER_REQUEST_SCALED', 1)
    response = client.post('/agent/stream', json=request())
    assert response.status_code == 402
    assert response.json()['code'] == 'REQUEST_CREDIT_LIMIT'
    assert not state['calls'] and not state['reserves']


def test_analysis_errors_release_reservation_and_never_publish_full_coverage(analysis_runtime, monkeypatch):
    client, state = analysis_runtime
    paid(analysis_runtime, monkeypatch)
    state['analysis'].bad_quote = True
    events = parse_sse_events(client.post('/agent/stream', json=request()).text)
    assert events[-1]['data']['type'] == 'error'
    assert not any(e['event'] == 'document_analysis_ready' for e in events)
    assert len(state['releases']) == 1
    assert state['charges'][0][1]['success'] is False


def test_analysis_abort_before_calls_releases_reservation_without_charge(analysis_runtime, monkeypatch):
    client, _ = analysis_runtime
    state = paid(analysis_runtime, monkeypatch)
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(api, 'register_cancellation_event', lambda request_id: cancel)
    events = parse_sse_events(client.post('/agent/stream', json=request()).text)
    assert events[-1]['data']['aborted'] is True
    assert not state['calls'] and not state['charges']
    assert len(state['releases']) == 1


def test_analysis_abort_during_extraction_refunds_unfinished_work(analysis_runtime, monkeypatch):
    client, _ = analysis_runtime
    state = paid(analysis_runtime, monkeypatch)
    cancel = threading.Event()
    monkeypatch.setattr(api, 'register_cancellation_event', lambda request_id: cancel)
    original = ScriptedAnalysis.__call__
    def abort(self, *args):
        reply = original(self, *args)
        cancel.set()
        return reply
    monkeypatch.setattr(ScriptedAnalysis, '__call__', abort)
    events = parse_sse_events(client.post('/agent/stream', json=request()).text)
    assert events[-1]['data']['aborted'] is True
    assert state['calls'] and all(stage == 'document_analysis_extract' for stage, _ in state['calls'])
    assert not state['charges'] and len(state['releases']) == 1
    assert not any(e['event'] == 'document_analysis_ready' for e in events)


def test_analysis_trace_and_body_logs_do_not_record_private_evidence(analysis_runtime, monkeypatch, caplog):
    client, _ = analysis_runtime
    monkeypatch.setattr(settings, 'log_request_bodies', True)
    with caplog.at_level(logging.INFO):
        response = client.post('/agent/process', json=request())
    assert response.status_code == 200
    assert FIRST not in caplog.text and LAST not in caplog.text
