"""Exercise real endpoint, routing, factory and streaming; replace provider calls and account services."""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agno.agent import RunEvent
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.document_context import CONTEXT_ERROR, DocumentContext, prepare_context
from src.document_edit.blocks import fingerprint
from tests.design.test_design_api import parse_sse_events

SOURCE = '# Proposal\n\nThe launch codename is Velvet Mango.\n\n## Conclusion\n\nStart with a small cohort.\n'


def test_document_context_defaults_enabled_and_respects_explicit_operator_override(monkeypatch):
    from config import Settings
    monkeypatch.delenv('DOCUMENT_CONTEXT_ENABLED', raising=False)
    assert Settings(_env_file=None).document_context_enabled is True
    monkeypatch.setenv('DOCUMENT_CONTEXT_ENABLED', 'false')
    assert Settings(_env_file=None).document_context_enabled is False


def body(**changes):
    return {'type': 'general', 'query': 'What is the launch codename?', 'model': 'openai',
            'api_key': 'endpoint-key', 'document_context': {'text': SOURCE}, **changes}


@pytest.fixture
def runtime(monkeypatch):
    state = {'calls': [], 'error': None, 'charges': [], 'reserves': [], 'releases': [], 'output_tokens': 4}
    monkeypatch.setattr(settings, 'document_context_enabled', True)
    monkeypatch.setattr(settings, 'task_routing_enabled', False)
    monkeypatch.setattr(settings, 'provider_fallback_enabled', False)
    monkeypatch.setattr(api, 'CREDIT_PRECHECK_ENABLED', False)
    monkeypatch.setattr(api, 'credit_manager', None)
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('provider-key', {
        'user_id': 'u1', 'requires_user_api_keys': True, 'user_keys': {}})))
    monkeypatch.setattr(api, 'fire_credit_deduction', lambda *a, **kw: state['charges'].append((a, kw)))
    monkeypatch.setattr(api, 'fire_credit_deduction_and_release', lambda **kw: state['charges'].append(kw))
    def no_history(*args, **kwargs):
        pytest.fail('Document context must not be saved in prompt history')
    monkeypatch.setattr(api, 'fire_prompt_history_save', no_history)

    def run(agent, prompt, **kwargs):
        state['calls'].append({'prompt': prompt, 'tools': agent.tools, 'instructions': agent.instructions,
            'model': agent.model.id, 'output_limit': getattr(agent.model, 'max_completion_tokens', None)})
        if state['error']:
            raise state['error']
        yield SimpleNamespace(event=RunEvent.run_content, content='Velvet Mango.', metrics=None)
        yield SimpleNamespace(event=RunEvent.run_completed, content=None, metrics={'input_tokens': 100, 'output_tokens': state['output_tokens']})
    monkeypatch.setattr(tpa.Agent, 'run', run)
    return TestClient(api.app, raise_server_exceptions=False), state


@pytest.mark.parametrize('action', [None, 'ask', 'draft'])
def test_process_uses_real_factory_without_tools_and_returns_context_metadata(runtime, action):
    client, state = runtime
    request = body(**({'document_action': action} if action else {}))
    result = client.post('/agent/process', json=request)
    assert result.status_code == 200, result.text
    result = result.json()
    assert result['document_context']['action'] == (action or 'ask')
    assert result['document_context']['coverage'] == 'full'
    assert result['document_context']['source_sha256'] == fingerprint(SOURCE)
    assert result['document_edit'] is None and result['design'] is None
    call, = state['calls']
    assert not call['tools']
    assert call['model'] == tpa.ADVANCED_MODEL_IDS['openai']
    assert call['output_limit'] == (5120 if action == 'draft' else 6144)
    assert SOURCE not in '\n'.join(call['instructions'])
    assert 'untrusted' in '\n'.join(call['instructions'])
    data = json.JSONDecoder().raw_decode(call['prompt'])[0]
    assert ''.join(e['text'] for e in data['excerpts']) == SOURCE
    assert not state['charges']


def test_stream_and_process_metadata_match_and_stream_keeps_existing_content_contract(runtime):
    client, state = runtime
    process = client.post('/agent/process', json=body()).json()
    streamed = client.post('/agent/stream', json=body())
    events = parse_sse_events(streamed.text)
    assert [e['event'] or e['data'].get('type') for e in events] == ['start', 'document_context_ready', 'content', 'done']
    assert events[1]['data'] == process['document_context']
    assert events[2]['data']['content'] == process['markdown_response']
    assert len(state['calls']) == 2


@pytest.mark.parametrize('path', ['/agent/process', '/agent/stream'])
@pytest.mark.parametrize('changed', [
    {'document_text': SOURCE}, {'type': 'fix_grammar'}, {'query': ' '}, {'query': 'x' * 8001},
    {'query': 'bad\ud800'}, {'query': 'bad\x00'}, {'secondary_prompt': 'extra'},
    {'document_action': 'unknown'}, {'existing_design_ast': {}}, {'image_provider': 'openai'},
    {'image_urls': ['https://example.org/image.png']},
    {'image_params': {'format': 'png'}}, {'blog_options': {}},
    {'document_context': {'text': SOURCE, 'cursor_utf16': 99999}},
    {'document_context': {'text': SOURCE, 'unexpected': True}},
    {'document_action': 'edit', 'query': 'x' * 2001},
    {'document_action': 'edit', 'document_context': {'text': ''}},
])
def test_invalid_new_contract_fails_before_provider_work(runtime, path, changed):
    client, state = runtime
    response = client.post(path, content=json.dumps(body(**changed)), headers={'Content-Type': 'application/json'})
    assert response.status_code == 400, response.text
    assert not state['calls']


@pytest.mark.parametrize('path', ['/agent/process', '/agent/stream'])
def test_action_without_source_and_disabled_feature_are_explicit_errors(runtime, monkeypatch, path):
    client, state = runtime
    request = body(document_action='ask')
    request.pop('document_context')
    assert client.post(path, json=request).status_code == 400
    monkeypatch.setattr(settings, 'document_context_enabled', False)
    response = client.post(path, json=body())
    assert response.status_code == 503
    assert response.json()['code'] == 'DOCUMENT_CONTEXT_DISABLED'
    assert not state['calls']


def test_context_task_bypasses_every_competing_router_even_when_query_requests_other_modes(runtime, monkeypatch):
    client, state = runtime
    def forbidden(*args, **kwargs):
        pytest.fail('A document task reached a competing router')
    for target in ['src.request_plan.plan_smart_request', 'src.request_plan.plan_design_request',
                   'src.task_router.task_for_request', 'src.blog.intent.detect_blog_intent']:
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(tpa, '_classify_image_request', forbidden)
    monkeypatch.setattr(tpa, 'determine_agent_type', forbidden)
    response = client.post('/agent/stream', json=body(query='Write a blog, generate code and create an image'))
    assert response.status_code == 200
    assert 'document_context_ready' in response.text
    assert not state['calls'][0]['tools']


@pytest.mark.parametrize('source', ['', 'x' * 8000])
def test_no_model_answers_skip_reservation_settlement_and_history(runtime, monkeypatch, source):
    client, state = runtime
    monkeypatch.setattr(api, 'CREDIT_PRECHECK_ENABLED', True)
    monkeypatch.setattr(settings, 'task_routing_enabled', True)
    monkeypatch.setattr(settings, 'provider_fallback_enabled', True)
    monkeypatch.setattr(api, 'credit_manager', SimpleNamespace(reserve=lambda *a: pytest.fail('No-model response reserved credits')))
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=(None, {
        'user_id': 'u1', 'requires_user_api_keys': False, 'user_keys': {}})))
    request = body(document_context={'text': source}, original_prompt='do not save this')
    streamed = client.post('/agent/stream', json=request)
    events = parse_sse_events(streamed.text)
    assert [e['event'] or e['data'].get('type') for e in events] == ['start', 'document_context_ready', 'content', 'done']
    assert events[1]['data']['status'] == 'needs_context'
    assert events[1]['data']['billable_input_chars'] == 0
    assert not state['calls'] and not state['charges']
    processed = client.post('/agent/process', json=request).json()
    assert processed['document_context'] == events[1]['data']


def test_large_draft_exposes_partial_coverage_warning_on_both_endpoints(runtime):
    client, _ = runtime
    request = body(document_context={'text': 'word ' * 8000}, document_action='draft')
    process = client.post('/agent/process', json=request).json()
    assert process['document_context']['coverage'] == 'partial'
    assert process['warnings'][0]['kind'] == 'partial_document_context'
    events = parse_sse_events(client.post('/agent/stream', json=request).text)
    assert events[1]['data'] == process['document_context']
    assert events[-1]['data']['warnings'] == process['warnings']


def paid(runtime, monkeypatch, *, allow=True):
    _, state = runtime
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('provider-key', {
        'user_id': 'u1', 'requires_user_api_keys': False, 'user_keys': {}})))
    def reserve(user, amount):
        state['reserves'].append(amount)
        return allow, 9999
    monkeypatch.setattr(api, 'CREDIT_PRECHECK_ENABLED', True)
    monkeypatch.setattr(api, 'credit_manager', SimpleNamespace(reserve=reserve,
        schedule_release=lambda *a, **kw: state['releases'].append((a, kw))))
    return state


def test_stream_billing_includes_the_actual_serialized_context_and_settles_once(runtime, monkeypatch):
    client, _ = runtime
    state = paid(runtime, monkeypatch)
    events = parse_sse_events(client.post('/agent/stream', json=body()).text)
    metadata = events[1]['data']
    assert metadata['billable_input_chars'] == metadata['context_chars'] + len(body()['query'])
    assert state['reserves'][0] == api.text_credits_scaled(metadata['billable_input_chars'], 0)
    charge, = state['charges']
    assert charge['input_chars'] == metadata['billable_input_chars']
    assert charge['output_chars'] == len('Velvet Mango.')
    assert charge['reserved_scaled'] == sum(state['reserves'])


@pytest.mark.parametrize('limit', ['balance', 'request'])
def test_credit_rejection_precedes_provider_work(runtime, monkeypatch, limit):
    client, _ = runtime
    state = paid(runtime, monkeypatch, allow=False)
    if limit == 'request':
        monkeypatch.setattr(api, 'MAX_RESERVATION_PER_REQUEST_SCALED', 1)
    response = client.post('/agent/stream', json=body())
    assert response.status_code == 402
    assert not state['calls'] and not state['charges']
    assert bool(state['reserves']) == (limit == 'balance')


def test_precancelled_request_releases_reservation_without_model_or_charge(runtime, monkeypatch):
    client, _ = runtime
    state = paid(runtime, monkeypatch)
    event = threading.Event()
    event.set()
    monkeypatch.setattr(api, 'register_cancellation_event', lambda rid: event)
    response = client.post('/agent/stream', json=body())
    assert response.status_code == 200
    assert not state['calls'] and not state['charges']
    assert state['releases'][0][0][1] == sum(state['reserves'])


@pytest.mark.parametrize('path', ['/agent/process', '/agent/stream'])
def test_provider_errors_and_body_logs_do_not_expose_source_or_task(runtime, monkeypatch, caplog, path):
    client, state = runtime
    monkeypatch.setattr(settings, 'log_request_bodies', True)
    state['error'] = RuntimeError(SOURCE + ' private-provider-detail')
    with caplog.at_level(logging.INFO):
        response = client.post(path, json=body(query='SENSITIVE_TASK'))
    assert CONTEXT_ERROR in response.text
    for private in [SOURCE, 'Velvet Mango', 'private-provider-detail', 'SENSITIVE_TASK']:
        assert private not in caplog.text
        assert private not in response.text


def test_successful_body_logging_excludes_generated_quotes(runtime, monkeypatch, caplog):
    client, _ = runtime
    monkeypatch.setattr(settings, 'log_request_bodies', True)
    with caplog.at_level(logging.INFO):
        response = client.post('/agent/process', json=body())
    assert response.status_code == 200
    assert 'Velvet Mango' not in caplog.text


def test_output_limit_warning_is_returned_and_preserves_partial_context_warning(runtime):
    client, state = runtime
    state['output_tokens'] = 5120
    request = body(document_context={'text': 'word ' * 8000}, document_action='draft')
    result = client.post('/agent/process', json=request).json()
    assert [w['kind'] for w in result['warnings']] == ['partial_document_context', 'output_limit_reached']
    events = parse_sse_events(client.post('/agent/stream', json=request).text)
    assert events[-1]['data']['warnings'] == result['warnings']


def test_paid_provider_failure_releases_reserved_credit_and_records_failure_once(runtime, monkeypatch):
    client, _ = runtime
    state = paid(runtime, monkeypatch)
    state['error'] = RuntimeError('private model error')
    events = parse_sse_events(client.post('/agent/stream', json=body()).text)
    assert events[-1]['data']['type'] == 'error'
    assert len(state['releases']) == len(state['charges']) == 1
    args, kwargs = state['charges'][0]
    assert kwargs['success'] is False
    assert args[1] == events[1]['data']['billable_input_chars']
    assert state['releases'][0][0][1] == sum(state['reserves'])


def test_abort_after_content_bills_only_emitted_output_and_cannot_restart(runtime, monkeypatch):
    client, _ = runtime
    state = paid(runtime, monkeypatch)
    cancel = threading.Event()
    monkeypatch.setattr(api, 'register_cancellation_event', lambda rid: cancel)
    def run(agent, prompt, **kwargs):
        state['calls'].append(prompt)
        yield SimpleNamespace(event=RunEvent.run_content, content='First.', metrics=None)
        cancel.set()
        yield SimpleNamespace(event=RunEvent.run_content, content='Never emit this.', metrics=None)
    monkeypatch.setattr(tpa.Agent, 'run', run)
    events = parse_sse_events(client.post('/agent/stream', json=body()).text)
    assert events[-1]['data']['aborted'] is True
    assert ''.join(e['data']['content'] for e in events if e['data'].get('type') == 'content') == 'First.'
    assert state['charges'][0]['output_chars'] == len('First.')
    assert len(state['calls']) == 1


def test_provider_error_after_visible_content_does_not_fall_back(runtime, monkeypatch):
    from src.task_router import KeyResolver
    client, state = runtime
    monkeypatch.setattr(settings, 'provider_fallback_enabled', True)
    resolver = KeyResolver({}, {'openai_api_key': 'o', 'anthropic_api_key': 'a'},
        lambda **kw: kw['default_keys'].get('anthropic_api_key' if kw['model_provider'] == 'claude' else kw['model_provider'] + '_api_key'))
    def run(agent, prompt, **kwargs):
        state['calls'].append(prompt)
        yield SimpleNamespace(event=RunEvent.run_content, content='Partial.', metrics=None)
        raise RuntimeError('503 unavailable')
    monkeypatch.setattr(tpa.Agent, 'run', run)
    prepared = prepare_context(DocumentContext(text=SOURCE), body()['query'])
    generator = tpa.process_user_query_stream(body()['query'], api_key='k', key_resolver=resolver, prepared_context=prepared)
    assert next(generator)['__event__']['event'] == 'document_context_ready'
    assert next(generator) == 'Partial.'
    with pytest.raises(RuntimeError):
        next(generator)
    assert len(state['calls']) == 1


def test_concurrent_api_requests_keep_source_and_actual_model_inputs_isolated(runtime, monkeypatch):
    client, state = runtime
    def run(agent, prompt, **kwargs):
        state['calls'].append(prompt)
        data = json.JSONDecoder().raw_decode(prompt)[0]
        text = ''.join(e['text'] for e in data['excerpts'])
        yield SimpleNamespace(event=RunEvent.run_content, content=text, metrics=None)
    monkeypatch.setattr(tpa.Agent, 'run', run)
    sources = [f'Private document {i} 👩‍💻' for i in range(16)]
    def request(source):
        response = client.post('/agent/process', json=body(document_context={'text': source}))
        assert response.status_code == 200
        return response.json()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(request, sources))
    for source, result in zip(sources, results):
        assert result['markdown_response'] == source
        assert result['document_context']['source_sha256'] == fingerprint(source)
    assert len(state['calls']) == len(sources)


def test_new_edit_uses_the_existing_patch_service_and_preserves_stale_checks(monkeypatch):
    from tests.document_edit.test_api import runtime as edit_runtime, body as edit_body
    client, state = edit_runtime.__wrapped__(monkeypatch)
    monkeypatch.setattr(settings, 'document_context_enabled', True)
    request = edit_body()
    source = request.pop('document_text')
    request.update(document_context={'text': source}, document_action='edit')
    response = client.post('/agent/process', json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['document_edit']['status'] == 'proposed'
    assert result['document_context'] is None
    assert len(state['calls']) == 3
    events = parse_sse_events(client.post('/agent/stream', json=request).text)
    assert [e['event'] or e['data'].get('type') for e in events] == ['start', 'document_edit_ready', 'content', 'done']
    stale = client.post('/agent/documents/apply', json={'document_text': source + 'Changed.',
        'base_sha256': result['document_edit']['base_sha256'], 'patches': result['document_edit']['patches']})
    assert stale.status_code == 409


@pytest.mark.parametrize('byok', [False, True])
def test_document_fallback_respects_eligible_keys_and_preserves_evidence(runtime, monkeypatch, byok):
    from src.task_router import KeyResolver
    _, state = runtime
    monkeypatch.setattr(settings, 'provider_fallback_enabled', True)
    keys = {'anthropic_api_key': 'own-a'} if byok else {'openai_api_key': 'platform-o', 'anthropic_api_key': 'platform-a'}
    resolver = KeyResolver(keys if byok else {}, None if byok else keys,
        lambda **kw: (kw['user_keys'] if byok else kw['default_keys']).get('anthropic_api_key' if kw['model_provider'] == 'claude' else kw['model_provider'] + '_api_key'))
    original = tpa.create_agent
    attempted = []
    def create(kind, provider, key, *args, **kwargs):
        attempted.append((provider, key))
        if provider == 'openai':
            raise RuntimeError('401 invalid api key')
        return original(kind, provider, key, *args, **kwargs)
    monkeypatch.setattr(tpa, 'create_agent', create)
    prepared = prepare_context(DocumentContext(text=SOURCE), body()['query'])
    chunks = list(tpa.process_user_query_stream(body()['query'], api_key='unused', key_resolver=resolver,
        prepared_context=prepared, is_byok=byok, report_model=True))
    assert ''.join(c for c in chunks if isinstance(c, str)) == 'Velvet Mango.'
    assert state['calls'][0]['prompt'] == prepared.prompt
    assert attempted == ([('claude', 'own-a')] if byok else [('openai', 'platform-o'), ('claude', 'platform-a')])
