import multiprocessing
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fakeredis
import pytest
import redis
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

import api
from src import text_processing_agent as tpa
from src.redis_manager import RedisManager


STREAM = {'type': 'fix_grammar', 'query': 'some text', 'model': 'openai', 'api_key': 'owner-licence'}


def configure_worker(monkeypatch, redis_url):
    monkeypatch.setattr(api, 'get_redis_manager', lambda: RedisManager(redis_url))
    monkeypatch.setattr(RedisManager, 'resolve_api_key_from_user_keys', lambda self, **kw: 'offline-provider')
    monkeypatch.setattr(api, 'start_billing_delivery', lambda client: None)
    monkeypatch.setattr(api, 'stop_billing_delivery', lambda: None)
    monkeypatch.setattr(tpa, 'prewarm_models', lambda keys: 0)
    for feature in ('voice_flow_enabled', 'model_preflight_on_start', 'task_routing_enabled', 'provider_fallback_enabled'):
        monkeypatch.setattr(api.settings, feature, False)


def stream_in_other_worker(redis_url, control):
    with control, pytest.MonkeyPatch.context() as patch:
        configure_worker(patch, redis_url)
        register = api.register_cancellation_event
        cancel = api.cancel_request_by_id
        captured, calls = {}, []

        def capture_registration(request_id, user_id):
            event = register(request_id, user_id)
            captured.update(request_id=request_id, cancel=event)
            return event

        def observe_cancellation(request_id, user_id):
            matched = cancel(request_id, user_id)
            if user_id == 'other':
                control.send(not matched and not captured['cancel'].is_set())
            return matched

        def provider_attempt(*args, **kwargs):
            calls.append('attempt')
            assert len(calls) == 1
            control.send((os.getpid(), captured['request_id']))
            control.send(('owner-cancelled', captured['cancel'].wait(5)))
            raise RuntimeError('429 rate limit')

        patch.setattr(api, 'register_cancellation_event', capture_registration)
        patch.setattr(api, 'cancel_request_by_id', observe_cancellation)
        patch.setattr(tpa, 'create_agent', lambda *args, **kwargs: SimpleNamespace(run=provider_attempt))
        with TestClient(api.app) as client:
            response = client.post('/agent/stream', json=STREAM)
        control.send((response.status_code, calls, response.text))


def test_abort_through_different_worker_stops_provider_retry(monkeypatch):
    try:
        server = fakeredis.TcpFakeServer(('127.0.0.1', 0))
    except (PermissionError, OSError):
        pytest.skip('Local sockets are unavailable')
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    redis_url = f'redis://127.0.0.1:{server.server_address[1]}/0'
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=stream_in_other_worker, args=(redis_url, child))
    try:
        try:
            with redis.Redis.from_url(redis_url) as store:
                for user in ('owner', 'other'):
                    store.hset(f'license_key:{user}-licence', mapping={
                        'isValid': 'true', 'status': 'active', 'requires_user_api_keys': 'true', 'user_id': user,
                    })
        except (redis.exceptions.ConnectionError, PermissionError, OSError):
            pytest.skip('Local sockets are unavailable')
        configure_worker(monkeypatch, redis_url)
        with TestClient(api.app) as client:
            worker.start()
            child.close()
            assert parent.poll(12), 'The stream worker never started provider work'
            stream_pid, request_id = parent.recv()
            assert stream_pid != os.getpid()
            for user in ('other', 'owner'):
                result = client.post('/agent/abort', json={'request_id': request_id, 'api_key': f'{user}-licence'})
                assert result.status_code == 200 and result.json() == {'success': True}
                if user == 'other':
                    assert parent.poll(3) and parent.recv() is True
            assert parent.poll(8), 'The stream worker did not observe the abort'
            assert parent.recv() == ('owner-cancelled', True)
            assert parent.poll(3), 'The stream worker did not finish'
            status, calls, body = parent.recv()
            assert status == 200 and calls == ['attempt'] and '"type": "error"' not in body
        worker.join(5)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(3)
        parent.close()
        child.close()
        server.shutdown()
        server.server_close()
        serving.join(3)


async def test_stream_can_be_aborted_at_response_headers(monkeypatch):
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('offline-provider', {
        'user_id': 'owner', 'requires_user_api_keys': True, 'user_keys': {},
    })))
    monkeypatch.setattr(api.settings, 'task_routing_enabled', False)
    monkeypatch.setattr(api.settings, 'provider_fallback_enabled', False)
    calls = []

    def provider(*args, **kwargs):
        calls.append('attempt')
        yield 'late content'

    monkeypatch.setattr(api, 'process_user_query_stream', provider)
    response = await api.stream_text(api.StreamRequest(**STREAM), BackgroundTasks())
    request_id = response.headers['X-Request-ID']

    async def send(message):
        if message['type'] == 'http.response.start':
            assert api.cancel_request_by_id(request_id, 'owner')

    await response.stream_response(send)
    assert not calls
    assert not api.cancel_request_by_id(request_id, 'owner')


@pytest.mark.parametrize('publish', [Mock(return_value=0), Mock(side_effect=redis.ConnectionError('offline'))])
def test_remote_abort_reports_publish_failure(monkeypatch, publish):
    monkeypatch.setattr(api, 'redis_manager', SimpleNamespace(
        validate_license=lambda licence: {'valid': True, 'user_id': 'owner'},
        redis_client=SimpleNamespace(publish=publish),
    ))
    result = TestClient(api.app).post('/agent/abort', json={'request_id': 'remote-stream', 'api_key': 'owner-licence'})
    assert result.status_code == 503
    publish.assert_called_once_with(api.ABORT_CHANNEL, '["remote-stream", "owner"]')
