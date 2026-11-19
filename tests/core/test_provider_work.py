import asyncio
import threading
import time
from types import SimpleNamespace

import anyio
import httpx
import pytest
from fastapi import HTTPException

import api
import src.provider_work as work
import src.text_processing_agent as tpa
from src.longform.ordered_stream import run_ordered


async def test_licence_check_completes_while_provider_pool_is_busy(monkeypatch):
    workers = work._Workers(1, 'test-provider')
    monkeypatch.setattr(work, '_request_workers', workers)
    checked = []

    def validate(key):
        checked.append(key)
        return {'valid': True, 'user_id': 'u', 'license': {'requires_user_api_keys': True}}

    monkeypatch.setattr(api, 'redis_manager', SimpleNamespace(
        get_cached_user_id=lambda key: None, validate_license=validate, get_user_api_keys=lambda user: {},
        resolve_api_key_from_user_keys=lambda **kwargs: 'offline-provider', update_heartbeat=lambda key: None))
    monkeypatch.setattr(api.settings, 'task_routing_enabled', False)
    monkeypatch.setattr(api.settings, 'provider_fallback_enabled', False)
    loop = asyncio.get_running_loop()
    started, release = asyncio.Event(), threading.Event()
    captured = {}

    def slow_stream(*args, **kwargs):
        captured.update(kwargs)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(4)
        yield 'late content'

    monkeypatch.setattr(api, 'process_user_query_stream', slow_stream)
    limiter = anyio.to_thread.current_default_thread_limiter()
    before = limiter.total_tokens
    limiter.total_tokens = 1
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://offline.invalid') as client:
            stream = asyncio.create_task(client.post('/agent/stream', json={
                'type': 'fix_grammar', 'query': 'text', 'model': 'openai', 'api_key': 'stream-licence'}))
            try:
                await asyncio.wait_for(started.wait(), 2)
                abort = await asyncio.wait_for(client.post('/agent/abort', json={
                    'request_id': captured['request_id'], 'api_key': 'abort-licence'}), 1)
                assert abort.status_code == 200 and not stream.done()
                assert checked == ['stream-licence', 'abort-licence'] and captured['cancel_event'].is_set()
            finally:
                release.set()
                response = await stream
        assert response.status_code == 200 and 'late content' not in response.text
    finally:
        limiter.total_tokens = before
        workers.pool.shutdown()


async def test_request_deadline_skips_queued_work(monkeypatch):
    workers = work._Workers(1, 'test-provider')
    monkeypatch.setattr(work, '_request_workers', workers)
    release = threading.Event()
    active = workers.submit(release.wait, 3)
    cancel = threading.Event()
    calls = []
    try:
        with pytest.raises(HTTPException) as error:
            await work.run_provider_work(calls.append, 'started', stop_event=cancel, deadline=time.monotonic() + 0.05)
        assert error.value.status_code == 504 and cancel.is_set()
        with pytest.raises(HTTPException) as busy:
            await work.run_provider_work(calls.append, 'over capacity')
        assert busy.value.status_code == 503
    finally:
        release.set()
        active.result(timeout=2)
        workers.pool.shutdown()
    assert not calls


def test_cancellation_skips_provider_retry(monkeypatch):
    cancel = threading.Event()
    runs = []

    def run(*args, **kwargs):
        runs.append('attempt')
        cancel.set()
        raise RuntimeError('429 rate limit')

    monkeypatch.setattr(tpa, 'create_agent', lambda *args, **kwargs: SimpleNamespace(run=run))
    assert list(tpa.process_user_query_stream('clean this', 'openai', 'offline-key', force_agent_type='grammar',
                                              cancel_event=cancel)) == []
    assert runs == ['attempt']


def test_cancelled_ordered_pipeline_schedules_no_more_units():
    cancel = threading.Event()
    started = []
    units = run_ordered([lambda i=i: started.append(i) or i for i in range(5)], parallel=1, cancel_event=cancel)
    assert next(units) == 0
    cancel.set()
    with pytest.raises(work.ProviderCancelled):
        next(units)
    assert started == [0]


def test_closing_a_pipeline_stops_followup_work_in_started_units():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    later_calls = []

    def unit():
        started.set()
        try:
            assert release.wait(2)
            work.check_provider_call()
            later_calls.append('follow-up')
        finally:
            finished.set()

    pipeline = run_ordered([lambda: 'first', unit], parallel=2)
    try:
        assert next(pipeline) == 'first' and started.wait(2)
    finally:
        pipeline.close()
        release.set()
    assert finished.wait(2) and not later_calls


@pytest.mark.parametrize('provider', ['openai', 'claude', 'grok', 'deepseek', 'gemini'])
def test_sdk_receives_its_transport_deadline(provider):
    model = tpa.get_model(provider, 'offline-key', timeout=7, cache_model=False, managed_retries=True)
    client = model.get_client()
    if provider == 'gemini':
        options = client._api_client._http_options
        assert options.timeout == 7000 and options.retry_options.attempts == 1
    else:
        assert client.timeout == 7 and client.max_retries == 0
