"""Exercise credit reservations and settlement through the real stream route."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fakeredis
import httpx
import pytest
from anyio import CancelScope
from fastapi import BackgroundTasks

import api
from src import billing_delivery, provider_work, webhook
from src.credit_manager import CreditManager
from tests.design.test_design_api import parse_sse_events


REQUEST = {'type': 'fix_grammar', 'query': 'x', 'model': 'openai', 'api_key': 'test'}
PENDING = billing_delivery.pending_credits_key('u1')


@pytest.fixture
def paid_stream(monkeypatch):
    store = fakeredis.FakeRedis(decode_responses=True)
    store.hset('subscription:u1', 'remaining_credits', 100)
    credits = CreditManager(store)
    box = billing_delivery.BillingDelivery(store, 'https://billing.invalid', 'test-secret')
    monkeypatch.setattr(webhook, '_billing_delivery', box)
    monkeypatch.setattr(api, 'CREDIT_PRECHECK_ENABLED', True)
    monkeypatch.setattr(api, 'CREDIT_SETTLE_DELAY_SECONDS', 0)
    monkeypatch.setattr(api.settings, 'task_routing_enabled', False)
    monkeypatch.setattr(api.settings, 'provider_fallback_enabled', False)
    monkeypatch.setattr(api, 'credit_manager', credits)
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('provider-key', {
        'user_id': 'u1', 'requires_user_api_keys': False, 'user_keys': {}})))
    monkeypatch.setattr(api, 'process_user_query_stream', lambda *a, **kw: iter(['text']))
    return SimpleNamespace(store=store, credits=credits, box=box)


@pytest.mark.parametrize('chunks,balance,cap,reservations,delivered', [
    pytest.param([35000], 100, 1000, [10, 350], 35000, id='large-chunk'),
    pytest.param([3999, 1], 100, 40, [10, 30], 3999, id='exact-cap'),
    pytest.param([999, 3000], 2, 1000, [10, 30], 999, id='insufficient-for-whole-chunk'),
])
async def test_stream_batches_reservations_and_bills_only_sent_text(paid_stream, monkeypatch, chunks, balance, cap, reservations, delivered):
    paid_stream.store.hset('subscription:u1', 'remaining_credits', balance)
    monkeypatch.setattr(api, 'MAX_RESERVATION_PER_REQUEST_SCALED', cap)
    monkeypatch.setattr(api, 'process_user_query_stream', lambda *a, **kw: ('a' * size for size in chunks))
    reserve = Mock(wraps=paid_stream.credits.reserve)
    monkeypatch.setattr(paid_stream.credits, 'reserve', reserve)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://offline.invalid') as client:
        response = await client.post('/agent/stream', json=REQUEST)

    assert response.status_code == 200
    events = [event['data'] for event in parse_sse_events(response.text)]
    assert sum(len(event['content']) for event in events if event['type'] == 'content') == delivered
    limits = [event['output_chars'] for event in events if event['type'] == 'credit_limit_reached']
    assert limits == ([delivered] if delivered < sum(chunks) else [])
    assert [args[1] for args, _ in reserve.call_args_list] == reservations

    held = api.text_credits_scaled(1, delivered)
    assert paid_stream.store.get(PENDING) == str(held)
    assert paid_stream.credits.peek_available('u1') == balance * api.CREDIT_SCALE - held
    assert paid_stream.store.get('credits_reserved:u1') is None
    event_id, = paid_stream.store.hkeys(billing_delivery.JOBS)
    post = Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(billing_delivery.requests, 'post', post)
    assert paid_stream.box.process_once()
    assert paid_stream.box.process_once()
    assert paid_stream.box.process_once() is False
    assert post.call_count == 1
    assert post.call_args.kwargs['headers']['Idempotency-Key'] == event_id
    assert post.call_args.kwargs['json']['output_chars'] == delivered
    assert paid_stream.store.get(PENDING) is None


@pytest.mark.parametrize('operation', ['reserve', 'enqueue'])
async def test_slow_credit_io_does_not_block_health_requests(paid_stream, monkeypatch, operation):
    loop = asyncio.get_running_loop()
    started, release = asyncio.Event(), threading.Event()
    target = paid_stream.credits if operation == 'reserve' else paid_stream.box
    original = getattr(target, operation)
    loop_thread = threading.get_ident()

    def slow_io(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, operation, slow_io)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://offline.invalid') as client:
        request = asyncio.create_task(client.post('/agent/stream', json=REQUEST))
        try:
            await asyncio.wait_for(started.wait(), 2)
            health = await client.get('/health')
            assert health.status_code == 200 and not request.done()
        finally:
            release.set()
            response = await request
    assert response.status_code == 200
    assert paid_stream.store.hlen(billing_delivery.JOBS) == 1


async def test_cancelled_stream_persists_its_billing_handoff_once(paid_stream):
    response = await api.stream_text(api.StreamRequest(**REQUEST), BackgroundTasks())
    iterator = response.body_iterator
    assert '"type": "start"' in await anext(iterator)
    assert '"content": "text"' in await anext(iterator)
    with CancelScope() as scope:
        scope.cancel()
        await iterator.aclose()

    job, = paid_stream.store.hvals(billing_delivery.JOBS)
    assert json.loads(job)['payload']['output_chars'] == 4
    assert paid_stream.store.get(PENDING) == '10'
    assert paid_stream.store.get('credits_reserved:u1') is None


async def test_aborting_queued_stream_releases_credit_hold_without_billing(paid_stream, monkeypatch):
    loop = asyncio.get_running_loop()
    queued, hold_released = asyncio.Event(), asyncio.Event()
    unblock = threading.Event()
    workers = provider_work._Workers(1, 'test-provider')
    active = workers.submit(unblock.wait, 4)
    submit = workers.submit
    release = paid_stream.credits.release
    calls = []

    def notify_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        queued.set()
        return future

    def release_hold(*args, **kwargs):
        release(*args, **kwargs)
        loop.call_soon_threadsafe(hold_released.set)

    def provider(*args, **kwargs):
        calls.append('started')
        yield 'late text'

    monkeypatch.setattr(workers, 'submit', notify_submit)
    monkeypatch.setattr(provider_work, '_request_workers', workers)
    monkeypatch.setattr(paid_stream.credits, 'release', release_hold)
    monkeypatch.setattr(api, 'process_user_query_stream', provider)
    response = await api.stream_text(api.StreamRequest(**REQUEST), BackgroundTasks())
    stream = response.body_iterator
    try:
        first = await anext(stream)
        request_id = parse_sse_events(first)[0]['data']['request_id']
        consume = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(queued.wait(), 1)
            assert api.cancel_request_by_id(request_id, 'u1')
        finally:
            unblock.set()
            await consume
        await stream.aclose()
        await asyncio.wait_for(hold_released.wait(), 1)
        assert not calls
        assert paid_stream.store.hlen(billing_delivery.JOBS) == 0
        assert paid_stream.store.get('credits_reserved:u1') is None
    finally:
        unblock.set()
        active.result(timeout=1)
        workers.pool.shutdown()
