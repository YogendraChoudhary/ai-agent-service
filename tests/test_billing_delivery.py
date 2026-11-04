"""Run the real billing Lua scripts against an isolated Redis emulator, never the account store."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import fakeredis
import pytest
import requests

import src.billing_delivery as delivery
import src.webhook as webhook
from src.credit_manager import CreditManager


@pytest.fixture
def world(monkeypatch):
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.hset('subscription:u1', mapping={'remaining_credits': 10, 'purchased_credits': 0})
    now = [1000.0]
    monkeypatch.setattr(delivery, 'time', SimpleNamespace(time=lambda: now[0]))
    box = delivery.BillingDelivery(redis, 'https://billing.invalid', 'test-secret')
    credits = CreditManager(redis)
    monkeypatch.setattr(webhook, '_billing_delivery', box)
    return SimpleNamespace(redis=redis, box=box, credits=credits, now=now)


def bill(world, *, event_id='event-1', reserved=10, delay=0, **changes):
    payload = webhook._payload('u1', 100, 200, 'text', 0, 'openai', 'test-model', 20, True, None, None)
    payload.update(changes)
    return world.box.enqueue(payload, event_id=event_id, reserved_scaled=reserved, hold_scaled=reserved, settle_delay=delay)


def job(world, event_id='event-1'):
    return json.loads(world.redis.hget(delivery.JOBS, event_id))


def test_enqueue_atomically_transfers_active_credit_to_a_nonexpiring_hold(world):
    assert world.credits.reserve('u1', 90) == (True, 10)
    bill(world, reserved=90)
    assert world.redis.get('credits_reserved:u1') is None
    assert world.redis.get(delivery.pending_credits_key('u1')) == '90'
    assert world.redis.ttl(delivery.pending_credits_key('u1')) == -1
    assert world.credits.peek_available('u1') == 10
    assert world.credits.reserve('u1', 20) == (False, 10)
    assert CreditManager(world.redis).peek_available('u1') == 10


def test_enqueue_and_completed_event_deduplication_never_transfer_another_requests_hold(world, monkeypatch):
    world.credits.reserve('u1', 20)
    bill(world)
    bill(world)
    assert world.redis.get('credits_reserved:u1') == '10'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    monkeypatch.setattr(delivery.requests, 'post', lambda *a, **kw: SimpleNamespace(status_code=200))
    assert world.box.process_once() and world.box.process_once()
    bill(world)
    assert world.redis.get('credits_reserved:u1') == '10'
    assert world.redis.get(delivery.pending_credits_key('u1')) is None
    with pytest.raises(ValueError, match='different usage'):
        bill(world, output_chars=300)


@pytest.mark.parametrize('failure', [500, 429, 401, 302, requests.Timeout('unknown receipt')])
def test_unacknowledged_delivery_keeps_credit_held_and_never_blindly_reposts(world, monkeypatch, failure):
    world.credits.reserve('u1', 100)
    bill(world, reserved=100)
    calls = []
    def post(*a, **kw):
        calls.append(kw)
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(status_code=failure)
    monkeypatch.setattr(delivery.requests, 'post', post)
    assert world.box.process_once()
    world.now[0] += 1000
    assert world.box.process_once()
    assert len(calls) == 1 and calls[0]['allow_redirects'] is False
    assert calls[0]['headers']['Idempotency-Key'] == 'event-1'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '100'
    assert world.credits.reserve('u1', 10) == (False, 0)
    assert job(world)['state'] == 'queued'


def test_confirmed_idempotent_receiver_retries_with_same_key_and_releases_only_after_ack_delay(world, monkeypatch):
    world.box.allow_retries = True
    world.credits.reserve('u1', 10)
    bill(world, delay=3)
    calls = []
    def post(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(status_code=500 if len(calls) == 1 else 200)
    monkeypatch.setattr(delivery.requests, 'post', post)
    world.box.process_once()
    assert world.box.process_once() is False
    world.now[0] += 2
    world.box.process_once()
    assert calls[0]['json'] == calls[1]['json']
    assert calls[0]['headers']['Idempotency-Key'] == calls[1]['headers']['Idempotency-Key']
    assert job(world)['state'] == 'acknowledged'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    world.now[0] += 3
    world.box.process_once()
    assert world.redis.get(delivery.pending_credits_key('u1')) is None
    assert world.redis.hlen(delivery.JOBS) == 0 and world.redis.zcard(delivery.DUE) == 0


def test_post_commit_timeout_can_be_replayed_without_a_second_receiver_charge(world, monkeypatch):
    world.box.allow_retries = True
    bill(world)
    charged, attempts = set(), []
    def receiver(*a, **kw):
        key = kw['headers']['Idempotency-Key']
        charged.add(key)
        attempts.append(key)
        if len(attempts) == 1:
            raise requests.Timeout('receiver committed but response was lost')
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(delivery.requests, 'post', receiver)
    world.box.process_once()
    world.now[0] += 2
    world.box.process_once()
    world.box.process_once()
    assert attempts == ['event-1', 'event-1'] and charged == {'event-1'}
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_new_worker_recovers_queued_and_acknowledged_work(world, monkeypatch):
    world.credits.reserve('u1', 10)
    bill(world, delay=3)
    second = delivery.BillingDelivery(world.redis, 'https://billing.invalid', 'test-secret')
    calls = []
    monkeypatch.setattr(delivery.requests, 'post', lambda *a, **kw: calls.append(kw) or SimpleNamespace(status_code=200))
    second.process_once()
    world.now[0] += 3
    third = delivery.BillingDelivery(world.redis, 'https://billing.invalid', 'test-secret')
    third.process_once()
    assert len(calls) == 1
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_only_one_worker_can_send_a_claimed_event(world, monkeypatch):
    bill(world)
    second = delivery.BillingDelivery(world.redis, 'https://billing.invalid', 'test-secret')
    entered, finish = threading.Event(), threading.Event()
    calls = []
    def post(*a, **kw):
        calls.append(kw)
        entered.set()
        assert finish.wait(3)
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(delivery.requests, 'post', post)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(world.box.process_once)
        assert entered.wait(3)
        assert second.process_once() is False
        finish.set()
        assert active.result()
    assert len(calls) == 1
    assert second.process_once()
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_expired_lease_cannot_let_an_old_worker_overwrite_new_acknowledgement(world, monkeypatch):
    world.box.allow_retries = True
    bill(world)
    second = delivery.BillingDelivery(world.redis, 'https://billing.invalid', 'test-secret', allow_retries=True)
    entered, finish = threading.Event(), threading.Event()
    def post(*a, **kw):
        if not entered.is_set():
            entered.set()
            assert finish.wait(3)
            return SimpleNamespace(status_code=500)
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(delivery.requests, 'post', post)
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(world.box.process_once)
        assert entered.wait(3)
        world.now[0] += delivery.LEASE_SECONDS + 1
        assert second.process_once()
        finish.set()
        active.result()
    assert job(world)['state'] == 'acknowledged'
    second.process_once()
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_missing_configuration_retains_the_pending_charge_without_a_network_call(world, monkeypatch):
    world.box.url = ''
    bill(world)
    monkeypatch.setattr(delivery.requests, 'post', lambda *a, **kw: pytest.fail('No configured billing receiver'))
    world.box.process_once()
    assert job(world)['attempts'] == 0
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'


def test_background_worker_drains_acknowledged_messages_and_stops_cleanly(world, monkeypatch):
    bill(world)
    released = threading.Event()
    complete = world.box._complete
    def observed_complete(**kwargs):
        result = complete(**kwargs)
        released.set()
        return result
    monkeypatch.setattr(world.box, '_complete', observed_complete)
    monkeypatch.setattr(delivery.requests, 'post', lambda *a, **kw: SimpleNamespace(status_code=200))
    world.box.start()
    try:
        assert released.wait(3)
    finally:
        world.box.stop()
    assert not world.box._thread.is_alive()
    assert world.redis.hlen(delivery.JOBS) == 0
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_both_webhook_entry_points_persist_usage_before_returning_and_failure_telemetry_has_no_hold(world):
    world.credits.reserve('u1', 10)
    webhook.fire_credit_deduction_and_release('u1', 100, 200, 'text', 0, 10, world.credits, 3, billing_event_id='reserved')
    webhook.fire_credit_deduction('u1', 100, 200, billing_event_id='unreserved')
    webhook.fire_credit_deduction('u1', 0, 0, success=False, billing_event_id='failure')
    assert world.redis.hlen(delivery.JOBS) == 3
    assert world.redis.get(delivery.pending_credits_key('u1')) == '20'
    assert world.redis.get('credits_reserved:u1') is None
    assert job(world, 'failure')['hold_scaled'] == 0


def test_enqueue_failure_does_not_release_an_active_credit_reservation(world, monkeypatch):
    world.credits.reserve('u1', 10)
    monkeypatch.setattr(world.box, '_enqueue', lambda **kw: (_ for _ in ()).throw(ConnectionError('Redis down')))
    with pytest.raises(ConnectionError):
        webhook.fire_credit_deduction_and_release('u1', 100, 200, 'text', 0, 10, world.credits, 0)
    assert world.redis.get('credits_reserved:u1') == '10'


@pytest.mark.parametrize('options, expected_hold', [
    ({'input_chars': 1100, 'output_chars': 1100}, 30),
    ({'operation_type': 'autocomplete'}, 2),
    ({'operation_type': 'image', 'image_count': 2}, 100),
    ({'operation_type': 'mixed', 'image_count': 1}, 60),
    ({'operation_type': 'image', 'images': [{'credits_scaled': 80}]}, 80),
    ({'design': {'credits_scaled': 30}}, 30),
    ({'design': {'credits_scaled': 30}, 'image_count': 1}, 80),
])
def test_unreserved_billing_holds_cover_each_existing_price_type(world, options, expected_hold):
    request = dict(user_id='u1', input_chars=100, output_chars=100)
    request.update(options)
    webhook.fire_credit_deduction(**request)
    assert world.redis.get(delivery.pending_credits_key('u1')) == str(expected_hold)


def test_real_api_keeps_failed_billing_held_and_refuses_further_paid_work(monkeypatch):
    from unittest.mock import AsyncMock
    from fastapi.testclient import TestClient
    import api
    import src.text_processing_agent as tpa
    from config import settings
    from tests.test_document_context_api import runtime, body
    _, state = runtime.__wrapped__(monkeypatch)
    store = fakeredis.FakeRedis(decode_responses=True)
    store.hset('subscription:u1', mapping={'remaining_credits': 1, 'purchased_credits': 0})
    monkeypatch.setattr(api, 'get_redis_manager', lambda: SimpleNamespace(redis_client=store, close=lambda: None))
    monkeypatch.setattr(tpa, 'prewarm_models', lambda *args: 0)
    monkeypatch.setattr(settings, 'voice_flow_enabled', False)
    monkeypatch.setattr(api, 'CREDIT_PRECHECK_ENABLED', True)
    monkeypatch.setattr(api, 'CREDIT_SETTLE_DELAY_SECONDS', 0)
    monkeypatch.setattr(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('synthetic-key', {
        'user_id': 'u1', 'requires_user_api_keys': False, 'user_keys': {}})))
    monkeypatch.setattr(api, 'fire_credit_deduction_and_release', webhook.fire_credit_deduction_and_release)
    monkeypatch.setattr(webhook, 'BACKEND_WEBHOOK_URL', 'https://billing.invalid')
    monkeypatch.setattr(webhook, 'WEBHOOK_SECRET', 'test-secret')
    monkeypatch.setattr(webhook, '_billing_delivery', None)
    monkeypatch.setattr(delivery.BillingDelivery, 'start', lambda self: None)
    posted = []
    def receiver(*args, **kwargs):
        posted.append(kwargs)
        if len(posted) == 1:
            return SimpleNamespace(status_code=500)
        store.hset('subscription:u1', 'remaining_credits', 0)
        return SimpleNamespace(status_code=200)
    monkeypatch.setattr(delivery.requests, 'post', receiver)
    with TestClient(api.app) as client:
        response = client.post('/agent/stream', json=body())
        assert response.status_code == 200 and 'Velvet Mango.' in response.text
        box = webhook._billing_delivery
        assert box.process_once()
        assert store.get(delivery.pending_credits_key('u1')) == '10'
        blocked = client.post('/agent/stream', json=body())
        assert blocked.status_code == 402 and len(state['calls']) == 1
        box.allow_retries = True
        event_id, = store.hkeys(delivery.JOBS)
        store.zadd(delivery.DUE, {event_id: 0})
        assert box.process_once() and box.process_once()
        assert len(posted) == 2
        assert posted[0]['headers']['Idempotency-Key'] == posted[1]['headers']['Idempotency-Key']
        assert store.get(delivery.pending_credits_key('u1')) is None
        assert client.post('/agent/stream', json=body()).status_code == 402
    assert webhook._billing_delivery is None
