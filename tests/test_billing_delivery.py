"""Run the real billing Lua scripts against an isolated Redis emulator, never the account store."""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, call

import fakeredis
import pytest
import requests
from redis.exceptions import ConnectionError as RedisConnectionError

import src.billing_delivery as delivery
import src.webhook as webhook
from src.credit_manager import CreditManager


@pytest.fixture
def world(monkeypatch):
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.hset('subscription:u1', mapping={'remaining_credits': 10, 'purchased_credits': 0})
    now = [1000.0]

    def clock():
        return now[0]

    monkeypatch.setattr(delivery, 'time', SimpleNamespace(time=clock, monotonic=clock))
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
    assert world.box.process_once() is False
    assert len(calls) == 1 and calls[0]['allow_redirects'] is False
    assert calls[0]['headers']['Idempotency-Key'] == 'event-1'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '100'
    assert world.credits.reserve('u1', 10) == (False, 0)
    assert job(world)['state'] == 'reconciliation'
    assert world.redis.zcard(delivery.DUE) == world.redis.hlen(delivery.LEASES) == 0
    world.box.allow_retries = True
    with pytest.raises(ValueError, match='no active receiver deduplication window'):
        world.box.requeue('event-1')


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


def test_never_attempted_charge_can_be_requeued_once_after_configuration_recovers(world, monkeypatch):
    world.box.url = ''
    bill(world, delay=3)
    post = Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(delivery.requests, 'post', post)
    assert world.box.process_once()
    assert job(world)['attempts'] == 0
    assert job(world)['state'] == 'reconciliation'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    assert world.box.process_once() is False and post.call_count == 0
    with pytest.raises(ValueError, match='configuration'):
        world.box.requeue('event-1')
    world.box.url = 'https://billing.invalid'
    world.box._wake.clear()
    second = delivery.BillingDelivery(world.redis, world.box.url, world.box.secret)
    barrier = threading.Barrier(2)
    read = world.redis.hget

    def read_before_either_write(*args):
        raw = read(*args)
        barrier.wait(timeout=3)
        return raw

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
        patch.setattr(world.redis, 'hget', read_before_either_write)
        results = list(pool.map(lambda box: box.requeue('event-1'), [world.box, second]))
    assert sorted(results) == [False, True]
    assert world.box._wake.is_set() or second._wake.is_set()
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    assert world.box.process_once() and world.box.process_once() is False
    assert post.call_count == 1 and post.call_args.kwargs['headers']['Idempotency-Key'] == 'event-1'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    world.now[0] += 3
    assert second.process_once() and world.box.requeue('event-1') is False
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


def test_manual_replay_uses_the_original_payload_and_key_with_receiver_deduplication(world, monkeypatch):
    world.box.allow_retries = True
    bill(world, images=[])
    post = Mock(side_effect=[SimpleNamespace(status_code=500), SimpleNamespace(status_code=200)])
    monkeypatch.setattr(delivery.requests, 'post', post)
    assert world.box.process_once()
    world.box.allow_retries = False
    world.now[0] += 2
    assert world.box.process_once() and job(world)['state'] == 'reconciliation'
    world.box.allow_retries = True
    assert world.box.requeue('event-1')
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    assert world.box.process_once() and world.box.process_once()
    assert post.call_count == 2 and post.call_args_list[0] == post.call_args_list[1]
    assert world.redis.get(delivery.pending_credits_key('u1')) is None


@pytest.mark.parametrize('legacy', [False, True])
def test_old_attempts_are_parked_on_the_next_claim_without_unsafe_replay(world, monkeypatch, legacy):
    world.box.allow_retries = True
    bill(world)
    post = Mock(return_value=SimpleNamespace(status_code=500))
    monkeypatch.setattr(delivery.requests, 'post', post)
    if legacy:
        existing = job(world)
        existing['attempts'] = 1
        world.redis.hset(delivery.JOBS, 'event-1', json.dumps(existing))
    else:
        assert world.box.process_once()
        world.now[0] += delivery.RETRY_WINDOW_SECONDS
    assert world.box.process_once() and world.box.process_once() is False
    assert post.call_count == (0 if legacy else 1)
    assert job(world)['state'] == 'reconciliation'
    assert world.redis.get(delivery.pending_credits_key('u1')) == '10'
    with pytest.raises(ValueError, match='no active receiver deduplication window'):
        world.box.requeue('event-1')


def test_idle_worker_claims_once_per_minute_without_an_extra_schedule_lookup(world, monkeypatch):
    assert world.box.process_once() is False  # Load the Lua script before counting steady-state checks.
    execute = Mock(wraps=world.redis.execute_command)
    monkeypatch.setattr(world.redis, 'execute_command', execute)
    waits = []

    def advance(seconds):
        waits.append(seconds)
        world.now[0] += seconds
        if len(waits) == 3:
            world.box._stop.set()

    monkeypatch.setattr(world.box._wake, 'wait', advance)
    world.box.start()
    world.box._thread.join(timeout=3)
    world.box.stop()
    assert waits == [60, 60, 60]
    assert [args[0] for args, _ in execute.call_args_list] == ['EVALSHA'] * 3


def test_background_worker_waits_for_retry_and_settlement_with_credits_held(world, monkeypatch):
    world.box.allow_retries = True
    assert world.credits.reserve('u1', 10)[0]
    bill(world, delay=63.25)
    world.redis.zadd(delivery.DUE, {'stale-event': world.now[0] - 1})
    calls, waits = [], []

    def post(*args, **kwargs):
        calls.append((world.now[0], kwargs['headers']['Idempotency-Key']))
        return SimpleNamespace(status_code=500 if len(calls) == 1 else 200)

    def advance(seconds):
        hold = world.redis.get(delivery.pending_credits_key('u1'))
        waits.append((seconds, hold))
        world.now[0] += seconds
        if hold is None or len(waits) == 5:
            world.box._stop.set()

    monkeypatch.setattr(delivery.requests, 'post', post)
    monkeypatch.setattr(world.box._wake, 'wait', advance)
    world.box.start()
    world.box._thread.join(timeout=3)
    world.box.stop()
    assert waits == [(0, '10'), (2, '10'), (60, '10'), (3.25, '10'), (60, None)]
    assert calls == [(1000, 'event-1'), (1002, 'event-1')]
    assert world.redis.hlen(delivery.JOBS) == world.redis.zcard(delivery.DUE) == 0


@pytest.mark.parametrize('jitter', [0, 2])
def test_background_worker_jitters_backoff_and_logs_redis_state_changes(world, monkeypatch, caplog, jitter):
    caplog.set_level(logging.INFO, logger=delivery.__name__)
    bill(world)
    quota_error = delivery.ResponseError('max requests limit exceeded. Limit: 500000, Usage: 500001')
    connection_error = RedisConnectionError('private connection details')
    claim = world.box._claim
    post = Mock(return_value=SimpleNamespace(status_code=200))
    sample_jitter = Mock(return_value=jitter)
    retries, holds, idle = [], [], Mock()

    def claim_when_available(**kwargs):
        if len(retries) < 3 or idle.called:
            raise quota_error
        if len(retries) < 7:
            raise connection_error
        return claim(**kwargs)

    def wait_before_retry(seconds):
        retries.append(seconds)
        world.now[0] += seconds
        holds.append(world.redis.get(delivery.pending_credits_key('u1')))
        world.box._wake.set()
        if len(retries) == 8:
            world.box._stop.set()

    monkeypatch.setattr(delivery.requests, 'post', post)
    monkeypatch.setattr(delivery, 'random', SimpleNamespace(uniform=sample_jitter))
    monkeypatch.setattr(world.box, '_claim', claim_when_available)
    monkeypatch.setattr(world.box._stop, 'wait', wait_before_retry)
    monkeypatch.setattr(world.box._wake, 'wait', idle)
    world.box.start()
    world.box._thread.join(timeout=3)
    world.box.stop()
    assert retries == [delay + jitter for delay in [2, 4, 8, 16, 32, 58, 58, 2]]
    assert sample_jitter.call_args_list == [call(0, 2)] * 8
    assert holds == ['10'] * 7 + [None]
    assert post.call_count == 1
    assert idle.call_args_list == [call(60)]
    messages = [record.getMessage() for record in caplog.records if record.name == delivery.__name__]
    assert len(messages) == 4
    assert messages[0].startswith('Billing delivery paused (quota_exhausted).')
    assert messages[0].endswith('Upstash max requests limit exceeded. Limit 500000, usage 500001.')
    assert messages[1].startswith('Billing delivery paused (ConnectionError).')
    assert messages[2] == f'Billing delivery recovered after {sum(retries[:7]):.1f}s and 7 failed checks.'
    assert messages[3] == messages[0]
    assert 'private connection details' not in caplog.text


def test_enqueue_wakes_idle_worker_for_delivery_and_settlement(world, monkeypatch):
    parked, released = threading.Event(), threading.Event()
    wake_wait = world.box._wake.wait
    complete = world.box._complete

    def observe_wait(seconds):
        parked.set()
        return wake_wait(seconds)

    def observed_complete(**kwargs):
        result = complete(**kwargs)
        released.set()
        return result

    monkeypatch.setattr(world.box._wake, 'wait', observe_wait)
    monkeypatch.setattr(world.box, '_complete', observed_complete)
    monkeypatch.setattr(delivery.requests, 'post', lambda *a, **kw: SimpleNamespace(status_code=200))
    world.box.start()
    try:
        assert parked.wait(3)
        bill(world)
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


def test_real_api_keeps_uncertain_billing_held_until_confirmed_without_a_second_post(monkeypatch):
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
        raise requests.Timeout('receiver charged the customer but the reply was lost')
    monkeypatch.setattr(delivery.requests, 'post', receiver)
    with TestClient(api.app) as client:
        response = client.post('/agent/stream', json=body())
        assert response.status_code == 200 and 'Velvet Mango.' in response.text
        box = webhook._billing_delivery
        assert box.process_once()
        assert store.get(delivery.pending_credits_key('u1')) == '10'
        blocked = client.post('/agent/stream', json=body())
        assert blocked.status_code == 402 and len(state['calls']) == 1
        event_id, = store.hkeys(delivery.JOBS)
        store.hset('subscription:u1', 'remaining_credits', 0)
        assert box.requeue(event_id, confirm_charged=True)
        assert box.requeue(event_id, confirm_charged=True) is False
        assert store.get(delivery.pending_credits_key('u1')) == '10'
        assert box.process_once() and box.process_once() is False
        assert len(posted) == 1
        assert store.get(delivery.pending_credits_key('u1')) is None
        assert client.post('/agent/stream', json=body()).status_code == 402
    assert webhook._billing_delivery is None
