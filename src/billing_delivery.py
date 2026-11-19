"""Redis-backed billing delivery and non-expiring holds for unacknowledged charges."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
import uuid

import requests
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)
JOBS = 'agent:billing:jobs'
DUE = 'agent:billing:due'
LEASES = 'agent:billing:leases'
LEASE_SECONDS = 30
RETRY_WINDOW_SECONDS = 6 * 24 * 60 * 60  # Stay within the receiver's seven-day deduplication guarantee.
MAX_POLL_SECONDS = 60
INITIAL_REDIS_BACKOFF_SECONDS = 2


def pending_credits_key(user_id):
    return f'credits_pending_billing:{user_id}'


_ENQUEUE = """
local existing = redis.call('HGET', KEYS[1], ARGV[1])
local done = redis.call('GET', KEYS[6])
if existing then
  if cjson.decode(existing).fingerprint ~= ARGV[3] then return -1 end
  return 0
end
if done then
  if done ~= ARGV[3] then return -1 end
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1])
local hold = tonumber(ARGV[5])
if hold > 0 then redis.call('INCRBY', KEYS[4], hold) end
local reserved = tonumber(ARGV[6])
local active = tonumber(redis.call('GET', KEYS[5])) or 0
if reserved > 0 then
  if active <= reserved then redis.call('DEL', KEYS[5])
  else redis.call('DECRBY', KEYS[5], reserved) end
end
return 1
"""

_CLAIM = """
local next_job = redis.call('ZRANGE', KEYS[2], 0, 0, 'WITHSCORES')
if #next_job == 0 then return {} end
if tonumber(next_job[2]) > tonumber(ARGV[1]) then return {next_job[2]} end
local id = next_job[1]
local job = redis.call('HGET', KEYS[1], id)
if not job then
  redis.call('ZREM', KEYS[2], id)
  return {ARGV[1]} -- another job could already be due
end
redis.call('ZADD', KEYS[2], ARGV[2], id)
redis.call('HSET', KEYS[3], id, ARGV[3])
return {id, job}
"""

_UPDATE = """
if redis.call('HGET', KEYS[3], ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
if ARGV[4] == '' then redis.call('ZREM', KEYS[2], ARGV[1])
else redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1]) end
if ARGV[5] == '1' then redis.call('HDEL', KEYS[3], ARGV[1]) end
return 1
"""

_REQUEUE = """
if redis.call('HGET', KEYS[1], ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1])
return 1
"""

_COMPLETE = """
if redis.call('HGET', KEYS[3], ARGV[1]) ~= ARGV[2] then return 0 end
local raw = redis.call('HGET', KEYS[1], ARGV[1])
if not raw then return 0 end
local job = cjson.decode(raw)
if job.state ~= 'acknowledged' then return 0 end
local pending = tonumber(redis.call('GET', KEYS[4])) or 0
if pending <= job.hold_scaled then redis.call('DEL', KEYS[4])
else redis.call('DECRBY', KEYS[4], job.hold_scaled) end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('SET', KEYS[5], job.fingerprint, 'EX', 604800)
return 1
"""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class BillingDelivery:
    def __init__(self, redis_client, url, secret, *, allow_retries=False):
        self.redis = redis_client
        self.url, self.secret, self.allow_retries = url, secret, allow_retries
        self._enqueue = redis_client.register_script(_ENQUEUE)
        self._claim = redis_client.register_script(_CLAIM)
        self._update = redis_client.register_script(_UPDATE)
        self._requeue = redis_client.register_script(_REQUEUE)
        self._complete = redis_client.register_script(_COMPLETE)
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None

    def enqueue(self, payload, *, event_id=None, reserved_scaled=0, hold_scaled=0, settle_delay=0):
        if not payload.get('user_id') or min(reserved_scaled, hold_scaled, settle_delay) < 0:
            raise ValueError('Billing requires a user and nonnegative credit amounts')
        event_id = event_id or uuid.uuid4().hex
        if not isinstance(event_id, str) or not event_id or len(event_id) > 200:
            raise ValueError('Invalid billing event ID')
        job = {'payload': payload, 'hold_scaled': hold_scaled, 'reserved_scaled': reserved_scaled,
               'settle_delay': settle_delay, 'state': 'queued', 'attempts': 0}
        job['fingerprint'] = hashlib.sha256(_json(job).encode()).hexdigest()
        result = self._enqueue(keys=[JOBS, DUE, LEASES, pending_credits_key(payload['user_id']),
            f"credits_reserved:{payload['user_id']}", f'agent:billing:done:{event_id}'],
            args=[event_id, _json(job), job['fingerprint'], time.time(), hold_scaled, reserved_scaled])
        if int(result) < 0:
            raise ValueError('Billing event ID already belongs to different usage')
        self._wake.set()
        return event_id

    def _save(self, event_id, token, job, due, *, release=True):
        return bool(self._update(keys=[JOBS, DUE, LEASES],
            args=[event_id, token, _json(job), '' if due is None else due, int(release)]))

    def _can_retry(self, job, now):
        return self.allow_retries and now < job.get('retry_until', 0)

    def requeue(self, event_id: str, *, confirm_charged: bool = False) -> bool:
        """Schedule delivery or settlement of one reconciled event without changing its hold."""
        raw = self.redis.hget(JOBS, event_id)
        if raw is None:
            return False
        job = json.loads(raw)
        if job['state'] != 'reconciliation':
            return False
        now = time.time()
        if confirm_charged:
            if not job['attempts']:
                raise ValueError('Only an attempted charge can be confirmed')
            job['state'] = 'acknowledged'
            due = now + job['settle_delay']
        else:
            if job['attempts'] and not self._can_retry(job, now):
                raise ValueError('This attempted charge has no active receiver deduplication window')
            if not self.url or not self.secret:
                raise ValueError('Billing webhook configuration is required for delivery')
            job['state'] = 'queued'
            due = now
        scheduled = bool(self._requeue(keys=[JOBS, DUE], args=[event_id, raw, _json(job), due]))
        if scheduled:
            self._wake.set()
        return scheduled

    def process_once(self) -> bool:
        processed, _ = self._process_due_job()
        return processed

    def _process_due_job(self) -> tuple[bool, float | None]:
        """Process one job or return the earliest time to check again."""
        now, token = time.time(), uuid.uuid4().hex
        claimed = self._claim(keys=[JOBS, DUE, LEASES], args=[now, now + LEASE_SECONDS, token])
        if not claimed:
            return False, None
        if len(claimed) == 1:
            return False, float(claimed[0])
        event_id, raw = claimed
        if isinstance(event_id, bytes):
            event_id = event_id.decode()
        job = json.loads(raw)
        if job['state'] == 'acknowledged':
            self._complete(keys=[JOBS, DUE, LEASES, pending_credits_key(job['payload']['user_id']),
                f'agent:billing:done:{event_id}'], args=[event_id, token])
            return True, None
        # A failed response or timeout may follow a committed charge; retries need receiver-side deduplication.
        if not self.url or not self.secret or (job['attempts'] and not self._can_retry(job, now)):
            job['state'] = 'reconciliation'
            if self._save(event_id, token, job, None):
                logger.error('Billing event %s held pending configuration or reconciliation', event_id)
            return True, None
        if not job['attempts'] and self.allow_retries:
            job['retry_until'] = now + RETRY_WINDOW_SECONDS
        job['attempts'] += 1
        if not self._save(event_id, token, job, now + LEASE_SECONDS, release=False):
            return True, None
        try:
            response = requests.post(self.url, json=job['payload'], headers={
                'Content-Type': 'application/json', 'X-Webhook-Secret': self.secret,
                'Idempotency-Key': event_id}, timeout=10, allow_redirects=False)
            acknowledged = 200 <= response.status_code < 300
            job['last_error'] = None if acknowledged else f'HTTP {response.status_code}'
        except requests.RequestException as error:
            acknowledged = False
            job['last_error'] = type(error).__name__
        now = time.time()
        if acknowledged:
            job['state'] = 'acknowledged'
            due = now + job['settle_delay']
        else:
            due = None
            if self._can_retry(job, now):
                due = min(job['retry_until'], now + min(300, 2 ** min(job['attempts'], 9)))
            else:
                job['state'] = 'reconciliation'
            logger.error('Billing event %s unacknowledged (%s); credits remain held', event_id, job['last_error'])
        self._save(event_id, token, job, due)
        return True, None

    def start(self):
        def work():
            error_delay = INITIAL_REDIS_BACKOFF_SECONDS
            last_error = None
            failed_checks = 0
            outage_started = 0.0
            while True:
                self._wake.clear()
                if self._stop.is_set():
                    return
                try:
                    processed, next_due = self._process_due_job()
                except Exception as error:
                    retry_in = error_delay + random.uniform(0, INITIAL_REDIS_BACKOFF_SECONDS)
                    reason = type(error).__name__
                    message = str(error) if isinstance(error, ResponseError) else ''
                    if 'max requests limit exceeded' in message.lower():
                        reason = 'quota_exhausted'
                    if last_error is None:
                        outage_started = time.monotonic()
                    if reason != last_error:
                        detail = ''
                        if reason == 'quota_exhausted':
                            detail = ' Upstash max requests limit exceeded.'
                            quota = re.search(r'Limit:\s*(\d+),\s*Usage:\s*(\d+)', message, re.IGNORECASE)
                            if quota:
                                detail += f' Limit {quota[1]}, usage {quota[2]}.'
                        logger.error('Billing delivery paused (%s). Retrying in %.1fs with pending charges retained.%s',
                                     reason, retry_in, detail)
                        last_error = reason
                    failed_checks += 1
                    self._stop.wait(retry_in)
                    error_delay = min(MAX_POLL_SECONDS - INITIAL_REDIS_BACKOFF_SECONDS, error_delay * 2)
                    continue
                if last_error is not None:
                    logger.info('Billing delivery recovered after %.1fs and %d failed checks.',
                                time.monotonic() - outage_started, failed_checks)
                    last_error = None
                    failed_checks = 0
                error_delay = INITIAL_REDIS_BACKOFF_SECONDS
                if not processed:
                    wait = MAX_POLL_SECONDS
                    if next_due is not None:
                        wait = min(wait, max(0, next_due - time.time()))
                    self._wake.wait(wait)
        self._thread = threading.Thread(target=work, name='billing-delivery', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=12)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Requeue one held billing event in the configured Redis database.')
    parser.add_argument('event_id')
    parser.add_argument('--confirm-charged', action='store_true',
                        help='Settle without sending a webhook after verifying the receiver recorded the charge.')
    args = parser.parse_args()

    from config import settings
    from .redis_manager import RedisManager

    manager = RedisManager(settings.redis_url)
    try:
        box = BillingDelivery(manager.redis_client, settings.backend_webhook_url, settings.webhook_secret,
                              allow_retries=settings.billing_webhook_idempotent)
        try:
            scheduled = box.requeue(args.event_id, confirm_charged=args.confirm_charged)
        except ValueError as error:
            parser.exit(1, f'{error}\n')
        if not scheduled:
            parser.exit(1, 'Event is not awaiting reconciliation.\n')
        print(f'Scheduled billing event {args.event_id} for {"settlement" if args.confirm_charged else "delivery"}.')
    finally:
        manager.close()


if __name__ == '__main__':
    main()
