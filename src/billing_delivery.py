"""Redis-backed billing delivery and non-expiring holds for unacknowledged charges."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid

import requests

logger = logging.getLogger(__name__)
JOBS = 'agent:billing:jobs'
DUE = 'agent:billing:due'
LEASES = 'agent:billing:leases'
LEASE_SECONDS = 30


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
local ids = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1], 'LIMIT', 0, 1)
if #ids == 0 then return {} end
local id = ids[1]
local job = redis.call('HGET', KEYS[1], id)
if not job then redis.call('ZREM', KEYS[2], id); return {} end
redis.call('ZADD', KEYS[2], ARGV[2], id)
redis.call('HSET', KEYS[3], id, ARGV[3])
return {id, job}
"""

_UPDATE = """
if redis.call('HGET', KEYS[3], ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[1])
if ARGV[5] == '1' then redis.call('HDEL', KEYS[3], ARGV[1]) end
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
            args=[event_id, token, _json(job), due, int(release)]))

    def process_once(self):
        now, token = time.time(), uuid.uuid4().hex
        claimed = self._claim(keys=[JOBS, DUE, LEASES], args=[now, now + LEASE_SECONDS, token])
        if not claimed:
            return False
        event_id, raw = claimed
        if isinstance(event_id, bytes):
            event_id = event_id.decode()
        job = json.loads(raw)
        if job['state'] == 'acknowledged':
            self._complete(keys=[JOBS, DUE, LEASES, pending_credits_key(job['payload']['user_id']),
                f'agent:billing:done:{event_id}'], args=[event_id, token])
            return True
        # A failed response or timeout may follow a committed charge; retries need receiver-side deduplication.
        if not self.url or not self.secret or (job['attempts'] and not self.allow_retries):
            self._save(event_id, token, job, now + 300)
            logger.error('Billing event %s held pending configuration or reconciliation', event_id)
            return True
        job['attempts'] += 1
        if not self._save(event_id, token, job, now + LEASE_SECONDS, release=False):
            return True
        try:
            response = requests.post(self.url, json=job['payload'], headers={
                'Content-Type': 'application/json', 'X-Webhook-Secret': self.secret,
                'Idempotency-Key': event_id}, timeout=10, allow_redirects=False)
            acknowledged = 200 <= response.status_code < 300
            job['last_error'] = None if acknowledged else f'HTTP {response.status_code}'
        except requests.RequestException as error:
            acknowledged = False
            job['last_error'] = type(error).__name__
        if acknowledged:
            job['state'] = 'acknowledged'
            due = time.time() + job['settle_delay']
        else:
            due = time.time() + min(300, 2 ** min(job['attempts'], 9)) if self.allow_retries else time.time() + 300
            logger.error('Billing event %s unacknowledged (%s); credits remain held', event_id, job['last_error'])
        self._save(event_id, token, job, due)
        return True

    def start(self):
        def work():
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    if self.process_once():
                        continue
                except Exception as error:
                    logger.error('Billing delivery paused (%s); pending charges retained', type(error).__name__)
                self._wake.wait(1)
        self._thread = threading.Thread(target=work, name='billing-delivery', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=12)
