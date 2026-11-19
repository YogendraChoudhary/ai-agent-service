"""Bound synchronous provider work independently of the API's Redis thread capacity."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar, copy_context

from fastapi import HTTPException

MODEL_CALL_TIMEOUT_SECONDS = 150.0
MODEL_REQUEST_TIMEOUT_SECONDS = 300.0
AUTOCOMPLETE_TIMEOUT_SECONDS = 8.0


class ProviderCancelled(Exception):
    pass


class ProviderBusy(Exception):
    pass


class _Workers:
    def __init__(self, workers, name):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.slots = threading.BoundedSemaphore(workers * 2)

    def submit(self, func, *args, **kwargs):
        if not self.slots.acquire(blocking=False):
            raise ProviderBusy('Model workers are busy. Try again shortly.')
        try:
            future = self.pool.submit(copy_context().run, func, *args, **kwargs)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future


_request_workers = _Workers(16, 'provider-request')
_call_workers = _Workers(32, 'provider-call')
_END = object()
_active_request = ContextVar('provider_request', default=(None, float('inf')))
provider_batch_stop = ContextVar('provider_batch_stop', default=None)


def check_cancelled(event):
    if event is not None and event.is_set():
        raise ProviderCancelled()


def check_provider_call():
    event, deadline = _active_request.get()
    check_cancelled(event)
    check_cancelled(provider_batch_stop.get())
    if time.monotonic() >= deadline:
        raise TimeoutError('Model request timed out')


def call_with_timeout(func, seconds, what='The model call'):
    if not seconds:
        return func()
    event, request_deadline = _active_request.get()
    deadline = min(time.monotonic() + seconds, request_deadline)
    message = f'{what} timed out after {seconds:g}s'

    def run():
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        check_cancelled(event)
        check_cancelled(provider_batch_stop.get())
        token = _active_request.set((event, deadline))
        try:
            return func()
        finally:
            _active_request.reset(token)

    future = _call_workers.submit(run)
    try:
        return future.result(timeout=max(0, deadline - time.monotonic()))
    except FutureTimeout as error:
        if future.done() and future.exception() is error:
            raise
        raise TimeoutError(message) from None


async def _await_worker(future):
    future = asyncio.wrap_future(future)
    # A cancelled waiter leaves the slot occupied until the worker exits.
    future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    return await asyncio.shield(future)


async def run_provider_call(func, *args, stop_event=None):
    def run():
        check_cancelled(stop_event)
        check_provider_call()
        return func(*args)

    return await _await_worker(_call_workers.submit(run))


async def run_provider_work(func, *args, stop_event=None, deadline=None, **kwargs):
    deadline = deadline if deadline is not None else time.monotonic() + MODEL_REQUEST_TIMEOUT_SECONDS
    stop_event = stop_event if stop_event is not None else threading.Event()

    def run():
        token = _active_request.set((stop_event, deadline))
        try:
            check_provider_call()
            return func(*args, **kwargs)
        finally:
            _active_request.reset(token)

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Model request timed out')
        return await asyncio.wait_for(_await_worker(_request_workers.submit(run)), remaining)
    except ProviderBusy as error:
        raise HTTPException(503, str(error), headers={'Retry-After': '2'}) from error
    except (TimeoutError, asyncio.TimeoutError) as error:
        stop_event.set()
        raise HTTPException(504, 'Model request timed out') from error
    except asyncio.CancelledError:
        stop_event.set()
        raise


async def iterate_provider(iterator, stop_event):
    deadline = time.monotonic() + MODEL_REQUEST_TIMEOUT_SECONDS
    while not stop_event.is_set():
        try:
            item = await run_provider_work(next, iterator, _END, stop_event=stop_event, deadline=deadline)
        except ProviderCancelled:
            return
        if item is _END:
            return
        yield item
