"""
Run independent units in parallel and hand their results back in document order.

  for text in run_ordered([lambda: write(intro), lambda: write(s1), ...], parallel=3):
      yield text

Unit i is yielded as soon as it AND every unit before it are done, so a slow early unit holds back later ones
while a fast later one is already finished when its turn comes. At most `parallel` units run at once.

Failure: when a unit raises, the units that have not started are cancelled (nothing keeps spending money behind a
request that is going to fail) and the error is raised at that unit's position. Units start in submission order,
so everything before the failed one has started and is yielded normally. Closing the generator (the client went
away) cancels whatever has not started in the same way; a call already in flight cannot be interrupted.

With progress=True a {"__progress__": {"chunk": i, "of": n}} sentinel follows each result, for the
`refine_progress` event.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextvars import copy_context
from threading import Event
from typing import Any, Callable, Iterator, Sequence

from src.provider_work import check_cancelled, provider_batch_stop


def run_ordered(
    tasks: Sequence[Callable[[], Any]],
    parallel: int = 3,
    *,
    progress: bool = False,
    thread_name_prefix: str = "ordered",
    cancel_event=None,
) -> Iterator[Any]:
    n = len(tasks)
    if n == 0:
        return
    limit = max(1, min(int(parallel), n))
    pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix=thread_name_prefix)
    failed = Event()
    closed = Event()

    def run(task):
        check_cancelled(cancel_event)
        if failed.is_set():
            raise CancelledError()
        token = provider_batch_stop.set(closed)
        try:
            return task()
        except BaseException:
            failed.set()
            raise
        finally:
            provider_batch_stop.reset(token)

    def submit(index):
        return pool.submit(copy_context().run, run, tasks[index])

    futures = {}
    try:
        for index in range(limit):
            check_cancelled(cancel_event)
            futures[index] = submit(index)
        for i in range(n):
            check_cancelled(cancel_event)
            result = futures.pop(i).result()
            check_cancelled(cancel_event)
            yield result
            if progress:
                yield {"__progress__": {"chunk": i + 1, "of": n}}
            if i + limit < n and not failed.is_set():
                check_cancelled(cancel_event)
                futures[i + limit] = submit(i + limit)
    finally:
        closed.set()
        failed.set()
        pool.shutdown(wait=False, cancel_futures=True)
