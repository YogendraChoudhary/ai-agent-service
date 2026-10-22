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

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence


def run_ordered(
    tasks: Sequence[Callable[[], Any]],
    parallel: int = 3,
    *,
    progress: bool = False,
    thread_name_prefix: str = "ordered",
) -> Iterator[Any]:
    n = len(tasks)
    if n == 0:
        return
    pool = ThreadPoolExecutor(max_workers=max(1, min(int(parallel), n)), thread_name_prefix=thread_name_prefix)

    def cancel_rest(future) -> None:
        if not future.cancelled() and future.exception() is not None:
            for f in futures:
                f.cancel()                      # only futures that have not started can be cancelled

    futures = [pool.submit(task) for task in tasks]
    for f in futures:
        f.add_done_callback(cancel_rest)
    try:
        for i, future in enumerate(futures):
            yield future.result()               # raises the unit's own exception
            if progress:
                yield {"__progress__": {"chunk": i + 1, "of": n}}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
