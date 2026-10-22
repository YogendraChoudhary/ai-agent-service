"""run_ordered: results in input order, bounded parallelism, cancellation behind a failure and on close."""

import threading
import time
import unittest

from src.longform.ordered_stream import run_ordered


class TestOrder(unittest.TestCase):

    def test_results_come_back_in_input_order_even_when_later_units_finish_first(self):
        delays = [0.15, 0.0, 0.05, 0.0]
        out = list(run_ordered([lambda d=d, i=i: (time.sleep(d), i)[1] for i, d in enumerate(delays)], parallel=4))
        self.assertEqual(out, [0, 1, 2, 3])

    def test_a_finished_later_unit_waits_for_its_turn_but_is_not_recomputed(self):
        runs = []
        tasks = [lambda: (time.sleep(0.1), runs.append(0), 0)[2], lambda: (runs.append(1), 1)[1]]
        self.assertEqual(list(run_ordered(tasks, parallel=2)), [0, 1])
        self.assertEqual(sorted(runs), [0, 1])

    def test_no_tasks_yields_nothing(self):
        self.assertEqual(list(run_ordered([], parallel=3)), [])

    def test_parallelism_is_bounded(self):
        live, peak, lock = 0, 0, threading.Lock()

        def task():
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.03)
            with lock:
                live -= 1
            return 1
        self.assertEqual(sum(run_ordered([task] * 9, parallel=3)), 9)
        self.assertEqual(peak, 3)

    def test_progress_sentinels_follow_each_result(self):
        out = list(run_ordered([lambda: "a", lambda: "b"], parallel=1, progress=True))
        self.assertEqual(out, ["a", {"__progress__": {"chunk": 1, "of": 2}},
                               "b", {"__progress__": {"chunk": 2, "of": 2}}])


class TestFailureAndCancellation(unittest.TestCase):

    def test_the_error_is_raised_at_the_failed_units_position(self):
        def boom():
            raise ValueError("unit 2 failed")
        got = []
        with self.assertRaisesRegex(ValueError, "unit 2 failed"):
            for r in run_ordered([lambda: "a", lambda: "b", boom, lambda: "d"], parallel=1):
                got.append(r)
        self.assertEqual(got, ["a", "b"])

    def test_units_that_have_not_started_are_cancelled_after_a_failure(self):
        started = []

        def boom():
            started.append("boom")
            raise RuntimeError("down")

        def later(i):
            def run():
                started.append(i)
                time.sleep(0.05)
                return i
            return run
        with self.assertRaises(RuntimeError):
            list(run_ordered([boom] + [later(i) for i in range(6)], parallel=1))
        time.sleep(0.2)
        self.assertEqual(started, ["boom"])

    def test_closing_the_generator_cancels_units_that_have_not_started(self):
        started = []

        def slow(i):
            def run():
                started.append(i)
                time.sleep(0.05)
                return i
            return run
        gen = run_ordered([slow(i) for i in range(8)], parallel=2)
        self.assertEqual(next(gen), 0)
        gen.close()
        time.sleep(0.3)
        self.assertLess(len(started), 8)


if __name__ == "__main__":
    unittest.main()
