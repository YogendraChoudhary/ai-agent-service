"""call_with_timeout: a call that does not answer in time is a retryable error, not a request that never ends."""

import threading
import time
import unittest

from src.longform.llm import call_with_timeout
from src.provider_errors import can_fall_back, is_retryable


class TestCallWithTimeout(unittest.TestCase):

    def test_a_call_that_answers_in_time_returns_its_value(self):
        self.assertEqual(call_with_timeout(lambda: "reply", 1.0), "reply")

    def test_no_limit_means_a_plain_call(self):
        for seconds in (None, 0):
            self.assertEqual(call_with_timeout(lambda: "reply", seconds), "reply")
        names = []
        call_with_timeout(lambda: names.append(threading.current_thread().name), None)
        self.assertEqual(names, [threading.current_thread().name])                        # no thread was started

    def test_its_own_exception_comes_back_as_it_was_raised(self):
        class Boom(Exception):
            pass
        with self.assertRaises(Boom):
            call_with_timeout(lambda: (_ for _ in ()).throw(Boom("x")), 1.0)

    def test_a_call_that_does_not_answer_raises_a_timeout_the_fallback_rules_retry(self):
        release = threading.Event()
        started = time.perf_counter()
        with self.assertRaises(TimeoutError) as raised:
            call_with_timeout(lambda: release.wait(5), 0.2, "The gemini call")
        release.set()
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(str(raised.exception), "The gemini call timed out after 0.2s")
        self.assertTrue(is_retryable(raised.exception))
        self.assertTrue(can_fall_back(raised.exception))

    def test_the_stuck_call_does_not_keep_the_process_alive(self):
        release = threading.Event()
        with self.assertRaises(TimeoutError):
            call_with_timeout(lambda: release.wait(5), 0.1)
        stuck = [t for t in threading.enumerate() if t.name == "llm-call"]
        release.set()
        self.assertTrue(all(t.daemon for t in stuck))


if __name__ == "__main__":
    unittest.main()
