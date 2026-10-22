"""
get_model under concurrency: threads that ask for the same model at once share one instance, and a Gemini model has its client built
before anyone uses it. Without that, the first parallel calls on a cold model each built a client and the loser was garbage collected
(google-genai then closes its HTTP connection) while its request was running: "Cannot send a request, as the client has been closed".
"""

import threading
import time
import unittest
from unittest.mock import patch

import src.text_processing_agent as tpa
from agno.models.google import Gemini


class SlowGemini(Gemini):
    """Building a model takes a moment, so every thread passes the cache check before any of them has stored its model."""

    def __init__(self, *args, **kwargs):
        time.sleep(0.05)
        super().__init__(*args, **kwargs)


def get_all(n, provider="gemini", **kw):
    barrier, models = threading.Barrier(n), []
    lock = threading.Lock()

    def work():
        barrier.wait()
        model = tpa.get_model(provider, "key-race", use_advanced=False, max_tokens=2000, **kw)
        with lock:
            models.append(model)
    threads = [threading.Thread(target=work) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return models


class TestTheModelCacheUnderConcurrency(unittest.TestCase):

    def setUp(self):
        for p in (patch.dict(tpa._MODEL_CACHE, {}, clear=True), patch.object(tpa, "Gemini", SlowGemini)):
            p.start()
            self.addCleanup(p.stop)

    def test_threads_asking_at_once_for_the_same_model_share_one(self):
        models = get_all(8)
        self.assertEqual(len(models), 8)
        self.assertEqual(len({id(m) for m in models}), 1)
        self.assertEqual(len(tpa._MODEL_CACHE), 1)

    def test_a_gemini_model_comes_with_its_client_already_built_and_every_caller_gets_the_same_one(self):
        models = get_all(8)
        self.assertIsNotNone(models[0].client)
        self.assertEqual(len({id(m.client) for m in models}), 1)

    def test_a_later_caller_gets_the_cached_model_and_client_unchanged(self):
        (first, *_) = get_all(2)
        again = tpa.get_model("gemini", "key-race", use_advanced=False, max_tokens=2000)
        self.assertIs(again, first)
        self.assertIs(again.client, first.client)

    def test_other_providers_are_shared_the_same_way_and_left_alone(self):
        models = get_all(6, provider="openai")
        self.assertEqual(len({id(m) for m in models}), 1)

    def test_a_model_that_is_not_cached_is_not_touched(self):
        model = tpa.get_model("gemini", "key-race", use_advanced=False, cache_model=False)
        self.assertIsNone(model.client)
        self.assertEqual(tpa._MODEL_CACHE, {})

    def test_building_the_client_ahead_never_fails_a_request(self):
        class Broken(Gemini):
            def get_client(self):
                raise ValueError("no api key")
        broken = Broken(id="x")
        tpa._warm_client(broken)                          # the call itself will report the problem
        tpa._warm_client(object())                        # not a Gemini model: nothing to do


if __name__ == "__main__":
    unittest.main()
