"""
AgnoLLM streaming: a call made with `on_text` is streamed the way the chat path streams (agno content events), returns the same
reply as a plain call, records its usage, and fails and falls back exactly as a plain call does. The agent is faked at the
factory (tests/blog/fakes.py); BLOG_STREAM_TITLE is the setting that turns the title-first stream on.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.text_processing_agent as tpa
from agno.agent import RunEvent
from config import Settings
from src.blog.llm import AgnoLLM, CallSpec
from src.observability import finish_trace, get_trace, start_trace
from src.task_router import Candidate
from tests.blog.fakes import FakeAgentFactory

OPENAI, CLAUDE = Candidate("openai", True), Candidate("claude", True)
KEYS = {"openai": "o", "claude": "a"}
USER = "REQUEST:\n" + json.dumps({"plan": {"target_words": 1450, "images": 3}, "write": {"id": "s1", "target_words": 200},
                                  "limits": {"max_words_without_visual_break": 220}})


class TestSetting(unittest.TestCase):

    def test_it_is_on_by_default_and_the_environment_can_turn_it_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BLOG_STREAM_TITLE", None)
            self.assertTrue(Settings().blog_stream_title)
        for raw, want in (("false", False), ("0", False), ("true", True), ("1", True)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"BLOG_STREAM_TITLE": raw}):
                self.assertEqual(Settings().blog_stream_title, want)


class TestStreamedCalls(unittest.TestCase):

    def setUp(self):
        self.factory = FakeAgentFactory()
        for p in (patch.object(tpa, "create_agent", self.factory), patch("time.sleep")):
            p.start()
            self.addCleanup(p.stop)

    def llm(self, candidates=(OPENAI, CLAUDE), **kw):
        return AgnoLLM(list(candidates), KEYS.get, **kw)

    def call(self, llm, on_text=None, stage="outline"):
        spec = CallSpec(stage, 0.4, 2000, on_text=on_text)
        messages = [{"role": "user", "content": USER}]
        return llm("You plan editorial blog posts as JSON", messages, spec)

    def test_a_streamed_call_returns_the_same_reply_and_feeds_the_listener_what_is_written_so_far(self):
        seen = []
        streamed = self.call(self.llm(), seen.append)
        plain = self.call(self.llm())
        self.assertEqual(streamed, plain)
        self.assertEqual(self.factory.streamed, [True, False])
        self.assertEqual(seen[-1], streamed)
        self.assertGreater(len(seen), 5)
        self.assertTrue(all(later.startswith(earlier) and len(later) > len(earlier) for earlier, later in zip(seen, seen[1:])))

    def test_a_call_without_a_listener_is_not_streamed(self):
        self.call(self.llm())
        self.assertEqual(self.factory.streamed, [False])

    def test_a_listener_that_raises_never_fails_the_call(self):
        def broken(text):
            raise ValueError("listener bug")
        with self.assertLogs("src.longform.llm", level="WARNING"):
            reply = self.call(self.llm(), broken)
        self.assertTrue(reply.startswith("{"))

    def test_a_provider_that_dies_mid_stream_falls_back_and_the_listener_starts_again_from_nothing(self):
        self.factory.mid_stream = {"openai": RuntimeError("503 overloaded")}
        seen = []
        llm = self.llm()
        reply = self.call(llm, seen.append)
        self.assertEqual(self.factory.providers(), ["openai", "claude"])
        self.assertEqual(llm.fallbacks, [{"from": "openai", "to": "claude", "error": "503"}])
        self.assertEqual(seen.count(seen[0]), 2)                              # the first chunk, once per provider
        self.assertEqual(seen[-1], reply)

    def test_an_error_event_in_the_stream_is_a_failed_run_like_any_other(self):
        def create_agent(*args, **kwargs):
            error = SimpleNamespace(event=RunEvent.run_error, content="429 rate limit", metrics=None)
            return SimpleNamespace(model=SimpleNamespace(id="x"), run=lambda history, stream=False, stream_events=False: iter([error]))
        with patch.object(tpa, "create_agent", create_agent):
            with self.assertRaisesRegex(RuntimeError, "openai request failed: 429 rate limit"):
                self.call(self.llm(candidates=(OPENAI,)), lambda text: None)

    def test_the_usage_of_a_streamed_call_is_recorded_from_its_last_event(self):
        metrics = SimpleNamespace(to_dict=lambda: {"input_tokens": 1800, "output_tokens": 950})

        def create_agent(*args, **kwargs):
            events = [SimpleNamespace(event=RunEvent.run_content, content='{"title": "T"}', metrics=None),
                      SimpleNamespace(event=RunEvent.run_completed, content=None, metrics=metrics)]
            return SimpleNamespace(model=SimpleNamespace(id="x"), run=lambda history, stream=False, stream_events=False: iter(events))
        start_trace("stream-usage", "/agent/stream")
        try:
            with patch.object(tpa, "create_agent", create_agent):
                reply = self.call(self.llm(candidates=(OPENAI,), request_id="stream-usage"), lambda text: None)
            attrs = get_trace("stream-usage").attrs
        finally:
            finish_trace("stream-usage")
        self.assertEqual(reply, '{"title": "T"}')
        self.assertEqual((attrs["tokens_in"], attrs["tokens_out"]), (1800, 950))

    def test_a_streamed_outline_is_attributed_like_a_plain_one(self):
        start_trace("stream-attr", "/agent/stream")
        try:
            llm = self.llm(request_id="stream-attr")
            self.call(llm, lambda text: None)
            attrs = get_trace("stream-attr").attrs
        finally:
            finish_trace("stream-attr")
        self.assertIn("outline_model", attrs)
        self.assertEqual([c["stage"] for c in attrs["llm_calls"]], ["outline"])


if __name__ == "__main__":
    unittest.main()
