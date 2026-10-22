"""Blog requests show up in the latency stats, grouped by mode, with time to first content."""

import unittest
from unittest.mock import patch

from src import observability as obs
from src.blog.plan import BlogRequest, plan_blog
from src.blog.service import generate_blog_stream
from tests.blog.fakes import FakeImageTool, ScriptedLLM


class TestModeStats(unittest.TestCase):

    def test_blog_requests_are_grouped_by_mode_and_model(self):
        with patch.object(obs, "_RING", obs.deque(maxlen=50)):
            request = BlogRequest.from_options("Write a blog post about tea", "", {})
            obs.start_trace("m1", "/agent/stream", type="generate_blog", chat_model="openai")
            obs.trace_set("m1", model_used="openai/gpt-5.2")
            list(generate_blog_stream(request, plan_blog(request), ScriptedLLM(), None, request_id="m1"))
            obs.trace_mark("m1", "first_token")
            summary = obs.finish_trace("m1")
            stats = obs.latency_stats()
        self.assertEqual(summary["mode"], "blog")
        self.assertEqual(list(stats["modes"]), ["blog | openai/gpt-5.2"])
        self.assertEqual(stats["modes"]["blog | openai/gpt-5.2"]["count"], 1)
        self.assertEqual(stats["mode_first_content"]["blog | openai/gpt-5.2"]["count"], 1)

    def test_requests_without_a_mode_are_not_in_the_mode_groups(self):
        with patch.object(obs, "_RING", obs.deque(maxlen=50)):
            obs.start_trace("m2", "/agent/stream", type="general", chat_model="openai")
            obs.finish_trace("m2")
            stats = obs.latency_stats()
        self.assertEqual((stats["modes"], stats["mode_first_content"]), ({}, {}))
        self.assertIn("requests", stats)                      # the existing groups are still there


if __name__ == "__main__":
    unittest.main()
