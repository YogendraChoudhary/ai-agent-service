"""generate_blog is long-form writing: OpenAI first (unless the user chose a model), with the writing fallback tree."""

import unittest
from unittest.mock import Mock

import pytest

from config import settings
from src import text_processing_agent as writer
from src.blog import runtime
from src.task_router import WRITING_REQUEST_TYPES, Candidate, resolve_route, task_for_request
from tests.blog.fakes import filler

ALL = {"openai", "claude", "gemini", "grok", "deepseek"}


class TestBlogRouting(unittest.TestCase):

    def test_generate_blog_is_a_writing_request(self):
        self.assertIn("generate_blog", WRITING_REQUEST_TYPES)
        task, attrs = task_for_request("generate_blog", "write something about tea")
        self.assertEqual((task, attrs["intent_rule"], attrs["classifier_used"]), ("writing", "explicit_type", False))

    def test_the_default_provider_is_openai_on_the_advanced_tier(self):
        route = resolve_route("writing", "gemini", "default", ALL, True)
        self.assertEqual(route[0], Candidate("openai", True))
        self.assertEqual([c.provider for c in route], ["openai", "gemini", "claude"])

    def test_a_model_the_user_chose_goes_first(self):
        route = resolve_route("writing", "claude", "user", ALL, True)
        self.assertEqual(route[0].provider, "claude")

    def test_only_providers_with_keys_are_candidates(self):
        self.assertEqual([c.provider for c in resolve_route("writing", "gemini", "default", {"gemini", "claude"}, True)],
                         ["gemini", "claude"])


@pytest.mark.parametrize(("query", "contract", "expected"), [
    ("Write an article on Weather Forcast in 100 Words.", True, "write"),
    ("Write a 100-word article on weather", True, "write"),
    ("Write an article on weather in 3 sentences", True, "write"),
    ("Write an article on weather in 800 words", True, "write"),
    ("Write an article on weather in 1,500 words", True, "blog"),
    ("Write a 1,500-word article on weather", True, "blog"),
    ("Write an article on weather", True, "blog"),
    ("Write an article on weather in 100 words", False, "blog"),
])
def test_article_length_controls_automatic_blog_promotion(monkeypatch, query, contract, expected):
    monkeypatch.setattr(settings, "length_contract_enabled", contract)
    monkeypatch.setattr(settings, "length_repair_enabled", False)
    monkeypatch.setattr(settings, "blog_ai_enabled", True)
    monkeypatch.setattr(settings, "blog_intent_regex_enabled", True)
    monkeypatch.setattr(settings, "blog_intent_shadow_mode", False)
    blog = Mock(return_value=iter(["Blog"]))
    write = Mock(return_value=iter([filler(100)]))
    monkeypatch.setattr(runtime, "stream_blog", blog)
    monkeypatch.setattr(writer, "_agent_text", write)
    monkeypatch.setattr(writer, "create_agent", Mock())
    list(writer.process_user_query_stream(query, api_key="test", force_agent_type=writer.FREE_FORM_AGENT))
    if expected == "blog":
        blog.assert_called_once()
        write.assert_not_called()
    else:
        blog.assert_not_called()
        write.assert_called_once()
        prompt = write.call_args.args[1]
        assert "LENGTH:" in prompt
        assert "100" not in prompt.split("\n\n", 1)[0]


if __name__ == "__main__":
    unittest.main()
