"""generate_blog is long-form writing: OpenAI first (unless the user chose a model), with the writing fallback tree."""

import unittest

from src.task_router import WRITING_REQUEST_TYPES, Candidate, resolve_route, task_for_request

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


if __name__ == "__main__":
    unittest.main()
