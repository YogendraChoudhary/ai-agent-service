"""
D15 / Phase 0: an explicit output budget on model calls. Claude's 8,192-token default silently cut long answers;
callers that know how much they expect now say so, and reasoning models get headroom for hidden tokens.
"""

import unittest
from unittest.mock import patch

import src.text_processing_agent as tpa
from src.models import OUTPUT_TOKEN_CEILING, REASONING_HEADROOM_TOKENS, output_token_budget, reasoning_headroom


def limit_of(model):
    return {k: getattr(model, k) for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if hasattr(model, k)}


class TestBudgetMaths(unittest.TestCase):

    def test_reasoning_models_get_headroom_others_do_not(self):
        for model_id in ("gpt-5.2", "gpt-5", "o3-mini", "grok-4.6"):
            with self.subTest(model=model_id):
                self.assertEqual(reasoning_headroom(model_id), REASONING_HEADROOM_TOKENS)
        for model_id in ("gpt-4o-mini", "claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-chat",
                         "deepseek-reasoner", "grok-4.20-0309-non-reasoning", ""):
            with self.subTest(model=model_id):
                self.assertEqual(reasoning_headroom(model_id), 0)

    def test_budget_is_visible_tokens_plus_headroom_and_never_above_the_ceiling(self):
        self.assertEqual(output_token_budget("claude-opus-5", 1500), 1500)
        self.assertEqual(output_token_budget("gpt-5.2", 1500), 1500 + REASONING_HEADROOM_TOKENS)
        self.assertEqual(output_token_budget("claude-opus-5", 10 ** 6), OUTPUT_TOKEN_CEILING)

    def test_rounding_keeps_the_model_cache_bounded(self):
        self.assertEqual([tpa._round_max_tokens(n) for n in (None, 0, -5, 1, 1024, 1025, 3000)], [None, None, None, 1024, 1024, 2048, 3072])


class TestGetModel(unittest.TestCase):

    def model(self, provider, advanced=True, **kw):
        return tpa.get_model(provider, "dummy", use_advanced=advanced, cache_model=False, **kw)

    def test_default_behaviour_is_unchanged(self):
        self.assertEqual(limit_of(self.model("claude"))["max_tokens"], 8192)
        for provider in ("openai", "gemini", "grok", "deepseek"):
            with self.subTest(provider=provider):
                self.assertEqual({v for v in limit_of(self.model(provider)).values()}, {None})

    def test_each_provider_gets_its_own_limit_field(self):
        self.assertEqual(limit_of(self.model("claude", max_tokens=700)), {"max_tokens": 1024})
        self.assertEqual(limit_of(self.model("gemini", max_tokens=700)), {"max_output_tokens": 1024})
        self.assertEqual(limit_of(self.model("openai", max_tokens=700))["max_completion_tokens"], 1024 + REASONING_HEADROOM_TOKENS)
        self.assertIsNone(limit_of(self.model("openai", max_tokens=700))["max_tokens"])     # gpt-5 rejects max_tokens
        self.assertEqual(limit_of(self.model("openai", advanced=False, max_tokens=700))["max_tokens"], 1024)
        self.assertEqual(limit_of(self.model("deepseek", advanced=False, max_tokens=700))["max_tokens"], 1024)
        self.assertEqual(limit_of(self.model("grok", max_tokens=700))["max_tokens"], 1024 + REASONING_HEADROOM_TOKENS)

    def test_the_deepseek_tool_downgrade_keeps_the_limit(self):
        m = self.model("deepseek", needs_tools=True, max_tokens=500)
        self.assertEqual((m.id, m.max_tokens), ("deepseek-chat", 1024))

    def test_models_are_cached_per_rounded_budget(self):
        with patch.dict(tpa._MODEL_CACHE, {}, clear=True):
            a = tpa.get_model("openai", "key-A", max_tokens=700)
            self.assertIs(a, tpa.get_model("openai", "key-A", max_tokens=1000))       # same 1,024 bucket
            self.assertIsNot(a, tpa.get_model("openai", "key-A", max_tokens=1100))    # next bucket
            self.assertIsNot(a, tpa.get_model("openai", "key-A"))                     # no budget at all
            self.assertEqual(len(tpa._MODEL_CACHE), 3)

    def test_byok_models_are_never_cached(self):
        with patch.dict(tpa._MODEL_CACHE, {}, clear=True):
            self.model("openai", max_tokens=700)
            self.assertEqual(tpa._MODEL_CACHE, {})

    def test_prewarming_still_works_with_the_blog_agent_configured(self):
        with patch.dict(tpa._MODEL_CACHE, {}, clear=True):
            self.assertGreater(tpa.prewarm_models({"openai_api_key": "k"}), 0)


if __name__ == "__main__":
    unittest.main()
