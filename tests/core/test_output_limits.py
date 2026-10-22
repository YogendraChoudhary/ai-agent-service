"""
D15 / Phase 0: an explicit output budget on model calls. Claude's 8,192-token default silently cut long answers;
callers that know how much they expect now say so, and reasoning models get headroom for hidden tokens.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.text_processing_agent as tpa
from agno.agent import RunEvent
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


class TestRefinementBudget(unittest.TestCase):
    """Phase 0: the four agents that polish a user's own text are given room for a whole document."""

    def test_the_budget_is_a_bit_over_the_input_between_the_limits(self):
        budget = lambda chars: tpa.refinement_max_tokens("grammar", "x" * chars)
        self.assertEqual(budget(100), 4096)                                          # a short text still gets the floor
        self.assertEqual(budget(30_000), 13_714)                                     # ~5,000 words: 1.6 x ~8,600 tokens
        self.assertEqual(budget(400_000), 16_384)                                    # never above the ceiling
        self.assertEqual(tpa.refinement_max_tokens("grammar", None), 4096)

    def test_only_the_four_refinement_agents_get_one(self):
        self.assertEqual(set(tpa.REFINEMENT_AGENTS), {"enhance", "grammar", "tone", "simplify"})
        for agent in tpa.REFINEMENT_AGENTS:
            self.assertIsNotNone(tpa.refinement_max_tokens(agent, "text"), agent)
        for agent in ("translation", "make_longer", "make_shorter", "completion", "smart_global", "blog_writer", "code_writer"):
            self.assertIsNone(tpa.refinement_max_tokens(agent, "x" * 30_000), agent)

    def test_the_single_call_path_hands_the_budget_to_the_agent(self):
        made = []

        def create_agent(agent_type, provider, key, request_id=None, **kw):
            made.append((agent_type, kw["max_tokens"]))
            event = SimpleNamespace(event=RunEvent.run_content, content="ok", metrics=None)
            return SimpleNamespace(model=SimpleNamespace(id="m"), run=lambda query, **k: iter([event]))
        query = "Fix: " + "word " * 6000
        with patch.object(tpa, "create_agent", create_agent):
            for agent in ("grammar", "enhance", "translation"):
                list(tpa.process_user_query_stream(query, model_provider="openai", api_key="k", force_agent_type=agent))
        self.assertEqual(made, [("grammar", tpa.refinement_max_tokens("grammar", query)),
                                ("enhance", tpa.refinement_max_tokens("enhance", query)), ("translation", None)])
        self.assertGreater(made[0][1], 13_000)                                       # sized from the text, not the floor


class TestRefinementRules(unittest.TestCase):
    """The prompt side of Phase 0: what must not move, said in the agents' own instructions."""

    def test_the_four_agents_carry_the_structure_and_their_own_length_rule(self):
        for agent in tpa.REFINEMENT_AGENTS:
            rules = tpa.AGENT_CONFIGS[agent]["instructions"]
            with self.subTest(agent=agent):
                self.assertIn(tpa.REFINE_STRUCTURE_RULE, rules)
                self.assertIn(tpa.REFINE_LENGTH_RULES[agent], rules)
                last_rule = next(i for i, r in enumerate(rules) if r.startswith("CRITICAL: Output ONLY"))
                self.assertLess(rules.index(tpa.REFINE_STRUCTURE_RULE), last_rule)     # before the output-format line

    def test_simplify_may_shrink_and_the_others_may_not(self):
        self.assertIn("somewhat shorter", tpa.REFINE_LENGTH_RULES["simplify"])
        for agent in ("grammar", "enhance", "tone"):
            self.assertNotIn("shorter", tpa.REFINE_LENGTH_RULES[agent])

    def test_no_other_agent_is_touched(self):
        for name, config in tpa.AGENT_CONFIGS.items():
            if name not in tpa.REFINEMENT_AGENTS:
                with self.subTest(agent=name):
                    self.assertNotIn(tpa.REFINE_STRUCTURE_RULE, config["instructions"] if isinstance(config["instructions"], list) else [])


if __name__ == "__main__":
    unittest.main()
