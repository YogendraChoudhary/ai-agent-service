"""
Claude's basic tier runs on claude-sonnet-5-5 since 2026-11-12. It takes no temperature, thinks by default (thinking counts against
max_tokens, so it gets headroom), and has its own optional effort dial, settings.claude_basic_effort (empty = the model's default).
"""

import unittest
from unittest.mock import patch

import src.models as M
import src.text_processing_agent as tpa
from config import Settings, settings
from src.models import REASONING_HEADROOM_TOKENS, reasoning_headroom


class TestDefaults(unittest.TestCase):

    def test_the_basic_default_is_sonnet_5_5(self):
        self.assertEqual(M.BASIC_MODEL_IDS["claude"], "claude-sonnet-5-5")

    def test_sonnet_5_5_gets_headroom_for_thinking_and_sonnet_5_does_not(self):
        self.assertEqual(reasoning_headroom("claude-sonnet-5-5"), REASONING_HEADROOM_TOKENS)
        self.assertEqual(reasoning_headroom("claude-sonnet-5"), 0)


class TestSetting(unittest.TestCase):

    def effort(self, value):
        return Settings(CLAUDE_BASIC_EFFORT=value).claude_basic_effort

    def test_empty_means_the_model_default(self):
        self.assertEqual(Settings().claude_basic_effort, "")
        self.assertEqual(self.effort(""), "")

    def test_valid_values_are_normalised_and_the_rest_fall_back_to_empty(self):
        for value in ("low", "HIGH", " medium "):
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), value.strip().lower())
        for value in ("minimal", "fast", "none"):
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), "")


class TestGetModel(unittest.TestCase):

    def model(self, basic_effort, advanced_effort="low", **kw):
        with patch.object(settings, "claude_basic_effort", basic_effort), patch.object(settings, "claude_advanced_effort", advanced_effort):
            return tpa.get_model("claude", "dummy", use_advanced=False, cache_model=False, temperature=0.3, **kw)

    def test_unset_effort_sends_nothing_and_no_temperature(self):
        m = self.model("")
        self.assertEqual((m.id, m.get_request_params()), ("claude-sonnet-5-5", {"max_tokens": 8192}))

    def test_a_configured_basic_effort_is_sent(self):
        for effort in ("low", "medium", "high"):
            with self.subTest(effort=effort):
                self.assertEqual(self.model(effort).get_request_params(), {"max_tokens": 8192, "output_config": {"effort": effort}})

    def test_basic_and_advanced_efforts_are_independent(self):
        with patch.object(settings, "claude_basic_effort", "high"), patch.object(settings, "claude_advanced_effort", "low"):
            basic = tpa.get_model("claude", "dummy", use_advanced=False, cache_model=False)
            advanced = tpa.get_model("claude", "dummy", use_advanced=True, cache_model=False)
        self.assertEqual((basic.output_config, advanced.output_config), ({"effort": "high"}, {"effort": "low"}))

    def test_thinking_is_never_switched_off(self):
        self.assertNotIn("thinking", self.model("low").get_request_params())      # {"type": "disabled"} is a 400 on Sonnet 5.5

    def test_the_light_tier_classifier_does_not_inherit_the_basic_effort(self):
        # intent_classifier builds its model through get_model(basic) and then swaps the ID to the light tier's Haiku
        from types import SimpleNamespace
        from src import intent_classifier
        seen = {}

        class FakeAgent:
            def __init__(self, model=None, **kw):
                seen.update(model.get_request_params())

            def run(self, text):
                return SimpleNamespace(content="code", status=None, metrics=None)

        with patch.object(settings, "claude_basic_effort", "high"), patch("agno.agent.Agent", FakeAgent), \
                patch.dict(M.AUTOCOMPLETE_MODEL_IDS, {"claude": "claude-haiku-4-5"}):
            intent_classifier._call("write a function", "claude", "k", 5.0)
        self.assertEqual(seen, {"max_tokens": 5, "temperature": 0})               # Haiku 4.5: no effort, no thinking fields


if __name__ == "__main__":
    unittest.main()
