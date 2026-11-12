"""
Claude's advanced tier runs on claude-opus-5-5 since 2026-11-11. It always thinks (thinking cannot be disabled and is billed as output,
inside max_tokens), so the tier sends an explicit effort from settings.claude_advanced_effort, leaves room for thinking in max_tokens, takes
no temperature, and reads replies by block type because a thinking block can come first.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.models as M
import src.text_processing_agent as tpa
from config import Settings, settings
from src.models import REASONING_HEADROOM_TOKENS, claude_effort_params


class TestHelper(unittest.TestCase):

    def test_the_advanced_default_is_opus_5_5(self):
        self.assertEqual(M.ADVANCED_MODEL_IDS["claude"], "claude-opus-5-5")

    def test_effort_goes_to_models_that_have_it(self):
        for model_id in ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-4-6"):
            with self.subTest(model=model_id):
                self.assertEqual(claude_effort_params(model_id, "low"), {"effort": "low"})

    def test_models_that_error_on_effort_get_nothing(self):
        for model_id in ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5-20251101", "gpt-5.2", "", None):
            with self.subTest(model=model_id):
                self.assertEqual(claude_effort_params(model_id, "low"), {})
        self.assertEqual(claude_effort_params("claude-opus-5-5", ""), {})


class TestSetting(unittest.TestCase):

    def effort(self, value):
        return Settings(CLAUDE_ADVANCED_EFFORT=value).claude_advanced_effort

    def test_valid_values_are_normalised(self):
        for value in ("low", "MEDIUM", " high ", "xhigh", "max"):
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), value.strip().lower())

    def test_the_default_is_low_and_anything_else_falls_back_to_it(self):
        for value in ("", "minimal", "fast", "none"):
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), "low")
        self.assertEqual(Settings().claude_advanced_effort, "low")


class TestGetModel(unittest.TestCase):

    def model(self, advanced, effort="low", **kw):
        with patch.object(settings, "claude_advanced_effort", effort):
            return tpa.get_model("claude", "dummy", use_advanced=advanced, cache_model=False, **kw)

    def test_advanced_tier_sends_the_configured_effort_and_no_temperature(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                m = self.model(True, effort, temperature=0.7)
                self.assertEqual(m.id, "claude-opus-5-5")
                self.assertEqual(m.get_request_params(), {"max_tokens": 8192, "output_config": {"effort": effort}})

    def test_the_basic_tier_does_not_follow_the_advanced_effort(self):
        m = self.model(False, "high")                                  # advanced effort set to high; basic effort left empty
        self.assertEqual(m.id, "claude-sonnet-5-5")
        self.assertNotIn("output_config", m.get_request_params())

    def test_thinking_is_never_switched_off(self):                    # Opus 5.5 returns 400 for {"type": "disabled"}
        self.assertNotIn("thinking", self.model(True).get_request_params())

    def test_a_change_of_effort_is_a_different_cached_model(self):
        with patch.object(settings, "claude_advanced_effort", "low"):
            low = tpa.get_model("claude", "key-effort-cache", use_advanced=True)
            self.assertIs(low, tpa.get_model("claude", "key-effort-cache", use_advanced=True))
        with patch.object(settings, "claude_advanced_effort", "high"):
            high = tpa.get_model("claude", "key-effort-cache", use_advanced=True)
        self.assertIsNot(low, high)
        self.assertEqual(high.output_config, {"effort": "high"})


class _Anthropic:
    def __init__(self, resp):
        self.sent, self._resp = {}, resp
        rec = self

        class _Messages:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                return rec._resp

        self.messages = _Messages()


def response(content, stop_reason="end_turn", details=None):
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=None, stop_details=details)


text = lambda t: SimpleNamespace(type="text", text=t)
thinking = lambda: SimpleNamespace(type="thinking", thinking="", signature="s")           # no .text, like the real block


class TestDesignCompletion(unittest.TestCase):

    def run_design(self, resp, effort="low"):
        client = _Anthropic(resp)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "a pricing card"}]
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client), patch.object(settings, "claude_advanced_effort", effort):
            return client.sent, tpa._run_design_json_completion(messages, None, 0.2, "claude", "k")

    def test_request_shape(self):
        sent, _ = self.run_design(response([text("{}")]))
        self.assertEqual(sent["model"], "claude-opus-5-5")
        self.assertEqual(sent["output_config"], {"effort": "low"})
        self.assertEqual(sent["max_tokens"], tpa._DESIGN_MAX_OUTPUT_TOKENS + REASONING_HEADROOM_TOKENS)
        self.assertNotIn("temperature", sent)
        self.assertNotIn("thinking", sent)

    def test_a_leading_thinking_block_does_not_hide_the_json(self):
        _, out = self.run_design(response([thinking(), text('{"ok": true}')]))
        self.assertEqual(out, '{"ok": true}')

    def test_truncation_still_raises_with_the_real_cap(self):
        with self.assertRaisesRegex(ValueError, "Design output truncated.*12288"):
            self.run_design(response([thinking()], stop_reason="max_tokens"))

    def test_a_refusal_is_an_explicit_error_not_an_empty_design(self):
        with self.assertRaisesRegex(ValueError, "declined the design request \\(cyber\\)"):
            self.run_design(response([], stop_reason="refusal", details=SimpleNamespace(category="cyber")))


if __name__ == "__main__":
    unittest.main()
