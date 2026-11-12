"""
Claude's light tier (autocomplete, code-intent classifier, voice formatter) runs on claude-haiku-5-5 since 2026-11-11.
Haiku 5.x thinks by default and thinking counts against max_tokens, so these calls turn it off and ask for the lowest effort;
Haiku 4.5 and older reject `output_config.effort`, so they must not get it.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.models as M
import src.text_processing_agent as tpa
from src.models import claude_light_request_params

LIGHT = {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}


class TestHelper(unittest.TestCase):

    def test_haiku_5_and_newer_get_thinking_off_and_lowest_effort(self):
        for model_id in ("claude-haiku-5-5", "claude-haiku-5", "claude-haiku-6"):
            with self.subTest(model=model_id):
                self.assertEqual(claude_light_request_params(model_id), LIGHT)

    def test_older_haiku_and_other_models_get_nothing(self):
        for model_id in ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-3-haiku-20240307", "claude-haiku-3-5",
                         "claude-sonnet-5-5", "claude-opus-5-5", "gpt-4o-mini", "", None):
            with self.subTest(model=model_id):
                self.assertEqual(claude_light_request_params(model_id), {})

    def test_each_call_gets_its_own_copy(self):
        a = claude_light_request_params("claude-haiku-5-5")
        a["thinking"]["type"] = "enabled"
        self.assertEqual(claude_light_request_params("claude-haiku-5-5"), LIGHT)

    def test_the_light_tier_default_is_haiku_5_5(self):
        self.assertEqual(M.AUTOCOMPLETE_MODEL_IDS["claude"], "claude-haiku-5-5")


class _Anthropic:
    def __init__(self, content):
        self.sent, self._content = {}, content
        rec = self

        class _Messages:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                return SimpleNamespace(stop_reason="end_turn", content=rec._content)

        self.messages = _Messages()


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking():
    return SimpleNamespace(type="thinking", thinking="", signature="s")          # no .text attribute, like the real block


class TestAutocomplete(unittest.TestCase):

    def run_autocomplete(self, model_id, content):
        client = _Anthropic(content)
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client), \
                patch.dict(M.AUTOCOMPLETE_MODEL_IDS, {"claude": model_id}):
            result = tpa.get_autocomplete_suggestion("Thanks so much for the quick", "", "claude", "k")
        return client.sent, result

    def test_haiku_5_5_request(self):
        sent, _ = self.run_autocomplete("claude-haiku-5-5", [text(" turnaround!")])
        self.assertEqual(sent["model"], "claude-haiku-5-5")
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertEqual(sent["output_config"], {"effort": "low"})
        self.assertEqual(sent["max_tokens"], 100)
        self.assertNotIn("temperature", sent)

    def test_haiku_4_5_request_is_unchanged(self):
        sent, _ = self.run_autocomplete("claude-haiku-4-5", [text(" turnaround!")])
        self.assertEqual(sent["temperature"], 0.3)
        self.assertNotIn("thinking", sent)
        self.assertNotIn("output_config", sent)

    def test_a_leading_thinking_block_does_not_hide_the_text(self):
        _, result = self.run_autocomplete("claude-haiku-5-5", [thinking(), text(" turnaround!")])
        self.assertEqual(result, " turnaround!")

    def test_a_reply_without_text_is_no_suggestion(self):                            # refusal, or max_tokens spent on thinking
        for content in ([], None, [thinking()]):
            with self.subTest(content=content):
                self.assertEqual(self.run_autocomplete("claude-haiku-5-5", content)[1], "")


class TestClassifier(unittest.TestCase):

    def agent_model(self, light_model):
        from src import intent_classifier
        seen = {}

        class FakeAgent:
            def __init__(self, model=None, **kw):
                seen.update(id=model.id, temperature=model.temperature, max_tokens=model.max_tokens,
                            thinking=model.thinking, output_config=model.output_config,
                            params=model.get_request_params())

            def run(self, text):
                return SimpleNamespace(content="code", status=None, metrics=None)

        with patch("agno.agent.Agent", FakeAgent), patch.dict(M.AUTOCOMPLETE_MODEL_IDS, {"claude": light_model}):
            self.assertEqual(intent_classifier._call("write a function", "claude", "k", 5.0), "code")
        return seen

    def test_haiku_5_5_request(self):
        seen = self.agent_model("claude-haiku-5-5")
        self.assertEqual(seen["id"], "claude-haiku-5-5")
        self.assertEqual(seen["params"], {"max_tokens": 8, "thinking": {"type": "disabled"}, "output_config": {"effort": "low"}})

    def test_haiku_4_5_request_is_unchanged(self):
        seen = self.agent_model("claude-haiku-4-5")
        self.assertEqual(seen["params"], {"max_tokens": 5, "temperature": 0})

    def test_other_providers_are_untouched(self):
        from src import intent_classifier
        seen = {}

        class FakeAgent:
            def __init__(self, model=None, **kw):
                seen.update(max_tokens=model.max_tokens, temperature=model.temperature)

            def run(self, text):
                return SimpleNamespace(content="other", status=None, metrics=None)

        with patch("agno.agent.Agent", FakeAgent):
            intent_classifier._call("hello", "openai", "k", 5.0)
        self.assertEqual(seen, {"max_tokens": 5, "temperature": 0})


class TestVoiceBackend(unittest.TestCase):

    def sent(self, model_id):
        from src.voice_flow.formatter.engine import ClaudeBackend
        seen = {}

        class FakeMessages:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(stop_reason="end_turn", content=[thinking(), text("ok")])

        class FakeClient:
            def __init__(self, **kw):
                self.messages = FakeMessages()

        with patch("anthropic.AsyncAnthropic", FakeClient):
            out = asyncio.run(ClaudeBackend("k", model_id).generate("s", [{"role": "user", "content": "hi"}], max_output_tokens=50, timeout=5))
        self.assertEqual(out, "ok")
        return seen

    def test_haiku_5_5_request(self):
        seen = self.sent("claude-haiku-5-5")
        self.assertEqual((seen["thinking"], seen["output_config"]), ({"type": "disabled"}, {"effort": "low"}))
        self.assertNotIn("temperature", seen)

    def test_haiku_4_5_request_is_unchanged(self):
        seen = self.sent("claude-haiku-4-5")
        self.assertEqual(seen["temperature"], 0)
        self.assertNotIn("thinking", seen)


if __name__ == "__main__":
    unittest.main()
