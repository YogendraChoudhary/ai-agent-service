"""
Claude and `temperature`. claude-opus-5, claude-sonnet-5 and every newer model answer 400 to any temperature
("`temperature` is deprecated for this model", probed live 2026-11-11), so a request that sends it never succeeds.
Every Claude request path must send it only to the models that still take it (src.models.claude_accepts_temperature).
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.models as M
import src.text_processing_agent as tpa
from src.models import claude_accepts_temperature

# what the live API did with temperature=0.3 (2026-11-11)
ACCEPTED = ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5-20251101")
REJECTED = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5-5",
            "claude-haiku-5-5")


class TestHelper(unittest.TestCase):

    def test_matches_what_the_live_api_accepted_and_rejected(self):
        for model_id in ACCEPTED:
            with self.subTest(model=model_id):
                self.assertTrue(claude_accepts_temperature(model_id))
        for model_id in REJECTED:
            with self.subTest(model=model_id):
                self.assertFalse(claude_accepts_temperature(model_id))

    def test_unknown_or_future_models_get_no_temperature(self):
        for model_id in ("claude-opus-6", "claude-fable-5-2", "claude-haiku-6", "", None):
            with self.subTest(model=model_id):
                self.assertFalse(claude_accepts_temperature(model_id))


class TestGetModel(unittest.TestCase):

    def model(self, advanced, temperature=0.7, **kw):
        return tpa.get_model("claude", "dummy", use_advanced=advanced, cache_model=False, temperature=temperature, **kw)

    def test_the_configured_models_send_no_temperature(self):
        for advanced in (True, False):
            for temperature in (0.0, 0.2, 0.7):
                with self.subTest(advanced=advanced, temperature=temperature):
                    m = self.model(advanced, temperature)
                    self.assertFalse(claude_accepts_temperature(m.id))
                    self.assertIsNone(m.temperature)
                    self.assertNotIn("temperature", m.get_request_params())

    def test_a_model_that_still_takes_temperature_keeps_it(self):
        with patch.dict(M.ADVANCED_MODEL_IDS, {"claude": "claude-opus-4-6"}):
            m = self.model(True, 0.2)
        self.assertEqual(m.temperature, 0.2)
        self.assertEqual(m.get_request_params()["temperature"], 0.2)

    def test_other_request_params_are_untouched(self):
        from config import settings
        from src.models import REASONING_HEADROOM_TOKENS
        with patch.object(settings, "claude_advanced_effort", "low"):
            advanced = self.model(True, 0.2, max_tokens=2000).get_request_params()
        # claude-opus-5-5: headroom for thinking and the effort dial, and still no temperature
        self.assertEqual(advanced, {"max_tokens": 2048 + REASONING_HEADROOM_TOKENS, "output_config": {"effort": "low"}})
        # the basic tier (claude-sonnet-5-5) also thinks by default: headroom, and no effort unless CLAUDE_BASIC_EFFORT is set
        self.assertEqual(self.model(False, 0.2, max_tokens=2000).get_request_params(), {"max_tokens": 2048 + REASONING_HEADROOM_TOKENS})

    def test_models_that_ignore_temperature_share_one_cached_instance(self):
        a = tpa.get_model("claude", "key-claude-temp-cache", use_advanced=True, temperature=0.2)
        b = tpa.get_model("claude", "key-claude-temp-cache", use_advanced=True, temperature=0.7)
        self.assertIs(a, b)


class _Anthropic:
    """A stand-in client that records every request."""

    def __init__(self):
        self.sent = {}
        rec = self

        class _Messages:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                return SimpleNamespace(stop_reason="end_turn", usage=None, content=[SimpleNamespace(type="text", text="{}")])

        self.messages = _Messages()


class TestDirectCalls(unittest.TestCase):

    def design(self, model_id):
        client = _Anthropic()
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client), \
                patch.dict(M.ADVANCED_MODEL_IDS, {"claude": model_id}):
            tpa._run_design_json_completion([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], None, 0.2, "claude", "k")
        return client.sent

    def autocomplete(self, model_id):
        client = _Anthropic()
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client), \
                patch.dict(M.AUTOCOMPLETE_MODEL_IDS, {"claude": model_id}):
            tpa.get_autocomplete_suggestion("Thanks so much for", "", "claude", "k")
        return client.sent

    def test_design_canvas(self):
        self.assertNotIn("temperature", self.design("claude-opus-5"))
        self.assertEqual(self.design("claude-opus-4-6")["temperature"], 0.2)

    def test_autocomplete(self):
        self.assertEqual(self.autocomplete("claude-haiku-4-5")["temperature"], 0.3)       # today's model still takes it
        self.assertNotIn("temperature", self.autocomplete("claude-haiku-5-5"))             # the planned upgrade does not
        self.assertEqual(self.autocomplete("claude-haiku-5-5")["max_tokens"], 100)


class TestIntentClassifier(unittest.TestCase):

    def temperature_seen_by_the_agent(self, autocomplete_model):
        from src import intent_classifier
        seen = {}

        class FakeAgent:
            def __init__(self, model=None, **kw):
                seen["id"], seen["temperature"] = model.id, model.temperature

            def run(self, text):
                return SimpleNamespace(content="code", status=None, metrics=None)

        with patch("agno.agent.Agent", FakeAgent), patch.dict(M.AUTOCOMPLETE_MODEL_IDS, {"claude": autocomplete_model}):
            self.assertEqual(intent_classifier._call("write a function", "claude", "k", 5.0), "code")
        return seen

    def test_haiku_4_5_keeps_its_deterministic_zero(self):
        self.assertEqual(self.temperature_seen_by_the_agent("claude-haiku-4-5"), {"id": "claude-haiku-4-5", "temperature": 0})

    def test_haiku_5_5_gets_none(self):
        self.assertEqual(self.temperature_seen_by_the_agent("claude-haiku-5-5"), {"id": "claude-haiku-5-5", "temperature": None})


class TestVoiceClaudeBackend(unittest.TestCase):

    def sent(self, model_id):
        from src.voice_flow.formatter.engine import ClaudeBackend
        seen = {}

        class FakeMessages:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")])

        class FakeClient:
            def __init__(self, **kw):
                self.messages = FakeMessages()

        with patch("anthropic.AsyncAnthropic", FakeClient):
            asyncio.run(ClaudeBackend("k", model_id).generate("s", [{"role": "user", "content": "hi"}], max_output_tokens=50, timeout=5))
        return seen

    def test_temperature_only_for_models_that_take_it(self):
        self.assertEqual(self.sent("claude-haiku-4-5")["temperature"], 0)
        self.assertNotIn("temperature", self.sent("claude-haiku-5-5"))


if __name__ == "__main__":
    unittest.main()
