"""
BLOG_OUTLINE_MODEL: a faster model for the FIRST outline attempt only. The outline is ~1,000 tokens of JSON, so its time
is generation speed (measured: gpt-5.2 12-15 s, gemini-3.5-flash-lite ~4 s). The repair and every part stay on the
drafting model, a request may only use providers its keys allow, and a failure falls back to the drafting providers.
"""

import json
import os
import unittest
from unittest.mock import patch

import src.models as models
import src.text_processing_agent as tpa
from config import Settings, settings
from src.blog.llm import AgnoLLM, CallSpec, model_id_for
from src.blog.runtime import outline_chain, parse_outline_model
from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS
from src.observability import finish_trace, get_trace, start_trace
from src.task_router import Candidate, KeyResolver
from tests.blog.fakes import FakeAgentFactory, ScriptedLLM, good_outline
from tests.blog.test_runtime import BRIEF, RuntimeCase

OPENAI, GEMINI, CLAUDE = Candidate("openai", True), Candidate("gemini", True), Candidate("claude", True)


class TestSetting(unittest.TestCase):

    def parse(self, value):
        with patch.dict(os.environ, {"BLOG_OUTLINE_MODEL": value}):
            return Settings().blog_outline_model

    def test_valid_values_are_normalised(self):
        for raw, want in (("", ""), ("gemini:basic", "gemini:basic"), (" Gemini ", "gemini"),
                          ("openai:advanced", "openai:advanced"), ("GROK:BASIC", "grok:basic")):
            with self.subTest(raw=raw):
                self.assertEqual(self.parse(raw), want)

    def test_invalid_values_mean_the_drafting_model(self):
        for raw in ("gemini:fast", "nonsense", "claude:", "gpt-5.2", "gemini:basic:x"):
            with self.subTest(raw=raw):
                self.assertEqual(self.parse(raw), "")

    def test_ships_unset(self):
        self.assertEqual(Settings.model_fields["blog_outline_model"].default, "")

    def test_parse(self):
        self.assertIsNone(parse_outline_model(""))
        self.assertIsNone(parse_outline_model(None))
        for junk in ("nope", "gemini:fast", "gemini:", "gemini:basic:x", ":basic", "gpt-5.2"):
            with self.subTest(junk=junk):
                self.assertIsNone(parse_outline_model(junk))        # never read as something else
        self.assertEqual(parse_outline_model("gemini"), ("gemini", False))          # a fast model is the point: basic
        self.assertEqual(parse_outline_model("gemini:basic"), ("gemini", False))
        self.assertEqual(parse_outline_model("openai:advanced"), ("openai", True))


class TestOutlineChain(unittest.TestCase):

    def test_nothing_configured_changes_nothing(self):
        self.assertIsNone(outline_chain("", [OPENAI, GEMINI], {"openai", "gemini"}))

    def test_the_fast_model_goes_first_and_the_drafting_providers_are_its_fallbacks(self):
        chain = outline_chain("gemini:basic", [OPENAI, CLAUDE], {"openai", "gemini", "claude"})
        self.assertEqual(chain, [Candidate("gemini", False), OPENAI, CLAUDE])

    def test_a_provider_the_request_may_not_use_is_never_chosen(self):
        # a BYOK account with only its own OpenAI key must not reach the platform's Gemini key
        self.assertIsNone(outline_chain("gemini:basic", [OPENAI], {"openai"}))

    def test_without_a_router_only_the_provider_in_use_can_serve(self):
        self.assertIsNone(outline_chain("gemini:basic", [OPENAI], None))
        self.assertEqual(outline_chain("openai:basic", [OPENAI], None), [Candidate("openai", False), OPENAI])

    def test_the_model_that_drafts_anyway_is_not_repeated(self):
        self.assertIsNone(outline_chain("openai:advanced", [OPENAI, GEMINI], {"openai", "gemini"}))
        self.assertIsNone(outline_chain("openai:advanced", [OPENAI], None))

    def test_a_blocked_model_is_never_chosen(self):
        with patch.object(models, "BLOCKED_MODELS", frozenset({BASIC_MODEL_IDS["gemini"]})):
            self.assertIsNone(outline_chain("gemini:basic", [OPENAI], {"openai", "gemini"}))
            self.assertIsNotNone(outline_chain("gemini:advanced", [OPENAI], {"openai", "gemini"}))

    def test_no_candidates_no_chain(self):
        self.assertIsNone(outline_chain("gemini:basic", [], {"gemini"}))

    def test_model_ids(self):
        self.assertEqual(model_id_for(Candidate("gemini", False)), BASIC_MODEL_IDS["gemini"])
        self.assertEqual(model_id_for(OPENAI), ADVANCED_MODEL_IDS["openai"])
        self.assertEqual(model_id_for(Candidate("deepseek", True), use_web=True), "deepseek-chat")


class TestAgnoLLMChains(unittest.TestCase):
    """AgnoLLM through the real create_agent path, with agents faked at the factory."""

    KEYS = {"openai": "o", "gemini": "g", "claude": "a"}

    def setUp(self):
        self.factory = FakeAgentFactory()
        for p in (patch.object(tpa, "create_agent", self.factory), patch("time.sleep")):
            p.start()
            self.addCleanup(p.stop)

    def llm(self, candidates=(OPENAI, CLAUDE), outline=(Candidate("gemini", False), OPENAI, CLAUDE), keys=None, **kw):
        keys = self.KEYS if keys is None else keys
        return AgnoLLM(list(candidates), keys.get, outline_candidates=list(outline) if outline else None, **kw)

    OUTLINE = CallSpec("outline", 0.4, 2000)
    REPAIR = CallSpec("outline_repair", 0.4, 2000)
    UNIT = CallSpec("unit", 0.7, 700, unit="s1")

    def call(self, llm, spec, system="You plan editorial blog posts as JSON", user=None):
        user = user or ("REQUEST:\n" + json.dumps(
            {"plan": {"target_words": 1400, "images": 3}, "write": {"id": "s1", "target_words": 200},
             "limits": {"max_words_without_visual_break": 220}}))
        messages = [{"role": "user", "content": user}]
        if spec.stage.endswith("repair"):
            messages += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "fix"}]
        return llm(system, messages, spec)

    def served(self):
        return [(c[1], c[3]["plan"].advanced_model) for c in self.factory.created]

    def test_the_first_outline_attempt_uses_the_fast_model_and_everything_else_the_drafting_model(self):
        llm = self.llm()
        self.call(llm, self.OUTLINE)
        self.call(llm, self.REPAIR)
        self.call(llm, self.UNIT, system="You are a senior editorial writer")
        self.assertEqual(self.served(), [("gemini", False), ("openai", True), ("openai", True)])
        self.assertEqual([c[2] for c in self.factory.created], ["g", "o", "o"])

    def test_without_a_fast_model_nothing_changes(self):
        llm = self.llm(outline=None)
        self.call(llm, self.OUTLINE)
        self.assertEqual(self.served(), [("openai", True)])

    def test_attribution_follows_the_drafting_model_not_the_fast_first_pass(self):
        start_trace("om1", "/agent/stream")
        try:
            llm = self.llm(request_id="om1")
            self.call(llm, self.OUTLINE)
            self.assertIsNone(llm.used)                                     # no part has been written yet
            self.assertEqual(llm.outline_used, ("gemini", BASIC_MODEL_IDS["gemini"]))
            self.assertEqual(llm.drafting_model(), ("openai", ADVANCED_MODEL_IDS["openai"]))
            self.assertNotIn("model_used", get_trace("om1").attrs)
            self.call(llm, self.UNIT, system="You are a senior editorial writer")
            attrs = get_trace("om1").attrs
        finally:
            finish_trace("om1")
        self.assertEqual(attrs["outline_model"], f"gemini/{BASIC_MODEL_IDS['gemini']}")
        self.assertEqual(attrs["model_used"], f"openai/{ADVANCED_MODEL_IDS['openai']}")
        self.assertEqual(llm.used, ("openai", ADVANCED_MODEL_IDS["openai"]))

    def test_a_fast_model_that_fails_falls_back_to_the_drafting_providers_for_that_call(self):
        self.factory.fail = {"gemini": RuntimeError("503 overloaded")}
        llm = self.llm()
        self.call(llm, self.OUTLINE)
        self.assertEqual(self.factory.providers(), ["gemini", "openai"])
        self.assertEqual(llm.fallbacks, [{"from": "gemini", "to": "openai", "error": "503"}])
        self.call(llm, self.OUTLINE)                                         # remembered: not tried a second time
        self.assertEqual(self.factory.runs["gemini"], 1)

    def test_a_drafting_provider_that_failed_during_the_outline_is_not_retried_by_every_part(self):
        self.factory.fail = {"gemini": RuntimeError("503 overloaded"), "openai": RuntimeError("503 overloaded")}
        llm = self.llm()
        self.call(llm, self.OUTLINE)                                         # gemini -> openai -> claude
        self.assertEqual(self.factory.providers(), ["gemini", "openai", "claude"])
        for _ in range(3):
            self.call(llm, self.UNIT, system="You are a senior editorial writer")
        self.assertEqual(self.factory.runs["openai"], 1)                     # the parts went straight to claude
        self.assertEqual(llm.drafting_model()[0], "claude")

    def test_every_call_is_logged_with_its_stage_model_and_seconds(self):
        self.factory.fail = {"gemini": RuntimeError("503 overloaded")}
        start_trace("om3", "/agent/stream")
        try:
            llm = self.llm(request_id="om3")
            self.call(llm, self.OUTLINE)
            self.call(llm, self.UNIT, system="You are a senior editorial writer")
            attrs = get_trace("om3").attrs
        finally:
            finish_trace("om3")
        self.assertEqual([(c["stage"], c["unit"], c["model"], c.get("error")) for c in attrs["llm_calls"]], [
            ("outline", None, f"gemini/{BASIC_MODEL_IDS['gemini']}", "503"),            # the failed attempt is logged too
            ("outline", None, f"openai/{ADVANCED_MODEL_IDS['openai']}", None),
            ("unit", "s1", f"openai/{ADVANCED_MODEL_IDS['openai']}", None)])
        self.assertTrue(all(isinstance(c["seconds"], float) and c["seconds"] >= 0 for c in attrs["llm_calls"]))
        self.assertEqual(llm.calls, attrs["llm_calls"])

    def test_a_missing_key_for_the_fast_model_skips_it_the_last_candidate_still_raises(self):
        llm = self.llm(keys={"openai": "o"})
        self.call(llm, self.OUTLINE)
        self.assertEqual(self.factory.providers(), ["openai"])
        with self.assertRaisesRegex(ValueError, "No API key configured for openai"):
            self.call(self.llm(candidates=(OPENAI,), outline=None, keys={}), self.OUTLINE)

    def test_the_last_candidate_is_always_tried_even_if_it_failed_before(self):
        self.factory.fail = {"openai": lambda n: RuntimeError("503 overloaded") if n == 1 else None}
        llm = self.llm(candidates=(OPENAI,), outline=None)
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.call(llm, self.UNIT)
        self.call(llm, self.UNIT)                                            # second call: tried again, and works
        self.assertEqual(self.factory.runs["openai"], 2)


class TestStreamWithAFastOutline(RuntimeCase):

    def routed(self, outline_model, keys=None, **kw):
        keys = keys if keys is not None else {"anthropic_api_key": "a", "openai_api_key": "o", "gemini_api_key": "g"}
        resolver = KeyResolver({}, keys, lambda **k: k["default_keys"].get("anthropic_api_key" if k["model_provider"] == "claude"
                                                                           else k["model_provider"] + "_api_key"))
        for name, value in (("task_routing_enabled", True), ("provider_fallback_enabled", True),
                            ("blog_outline_model", outline_model)):
            p = patch.object(settings, name, value)
            p.start()
            self.addCleanup(p.stop)
        values = dict(model_provider="gemini", api_key="g", model_source="default", key_resolver=resolver)
        values.update(kw)
        return self.stream(**values)

    def providers_and_tiers(self):
        return [(c[1], c[3]["plan"].advanced_model, c[3]["instructions"].startswith("You plan")) for c in self.factory.created]

    def test_a_platform_account_gets_the_fast_outline_and_the_drafting_model_for_the_rest(self):
        out = self.routed("gemini:basic")
        created = self.providers_and_tiers()
        self.assertEqual(created[0], ("gemini", False, True))
        self.assertTrue(all(c[:2] == ("openai", True) for c in created[1:]), created)
        self.assertEqual(len(self.factory.llm.calls_for("outline")), 1)
        self.assertEqual(out[0], {"__model_used__": ("openai", ADVANCED_MODEL_IDS["openai"])})      # who writes the post

    def test_a_weak_first_outline_is_repaired_by_the_drafting_model(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        self.factory.llm.outline = [bad, good_outline({"plan": {"target_words": 1400, "images": 3}})]
        self.routed("gemini:basic")
        self.assertEqual(self.providers_and_tiers()[:2], [("gemini", False, True), ("openai", True, True)])
        self.assertEqual([c[2].stage for c in self.factory.llm.calls[:2]], ["outline", "outline_repair"])

    def test_a_byok_account_without_that_providers_key_never_reaches_it(self):
        self.routed("gemini:basic", keys={"openai_api_key": "o"})
        self.assertEqual({c[1] for c in self.factory.created}, {"openai"})

    def test_a_failing_fast_model_costs_nothing_but_time(self):
        self.factory.fail = {"gemini": RuntimeError("503 overloaded")}
        out = self.routed("gemini:basic")
        self.assertEqual(self.providers_and_tiers()[:2], [("gemini", False, True), ("openai", True, True)])
        self.assertTrue(any(isinstance(c, str) for c in out))

    def test_without_a_router_a_different_provider_is_ignored(self):
        with patch.object(settings, "blog_outline_model", "gemini:basic"):
            self.stream()
        self.assertEqual({c[1] for c in self.factory.created}, {"openai"})

    def test_without_a_router_the_same_providers_fast_tier_can_be_used(self):
        with patch.object(settings, "blog_outline_model", "openai:basic"):
            self.stream()
        self.assertEqual(self.providers_and_tiers()[0], ("openai", False, True))
        self.assertEqual(self.providers_and_tiers()[1], ("openai", True, False))

    def test_unset_is_exactly_todays_behaviour(self):
        self.stream()
        self.assertEqual({(c[1], c[3]["plan"].advanced_model) for c in self.factory.created}, {("openai", True)})

    def test_the_fast_model_is_traced(self):
        start_trace("om2", "/agent/stream")
        try:
            self.routed("gemini:basic", request_id="om2")
            attrs = get_trace("om2").attrs
        finally:
            finish_trace("om2")
        self.assertEqual(attrs["outline_model"], f"gemini/{BASIC_MODEL_IDS['gemini']}")
        self.assertEqual(attrs["model_used"], f"openai/{ADVANCED_MODEL_IDS['openai']}")


if __name__ == "__main__":
    unittest.main()
