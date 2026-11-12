"""
OpenAI's advanced tier runs on gpt-6.1-terra since 2026-11-12, and the design canvas on gpt-6.1-sol. Probed live: gpt-6.x answers 400 to
max_tokens, and to temperature / top_p unless reasoning_effort is "none"; its function tools on Chat Completions need effort "none"; Sol has no
"none". Left at the default effort, terra's tail was slow (p95 first token 8.9 s), so effort is always sent.
"""
import unittest
from unittest.mock import patch

import src.models as M
import src.text_processing_agent as tpa
from config import Settings, settings
from src.models import (REASONING_HEADROOM_TOKENS, design_model_id, openai_effort_params, openai_reasoning_family, reasoning_headroom)


class TestModelFacts(unittest.TestCase):

    def test_defaults(self):
        self.assertEqual(M.ADVANCED_MODEL_IDS["openai"], "gpt-6.1-terra")
        self.assertEqual(M.BASIC_MODEL_IDS["openai"], "gpt-4o-mini")             # unchanged: kept for keystroke latency
        self.assertEqual(design_model_id("openai"), "gpt-6.1-sol")
        self.assertEqual(design_model_id("gemini"), M.ADVANCED_MODEL_IDS["gemini"])
        self.assertEqual(design_model_id("nobody", "fallback"), "fallback")

    def test_the_reasoning_family(self):
        for model_id in ("gpt-5.2", "gpt-5", "gpt-6.1-terra", "gpt-6-luna", "gpt-5.6-luna", "o3-mini"):
            with self.subTest(model=model_id):
                self.assertTrue(openai_reasoning_family(model_id))
        for model_id in ("gpt-4o-mini", "gpt-4.1", "gemini-3.8-flash", "", None):
            with self.subTest(model=model_id):
                self.assertFalse(openai_reasoning_family(model_id))

    def test_headroom_and_limit_field_follow_the_family(self):
        self.assertEqual(reasoning_headroom("gpt-6.1-terra"), REASONING_HEADROOM_TOKENS)
        self.assertEqual(tpa._output_limit_kwargs("openai", "gpt-6.1-terra", 1000), {"max_completion_tokens": 1000 + REASONING_HEADROOM_TOKENS})
        self.assertEqual(tpa._output_limit_kwargs("openai", "gpt-4o-mini", 1000), {"max_tokens": 1000})


class TestEffortParams(unittest.TestCase):

    def test_effort_is_sent_where_the_model_has_a_none_level(self):
        for model_id in ("gpt-6.1-terra", "gpt-6-luna", "gpt-6-sol", "gpt-5.2", "gpt-5.6-luna"):
            with self.subTest(model=model_id):
                self.assertEqual(openai_effort_params(model_id, "low"), {"reasoning_effort": "low"})

    def test_models_without_it_get_nothing(self):
        for model_id in ("gpt-4o-mini", "gpt-5", "gpt-5-mini", "o3", "claude-opus-5-5", "", None):
            with self.subTest(model=model_id):
                self.assertEqual(openai_effort_params(model_id, "none"), {})
        self.assertEqual(openai_effort_params("gpt-6.1-terra", ""), {})

    def test_function_tools_force_none(self):
        self.assertEqual(openai_effort_params("gpt-6.1-terra", "high", needs_tools=True), {"reasoning_effort": "none"})

    def test_sol_cannot_do_none(self):
        self.assertEqual(openai_effort_params("gpt-6.1-sol", "none"), {"reasoning_effort": "low"})
        self.assertEqual(openai_effort_params("gpt-6.1-sol", "high"), {"reasoning_effort": "high"})


class TestSetting(unittest.TestCase):

    def effort(self, value):
        return Settings(OPENAI_ADVANCED_EFFORT=value).openai_advanced_effort

    def test_default_is_none_and_bad_values_fall_back_to_it(self):
        self.assertEqual(Settings().openai_advanced_effort, "none")
        for value in ("low", "MEDIUM", " high ", "xhigh", "none"):
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), value.strip().lower())
        for value in ("", "max", "minimal", "fast"):                # max is Responses-only; minimal is rejected everywhere
            with self.subTest(value=value):
                self.assertEqual(self.effort(value), "none")


class TestGetModel(unittest.TestCase):

    def model(self, effort="none", advanced=True, **kw):
        with patch.object(settings, "openai_advanced_effort", effort):
            return tpa.get_model("openai", "dummy", use_advanced=advanced, cache_model=False, **kw)

    def test_advanced_tier_request(self):
        m = self.model("none", temperature=0.7, max_tokens=2000)
        self.assertEqual(m.id, "gpt-6.1-terra")
        self.assertIsNone(m.temperature)                                   # a 400 unless effort is none, and the app never sent it to gpt-5.x
        params = m.get_request_params()
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertEqual(params["max_completion_tokens"], 2048 + REASONING_HEADROOM_TOKENS)
        self.assertNotIn("max_tokens", params)
        self.assertNotIn("temperature", params)

    def test_the_configured_effort_is_used(self):
        for effort in ("low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort):
                self.assertEqual(self.model(effort).get_request_params()["reasoning_effort"], effort)

    def test_tool_using_agents_are_forced_to_none(self):
        self.assertEqual(self.model("high", needs_tools=True).get_request_params()["reasoning_effort"], "none")

    def test_the_basic_tier_is_untouched(self):
        m = self.model("high", advanced=False, temperature=0.3, max_tokens=700)
        self.assertEqual(m.id, "gpt-4o-mini")
        params = m.get_request_params()
        self.assertEqual((params["temperature"], params["max_tokens"]), (0.3, 1024))
        self.assertNotIn("reasoning_effort", params)

    def test_a_change_of_effort_is_a_different_cached_model(self):
        with patch.object(settings, "openai_advanced_effort", "none"):
            a = tpa.get_model("openai", "key-openai-effort-cache", use_advanced=True)
            self.assertIs(a, tpa.get_model("openai", "key-openai-effort-cache", use_advanced=True))
        with patch.object(settings, "openai_advanced_effort", "low"):
            b = tpa.get_model("openai", "key-openai-effort-cache", use_advanced=True)
        self.assertIsNot(a, b)

    def test_tool_and_plain_models_are_cached_separately(self):
        with patch.object(settings, "openai_advanced_effort", "low"):
            plain = tpa.get_model("openai", "key-openai-tools-cache", use_advanced=True)
            tools = tpa.get_model("openai", "key-openai-tools-cache", use_advanced=True, needs_tools=True)
        self.assertEqual((plain.reasoning_effort, tools.reasoning_effort), ("low", "none"))

    def test_the_old_model_still_works_if_rolled_back(self):
        with patch.dict(M.ADVANCED_MODEL_IDS, {"openai": "gpt-5.2"}):
            params = self.model("high", max_tokens=700).get_request_params()
        self.assertNotIn("reasoning_effort", params)        # gpt-5.2's default is already "none": exactly the request it always got
        self.assertIn("max_completion_tokens", params)
        self.assertNotIn("temperature", params)


class TestPins(unittest.TestCase):
    """Blog and code agents stay on gpt-5.2 (the user's choice, 2026-11-12); everything else in the advanced tier moves to terra."""

    def agent_model(self, agent_type, provider="openai"):
        return tpa.create_agent(agent_type, provider, "dummy-key", "rid", is_byok=True).model

    def test_the_pinned_agents_keep_gpt_5_2_with_the_request_they_always_sent(self):
        for agent_type in ("blog_writer", "code_writer", "code_explainer"):
            with self.subTest(agent=agent_type):
                m = self.agent_model(agent_type)
                self.assertEqual(m.id, "gpt-5.2")
                self.assertNotIn("reasoning_effort", m.get_request_params())
                self.assertNotIn("temperature", m.get_request_params())

    def test_the_rest_of_the_advanced_tier_runs_on_terra(self):
        for agent_type in ("enhance", "email_template", "smart_global", "document_editor", "make_longer"):
            with self.subTest(agent=agent_type):
                self.assertEqual(self.agent_model(agent_type).id, "gpt-6.1-terra")

    def test_the_basic_tier_and_other_providers_are_not_pinned(self):
        self.assertEqual(self.agent_model("grammar").id, "gpt-4o-mini")
        for agent_type in ("blog_writer", "code_writer"):
            self.assertEqual(self.agent_model(agent_type, "claude").id, "claude-opus-5-5")

    def test_advanced_model_id(self):
        self.assertEqual(M.advanced_model_id("openai", "blog_writer"), "gpt-5.2")
        self.assertEqual(M.advanced_model_id("openai", "enhance"), "gpt-6.1-terra")
        self.assertEqual(M.advanced_model_id("openai"), "gpt-6.1-terra")
        self.assertEqual(M.advanced_model_id("claude", "blog_writer"), M.ADVANCED_MODEL_IDS["claude"])
        self.assertIsNone(M.advanced_model_id("nobody"))

    def test_the_model_a_blog_or_longform_call_reports_is_the_model_that_ran(self):
        """
        Found by the live before/after run (2026-11-12): the pinned blog writer really ran gpt-5.2, but the cost record, the trace and
        the model reported to the client said gpt-6.1-terra, because src.longform.llm.model_id_for ignored the pin.
        """
        from src.longform.llm import AgnoLLM, model_id_for
        from src.task_router import Candidate
        openai, claude, basic = Candidate("openai", True), Candidate("claude", True), Candidate("openai", False)
        ran = self.agent_model("blog_writer").id
        self.assertEqual(ran, "gpt-5.2")
        self.assertEqual(model_id_for(openai, agent_type="blog_writer"), ran)
        self.assertEqual(model_id_for(openai), "gpt-6.1-terra")                              # no agent type: the tier's id, as before
        self.assertEqual(model_id_for(openai, agent_type="longform_refiner"), "gpt-6.1-terra")
        self.assertEqual(model_id_for(claude, agent_type="blog_writer"), M.ADVANCED_MODEL_IDS["claude"])
        self.assertEqual(model_id_for(basic, agent_type="blog_writer"), "gpt-4o-mini")        # a pin is for the advanced tier only
        self.assertEqual(AgnoLLM([openai], lambda p: "k", request_id="r").drafting_model(), ("openai", "gpt-5.2"))           # blog is the default agent
        self.assertEqual(AgnoLLM([openai], lambda p: "k", request_id="r", agent_type="longform_refiner").drafting_model(), ("openai", "gpt-6.1-terra"))

    def test_the_cost_of_a_pinned_call_is_recorded_under_the_pinned_model(self):
        from types import SimpleNamespace
        from src.blog.llm import CallSpec
        from src.longform.llm import AgnoLLM
        from src.observability import finish_trace, start_trace
        from src.task_router import Candidate

        class FakeAgent:
            model = SimpleNamespace(id="gpt-5.2")

            def run(self, history, stream=False):
                return SimpleNamespace(content="text", status=None, metrics={"input_tokens": 100, "output_tokens": 50})

        start_trace("pin-cost", "test")
        with patch.object(tpa, "create_agent", lambda *a, **k: FakeAgent()), patch("src.email_template.service.is_failed_run", lambda r: False):
            AgnoLLM([Candidate("openai", True)], lambda p: "k", request_id="pin-cost")._run(
                Candidate("openai", True), "k", "system", [{"role": "user", "content": "hi"}], CallSpec(stage="unit", temperature=0.7, max_tokens=500, unit="s1"))
        usage = (finish_trace("pin-cost") or {}).get("usage")
        self.assertEqual([u["model"] for u in usage], ["gpt-5.2"])

    def test_pinned_and_unpinned_agents_do_not_share_a_cached_model(self):
        blog = tpa.get_model("openai", "key-pin-cache", use_advanced=True, agent_type="blog_writer")
        other = tpa.get_model("openai", "key-pin-cache", use_advanced=True, agent_type="enhance")
        self.assertEqual((blog.id, other.id), ("gpt-5.2", "gpt-6.1-terra"))
        self.assertIs(other, tpa.get_model("openai", "key-pin-cache", use_advanced=True))


class TestVoiceBackend(unittest.TestCase):

    def test_gpt_6_models_are_known_reasoners(self):
        from src.voice_flow.formatter.engine import OpenAIBackend
        self.assertTrue(OpenAIBackend(api_key="k", model="gpt-6-luna")._reasons)
        self.assertFalse(OpenAIBackend(api_key="k", model="gpt-4o-mini")._reasons)


if __name__ == "__main__":
    unittest.main()
