"""
Deterministic request planning (fixes 2, 3, 5, 6, 7, 8 of the routing audit).

The LLM writes text and picture prompts. Code decides: how many images exist, whether web tools exist,
which model tier answers, which model draws the picture and with what settings, and what it costs.
"""

import inspect
import json
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import src.text_processing_agent as tpa
from config import settings
from src.credit_manager import IMAGE_CREDITS_SCALED, image_credits_scaled
from src.image_gen import ImageParams, plan_image_target
from src.models import BASIC_MODEL_IDS, ADVANCED_MODEL_IDS
from src.request_plan import (
    RequestPlan,
    ToolBudget,
    extract_image_count,
    image_count_requested,
    needs_advanced_model,
    needs_web,
    plan_smart_request,
)


class TestImageCount(unittest.TestCase):

    CASES = [
        ("write an article about AI with 2 images", 2),
        ("write a blog post with a hero image", 1),
        ("write a post with images", 2),                       # plural, no number: IMAGE_DEFAULT_PLURAL_COUNT
        ("include a couple of photos of the product", 2),
        ("write an essay with several images and a diagram", 4),
        ("add an infographic to the report", 1),
        ("write the history of the iphone with 3 images", 3),
        ("write about cats", 0),
        ("write an article on AI image generators", 0),         # the noun is the topic
        ("an article about image compression with 3 examples", 0),
        ("write about how to take a good photo", 0),
        ("top 5 photos of 2020", 0),
        ("", 0),
    ]

    def test_counts_come_from_the_users_words(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(image_count_requested(text), want)

    def test_count_is_capped_in_code(self):
        self.assertEqual(image_count_requested("write a guide with 10 pictures"), 10)
        self.assertEqual(extract_image_count("write a guide with 10 pictures"), settings.image_max_per_request)
        plan = plan_smart_request("write a report with six illustrations")
        self.assertEqual((plan.image_count, plan.image_requested, plan.image_capped), (4, 6, True))

    def test_none_is_safe(self):
        self.assertEqual(extract_image_count(None), 0)


class TestWebGate(unittest.TestCase):

    def test_freshness_and_research_cues_grant_web(self):
        for text in ("what is the latest news on EVs", "EV market share in 2026", "research the top competitors",
                     "summarise https://example.com/post", "current price of gold", "cite your sources"):
            with self.subTest(text=text):
                self.assertTrue(needs_web(text))

    def test_plain_writing_gets_no_web(self):
        for text in ("write a poem about autumn", "write a paragraph about our coffee shop", "draft a thank-you note"):
            with self.subTest(text=text):
                self.assertFalse(needs_web(text))

    def test_kill_switch_restores_always_on(self):
        with patch.object(settings, "web_tools_gate_enabled", False):
            self.assertTrue(needs_web("write a poem"))


class TestModelTier(unittest.TestCase):

    def test_short_conversational_writes_use_the_fast_tier(self):
        self.assertFalse(plan_smart_request("write a paragraph about our new coffee shop").advanced_model)

    def test_tools_longform_code_and_long_input_use_the_advanced_tier(self):
        for text in ("write an article about EVs", "write a python function to sort a list",
                     "write a 1,200 word story", "x" * (settings.tier_long_query_chars + 1)):
            with self.subTest(text=text[:40]):
                self.assertTrue(plan_smart_request(text).advanced_model)
        self.assertTrue(needs_advanced_model("hi", use_web=True, image_count=0))
        self.assertTrue(needs_advanced_model("hi", use_web=False, image_count=1))

    def test_kill_switch(self):
        with patch.object(settings, "model_tier_routing_enabled", False):
            self.assertTrue(plan_smart_request("write a haiku").advanced_model)


class TestToolBudget(unittest.TestCase):

    def test_grants_exactly_the_limit_even_under_threads(self):
        b = ToolBudget(3)
        granted = []
        threads = [threading.Thread(target=lambda: granted.append(b.take())) for _ in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sum(granted), 3)
        self.assertEqual((b.used, b.remaining), (3, 0))


class TestPromptBlocks(unittest.TestCase):

    def _text(self, plan):
        return "\n".join(tpa.build_smart_global_instructions(plan))

    def test_unplanned_prompt_is_the_full_original(self):
        t = self._text(None)
        self.assertIn("DECISION MAKING FOR IMAGES", t)
        self.assertIn("WEB SEARCH & PARALLEL RESEARCH", t)

    def test_planned_prompt_names_only_granted_tools(self):
        t = self._text(RequestPlan(image_count=0, use_web=False))
        self.assertNotIn("WEB SEARCH", t)
        self.assertIn("no web access", t)
        self.assertIn("Image generation is not available", t)
        self.assertNotIn("DECISION MAKING", t)

    def test_planned_prompt_fixes_the_image_count(self):
        t = self._text(RequestPlan(image_count=2, image_requested=2, use_web=True))
        self.assertIn("exactly 2 images", t)
        self.assertIn("WEB SEARCH", t)

    def test_cap_is_explained_to_the_model(self):
        t = self._text(RequestPlan(image_count=4, image_requested=6))
        self.assertIn("asked for 6 images but only 4", t)


def _agent(provider, text, *, is_byok=False, gemini_key="plat-gemini", pending=None):
    plan = plan_smart_request(text)
    ctx = {"document_id": None, "explicit_params": None, "user_query": text}
    with patch.object(settings, "default_gemini_api_key", gemini_key):
        return plan, tpa.create_agent(
            "smart_global", provider, "k", None, pending_images=pending if pending is not None else {},
            user_id="u", is_byok=is_byok, image_ctx=ctx, plan=plan,
        )


def _tool_names(agent):
    return [getattr(t, "__name__", type(t).__name__) for t in (agent.tools or [])]


class TestCreateAgentWithPlan(unittest.TestCase):

    def test_plain_request_gets_no_tools_and_the_fast_model(self):
        plan, agent = _agent("openai", "write a paragraph about our new coffee shop")
        self.assertEqual(_tool_names(agent), [])
        self.assertEqual(agent.model.id, BASIC_MODEL_IDS["openai"])

    def test_research_request_gets_web_tools_only(self):
        _, agent = _agent("claude", "write a report on the latest AI chip news")
        names = _tool_names(agent)
        self.assertIn("parallel_web_search", names)
        self.assertNotIn("generate_image_tool", names)
        self.assertEqual(agent.model.id, ADVANCED_MODEL_IDS["claude"])

    def test_single_image_request_gets_one_image_tool_and_no_web(self):
        _, agent = _agent("gemini", "write a blog post with a hero image")
        self.assertEqual(_tool_names(agent), ["generate_image_tool"])

    def test_multi_image_tool_only_when_two_or_more(self):
        _, agent = _agent("openai", "write a post with 3 images")
        self.assertEqual(_tool_names(agent), ["generate_image_tool", "generate_multiple_images"])

    def test_claude_and_deepseek_can_now_make_images(self):
        for provider in ("claude", "deepseek"):
            with self.subTest(provider=provider):
                _, agent = _agent(provider, "write a post with 2 images")
                self.assertIn("generate_image_tool", _tool_names(agent))

    def test_no_image_tool_when_no_provider_can_serve_it(self):
        # BYOK claude user, no platform key: the prompt says so, no tool exists to fabricate with
        plan, agent = _agent("claude", "write a post with 2 images", is_byok=True)
        self.assertEqual(_tool_names(agent), [])
        self.assertIn("Image generation is not available", "\n".join(agent.instructions))

    def test_tool_call_limit_is_a_backstop(self):
        _, agent = _agent("openai", "write about the latest EV news with 2 images")
        self.assertEqual(
            agent.tool_call_limit,
            settings.web_search_call_limit + settings.web_read_call_limit + 2 + 2,
        )

    def test_unplanned_call_keeps_the_old_full_toolset(self):
        with patch.object(settings, "default_gemini_api_key", None):
            agent = tpa.create_agent("smart_global", "openai", "k", None, pending_images={})
        names = _tool_names(agent)
        self.assertIn("generate_image_tool", names)
        self.assertIn("parallel_web_search", names)

    def test_other_agents_ignore_the_plan(self):
        with patch.object(settings, "default_gemini_api_key", "x"):
            agent = tpa.create_agent("grammar", "openai", "k", None, plan=RequestPlan(image_count=3, use_web=True))
        self.assertFalse(agent.tools)


class TestImageToolIsDeterministic(unittest.TestCase):

    def _tool(self, budget=None, credit_manager=None, is_byok=False, user_query="make it so", provider="openai",
              explicit=None):
        pending = {}
        ctx = {"document_id": None, "explicit_params": explicit, "user_query": user_query}
        tool = tpa._build_image_tool(
            provider, "chat-key", pending, "rid", "user1", credit_manager, is_byok, ctx, budget
        )
        return tool, pending

    def _run(self, tool, prompt="a lake", seen=None):
        def fake(**kw):
            if seen is not None:
                seen.update(kw)
            return {"cdn_url": "https://cdn/x.jpg", "width": 1, "height": 1, "format": "jpeg",
                    "provider": "gemini", "model": "gemini-3.1-flash-lite-image"}
        with patch.object(settings, "default_gemini_api_key", "plat"), \
                patch("src.image_gen.generate_image_unified", fake):
            out = tool(prompt)
            deadline = time.time() + 5
            time.sleep(0.05)
            return out

    def test_tool_signatures_have_no_llm_size_argument(self):
        tool, _ = self._tool()
        self.assertEqual(list(inspect.signature(tool).parameters), ["prompt"])
        multi = tpa.create_multi_image_tool(tool)
        self.assertEqual(list(inspect.signature(multi).parameters), ["prompts"])

    def test_budget_refuses_extra_images(self):
        tool, pending = self._tool(budget=ToolBudget(2))
        outs = [self._run(tool) for _ in range(3)]
        self.assertTrue(outs[0].startswith("Image URL:") and outs[1].startswith("Image URL:"))
        self.assertTrue(outs[2].startswith("ERROR: image limit reached"))
        self.assertEqual(len(pending), 2)

    def test_multi_tool_is_capped_by_the_budget_not_by_the_llm(self):
        budget = ToolBudget(2)
        tool, pending = self._tool(budget=budget)
        multi = tpa.create_multi_image_tool(tool, None, budget)
        with patch.object(settings, "default_gemini_api_key", "plat"), patch(
            "src.image_gen.generate_image_unified",
            lambda **kw: {"cdn_url": "u", "width": 1, "height": 1, "format": "jpeg"},
        ):
            out = multi(["a", "b", "c", "d", "e"])
        self.assertEqual(out.count("placeholder::"), 2)
        self.assertEqual(len(pending), 2)

    def test_users_words_not_the_llm_prompt_pick_model_and_pass_through(self):
        seen = {}
        tool, _ = self._tool(user_query="a post with a widescreen image using grok")
        self._run(tool, prompt="a calm lake, square crop, drawn using gemini", seen=seen)
        self.assertEqual(seen["selection_text"], "a post with a widescreen image using grok")
        p = ImageParams.from_prompt_and_overrides(seen["prompt"], seen["explicit_params"], selection_text=seen["selection_text"])
        self.assertEqual((p.provider, p.aspect_ratio), ("grok", "16:9"))

    def test_picture_settings_come_from_the_user_not_the_llm_prompt(self):
        llm = "A calm lake, 1:1 square, low quality draft"
        user = "write a blog post with a 16:9 high quality image"
        p = ImageParams.from_prompt_and_overrides(llm, None, selection_text=user)
        self.assertEqual((p.aspect_ratio, p.quality), ("16:9", "high"))
        p = ImageParams.from_prompt_and_overrides(llm, None, selection_text="")
        self.assertEqual((p.aspect_ratio, p.quality), ("1:1", "medium"))   # LLM wording ignored

    def test_reserves_the_price_of_the_model_that_will_run_and_records_it(self):
        cm = MagicMock()
        cm.reserve.return_value = (True, 100)
        with patch("src.credit_manager.IMAGE_CREDITS_BY_MODEL", {"gemini/gemini-3.1-flash-lite-image": 30}):
            tool, pending = self._tool(credit_manager=cm)
            self._run(tool)
        cm.reserve.assert_called_once_with("user1", 30)
        entry = next(iter(pending.values()))
        self.assertEqual((entry["provider"], entry["model"]), ("gemini", "gemini-3.1-flash-lite-image"))
        self.assertEqual((entry["credits_scaled"], entry["reserved_scaled"]), (30, 30))

    def test_byok_reserves_nothing(self):
        cm = MagicMock()
        tool, pending = self._tool(credit_manager=cm, is_byok=True)
        with patch("src.image_gen.generate_image_unified", lambda **kw: {"cdn_url": "u"}):
            tool("a lake")
        cm.reserve.assert_not_called()

    def test_insufficient_credits_skip_generation(self):
        cm = MagicMock()
        cm.reserve.return_value = (False, 0)
        tool, pending = self._tool(credit_manager=cm)
        with patch.object(settings, "default_gemini_api_key", "plat"):
            self.assertIn("insufficient credits", tool("a lake"))
        self.assertEqual(pending, {})

    def test_no_usable_provider_returns_an_error_not_a_dead_placeholder(self):
        tool, pending = self._tool(is_byok=True, provider="claude")
        with patch.object(settings, "default_gemini_api_key", "plat"):
            self.assertTrue(tool("a lake").startswith("ERROR:"))
        self.assertEqual(pending, {})


class TestWebBudgets(unittest.TestCase):

    def test_page_reads_are_budgeted(self):
        read = tpa.create_web_read_tool("rid", budget=ToolBudget(1))
        first = read("ftp://example.com")             # consumes the one allowed call (and is rejected on scheme)
        self.assertIn("only http and https", first)
        self.assertIn("page-read limit reached", read("https://example.com"))

    def test_parallel_search_is_budgeted_per_query(self):
        search = MagicMock()
        search.web_search.return_value = json.dumps([{"title": "t"}])
        tool = tpa.create_parallel_search_tool(search, "rid", budget=ToolBudget(2))
        first = json.loads(tool(["a", "b", "c"]))
        self.assertEqual(set(first), {"a", "b"})                       # truncated to the remaining budget
        self.assertIn("search limit reached", json.loads(tool(["d"]))["error"])


class TestCreditTable(unittest.TestCase):

    def test_default_is_the_flat_price(self):
        self.assertEqual(image_credits_scaled("gemini", "gemini-2.5-flash-image"), IMAGE_CREDITS_SCALED)

    def test_model_then_provider_then_default(self):
        table = {"openai/gpt-image-1": 80, "grok": 60}
        with patch("src.credit_manager.IMAGE_CREDITS_BY_MODEL", table):
            self.assertEqual(image_credits_scaled("openai", "gpt-image-1"), 80)
            self.assertEqual(image_credits_scaled("grok", "grok-imagine-image"), 60)
            self.assertEqual(image_credits_scaled("gemini", "x"), IMAGE_CREDITS_SCALED)

    def test_env_override_and_bad_json(self):
        with patch.object(settings, "image_credits_by_model", '{"openai/gpt-image-1": 90}'):
            self.assertEqual(image_credits_scaled("openai", "gpt-image-1"), 90)
        with patch.object(settings, "image_credits_by_model", "not json"):
            self.assertEqual(image_credits_scaled("openai", "gpt-image-1"), IMAGE_CREDITS_SCALED)

    def test_plan_image_target_matches_what_generation_will_use(self):
        with patch.object(settings, "default_gemini_api_key", "plat"):
            self.assertEqual(plan_image_target("openai", "k", False, None, None, "a cat"),
                             ("gemini", "gemini-3.1-flash-lite-image"))
            self.assertEqual(plan_image_target("openai", "k", False, None, None, "a cat using gemini 3.1"),
                             ("gemini", "gemini-3.1-flash-image"))
            self.assertEqual(plan_image_target("openai", "k", True, None, None, "a cat")[0], "openai")
            self.assertIsNone(plan_image_target("claude", "k", True, None, None, "a cat"))


class TestProcessPipeline(unittest.TestCase):
    """/agent/process goes through the stream pipeline and never returns dead placeholders."""

    def _pending(self):
        return {
            "__completion_callback__": None,
            "ok": {"status": "done", "cdn_url": "https://cdn/ok.jpg", "provider": "gemini",
                   "model": "gemini-2.5-flash-image", "credits_scaled": 50, "reserved_scaled": 50},
            "bad": {"status": "failed", "cdn_url": None, "error": "boom", "reserved_scaled": 50},
        }

    def test_placeholders_become_urls_and_failures_become_a_note(self):
        text = "Intro ![lake](placeholder::ok) middle ![river](placeholder::bad) end"
        out = tpa.resolve_image_placeholders(text, self._pending())
        self.assertIn("![lake](https://cdn/ok.jpg)", out)
        self.assertIn(tpa.IMAGE_FAILED_NOTE, out)
        self.assertNotIn("placeholder::", out)

    def test_bare_and_unknown_placeholders_never_survive(self):
        out = tpa.resolve_image_placeholders("see placeholder::ok and placeholder::ghost", self._pending())
        self.assertNotIn("placeholder::", out)
        self.assertIn("https://cdn/ok.jpg", out)

    def test_pipeline_waits_for_background_images(self):
        pending = {"__completion_callback__": None,
                   "p1": {"status": "generating", "cdn_url": None, "reserved_scaled": 0}}

        def fake_stream(query, **kw):
            yield "Here: ![x](placeholder::p1)"
            yield {"__pending_images__": pending}
            def finish():
                time.sleep(0.2)
                pending["p1"].update(status="done", cdn_url="https://cdn/late.jpg")
                kw["image_completion_callback"]()
            threading.Thread(target=finish, daemon=True).start()

        with patch.object(tpa, "process_user_query_stream", fake_stream):
            text, p = tpa.process_user_query_with_images("q", api_key="k", wait_seconds=5)
        self.assertEqual(text, "Here: ![x](https://cdn/late.jpg)")

    def test_timeout_replaces_unfinished_images(self):
        pending = {"p1": {"status": "generating", "cdn_url": None}}

        def fake_stream(query, **kw):
            yield "![x](placeholder::p1)"
            yield {"__pending_images__": pending}

        with patch.object(tpa, "process_user_query_stream", fake_stream):
            text, _ = tpa.process_user_query_with_images("q", api_key="k", wait_seconds=0.3)
        self.assertEqual(text, tpa.IMAGE_FAILED_NOTE)

    def test_process_user_query_returns_plain_text(self):
        def fake_stream(query, **kw):
            yield "hello"
            yield " world"

        with patch.object(tpa, "process_user_query_stream", fake_stream):
            self.assertEqual(tpa.process_user_query("q", api_key="k"), "hello world")

    def test_same_routing_arguments_as_the_stream_endpoint(self):
        seen = {}

        def fake_stream(query, **kw):
            seen.update(kw)
            yield "x"

        with patch.object(tpa, "process_user_query_stream", fake_stream):
            tpa.process_user_query_with_images(
                "q", api_key="k", user_id="u", image_routing=True, image_params={"provider": "grok"}
            )
        self.assertTrue(seen["image_routing"])
        self.assertEqual(seen["image_params"], {"provider": "grok"})
        self.assertEqual(seen["user_id"], "u")


class TestBillingHelpersAndEndpoint(unittest.TestCase):

    def test_image_billing_releases_every_reservation_but_reports_only_finished_images(self):
        import api
        pending = {
            "__completion_callback__": None,
            "a": {"status": "done", "provider": "gemini", "model": "m", "credits_scaled": 50, "reserved_scaled": 50},
            "b": {"status": "failed", "reserved_scaled": 50},
        }
        reserved, details = api.image_billing(pending)
        self.assertEqual(reserved, 100)
        self.assertEqual(details, [{"provider": "gemini", "model": "m", "credits_scaled": 50}])

    def test_image_only_requests_are_attributed_to_the_image_model(self):
        import api
        d = [{"provider": "gemini", "model": "gemini-2.5-flash-image", "credits_scaled": 50}]
        self.assertEqual(api.billed_provider_model("claude", "image", d), ("gemini", "gemini-2.5-flash-image"))
        self.assertEqual(api.billed_provider_model("claude", "mixed", d), ("claude", None))
        self.assertEqual(api.billed_provider_model("claude", "image", []), ("claude", None))

    def test_process_endpoint_runs_the_shared_pipeline_without_agent_billing(self):
        # /agent/process is called by the backend (apps/backend agentService.ts) with the
        # provider key it already resolved; the backend then bills text AND images by
        # counting them in the returned HTML (creditDeductionService.ts). So the agent must
        # not validate a licence, attach platform keys, or deduct credits itself.
        import api
        from fastapi.testclient import TestClient

        pending = {"__completion_callback__": None,
                   "a": {"status": "done", "provider": "gemini", "model": "gemini-2.5-flash-image",
                         "credits_scaled": 50, "reserved_scaled": 50}}
        captured = {}

        def fake_pipeline(query, **kw):
            captured.update(kw, query=query)
            return "![x](https://cdn/a.jpg)", pending

        # Mirrors the real resolver: with no user keys, only default_keys could yield a key.
        def resolve(**k):
            return "PLATFORM-KEY" if k.get("default_keys") else None

        validate = AsyncMock()
        with patch.object(api, "async_validate_and_get_api_key", validate), \
                patch.object(api, "process_user_query_with_images", fake_pipeline), \
                patch.object(api, "fire_credit_deduction") as fire, \
                patch.object(api, "fire_credit_deduction_and_release") as fire_release, \
                patch.object(api, "redis_manager", MagicMock(resolve_api_key_from_user_keys=resolve)):
            client = TestClient(api.app, raise_server_exceptions=False)
            r = client.post("/agent/process", json={
                "type": "general", "query": "draw a cat using grok", "model": "claude", "api_key": "sk-provider",
                "image_params": {"quality": "high"},
            })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["markdown_response"], "![x](https://cdn/a.jpg)")
        validate.assert_not_called()
        self.assertEqual(captured["api_key"], "sk-provider")
        # same routing inputs as /agent/stream: direct image route on for general chat, regex-chosen provider passed
        self.assertTrue(captured["image_routing"])
        self.assertEqual(captured["image_params"]["provider"], "grok")
        self.assertEqual(captured["image_params"]["quality"], "high")
        self.assertNotIn("api_key", captured["image_params"])   # never a platform key
        self.assertTrue(captured["is_byok"])
        self.assertIsNone(captured["user_id"])
        self.assertIsNone(captured["credit_manager"])
        fire.assert_not_called()
        fire_release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
