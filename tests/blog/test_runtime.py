"""
The pipeline bound to this app, through the UNMODIFIED production paths: process_user_query_stream's blog hook,
runtime.stream_blog, AgnoLLM (provider candidates and fallback), the real image tool (pending_images shape,
credit reservation) and create_agent. Only the network edges are faked: the agent factory and the image adapter.
"""

import re
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.image_gen as ig
import src.text_processing_agent as tpa
from config import settings
from src.blog.media import IMAGE_STYLE_SUFFIX
from src.models import ADVANCED_MODEL_IDS
from src.observability import finish_trace, get_trace, start_trace
from src.task_router import KeyResolver
from tests.blog.fakes import FakeAgentFactory, ScriptedLLM, good_body

BRIEF = "Write a blog post for team leads about running remote meetings that people actually want to attend."


class FakeCredits:
    def __init__(self, ok=True):
        self.ok, self.reserved = ok, []

    def reserve(self, user_id, scaled):
        self.reserved.append((user_id, scaled))
        return (self.ok, 1000)


class RuntimeCase(unittest.TestCase):

    def setUp(self):
        self.factory = FakeAgentFactory()
        self.adapter_calls = []

        def adapter(**kw):
            self.adapter_calls.append(kw)
            n = len(self.adapter_calls)
            return {"cdn_url": f"https://cdn.test/img{n}.jpg", "width": 1600, "height": 900, "format": "jpg",
                    "provider": "gemini", "model": "gemini-3.1-flash-lite-image"}
        for p in (patch.object(tpa, "create_agent", self.factory), patch("src.image_gen.generate_image_unified", adapter),
                  patch.object(ig.settings, "default_gemini_api_key", "platform-gem"), patch("time.sleep")):
            p.start()
            self.addCleanup(p.stop)

    def stream(self, query=BRIEF, **kw):
        kw.setdefault("model_provider", "openai")
        kw.setdefault("api_key", "user-key")
        kw.setdefault("user_id", "u1")
        kw.setdefault("force_agent_type", "blog_writer")
        kw.setdefault("report_model", True)
        return list(tpa.process_user_query_stream(query, **kw))

    @staticmethod
    def shape(out):
        def one(c):
            if isinstance(c, str):
                return "text"
            return next(iter(c)).strip("_") + (":" + c["__event__"]["type"] if "__event__" in c else "")
        return [one(c) for c in out]

    @staticmethod
    def document(out):
        return "".join(c for c in out if isinstance(c, str))

    @staticmethod
    def pending(out):
        return next(c["__pending_images__"] for c in out if isinstance(c, dict) and "__pending_images__" in c)


class TestStreamHook(RuntimeCase):

    def test_the_chunk_order_the_stream_endpoint_relies_on(self):
        out = self.stream()
        # the title opens the post while the rest of the outline is still being written; pictures start after the outline event
        self.assertEqual(self.shape(out), ["model_used", "text", "event:outline", "pending_images"] + ["text"] * 7)

    def test_the_old_chunk_order_is_one_flag_away(self):
        with patch.object(settings, "blog_stream_title", False):
            out = self.stream()
        self.assertEqual(self.shape(out), ["model_used", "event:outline", "pending_images"] + ["text"] * 8)

    def test_only_the_outline_call_is_streamed(self):
        self.stream()
        self.assertEqual(self.factory.streamed, [True] + [False] * 7)             # the outline; then intro, 5 sections, conclusion

    def test_the_post_is_the_same_with_the_title_early_or_not(self):
        pictures = lambda text: re.sub(r"placeholder::[0-9a-f-]{36}", "placeholder::ID", text)     # each run draws its own ids
        early = pictures(self.document(self.stream()))
        with patch.object(settings, "blog_stream_title", False):
            late = pictures(self.document(self.stream()))
        self.assertEqual(early, late)

    def test_the_pictures_dict_is_handed_over_the_moment_pictures_exist_so_an_abort_can_still_release_it(self):
        gen = tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                            force_agent_type="blog_writer", credit_manager=FakeCredits())
        seen = []
        for chunk in gen:
            seen.append(chunk)
            if isinstance(chunk, dict) and "__pending_images__" in chunk:
                break                                       # the client leaves as soon as it is told about the pictures
        gen.close()
        pending = next(c["__pending_images__"] for c in seen if isinstance(c, dict) and "__pending_images__" in c)
        self.assertEqual(len([k for k in pending if not k.startswith("__")]), 3)
        self.assertEqual(len([c for c in seen if isinstance(c, str)]), 1)             # only the title had been sent

    def test_the_pictures_dict_arrives_before_a_single_part_is_written(self):
        started = []

        def intro(req, spec):
            started.append(spec.unit)
            return good_body(req)
        self.factory.llm = ScriptedLLM(units={"intro": intro})
        gen = tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                            force_agent_type="blog_writer", credit_manager=FakeCredits())
        for chunk in gen:
            if isinstance(chunk, dict) and "__pending_images__" in chunk:
                break                                       # not "with the first part": the next chunk may be a part away
        gen.close()
        time.sleep(0.2)                                     # a part that had been started would have called the model by now
        self.assertEqual(started, [])

    def test_without_the_early_title_the_dict_is_handed_over_before_any_text(self):
        with patch.object(settings, "blog_stream_title", False):
            gen = tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                                force_agent_type="blog_writer", credit_manager=FakeCredits())
            seen = []
            for chunk in gen:
                seen.append(chunk)
                if isinstance(chunk, str):
                    break                                   # the client leaves right after the title
            gen.close()
        pending = next(c["__pending_images__"] for c in seen if isinstance(c, dict) and "__pending_images__" in c)
        self.assertEqual(len([k for k in pending if not k.startswith("__")]), 3)

    def test_a_client_that_leaves_after_the_title_never_starts_or_pays_for_a_picture(self):
        credits = FakeCredits()
        gen = tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                            force_agent_type="blog_writer", credit_manager=credits)
        for chunk in gen:
            if isinstance(chunk, str):
                break
        gen.close()
        self.assertEqual((self.adapter_calls, credits.reserved), ([], []))

    def test_without_report_model_that_chunk_is_absent(self):
        self.assertNotIn("model_used", self.shape(self.stream(report_model=False)))

    def test_the_model_reported_is_the_one_that_answered(self):
        out = self.stream()
        self.assertEqual(out[0], {"__model_used__": ("openai", ADVANCED_MODEL_IDS["openai"])})

    def test_the_agent_is_built_per_call_with_the_pipelines_prompt_and_budget(self):
        self.stream()
        outline, first_part = self.factory.created[0], self.factory.created[1]
        for agent_type, provider, key, kw in (outline, first_part):
            self.assertEqual((agent_type, provider, key), ("blog_writer", "openai", "user-key"))
        self.assertTrue(outline[3]["instructions"].startswith("You plan editorial blog posts"))
        self.assertEqual((outline[3]["temperature"], outline[3]["max_tokens"]), (0.4, 2050))
        self.assertEqual(first_part[3]["temperature"], 0.7)
        self.assertFalse(outline[3]["plan"].use_web)
        self.assertTrue(outline[3]["plan"].advanced_model)

    def test_style_notes_and_options_reach_the_model(self):
        self.stream(instruction="Friendly, no jargon", blog_options={"length": "short", "audience": "founders", "tone": "warm"})
        sent = self.factory.llm.calls[0][1][0]["content"]
        for fragment in ("Friendly, no jargon", "founders", "warm", '"target_words": 800', '"sections": 4'):
            self.assertIn(fragment, sent)
        self.assertEqual(len(self.factory.llm.calls_for("unit")), 6)       # intro + 4 sections + conclusion

    def test_the_pipeline_runs_in_a_fresh_thread_pool_not_the_tool_pool(self):
        names = set()

        def spy(req, spec):
            names.add(threading_name())
            return good_body(req)
        import threading
        threading_name = lambda: threading.current_thread().name
        self.factory.llm.units = {u: [spy] for u in ("intro", "s1", "s2")}
        self.stream()
        self.assertTrue(all(n.startswith("blog_unit") for n in names), names)


class TestPictures(RuntimeCase):

    def test_pictures_are_registered_in_the_shared_pending_dict_in_the_chat_path_shape(self):
        pending = self.pending(self.stream(credit_manager=FakeCredits()))
        entries = {k: v for k, v in pending.items() if not k.startswith("__")}
        self.assertEqual(len(entries), 3)
        for pid, entry in entries.items():
            self.assertTrue(re.fullmatch(r"[0-9a-f-]{36}", pid))
            self.assertEqual({"status", "cdn_url", "error", "prompt", "width", "height", "format", "provider", "model",
                              "credits_scaled", "reserved_scaled"} - set(entry), set())
            self.assertTrue(entry["prompt"].endswith(IMAGE_STYLE_SUFFIX))
            self.assertEqual(entry["provider"], "gemini")

    def test_the_text_references_exactly_the_pictures_that_were_started(self):
        out = self.stream()
        pending = self.pending(out)
        placeholders = set(re.findall(r"placeholder::([0-9a-f-]{36})", self.document(out)))
        self.assertEqual(placeholders, {k for k in pending if not k.startswith("__")})

    def test_pictures_are_drawn_wide_at_medium_quality_from_the_users_own_words(self):
        out = self.stream()
        self.pending(out)
        deadline = time.time() + 2
        while len(self.adapter_calls) < 3 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.adapter_calls), 3)
        for call in self.adapter_calls:
            self.assertEqual(call["explicit_params"], {"aspect_ratio": "16:9", "quality": "medium"})
            self.assertEqual(call["selection_text"], BRIEF)          # the brief, never the model's picture prompt
            self.assertEqual(call["user_id"], "u1")

    def test_explicit_image_params_win_over_the_blog_defaults(self):
        self.stream(image_params={"aspect_ratio": "1:1", "quality": "high"})
        deadline = time.time() + 2
        while not self.adapter_calls and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.adapter_calls[0]["explicit_params"], {"aspect_ratio": "1:1", "quality": "high"})

    def test_credits_are_reserved_per_picture_for_platform_keys_only(self):
        credits = FakeCredits()
        self.stream(credit_manager=credits)
        self.assertEqual(len(credits.reserved), 3)
        self.assertTrue(all(u == "u1" and n > 0 for u, n in credits.reserved))
        byok = FakeCredits()
        self.stream(credit_manager=byok, is_byok=True)
        self.assertEqual(byok.reserved, [])

    def test_a_picture_that_cannot_be_paid_for_is_skipped_with_a_warning(self):
        out = self.stream(credit_manager=FakeCredits(ok=False))
        warnings = next(c["__warnings__"] for c in out if isinstance(c, dict) and "__warnings__" in c)
        self.assertEqual([w["kind"] for w in warnings].count("image_skipped"), 3)
        self.assertIn("insufficient credits", warnings[0]["detail"])
        self.assertNotIn("placeholder::", self.document(out))
        self.assertNotIn("pending_images", self.shape(out))             # nothing started: nothing to settle

    def test_an_account_that_cannot_draw_gets_a_post_without_pictures(self):
        out = self.stream(model_provider="claude", api_key="claude-key", is_byok=True)
        self.assertNotIn("placeholder::", self.document(out))
        self.assertNotIn("pending_images", self.shape(out))
        self.assertEqual(self.adapter_calls, [])

    def test_a_platform_account_on_claude_draws_with_the_platform_image_key(self):
        out = self.stream(model_provider="claude", api_key="claude-key", is_byok=False)
        self.assertEqual(len([k for k in self.pending(out) if not k.startswith("__")]), 3)


class TestProviders(RuntimeCase):

    def routed(self, **kw):
        keys = KeyResolver({}, {"anthropic_api_key": "a", "openai_api_key": "o", "gemini_api_key": "g"},
                           lambda **k: k["default_keys"].get("anthropic_api_key" if k["model_provider"] == "claude"
                                                             else k["model_provider"] + "_api_key"))
        for name in ("task_routing_enabled", "provider_fallback_enabled"):
            p = patch.object(settings, name, True)
            p.start()
            self.addCleanup(p.stop)
        values = dict(model_provider="gemini", api_key="g", model_source="default", key_resolver=keys)
        values.update(kw)
        return self.stream(**values)

    def test_a_default_blog_request_goes_to_the_writing_provider_first(self):
        out = self.routed()
        self.assertEqual(self.factory.providers()[0], "openai")
        self.assertEqual(out[0]["__model_used__"][0], "openai")

    def test_a_provider_that_is_down_moves_the_call_on_and_the_next_calls_do_not_retry_it(self):
        self.factory.fail = {"openai": RuntimeError("503 overloaded")}
        out = self.routed()
        self.assertEqual(out[0]["__model_used__"][0], "gemini")
        self.assertEqual(self.factory.runs["openai"], 1)               # tried once, then remembered
        self.assertGreaterEqual(self.factory.runs["gemini"], 8)
        self.assertEqual(len(self.factory.llm.calls_for("unit")), 7)

    def test_a_dead_key_falls_back_without_a_retry(self):
        self.factory.fail = {"openai": RuntimeError("Error code: 401 - invalid x-api-key")}
        out = self.routed()
        self.assertEqual(self.factory.runs["openai"], 1)
        self.assertEqual(out[0]["__model_used__"][0], "gemini")

    def test_a_rate_limit_is_retried_once_on_the_same_provider(self):
        self.factory.fail = {"openai": lambda n: RuntimeError("429 rate limit") if n == 1 else None}
        out = self.routed()
        self.assertEqual(out[0]["__model_used__"][0], "openai")
        self.assertEqual(self.factory.providers()[:2], ["openai", "openai"])

    def test_a_bad_request_never_falls_back(self):
        self.factory.fail = {"openai": RuntimeError("400 bad request")}
        with self.assertRaisesRegex(RuntimeError, "400"):
            self.routed()
        self.assertEqual(set(self.factory.providers()), {"openai"})

    def test_every_provider_down_raises_the_last_error(self):
        self.factory.fail = {p: RuntimeError("503 overloaded") for p in ("openai", "gemini", "claude")}
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.routed()

    def test_the_fallback_is_traced(self):
        self.factory.fail = {"openai": RuntimeError("503 overloaded")}
        start_trace("bfb", "/agent/stream")
        try:
            self.routed(request_id="bfb")
            attrs = get_trace("bfb").attrs
        finally:
            finish_trace("bfb")
        self.assertEqual(attrs["fallbacks"], [{"from": "openai", "to": "gemini", "error": "503"}])
        self.assertEqual(attrs["model_used"], f"gemini/{ADVANCED_MODEL_IDS['gemini']}")
        self.assertEqual(attrs["task"], "writing")


class TestFailureAfterPicturesStarted(RuntimeCase):

    def test_the_caller_is_told_about_started_pictures_before_the_error(self):
        """Pictures are billed work: when a part fails, the stream endpoint still needs the dict to release the credits."""
        self.factory.llm.units = {"s3": [lambda req, spec: (_ for _ in ()).throw(RuntimeError("400 bad request"))]}
        gen = tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                            force_agent_type="blog_writer", credit_manager=FakeCredits())
        seen = []
        with self.assertRaisesRegex(RuntimeError, "400"):
            for chunk in gen:
                seen.append(chunk)
        pending = [c for c in seen if isinstance(c, dict) and "__pending_images__" in c]
        self.assertEqual(len(pending), 1)
        self.assertEqual(len([k for k in pending[0]["__pending_images__"] if not k.startswith("__")]), 3)

    def test_no_pictures_started_means_no_pending_chunk_on_failure(self):
        self.factory.llm.outline = ["nope"]
        seen = []
        with self.assertRaises(Exception):
            for chunk in tpa.process_user_query_stream(BRIEF, model_provider="openai", api_key="k", user_id="u1",
                                                       force_agent_type="blog_writer"):
                seen.append(chunk)
        self.assertEqual([c for c in seen if isinstance(c, dict) and "__pending_images__" in c], [])


class TestNonStreaming(RuntimeCase):

    def test_images_are_awaited_and_the_text_has_real_urls_and_the_warnings_come_back(self):
        meta = {}
        text, pending = tpa.process_user_query_with_images(
            BRIEF, model_provider="openai", api_key="k", force_agent_type="blog_writer", is_byok=True,
            wait_seconds=3, instruction="notes", blog_options={"length": "short"}, meta=meta)
        self.assertNotIn("placeholder::", text)
        self.assertEqual(len(re.findall(r"!\[[^\]]*\]\(https://cdn\.test/img\d\.jpg\)", text)), 2)      # short = 2 images
        self.assertEqual(meta.get("warnings", []), [])
        self.assertIn("notes", self.factory.llm.calls[0][1][0]["content"])

    def test_a_failed_picture_becomes_the_existing_note_never_a_dead_link(self):
        def broken(**kw):
            raise RuntimeError("image provider down")
        with patch("src.image_gen.generate_image_unified", broken):
            text, _ = tpa.process_user_query_with_images(BRIEF, model_provider="openai", api_key="k",
                                                         force_agent_type="blog_writer", is_byok=True, wait_seconds=3)
        self.assertIn(tpa.IMAGE_FAILED_NOTE, text)
        self.assertNotIn("placeholder::", text)

    def test_meta_is_optional(self):
        text, _ = tpa.process_user_query_with_images(BRIEF, model_provider="openai", api_key="k",
                                                     force_agent_type="blog_writer", is_byok=True, wait_seconds=3)
        self.assertTrue(text.startswith("# "))


class TestPromotionFromGeneralChat(RuntimeCase):
    """BLOG_INTENT_*: ships off; shadow logs only; on routes "write a blog post about ..." to the pipeline."""

    PROMPT = "Write a blog post about remote team meetings"

    def general(self, query=None, **kw):
        values = dict(force_agent_type=tpa.FREE_FORM_AGENT, image_routing=True, report_model=False)   # what the API passes for "general"
        values.update(kw)
        return self.stream(query or self.PROMPT, **values)

    def with_flags(self, **flags):
        for name, value in flags.items():
            p = patch.object(settings, name, value)
            p.start()
            self.addCleanup(p.stop)

    def reaches_blog(self, query=None, **kw):
        self.factory.created.clear()
        try:
            self.general(query, **kw)
        except Exception:
            pass
        return any(c[0] == "blog_writer" for c in self.factory.created)

    def test_off_by_default(self):
        self.assertFalse(self.reaches_blog())

    def test_on_routes_the_request_to_the_blog_pipeline_as_writing(self):
        self.with_flags(blog_ai_enabled=True, blog_intent_regex_enabled=True)
        start_trace("promo", "/agent/stream")
        try:
            self.assertTrue(self.reaches_blog(request_id="promo"))
            attrs = get_trace("promo").attrs
        finally:
            finish_trace("promo")
        self.assertEqual((attrs["task"], attrs["intent_rule"], attrs["blog_intent"]), ("writing", "anchored_blog", "anchored_blog"))

    def test_shadow_mode_only_logs_the_decision(self):
        self.with_flags(blog_ai_enabled=True, blog_intent_shadow_mode=True, blog_intent_regex_enabled=True)
        with self.assertLogs("src.text_processing_agent", level="INFO") as logs:
            self.assertFalse(self.reaches_blog())
        self.assertTrue(any("[BLOG SHADOW]" in line for line in logs.output))

    def test_never_when_the_blog_feature_itself_is_off(self):
        self.with_flags(blog_ai_enabled=False, blog_intent_regex_enabled=True)
        self.assertFalse(self.reaches_blog())

    def test_a_blog_post_about_code_is_content_not_code(self):
        # code_intent treats "write a blog post about <code topic>" as mixed content, so the two never collide
        self.with_flags(blog_ai_enabled=True, blog_intent_regex_enabled=True)
        self.assertTrue(self.reaches_blog("Write a blog post about Python functions"))

    def test_code_intent_and_reference_images_win(self):
        self.with_flags(blog_ai_enabled=True, blog_intent_regex_enabled=True)
        self.assertFalse(self.reaches_blog(self.PROMPT, image_urls=["https://cdn.test/a.png"]))
        with patch("src.code_intent.detect_code_intent", lambda q: {"kind": "code", "rule": "anchored_code"}):
            self.assertFalse(self.reaches_blog())              # code intent is decided first, as for images

    def test_only_the_free_form_chat_is_ever_promoted(self):
        self.with_flags(blog_ai_enabled=True, blog_intent_regex_enabled=True)
        for typed in ("grammar", "make_longer", "email_template", "code_writer"):
            with self.subTest(agent=typed):
                self.assertFalse(self.reaches_blog(force_agent_type=typed))


class TestAgentFactory(unittest.TestCase):
    """The real create_agent for the blog writer: per-call prompt, temperature and output budget, no Markdown nudge."""

    def agent(self, **kw):
        values = dict(plan=SimpleNamespace(use_web=False, advanced_model=True, image_count=0, use_tools=False,
                                           reasons={}), instructions="SYSTEM", temperature=0.4, max_tokens=900)
        values.update(kw)
        return tpa.create_agent("blog_writer", "openai", "k", "rid", is_byok=True, **values)

    def test_overrides_reach_the_agent_and_the_model(self):
        agent = self.agent()
        self.assertEqual(agent.instructions, "SYSTEM")
        self.assertFalse(agent.markdown)
        self.assertEqual(agent.model.max_completion_tokens, 1024 + 4096)      # gpt-5.x: visible + reasoning headroom
        self.assertFalse(getattr(agent, "tools", None))

    def test_the_date_is_given_so_research_can_resolve_relative_dates(self):
        self.assertTrue(self.agent().add_datetime_to_context)

    def test_web_tools_only_when_the_plan_grants_them_and_they_share_budgets(self):
        from src.request_plan import RequestPlan
        ctx = {}
        plan = RequestPlan(use_web=True, advanced_model=True)
        a = tpa.create_agent("blog_writer", "openai", "k", "rid", is_byok=True, image_ctx=ctx, plan=plan, instructions="S")
        self.assertEqual(sorted(t.__name__ for t in a.tools), ["parallel_web_search", "read_multiple_web_pages", "read_web_page"])
        budgets = ctx["web_budgets"]
        tpa.create_agent("blog_writer", "openai", "k", "rid", is_byok=True, image_ctx=ctx, plan=plan, instructions="S")
        self.assertIs(ctx["web_budgets"], budgets)                              # one budget for every part of the post

    def test_never_an_image_tool(self):
        from src.request_plan import RequestPlan
        a = tpa.create_agent("blog_writer", "openai", "k", "rid", is_byok=True, plan=RequestPlan(use_web=True, advanced_model=True),
                             instructions="S")
        self.assertNotIn("generate_image_tool", [t.__name__ for t in a.tools])

    def test_without_a_plan_the_blog_writer_still_gets_no_tools(self):
        a = tpa.create_agent("blog_writer", "openai", "k", "rid", is_byok=True, instructions="S")
        self.assertFalse(getattr(a, "tools", None))

    def test_other_agents_keep_their_own_prompt_and_temperature(self):
        a = tpa.create_agent("grammar", "openai", "k", "rid", is_byok=True)
        self.assertIn("grammar", a.instructions[0].lower())
        self.assertTrue(a.markdown)


class TestReaderRecordsWhatItReads(unittest.TestCase):
    """create_web_read_tool(on_read): the only way the lint can know a page was really read (rule D13)."""

    def setUp(self):
        p = patch.object(tpa, "_is_publicly_routable", lambda host: True)
        p.start()
        self.addCleanup(p.stop)

    def make(self, on_read, budget=None, html="<html><body><article><p>Hello page.</p></article></body></html>", fail=False):
        def get(url):
            if fail:
                raise RuntimeError("boom")
            return SimpleNamespace(text=html, raise_for_status=lambda: None)
        p = patch.object(tpa, "get_shared_web_client", lambda: SimpleNamespace(get=get))
        p.start()
        self.addCleanup(p.stop)
        return tpa.create_web_read_tool("rid", budget=budget, on_read=on_read)

    def test_a_page_that_was_read_is_reported(self):
        seen = []
        out = self.make(seen.append)("https://example.com/a")
        self.assertEqual(seen, ["https://example.com/a"])
        self.assertIn("Hello page", out)

    def test_errors_and_refused_reads_are_not_reports(self):
        seen = []
        self.assertTrue(self.make(seen.append, fail=True)("https://example.com/a").startswith("ERROR"))
        from src.request_plan import ToolBudget
        self.assertTrue(self.make(seen.append, budget=ToolBudget(0))("https://example.com/a").startswith("ERROR"))
        self.assertTrue(self.make(seen.append, html="<html><body></body></html>")("https://example.com/a").startswith("ERROR"))
        self.assertEqual(seen, [])

    def test_the_callback_is_optional(self):
        self.assertIn("Hello page", self.make(None)("https://example.com/a"))


if __name__ == "__main__":
    unittest.main()
