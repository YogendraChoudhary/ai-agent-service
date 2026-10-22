"""
type=generate_blog over HTTP: the 503 gate, request validation, the SSE sequence, warnings in `done` and in the
process response, and billing attribution. The pipeline is the real one; only the agent factory and the image
adapter are faked.
"""

import json
import re
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
import src.image_gen as ig
import src.text_processing_agent as tpa
from config import settings
from src.blog.plan import BRIEF_MAX_CHARS
from tests.blog.fakes import FakeAgentFactory, ScriptedLLM, filler

BRIEF = "Write a blog post for team leads about running remote meetings that people actually want to attend."
META = {"user_id": "u1", "requires_user_api_keys": False, "user_keys": {}}


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


class ApiCase(unittest.TestCase):

    def setUp(self):
        self.factory = FakeAgentFactory()
        self.adapter_calls = []

        def adapter(**kw):
            self.adapter_calls.append(kw)
            return {"cdn_url": f"https://cdn.test/img{len(self.adapter_calls)}.jpg", "width": 1600, "height": 900,
                    "format": "jpg", "provider": "gemini", "model": "gemini-3.1-flash-lite-image"}
        self.deductions = []
        patches = [
            patch.object(tpa, "create_agent", self.factory),
            patch("src.image_gen.generate_image_unified", adapter),
            patch.object(ig.settings, "default_gemini_api_key", "platform-gem"),
            patch.object(settings, "blog_ai_enabled", True),
            patch.object(settings, "task_routing_enabled", False),
            patch.object(settings, "provider_fallback_enabled", False),
            patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("k", META))),
            patch.object(api, "credit_manager", None),
            patch.object(api, "CREDIT_PRECHECK_ENABLED", False),
            patch.object(api, "fire_credit_deduction", lambda *a, **k: self.deductions.append((a, k))),
            patch.object(api, "fire_prompt_history_save"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def body(self, **kw):
        base = {"type": "generate_blog", "query": BRIEF, "model": "openai", "api_key": "lic"}
        base.update(kw)
        return base


class TestGateAndValidation(ApiCase):

    def test_disabled_by_default_means_503_on_both_endpoints(self):
        with patch.object(settings, "blog_ai_enabled", False):
            for path in ("/agent/process", "/agent/stream"):
                with self.subTest(path=path):
                    r = self.client.post(path, json=self.body())
                    self.assertEqual(r.status_code, 503, r.text)
                    self.assertEqual(r.json(), {"error": "AI blog generation is disabled"})
                    self.assertIn("x-request-id", r.headers)
        self.assertEqual(self.factory.created, [])

    def test_the_flag_does_not_affect_other_types(self):
        with patch.object(settings, "blog_ai_enabled", False), \
                patch.object(api, "process_user_query_with_images", lambda *a, **k: ("ok", {})):
            r = self.client.post("/agent/process", json=self.body(type="general"))
            self.assertEqual(r.status_code, 200, r.text)

    def test_blog_options_belong_to_generate_blog_only(self):
        for path in ("/agent/process", "/agent/stream"):
            with self.subTest(path=path):
                r = self.client.post(path, json=self.body(type="general", blog_options={"length": "short"}))
                self.assertEqual(r.status_code, 400)
                self.assertIn("blog_options is only valid with type generate_blog", r.json()["error"])

    def test_a_brief_is_a_topic_not_a_document(self):
        for path in ("/agent/process", "/agent/stream"):
            r = self.client.post(path, json=self.body(query="x" * (BRIEF_MAX_CHARS + 1)))
            with self.subTest(path=path):
                self.assertEqual(r.status_code, 400)
                self.assertIn("too long for a blog brief", r.json()["error"])
            self.assertEqual(self.client.post(path, json=self.body(query="  ")).status_code, 400)

    def test_bad_options_are_rejected_not_ignored(self):
        for options in ({"length": "huge"}, {"images": "many"}, {"images": 11}, {"images": -1}, {"tone": ""},
                        {"audience": "x" * 201}, {"lenght": "short"}, {"colour": "red"}):
            with self.subTest(options=options):
                self.assertEqual(self.client.post("/agent/process", json=self.body(blog_options=options)).status_code, 400)

    def test_good_options_are_accepted(self):
        for options in ({}, {"length": "short"}, {"images": "auto"}, {"images": 0}, {"images": 10},
                        {"length": "long", "images": 2, "tone": "warm", "audience": "founders"}):
            with self.subTest(options=options):
                self.assertEqual(self.client.post("/agent/process", json=self.body(blog_options=options)).status_code, 200)

    def test_the_type_is_listed_and_documented_in_the_openapi_schema(self):
        self.assertIn("generate_blog", api.VALID_AGENT_TYPES)
        schema = api.app.openapi()["components"]["schemas"]
        self.assertIn("warnings", schema["ProcessResponse"]["properties"])
        self.assertIn("blog_options", schema["StreamRequest"]["properties"])
        self.assertIn("blog_options", schema["ProcessRequest"]["properties"])


class TestProcessEndpoint(ApiCase):

    def post(self, **kw):
        return self.client.post("/agent/process", json=self.body(**kw))

    def test_returns_the_finished_post_with_real_image_urls_and_empty_warnings(self):
        r = self.post(secondary_prompt="Friendly, no jargon", blog_options={"length": "short"})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["warnings"], [])
        self.assertEqual(data["html_response"], data["markdown_response"])
        text = data["markdown_response"]
        self.assertTrue(text.startswith("# "))
        self.assertNotIn("placeholder::", text)
        self.assertEqual(len(re.findall(r"!\[[^\]]*\]\(https://cdn\.test/img\d\.jpg\)", text)), 2)
        sent = self.factory.llm.calls[0][1][0]["content"]
        self.assertIn("Friendly, no jargon", sent)
        self.assertIn('"target_words": 800', sent)

    def test_the_api_key_is_the_providers_and_the_post_is_not_metered_here(self):
        self.post()
        self.assertTrue(all(c[2] == "lic" for c in self.factory.created))        # the key the backend resolved
        self.assertTrue(all(c[3]["is_byok"] for c in self.factory.created))
        self.assertEqual(self.deductions, [])

    def test_warnings_are_returned(self):
        self.factory.llm.units = {"s2": f"{filler(240)}\n"}
        r = self.post()
        kinds = [w["kind"] for w in r.json()["warnings"]]
        self.assertIn("unit_lint", kinds)

    def test_a_post_that_cannot_be_planned_is_a_500_with_the_reason(self):
        self.factory.llm.outline = ["nope"]
        r = self.post()
        self.assertEqual(r.status_code, 500)
        self.assertIn("usable outline", r.json()["error"])

    def test_other_types_still_return_an_empty_warnings_list(self):
        with patch.object(api, "process_user_query_with_images", lambda *a, **k: ("ok", {})):
            self.assertEqual(self.post(type="general").json()["warnings"], [])


class TestStreamEndpoint(ApiCase):

    def post(self, **kw):
        return self.client.post("/agent/stream", json=self.body(**kw))

    def test_the_event_sequence(self):
        events = sse(self.post(blog_options={"length": "short"}))
        types = [e["type"] for e in events]
        self.assertEqual(types[:3], ["start", "content", "outline"])           # the title opens the post, then the skeleton
        i = types.index("done")
        self.assertEqual(types[3:i], ["content"] * 6)                           # intro, 4 sections, conclusion
        self.assertEqual(types[i + 1:], ["image_ready", "image_ready", "stream_end"])   # the short post has 2 pictures

    def test_the_old_event_sequence_is_one_flag_away_and_the_post_is_the_same(self):
        early = sse(self.post(blog_options={"length": "short"}))
        with patch.object(settings, "blog_stream_title", False):
            late = sse(self.post(blog_options={"length": "short"}))
        self.assertEqual([e["type"] for e in late][:3], ["start", "outline", "content"])
        text = lambda events: re.sub(r"placeholder::[0-9a-f-]{36}", "placeholder::ID",
                                     "".join(e["content"] for e in events if e["type"] == "content"))
        self.assertEqual(text(early), text(late))
        self.assertTrue(early[1]["content"].startswith("# "))

    def test_the_outline_event_lets_the_editor_draw_a_skeleton_at_once(self):
        outline = next(e for e in sse(self.post()) if e["type"] == "outline")
        self.assertEqual(outline["title"], "Remote meetings people actually want to join")
        self.assertEqual(len(outline["sections"]), 6)
        self.assertEqual(set(outline["sections"][0]), {"id", "heading"})

    def test_content_is_gfm_and_images_swap_in_through_the_existing_events(self):
        events = sse(self.post())
        text = "".join(e["content"] for e in events if e["type"] == "content")
        placeholders = re.findall(r"placeholder::([0-9a-f-]{36})", text)
        ready = [e for e in events if e["type"] == "image_ready"]
        self.assertEqual(sorted(e["placeholder_id"] for e in ready), sorted(placeholders))
        self.assertEqual(len(ready), 3)
        self.assertTrue(all(e["url"].startswith("https://cdn.test/") and e["width"] == 1600 for e in ready))
        self.assertIn("| Format | Best for | Watch out for |", text)
        self.assertIn("\n> ", text)

    def test_done_carries_no_warnings_key_when_clean(self):
        done = next(e for e in sse(self.post()) if e["type"] == "done")
        self.assertNotIn("warnings", done)
        self.assertEqual(done["pending_images"], 3)

    def test_done_carries_the_warnings_when_there_are_some(self):
        self.factory.llm.units = {"s2": f"{filler(240)}\n"}
        done = next(e for e in sse(self.post()) if e["type"] == "done")
        self.assertIn("unit_lint", [w["kind"] for w in done["warnings"]])

    def test_the_up_front_reservation_covers_the_post_not_just_the_brief(self):
        from src.credit_manager import text_credits_scaled
        reserved = []
        credits = type("C", (), {})()
        credits.reserve = lambda user_id, scaled: (reserved.append(scaled) or True, 10 ** 6)
        credits.schedule_release = lambda *a, **k: None
        with patch.object(api, "credit_manager", credits), patch.object(api, "CREDIT_PRECHECK_ENABLED", True), \
                patch.object(api, "fire_credit_deduction_and_release"), patch.object(api, "fire_credit_deduction"):
            self.post(blog_options={"length": "long"})
        self.assertEqual(reserved[0], text_credits_scaled(len(BRIEF), 6 * 2200))          # not text_credits_scaled(len, 0)
        self.assertGreater(reserved[0], text_credits_scaled(len(BRIEF), 0))

    def test_the_post_is_billed_as_mixed_with_the_images_that_finished(self):
        self.post()
        ((args, kwargs),) = self.deductions
        user_id, input_chars, output_chars, op_type, image_count = args
        self.assertEqual((user_id, op_type, image_count), ("u1", "mixed", 3))
        self.assertEqual(input_chars, len(BRIEF))
        self.assertGreater(output_chars, 4000)
        self.assertEqual(len(kwargs["images"]), 3)
        self.assertTrue(kwargs["success"])

    def test_a_failing_post_is_an_error_event_and_reported_to_billing_as_failed(self):
        self.factory.llm.outline = ["nope"]
        events = sse(self.post())
        self.assertEqual([e["type"] for e in events], ["start", "error"])
        self.assertIn("usable outline", events[-1]["error"])
        (args, kwargs), = self.deductions
        self.assertFalse(kwargs["success"])

    def test_pictures_started_before_a_failure_are_still_released(self):
        credits = type("Credits", (), {})()
        released = []
        credits.reserve = lambda user_id, scaled: (True, 10_000)
        credits.schedule_release = lambda user_id, scaled, delay_seconds=0: released.append(scaled)
        self.factory.llm.units = {"s3": [lambda req, spec: (_ for _ in ()).throw(RuntimeError("400 bad request"))]}
        with patch.object(api, "credit_manager", credits), patch.object(api, "CREDIT_PRECHECK_ENABLED", True), \
                patch.object(api, "fire_credit_deduction"):
            events = sse(self.post())
        self.assertEqual(events[-1]["type"], "error")
        from src.credit_manager import image_credits_scaled, text_credits_scaled
        pictures = 3 * image_credits_scaled("gemini", "gemini-3.1-flash-lite-image")
        self.assertEqual(len(released), 1)
        self.assertGreaterEqual(released[0], text_credits_scaled(len(BRIEF), 0) + pictures)   # nothing stays held

    def test_blog_options_and_style_notes_reach_the_stream_function(self):
        seen = {}

        def fake_stream(query, **kw):
            seen.update(kw, query=query)
            yield "hello"
        with patch.object(api, "process_user_query_stream", fake_stream):
            self.post(secondary_prompt="  warm  ", blog_options={"length": "long", "images": "auto"})
        self.assertEqual(seen["query"], BRIEF)                                  # no prefix: the brief is the query
        self.assertEqual(seen["instruction"], "  warm  ")                       # as the user wrote it; the plan strips it
        self.assertEqual(seen["blog_options"], {"length": "long", "images": "auto"})
        self.assertEqual(seen["force_agent_type"], "blog_writer")
        self.assertFalse(seen["image_routing"])

    def test_other_types_pass_no_blog_options(self):
        seen = {}

        def fake_stream(query, **kw):
            seen.update(kw)
            yield "hello"
        with patch.object(api, "process_user_query_stream", fake_stream):
            self.client.post("/agent/stream", json=self.body(type="general", secondary_prompt=""))
        self.assertEqual((seen["instruction"], seen["blog_options"]), (None, None))

    def test_additive_events_pass_through_untouched_for_clients_that_ignore_them(self):
        def fake_stream(query, **kw):
            yield {"__event__": {"type": "refine_progress", "chunk": 1, "of": 3}}
            yield "text"
            yield {"__warnings__": [{"kind": "x", "detail": "y"}]}
        with patch.object(api, "process_user_query_stream", fake_stream):
            events = sse(self.post())
        self.assertEqual([e["type"] for e in events], ["start", "refine_progress", "content", "done"])
        self.assertEqual(events[-1]["warnings"], [{"kind": "x", "detail": "y"}])


if __name__ == "__main__":
    unittest.main()
