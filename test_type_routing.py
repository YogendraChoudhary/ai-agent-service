"""
The declared request `type` decides the agent. User text never does.

Before this change the agent was guessed from substrings of the prefix plus the user's own content, so
9 of 14 realistic requests in the audit went to the wrong agent. `general` is always the free-form agent.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import api
import src.text_processing_agent as tpa
from src.text_processing_agent import (
    AGENT_CONFIGS,
    AGENT_FOR_REQUEST_TYPE,
    FREE_FORM_AGENT,
    agent_for_request_type,
    determine_agent_type,
    process_user_query_stream,
)
from test_image_routing import RouteTestCase, AgentPathReached

# (declared type, user text that the old keyword router read as a DIFFERENT operation)
HIJACK_MATRIX = [
    ("simplify",        "Please fix grammar issues in my contract draft before sending"),
    ("make_longer",     "Simplify the onboarding steps for new hires"),
    ("emojify",         "Here is a summary of our quarterly results"),
    ("fix_grammar",     "I want to translate this to French"),
    ("complete",        "Summarize the meeting notes: we agreed to ship"),
    ("make_shorter",    "Translate the menu to Spanish for our guests"),
    ("change_tone",     "Please expand on the summary section"),
    ("enhance_with_ai", "I need to make a table format for prices"),
    ("translate",       "make longer and add emoji and a bullet points list"),
    ("general",         "Write a complete guide to onboarding with 3 images"),
    ("general",         "Write a blog post about AI. Include a summary at the end and 2 images"),
    ("general",         "Research the latest EV market data and improve my intro paragraph"),
    ("general",         "Make a list of competitors and search the web for their pricing"),
    ("general",         "Draw a poster, then translate the caption to German"),
    ("general",         "write an article on how to condense your wardrobe"),
    ("general",         "What are the latest battery developments? Answer in 3 bullet points."),
    ("general",         "Fix grammar and simplify this, then summarize it as a table"),
]


class TestTable(unittest.TestCase):

    def test_every_declared_type_has_an_agent_and_nothing_else_does(self):
        self.assertEqual(set(AGENT_FOR_REQUEST_TYPE), set(api.VALID_AGENT_TYPES))

    def test_every_target_is_a_configured_agent(self):
        for request_type, agent in AGENT_FOR_REQUEST_TYPE.items():
            with self.subTest(request_type=request_type):
                self.assertIn(agent, AGENT_CONFIGS)

    def test_the_documented_mapping(self):
        self.assertEqual(AGENT_FOR_REQUEST_TYPE, {
            "fix_grammar": "grammar", "simplify": "simplify", "complete": "completion", "emojify": "emojify",
            "translate": "translation", "make_longer": "make_longer", "make_shorter": "make_shorter",
            "enhance_with_ai": "enhance", "change_tone": "tone", "general": "smart_global",
            "generate_email_template": "email_template", "extract_image_text": "text_extraction",
        })

    def test_general_is_the_free_form_agent(self):
        self.assertEqual(agent_for_request_type("general"), FREE_FORM_AGENT)
        self.assertEqual(FREE_FORM_AGENT, "smart_global")

    def test_unknown_type_is_an_error_not_a_guess(self):
        for bad in ("nope", "", "Fix_Grammar", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                agent_for_request_type(bad)


class TestTextCannotChangeTheAgent(RouteTestCase):
    """Through the agent layer: the agent that gets built is the declared one, whatever the text says."""

    def setUp(self):
        super().setUp()
        self.built = []

        def fake_create_agent(agent_type, *a, **k):
            self.built.append((agent_type, k.get("plan")))
            raise AgentPathReached({"agent_type": agent_type, **k})
        p = patch.object(tpa, "create_agent", fake_create_agent)
        p.start()
        self.addCleanup(p.stop)

    def _agent_for(self, declared, text):
        full = api.build_full_query(declared, text, "")
        with self.assertRaises(AgentPathReached) as ctx:
            list(process_user_query_stream(
                full, model_provider="gemini", api_key="k", user_id="u",
                force_agent_type=agent_for_request_type(declared), image_routing=(declared == "general"),
            ))
        return ctx.exception.kwargs["agent_type"]

    def test_hijack_matrix_routes_by_declared_type(self):
        for declared, text in HIJACK_MATRIX:
            with self.subTest(declared=declared, text=text):
                self.assertEqual(self._agent_for(declared, text), AGENT_FOR_REQUEST_TYPE[declared])

    def test_the_old_router_really_did_misroute_these(self):
        """Guards the matrix: if the legacy router agreed with the table these cases would prove nothing."""
        wrong = 0
        for declared, text in HIJACK_MATRIX:
            legacy, _ = determine_agent_type(api.build_full_query(declared, text, "").lower())
            wrong += legacy != AGENT_FOR_REQUEST_TYPE[declared]
        self.assertGreaterEqual(wrong, 12)

    def test_general_always_reaches_the_smart_agent_even_with_operation_words(self):
        for text in ("summarize this", "improve my intro", "translate it", "make this shorter", "fix grammar",
                     "create table of prices", "make a list", "enhance the paragraph", "tl;dr"):
            with self.subTest(text=text):
                self.assertEqual(self._agent_for("general", text), "smart_global")

    def test_planner_still_runs_for_a_general_request_the_old_router_stole(self):
        # "summary" used to send this to the no-tools summarize agent, so the 2 images were silently dropped
        self._agent_for("general", "Write a blog post about AI. Include a summary at the end and 2 images")
        agent_type, plan = self.built[-1]
        self.assertEqual(agent_type, "smart_global")
        self.assertEqual(plan.image_count, 2)

    def test_research_wording_keeps_web_tools_for_general(self):
        # "bullet points" used to send this to the list agent (no web tools), which answered from memory
        self._agent_for("general", "What are the latest battery developments in 2026? Answer in 3 bullet points.")
        _, plan = self.built[-1]
        self.assertTrue(plan.use_web)

    def test_typed_operations_get_no_plan(self):
        self._agent_for("simplify", "write a post with 3 images about the latest news")
        self.assertEqual(self.built[-1], ("simplify", None))


class TestImageFastPathUnaffected(RouteTestCase):

    def test_general_still_takes_the_direct_image_route(self):
        out = self.run_stream("draw a red bicycle", force_agent_type=agent_for_request_type("general"))
        self.assertTrue(out[0].startswith("!["))                       # placeholder markdown, agent never built
        self.wait_for_adapter(out[1]["__pending_images__"])
        self.assertEqual(len(self.calls), 1)

    def test_typed_operations_never_take_the_image_route(self):
        for declared in ("fix_grammar", "simplify", "translate", "enhance_with_ai"):
            with self.subTest(declared=declared):
                self.assertAgentPath("draw a red bicycle", force_agent_type=agent_for_request_type(declared),
                                     image_routing=False)

    def test_email_template_and_ocr_never_take_the_image_route(self):
        for declared in ("generate_email_template", "extract_image_text"):
            with self.subTest(declared=declared):
                self.assertAgentPath("draw a red bicycle", force_agent_type=agent_for_request_type(declared),
                                     image_routing=True)

    def test_legacy_keyword_router_is_unchanged_for_callers_that_name_no_agent(self):
        self.assertEqual(determine_agent_type("fix grammar: x")[0], "grammar")
        self.assertEqual(determine_agent_type("write something")[0], "smart_global")


class TestEndpointsPassTheDeclaredAgent(unittest.TestCase):
    """api.py wiring: /agent/process and /agent/stream both force the agent from the declared type."""

    META = {"user_id": "u1", "requires_user_api_keys": False, "user_keys": {}}

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(api.app, raise_server_exceptions=False)

    def test_process_forces_the_declared_agent_for_every_type_and_hijack_text(self):
        for declared, text in HIJACK_MATRIX:
            seen = {}

            def fake(query, **kw):
                seen.update(kw, query=query)
                return "ok", {}
            with self.subTest(declared=declared, text=text), \
                    patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("k", self.META))), \
                    patch.object(api, "process_user_query_with_images", fake), \
                    patch.object(api, "CREDIT_PRECHECK_ENABLED", False):
                r = self._client().post("/agent/process", json={
                    "type": declared, "query": text, "model": "gemini", "api_key": "lic"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(seen["force_agent_type"], AGENT_FOR_REQUEST_TYPE[declared])
                self.assertEqual(seen["image_routing"], declared == "general")

    def test_stream_forces_the_declared_agent_for_every_type_and_hijack_text(self):
        for declared, text in HIJACK_MATRIX:
            seen = {}

            def fake_stream(query, **kw):
                seen.update(kw, query=query)
                yield "hello"
            with self.subTest(declared=declared, text=text), \
                    patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("k", self.META))), \
                    patch.object(api, "process_user_query_stream", fake_stream), \
                    patch.object(api, "credit_manager", None), \
                    patch.object(api, "CREDIT_PRECHECK_ENABLED", False), \
                    patch.object(api, "fire_credit_deduction"), \
                    patch.object(api, "fire_prompt_history_save"):
                r = self._client().post("/agent/stream", json={
                    "type": declared, "query": text, "model": "gemini", "api_key": "lic"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(seen["force_agent_type"], AGENT_FOR_REQUEST_TYPE[declared])

    def test_the_endpoint_source_no_longer_guesses(self):
        import inspect
        src = inspect.getsource(api)
        self.assertNotIn("forced_agent_types", src)
        self.assertEqual(src.count("force_agent_type=agent_for_request_type(agent_type)"), 2)


if __name__ == "__main__":
    unittest.main()
