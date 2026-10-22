"""
Mode 1 over HTTP: the four typed requests (fix_grammar, simplify, change_tone, enhance_with_ai) with the flag on and off, the SSE
sequence with its additive `refine_progress` events and `warnings` in `done`, the process response, the fences a document may
start or end with, and billing attribution. The routing, the hook and the pipeline are the real ones; only the agent factory is fake.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS
from tests.longform.fakes import FakeAgents, drop_a_token, polish
from tests.longform.fixtures import FENCE, acceptance_document

META = {"user_id": "u1", "requires_user_api_keys": False, "user_keys": {}}
DOC = acceptance_document()
TYPES = {"fix_grammar": "grammar", "simplify": "simplify", "change_tone": "tone", "enhance_with_ai": "enhance"}


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def content_of(events):
    return "".join(e["content"] for e in events if e["type"] == "content")


class ApiCase(unittest.TestCase):

    def setUp(self):
        self.agents = FakeAgents()
        self.deductions = []
        patches = [
            patch.object(tpa, "create_agent", self.agents),
            patch.object(settings, "longform_refine_enabled", True),
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
        base = {"type": "fix_grammar", "query": DOC, "model": "openai", "api_key": "lic"}
        base.update(kw)
        return base

    def refiner_saw(self):
        return "\n".join(m["content"] for _, messages, _ in self.agents.refiner.calls for m in messages)


class TestStreamEndpoint(ApiCase):

    def post(self, **kw):
        return self.client.post("/agent/stream", json=self.body(**kw))

    def test_the_sequence_has_a_progress_event_after_every_verified_chunk(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        events = sse(r)
        self.assertEqual([e["type"] for e in events], ["start"] + ["content", "refine_progress"] * 6 + ["done"])
        self.assertEqual(content_of(events), DOC)
        self.assertEqual([e["chunk"] for e in events if e["type"] == "refine_progress"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(e["of"] == 6 for e in events if e["type"] == "refine_progress"))
        self.assertNotIn("warnings", events[-1])                         # nothing was restored: `done` is what it always was
        self.assertEqual(set(events[-1]), {"type", "duration", "pending_images"})

    def test_a_chunk_the_model_damaged_is_kept_as_written_and_reported_in_done(self):
        self.agents.refiner.behaviour = drop_a_token
        events = sse(self.post())
        self.assertEqual(content_of(events), DOC)                        # nothing lost, nothing moved
        warnings = events[-1]["warnings"]
        self.assertTrue(warnings)
        self.assertEqual({w["kind"] for w in warnings}, {"chunk_restored"})
        self.assertTrue(all(isinstance(w["chunk"], int) and "kept as written" in w["detail"] for w in warnings))

    def test_with_the_flag_off_the_stream_is_exactly_what_it_was(self):
        with patch.object(settings, "longform_refine_enabled", False):
            events = sse(self.post())
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertNotIn("refine_progress", [e["type"] for e in events])
        self.assertEqual(content_of(events), FakeAgents.SINGLE_ANSWER)
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "done")

    def test_short_plain_text_is_streamed_by_the_single_call_even_with_the_flag_on(self):
        events = sse(self.post(query="A short note with a tipo in it."))
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertNotIn("refine_progress", [e["type"] for e in events])

    def test_all_four_typed_requests_use_it_with_the_document_not_the_prompt_prefix(self):
        for api_type, agent in TYPES.items():
            with self.subTest(type=api_type):
                self.agents.created.clear()
                self.agents.refiner.calls.clear()
                events = sse(self.post(type=api_type, secondary_prompt="calm and clear"))
                self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})
                self.assertEqual(content_of(events), DOC)
                seen = self.refiner_saw()
                for prefix in ("Fix grammar:", "Simplify:", "Change tone", "Enhance:"):
                    self.assertNotIn(prefix, seen)                         # the model gets the document, not the single-call prompt
                self.assertEqual("calm and clear" in seen, agent in ("tone", "enhance"))

    def test_a_polished_document_keeps_its_length_and_its_protected_blocks(self):
        self.agents.refiner.behaviour = polish
        events = sse(self.post(type="enhance_with_ai", secondary_prompt="warmer"))
        text = content_of(events)
        self.assertNotEqual(text, DOC)
        self.assertEqual(text.count(FENCE), DOC.count(FENCE))
        self.assertEqual(events[-1]["type"], "done")

    def test_billing_names_the_model_that_served_the_chunks_and_counts_the_text_streamed(self):
        for api_type, ids in (("fix_grammar", BASIC_MODEL_IDS), ("enhance_with_ai", ADVANCED_MODEL_IDS)):
            with self.subTest(type=api_type):
                self.deductions.clear()
                self.post(type=api_type)
                ((args, kwargs),) = self.deductions
                user_id, input_chars, output_chars, op_type, image_count = args
                self.assertEqual((user_id, op_type, image_count), ("u1", "text", 0))
                self.assertEqual(output_chars, len(DOC))
                self.assertEqual((kwargs["provider"], kwargs["model"]), ("openai", ids["openai"]))
                self.assertTrue(kwargs["success"])

    def test_a_failing_provider_is_an_error_event_and_reported_to_billing_as_failed(self):
        self.agents.fail = {"openai": RuntimeError("400 invalid request")}
        events = sse(self.post())
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("done", [e["type"] for e in events])
        (args, kwargs), = self.deductions
        self.assertFalse(kwargs["success"])


class TestProcessEndpoint(ApiCase):

    def post(self, **kw):
        return self.client.post("/agent/process", json=self.body(**kw))

    def test_returns_the_whole_document_and_an_empty_warnings_list(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["markdown_response"], DOC)
        self.assertEqual(data["html_response"], DOC)
        self.assertEqual(data["warnings"], [])
        self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})

    def test_restored_chunks_are_warnings_and_the_document_is_complete(self):
        self.agents.refiner.behaviour = drop_a_token
        data = self.post().json()
        self.assertEqual(data["markdown_response"], DOC)
        self.assertTrue(data["warnings"])
        self.assertEqual({w["kind"] for w in data["warnings"]}, {"chunk_restored"})

    def test_all_four_typed_requests_use_it(self):
        for api_type in TYPES:
            with self.subTest(type=api_type):
                self.agents.created.clear()
                r = self.post(type=api_type, secondary_prompt="calm and clear")
                self.assertEqual(r.json()["markdown_response"], DOC)
                self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})

    def test_the_backends_key_is_used_and_nothing_is_metered_here(self):
        self.post()
        self.assertTrue(all(c[2] == "lic" for c in self.agents.created))
        self.assertTrue(all(c[3]["is_byok"] for c in self.agents.created))
        self.assertEqual(self.deductions, [])

    def test_a_document_that_starts_or_ends_with_a_fence_keeps_both(self):
        """The single call's wrapper stripping would eat a closing fence; verified chunks have no wrappers to strip."""
        text = f"```\nplain block\n```\n\n{DOC.rstrip()}\n\n{FENCE}"
        self.assertTrue(text.endswith("\n```") and text.startswith("```\n"))
        data = self.post(query=text).json()
        self.assertEqual(data["markdown_response"], text)

    def test_with_the_flag_off_a_wrapped_single_call_answer_is_unwrapped_as_before(self):
        self.agents.single = lambda query: "```markdown\nThe fixed text.\n```"
        with patch.object(settings, "longform_refine_enabled", False):
            data = self.post().json()
        self.assertEqual(data["markdown_response"], "The fixed text.")

    def test_a_provider_failure_is_a_500_with_the_reason(self):
        self.agents.fail = {"openai": RuntimeError("400 invalid request")}
        r = self.post()
        self.assertEqual(r.status_code, 500)
        self.assertIn("400", r.json()["error"])


if __name__ == "__main__":
    unittest.main()
