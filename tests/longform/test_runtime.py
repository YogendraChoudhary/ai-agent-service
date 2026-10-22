"""
The refinement pipeline inside process_user_query_stream: when it runs and when the single call still does, which tier and
settings each model call gets, what comes out and in what order, and what a failing provider does. The routing, the hook and the
pipeline are the real ones; only the agent factory is fake.
"""

import random
import time
import unittest
from unittest.mock import patch

import src.text_processing_agent as tpa
from config import settings
from src.longform.prompts import OPERATIONS, system_prompt
from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS
from src.task_router import KeyResolver
from tests.longform.fakes import FakeAgents, drop_a_token, polish
from tests.longform.fixtures import FENCE, QUOTE, TABLE, acceptance_document, paragraph

# What api.py builds for the single call, per agent type
PREFIX = {"grammar": "Fix grammar: ", "simplify": "Simplify: ", "tone": "Change tone to calm: ", "enhance": "Enhance: sharper, Text: "}


class RuntimeCase(unittest.TestCase):

    def setUp(self):
        self.agents = FakeAgents()
        for p in (patch.object(tpa, "create_agent", self.agents), patch.object(settings, "longform_refine_enabled", True)):
            p.start()
            self.addCleanup(p.stop)

    def stream(self, agent="grammar", text=None, **kw):
        text = acceptance_document() if text is None else text
        kw.setdefault("query", PREFIX[agent] + text)
        kw.setdefault("user_text", text)
        kw.setdefault("force_agent_type", agent)
        kw.setdefault("model_provider", "openai")
        kw.setdefault("api_key", "user-key")
        kw.setdefault("user_id", "u1")
        kw.setdefault("report_model", True)
        return list(tpa.process_user_query_stream(**kw))

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


class TestWhenThePipelineRuns(RuntimeCase):

    def test_a_long_document_goes_through_it_chunk_by_chunk(self):
        text = acceptance_document()
        out = self.stream(text=text)
        self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})
        self.assertEqual(len(self.agents.created), 6)                    # the acceptance document is six chunks
        self.assertEqual(self.document(out), text)                       # an identity model returns it byte for byte

    def test_with_the_flag_off_the_single_call_runs_exactly_as_before(self):
        with patch.object(settings, "longform_refine_enabled", False):
            out = self.stream()
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertEqual(self.document(out), FakeAgents.SINGLE_ANSWER)
        self.assertEqual(self.shape(out)[0], "model_used")
        self.assertNotIn("event:refine_progress", self.shape(out))

    def test_the_flag_is_off_by_default(self):
        self.assertFalse(type(settings).model_fields["longform_refine_enabled"].default)

    def test_short_plain_text_keeps_the_single_call(self):
        out = self.stream(text="A short note with a typo in it.")
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertEqual(self.document(out), FakeAgents.SINGLE_ANSWER)

    def test_a_short_text_with_a_table_or_a_quote_or_a_code_block_is_still_protected(self):
        for name, block in {"table": TABLE, "quote": QUOTE, "code": FENCE}.items():
            with self.subTest(block=name):
                self.agents.created.clear()
                text = f"Some words to polish.\n\n{block}\n\nA few more words.\n"
                out = self.stream(text=text)
                self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})
                self.assertEqual(self.document(out), text)

    def test_without_the_text_the_single_call_runs(self):
        """Callers that pass only the query (not the four typed endpoints) are untouched."""
        out = self.stream(user_text=None)
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertEqual(self.document(out), FakeAgents.SINGLE_ANSWER)

    def test_reference_images_keep_the_single_call(self):
        with patch.object(tpa, "build_agno_images", return_value=["image"]):
            self.stream(image_urls=["https://cdn.test/a.png"])
        self.assertEqual(self.agents.agent_types(), ["grammar"])

    def test_html_keeps_the_single_call(self):
        html = "".join(f"<p>{paragraph(random.Random(i), 40)}</p>\n" for i in range(30))
        self.stream(text=html)
        self.assertEqual(self.agents.agent_types(), ["grammar"])

    def test_text_that_already_looks_like_our_tokens_keeps_the_single_call(self):
        self.stream(text=acceptance_document() + "\n\nA stray [[TABLE_1]] in the user's own words.\n")
        self.assertEqual(self.agents.agent_types(), ["grammar"])

    def test_the_other_agent_types_never_use_it(self):
        text = acceptance_document()
        for agent in ("translation", "summarize", "make_longer", "make_shorter", "emojify", "completion", "table", "list"):
            with self.subTest(agent=agent):
                self.agents.created.clear()
                list(tpa.process_user_query_stream("Do it: " + text, model_provider="openai", api_key="k",
                                                   force_agent_type=agent, user_text=text))
                self.assertEqual(self.agents.agent_types(), [agent])

    def test_the_first_chunk_is_small_by_default_and_the_setting_turns_that_off(self):
        def smallest():
            passages = [c[1][0]["content"].split("<<<PASSAGE\n", 1)[1] for c in self.agents.refiner.calls]
            return min(len(p.split()) for p in passages)
        self.stream()
        self.assertLess(smallest(), 400)
        self.agents.refiner.calls.clear()
        with patch.object(settings, "longform_first_chunk_words", 0):
            self.stream()
        self.assertGreater(smallest(), 400)

    def test_if_the_pipeline_cannot_be_imported_here_the_single_call_keeps_working(self):
        """Its regular expressions need Python 3.11; on an older interpreter turning the flag on must not take the four types down."""
        import sys
        tpa._longform.cache_clear()
        self.addCleanup(tpa._longform.cache_clear)
        with patch.dict(sys.modules, {"src.longform.refine": None}), self.assertLogs(tpa.logger, level="ERROR") as logged:
            self.assertIsNone(tpa._longform())
            out = self.stream()
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.assertEqual(self.document(out), FakeAgents.SINGLE_ANSWER)
        self.assertEqual(len(logged.records), 1)                                           # logged once, not on every request
        self.assertIn("Long-form refinement is unavailable", logged.output[0])

    def test_the_minimum_length_is_a_setting(self):
        text = "\n\n".join(paragraph(random.Random(i), 100) for i in range(4))         # about 400 words, no structure
        self.stream(text=text)
        self.assertEqual(self.agents.agent_types(), ["grammar"])
        self.agents.created.clear()
        with patch.object(settings, "longform_refine_min_words", 300):
            self.stream(text=text)
        self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})


class TestWhatEveryModelCallGets(RuntimeCase):

    def test_the_tier_follows_the_operation_as_the_single_call_does(self):
        for agent in OPERATIONS:
            with self.subTest(operation=agent):
                self.agents.created.clear()
                tier = tpa.AGENT_CONFIGS[agent].get("use_advanced", True)
                out = self.stream(agent=agent)
                self.assertEqual({c[3]["plan"].advanced_model for c in self.agents.created}, {tier})
                ids = ADVANCED_MODEL_IDS if tier else BASIC_MODEL_IDS
                self.assertEqual(out[0], {"__model_used__": ("openai", ids["openai"])})
        self.assertTrue(tpa.AGENT_CONFIGS["enhance"].get("use_advanced", True))               # enhance stays on the advanced tier
        self.assertFalse(tpa.AGENT_CONFIGS["grammar"].get("use_advanced", True))

    def test_each_call_carries_the_operations_prompt_temperature_and_a_budget(self):
        for agent, operation in OPERATIONS.items():
            with self.subTest(operation=agent):
                self.agents.created.clear()
                self.stream(agent=agent)
                for agent_type, provider, key, kw in self.agents.created:
                    self.assertEqual((agent_type, provider, key), ("longform_refiner", "openai", "user-key"))
                    self.assertEqual(kw["instructions"], system_prompt(agent))
                    self.assertEqual(kw["temperature"], operation.temperature)
                    self.assertGreaterEqual(kw["max_tokens"], 1024)
                    self.assertFalse(kw["plan"].use_web)                         # no tools: the passage is the only input

    def test_the_users_account_details_reach_the_model_call(self):
        self.stream(user_id="user-7", is_byok=True)
        self.assertEqual({(c[3]["user_id"], c[3]["is_byok"]) for c in self.agents.created}, {("user-7", True)})

    def test_the_instruction_reaches_every_chunk_of_enhance_and_tone_only(self):
        for agent, line in (("enhance", "INSTRUCTION: make it punchier"), ("tone", "TONE: make it punchier")):
            with self.subTest(operation=agent):
                self.agents.refiner.calls.clear()
                self.stream(agent=agent, instruction="make it punchier")
                firsts = [c[1][0]["content"] for c in self.agents.refiner.calls]
                self.assertEqual(len(firsts), 6)
                self.assertTrue(all(line in m for m in firsts))
        for agent in ("grammar", "simplify"):
            with self.subTest(operation=agent):
                self.agents.refiner.calls.clear()
                self.stream(agent=agent, instruction="make it punchier")
                self.assertFalse(any("punchier" in c[1][0]["content"] for c in self.agents.refiner.calls))

    def test_every_chunk_is_sent_as_data_between_the_markers(self):
        self.stream()
        for system, messages, spec in self.agents.refiner.calls:
            self.assertEqual((len(messages), spec.stage), (1, "chunk"))
            self.assertIn("<<<PASSAGE\n", messages[0]["content"])


class TestWhatComesOut(RuntimeCase):

    def test_the_model_first_then_each_verified_chunk_followed_by_its_progress(self):
        out = self.stream()
        self.assertEqual(self.shape(out), ["model_used"] + ["text", "event:refine_progress"] * 6)
        events = [c["__event__"] for c in out if isinstance(c, dict) and "__event__" in c]
        self.assertEqual(events, [{"type": "refine_progress", "chunk": i, "of": 6} for i in range(1, 7)])

    def test_without_report_model_no_model_is_announced(self):
        out = self.stream(report_model=False)
        self.assertEqual(self.shape(out), ["text", "event:refine_progress"] * 6)

    def test_the_document_is_complete_after_polishing_and_keeps_every_protected_block(self):
        self.agents.refiner.behaviour = polish
        text = acceptance_document()
        out = self.document(self.stream(text=text))
        self.assertNotEqual(out, text)
        self.assertEqual(out.count(TABLE), text.count(TABLE))
        self.assertEqual(out.count(QUOTE), text.count(QUOTE))

    def test_a_chunk_the_model_damages_is_restored_and_reported_once_at_the_end(self):
        self.agents.refiner.behaviour = drop_a_token
        text = acceptance_document()
        out = self.stream(text=text)
        self.assertEqual(self.document(out), text)                        # nothing was lost
        warnings = [c["__warnings__"] for c in out if isinstance(c, dict) and "__warnings__" in c]
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0])
        self.assertEqual({w["kind"] for w in warnings[0]}, {"chunk_restored"})
        self.assertIs(out[-1], next(c for c in out if isinstance(c, dict) and "__warnings__" in c))      # the last thing yielded
        self.assertLess(self.shape(out).index("event:refine_progress"), self.shape(out).index("warnings"))

    def test_no_warnings_sentinel_when_nothing_was_restored(self):
        self.assertNotIn("warnings", self.shape(self.stream()))

    def test_the_non_streaming_entry_point_returns_the_text_and_the_warnings(self):
        self.agents.refiner.behaviour = drop_a_token
        text = acceptance_document()
        meta = {}
        result, pending = tpa.process_user_query_with_images(
            PREFIX["grammar"] + text, model_provider="openai", api_key="k", force_agent_type="grammar", is_byok=True,
            user_text=text, meta=meta)
        self.assertEqual(result, text)
        self.assertTrue(meta["warnings"])
        self.assertEqual({w["kind"] for w in meta["warnings"]}, {"chunk_restored"})

    def test_the_trace_records_the_pipeline_for_the_dashboards(self):
        from src.observability import finish_trace, get_trace, latency_stats, start_trace
        start_trace("req-trace", "/agent/stream")
        self.stream(request_id="req-trace")
        attrs = get_trace("req-trace").attrs
        finish_trace("req-trace", "ok")
        self.assertEqual(attrs["mode"], "longform")
        self.assertEqual(attrs["longform"]["operation"], "grammar")
        self.assertEqual((attrs["longform"]["chunks"], attrs["longform"]["restored"]), (6, 0))
        self.assertEqual(attrs["model_used"], f"openai/{BASIC_MODEL_IDS['openai']}")
        self.assertEqual(len(attrs["llm_calls"]), 6)
        self.assertIn(f"longform | openai/{BASIC_MODEL_IDS['openai']}", latency_stats()["modes"])      # the latency dashboards


class TestProviderFallback(RuntimeCase):
    """A chunk's call moves to the next provider on the errors the single call moves on, and on nothing else."""

    def routed(self, keys=None, fallback=True, **kw):
        keys = keys if keys is not None else {"anthropic_api_key": "a", "openai_api_key": "o", "gemini_api_key": "g"}
        resolver = KeyResolver({}, keys, lambda **k: k["default_keys"].get("anthropic_api_key" if k["model_provider"] == "claude"
                                                                           else k["model_provider"] + "_api_key"))
        for name, value in (("task_routing_enabled", True), ("provider_fallback_enabled", fallback)):
            p = patch.object(settings, name, value)
            p.start()
            self.addCleanup(p.stop)
        values = dict(model_provider="openai", api_key="o", model_source="user", key_resolver=resolver)
        values.update(kw)
        return self.stream(**values)

    def test_a_failing_provider_costs_nothing_but_time(self):
        self.agents.fail = {"openai": RuntimeError("503 overloaded")}
        text = acceptance_document()
        out = self.routed(text=text)
        self.assertEqual(self.document(out), text)
        served = out[0]["__model_used__"][0]
        self.assertNotEqual(served, "openai")                                          # who actually wrote it
        self.assertLessEqual(self.agents.runs["openai"], settings.longform_parallel)      # at most the chunks in flight paid for it
        self.assertEqual(self.agents.runs.get(served, 0), 6)

    def test_a_bad_request_is_not_retried_elsewhere(self):
        self.agents.fail = {"openai": RuntimeError("400 invalid request")}
        with self.assertRaises(RuntimeError):
            self.routed()
        self.assertEqual(set(self.agents.runs), {"openai"})

    def test_with_fallback_off_the_first_failure_is_the_answer(self):
        self.agents.fail = {"openai": RuntimeError("503 overloaded")}
        with self.assertRaises(RuntimeError):
            self.routed(fallback=False)
        self.assertEqual(set(self.agents.runs), {"openai"})

    def test_a_byok_account_only_reaches_the_providers_it_has_keys_for(self):
        self.agents.fail = {"openai": RuntimeError("503 overloaded")}
        with self.assertRaises(RuntimeError):
            self.routed(keys={"openai_api_key": "o"})
        self.assertEqual(set(self.agents.runs), {"openai"})

    def test_a_rate_limit_is_retried_once_on_the_same_provider_first(self):
        with patch("time.sleep") as sleep:
            self.agents.fail = {"openai": lambda n: RuntimeError("429 rate limit") if n == 1 else None}
            text = acceptance_document()
            out = self.routed(text=text)
        self.assertEqual(self.document(out), text)
        self.assertEqual(out[0]["__model_used__"][0], "openai")
        self.assertTrue(sleep.called)

    def test_a_call_that_never_comes_back_times_out_and_the_next_provider_serves_the_passage(self):
        self.agents.hang = {"openai": 5.0}
        text = acceptance_document()
        started = time.perf_counter()
        with patch.object(settings, "longform_call_timeout_seconds", 0.3):
            out = self.routed(text=text)
        self.assertLess(time.perf_counter() - started, 4.0)                             # not 5 s: nobody waited for the hung calls
        self.assertEqual(self.document(out), text)
        self.assertNotEqual(out[0]["__model_used__"][0], "openai")

    def test_without_fallback_a_hung_call_is_an_error_not_a_hang(self):
        self.agents.hang = {"openai": 5.0}
        started = time.perf_counter()
        with patch.object(settings, "longform_call_timeout_seconds", 0.3), self.assertRaises(TimeoutError) as raised:
            self.routed(fallback=False)
        self.assertIn("timed out after 0.3s", str(raised.exception))
        self.assertLess(time.perf_counter() - started, 4.0)

    def test_a_timeout_of_zero_waits_as_long_as_it_takes(self):
        self.agents.hang = {"openai": 0.6}
        text = acceptance_document()
        with patch.object(settings, "longform_call_timeout_seconds", 0):
            out = self.routed(text=text)
        self.assertEqual(self.document(out), text)
        self.assertEqual(out[0]["__model_used__"][0], "openai")

    def test_the_tier_is_kept_on_the_fallback_provider(self):
        self.agents.fail = {"openai": RuntimeError("503 overloaded")}
        out = self.routed(agent="enhance")
        served = out[0]["__model_used__"]
        self.assertEqual(served[1], ADVANCED_MODEL_IDS[served[0]])


if __name__ == "__main__":
    unittest.main()
