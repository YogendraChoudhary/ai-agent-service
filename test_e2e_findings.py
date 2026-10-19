"""Regression tests for problems found by the local end-to-end run, and for request observability."""

import json
import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agno.agent import RunEvent

import src.observability as obs
import src.text_processing_agent as tpa
from config import settings
from src.intent import detect_image_intent


class TestWoodenTableIsNotAText(unittest.TestCase):

    def test_furniture_table_does_not_block_a_drawing_request(self):
        for q in ("draw a green apple on a wooden table", "draw a dining table set for dinner"):
            with self.subTest(q=q):
                self.assertEqual(detect_image_intent(q)["kind"], "generate")

    def test_document_tables_still_go_to_the_text_agent(self):
        for q in ("draw a table of quarterly results", "create a table of prices", "make a comparison table"):
            with self.subTest(q=q):
                self.assertEqual(detect_image_intent(q)["kind"], "none")


class _FakeAgent:
    def __init__(self, events):
        self.events = events

    def run(self, query, stream=True, images=None):
        return iter(self.events)


def _run(events, **kw):
    with patch.object(tpa, "create_agent", lambda *a, **k: _FakeAgent(events)):
        return list(tpa.process_user_query_stream("fix grammar: me goes", model_provider="claude", api_key="k", **kw))


class TestModelErrorsAreNotSwallowed(unittest.TestCase):

    def test_run_error_raises_instead_of_returning_an_empty_success(self):
        err = SimpleNamespace(event=RunEvent.run_error, content="Error code: 401 - authentication_error")
        with self.assertRaises(RuntimeError) as ctx:
            _run([err])
        self.assertIn("claude request failed", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    def test_error_after_partial_output_still_raises(self):
        ok = SimpleNamespace(event=RunEvent.run_content, content="Hel")
        err = SimpleNamespace(event=RunEvent.run_error, content="boom")
        with self.assertRaises(RuntimeError):
            _run([ok, err])

    def test_empty_response_without_images_is_an_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            _run([])
        self.assertIn("empty response", str(ctx.exception))

    def test_normal_content_is_unchanged(self):
        evs = [SimpleNamespace(event=RunEvent.run_content, content="He and I"),
               SimpleNamespace(event=RunEvent.run_content, content=" went.")]
        self.assertEqual("".join(_run(evs)), "He and I went.")

    def test_error_message_is_single_line_and_short(self):
        msg = tpa._agent_error_message("claude", "line1\nline2 " + "x" * 1000)
        self.assertNotIn("\n", msg)
        self.assertLessEqual(len(msg), 340)


class TestRedaction(unittest.TestCase):

    def test_secrets_are_redacted_at_any_depth_and_input_is_untouched(self):
        body = {"api_key": "LICENSE-123", "query": "hi", "image_params": {"api_key": "sk-x", "quality": "high"},
                "items": [{"authorization": "Bearer abc"}]}
        out = obs.redact(body)
        self.assertEqual(out["api_key"], "***redacted***")
        self.assertEqual(out["image_params"], {"api_key": "***redacted***", "quality": "high"})
        self.assertEqual(out["items"][0]["authorization"], "***redacted***")
        self.assertEqual(body["api_key"], "LICENSE-123")
        self.assertNotIn("LICENSE-123", json.dumps(out))


class TestTraces(unittest.TestCase):

    def setUp(self):
        obs._RING.clear()

    def test_summary_has_marks_attrs_images_and_total(self):
        obs.start_trace("t1", "/agent/stream", type="general")
        obs.trace_mark("t1", "auth_done")
        obs.trace_set("t1", route="direct_image")
        obs.trace_mark("t1", "auth_done")                      # second mark must not overwrite the first
        s = obs.finish_trace("t1", images=[{"status": "done", "provider": "gemini", "model": "m", "seconds": 2.0}], http_status=200)
        self.assertEqual((s["route"], s["status"], s["http_status"]), ("direct_image", "ok", 200))
        self.assertIn("auth_done", s["marks"])
        self.assertGreaterEqual(s["total_seconds"], 0)
        self.assertEqual(s["images"][0]["model"], "m")

    def test_marking_an_unknown_request_is_a_noop(self):
        obs.trace_mark("nope", "x")
        obs.trace_set(None, a=1)
        self.assertIsNone(obs.finish_trace("nope"))

    def test_finish_removes_the_trace_so_nothing_leaks(self):
        obs.start_trace("t2", "/agent/process")
        obs.finish_trace("t2")
        self.assertIsNone(obs.get_trace("t2"))

    def test_jsonl_file_is_written_when_configured(self):
        with tempfile.NamedTemporaryFile("r+", suffix=".jsonl") as fh, patch.object(settings, "request_log_file", fh.name):
            obs.start_trace("t3", "/agent/stream", type="general")
            obs.finish_trace("t3", error="x" * 1000, status="error", http_status=500)
            fh.seek(0)
            row = json.loads(fh.read().strip())
        self.assertEqual((row["request_id"], row["status"]), ("t3", "error"))
        self.assertLess(len(row["error"]), 500)

    def test_stats_group_by_route_and_image_model(self):
        for i, secs in enumerate((1.0, 2.0, 3.0), 1):
            obs._RING.append({"endpoint": "/agent/stream", "route": "agent", "status": "ok", "total_seconds": secs,
                              "images": [{"status": "done", "provider": "gemini", "model": "m", "seconds": secs}]})
        st = obs.latency_stats()
        self.assertEqual(st["requests"]["/agent/stream | agent"]["p50"], 2.0)
        self.assertEqual(st["image_generation"]["gemini/m"]["max"], 3.0)

    def test_bodies_are_logged_only_when_enabled_and_redacted(self):
        with self.assertLogs("observability", level="INFO") as cm, patch.object(settings, "log_request_bodies", True):
            obs.log_request_body("r", "/agent/stream", {"api_key": "SECRET", "query": "hello"})
        self.assertIn("hello", cm.output[0])
        self.assertNotIn("SECRET", cm.output[0])
        with patch.object(settings, "log_request_bodies", False):
            with self.assertRaises(AssertionError):            # assertLogs fails when nothing is logged
                with self.assertLogs("observability", level="INFO"):
                    obs.log_request_body("r", "/agent/stream", {"query": "hello"})

    def test_long_bodies_are_clipped(self):
        with self.assertLogs("observability", level="INFO") as cm, patch.object(settings, "log_request_bodies", True), \
                patch.object(settings, "log_body_max_chars", 100):
            obs.log_response_body("r", "/agent/process", {"text": "y" * 5000})
        self.assertIn("chars]", cm.output[0])
        self.assertLess(len(cm.output[0]), 400)


class TestMetricsEndpoint(unittest.TestCase):

    def test_off_by_default_on_when_enabled(self):
        import api
        from fastapi.testclient import TestClient
        client = TestClient(api.app, raise_server_exceptions=False)
        with patch.object(settings, "metrics_endpoint_enabled", False):
            self.assertEqual(client.get("/agent/metrics").status_code, 404)
        with patch.object(settings, "metrics_endpoint_enabled", True):
            r = client.get("/agent/metrics")
            self.assertEqual(r.status_code, 200)
            self.assertIn("stats", r.json())

    def test_every_response_carries_a_response_time_header(self):
        import api
        from fastapi.testclient import TestClient
        r = TestClient(api.app).get("/health")
        self.assertTrue(r.headers["X-Response-Time-Ms"].isdigit())


if __name__ == "__main__":
    unittest.main()
