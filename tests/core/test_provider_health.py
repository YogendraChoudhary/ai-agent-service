"""Rollout safeguards: provider/model health counters, alerts, and where the app feeds them."""
import logging
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api
import src.models as M
import src.text_processing_agent as tpa
from config import settings
from src.provider_health import HEALTH, ProviderHealth, classify_error


class StatusError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


class TestClassify(unittest.TestCase):

    def test_what_each_error_means(self):
        cases = {
            "key_rejected": [StatusError(401, "invalid x-api-key"), StatusError(403, "permission denied"), StatusError(402, "prepayment credits depleted"),
                             Exception("Error code: 400 - Your credit balance is too low")],
            "model_missing": [StatusError(404, "model: claude-x"), Exception("The model 'gpt-image-1' does not exist."), Exception("model_not_found")],
            "request_rejected": [StatusError(400, "`temperature` is deprecated for this model."), StatusError(422, "bad"),
                                 Exception("Unsupported parameter: 'max_tokens' is not supported with this model")],
            "rate_limited": [StatusError(429, "slow down"), Exception("RESOURCE_EXHAUSTED")],
            "server_error": [StatusError(500, "boom"), StatusError(529, "overloaded"), ConnectionError("connection reset")],
        }
        for kind, errors in cases.items():
            for error in errors:
                with self.subTest(kind=kind, error=str(error)[:50]):
                    self.assertEqual(classify_error(error), kind)


class TestCounting(unittest.TestCase):

    def setUp(self):
        self.h = ProviderHealth(window=60)

    def calls(self, provider, model, ok, **failures):
        for _ in range(ok):
            self.h.record(provider, model, "ok")
        for kind, n in failures.items():
            for _ in range(n):
                self.h.record(provider, model, kind, detail=f"{kind} sample")

    def test_snapshot_counts_per_model(self):
        self.calls("claude", "claude-opus-5-5", 8, truncated=1)
        self.calls("openai", "gpt-6.1-terra", 5)
        snap = self.h.snapshot()
        self.assertEqual(snap["models"]["claude/claude-opus-5-5"]["calls"], 9)
        self.assertEqual(snap["models"]["claude/claude-opus-5-5"]["failures"], {"truncated": 1})
        self.assertEqual(snap["models"]["openai/gpt-6.1-terra"], {"calls": 5, "ok": 5, "failures": {}, "last_error": {}})

    def test_old_events_leave_the_window(self):
        self.calls("claude", "m", 3, key_rejected=2)
        with patch("src.provider_health.time.time", return_value=time.time() + 120):
            snap = self.h.snapshot()
        self.assertEqual((snap["models"], snap["alerts"]), ({}, []))

    def test_a_rejected_platform_key_is_critical_but_a_users_own_key_is_not(self):
        self.h.record("claude", "claude-opus-5-5", "key_rejected", byok=True, detail="401")
        self.assertEqual(self.h.alerts(), [])
        self.h.record("claude", "claude-opus-5-5", "key_rejected", detail="401 invalid x-api-key")
        (alert,) = self.h.alerts()
        self.assertEqual((alert["code"], alert["severity"], alert["model"]), ("key_rejected", "critical", "claude-opus-5-5"))
        self.assertIn("fall back", alert["what"])

    def test_one_missing_model_is_critical(self):
        self.h.record("openai", "gpt-image-1", "model_missing", detail="The model 'gpt-image-1' does not exist.")
        self.assertEqual([(a["code"], a["severity"]) for a in self.h.alerts()], [("model_missing", "critical")])

    def test_a_request_shape_rejection_needs_three_or_a_two_percent_rate(self):
        self.calls("claude", "claude-sonnet-5-5", 0, request_rejected=2)
        self.assertEqual(self.h.alerts(), [])
        self.calls("claude", "claude-sonnet-5-5", 0, request_rejected=1)
        self.assertEqual([a["code"] for a in self.h.alerts()], ["request_rejected"])
        self.assertIn("sample", self.h.alerts()[0]["sample"])
        other = ProviderHealth()
        for _ in range(120):
            other.record("claude", "m", "ok")
        other.record("claude", "m", "request_rejected")
        other.record("claude", "m", "request_rejected")
        self.assertEqual([a["code"] for a in other.alerts()], [])           # 2 of 122 is under the 2% rate and under 3
        other.record("claude", "m", "request_rejected")
        self.assertEqual([a["code"] for a in other.alerts()], ["request_rejected"])

    def test_a_single_bad_request_is_not_logged_as_an_alert_but_the_third_is(self):
        with self.assertNoLogs("src.provider_health", level="ERROR"):
            self.h.record("claude", "m", "request_rejected", detail="prompt too long")
            self.h.record("claude", "m", "request_rejected", detail="prompt too long")
        with self.assertLogs("src.provider_health", level="ERROR") as logs:
            self.h.record("claude", "m", "request_rejected", detail="prompt too long")
        self.assertIn("ROLLOUT_ALERT", logs.output[0])

    def test_truncation_and_empty_answers_alert_only_at_volume_and_rate(self):
        self.calls("openai", "gpt-6.1-terra", 15, truncated=2)             # 17 calls: too few to judge
        self.assertEqual(self.h.alerts(), [])
        self.calls("openai", "gpt-6.1-terra", 0, truncated=3)              # 20 calls, 5 truncated = 25%
        self.assertEqual([a["code"] for a in self.h.alerts()], ["truncated"])

    def test_a_high_failure_rate_alerts_unless_a_sharper_alert_already_explains_it(self):
        self.calls("gemini", "g", 5, server_error=6)
        self.assertEqual([a["code"] for a in self.h.alerts()], ["high_failure_rate"])
        sharper = ProviderHealth()
        for _ in range(6):
            sharper.record("gemini", "g", "key_rejected")
        for _ in range(6):
            sharper.record("gemini", "g", "server_error")
        self.assertEqual([a["code"] for a in sharper.alerts()], ["key_rejected"])

    def test_fallbacks_are_counted_by_pair_and_reason(self):
        self.h.record_fallback("claude", "openai", "401")
        self.h.record_fallback("claude", "openai", "401")
        self.assertEqual(self.h.snapshot()["fallbacks"], {"claude->openai (401)": 2})

    def test_keys_in_error_text_are_never_stored(self):
        self.h.record("claude", "m", "request_rejected", detail="bad header x-api-key: sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        stored = self.h.snapshot()["models"]["claude/m"]["last_error"]["request_rejected"]
        self.assertNotIn("KLMNOPQRSTUVWXYZ", stored)
        self.assertIn("sk-ant-ap...", stored)

    def test_a_critical_alert_is_logged_loudly_once_per_hour(self):
        with self.assertLogs("src.provider_health", level="ERROR") as logs:
            for _ in range(5):
                self.h.record("claude", "claude-opus-5-5", "key_rejected", detail="401")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("ROLLOUT_ALERT", logs.output[0])
        self.assertIn("provider=claude", logs.output[0])

    def test_a_byok_failure_is_not_logged_as_an_alert(self):
        logger = logging.getLogger("src.provider_health")
        with patch.object(logger, "error") as error:
            self.h.record("claude", "m", "key_rejected", byok=True)
        error.assert_not_called()


class TestWhereTheAppFeedsIt(unittest.TestCase):

    def autocomplete(self, create):
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client):
            return tpa.get_autocomplete_suggestion("Thanks for", "", "claude", "k")

    def test_autocomplete_success_and_swallowed_failure_are_both_counted(self):
        ok = lambda **kw: SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=" the help")])
        self.assertEqual(self.autocomplete(ok), " the help")

        def rejected(**kw):
            raise StatusError(400, "`temperature` is deprecated for this model.")
        self.assertEqual(self.autocomplete(rejected), "")                       # the user sees nothing ...
        counts = HEALTH.snapshot()["models"]["claude/claude-haiku-5-5"]
        self.assertEqual((counts["ok"], counts["failures"]), (1, {"request_rejected": 1}))   # ... the rollout view does

    def test_a_byok_autocomplete_failure_is_flagged_as_such(self):
        client = SimpleNamespace(messages=SimpleNamespace(create=MagicMock(side_effect=StatusError(401, "invalid x-api-key"))))
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client):
            tpa.get_autocomplete_suggestion("Thanks for", "", "claude", "k", is_byok=True)
        self.assertEqual(HEALTH.alerts(), [])

    def test_stream_failures_use_the_model_the_agent_runs_on(self):
        tpa._record_stream_failure("openai", "blog_writer", True, StatusError(429, "rate limit"), False)
        tpa._record_stream_failure("openai", "enhance", True, RuntimeError("The openai model returned an empty response"), False)
        tpa._record_stream_failure("claude", "grammar", False, StatusError(401, "invalid x-api-key"), False)
        models = HEALTH.snapshot()["models"]
        self.assertEqual(models["openai/gpt-5.2"]["failures"], {"rate_limited": 1})          # pinned blog agent
        self.assertEqual(models["openai/gpt-6.1-terra"]["failures"], {"empty": 1})
        self.assertEqual(models["claude/claude-sonnet-5-5"]["failures"], {"key_rejected": 1})

    def test_design_failures_are_told_apart(self):
        tpa._design_failure_kind("claude", "claude-opus-5-5", ValueError("Design output truncated by model at max_tokens limit (12288)"), False)
        tpa._design_failure_kind("claude", "claude-opus-5-5", ValueError("Claude declined the design request (cyber)"), False)
        tpa._design_failure_kind("claude", "claude-opus-5-5", StatusError(400, "bad"), False)
        self.assertEqual(HEALTH.snapshot()["models"]["claude/claude-opus-5-5"]["failures"], {"truncated": 1, "refused": 1, "request_rejected": 1})

    def test_a_withdrawn_image_model_is_counted_even_though_the_chain_recovers(self):
        from src.image_gen import ImageParams, _call_openai_with_fallback
        item = MagicMock(b64_json="iVBORw0KGgo=", url=None)
        client = MagicMock()
        client.images.generate.side_effect = [Exception("The model 'gpt-image-1' does not exist."), MagicMock(data=[item])]
        with patch("openai.OpenAI", return_value=client):
            _call_openai_with_fallback("k", "a cat", ImageParams(), model_id="gpt-image-1")
        models = HEALTH.snapshot()["models"]
        self.assertEqual(models["openai/gpt-image-1"]["failures"], {"model_missing": 1})
        self.assertEqual(models["openai/gpt-image-2.5-flare"]["ok"], 1)
        self.assertEqual([a["code"] for a in HEALTH.alerts()], ["model_missing"])

    def test_the_classifier_counts_a_failed_run(self):
        from agno.run.base import RunStatus
        from src import intent_classifier
        bad = SimpleNamespace(status=RunStatus.error, content="Error code: 401 - invalid x-api-key")
        self.assertEqual(intent_classifier._answer(bad, "claude", "claude-haiku-5-5", RunStatus), "")
        self.assertEqual(HEALTH.snapshot()["models"]["claude/claude-haiku-5-5"]["failures"], {"key_rejected": 1})


class TestEndpoint(unittest.TestCase):

    def get(self, headers=None):
        return TestClient(api.app).get("/agent/health/providers", headers=headers or {})

    def test_hidden_without_the_secret_or_the_metrics_flag(self):
        with patch.object(settings, "metrics_endpoint_enabled", False), patch.object(settings, "agent_backend_secret", "s3cret"):
            self.assertEqual(self.get().status_code, 404)
            self.assertEqual(self.get({"X-Agent-Backend-Secret": "wrong"}).status_code, 404)

    def test_the_backend_secret_opens_it_and_status_follows_the_alerts(self):
        with patch.object(settings, "metrics_endpoint_enabled", False), patch.object(settings, "agent_backend_secret", "s3cret"):
            body = self.get({"X-Agent-Backend-Secret": "s3cret"}).json()
            self.assertEqual((body["status"], body["alerts"]), ("ok", []))
            HEALTH.record("claude", "claude-opus-5-5", "key_rejected", detail="401")
            body = self.get({"X-Agent-Backend-Secret": "s3cret"}).json()
        self.assertEqual(body["status"], "alert")
        self.assertEqual(body["alerts"][0]["code"], "key_rejected")
        self.assertIn("claude/claude-opus-5-5", body["models"])
        self.assertNotIn("401 invalid", str(body.get("preflight")))

    def test_the_metrics_flag_opens_it_for_local_use(self):
        with patch.object(settings, "metrics_endpoint_enabled", True):
            self.assertEqual(self.get().status_code, 200)


class TestStartup(unittest.TestCase):
    """The lifespan starts the model check in a background thread when MODEL_PREFLIGHT_ON_START is on, and says so when overrides are active."""

    def start_app(self, enabled, overrides=None):
        import threading
        called = threading.Event()
        patches = [patch.object(settings, "model_preflight_on_start", enabled), patch.object(settings, "voice_flow_enabled", False),
                   patch("src.model_preflight.run_preflight", lambda *a, **k: called.set())]
        if overrides is not None:
            patches.append(patch.object(M, "MODEL_ENV_OVERRIDES", overrides))
        for p in patches:
            p.start()
        try:
            with self.assertLogs("api", level="WARNING") as logs, TestClient(api.app):
                api.logger.warning("sentinel")                         # assertLogs needs at least one record
                ran = called.wait(2)
            return ran, "\n".join(logs.output)
        finally:
            for p in patches:
                p.stop()

    def test_the_check_runs_at_startup_when_enabled(self):
        self.assertTrue(self.start_app(True)[0])

    def test_and_not_when_disabled(self):
        self.assertFalse(self.start_app(False)[0])

    def test_active_overrides_are_announced(self):
        _, log = self.start_app(False, {"ADVANCED/openai": "gpt-5.2"})
        self.assertIn("Model overrides from the environment are active", log)
        self.assertIn("gpt-5.2", log)


if __name__ == "__main__":
    unittest.main()
