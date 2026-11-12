"""Rollout safeguards: the model/key preflight, and MODEL_<TIER>_<PROVIDER> overrides for rollback without a deploy."""
import logging
import unittest
from unittest.mock import MagicMock, patch

import src.model_preflight as P
import src.models as M
import src.text_processing_agent as tpa
from config import Settings, settings
from src.provider_health import HEALTH


class StatusError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


class TestConfiguredModels(unittest.TestCase):

    def test_every_job_a_model_is_configured_for_is_listed(self):
        wanted = P.configured_models()
        self.assertEqual(wanted["claude"]["claude-opus-5-5"], ["advanced tier"])
        self.assertEqual(wanted["claude"]["claude-haiku-5-5"], ["autocomplete / classifier / voice"])
        self.assertIn("design canvas", wanted["openai"]["gpt-6.1-sol"])
        self.assertEqual(sorted(wanted["openai"]["gpt-5.2"]), ["pinned: blog_writer", "pinned: code_explainer", "pinned: code_writer"])
        self.assertIn("image default", wanted["openai"]["gpt-image-2.5-flare"])
        self.assertIn("image fallback", wanted["openai"]["gpt-image-1.5"])
        self.assertIn("image default", wanted["gemini"]["gemini-3.1-flash-lite-image"])


class TestCheckProvider(unittest.TestCase):
    MODELS = {"m-ok": ["advanced tier"], "m-gone": ["image default"]}

    def check(self, lookup, key="k"):
        with patch.object(P, "_lookup", side_effect=lookup):
            return P.check_provider("openai", key, self.MODELS)

    def test_all_present(self):
        result = self.check(lambda *a: None)
        self.assertEqual((result["key"], result["problems"]), ("accepted", []))
        self.assertTrue(all(row["ok"] for row in result["models"].values()))

    def test_a_missing_model_is_a_problem_naming_it(self):
        def lookup(provider, key, model_id, timeout):
            if model_id == "m-gone":
                raise StatusError(404, "The model 'm-gone' does not exist")
        result = self.check(lookup)
        self.assertEqual(result["key"], "accepted")
        self.assertEqual([(p["model"], p["detail"].split(":")[0]) for p in result["problems"]], [("m-gone", "model_missing")])
        self.assertFalse(result["models"]["m-gone"]["ok"])
        self.assertTrue(result["models"]["m-ok"]["ok"])

    def test_a_rejected_key_is_one_finding_not_one_per_model(self):
        def lookup(*a):
            raise StatusError(401, "invalid x-api-key")
        result = self.check(lookup)
        self.assertEqual(result["key"], "rejected")
        self.assertEqual([p["model"] for p in result["problems"]], ["(key)"])

    def test_an_unreachable_provider_is_not_reported_as_a_broken_rollout(self):
        def lookup(*a):
            raise ConnectionError("connection reset by peer")
        result = self.check(lookup)
        self.assertEqual((result["key"], result["problems"]), ("unreachable", []))
        self.assertIsNone(result["models"]["m-ok"]["ok"])

    def test_no_key_means_not_configured(self):
        self.assertEqual(self.check(lambda *a: None, key=None), {"key": "not configured", "models": {}, "problems": []})


class TestOpenAILookup(unittest.TestCase):
    """models.retrieve answers 200 for the withdrawn gpt-image-1; the account's model list is the truth."""

    def setUp(self):
        P._OPENAI_IDS.clear()

    def test_a_model_missing_from_the_list_is_missing_even_if_retrieve_would_answer(self):
        client = MagicMock()
        client.models.list.return_value = [MagicMock(id="gpt-image-2.5-flare"), MagicMock(id="gpt-6.1-terra")]
        with patch("openai.OpenAI", return_value=client):
            P._lookup("openai", "k", "gpt-6.1-terra", 5)
            with self.assertRaisesRegex(LookupError, "does not exist"):
                P._lookup("openai", "k", "gpt-image-1", 5)
        client.models.retrieve.assert_not_called()
        client.models.list.assert_called_once()                                 # one list call serves every lookup of the run

    def test_it_is_classified_as_a_missing_model(self):
        from src.provider_health import classify_error
        self.assertEqual(classify_error(LookupError("The model 'gpt-image-1' does not exist in this account's model list")), "model_missing")


class TestRunPreflight(unittest.TestCase):

    def test_problems_are_stored_alerted_and_logged(self):
        def lookup(provider, key, model_id, timeout):
            if model_id == "gpt-image-2.5-flare":
                raise StatusError(404, "The model 'gpt-image-2.5-flare' does not exist.")
        with patch.object(P, "_lookup", side_effect=lookup), self.assertLogs("src.model_preflight", level="ERROR") as logs:
            report = P.run_preflight({"claude": "k", "openai": "k", "gemini": None})
        self.assertFalse(report["ok"])
        self.assertEqual([(p["provider"], p["model"]) for p in report["problems"]], [("openai", "gpt-image-2.5-flare")])
        self.assertIn("ROLLOUT_ALERT preflight provider=openai model=gpt-image-2.5-flare", logs.output[0])
        self.assertIs(HEALTH.preflight, report)
        alerts = HEALTH.alerts()
        self.assertEqual([(a["code"], a["severity"]) for a in alerts], [("preflight", "critical")])
        self.assertEqual(report["providers"]["gemini"]["key"], "not configured")

    def test_a_clean_run_is_ok_and_quiet(self):
        with patch.object(P, "_lookup", return_value=None), self.assertNoLogs("src.model_preflight", level="ERROR"):
            report = P.run_preflight({"claude": "k", "openai": "k", "gemini": "k"})
        self.assertTrue(report["ok"])
        self.assertEqual(HEALTH.alerts(), [])

    def test_record_false_leaves_the_health_view_alone(self):
        with patch.object(P, "_lookup", side_effect=StatusError(401, "invalid x-api-key")):
            report = P.run_preflight({"claude": "k", "openai": None, "gemini": None}, record=False)
        self.assertFalse(report["ok"])
        self.assertIsNone(HEALTH.preflight)

    def test_the_setting_defaults_on_and_the_test_environment_turns_it_off(self):
        self.assertIs(Settings.model_fields['model_preflight_on_start'].default, True)    # on in production ...
        self.assertFalse(settings.model_preflight_on_start)                               # ... off here: conftest sets MODEL_PREFLIGHT_ON_START=false


class TestEnvOverrides(unittest.TestCase):

    def setUp(self):
        self.patches = [patch.dict(table) for table in M.MODEL_TIERS.values()]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_overrides_replace_only_what_is_named(self):
        before = dict(M.BASIC_MODEL_IDS)
        changed = M.apply_model_env_overrides({"MODEL_ADVANCED_OPENAI": " gpt-5.2 ", "MODEL_AUTOCOMPLETE_CLAUDE": "claude-haiku-4-5",
                                               "MODEL_DESIGN_OPENAI": "gpt-5.2"})
        self.assertEqual(changed, {"ADVANCED/openai": "gpt-5.2", "AUTOCOMPLETE/claude": "claude-haiku-4-5", "DESIGN/openai": "gpt-5.2"})
        self.assertEqual(M.ADVANCED_MODEL_IDS["openai"], "gpt-5.2")
        self.assertEqual(M.AUTOCOMPLETE_MODEL_IDS["claude"], "claude-haiku-4-5")
        self.assertEqual(M.design_model_id("openai"), "gpt-5.2")
        self.assertEqual(M.BASIC_MODEL_IDS, before)

    def test_blank_unknown_and_unchanged_values_do_nothing(self):
        self.assertEqual(M.apply_model_env_overrides({"MODEL_ADVANCED_OPENAI": "  ", "MODEL_ADVANCED_NOBODY": "x", "MODEL_NOPE_OPENAI": "x",
                                                      "MODEL_ADVANCED_CLAUDE": M.ADVANCED_MODEL_IDS["claude"]}), {})

    def test_the_environment_is_the_default_source(self):
        with patch.dict("os.environ", {"MODEL_BASIC_CLAUDE": "claude-sonnet-5"}):
            self.assertEqual(M.apply_model_env_overrides(), {"BASIC/claude": "claude-sonnet-5"})
        self.assertEqual(M.BASIC_MODEL_IDS["claude"], "claude-sonnet-5")

    def test_no_override_is_active_in_the_test_environment(self):
        self.assertEqual(M.MODEL_ENV_OVERRIDES, {})

    def test_a_rolled_back_model_sends_the_request_it_expects(self):
        M.apply_model_env_overrides({"MODEL_ADVANCED_CLAUDE": "claude-opus-5", "MODEL_ADVANCED_OPENAI": "gpt-5.2",
                                     "MODEL_AUTOCOMPLETE_CLAUDE": "claude-haiku-4-5"})
        claude = tpa.get_model("claude", "dummy", use_advanced=True, cache_model=False, temperature=0.7)
        self.assertEqual((claude.id, "temperature" in claude.get_request_params()), ("claude-opus-5", False))   # Opus 5 still takes none
        openai = tpa.get_model("openai", "dummy", use_advanced=True, cache_model=False)
        self.assertEqual(openai.id, "gpt-5.2")
        self.assertNotIn("reasoning_effort", openai.get_request_params())
        self.assertEqual(M.claude_light_request_params(M.AUTOCOMPLETE_MODEL_IDS["claude"]), {})         # Haiku 4.5: no effort field


if __name__ == "__main__":
    unittest.main()
