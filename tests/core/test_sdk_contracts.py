"""
The installed provider SDKs must accept every argument the app sends (src/sdk_contracts.py).

Production failure this guards (2026-11-20): every Claude call failed with "Messages.stream() got an unexpected keyword argument
'output_config'" because the server had an older anthropic SDK. The first test runs the check against the REAL installed packages,
never fakes; the rest check the checker, the runtime classification, the fallback and what the user sees.
"""
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agno.agent import RunEvent
from anthropic.resources.messages import Messages

import src.text_processing_agent as tpa
from config import settings
from src import sdk_contracts as C
from src.provider_errors import can_fall_back, is_provider_unusable, is_sdk_contract_error, public_error_message
from src.provider_health import HEALTH, ProviderHealth, classify_error
from src.task_router import KeyResolver

PROD_ERROR = "claude request failed: Messages.stream() got an unexpected keyword argument 'output_config'"


def without(method, name):
    """A stand-in for an older SDK method: the same signature minus one parameter."""
    sig = inspect.signature(method)

    def old(*a, **kw):
        raise AssertionError("never called")
    old.__signature__ = sig.replace(parameters=[p for p in sig.parameters.values() if p.name != name])
    return old


def test_the_installed_sdks_accept_everything_the_app_sends():
    report = C.check_sdk_contracts()
    assert report["ok"], "\n".join(f["message"] for f in report["findings"])
    assert report["checked"] >= 40 and all(report["versions"].values())


def test_an_sdk_without_output_config_is_caught_on_the_paths_that_send_it(monkeypatch):
    monkeypatch.setattr(Messages, "stream", without(Messages.stream, "output_config"))
    report = C.check_sdk_contracts()
    messages = [f["message"] for f in report["findings"]]
    assert not report["ok"]
    assert "agno claude advanced -> stream: Messages.stream() does not accept output_config" in messages
    assert any("light (classifier)" in m for m in messages)
    assert all("output_config" in m for m in messages)          # nothing else is flagged


def test_direct_calls_are_checked_too(monkeypatch):
    monkeypatch.setattr(Messages, "create", without(Messages.create, "output_config"))
    messages = [f["message"] for f in C.check_sdk_contracts()["findings"]]
    assert "design canvas claude: Messages.create() does not accept output_config" in messages


def test_an_sdk_without_temperature_is_caught_too(monkeypatch):
    # anthropic 1.x removed temperature from stream(); agno sends it on models that take it (the basic tier's Sonnet does not).
    monkeypatch.setattr(Messages, "stream", without(Messages.stream, "temperature"))
    sent_temperature = any("temperature" in tpa.get_model("claude", "k", use_advanced=a, cache_model=False).get_request_params()
                           for a in (True, False))
    assert C.check_sdk_contracts()["ok"] is (not sent_temperature)


def test_kwargs_methods_cannot_be_rejected_statically():
    assert C.unsupported_params(lambda **kw: None, {"anything"}) == ()
    assert C.unsupported_params(lambda model, messages: None, {"model", "output_config"}) == ("output_config",)


def test_genai_type_keywords_are_checked(monkeypatch, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "x.py").write_text(
        "from google.genai import types\n"
        "types.GenerateContentConfig(temperature=0, max_output_tokens=5)\n"
        "types.GenerateContentConfig(temperature=0, made_up_option=1)\n")
    monkeypatch.setattr(C, "_SRC", tmp_path / "pkg")
    (finding,) = C._genai_type_findings()
    assert finding.unsupported == ("made_up_option",) and finding.path.endswith("x.py:3 types.GenerateContentConfig")


def test_the_check_never_raises_when_a_path_cannot_be_built(monkeypatch):
    monkeypatch.setattr(C, "_direct_paths", lambda: (_ for _ in ()).throw(ImportError("no module named x")))
    report = C.check_sdk_contracts()
    assert not report["ok"] and "could not build the request: ImportError" in report["findings"][0]["message"]


def test_startup_check_warns_or_refuses(monkeypatch, caplog):
    monkeypatch.setattr(Messages, "stream", without(Messages.stream, "output_config"))
    HEALTH.reset()
    with caplog.at_level(logging.CRITICAL, logger="src.sdk_contracts"):
        report = C.run_startup_check("warn")
    assert not report["ok"] and HEALTH.sdk_contracts is report
    assert any("event=sdk_contract_mismatch" in r.message and r.levelno == logging.CRITICAL for r in caplog.records)
    assert any(a["code"] == "sdk_contract_mismatch" and a["severity"] == "critical" for a in HEALTH.snapshot()["alerts"])
    with pytest.raises(RuntimeError, match="Install requirements.lock"):
        C.run_startup_check("fail")
    HEALTH.reset()


def test_setting_only_takes_warn_or_fail(monkeypatch):
    from config import Settings
    monkeypatch.setenv("SDK_CONTRACT_ON_MISMATCH", "ignore")
    with pytest.raises(Exception):
        Settings()


# ---------------------------------------------------------------------------
# At request time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, mismatch", [
    (PROD_ERROR, True),
    ("Completions.create() got an unexpected keyword argument 'verbosity'", True),
    ("generate() missing 1 required positional argument: 'config'", True),
    ("Error code: 401 - invalid x-api-key", False),
    ("400 bad request", False),
])
def test_classification(text, mismatch):
    error = RuntimeError(text)
    assert is_sdk_contract_error(error) is mismatch
    assert (classify_error(error) == "sdk_contract_mismatch") is mismatch
    if mismatch:
        assert is_provider_unusable(error) and can_fall_back(error)


def test_a_mismatch_alerts_on_the_first_call_even_with_a_users_key(caplog):
    health = ProviderHealth()
    with caplog.at_level(logging.CRITICAL, logger="src.provider_health"):
        health.record_error("claude", "claude-opus-5-5", RuntimeError(PROD_ERROR), byok=True)
    assert any("event=sdk_contract_mismatch" in r.message for r in caplog.records)
    (alert,) = health.alerts()
    assert alert["code"] == "sdk_contract_mismatch" and alert["severity"] == "critical"


def test_the_stream_falls_back_and_says_why(caplog):
    keys = KeyResolver({}, {"anthropic_api_key": "a", "openai_api_key": "o", "gemini_api_key": "g"},
                       lambda **kw: kw["default_keys"].get("anthropic_api_key" if kw["model_provider"] == "claude"
                                                           else kw["model_provider"] + "_api_key"))
    calls = []

    def create(agent_type, provider, key, *a, **kw):
        calls.append(provider)
        events = ([SimpleNamespace(event=RunEvent.run_error, content=PROD_ERROR.split(": ", 1)[1])] if provider == "claude"
                  else [SimpleNamespace(event=RunEvent.run_content, content="Done.")])
        return SimpleNamespace(run=lambda *a, **kw: iter(events))

    HEALTH.reset()
    with patch.object(settings, "task_routing_enabled", True), patch.object(settings, "provider_fallback_enabled", True), \
            patch.object(tpa, "create_agent", side_effect=create), patch.object(tpa.time, "sleep"), \
            caplog.at_level(logging.CRITICAL):
        out = list(tpa.process_user_query_stream("Put 50 more words on Antarctic weather", model_provider="claude", api_key="a",
                                                 model_source="user", key_resolver=keys, report_model=True))
    assert calls[:2] == ["claude", "openai"] and out[0]["__model_used__"][0] == "openai"
    assert "claude->openai (sdk_contract_mismatch)" in HEALTH.snapshot()["fallbacks"]
    assert any("event=sdk_contract_mismatch" in r.message for r in caplog.records)
    HEALTH.reset()


@pytest.mark.parametrize("error, shown", [
    (RuntimeError(PROD_ERROR), "The Claude model is temporarily unavailable because of a problem on our side."),
    (TypeError("'NoneType' object is not subscriptable"), "Something went wrong on our side. Please try again."),
    (KeyError("choices"), "Something went wrong on our side. Please try again."),
    (RuntimeError("openai request failed: Error code: 401 - Incorrect API key provided"),
     "openai request failed: Error code: 401 - Incorrect API key provided"),          # actionable: unchanged
    (ValueError("brief must contain at least 3 characters of text"), "brief must contain at least 3 characters of text"),
])
def test_what_the_user_sees(error, shown):
    message = public_error_message(error)
    assert message.startswith(shown) and "output_config" not in message and "NoneType" not in message
