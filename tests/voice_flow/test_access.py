"""Licence and credits for POST /v1/voice/dictate (src/voice_flow/access.py), with a fake Redis, credit manager and webhook."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from src.credit_manager import text_credits_scaled
from src.redis_manager import RedisManager
from src.voice_flow import router as router_module
from src.voice_flow.access import Caller, LicenceGate
from src.models import AUTOCOMPLETE_MODEL_IDS
from src.voice_flow.formatter.engine import Breaker, VoiceFormattingService, build_backend_for
from src.voice_flow.schemas import DictateContext
from tests.voice_flow.test_engine import FakeBackend
from tests.voice_flow.test_router import FakeTranscriber, dictate, wav

AUTH = {"authorization": "Bearer lic_ok"}
TASKS = "We have two tasks:\n1. Update the schema\n2. Deploy the service"


class FakeRedis(RedisManager):
    """The licence and key lookups of the real manager (including its key-resolution rules), without a network."""

    def __init__(self, *, valid=True, error="License key not found", code=None, status="active", byok=False, keys=None):
        self.valid, self.error, self.code, self.status, self.byok = valid, error, code, status, byok
        self.keys = {"use_own_keys": False, **(keys or {})}
        self.heartbeats = []

    def validate_license(self, license_key):
        if not self.valid:
            return {"valid": False, "user_id": None, "license": None, "error": self.error, "code": self.code}
        return {"valid": True, "user_id": "u1", "license": {"subscription_status": self.status, "requires_user_api_keys": self.byok}}

    def get_user_api_keys(self, user_id):
        return dict(self.keys)

    def update_heartbeat(self, license_key):
        self.heartbeats.append(license_key)


class FakeCredits:
    def __init__(self, balance=1000):
        self.balance, self.reserve_calls, self.released = balance, [], []
        self.reserved = 0

    def reserve(self, user_id, scaled):
        self.reserve_calls.append((user_id, scaled))
        available = self.balance - self.reserved
        if available < scaled:
            return False, available
        self.reserved += scaled
        return True, available - scaled

    def schedule_release(self, user_id, scaled, delay_seconds=0):
        self.released.append(scaled)
        self.reserved -= scaled


def make_host(redis, credits, precheck=True):
    host = SimpleNamespace(redis_manager=redis, credit_manager=credits, CREDIT_PRECHECK_ENABLED=precheck,
                           CREDIT_SETTLE_DELAY_SECONDS=0.0, deductions=[], failures=[])

    def fire(user_id, input_chars, output_chars, operation_type="text", image_count=0, **kw):
        (host.deductions if kw.get("success", True) else host.failures).append(
            {"user_id": user_id, "input_chars": input_chars, "output_chars": output_chars, "operation_type": operation_type, **kw})

    def fire_release(*, credit_manager, reserved_scaled, **kw):
        fire(kw.pop("user_id"), kw.pop("input_chars"), kw.pop("output_chars"), kw.pop("operation_type"), reserved_scaled=reserved_scaled, **kw)
        credit_manager.schedule_release(None, reserved_scaled)

    host.fire_credit_deduction, host.fire_credit_deduction_and_release = fire, fire_release
    return host


@pytest.fixture
def world(monkeypatch):
    backend = FakeBackend(TASKS)
    transcriber = FakeTranscriber()
    redis, credits = FakeRedis(), FakeCredits()
    host = make_host(redis, credits)
    monkeypatch.setattr(router_module, "_service", VoiceFormattingService(backend=backend, timeout=1.0, breaker=Breaker(), mode="always"))
    monkeypatch.setattr(router_module, "get_transcriber", lambda: transcriber)
    monkeypatch.setattr(router_module, "_gate", LicenceGate(host))
    monkeypatch.setattr(router_module, "_own_formatters", type(router_module._own_formatters)())
    monkeypatch.setattr(settings, "voice_flow_token", "")
    app = FastAPI()
    app.include_router(router_module.router)
    return SimpleNamespace(client=TestClient(app), app=app, backend=backend, transcriber=transcriber, redis=redis,
                           credits=credits, host=host)


class TestWhoMayDictate:
    def test_a_licence_key_is_required(self, world):
        assert dictate(world.client).status_code == 401
        assert dictate(world.client, headers={"authorization": "Basic abc"}).status_code == 401
        assert dictate(world.client, headers={"authorization": "Bearer "}).status_code == 401
        assert world.transcriber.calls == [] and world.credits.reserve_calls == []

    def test_an_invalid_licence_is_403(self, world):
        world.redis.valid = False
        assert dictate(world.client, headers=AUTH).status_code == 403
        world.redis.code = "LICENSE_EXPIRED"
        r = dictate(world.client, headers=AUTH)
        assert r.status_code == 403 and r.json()["detail"]["code"] == "LICENSE_EXPIRED"
        assert world.transcriber.calls == []

    def test_the_licence_service_being_down_is_503(self, world):
        world.redis.valid, world.redis.code, world.redis.error = False, "LICENSE_SERVICE_UNAVAILABLE", "Redis is down"
        assert dictate(world.client, headers=AUTH).status_code == 503
        world.host.redis_manager = None
        assert dictate(world.client, headers=AUTH).status_code == 503

    def test_a_plan_that_is_past_due_is_402_unless_the_account_uses_its_own_keys(self, world):
        world.redis.status = "past_due"
        assert dictate(world.client, headers=AUTH).status_code == 402
        world.redis.byok, world.redis.keys["gemini_api_key"] = True, "g-key"
        assert dictate(world.client, headers=AUTH).status_code == 200

    def test_a_good_licence_is_let_in_and_counted_as_active(self, world):
        assert dictate(world.client, headers=AUTH).status_code == 200
        assert world.redis.heartbeats == ["lic_ok"]

    def test_the_status_route_stays_open(self, world):
        assert world.client.get("/v1/voice/status").status_code == 200


class TestWhatItCosts:
    def test_a_delivered_dictation_is_billed_as_text_by_its_formatted_characters(self, world):
        r = dictate(world.client, wav(1.0), headers=AUTH)
        assert r.status_code == 200
        [deduction] = world.host.deductions
        assert deduction["user_id"] == "u1" and deduction["operation_type"] == "text" and deduction["success"] is True
        assert deduction["input_chars"] == 0 and deduction["output_chars"] == len(r.json()["formatted_text"]) == len(TASKS)
        assert deduction["provider"] == "fake" and deduction["duration_ms"] >= 0
        assert world.credits.reserve_calls == [("u1", 10)]                      # 1 credit held ...
        assert world.credits.released == [10] and deduction["reserved_scaled"] == 10        # ... and let go once the backend was told
        assert text_credits_scaled(0, deduction["output_chars"]) == 10          # the charge the backend computes: 1 credit

    def test_the_price_is_one_credit_per_thousand_characters_at_least_one(self):
        assert [text_credits_scaled(0, n) for n in (1, 59, 1000, 1001, 4500)] == [10, 10, 10, 20, 50]

    def test_the_hold_grows_with_the_length_of_the_recording(self, world):
        gate = world.app and router_module._gate
        caller = Caller("u1", billable=True)
        assert [gate.reserve(caller, s) for s in (0.5, 30, 60, 600)] == [10, 10, 20, 120]

    def test_not_enough_credits_is_402_and_nothing_runs(self, world):
        world.credits.balance = 5
        r = dictate(world.client, wav(1.0), headers=AUTH)
        assert r.status_code == 402
        assert r.json()["detail"] == {"error": "Insufficient credits", "code": "INSUFFICIENT_CREDITS", "available": 5, "needed": 10}
        assert world.transcriber.calls == [] and world.host.deductions == [] and world.credits.released == []

    def test_nothing_heard_costs_nothing_and_releases_the_hold(self, world):
        world.transcriber.text = ""
        r = dictate(world.client, wav(1.0), headers=AUTH)
        assert r.status_code == 200 and r.json()["formatted_text"] == ""
        assert world.credits.reserve_calls == [("u1", 10)] and world.credits.released == [10]
        assert world.host.deductions == [] and world.host.failures == []

    def test_a_key_bounce_reserves_and_costs_nothing(self, world):
        assert dictate(world.client, wav(0.02), headers=AUTH).status_code == 200
        assert world.credits.reserve_calls == [] and world.host.deductions == [] and world.transcriber.calls == []

    def test_a_refused_recording_costs_nothing(self, world, monkeypatch):
        monkeypatch.setattr(settings, "voice_max_utterance_seconds", 5)
        assert dictate(world.client, wav(6.0), headers=AUTH).status_code == 413
        assert dictate(world.client, b"not audio", headers=AUTH).status_code == 400
        assert world.credits.reserve_calls == [] and world.host.deductions == []

    def test_a_formatter_that_fell_back_to_rules_is_still_billed(self, world):
        world.backend.error = RuntimeError("boom")
        r = dictate(world.client, wav(1.0), headers=AUTH)
        assert r.json()["source"] == "rules" and len(world.host.deductions) == 1

    def test_a_crash_releases_the_hold_and_is_reported_not_charged(self, world, monkeypatch):
        async def boom(*a, **k):
            raise RuntimeError("whisper died")
        monkeypatch.setattr(world.transcriber, "transcribe", boom)
        client = TestClient(world.app, raise_server_exceptions=False)
        assert dictate(client, wav(1.0), headers=AUTH).status_code == 500
        assert world.credits.released == [10] and world.host.deductions == []
        [failure] = world.host.failures
        assert failure["success"] is False and failure["error_code"] == "RuntimeError" and failure["output_chars"] == 0

    def test_a_failed_webhook_handoff_releases_the_hold(self, world):
        def fails(**kw):
            raise RuntimeError("backend webhook broke")
        world.host.fire_credit_deduction_and_release = fails
        client = TestClient(world.app, raise_server_exceptions=False)
        assert dictate(client, wav(1.0), headers=AUTH).status_code == 500
        assert world.credits.released == [10] and len(world.host.failures) == 1

    def test_without_the_credit_precheck_the_backend_is_still_told(self, world):
        world.host.CREDIT_PRECHECK_ENABLED = False
        assert dictate(world.client, wav(1.0), headers=AUTH).status_code == 200
        assert world.credits.reserve_calls == [] and len(world.host.deductions) == 1
        assert "reserved_scaled" not in world.host.deductions[0]

    def test_credit_service_unavailable_is_refused_before_transcription(self, world):
        world.host.credit_manager = None
        assert dictate(world.client, headers=AUTH).status_code == 503
        assert world.transcriber.calls == [] and world.host.deductions == []

    @pytest.mark.parametrize("balance, status", [(10, 402), (20, 200)])
    def test_actual_output_above_the_estimate_requires_a_top_up(self, world, balance, status):
        world.credits.balance = balance
        world.transcriber.text = "we should review the schema before deploying the service. " * 25
        world.backend.reply = world.transcriber.text.strip()
        response = dictate(world.client, headers=AUTH)
        assert response.status_code == status
        assert world.credits.reserve_calls == [("u1", 10), ("u1", 10)]
        assert world.credits.reserved == 0
        if status == 200:
            assert len(response.json()["formatted_text"]) > 1000
            assert world.host.deductions[0]["reserved_scaled"] == 20
        else:
            assert world.host.deductions == [] and world.credits.released == [10]

    async def test_cancellation_releases_the_hold(self, world, monkeypatch):
        monkeypatch.setattr(world.transcriber, "transcribe", AsyncMock(side_effect=asyncio.CancelledError))
        with pytest.raises(asyncio.CancelledError):
            await router_module._dictate(wav(), DictateContext(), Caller("u1", billable=True))
        assert world.credits.released == [10] and world.host.deductions == []

    def test_whitespace_transcript_costs_nothing(self, world):
        world.transcriber.text = "   "
        assert dictate(world.client, headers=AUTH).json()["formatted_text"] == ""
        assert world.credits.released == [10] and world.host.deductions == []

    def test_empty_formatted_output_costs_nothing(self, world, monkeypatch):
        from src.voice_flow.formatter.post_processor import Stitched
        original = router_module._format

        async def empty(*args, **kwargs):
            _, result, post_ms = await original(*args, **kwargs)
            return Stitched(text="", leading_space=False, trailing_space=False), result, post_ms

        monkeypatch.setattr(router_module, "_format", empty)
        assert dictate(world.client, headers=AUTH).json()["formatted_text"] == ""
        assert world.credits.released == [10] and world.host.deductions == []


class TestAccountsOnTheirOwnKeys:
    @pytest.fixture
    def own(self, world, monkeypatch):
        built, own_backend = [], FakeBackend(TASKS)
        monkeypatch.setattr(router_module, "build_backend_for", lambda provider, key: built.append((provider, key)) or own_backend)
        world.built, world.own_backend = built, own_backend
        return world

    def test_such_an_account_is_not_charged_and_formats_with_its_own_key(self, own):
        own.redis.byok, own.redis.keys["openai_api_key"] = True, "sk-user-openai"
        r = dictate(own.client, wav(1.0), headers=AUTH)
        assert r.status_code == 200 and r.json()["source"] == "llm"
        assert own.built == [("openai", "sk-user-openai")]
        assert len(own.own_backend.calls) == 1 and own.backend.calls == []        # the platform's formatter was not used
        assert own.credits.reserve_calls == [] and own.host.deductions == [] and own.host.failures == []

    def test_the_platforms_provider_is_preferred_when_both_keys_exist(self, own, monkeypatch):
        own.redis.byok = True
        own.redis.keys.update(openai_api_key="sk-o", gemini_api_key="g-key")
        monkeypatch.setattr(settings, "voice_format_provider", "gemini")
        dictate(own.client, wav(1.0), headers=AUTH)
        monkeypatch.setattr(settings, "voice_format_provider", "openai")
        own.redis.keys["openai_api_key"] = "sk-o2"                                # a different key: a different formatter
        dictate(own.client, wav(1.0), headers=AUTH)
        assert own.built == [("gemini", "g-key"), ("openai", "sk-o2")]

    def test_the_use_my_own_keys_setting_on_a_platform_plan_formats_with_the_users_key_but_is_still_charged(self, own):
        own.redis.keys.update(use_own_keys=True, gemini_api_key="g-key")
        assert dictate(own.client, wav(1.0), headers=AUTH).status_code == 200
        assert own.built == [("gemini", "g-key")]
        assert own.credits.reserve_calls == [("u1", 10)] and len(own.host.deductions) == 1

    def test_such_an_account_without_a_usable_key_is_refused_and_no_platform_resource_is_used(self, own):
        own.redis.byok = True
        own.redis.keys.update(assemblyai_api_key="asr-only", gemini_api_key=" ")
        r = dictate(own.client, wav(1.0), headers=AUTH)
        assert r.status_code == 400 and r.json()["detail"]["code"] == "NO_API_KEY"
        assert own.transcriber.calls == [] and own.backend.calls == [] and own.built == []

    def test_a_platform_account_never_sees_another_accounts_key(self, own):
        own.redis.keys.update(use_own_keys=False, gemini_api_key="g-key")            # a stored key is ignored unless the setting is on
        assert dictate(own.client, wav(1.0), headers=AUTH).status_code == 200
        assert own.built == [] and len(own.backend.calls) == 1

    @pytest.mark.parametrize("provider, field", [("claude", "anthropic_api_key"), ("grok", "grok_api_key"), ("deepseek", "deepseek_api_key")])
    def test_other_supported_personal_keys_work_without_platform_keys_or_credits(self, own, provider, field):
        own.redis.byok, own.redis.keys[field] = True, "personal-key"
        assert dictate(own.client, headers=AUTH).status_code == 200
        assert own.built == [(provider, "personal-key")]
        assert own.backend.calls == [] and own.credits.reserve_calls == [] and own.host.deductions == []

    def test_personal_provider_failure_never_calls_the_platform_formatter(self, own):
        own.redis.byok, own.redis.keys["openai_api_key"] = True, "personal-key"
        own.own_backend.error = RuntimeError("personal provider unavailable")
        assert dictate(own.client, headers=AUTH).json()["source"] == "rules"
        assert own.backend.calls == [] and own.credits.reserve_calls == [] and own.host.deductions == []

    def test_each_key_has_its_own_formatter_and_the_cache_is_bounded(self, own, monkeypatch):
        monkeypatch.setattr(router_module, "OWN_FORMATTERS_MAX", 2)
        a, a_again = (router_module._formatter_for(Caller(own_key=("gemini", "k1"))) for _ in range(2))
        b = router_module._formatter_for(Caller(own_key=("gemini", "k2")))
        assert a is a_again and a is not b and a.breaker is not b.breaker          # one account's bad key pauses nobody else
        assert router_module._formatter_for(Caller(own_key=("gemini", "k1"))) is a  # make k1 the most recently used
        router_module._formatter_for(Caller(own_key=("openai", "k3")))              # the third evicts the least recently used (k2)
        assert len(router_module._own_formatters) == 2
        assert router_module._formatter_for(Caller(own_key=("gemini", "k1"))) is a
        assert all("k1" not in name and "k2" not in name for name in router_module._own_formatters)    # keys are never used as names


class TestPersonalBackends:
    @pytest.mark.parametrize("provider", AUTOCOMPLETE_MODEL_IDS)
    def test_only_light_models_and_personal_keys_are_used(self, monkeypatch, provider):
        monkeypatch.setattr(settings, "voice_format_provider", provider)
        monkeypatch.setattr(settings, "voice_format_model", "expensive-platform-override")
        backend = build_backend_for(provider, "personal-key")
        assert backend.name == provider and backend.api_key == "personal-key"
        assert backend.model == AUTOCOMPLETE_MODEL_IDS[provider]
        with pytest.raises(ValueError):
            build_backend_for(provider, " ")

    @pytest.mark.parametrize("provider", ["grok", "deepseek", "claude"])
    async def test_personal_backends_send_the_prompt_and_return_only_text(self, provider):
        backend = build_backend_for(provider, "personal-key")
        turns = [{"role": "user", "content": "hello"}]
        if provider == "claude":
            response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Hello.")])
            create = AsyncMock(return_value=response)
            backend._client = SimpleNamespace(messages=SimpleNamespace(create=create))
        else:
            response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Hello."))])
            create = AsyncMock(return_value=response)
            backend._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            from src.text_processing_agent import OPENAI_COMPATIBLE_BASES
            assert backend.base_url == OPENAI_COMPATIBLE_BASES[provider]
        assert await backend.generate("Format only", turns, max_output_tokens=100, timeout=1.0) == "Hello."
        params = create.call_args.kwargs
        assert params["model"] == AUTOCOMPLETE_MODEL_IDS[provider] and params["max_tokens"] == 100
        assert params["messages"] == (turns if provider == "claude" else [{"role": "system", "content": "Format only"}, *turns])


class TestWiring:
    def test_the_production_app_installs_the_gate(self):
        import api
        if not settings.voice_flow_enabled:
            pytest.skip("voice is switched off")
        assert isinstance(router_module._gate, LicenceGate) and router_module._gate.host is api

    def test_a_dictation_through_the_real_app_is_billed_through_the_real_webhook_path(self, monkeypatch):
        import api
        redis, credits, transcriber = FakeRedis(), FakeCredits(), FakeTranscriber()
        sent, released = [], threading.Event()

        def release(user_id, scaled):
            credits.schedule_release(user_id, scaled)
            released.set()

        monkeypatch.setattr(credits, "release", release, raising=False)
        monkeypatch.setattr(api, "redis_manager", redis)
        monkeypatch.setattr(api, "credit_manager", credits)
        monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
        monkeypatch.setattr(api, "CREDIT_SETTLE_DELAY_SECONDS", 0.0)
        monkeypatch.setattr("src.webhook.BACKEND_WEBHOOK_URL", "https://billing.invalid/voice-test")
        monkeypatch.setattr("src.webhook.WEBHOOK_SECRET", "test-secret")
        monkeypatch.setattr("src.webhook.requests.post", lambda *a, **kw: sent.append(kw) or SimpleNamespace(status_code=200))
        monkeypatch.setattr(router_module, "_service", VoiceFormattingService(backend=FakeBackend(TASKS), timeout=1.0, breaker=Breaker(), mode="always"))
        monkeypatch.setattr(router_module, "get_transcriber", lambda: transcriber)
        client = TestClient(api.app, raise_server_exceptions=False)
        assert dictate(client, wav(1.0)).status_code == 401
        r = dictate(client, wav(1.0), headers=AUTH)
        assert r.status_code == 200 and r.json()["formatted_text"] == TASKS
        assert released.wait(2)
        assert credits.reserve_calls == [("u1", 10)] and credits.released == [10]
        assert sent[0]["json"] == {
            "user_id": "u1", "input_chars": 0, "output_chars": len(TASKS), "operation_type": "text", "image_count": 0,
            "provider": "fake", "model": None, "duration_ms": sent[0]["json"]["duration_ms"],
            "success": True, "error_code": None,
        }
        assert sent[0]["headers"]["X-Webhook-Secret"] == "test-secret"
        assert dictate(client, b"nope", headers=AUTH).json() == {"error": "could not decode the audio"}
        monkeypatch.setattr(router_module, "_in_flight", router_module.MAX_DICTATIONS_IN_FLIGHT)
        busy = dictate(client, headers=AUTH)
        assert busy.status_code == 429 and busy.headers["retry-after"] == "2"
