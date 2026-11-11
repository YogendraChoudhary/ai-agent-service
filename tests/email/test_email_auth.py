"""/agent/email/* take the Eddyter licence key and are checked and billed like /agent/stream."""
import base64
import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import api
from config import settings
from src.email_template.categories import load_example
from src.email_template.service import EmailProviderError
from tests.email.test_email_service import FakeLLM

REMINDER = load_example("reminder_deadline")
MANAGED = ("platform-key", {"user_id": "u-managed", "requires_user_api_keys": False, "user_keys": {}})
BYOK = ("user-key", {"user_id": "u-byok", "requires_user_api_keys": True, "user_keys": {}})


def body(**kw):
    inp = REMINDER["input"]
    return {"model": "openai", "api_key": "licence-key", "brief": inp["brief"],
            "allowed_merge_tags": inp["allowed_merge_tags"], "links": {"training_portal": "https://portal.example.com"}, **kw}


class Credits:
    def __init__(self, ok=True):
        self.ok, self.reserved, self.released = ok, [], []

    def reserve(self, user, amount):
        self.reserved.append((user, amount))
        return self.ok, 7

    def schedule_release(self, user, amount, delay_seconds=0):
        self.released.append((user, amount))

    def release(self, user, amount):
        self.released.append((user, amount))


@pytest.fixture
def client(monkeypatch):
    state = {"charges": [], "credits": Credits(), "licence": MANAGED}
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(side_effect=lambda *a, **kw: state["licence"]))
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
    monkeypatch.setattr(api, "credit_manager", state["credits"])
    monkeypatch.setattr(api, "fire_credit_deduction_and_release", lambda **kw: state["charges"].append(kw))
    monkeypatch.setattr(api, "fire_credit_deduction", lambda *a, **kw: state["charges"].append((a, kw)))
    return TestClient(api.app, raise_server_exceptions=False), state


def generate(client, llm, **kw):
    with patch("src.email_template.service.make_agno_llm", return_value=llm) as made:
        return client.post("/agent/email/generate", json=body(**kw)), made


@pytest.mark.parametrize("status, detail", [
    (403, {"error": "Your access to the Eddyter editor has been suspended by an administrator.", "code": "EDITOR_ACCESS_BLOCKED"}),
    (403, "Invalid license: License key not found"),
    (402, "Your payment is past due. Please update your payment method to continue using AI features."),
    (400, "No API key configured for openai. Please add your API key in settings."),
    (503, "License validation service unavailable"),
])
def test_licence_errors_are_the_stream_errors_and_no_model_runs(client, monkeypatch, status, detail):
    http, state = client
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(side_effect=HTTPException(status_code=status, detail=detail)))
    llm = FakeLLM(REMINDER["output"])
    r, _ = generate(http, llm)
    # Same body as /agent/stream: a dict detail as-is ({"error", "code"}), otherwise {"error": detail}.
    assert r.status_code == status and r.json() == (detail if isinstance(detail, dict) else {"error": detail})
    assert not llm.calls and not state["charges"] and not state["credits"].reserved


def test_a_managed_licence_runs_on_the_platform_key_and_is_billed_as_text(client):
    http, state = client
    r, made = generate(http, FakeLLM(REMINDER["output"]))
    assert r.status_code == 200, r.text
    assert made.call_args.args[:2] == ("openai", "platform-key") and made.call_args.kwargs["is_byok"] is False
    (charge,) = state["charges"]
    assert charge["operation_type"] == "text" and charge["user_id"] == "u-managed" and charge["image_count"] == 0
    assert charge["input_chars"] == len(REMINDER["input"]["brief"])
    template = r.json()["templates"][0]["template"]
    assert 0 < charge["output_chars"] < len(str(template))      # readable text, not the JSON around it
    assert charge["reserved_scaled"] == state["credits"].reserved[0][1] and charge["provider"] == "openai"


def test_a_byok_licence_uses_the_saved_key_and_is_not_billed(client):
    http, state = client
    state["licence"] = BYOK
    r, made = generate(http, FakeLLM(REMINDER["output"]))
    assert r.status_code == 200 and made.call_args.args[:2] == ("openai", "user-key")
    assert not state["charges"] and not state["credits"].reserved


def test_insufficient_credits_is_402_before_any_model_call(client):
    http, state = client
    state["credits"].ok = False
    llm = FakeLLM(REMINDER["output"])
    r, _ = generate(http, llm)
    assert r.status_code == 402 and r.json()["code"] == "INSUFFICIENT_CREDITS" and r.json()["available"] == 7
    assert not llm.calls and not state["charges"]


def test_missing_credit_service_is_503_for_managed_users(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(api, "credit_manager", None)
    r, _ = generate(http, FakeLLM(REMINDER["output"]))
    assert r.status_code == 503


@pytest.mark.parametrize("reply, status", [("nope", 422), (EmailProviderError("openai request failed"), 502)])
def test_a_failed_generation_releases_the_reservation_and_charges_nothing(client, reply, status):
    http, state = client

    def llm(system, messages):
        if isinstance(reply, Exception):
            raise reply
        return reply
    r, _ = generate(http, llm)
    assert r.status_code == status and not state["charges"]
    assert state["credits"].released == state["credits"].reserved


def test_model_source_default_routes_writing_and_falls_back(client, monkeypatch):
    http, state = client
    monkeypatch.setattr(settings, "task_routing_enabled", True)
    monkeypatch.setattr(settings, "provider_fallback_enabled", True)

    class Redis:
        @staticmethod
        def resolve_api_key_from_user_keys(user_keys, model_provider, default_keys):
            return {"openai": "o-key", "claude": "c-key", "gemini": "g-key"}.get(model_provider)
    monkeypatch.setattr(api, "redis_manager", Redis())
    used = []

    def factory(provider, key, **kw):
        used.append((provider, key))
        if provider == "openai":
            def down(system, messages):
                raise EmailProviderError("openai overloaded")
            return down
        return FakeLLM(REMINDER["output"])
    with patch("src.email_template.service.make_agno_llm", side_effect=factory):
        r = http.post("/agent/email/generate", json=body(model="gemini", model_source="default"))
    assert r.status_code == 200, r.text
    assert used[0] == ("openai", "o-key") and used[1][0] == "gemini"      # writing default first, then the fallback tree
    assert state["charges"][0]["provider"] == "gemini"


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (1536, 1024), (10, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_image_uploads_go_to_the_licence_user_whatever_user_id_says(client):
    http, state = client
    with patch("src.s3_service.upload_image_from_bytes_detailed", return_value={"cdn_url": "https://cdn.test/a.jpg"}) as up:
        r = http.post("/agent/email/image", json={"api_key": "licence-key", "slot": "hero", "image_base64": _png(),
                                                  "user_id": "someone-else"})
    assert r.status_code == 200 and up.call_args.args[2] == "u-managed" and not state["charges"]


def test_image_needs_a_valid_licence(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(side_effect=HTTPException(403, "Invalid license")))
    with patch("src.s3_service.upload_image_from_bytes_detailed") as up:
        r = http.post("/agent/email/image", json={"api_key": "bad", "slot": "hero", "image_base64": _png()})
    assert r.status_code == 403 and not up.called


@pytest.mark.parametrize("configured, sent, raw_mode", [
    ("s3cret", "s3cret", True), ("s3cret", "wrong", False), ("s3cret", None, False), ("", "", False), ("", "anything", False),
])
def test_raw_provider_key_mode_needs_the_backend_secret(client, monkeypatch, configured, sent, raw_mode):
    http, state = client
    monkeypatch.setattr(settings, "agent_backend_secret", configured)
    licence = AsyncMock(return_value=MANAGED)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", licence)
    headers = {"X-Agent-Backend-Secret": sent} if sent is not None else {}
    with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM(REMINDER["output"])) as made:
        r = http.post("/agent/email/generate", json=body(api_key="sk-provider"), headers=headers)
    assert r.status_code == 200
    if raw_mode:
        assert not licence.called and made.call_args.args[1] == "sk-provider" and not state["charges"]
    else:
        assert licence.called and made.call_args.args[1] == "platform-key" and state["charges"]


def test_backend_secret_caller_may_choose_the_upload_folder(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(settings, "agent_backend_secret", "s3cret")
    with patch("src.s3_service.upload_image_from_bytes_detailed", return_value={"cdn_url": "https://cdn.test/a.jpg"}) as up:
        r = http.post("/agent/email/image", json={"api_key": "x", "slot": "hero", "image_base64": _png(), "user_id": "u-42"},
                      headers={"X-Agent-Backend-Secret": "s3cret"})
    assert r.status_code == 200 and up.call_args.args[2] == "u-42"


def test_output_chars_count_readable_text_only():
    result = {"templates": [{"template": {"subject_options": ["Hello"], "blocks": [
        {"id": "b1", "type": "heading", "text": "Big news"}, {"id": "b2", "type": "button", "label": "Go", "href": "link:x"},
        {"id": "b3", "type": "header", "logo_asset": "asset:logo", "theme": {"accent_color": "#22C3E6"}}]}}]}
    assert api.email_output_chars(result) == len("Hello") + len("Big news") + len("Go")
