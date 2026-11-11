"""/agent/process takes a raw provider key, so with AGENT_BACKEND_SECRET set only the backend may call it."""
import pytest

from config import settings
from tests.test_document_context_api import runtime  # noqa: F401  (fixture: scripted model, no network)

BODY = {"type": "general", "query": "Write one line about Velvet Mango", "model": "openai", "api_key": "sk-provider"}


@pytest.mark.parametrize("header", [None, "wrong", ""])
def test_with_a_secret_configured_other_callers_get_401_and_no_model_runs(runtime, monkeypatch, header):  # noqa: F811
    client, state = runtime
    monkeypatch.setattr(settings, "agent_backend_secret", "s3cret")
    headers = {} if header is None else {"X-Agent-Backend-Secret": header}
    r = client.post("/agent/process", json=BODY, headers=headers)
    assert r.status_code == 401 and r.json() == {"error": "Unauthorized", "code": "BACKEND_SECRET_REQUIRED"}
    assert "X-Request-ID" in r.headers and not state["calls"]


def test_the_backend_with_the_secret_is_served(runtime, monkeypatch):  # noqa: F811
    client, state = runtime
    monkeypatch.setattr(settings, "agent_backend_secret", "s3cret")
    r = client.post("/agent/process", json=BODY, headers={"X-Agent-Backend-Secret": "s3cret"})
    assert r.status_code == 200 and r.json()["success"] is True and len(state["calls"]) == 1


def test_without_a_secret_the_route_stays_open_as_before(runtime, monkeypatch):  # noqa: F811
    client, state = runtime
    monkeypatch.setattr(settings, "agent_backend_secret", "")
    r = client.post("/agent/process", json=BODY)
    assert r.status_code == 200 and len(state["calls"]) == 1


def test_the_stream_route_never_needs_the_secret(runtime, monkeypatch):  # noqa: F811
    client, _ = runtime
    monkeypatch.setattr(settings, "agent_backend_secret", "s3cret")
    r = client.post("/agent/stream", json={**BODY, "api_key": "licence-key"})
    assert r.status_code == 200
