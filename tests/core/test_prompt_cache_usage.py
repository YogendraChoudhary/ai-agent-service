import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agno.agent import Agent
from agno.metrics import RunMetrics, accumulate_model_metrics
from agno.models.response import ModelResponse
from anthropic.types import Usage
from fastapi.testclient import TestClient
from google.genai.types import GenerateContentResponseUsageMetadata
from openai import OpenAI
from openai.types import CompletionUsage

import api
from src import text_processing_agent as tpa
from src.observability import finish_trace, recent_summaries, record_provider_usage, record_text_usage, start_trace


@pytest.mark.parametrize("provider,usage,read,write,miss", [
    ("claude", Usage(input_tokens=30, output_tokens=10, cache_read_input_tokens=60,
                    cache_creation_input_tokens=10), 60, 10, None),
    ("openai", CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                               prompt_tokens_details={"cached_tokens": 60, "cache_write_tokens": 10}), 60, 10, None),
    ("gemini", GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=10,
                                                   cached_content_token_count=60), 60, None, None),
    ("grok", CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                             prompt_tokens_details={"cached_tokens": 60}), 60, None, None),
    ("deepseek", CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                                 prompt_cache_hit_tokens=60, prompt_cache_miss_tokens=40), 60, None, 40),
])
def test_cache_counters_survive_agno_and_direct_sdk_usage(provider, usage, read, write, miss):
    model = tpa.get_model(provider, "unused", cache_model=False)
    metrics = RunMetrics()
    for _ in range(2):
        response = ModelResponse(response_usage=model._get_metrics(usage))
        accumulate_model_metrics(response, model, model.model_type, metrics)
    start_trace("cache-parity", "/agent/stream")
    record_text_usage("cache-parity", model.id, metrics, feature="chat")
    sdk_response = SimpleNamespace(usage=usage, usage_metadata=usage)
    record_provider_usage("cache-parity", "autocomplete", provider, model.id, sdk_response)
    shared, direct = finish_trace("cache-parity")["cache_usage"]
    for row, calls, feature in [(shared, 2, "chat"), (direct, 1, "autocomplete")]:
        assert row == {
            "provider": provider, "model": model.id, "feature": feature,
            "calls": calls, "read_reported_calls": calls, "hit_calls": calls,
            "write_reported_calls": calls if write is not None else 0,
            "input_tokens": calls * 100, "input_tokens_with_read_data": calls * 100,
            "cache_read_tokens": calls * read,
            "cache_write_tokens": None if write is None else calls * write,
            "cache_miss_tokens": None if miss is None else calls * miss,
        }


def test_missing_usage_and_reported_zero_are_distinct():
    start_trace("cache-unknown", "/agent/autocomplete")
    record_provider_usage("cache-unknown", "autocomplete", "openai", "gpt-4o-mini", SimpleNamespace())
    usage = CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                            prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0})
    record_provider_usage("cache-unknown", "autocomplete", "openai", "gpt-4o-mini", SimpleNamespace(usage=usage))
    unknown, miss = finish_trace("cache-unknown")["cache_usage"]
    assert (unknown["read_reported_calls"], unknown["cache_read_tokens"], unknown["cache_write_tokens"]) == (0, None, None)
    assert (miss["read_reported_calls"], miss["cache_read_tokens"], miss["cache_write_tokens"], miss["hit_calls"]) == (1, 0, 0, 0)


def test_stream_tracks_each_model_call_once_when_a_tool_runs():
    def lookup() -> str:
        """Look up a colour."""
        return "blue"

    requests = []
    model = tpa.get_model("openai", "unused", cache_model=False)

    def reply(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        delta = {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_one", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }]} if first else {"role": "assistant", "content": "Blue."}
        usage = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                 "prompt_tokens_details": {"cached_tokens": 0 if first else 60, "cache_write_tokens": 60 if first else 0}}
        frames = [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if first else "stop"}]},
            {"choices": [], "usage": usage},
        ]
        events = ["data: " + json.dumps({"id": f"response-{len(requests)}", "created": 0, "model": model.id,
                                        "object": "chat.completion.chunk", **frame}) for frame in frames]
        return httpx.Response(200, text="\n\n".join([*events, "data: [DONE]", ""]),
                              headers={"content-type": "text/event-stream"})

    with OpenAI(api_key="unused", http_client=httpx.Client(transport=httpx.MockTransport(reply))) as client:
        model.client = client
        agent = Agent(model=model, tools=[lookup], telemetry=False)
        start_trace("cache-tool-stream", "/agent/stream")
        content = "".join(tpa._agent_text(agent, "Look up a colour.", None, "openai", True, "cache-tool-stream"))
    (usage,) = finish_trace("cache-tool-stream")["cache_usage"]
    assert content == "Blue." and len(requests) == 2
    assert (usage["calls"], usage["hit_calls"], usage["read_reported_calls"]) == (2, 1, 2)
    assert (usage["input_tokens"], usage["cache_read_tokens"], usage["cache_write_tokens"]) == (200, 60, 60)
    assert all(not any(key.startswith("prompt_cache_") for key in request) for request in requests)


def test_autocomplete_records_cache_usage_in_request_summary(monkeypatch):
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("unused", {
        "user_id": "test", "requires_user_api_keys": True, "user_keys": None,
    })))
    mock_reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="<new>some help.</new>")],
        usage=Usage(input_tokens=30, output_tokens=10, cache_read_input_tokens=600, cache_creation_input_tokens=0),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: mock_reply))
    monkeypatch.setattr(tpa, "_get_cached_anthropic_client", lambda key: client)
    response = TestClient(api.app).post("/agent/autocomplete", json={"text_before": "I need", "api_key": "lic", "model": "claude"})
    assert response.status_code == 200
    summary = recent_summaries(1)[0]
    assert summary["request_id"] == response.headers["x-request-id"]
    (usage,) = summary["cache_usage"]
    assert usage["feature"] == "autocomplete" and usage["cache_read_tokens"] == 600
    assert usage["cache_write_tokens"] == 0 and usage["input_tokens"] == 630


def test_design_keeps_usage_when_provider_output_is_truncated(monkeypatch):
    response = SimpleNamespace(stop_reason="max_tokens", usage=Usage(
        input_tokens=30, output_tokens=10, cache_read_input_tokens=600, cache_creation_input_tokens=40,
    ))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(tpa, "_get_cached_anthropic_client", lambda key: client)
    start_trace("cache-design", "/agent/stream")
    with pytest.raises(ValueError, match="truncated"):
        tpa._run_design_json_completion(
            [{"role": "user", "content": "A card"}], None, 0.2, "claude", "unused", "cache-design",
        )
    (usage,) = finish_trace("cache-design")["cache_usage"]
    assert usage["feature"] == "design_canvas"
    assert (usage["cache_read_tokens"], usage["cache_write_tokens"], usage["input_tokens"]) == (600, 40, 670)
