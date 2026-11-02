"""Real Enhance routing and HTTP contracts, with only provider calls replaced."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.charts.runtime import stream_graph_suggestion
from tests.longform.fakes import FakeAgents
from tests.design.test_design_api import parse_sse_events
from tests.charts.test_charts import EXTRACTED, SOURCE


@pytest.fixture
def integration(monkeypatch):
    calls, agents = [], FakeAgents(single=lambda query: "Improved prose, separate from the suggested graph.")
    state = {"response": json.dumps(EXTRACTED), "error": None}

    class ExtractionAgent:
        def run(self, history, **kw):
            calls.append(history)
            if state["error"]:
                raise state["error"]
            return SimpleNamespace(content=state["response"], metrics=None)

    def factory(agent_type, *args, **kwargs):
        return ExtractionAgent() if agent_type == "chart_extractor" else agents(agent_type, *args, **kwargs)

    for name, value in [("chart_suggestions_enabled", True), ("longform_refine_enabled", False),
                        ("task_routing_enabled", False), ("provider_fallback_enabled", False)]:
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(tpa, "create_agent", factory)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("key", {
        "user_id": "u1", "requires_user_api_keys": True, "user_keys": {}})))
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", False)
    monkeypatch.setattr(api, "credit_manager", None)
    monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
    return TestClient(api.app), state, calls, agents


def body(**kw):
    return {"type": "enhance_with_ai", "query": SOURCE, "model": "openai", "api_key": "key", **kw}


def test_process_returns_optional_suggestion_without_inserting_graph_into_prose(integration):
    client, _, calls, _ = integration
    response = client.post("/agent/process", json=body())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["markdown_response"] == "Improved prose, separate from the suggested graph."
    assert result["graph_suggestions"][0]["status"] == "suggested"
    assert result["graph_suggestions"][0]["chart"]["data"]["points"][1]["value"] == "18000"
    assert "<svg" in result["graph_suggestions"][0]["preview_svg"]
    assert SOURCE in calls[0][0].content
    assert result["graph_suggestion_error"] is None


def test_stream_has_a_separate_suggestion_after_content_and_before_done(integration):
    client, _, _, _ = integration
    response = client.post("/agent/stream", json=body())
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    graph = next(e for e in events if e["event"] == "graph_suggestion")
    assert graph["data"]["chart"]["data"]["points"][0]["value"] == "12000"
    assert events.index(graph) > next(i for i, e in enumerate(events) if e["data"].get("type") == "content")
    assert events[-1]["data"]["type"] == "done"
    assert all("preview_svg" not in e["data"].get("content", "") for e in events)


def test_protected_tables_are_extracted_from_original_before_freezing(integration, monkeypatch):
    client, state, calls, _ = integration
    table = "Monthly revenue (USD).\n\n| Month | Revenue |\n|---|---:|\n| January | 12,000 |\n| February | 18,000 |\n| March | 15,000 |\n"
    points = [{"label": p["label"], "value_text": p["value_text"],
               "excerpt": f'| {p["label"]} | {p["value_text"]} |'} for p in EXTRACTED["points"]]
    state["response"] = json.dumps({**EXTRACTED, "points": points})
    monkeypatch.setattr(settings, "longform_refine_enabled", True)
    result = client.post("/agent/process", json=body(query=table)).json()
    assert result["markdown_response"] == table
    assert result["graph_suggestions"][0]["chart"]["data"]["points"][2]["value"] == "15000"
    assert "12,000" in calls[0][0].content and "[[TABLE_" not in calls[0][0].content


@pytest.mark.parametrize("disabled", ["flag", "request", "other_type", "no_data"])
def test_disabled_optout_other_types_and_prose_without_data_make_no_extraction_call(integration, monkeypatch, disabled):
    client, _, calls, _ = integration
    kw = {}
    if disabled == "flag":
        monkeypatch.setattr(settings, "chart_suggestions_enabled", False)
    elif disabled == "request":
        kw["suggest_graphs"] = False
    elif disabled == "other_type":
        kw["type"] = "fix_grammar"
    else:
        kw["query"] = "Our store had a strong start to the year."
    result = client.post("/agent/process", json=body(**kw)).json()
    assert result["success"] and not result["graph_suggestions"]
    assert calls == []


@pytest.mark.parametrize("bad_response", ["not JSON", json.dumps({"points": []}),
                                          json.dumps({**EXTRACTED, "unit": "EUR"})])
def test_missing_or_invalid_data_never_breaks_enhance(integration, bad_response):
    client, state, _, _ = integration
    state["response"] = bad_response
    result = client.post("/agent/process", json=body()).json()
    assert result["success"]
    assert result["graph_suggestions"] == []
    assert result["markdown_response"] == "Improved prose, separate from the suggested graph."
    if bad_response != json.dumps({"points": []}):
        assert result["graph_suggestion_error"]["code"] == "graph_suggestion_unavailable"


def test_provider_failure_is_optional_and_has_no_sensitive_error_text(integration):
    client, state, _, _ = integration
    state["error"] = RuntimeError("do not disclose this internal provider detail")
    response = client.post("/agent/stream", json=body())
    events = parse_sse_events(response.text)
    assert any(e["event"] == "graph_suggestion_unavailable" for e in events)
    assert events[-1]["data"]["type"] == "done"
    assert "do not disclose" not in response.text


def test_abort_prevents_extraction_call():
    stop = threading.Event()
    stop.set()
    assert list(stream_graph_suggestion(SOURCE, candidates=[], key_for=lambda p: pytest.fail("key requested"),
                                        is_byok=True, cancel_event=stop)) == []


def test_graph_usage_does_not_overwrite_the_prose_trace(integration):
    from src.longform.llm import AgnoLLM, CallSpec
    from src.observability import start_trace, finish_trace
    from src.task_router import Candidate
    trace = start_trace("graph-usage-check", "test", model_used="primary-prose-model", llm_calls=[{"stage": "chunk"}])
    llm = AgnoLLM([Candidate("openai", True)], lambda p: "key", request_id="graph-usage-check", agent_type="chart_extractor")
    try:
        llm("extract", [{"role": "user", "content": SOURCE}], CallSpec("graph_extract", 0, 1000))
        assert trace.attrs["model_used"] == "primary-prose-model"
        assert trace.attrs["llm_calls"] == [{"stage": "chunk"}]
        assert trace.attrs["graph_llm_calls"][0]["stage"] == "graph_extract"
        assert trace.attrs["graph_model_used"].startswith("openai/")
    finally:
        finish_trace("graph-usage-check")


def test_create_then_type_color_and_value_edits_round_trip_over_http(integration):
    client, _, _, _ = integration
    suggestion = client.post("/agent/process", json=body()).json()["graph_suggestions"][0]
    original = suggestion["chart"]
    current = original
    for patch in [{"chart_type": "line"}, {"color": "#FF0000"}, {"chart_type": "pie"}, {"chart_type": "bar"}]:
        response = client.post("/agent/charts/edit", json={"chart": current, "patch": patch})
        assert response.status_code == 200, response.text
        current = response.json()["chart"]
        assert current["data"] == original["data"]
        assert current["chart_id"] == original["chart_id"]
    edited = client.post("/agent/charts/edit", json={"chart": current, "patch": {"values": {"p2": "19000"}}}).json()["chart"]
    assert edited["data"]["points"][0] == original["data"]["points"][0]
    assert edited["data"]["points"][2] == original["data"]["points"][2]
    assert edited["data"]["points"][1]["value"] == "19000"
    assert edited["data"]["points"][1]["source"] == original["data"]["points"][1]["source"]
