"""Exact phrases, HTML safety, Unicode ranges, bounded repair and API routing."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.key_points import KeyPointValidationError, Selection, highlight_key_points, resolve_highlights
from tests.design.test_design_api import parse_sse_events

SOURCE = "The launch is scheduled for 15 November. All teams must complete testing by 10 November. The approved budget is USD 20,000. This note is for internal planning."
POINTS = ["launch is scheduled for 15 November", "complete testing by 10 November", "approved budget is USD 20,000"]


def test_exact_source_excerpts_are_ordered_and_wrapped_without_changing_text():
    result = resolve_highlights(SOURCE, Selection(key_points=list(reversed(POINTS))))
    assert result.key_points == POINTS
    assert result.highlighted_html.count("<mark>") == 3
    assert result.highlighted_html.replace("<mark>", "").replace("</mark>", "") == '<span style="white-space:pre-wrap">'+SOURCE+'</span>'
    for span in result.spans:
        assert SOURCE.encode("utf-16-le")[span.start_utf16*2:span.end_utf16*2].decode("utf-16-le") == span.text


@pytest.mark.parametrize("source,points", [
    (SOURCE, ["USD 25,000"]), (SOURCE, ["launch is scheduled"] * 2),
    (SOURCE, ["launch is scheduled", "launch is scheduled for 15 November"]),
    (SOURCE, ["   "]), (SOURCE, ["aunch"]),
    ("repeat repeat", ["repeat"]), ("aaaa", ["aa"]),
    (SOURCE, ["20,00"]),
    ("We won't launch until testing finishes.", ["won"]),
    ("The release_id is recorded for future use.", ["release"]),
    ("The cafe\u0301 opens on Friday.", ["cafe"]),
])
def test_rewritten_ambiguous_overlapping_and_fragment_highlights_are_rejected(source, points):
    with pytest.raises(ValueError):
        resolve_highlights(source, Selection(key_points=points))


@pytest.mark.parametrize("source,points", [
    ("Revenue grew 20%, costs fell 10%.", ["Revenue grew 20%", "costs fell 10%"]),
    ("The deadline is Friday.", ["The deadline is Friday."]),
])
def test_dense_short_passages_can_retain_all_essential_facts(source, points):
    assert resolve_highlights(source, Selection(key_points=points)).key_points == points


@pytest.mark.parametrize("quote", ["'", "’"])
def test_quoted_whole_words_are_not_mistaken_for_contraction_fragments(quote):
    source=f"The event is called {quote}Launch{quote} and starts on Friday."
    assert resolve_highlights(source, Selection(key_points=["Launch"])).key_points==["Launch"]


def test_safe_html_and_unicode_offsets():
    source = '🚀 Launch <script>alert("x")</script> on Friday. Keep all other words unchanged.'
    result = resolve_highlights(source, Selection(key_points=["on Friday"]))
    assert "<script>" not in result.highlighted_html
    assert "&lt;script&gt;" in result.highlighted_html
    span, = result.spans
    assert span.start_utf16 == source.index("on Friday") + 1
    assert source.encode("utf-16-le")[span.start_utf16*2:span.end_utf16*2].decode("utf-16-le") == "on Friday"


def test_empty_selection_is_a_valid_noop():
    result = resolve_highlights("Hello there!", Selection(key_points=[]))
    assert result.key_points == result.spans == []
    assert "<mark>" not in result.highlighted_html


def test_selection_count_is_bounded():
    with pytest.raises(ValueError):
        Selection(key_points=[str(n) for n in range(6)])


def test_repair_is_bounded_and_cannot_invent_a_rewritten_point():
    calls = []
    def model(system, messages, spec):
        calls.append(spec.stage)
        return json.dumps({"key_points": ["invented quote"]}) if spec.stage == "key_points" else json.dumps({"key_points": POINTS})
    assert highlight_key_points(SOURCE, model).key_points == POINTS
    assert calls == ["key_points", "key_points_repair"]
    calls.clear()
    def invalid(*args):
        calls.append(1)
        return "not JSON"
    with pytest.raises(KeyPointValidationError):
        highlight_key_points(SOURCE, invalid)
    assert len(calls) == 2


def test_abort_during_invalid_first_response_does_not_start_repair():
    stop = threading.Event()
    calls = []
    def model(*args):
        calls.append(1)
        stop.set()
        return "not JSON"
    assert highlight_key_points(SOURCE, model, cancel_event=stop) is None
    assert len(calls) == 1


@pytest.fixture
def runtime(monkeypatch):
    state = {"reply": json.dumps({"key_points": POINTS}), "calls": [], "error": None}
    def factory(agent_type, provider, key, rid=None, **kwargs):
        assert agent_type == "key_point_highlighter"
        assert not kwargs["plan"].advanced_model and not kwargs["plan"].use_tools
        state["calls"].append((provider,key))
        class Model:
            def run(self, history, **kw):
                if state["error"]:
                    raise state["error"]
                assert json.loads(history[0].content)["source_text"] == state.get("source", SOURCE)
                return SimpleNamespace(content=state["reply"], metrics=None)
        return Model()
    monkeypatch.setattr(tpa,"create_agent",factory)
    monkeypatch.setattr(settings,"task_routing_enabled",False)
    monkeypatch.setattr(settings,"provider_fallback_enabled",False)
    monkeypatch.setattr(api,"CREDIT_PRECHECK_ENABLED",False)
    monkeypatch.setattr(api,"credit_manager",None)
    monkeypatch.setattr(api,"async_validate_and_get_api_key",AsyncMock(return_value=("provider-key",{
        "user_id":"u1","requires_user_api_keys":True,"user_keys":{}})))
    monkeypatch.setattr(api,"fire_prompt_history_save",lambda *a,**kw:None)
    return TestClient(api.app,raise_server_exceptions=False),state


def body(**kw):
    return {"type":"highlight_key_points","query":SOURCE,"model":"openai","api_key":"key",**kw}


def test_process_returns_original_text_and_highlight_metadata(runtime):
    client,state=runtime
    response=client.post("/agent/process",json=body())
    assert response.status_code==200,response.text
    result=response.json()
    assert result["markdown_response"]==result["html_response"]==SOURCE
    assert result["highlighting"]["key_points"]==POINTS
    assert len(result["highlighting"]["spans"])==3
    assert state["calls"]==[("openai","key")]


def test_stream_emits_highlights_before_unchanged_content_and_done(runtime):
    client,_=runtime
    response=client.post("/agent/stream",json=body())
    events=parse_sse_events(response.text)
    assert [e["event"] or e["data"].get("type") for e in events]==["start","highlight_ready","content","done"]
    assert events[1]["data"]["key_points"]==POINTS
    assert events[2]["data"]["content"]==SOURCE


@pytest.mark.parametrize("invalid", [{"query":" "},{"query":"x"*10001},{"secondary_prompt":"rewrite"},
    {"image_provider":"openai"},{"existing_design_ast":{}},{"document_text":"Some document."}])
def test_invalid_requests_do_not_call_a_provider(runtime,invalid):
    client,state=runtime
    response=client.post("/agent/process",json=body(**invalid))
    assert response.status_code==400,response.text
    assert not state["calls"]


def test_invalid_output_fails_before_any_text_or_highlights_are_emitted(runtime):
    client,state=runtime
    state["reply"]="invalid JSON"
    response=client.post("/agent/process",json=body())
    assert response.status_code==422
    assert len(state["calls"])==2
    state["calls"].clear()
    events=parse_sse_events(client.post("/agent/stream",json=body()).text)
    assert [e["data"].get("type") for e in events]==["start","error"]
    assert len(state["calls"])==2


def test_source_ending_in_a_code_fence_is_not_stripped(runtime):
    client,state=runtime
    state["source"]="A note with no important points.\n\n```text\nunchanged\n```"
    state["reply"]='{"key_points":[]}'
    result=client.post("/agent/process",json=body(query=state["source"])).json()
    assert result["markdown_response"]==state["source"]


def test_provider_failure_remains_an_error(runtime):
    client,state=runtime
    state["error"]=RuntimeError("provider unavailable")
    assert client.post("/agent/process",json=body()).status_code==500
