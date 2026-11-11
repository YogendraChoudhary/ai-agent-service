"""An Enhance instruction that asks for a chart always gets graph_suggestion or a clear "can't chart this"."""
import json

import pytest

import api
from api import ProcessRequest
from config import settings
from src.charts.request import asks_for_chart, requested_chart_type
from tests.charts.test_charts import EXTRACTED, SOURCE
from tests.charts.test_integration import integration  # noqa: F401  (fixture)
from tests.design.test_design_api import parse_sse_events

PROSE = "Improved prose, separate from the suggested graph."


@pytest.mark.parametrize("query", [
    "Make a bar chart of this data", "Can you create a pie chart from the table?", "chart these numbers",
    "show me a graph of revenue by month", "Turn this table into a line graph", "Plot monthly sales",
    "please visualise these figures", "Add a chart below the Pricing heading", "show this as a chart",
])
def test_explicit_chart_requests_are_recognised(query):
    assert asks_for_chart(query)


@pytest.mark.parametrize("query", [
    "What chart should I use for this?", "Explain this graph", "How do I make a chart in Excel?",
    "Make a flowchart of the signup process", "Create an org chart for the team", "draw a gantt chart",
    "Write a blog post about charts", "summarize the plot of the novel", "help me chart a course for my career",
    "build a knowledge graph", "Make a table of these numbers", "Make it more persuasive",
])
def test_questions_diagrams_and_other_instructions_are_not_chart_requests(query):
    assert not asks_for_chart(query)


@pytest.mark.parametrize("query", ["Tighten the wording and chart the revenue", "Fix typos, then graph the results",
                                   "Polish it; also plot monthly signups", "Improve and add structure, chart the figures"])
def test_a_chart_verb_later_in_the_instruction_is_recognised(query):
    assert asks_for_chart(query, allow_writing=True)


@pytest.mark.parametrize("query", ["help me chart a course and plan the trip", "plan and chart our course",
                                   "the chart the team built is great"])
def test_chart_as_an_idiom_or_noun_is_not_a_request(query):
    assert not asks_for_chart(query, allow_writing=True)


def test_writing_verbs_do_not_opt_out_of_an_enhance_chart():
    assert not asks_for_chart("Rewrite this and add a chart")
    assert asks_for_chart("Rewrite this and add a chart", allow_writing=True)


@pytest.mark.parametrize("query, expected", [
    ("Make a pie chart", "pie"), ("donut chart please", "pie"), ("turn it into a line graph", "line"),
    ("chart these numbers", "bar"), ("make a bar chart", "bar"),
])
def test_requested_chart_type(query, expected):
    assert requested_chart_type(query) == expected


def enhance(**kw):
    return {"type": "enhance_with_ai", "query": SOURCE, "secondary_prompt": "Make it clearer and add a pie chart",
            "model": "openai", "api_key": "key", **kw}


def test_process_returns_prose_and_the_requested_chart_type(integration):  # noqa: F811
    client, _, calls, agents = integration
    result = client.post("/agent/process", json=enhance()).json()
    assert result["markdown_response"] == PROSE
    suggestion = result["graph_suggestions"][0]
    assert suggestion["status"] == "suggested" and suggestion["chart"]["presentation"]["chart_type"] == "pie"
    assert [p["value"] for p in suggestion["chart"]["data"]["points"]] == ["12000", "18000", "15000"]
    assert result["graph_suggestion_error"] is None and len(calls) == 1
    # The writer is told the chart comes separately, so it adds no table for it.
    assert "do not add a table" in " ".join(agents.created[0][3]["instructions"])


def test_stream_sends_prose_then_exactly_one_chart_event(integration):  # noqa: F811
    client, _, _, _ = integration
    events = parse_sse_events(client.post("/agent/stream", json=enhance()).text)
    charts = [e for e in events if e["event"] in ("graph_suggestion", "graph_suggestion_unavailable")]
    assert [e["event"] for e in charts] == ["graph_suggestion"]
    assert events.index(charts[0]) > next(i for i, e in enumerate(events) if e["data"].get("type") == "content")
    assert events[-1]["data"]["type"] == "done"


def test_no_numbers_is_not_chartable_without_an_extraction_call(integration):  # noqa: F811
    client, _, calls, _ = integration
    events = parse_sse_events(client.post("/agent/stream", json=enhance(query="Our store had a strong start.")).text)
    error = next(e["data"] for e in events if e["event"] == "graph_suggestion_unavailable")
    assert error["code"] == "not_chartable" and error["requested"] is True and "your text" in error["message"]
    assert not calls


@pytest.mark.parametrize("response, code", [
    (json.dumps({"points": []}), "not_chartable"),
    (json.dumps({**EXTRACTED, "points": [{**EXTRACTED["points"][0], "value_text": "99,000"}, *EXTRACTED["points"][1:]]}), "chart_failed"),
    ("not json", "chart_failed"),
])
def test_unusable_or_ungrounded_extraction_reports_why(integration, response, code):  # noqa: F811
    client, state, calls, _ = integration
    state["response"] = response
    result = client.post("/agent/process", json=enhance()).json()
    assert result["graph_suggestions"] == [] and result["markdown_response"] == PROSE
    assert result["graph_suggestion_error"]["code"] == code and len(calls) == 1


def test_provider_failure_is_a_clean_chart_failed_event(integration):  # noqa: F811
    client, state, _, _ = integration
    state["error"] = RuntimeError("secret provider detail sk-123")
    events = parse_sse_events(client.post("/agent/stream", json=enhance()).text)
    error = next(e["data"] for e in events if e["event"] == "graph_suggestion_unavailable")
    assert error["code"] == "chart_failed" and "sk-123" not in json.dumps(events)


def test_pie_of_negative_values_falls_back_to_bar(integration):  # noqa: F811
    client, state, _, _ = integration
    source = "Profit was -5 in January and 10 in February."
    state["response"] = json.dumps({"title": "Profit", "unit": "", "points": [
        {"label": "January", "value_text": "-5", "excerpt": "-5 in January"},
        {"label": "February", "value_text": "10", "excerpt": "10 in February"}]})
    result = client.post("/agent/process", json=enhance(query=source)).json()
    assert result["graph_suggestions"][0]["chart"]["presentation"]["chart_type"] == "bar"


def test_enhance_without_a_chart_instruction_keeps_the_silent_optional_suggestion(integration):  # noqa: F811
    client, _, calls, _ = integration
    result = client.post("/agent/process", json=enhance(query="Our store had a strong start.",
                                                        secondary_prompt="Make it clearer")).json()
    assert result["graph_suggestions"] == [] and result["graph_suggestion_error"] is None and not calls


@pytest.mark.parametrize("change", ["write_mode", "flag", "opt_out", "other_type"])
def test_write_mode_and_other_types_never_intercept(integration, monkeypatch, change):  # noqa: F811
    client, _, calls, agents = integration
    kw = {}
    if change == "write_mode":
        kw = {"type": "general", "query": "Make a bar chart: " + SOURCE, "secondary_prompt": None}
    elif change == "flag":
        monkeypatch.setattr(settings, "chart_suggestions_enabled", False)
    elif change == "opt_out":
        kw["suggest_graphs"] = False
    else:
        kw = {"type": "change_tone", "secondary_prompt": "formal, and add a bar chart"}
    assert api.chart_request_for(ProcessRequest(**enhance(**kw))) is None
    if change == "write_mode":
        result = client.post("/agent/process", json=enhance(**kw)).json()
        assert result["graph_suggestions"] == [] and agents.created and not calls
