"""Charts after Enhance: optional suggestions, or the chart the instruction asked for; never written into the prose."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from config import settings
from src.longform.llm import AgnoLLM
from src.observability import trace_set
from .extract import has_candidate_data, suggest_chart
from .render import graph_suggestion
from .schema import ChartDocument, ChartType

logger = logging.getLogger(__name__)

NOT_CHARTABLE_MESSAGE = ("This can't be charted: no set of two or more comparable values, each stated with its "
                         "label, was found in {where}. Add or select the figures to chart and try again.")
CHART_FAILED_MESSAGE = "A chart could not be produced from this content. Nothing was changed; please retry."
REQUESTED_REASON = "Built only from values stated in your content. Review before inserting."


def stream_graph_suggestion(source, *, candidates, key_for, is_byok, user_id=None, request_id=None, cancel_event=None):
    if not has_candidate_data(source) or (cancel_event is not None and cancel_event.is_set()):
        return
    try:
        llm = AgnoLLM(candidates, key_for, is_byok=is_byok, user_id=user_id, request_id=request_id,
                      agent_type="chart_extractor", timeout=settings.chart_extraction_timeout_seconds)
        chart = suggest_chart(source, llm)
        if cancel_event is not None and cancel_event.is_set():
            return
        trace_set(request_id, graph_suggestion="suggested" if chart else "no_comparable_data")
        if chart:
            yield {"__event__": {"event": "graph_suggestion", "data": graph_suggestion(chart).model_dump(mode="json")}}
    except Exception as error:
        # No content or provider credentials in the public failure event.
        logger.warning("[%s] Optional graph suggestion unavailable (%s)", request_id, type(error).__name__)
        trace_set(request_id, graph_suggestion="unavailable", graph_error=type(error).__name__)
        yield {"__event__": {"event": "graph_suggestion_unavailable", "data": {
            "code": "graph_suggestion_unavailable", "message": "Your text was enhanced, but a verified graph suggestion could not be produced."}}}


def _unavailable(code, message):
    return {"__event__": {"event": "graph_suggestion_unavailable",
                          "data": {"code": code, "message": message, "requested": True}}}


def _with_type(chart: ChartDocument, chart_type: ChartType) -> ChartDocument:
    if chart_type == chart.presentation.chart_type:
        return chart
    data = chart.model_dump(mode="json")
    data["presentation"]["chart_type"] = chart_type
    try:
        return ChartDocument.model_validate(data)
    except ValidationError:
        return chart  # e.g. a pie of negative values: keep the default bar rather than refuse the chart


def stream_requested_chart(source, *, where, chart_type: ChartType = "bar", candidates, key_for, is_byok,
                           user_id=None, request_id=None, report_model=False, cancel_event=None):
    """
    An Enhance instruction asked for a chart: after the prose, emit exactly one `graph_suggestion` (the editable
    eddyter_chart) or one `graph_suggestion_unavailable` with code `not_chartable` / `chart_failed` and `requested: true`.
    """
    trace_set(request_id, chart_requested=True)
    if cancel_event is not None and cancel_event.is_set():
        return
    if not has_candidate_data(source):
        # Fewer than two numbers: no model call.
        trace_set(request_id, graph_suggestion="no_comparable_data")
        yield _unavailable("not_chartable", NOT_CHARTABLE_MESSAGE.format(where=where))
        return
    llm = AgnoLLM(candidates, key_for, is_byok=is_byok, user_id=user_id, request_id=request_id,
                  agent_type="chart_extractor", timeout=settings.chart_extraction_timeout_seconds,
                  cancel_event=cancel_event)
    try:
        chart = suggest_chart(source, llm)
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            return
        # Ungrounded evidence, timeouts and provider errors alike; no content or credentials in the public event.
        logger.warning("[%s] Requested chart unavailable (%s)", request_id, type(error).__name__)
        trace_set(request_id, graph_suggestion="unavailable", graph_error=type(error).__name__)
        yield _unavailable("chart_failed", CHART_FAILED_MESSAGE)
        return
    if cancel_event is not None and cancel_event.is_set():
        return
    if report_model and llm.used:
        yield {"__model_used__": llm.used}
    if chart is None:
        trace_set(request_id, graph_suggestion="no_comparable_data")
        yield _unavailable("not_chartable", NOT_CHARTABLE_MESSAGE.format(where=where))
        return
    trace_set(request_id, graph_suggestion="suggested")
    suggestion = graph_suggestion(_with_type(chart, chart_type)).model_copy(update={"reason": REQUESTED_REASON})
    yield {"__event__": {"event": "graph_suggestion", "data": suggestion.model_dump(mode="json")}}
