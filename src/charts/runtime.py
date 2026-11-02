"""Optional suggestions after Enhance; failures never discard the improved document."""
from __future__ import annotations

import logging

from config import settings
from src.longform.llm import AgnoLLM
from src.observability import trace_set
from .extract import has_candidate_data, suggest_chart
from .render import graph_suggestion

logger = logging.getLogger(__name__)


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
