"""Explicit chart requests in an Enhance instruction ("add a bar chart of these numbers"); zero model calls."""
from __future__ import annotations

import re

from .schema import ChartType

_POLITE = re.compile(r"^(?:(?:please|can you|could you|would you|will you|kindly)\s+)+", re.I)
# Questions about charts are answered in chat, not charted: "what chart should I use?", "explain this graph".
_QUESTION = re.compile(r"^(?:what|which|why|how|when|where|who|is|are|does|do|did|should|explain|describe|"
                       r"summari[sz]e|analy[sz]e|interpret|review|critique|compare)\b", re.I)
_NOUN = r"(?:(?:bar|column|line|pie|donut|doughnut|area)\s+)?(?:charts?|graphs?|plots?)"
_VERB = (r"(?:make|create|draw|plot|generate|build|produce|render|show|give|add|insert|turn|convert|"
         r"visuali[sz]e|display|put|want|need|prepare)")
_REQUESTS = [
    # "make a bar chart of revenue", "show me a graph of the results", "turn this table into a pie chart"
    re.compile(r"\b" + _VERB + r"\b(?:[^.;!?\n]){0,80}?\b" + _NOUN + r"\b", re.I),
    # "chart these numbers", "graph the results", "plot monthly sales"
    re.compile(r"^(?:chart|graph|plot)\s+(?:this|these|that|those|it|them|the|my|our|monthly|yearly|annual|"
               r"quarterly|weekly|daily)\b", re.I),
    # "... as a chart", "... into a line graph", "visualise these figures"
    re.compile(r"\b(?:as|into|in)\s+(?:a|an)\s+" + _NOUN + r"\b", re.I),
    re.compile(r"\bvisuali[sz]e\s+(?:this|these|that|those|it|them|the|my|our)\b", re.I),
    # "tighten the wording and chart the revenue", "…, then graph the results", "also plot signups"
    re.compile(r"(?:\b(?:and|then|also|plus)|[,;])\s+(?:(?:please|also|then)\s+)?(?:chart|graph|plot)\s+(?:this|these|that|those|it|"
               r"them|the|my|our|monthly|yearly|annual|quarterly|weekly|daily|each|all)\b", re.I),
]
# Charts and graphs that are diagrams or idioms, not data series.
_NOT_DATA = re.compile(
    r"\b(?:flow|org|organi[sz]ational|organi[sz]ation|gantt|seating|star|birth|natal|astrolog\w*|nautical|tide|pert|"
    r"sankey|mind|process)[\s-]*(?:charts?|graphs?|maps?)\b|\bflowcharts?\b|\bknowledge\s+graphs?\b|"
    r"\bgraph\s+(?:theory|database|ql|api|neural)\b|\bgraphql\b|\bplot\s+(?:twist|line|of)\b|"
    r"\bchart\s+(?:a|your|our|the|my)\s+(?:course|path|way|future)\b", re.I)
# A request for written content that also wants a chart ("write a report and add a chart") is answered by the writer.
_WRITING = re.compile(r"\b(?:write|draft|compose|rewrite|summari[sz]e|explain)\b", re.I)


def asks_for_chart(query: str, *, allow_writing: bool = False) -> bool:
    """
    True when the user explicitly asks for a data chart or graph (not a question about one, nor a diagram).
    allow_writing: an Enhance instruction ("rewrite this and add a chart") is always about writing, so writing verbs
    do not opt out there.
    """
    text = _POLITE.sub("", query.strip())
    if not text or _QUESTION.match(text) or _NOT_DATA.search(text) or (not allow_writing and _WRITING.search(text)):
        return False
    return any(pattern.search(text) for pattern in _REQUESTS)


def requested_chart_type(query: str) -> ChartType:
    """The chart type the user named; bar when none was named."""
    if re.search(r"\b(?:pie|donut|doughnut)\b", query, re.I):
        return "pie"
    if re.search(r"\bline\s+(?:charts?|graphs?|plots?)\b|\btrend\s*line\b", query, re.I):
        return "line"
    return "bar"
