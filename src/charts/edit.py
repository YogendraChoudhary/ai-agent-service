"""Scoped edits: no generation call is allowed to rewrite an existing graph's data."""
from __future__ import annotations

import re

from .schema import ChartDocument, ChartPatch

_COLORS = {"red": "#DC2626", "blue": "#2563EB", "green": "#16A34A", "orange": "#EA580C",
           "purple": "#9333EA", "black": "#000000", "gray": "#6B7280", "grey": "#6B7280"}


def patch_from_instruction(instruction: str) -> ChartPatch:
    """A small, explicit language for presentation edits; unsupported requests fail without mutation."""
    changes = {}
    for clause in re.split(r"\s+(?:and|then)\s+|\s*[;,]\s*", instruction.strip().rstrip(".!"), flags=re.I):
        clause = re.sub(r"^please\s+", "", clause.strip(), flags=re.I)
        kind = re.fullmatch(r"(?:(?:change|switch|convert|set)\s+(?:(?:the|this)\s+)?(?:(?:graph|chart)\s+)?(?:type\s+)?(?:to\s+)?|make\s+it\s+(?:a\s+)?)(bar|line|pie)(?:\s+(?:graph|chart))?", clause, re.I)
        color = re.fullmatch(r"(?:change|set)\s+(?:(?:the|this)\s+)?(?:(?:graph|chart)\s+)?colou?r\s+to\s+(#[0-9a-f]{6}|red|blue|green|orange|purple|black|gr[ae]y)", clause, re.I)
        if kind:
            key, value = "chart_type", kind[1].lower()
        elif color:
            key, value = "color", _COLORS.get(color[1].lower(), color[1].upper())
        else:
            raise ValueError("Use 'change graph type to line' or 'change color to red' (bar/line/pie, named colors or #RRGGBB). Use an explicit patch to edit individual values, labels or colors.")
        if key in changes:
            raise ValueError("Conflicting or repeated edit; supply one change per field")
        changes[key] = value
    return ChartPatch.model_validate(changes)


def apply_patch(chart: ChartDocument, patch: ChartPatch) -> ChartDocument:
    raw = chart.model_dump(mode="json")
    ids = {p.id for p in chart.data.points}
    fields = patch.model_fields_set
    for name in ("values", "labels", "point_colors"):
        if name in fields and set(getattr(patch, name)) - ids:
            raise ValueError(f"{name} references an unknown point ID")
    for name in ("chart_type", "color"):
        if name in fields:
            raw["presentation"][name] = getattr(patch, name)
    if "point_colors" in fields:
        raw["presentation"]["point_colors"].update(patch.point_colors)
    for point in raw["data"]["points"]:
        if "values" in fields and point["id"] in patch.values:
            point["value"] = patch.values[point["id"]]
        if "labels" in fields and point["id"] in patch.labels:
            point["label"] = patch.labels[point["id"]]
    if "title" in fields:
        raw["title"] = patch.title
    if "unit" in fields:
        raw["data"]["unit"] = patch.unit
    # Revalidate the whole result atomically: a failed edit never mutates the caller's document.
    return ChartDocument.model_validate(raw)
