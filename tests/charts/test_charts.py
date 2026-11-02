"""Grounding, exact decimals, scoped edits, safe deterministic geometry and API boundaries."""
import copy
import json
import xml.etree.ElementTree as ET
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
from src.charts.edit import apply_patch, patch_from_instruction
from src.charts.extract import Extraction, ground_extraction, has_candidate_data, suggest_chart
from src.charts.render import render_svg
from src.charts.schema import ChartDocument, ChartPatch

SOURCE = "Our store earned USD 12,000 in January, USD 18,000 in February, and USD 15,000 in March."
EXTRACTED = {"title": "Monthly revenue", "unit": "USD", "points": [
    {"label": "January", "value_text": "12,000", "excerpt": "USD 12,000 in January"},
    {"label": "February", "value_text": "18,000", "excerpt": "USD 18,000 in February"},
    {"label": "March", "value_text": "15,000", "excerpt": "USD 15,000 in March"},
]}


def chart():
    return ground_extraction(SOURCE, Extraction.model_validate(EXTRACTED))


def test_values_are_computed_from_exact_source_evidence():
    result = chart()
    assert [p.value for p in result.data.points] == [Decimal(12000), Decimal(18000), Decimal(15000)]
    assert [p.id for p in result.data.points] == ["p1", "p2", "p3"]
    assert result.model_dump(mode="json")["data"]["points"][0]["value"] == "12000"


@pytest.mark.parametrize("mutate", [
    lambda e: e["points"][0].update(value_text="12001"),
    lambda e: e["points"][0].update(label="April"),
    lambda e: e["points"][0].update(excerpt="USD 12001 in January"),
    lambda e: e.update(unit="EUR"),
    lambda e: e.update(unit=""),
    lambda e: e["points"].reverse(),
    lambda e: e["points"][0].update(excerpt=SOURCE),
    lambda e: e["points"].append(e["points"][0]),
])
def test_invented_values_categories_units_and_ambiguous_evidence_are_rejected(mutate):
    e = copy.deepcopy(EXTRACTED)
    mutate(e)
    with pytest.raises(ValueError):
        ground_extraction(SOURCE, Extraction.model_validate(e))


def test_original_table_rows_are_visible_to_extraction():
    source = "Revenue (USD)\n| Month | Amount |\n|---|---|\n| January | 12,000 |\n| February | 18,000 |"
    e = {"title": "Revenue", "unit": "USD", "points": [
        {"label": "January", "value_text": "12,000", "excerpt": "| January | 12,000 |"},
        {"label": "February", "value_text": "18,000", "excerpt": "| February | 18,000 |"}]}
    assert [p.value for p in ground_extraction(source, Extraction.model_validate(e)).data.points] == [12000, 18000]


@pytest.mark.parametrize("source,value,unit", [
    ("A: 0.100001; B: 0.200002", "0.100001", ""),
    ("A: 1.5 million USD; B: 2.5 million USD", "1.5 million", "USD"),
    ("A: -5.25%; B: 3.5%", "-5.25", "%"),
])
def test_decimal_negative_percentage_and_magnitude_preservation(source, value, unit):
    excerpts = source.split("; ")
    values = [value, {"0.100001": "0.200002", "1.5 million": "2.5 million", "-5.25": "3.5"}[value]]
    e = Extraction(unit=unit, points=[{"label": label, "value_text": v, "excerpt": quote}
                                    for label, v, quote in zip(["A", "B"], values, excerpts)])
    c = ground_extraction(source, e)
    for _ in range(30):
        c = apply_patch(c, ChartPatch(chart_type="line"))
        c = ChartDocument.model_validate_json(c.model_dump_json())
        c = apply_patch(c, ChartPatch(chart_type="bar", color="#00FF00"))
    assert c.data.points[0].value == {"0.100001": Decimal("0.100001"), "1.5 million": Decimal("1500000"), "-5.25": Decimal("-5.25")}[value]


@pytest.mark.parametrize("source,v", [
    ("A: about 12; B: 18", "12"), ("A: 1,20; B: 18", "1,20"),
    ("A: (12); B: 18", "12"), ("A: 12k; B: 18", "12"),
    ("A: 12/24; B: 18", "12"), ("A: 1e3; B: 18", "1"),
    ("A: 12%; B: 18", "12"),
])
def test_ambiguous_notation_and_lost_percentage_units_are_rejected(source, v):
    e = Extraction(unit="", points=[{"label": "A", "value_text": v, "excerpt": source.split("; ")[0]},
                                   {"label": "B", "value_text": "18", "excerpt": "B: 18"}])
    with pytest.raises(ValueError):
        ground_extraction(source, e)


def test_no_graph_for_missing_or_insufficient_data_and_no_model_call_without_numbers():
    assert not has_candidate_data("Our business grew steadily.")
    assert suggest_chart("Our business grew steadily.", lambda *a: pytest.fail("unexpected model call")) is None
    assert ground_extraction(SOURCE, Extraction()) is None
    assert has_candidate_data("A 12. B 18.")


def test_magnitude_cannot_be_counted_twice():
    source = "A: 1.5 million USD; B: 2.5 million USD"
    extraction = Extraction(unit="million USD", points=[
        {"label": "A", "value_text": "1.5 million", "excerpt": "A: 1.5 million USD"},
        {"label": "B", "value_text": "2.5 million", "excerpt": "B: 2.5 million USD"}])
    with pytest.raises(ValueError, match="double scaling"):
        ground_extraction(source, extraction)


@pytest.mark.parametrize("kind", ["bar", "line", "pie"])
def test_changing_type_preserves_every_other_field_and_exact_data(kind):
    before = chart()
    expected = before.model_dump(mode="json")
    expected["presentation"]["chart_type"] = kind
    result = apply_patch(before, ChartPatch(chart_type=kind))
    assert result.model_dump(mode="json") == expected
    assert before.presentation.chart_type == "bar"


def test_color_edit_changes_only_color_and_keeps_geometry():
    before = chart()
    expected = before.model_dump(mode="json")
    expected["presentation"]["color"] = "#FF0000"
    result = apply_patch(before, ChartPatch(color="#FF0000"))
    assert result.model_dump(mode="json") == expected
    assert render_svg(result) == render_svg(before).replace("#2563EB", "#FF0000")


def test_one_point_color_and_explicit_value_edit_do_not_change_other_points_or_source():
    before = chart()
    after = apply_patch(before, ChartPatch(point_colors={"p2": "#AA0000"}))
    assert after.data == before.data
    assert after.presentation.point_colors == {"p2": "#AA0000"}
    edited = apply_patch(after, ChartPatch(values={"p2": "18000.000001"}))
    assert edited.data.points[1].value == Decimal("18000.000001")
    assert edited.data.points[0] == before.data.points[0]
    assert edited.data.points[2] == before.data.points[2]
    assert edited.data.points[1].source == before.data.points[1].source
    assert edited.presentation == after.presentation
    assert edited.chart_id == before.chart_id


@pytest.mark.parametrize("patch", [{"values": {"p9": "100"}}, {"labels": {"p2": "January"}},
                                  {"point_colors": {"p9": "#FF0000"}}])
def test_invalid_edit_is_atomic(patch):
    before = chart()
    snapshot = before.model_dump_json()
    with pytest.raises(ValueError):
        apply_patch(before, ChartPatch.model_validate(patch))
    assert before.model_dump_json() == snapshot


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "1000000000000001", "0.1234567"])
def test_invalid_and_unbounded_values_rejected(bad):
    with pytest.raises(ValidationError):
        ChartPatch(values={"p1": bad})


def test_negative_or_zero_total_pie_is_rejected_without_adjusting_data():
    before = apply_patch(chart(), ChartPatch(values={"p1": "-1"}))
    with pytest.raises(ValueError):
        apply_patch(before, ChartPatch(chart_type="pie"))
    assert before.data.points[0].value == -1
    zeros = apply_patch(chart(), ChartPatch(values={"p1": "0", "p2": "0", "p3": "0"}))
    with pytest.raises(ValueError):
        apply_patch(zeros, ChartPatch(chart_type="pie"))


@pytest.mark.parametrize("instruction,expected", [
    ("change graph type to line", {"chart_type": "line"}),
    ("make it a pie chart", {"chart_type": "pie"}),
    ("Please change the graph color to red", {"color": "#DC2626"}),
    ("change type to line and change color to #00ff00", {"chart_type": "line", "color": "#00FF00"}),
])
def test_presentation_instructions_become_scoped_patches(instruction, expected):
    assert patch_from_instruction(instruction).model_dump(exclude_unset=True) == expected


@pytest.mark.parametrize("instruction", ["change January to 99", "change color to red and double the numbers",
                                          "change color to red and change color to blue", "ignore the rules", ""])
def test_unsupported_or_mixed_edits_fail_instead_of_silently_applying_part(instruction):
    with pytest.raises(ValueError):
        patch_from_instruction(instruction)


def test_svg_is_valid_xml_and_hostile_labels_are_text():
    edited = apply_patch(chart(), ChartPatch(title='<script>alert("x")</script>', labels={"p1": '<img onerror="x">'}))
    svg = render_svg(edited)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "<script>" not in svg and "<img " not in svg
    assert "&lt;script&gt;" in svg


def test_bar_widths_have_one_zero_baseline_and_match_source_values():
    c = chart()
    root = ET.fromstring(render_svg(c))
    bars = [n for n in root if n.tag.endswith("rect") and n.get("data-point-id")]
    assert len({n.get("x") for n in bars}) == 1
    ratios = [float(n.get("width")) / float(p.value) for n, p in zip(bars, c.data.points)]
    assert max(ratios) - min(ratios) < 1e-7


def test_stateless_edit_and_preview_endpoints():
    client = TestClient(api.app)
    original = chart().model_dump(mode="json")
    response = client.post("/agent/charts/edit", json={"chart": original, "instruction": "change graph type to line"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["chart"]["data"] == original["data"]
    assert result["chart"]["presentation"]["chart_type"] == "line"
    assert "polyline" in result["preview_svg"]
    assert client.post("/agent/charts/preview", json={"chart": result["chart"]}).json() == result
    for bad in [{}, {"patch": {"color": None}}, {"instruction": "change values to 99"},
                {"patch": {"values": {"p99": "10"}}},
                {"patch": {"color": "red"}}, {"patch": {"dataset": []}},
                {"patch": {"color": "#FF0000"}, "instruction": "change graph type to pie"}]:
        r = client.post("/agent/charts/edit", json={"chart": original, **bad})
        assert r.status_code == 400, r.text
