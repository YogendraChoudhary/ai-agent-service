"""Exercise Enhance -> optional graph -> scoped edits, without frontend code.

  .venv/bin/python scripts/demo_editable_graph.py          # scripted provider; no network
  .venv/bin/python scripts/demo_editable_graph.py --live   # real configured OpenAI model

Both modes use the real HTTP routes, extraction validator and SVG renderer. Redis,
storage, credit hooks and app startup are excluded. Receipts contain no provider key.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update({
    "REDIS_URL": "", "BACKEND_WEBHOOK_URL": "", "WEBHOOK_SECRET": "",
    "CREDIT_PRECHECK_ENABLED": "false", "VOICE_FLOW_ENABLED": "false",
    "CHART_SUGGESTIONS_ENABLED": "true", "LONGFORM_REFINE_ENABLED": "false",
    "TASK_ROUTING_ENABLED": "false", "PROVIDER_FALLBACK_ENABLED": "false",
    "LOG_REQUEST_BODIES": "false",
})

from agno.agent import RunEvent
from fastapi.testclient import TestClient
import api
from config import settings
from src import text_processing_agent as tpa

SOURCE = ("Our online store's revenue was USD 12,000 in January, USD 18,000 in February, "
          "and USD 15,000 in March. February had the highest revenue of the three months.")
EVIDENCE = {"title": "Monthly revenue", "unit": "USD", "points": [
    {"label": month, "value_text": value, "excerpt": f"USD {value} in {month}"}
    for month, value in [("January", "12,000"), ("February", "18,000"), ("March", "15,000")]
]}


def scripted_agent(agent_type, *args, **kwargs):
    class ScriptedAgent:
        def run(self, history, stream=False, **kw):
            if agent_type == "chart_extractor":
                return SimpleNamespace(content=json.dumps(EVIDENCE), metrics=None)
            prose = "Our store generated USD 12,000 in January, USD 18,000 in February and USD 15,000 in March. February led the three months."
            return iter([SimpleNamespace(event=RunEvent.run_content, content=prose)])
    return ScriptedAgent()


def run(live=False):
    logging.getLogger().setLevel(logging.WARNING)
    out = ROOT / "local_run" / "editable_graph_demo" / ("live" if live else "scripted")
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    client = TestClient(api.app, raise_server_exceptions=False)

    def post(path, body):
        start = time.perf_counter()
        response = client.post(path, json=body)
        record = {"path": path, "request": {**body, **({"api_key": "[redacted]"} if "api_key" in body else {})},
                  "http_status": response.status_code, "seconds": round(time.perf_counter() - start, 3),
                  "request_id": response.headers.get("x-request-id"), "response": response.json()}
        receipts.append(record)
        (out / "receipts.json").write_text(json.dumps({"live_provider": live, "requests": receipts}, indent=2))
        if response.status_code != 200:
            raise RuntimeError(f"{path} failed; see {out / 'receipts.json'}")
        return record["response"]

    key = settings.default_openai_api_key if live else "scripted-provider-key"
    if not key:
        raise RuntimeError("DEFAULT_OPENAI_API_KEY is required for --live")
    with nullcontext() if live else patch.object(tpa, "create_agent", scripted_agent):
        enhanced = post("/agent/process", {"type": "enhance_with_ai", "query": SOURCE,
                        "secondary_prompt": "Polish this for a business report.", "model": "openai",
                        "api_key": key, "suggest_graphs": True})
    if not enhanced["graph_suggestions"]:
        raise RuntimeError(f"The model returned no verified suggestion; inspect {out / 'receipts.json'}")
    original = enhanced["graph_suggestions"][0]
    chart = original["chart"]
    assert [(p["label"], p["value"]) for p in chart["data"]["points"]] == [("January", "12000"), ("February", "18000"), ("March", "15000")]
    assert chart["data"]["unit"] == "USD"
    line = post("/agent/charts/edit", {"chart": chart, "patch": {"chart_type": "line"}})
    red = post("/agent/charts/edit", {"chart": line["chart"], "instruction": "change color to red"})
    pie = post("/agent/charts/edit", {"chart": red["chart"], "patch": {"chart_type": "pie"}})
    for view in [line, red, pie]:
        assert view["chart"]["data"] == chart["data"]
        assert view["chart"]["chart_id"] == chart["chart_id"]
    changed = post("/agent/charts/edit", {"chart": red["chart"], "patch": {"values": {"p2": "19000.25"}}})
    assert changed["chart"]["data"]["points"][0] == chart["data"]["points"][0]
    assert changed["chart"]["data"]["points"][2] == chart["data"]["points"][2]
    assert changed["chart"]["data"]["points"][1]["value"] == "19000.25"
    assert changed["chart"]["presentation"] == red["chart"]["presentation"]
    cards = []
    for title, view in [("Suggested bar graph", original), ("Type changed to line — same data", line),
                        ("Color changed to red — same data and type", red), ("Type changed to pie — same data and color", pie),
                        ("Explicit data edit — only February changed", changed)]:
        cards.append(f'<section><h2>{html.escape(title)}</h2>{view["preview_svg"]}<details><summary>Editable chart document</summary><pre>{html.escape(json.dumps(view["chart"], indent=2))}</pre></details></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Editable graph Python demonstration</title><style>body{font:16px/1.5 Arial,sans-serif;color:#182434;max-width:1000px;margin:40px auto;padding:0 24px}'
            'section{border-top:1px solid #cbd5e1;padding:22px 0}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:16px}h2{font-size:21px}</style>'
            '<h1>Editable graph consistency demonstration</h1>'
            f'<p>Provider mode: {"real configured OpenAI model" if live else "scripted model, no network"}. All previews and edits use the production Python routes.</p>'
            '<h2>Original content</h2><p>' + html.escape(SOURCE) + '</p><h2>Enhanced text</h2><p>' + html.escape(enhanced["markdown_response"]) + '</p>'
            '<p>The optional suggestion is separate from the prose. A frontend must offer Insert and Dismiss; this page displays Python API results only.</p>'
            + ''.join(cards) + '<p>All type/color edits preserved the complete data object. The final example deliberately changes only p2 to 19000.25.</p>'
            '<p><a href="receipts.json">Actual request and response receipts, with provider key redacted</a></p></html>')
    (out / "example.html").write_text(page)
    (out / "original_chart.json").write_text(json.dumps(chart, indent=2))
    for name, view in [("bar", original), ("line", line), ("red_line", red), ("pie", pie)]:
        (out / f"{name}.svg").write_text(view["preview_svg"])
    print(json.dumps({"live_provider": live, "requests": len(receipts), "data_preserved_on_type_and_color_edits": True,
                      "one_explicit_value_edit_verified": True, "example": str(out / "example.html"),
                      "receipts": str(out / "receipts.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    run(parser.parse_args().live)
