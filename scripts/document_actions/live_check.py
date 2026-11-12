"""Live before/after check of the five document-action behaviours (real provider calls, no Redis, no billing).

  .venv/bin/python scripts/document_actions/live_check.py --label before --out <file.json>
  .venv/bin/python scripts/document_actions/live_check.py --label after  --out <file.json> [--provider openai|claude]

Drives the production /agent/process route through TestClient with the keys in .env, one case at a time, and records the
status, text, chosen mode and wall-clock latency of each. Local-only: not part of the test suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update({"REDIS_URL": "", "BACKEND_WEBHOOK_URL": "", "WEBHOOK_SECRET": "", "CREDIT_PRECHECK_ENABLED": "false",
                   "VOICE_FLOW_ENABLED": "false", "TASK_ROUTING_ENABLED": "false", "PROVIDER_FALLBACK_ENABLED": "false",
                   "LOG_REQUEST_BODIES": "false", "MODEL_PREFLIGHT_ON_START": "false"})

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from config import settings  # noqa: E402
from src.document_placement import resolve_placement  # noqa: E402

DOC = ("# Acme Annual Report\n\n"
       "## About\n\nAcme builds tools that help small teams plan their work.\n\n"
       "## Contact\n\nReach us during business hours.\n\n"
       "## Top 10 list\n\n1. Planning\n2. Reviews\n3. Reports\n\n"
       "## Office\n\nOur office is at 12 Park Lane, Springfield.\n\n"
       "> Quality means doing it right when no one is watching.\n\n"
       "## Pricing\n\nThe Team plan costs USD 20 per seat per month.\n")
CONTACT_ONLY = "# Acme\n\n## Contact\n\nReach us during business hours.\n\n## Careers\n\nWe hire in spring.\n"
THIN = "# Notes\n\nA short note.\n"


def cases():
    # (id, item, action, document, query)
    return [
        ("1a", "Draft: exact text only", "draft", CONTACT_ONLY, "add Acme Inc. below the Contact heading"),
        ("1b", "Draft: look-alike must still write", "draft", CONTACT_ONLY, "add a short company bio below the Contact heading"),
        ("1c", "Draft: exact product name", "draft", DOC, "add Acme Cloud below the Pricing heading"),
        ("1d", "Draft: exact phone number", "draft", DOC, "add 555-0100 below the Contact heading"),
        ("2a", "Edit: (2024) next to the title", "edit", DOC, "add (2024) next to the title"),
        ("2b", "Edit: phone number after the address", "edit", DOC, "add a phone number after the address"),
        ("2d", "Edit: a date after the title (no value)", "edit", DOC, "add a date after the title"),
        ("2c", "Edit: look-alike must still refuse", "edit", DOC, "add a short FAQ below the Pricing heading"),
        ("3a", "Edit: title", "edit", DOC, "make the title shorter"),
        ("3b", "Edit: heading", "edit", DOC, "rename the Contact heading to Get in touch"),
        ("3c", "Edit: quote", "edit", DOC, "make the quote shorter"),
        ("4a", "Draft: next to the Top 10 list", "draft", DOC, "add a note next to the Top 10 list"),
        ("4b", "Draft: beside the Office heading", "draft", DOC, "add a line beside the Office heading"),
        ("4c", "Draft: below the heading (none named)", "draft", CONTACT_ONLY, "add a line below the heading"),
        ("4d", "Draft: top of the document (look-alike)", "draft", DOC, "add a one-line summary at the top of the document"),
        ("5a", "Auto: question", "auto", DOC, "What does the Team plan cost?"),
        ("5b", "Auto: add text at a place", "auto", DOC, "add a short note about annual billing below the Pricing heading"),
        ("5c", "Auto: rewrite", "auto", DOC, "make the About section more concise"),
        ("5d", "Auto: write something standalone", "auto", THIN, "Write a Python function that checks if an email address is valid"),
        ("5e", "Auto: tiny addition", "auto", DOC, "add (2024) next to the title"),
        ("5g", "Auto: empty page", "auto", "", "Make it shorter"),
        ("5h", "Auto: unclear wording (needs the model)", "auto", DOC, "Write a closing sentence for the Conclusion"),
        ("5f", "Auto: large rewrite stays out of Edit-inline", "auto", DOC, "write a new section about our security practices after the Pricing section"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", default="openai", choices=["openai", "claude"])
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    key = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key}[args.provider]
    client = TestClient(api.app, raise_server_exceptions=False)
    rows = []
    for cid, item, action, doc, query in cases():
        if args.only and cid not in args.only.split(","):
            continue
        body = {"type": "general", "query": query, "model": args.provider, "api_key": key,
                "document_context": {"text": doc}, "document_action": action}
        started = time.perf_counter()
        r = client.post("/agent/process", json=body)
        seconds = round(time.perf_counter() - started, 2)
        row = {"id": cid, "item": item, "action": action, "query": query, "http": r.status_code, "seconds": seconds}
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code == 200:
            edit = data.get("document_edit")
            row.update(text=data.get("markdown_response") if not edit else None,
                       edit_status=edit and edit["status"], edit_message=edit and edit["message"],
                       patches=edit and [{"before": p["before"], "after": p["after"]} for p in edit["patches"]],
                       context=data.get("document_context") and {k: data["document_context"].get(k) for k in ("action", "status", "message", "placement")},
                       chosen=data.get("document_action_resolved"), needs_context=data.get("needs_context"))
        else:
            row["error"] = data.get("detail") or data.get("error")
        rows.append(row)
        print(f"[{cid}] {item}: HTTP {r.status_code} in {seconds}s", flush=True)
    # Placement is deterministic and offline: record it for the same phrases.
    placements = []
    for cid, item, action, doc, query in cases():
        if cid.startswith("4"):
            p, q = resolve_placement(doc, query)
            placements.append({"id": cid, "query": query, "placement": p and {"position": p.position, "anchor": p.text[:60]}, "question": q})
    Path(args.out).write_text(json.dumps({"label": args.label, "provider": args.provider, "rows": rows, "placements": placements}, indent=2, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
