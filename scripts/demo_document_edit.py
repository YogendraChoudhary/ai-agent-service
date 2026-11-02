"""Selection-free editing, exact untouched-region proofs and stale-patch evidence.

  .venv/bin/python scripts/demo_document_edit.py          # scripted provider; no network
  .venv/bin/python scripts/demo_document_edit.py --live   # real configured OpenAI provider

Both modes exercise production HTTP, planning, validation and application code.
No Redis, licences, storage or external webhooks are contacted. Only synthetic documents are used.
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
os.environ.update({"REDIS_URL": "", "BACKEND_WEBHOOK_URL": "", "WEBHOOK_SECRET": "",
    "CREDIT_PRECHECK_ENABLED": "false", "VOICE_FLOW_ENABLED": "false", "TASK_ROUTING_ENABLED": "false",
    "PROVIDER_FALLBACK_ENABLED": "false", "LOG_REQUEST_BODIES": "false"})

from fastapi.testclient import TestClient
import api
from config import settings
from src import text_processing_agent as tpa
from src.document_edit.blocks import document_blocks, fingerprint

BLOG = ("# Launch guide\n\n## Introduction\n\nThis introduction is very very useful for readers who want a clear place to begin.\n\n"
        "## Benefits\n\nOur tool saves time for teams. Saving time is a benefit because it saves teams time.\n\n"
        "## Pricing\n\nThe price is USD 20 per month in 2026.\n\n"
        "| Plan | Monthly price |\n|---|---:|\n| Team | USD 20 |\n\n"
        "> Keep this customer quote exactly as written.\n\n![Product](https://example.org/product.png)\n\n"
        "```python\nprice = 20\n```\n\n## Conclusion\n\nStart with a small team and review the results before expanding.\n")
INSTRUCTION = "Make the Introduction more engaging and remove repetition in Benefits. Leave Pricing and Conclusion unchanged."
SEMANTIC = ("The opening explains why clear documentation matters.\n\n"
            "Our onboarding process is confusing and confusing for new users who are starting their first project.\n\n"
            "The closing recommendation is to review progress every month.\n")
SEMANTIC_INSTRUCTION = "Remove repetition from the paragraph about onboarding. Keep the other paragraphs exactly unchanged."
AMBIGUOUS = "## FAQ\n\nThe first answer stays here.\n\n## FAQ\n\nThe second answer stays here.\n"


def scripted_agent(agent_type, *args, **kwargs):
    assert agent_type == "document_editor"
    class Model:
        def run(self, history, **kw):
            data = json.loads(history[0].content)
            if "allowed_block_ids" in data:
                selected = [b for b in data["document"] if b["editable"] and
                            (b["section"][-1:] in (["Introduction"], ["Benefits"]) if b["section"] else "onboarding" in b["text"])]
                response = {"status":"ready", "message":"The matching paragraphs can be refined without changing other sections.",
                    "targets":[{"block_id":b["id"], "request_quote":data["USER_INSTRUCTION"],
                                "reason":"This paragraph contains the wording or repetition identified by the request."} for b in selected]}
                content = json.dumps(response)
            else:
                content = data["TARGET"].replace("very very useful", "particularly helpful").replace(
                    "Our tool saves time for teams. Saving time is a benefit because it saves teams time.",
                    "Our tool helps teams save time and focus on their most important work.").replace("confusing and confusing", "confusing")
            return SimpleNamespace(content=content, metrics=None)
    return Model()


def unchanged_proof(source, result, patches):
    """Check output slices independently, including gaps around and after every replacement."""
    blocks = {b.id: b for b in document_blocks(source)}
    old_at = new_at = 0
    proof = []
    for p in sorted(patches, key=lambda p: blocks[p["block_id"]].start):
        block = blocks[p["block_id"]]
        before = source[old_at:block.start]
        after = result[new_at:new_at + len(before)]
        assert before == after, "An untargeted region changed"
        proof.append({"original_start":old_at, "original_end":block.start, "characters":len(before),
                      "before_sha256":fingerprint(before), "after_sha256":fingerprint(after), "identical":True})
        new_at += len(before)
        assert result[new_at:new_at + len(p["after"])] == p["after"]
        new_at += len(p["after"])
        old_at = block.end
    assert source[old_at:] == result[new_at:]
    proof.append({"original_start":old_at, "original_end":len(source), "characters":len(source[old_at:]),
                  "before_sha256":fingerprint(source[old_at:]), "after_sha256":fingerprint(result[new_at:]), "identical":True})
    return proof


def run(live):
    logging.getLogger().setLevel(logging.WARNING)
    out = ROOT / "local_run" / "document_edit_demo" / ("live" if live else "scripted")
    out.mkdir(parents=True, exist_ok=True)
    client, receipts, examples = TestClient(api.app, raise_server_exceptions=False), [], []
    key = settings.default_openai_api_key if live else "scripted-provider-key"
    if not key:
        raise RuntimeError("--live needs the configured DEFAULT_OPENAI_API_KEY")

    def post(path, body):
        started = time.perf_counter()
        r = client.post(path, json=body)
        receipts.append({"path":path, "request":{**body, **({"api_key":"[redacted]"} if "api_key" in body else {})},
                         "http_status":r.status_code, "seconds":round(time.perf_counter()-started, 3),
                         "request_id":r.headers.get("x-request-id"), "response":r.json()})
        (out / "receipts.json").write_text(json.dumps({"live_provider":live,"requests":receipts}, indent=2))
        return r

    with nullcontext() if live else patch.object(tpa, "create_agent", scripted_agent):
        for title, source, instruction, status in [("Named blog sections", BLOG, INSTRUCTION, "proposed"),
            ("Paragraph located by meaning, without headings", SEMANTIC, SEMANTIC_INSTRUCTION, "proposed"),
            ("Ambiguous duplicate heading", AMBIGUOUS, "Improve FAQ", "needs_clarification")]:
            r = post("/agent/process", {"type":"general", "query":instruction, "document_text":source,
                                      "model":"openai", "api_key":key})
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["document_edit"]["status"] == status, f"Inspect {out / 'receipts.json'}; expected {status}"
            metadata, result = data["document_edit"], data["markdown_response"]
            editable = [b for b in document_blocks(source) if b.editable]
            expected = {b.id for b in editable if (b.section[-1:] in (("Introduction",), ("Benefits",))
                        if title == "Named blog sections" else "onboarding" in b.text)} if status == "proposed" else set()
            assert {p["block_id"] for p in metadata["patches"]} == expected, "The model selected the wrong paragraphs"
            proof = unchanged_proof(source, result, metadata["patches"])
            assert fingerprint(result) == metadata["result_sha256"]
            examples.append({"title":title, "source":source, "instruction":instruction, "result":result,
                             "metadata":metadata, "untouched_regions":proof})
    first = examples[0]
    source, metadata = first["source"], first["metadata"]
    accepted = post("/agent/documents/apply", {"document_text":source, "base_sha256":metadata["base_sha256"], "patches":metadata["patches"]})
    assert accepted.status_code == 200 and accepted.json()["document_text"] == first["result"]
    stale = post("/agent/documents/apply", {"document_text":source+"\nA newer edit.", "base_sha256":metadata["base_sha256"], "patches":metadata["patches"]})
    assert stale.status_code == 409
    (out / "proof.json").write_text(json.dumps({"examples":examples, "stale_edit_rejected":True}, indent=2))
    sections = []
    for example in examples:
        details = html.escape(json.dumps(example["metadata"], indent=2))
        sections.append('<section><h2>'+html.escape(example["title"])+'</h2><p><strong>Instruction:</strong> '+html.escape(example["instruction"])+
            '</p><p>Status: <strong>'+example["metadata"]["status"]+'</strong></p><div class="pair"><div><h3>Original</h3><pre>'+html.escape(example["source"])+
            '</pre></div><div><h3>Returned document</h3><pre>'+html.escape(example["result"])+
            '</pre></div></div><p>All untargeted regions verified byte for byte; hashes are in proof.json.</p><details><summary>Patch evidence</summary><pre>'+details+'</pre></details></section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Document section editing evidence</title><style>body{font:16px/1.5 Arial,sans-serif;max-width:1300px;margin:32px auto;padding:0 24px;color:#182434}section{border-top:1px solid #ccd5df;margin-top:28px;padding:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:18px}h3{font-size:17px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>Editing without a selection</h1><p>Provider: '+('real configured OpenAI model' if live else 'scripted model; no network')+'. These are production Python API results. No frontend editor was changed.</p>'+''.join(sections)+'<p>Accepted patches replayed successfully. A changed document was rejected with HTTP 409.</p><p><a href="receipts.json">Redacted requests and responses</a> · <a href="proof.json">Untouched-region hashes and proofs</a></p></html>'
    (out / "example.html").write_text(page)
    print(json.dumps({"live_provider":live, "examples":len(examples), "requests":len(receipts), "all_untargeted_regions_exact":True,
                      "stale_edit_rejected":True, "example":str(out / "example.html"), "proof":str(out / "proof.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    run(parser.parse_args().live)
