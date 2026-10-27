"""
Live Mode 8 (editable UI design) checks against the local server (scripts/run_local_server.py): real models, real
routing, nothing outside this machine touched except the model providers.

  python scripts/local_design_requests.py              # all scenarios
  python scripts/local_design_requests.py 1 2 3        # only these (follow-ups need the scenario they edit)
  python scripts/local_design_requests.py --from local_run/design_live/run_X.json 10
                                                       # re-use run_X's designs for follow-ups; the new file holds
                                                       # run_X's results plus (or replacing) the ones run now

For every request it records the request body (keys redacted), every SSE event with client-side timings, the returned
DesignDocumentAST (re-validated here with the production validator), for follow-up edits a node-by-node diff against
the design it edited, and the server's own trace for the same request (provider and model per call, tokens, estimated
cost, harness attempts, auto-heal). Results go to local_run/design_live/run_<timestamp>.json.

The vision scenario uses local_run/images/design_ref_ask_anything.jpg, which the local server serves under its CDN_URL.
"""

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.design_canvas.schema import DesignDocumentAST          # noqa: E402
from src.design_canvas.validate import validate_design_ast      # noqa: E402

BASE = "http://127.0.0.1:8000"
LICENSE = "local-dev-license"
OUT_DIR = ROOT / "local_run" / "design_live"
REFERENCE = f"{BASE}/local-images/design_ref_ask_anything.jpg"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def stream(body, abort_after_init=False):
    """POST /agent/stream; returns events (named and typed) with the second each arrived."""
    t0 = time.perf_counter()
    events, pending_name, rid, abort_at = [], None, None, None
    with httpx.Client(timeout=300) as client, client.stream("POST", f"{BASE}/agent/stream", json=body) as r:
        rid = r.headers.get("X-Request-ID")
        if r.status_code != 200:
            return {"http": r.status_code, "error": r.read().decode()[:500], "rid": rid,
                    "total": round(time.perf_counter() - t0, 2), "events": []}
        for line in r.iter_lines():
            if line.startswith("event: "):
                pending_name = line[7:].strip()
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            now = round(time.perf_counter() - t0, 2)
            name = pending_name or data.get("type")
            pending_name = None
            events.append({"event": name, "t": now, "data": data})
            if name == "start":
                rid = data.get("request_id") or rid
            if abort_after_init and name == "init" and abort_at is None:
                abort_at = now
                threading.Thread(target=lambda: httpx.post(
                    f"{BASE}/agent/abort", json={"request_id": rid, "api_key": LICENSE}, timeout=10), daemon=True).start()
    return {"http": 200, "rid": rid, "total": round(time.perf_counter() - t0, 2), "events": events, "abort_at": abort_at}


def process(body):
    """POST /agent/process; it takes the provider's own key (the backend resolves and bills it), read from .env."""
    from config import settings
    key = {"claude": settings.default_anthropic_api_key, "openai": settings.default_openai_api_key,
           "gemini": settings.default_gemini_api_key}[body["model"]]
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/agent/process", json={**body, "api_key": key}, timeout=300)
    j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:500]}
    return {"http": r.status_code, "rid": r.headers.get("X-Request-ID"), "total": round(time.perf_counter() - t0, 2),
            "json": j}


def server_trace(rid):
    for _ in range(10):
        recent = httpx.get(f"{BASE}/agent/metrics?limit=200", timeout=10).json()["recent"]
        hit = next((s for s in recent if s.get("request_id") == rid), None)
        if hit:
            return hit
        time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Design helpers
# ---------------------------------------------------------------------------

def walk(node, depth=0):
    yield node, depth
    for child in node.get("children", []):
        yield from walk(child, depth + 1)


def nodes(design):
    return [n for n, _ in walk(design["root"])]


def texts(design):
    return [" ".join(s["text"] for s in n["spans"]) for n in nodes(design) if n["node_type"] == "text"]


def check_valid(design, existing=None):
    ast = DesignDocumentAST.model_validate(design)
    return validate_design_ast(ast, existing_ast_json=json.dumps(existing) if existing else None)


def _field_changes(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for k in sorted(set(before) | set(after)):
            if k == "children":
                continue
            yield from _field_changes(before.get(k), after.get(k), f"{path}.{k}" if path else k)
    elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for i, (b, a) in enumerate(zip(before, after)):
            yield from _field_changes(b, a, f"{path}[{i}]")
    elif before != after:
        yield {"field": path, "before": before, "after": after}


def diff(before, after):
    """Node-level diff of two designs, by id (the editor's handle on a node)."""
    b = {n["id"]: n for n in nodes(before)}
    a = {n["id"]: n for n in nodes(after)}
    changed = {}
    for nid in b:
        if nid in a:
            changes = list(_field_changes(b[nid], a[nid]))
            if changes:
                changed[nid] = changes
    return {"removed": [{"id": i, "node_type": b[i]["node_type"]} for i in b if i not in a],
            "added": [{"id": i, "node_type": a[i]["node_type"]} for i in a if i not in b],
            "changed": changed,
            "kept_unchanged": [i for i in b if i in a and i not in changed]}


def luminance(hex_color):
    h = hex_color.lstrip("#")[:6]
    def lin(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(h[0:2]) + 0.7152 * lin(h[2:4]) + 0.0722 * lin(h[4:6])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def stream_body(query, **extra):
    # model is what the user has selected in the editor; model_source=default lets the design route pick (claude first)
    return {"type": "general", "query": query, "model": "openai", "model_source": "default", "api_key": LICENSE, **extra}


def design_of(res):
    ev = next((e for e in res.get("events", []) if e["event"] == "design_ready"), None)
    return ev["data"] if ev else None


def names(res):
    return [e["event"] for e in res.get("events", [])]


SCENARIOS = {}


def scenario(n, title, kind, follows=None):
    def wrap(fn):
        SCENARIOS[n] = {"title": title, "kind": kind, "follows": follows, "fn": fn}
        return fn
    return wrap


@scenario(1, "Create from a prompt", "create")
def s1(prev):
    body = stream_body("Design an onboarding card UI for a meditation app: dark background, a calm two-tone headline, "
                       "a short description, a simple illustration of a moon over waves, and a 'Start breathing' button.")
    res = stream(body)
    d = design_of(res)
    checks = [("init, design_ready and complete events, in that order",
               [x for x in names(res) if x in ("init", "design_ready", "complete")] == ["init", "design_ready", "complete"]),
              ("design passes schema + every domain check", d is not None and check_valid(d) == []),
              ("has a button and a vector illustration",
               d is not None and {"button", "vector_illustration"} <= {n["node_type"] for n in nodes(d)})]
    return body, res, d, None, checks


@scenario(2, "Follow-up edit: change copy and colour", "edit", follows=1)
def s2(prev):
    before = prev[1]
    body = stream_body("Change the button text to 'Begin session' and make the headline mint green (#34F5A6).",
                       existing_design_ast=before)
    res = stream(body)
    d = design_of(res)
    dd = diff(before, d) if d else None
    buttons = [n for n in nodes(d) if n["node_type"] == "button"] if d else []
    h1 = [n for n in nodes(d) if n["node_type"] == "text" and n["tag"] in ("h1", "h2")] if d else []
    checks = [("design passes schema + every domain check (incl. id preservation)", d is not None and check_valid(d, before) == []),
              ("button now says 'Begin session'", any("Begin session" in b["label"] for b in buttons)),
              ("headline uses #34F5A6", any(s["color"].upper() == "#34F5A6" for n in h1[:1] for s in n["spans"])),
              ("no node was removed", dd is not None and dd["removed"] == [])]
    return body, res, d, {"edits": 1, "diff": dd}, checks


@scenario(3, "Follow-up edit: delete a node", "edit", follows=2)
def s3(prev):
    before = prev[2]
    body = stream_body("Remove the illustration.", existing_design_ast=before, type="edit_design")
    res = stream(body)
    d = design_of(res)
    dd = diff(before, d) if d else None
    b_nodes = {n["id"]: n for n in nodes(before)}
    illus_ids = {i for i, n in b_nodes.items() if n["node_type"] == "vector_illustration"}
    checks = [("design passes schema + every domain check", d is not None and check_valid(d, before) == []),
              ("no vector illustration left", d is not None and all(n["node_type"] != "vector_illustration" for n in nodes(d))),
              ("only the illustration (and an emptied wrapper) was removed",
               dd is not None and all(r["id"] in illus_ids or b_nodes[r["id"]]["node_type"] == "container" for r in dd["removed"])),
              ("accepted on the first attempt (deleting is not treated as an error)",
               (next((e for e in res["events"] if e["event"] == "complete"), {}).get("data", {}).get("attempts") == 1))]
    return body, res, d, {"edits": 2, "diff": dd}, checks


@scenario(4, "Recreate from a screenshot (JPEG)", "vision")
def s4(prev):
    body = stream_body("Recreate this screenshot as an editable design.", image_urls=[REFERENCE])
    res = stream(body)
    d = design_of(res)
    all_text = " ".join(texts(d)) if d else ""
    checks = [("design passes schema + every domain check", d is not None and check_valid(d) == []),
              ("headline, section heading and button copy recreated",
               all(t in all_text for t in ("Anything", "Open-ended questions")) and d is not None
               and any("Get started" in n.get("label", "") for n in nodes(d))),
              ("speech bubbles as an editable vector with several layers",
               d is not None and any(n["node_type"] == "vector_illustration" and len(n["layers"]) >= 2 for n in nodes(d))),
              ("both navigation arrows as icon buttons",
               d is not None and {"arrow_left", "arrow_right"} <= {n.get("icon_name") for n in nodes(d)}
               or d is not None and {"chevron_left", "chevron_right"} <= {n.get("icon_name") for n in nodes(d)}),
              ("dark background kept", d is not None and luminance(d["root"]["style"].get("background_color") or "#FFFFFF") < 0.1)]
    return body, res, d, None, checks


@scenario(5, "Follow-up edit on the screenshot design: light theme", "edit", follows=4)
def s5(prev):
    return _light_theme(prev)


@scenario(10, "Same light-theme follow-up, sent as type=edit_design", "edit", follows=4)
def s10(prev):
    return _light_theme(prev, type="edit_design")


def _light_theme(prev, **extra):
    before = prev[4]
    body = stream_body("Switch this to a light theme: white background and dark text. Keep the mint accents.",
                       existing_design_ast=before, **extra)
    res = stream(body)
    d = design_of(res)
    dd = diff(before, d) if d else None
    checks = [("design passes schema + every domain check (contrast checked on the new background)",
               d is not None and check_valid(d, before) == []),
              ("background is now light", d is not None and luminance(d["root"]["style"].get("background_color") or "#000000") > 0.8),
              ("node ids kept (the editor's selection survives)",
               dd is not None and len(dd["removed"]) == 0)]
    return body, res, d, {"edits": 4, "diff": dd}, checks


@scenario(6, "Routing: 'banner image' wording is a design, not a picture", "create")
def s6(prev):
    body = stream_body("Design a hero section with a banner image for a specialty coffee shop.")
    res = stream(body)
    d = design_of(res)
    contents = [e for e in res["events"] if e["event"] == "content"]
    checks = [("routed to the design engine (design_ready event)", d is not None),
              ("no raster image placeholder streamed", not any("placeholder::" in e["data"].get("content", "") for e in contents)),
              ("design passes schema + every domain check", d is not None and check_valid(d) == [])]
    return body, res, d, None, checks


@scenario(7, "/agent/process parity with type=generate_design (OpenAI)", "create")
def s7(prev):
    # /agent/process calls the given provider only (the backend resolved its key); OpenAI here also exercises gpt-5.2
    body = {"type": "generate_design", "model": "openai",
            "query": "A pricing card for a Pro plan at $12/month with three feature bullets and an Upgrade button."}
    res = process(body)
    d = (res.get("json") or {}).get("design")
    checks = [("HTTP 200", res["http"] == 200),
              ("design returned in the `design` field", d is not None),
              ("markdown_response carries the same AST as JSON",
               d is not None and json.loads(res["json"].get("markdown_response") or "{}").get("design_id") == d["design_id"]),
              ("design passes schema + every domain check", d is not None and check_valid(d) == [])]
    return {**body, "api_key": "***provider key from .env***"}, res, d, None, checks


@scenario(8, "Routing control: ordinary chat stays ordinary", "control")
def s8(prev):
    body = stream_body("Write a haiku about the ocean.")
    res = stream(body)
    text = "".join(e["data"].get("content", "") for e in res["events"] if e["event"] == "content")
    checks = [("answered as text", len(text.strip()) > 0),
              ("no design events", not ({"init", "design_ready"} & set(names(res))))]
    res["text"] = text
    return body, res, None, None, checks


@scenario(9, "/agent/abort while a design is generating", "abort")
def s9(prev):
    body = stream_body("Design a dashboard widget UI showing weekly revenue with a sparkline illustration.")
    res = stream(body, abort_after_init=True)
    after = round(res["total"] - (res.get("abort_at") or 0), 2)
    res["seconds_from_abort_to_stream_end"] = after
    checks = [("abort sent right after init", res.get("abort_at") is not None),
              ("no design delivered", "design_ready" not in names(res)),
              ("stream ended within 2 s of the abort", after < 2.0)]
    return body, res, None, None, checks


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main(selected, previous=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    designs, results = {}, []
    if previous:
        designs = {r["n"]: r["design"] for r in previous if r.get("design")}
    for n in sorted(selected):
        sc = SCENARIOS[n]
        if sc["follows"] and sc["follows"] not in designs:
            print(f"\n#{n} {sc['title']}: skipped (needs #{sc['follows']})")
            continue
        print(f"\n#{n} {sc['title']} ...", flush=True)
        try:
            body, res, design, edit, checks = sc["fn"](designs)
        except Exception as err:   # a scenario that crashes is a result too
            results.append({"n": n, "title": sc["title"], "kind": sc["kind"], "crash": repr(err)})
            print(f"   CRASH {err!r}")
            continue
        if design is not None:
            designs[n] = design
        srv = server_trace(res.get("rid")) if res.get("rid") else None
        req_body = {k: ("***redacted***" if k == "api_key" else v) for k, v in body.items()}
        if "existing_design_ast" in req_body:
            req_body["existing_design_ast"] = f"<design from #{sc['follows']}>"
        results.append({"n": n, "title": sc["title"], "kind": sc["kind"], "follows": sc["follows"],
                        "endpoint": "/agent/process" if n == 7 else "/agent/stream", "request": req_body,
                        "response": {k: v for k, v in res.items() if k != "json"} | (
                            {"json_keys": sorted((res.get("json") or {}).keys())} if "json" in res else {}),
                        "design": design, "edit": edit, "checks": [{"check": c, "ok": bool(ok)} for c, ok in checks],
                        "server": srv})
        for c, ok in checks:
            print(f"   {'PASS' if ok else 'FAIL'}  {c}")
        if srv:
            usage = srv.get("usage") or []
            print(f"   server: {srv.get('total_seconds')}s, calls={[u['model'] for u in usage]}, "
                  f"tokens={srv.get('tokens_in')}/{srv.get('tokens_out')}, cost=${srv.get('est_cost_usd') or 0:.4f}, "
                  f"attempts={[a['outcome'] for a in srv.get('design_attempts') or []]}")
    if previous:
        rerun = {r["n"] for r in results}
        results = sorted([r for r in previous if r["n"] not in rerun] + results, key=lambda r: r["n"])
    credits = httpx.get(f"{BASE}/local/credits", timeout=10).json()
    out = OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"base": BASE, "reference_image": REFERENCE, "results": results, "credits": credits},
                              indent=2, ensure_ascii=False))
    passed = sum(c["ok"] for r in results for c in r.get("checks", []))
    total = sum(len(r.get("checks", [])) for r in results)
    print(f"\n{passed}/{total} checks passed — {out}")


if __name__ == "__main__":
    args, previous = sys.argv[1:], None
    if "--from" in args:
        i = args.index("--from")
        previous = json.loads(Path(args[i + 1]).read_text())["results"]
        args = args[:i] + args[i + 2:]
    main([int(a) for a in args] or list(SCENARIOS), previous)
