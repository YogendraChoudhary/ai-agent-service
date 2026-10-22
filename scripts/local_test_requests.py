"""
End-to-end checks against the local server (scripts/run_local_server.py).

  python scripts/local_test_requests.py            # all scenarios
  python scripts/local_test_requests.py 3 5        # only scenarios 3 and 5

/agent/process takes the provider's own key, so those scenarios read the platform key from the settings (.env); the
local server never sees the licence for them. For every request it prints the request body (licence key redacted), the response, and the timings measured
FROM THE CLIENT (time to first byte, first token, each image, end of stream). It then pulls the server's own
trace for the same X-Request-ID from /agent/metrics so client and server numbers can be compared, and checks
what each scenario is supposed to prove (routing, model, image count, no dead placeholders).
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
LICENSE = "local-dev-license"
RUN_DIR = Path(__file__).resolve().parent.parent / "local_run"


def redact(body):
    return {k: ("***redacted***" if "key" in k else v) for k, v in body.items()}


def stream_request(client, body):
    t0 = time.perf_counter()
    ev, text, images, rid = [], [], [], None
    ttfb = first_token = done = ended = outline = warnings = None
    outline_at = None
    progress = []
    with client.stream("POST", f"{BASE}/agent/stream", json=body, timeout=240) as r:
        ttfb = time.perf_counter() - t0
        rid = r.headers.get("X-Request-ID")
        if r.status_code != 200:
            return {"http": r.status_code, "body": r.read().decode(), "rid": rid, "total": time.perf_counter() - t0}
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            e = json.loads(line[6:])
            now = time.perf_counter() - t0
            kind = e.get("type")
            if kind == "content":
                first_token = first_token or now
                text.append(e["content"])
            elif kind == "image_ready":
                images.append({"t": round(now, 2), "url": e["url"], "w": e.get("width"), "h": e.get("height")})
            elif kind == "outline":
                outline, outline_at = e, now
            elif kind == "refine_progress":
                progress.append(e)
            elif kind == "done":
                done = now
                warnings = e.get("warnings")
            elif kind == "stream_end":
                ended = now
            elif kind in ("error", "image_failed", "credit_limit_reached"):
                ev.append({"t": round(now, 2), **e})
    total = time.perf_counter() - t0
    return {"http": 200, "rid": rid, "text": "".join(text), "images": images, "events": ev,
            "ttfb": ttfb, "first_token": first_token, "done": done, "stream_end": ended, "total": total,
            "outline": outline, "outline_at": outline_at, "warnings": warnings, "progress": progress}


def provider_key(model):
    """/agent/process takes the PROVIDER's own key (the backend resolves it and bills the call), not a licence: read the platform key."""
    sys.path.insert(0, str(RUN_DIR.parent))
    from config import settings
    return {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key, "gemini": settings.default_gemini_api_key,
            "grok": settings.default_grok_api_key, "deepseek": settings.default_deepseek_api_key}.get(model)


def process_request(client, body):
    t0 = time.perf_counter()
    body = {**body, "api_key": provider_key(body.get("model")) or body["api_key"]}
    r = client.post(f"{BASE}/agent/process", json=body, timeout=240)
    out = {"http": r.status_code, "rid": r.headers.get("X-Request-ID"), "total": time.perf_counter() - t0,
           "server_ms_header": r.headers.get("X-Response-Time-Ms")}
    try:
        j = r.json()
    except ValueError:
        j = {"raw": r.text}
    out["json"] = j
    out["text"] = j.get("markdown_response", "")
    return out


def req(**kw):
    return {"api_key": LICENSE, **kw}


# (title, endpoint, body, checks)  checks: list of (description, fn(result, server_summary) -> bool)
def n_images(res): return len(res.get("images", []))
def has_model(res, srv, provider, model):
    return any(i.get("provider") == provider and i.get("model") == model and i.get("status") == "done" for i in (srv or {}).get("images", []))
def no_dead(res, srv): return "placeholder::" not in res.get("text", "")
def blog_tables(res): return len(re.findall(r"^\|.+\|[ \t]*\n\|[\s:|-]+\|[ \t]*$", res.get("text", ""), re.M))
def blog_quotes(res): return len(re.findall(r"^> ", res.get("text", ""), re.M))
def blog_h2(res): return len(re.findall(r"^## ", res.get("text", ""), re.M))

# Mode 1 (LONGFORM_REFINE_ENABLED, on in run_local_server.py): the project's own README is the document, because it is a real
# one: headings, lists, links, tables and code blocks. Whatever the model does to the prose, those must come back as they went.
DOC_PATH = Path(__file__).resolve().parent.parent / "README.md"
DOC = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
def fences(text): return re.findall(r"^```.*?^```", text, re.S | re.M)
def tables_of(text): return re.findall(r"(?:^\|.*\|[ \t]*\n)+", text, re.M)
def heading_levels(text):
    outside = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)           # a # inside a code block is a comment
    return [len(m) for m in re.findall(r"^(#{1,6}) ", outside, re.M)]
def urls_of(text): return sorted(u.rstrip(".,;:!?") for u in re.findall(r"https?://[^\s)>\]\"']+", text))
def kept(res, *what):
    return all(f(res["text"]) == f(DOC) for f in what)

SCENARIOS = [
    ("health", "GET /health", None, [("200 OK", lambda r, s: r["http"] == 200)]),

    ("plain writing: fast tier, no tools, no images", "stream",
     req(type="general", model="gemini", query="write a two-sentence description of a small coffee shop"),
     [("text returned", lambda r, s: len(r["text"]) > 40), ("no images", lambda r, s: n_images(r) == 0),
      ("agent route, planner granted no tools", lambda r, s: s and s.get("route") == "agent" and s["plan"] == {"images": 0, "web": False, "tier": "fast"})]),

    ("composite: 'with 2 images' -> exactly 2 images", "stream",
     req(type="general", model="gemini", query="write a short blog post (about 100 words) about a small coffee shop with 2 images"),
     [("exactly 2 images", lambda r, s: n_images(r) == 2), ("planner said 2 images", lambda r, s: s and s["plan"]["images"] == 2),
      ("drawn by gemini-3.1-flash-lite (default)", lambda r, s: has_model(r, s, "gemini", "gemini-3.1-flash-lite-image")),
      ("2 image placeholders in the text", lambda r, s: r["text"].count("placeholder::") == 2)]),

    ("direct image, no model named -> default Gemini lite", "stream",
     req(type="general", model="claude", query="draw a red bicycle leaning against a blue wall"),
     [("direct image route", lambda r, s: s and s.get("route") == "direct_image"), ("1 image", lambda r, s: n_images(r) == 1),
      ("default model gemini-3.1-flash-lite-image even with a Claude chat model", lambda r, s: has_model(r, s, "gemini", "gemini-3.1-flash-lite-image"))]),

    ("direct image at 4K -> routed to the model that really does 4K", "stream",
     req(type="general", model="gemini", query="draw a mountain lake poster in 4k"),
     [("drawn by gemini-3.1-flash-image (lite is 1K only)", lambda r, s: has_model(r, s, "gemini", "gemini-3.1-flash-image")),
      ("really larger than 1K", lambda r, s: r["images"] and max(r["images"][0]["w"] or 0, r["images"][0]["h"] or 0) >= 2048)]),

    ("direct image, 'using gemini 2.5' (explicit older model still selectable)", "stream",
     req(type="general", model="gemini", query="draw a teapot on a shelf using gemini 2.5"),
     [("drawn by gemini-2.5-flash-image", lambda r, s: has_model(r, s, "gemini", "gemini-2.5-flash-image"))]),

    ("direct image, 'using grok imagine' (regex picks the model)", "stream",
     req(type="general", model="gemini", query="draw a red bicycle leaning against a blue wall using grok imagine"),
     [("1 image", lambda r, s: n_images(r) == 1), ("drawn by grok-imagine-image", lambda r, s: has_model(r, s, "grok", "grok-imagine-image"))]),

    ("direct image, 'using gemini 3.1' -> exact model", "stream",
     req(type="general", model="openai", query="draw a lighthouse at dusk with gemini 3.1"),
     [("drawn by gemini-3.1-flash-image", lambda r, s: has_model(r, s, "gemini", "gemini-3.1-flash-image"))]),

    ("direct image, 'using gpt' (slowest provider)", "stream",
     req(type="general", model="gemini", query="draw a lighthouse at dusk using gpt"),
     [("drawn by gpt-image-1", lambda r, s: has_model(r, s, "openai", "gpt-image-1"))]),

    ("research request: web tools granted, advanced tier", "stream",
     req(type="general", model="gemini", query="What are the latest developments in solid-state batteries in 2026? Keep it to three short sentences."),
     [("planner granted web", lambda r, s: s and s["plan"]["web"] is True and s["plan"]["tier"] == "advanced"), ("answer returned", lambda r, s: len(r["text"]) > 60)]),

    ("REGRESSION: 'bullet points' in a research question used to hijack it to the no-web list agent", "stream",
     req(type="general", model="gemini", query="What are the latest developments in solid-state batteries in 2026? Answer in 3 bullet points."),
     [("free-form agent, not 'list'", lambda r, s: s and s.get("agent_type") == "smart_global"),
      ("planner granted web", lambda r, s: s and s["plan"]["web"] is True), ("answer returned", lambda r, s: len(r["text"]) > 60)]),

    ("REGRESSION: 'summary' + '2 images' used to hijack to the no-tools summarize agent (images dropped)", "stream",
     req(type="general", model="gemini", query="Write a 60-word blog post about a small coffee shop. Include a summary line at the end and 2 images."),
     [("free-form agent, not 'summarize'", lambda r, s: s and s.get("agent_type") == "smart_global"),
      ("exactly 2 images generated", lambda r, s: n_images(r) == 2)]),

    ("REGRESSION: declared 'simplify' whose text says 'fix grammar' used to run the grammar agent", "process",
     req(type="simplify", model="gemini", query="Please fix grammar issues in this sentence: The committee's deliberations notwithstanding, the aforementioned stipulations remain in force."),
     [("ran the simplify agent", lambda r, s: s and s.get("agent_type") == "simplify"), ("200", lambda r, s: r["http"] == 200)]),

    ("REGRESSION: declared 'make_shorter' whose text says 'translate' used to run the translator", "process",
     req(type="make_shorter", model="gemini", query="Translate the menu to Spanish for our guests: we proudly serve freshly roasted coffee, warm pastries and seasonal sandwiches every single morning."),
     [("ran the make_shorter agent", lambda r, s: s and s.get("agent_type") == "make_shorter"), ("200", lambda r, s: r["http"] == 200)]),

    ("claude chat model: a failure must be visible, never an empty 200", "process",
     req(type="fix_grammar", model="claude", query="me and him goes to the market"),
     [("never an empty 200", lambda r, s: not (r["http"] == 200 and not r["text"].strip())),
      ("a failure names the provider", lambda r, s: r["http"] == 200 or "claude request failed" in json.dumps(r.get("json", {})))]),

    ("/agent/process with an image: real URL, no dead placeholder", "process",
     req(type="general", model="gemini", query="write two sentences about a lighthouse with a hero image"),
     [("200", lambda r, s: r["http"] == 200), ("no dead placeholders", lambda r, s: no_dead(r, s)),
      ("has a real image URL", lambda r, s: "/local-images/" in r["text"])]),

    ("/agent/process image_params + regex model (grok)", "process",
     req(type="general", model="claude", query="draw a green apple on a wooden table using grok imagine", image_params={"quality": "high"}),
     [("200", lambda r, s: r["http"] == 200), ("real image URL", lambda r, s: "/local-images/" in r["text"]),
      ("drawn by grok", lambda r, s: has_model(r, s, "grok", "grok-imagine-image"))]),

    ("typed text operation: fix_grammar", "process",
     req(type="fix_grammar", model="gemini", query="me and him goes to the market yesterday and buyed two apple"),
     [("200", lambda r, s: r["http"] == 200), ("corrected text returned", lambda r, s: len(r["text"]) > 20)]),

    ("generate_blog (stream): skeleton first, ordered sections, a table, pull quotes, 3 pictures", "stream",
     req(type="generate_blog", model="openai", secondary_prompt="Friendly and practical, no jargon",
         query="Write a blog post for team leads about running remote meetings that people actually want to attend.",
         blog_options={"length": "standard"}),
     [("200", lambda r, s: r["http"] == 200),
      ("the title opens the post before the outline event (BLOG_STREAM_TITLE)", lambda r, s: r.get("outline") and r["first_token"] < r["outline_at"]),
      ("one heading per outline entry", lambda r, s: blog_h2(r) == len(r["outline"]["sections"])),
      ("at least 1 table", lambda r, s: blog_tables(r) >= 1), ("at least 2 pull quotes", lambda r, s: blog_quotes(r) >= 2),
      ("exactly 3 pictures, all delivered", lambda r, s: n_images(r) == 3 and r["text"].count("placeholder::") == 3),
      ("no image_failed / error events", lambda r, s: not r["events"]),
      ("planner recorded the blog", lambda r, s: s and s.get("mode") == "blog" and s["blog"]["images"] == 3),
      ("first content within 12 s (plan target)", lambda r, s: r["first_token"] is not None and r["first_token"] <= 12)]),

    ("generate_blog (process): short post, 1 picture, real URL, warnings listed", "process",
     req(type="generate_blog", model="gemini", query="Write a short blog post about choosing a first project management tool for a small business",
         blog_options={"length": "short", "images": 1}),
     [("200", lambda r, s: r["http"] == 200), ("a table", lambda r, s: blog_tables(r) >= 1),
      ("no dead placeholders", lambda r, s: no_dead(r, s)), ("one real image URL", lambda r, s: r["text"].count("/local-images/") == 1),
      ("warnings is a list", lambda r, s: isinstance(r["json"].get("warnings"), list))]),

    ("generate_blog: 'without images' in the brief means none are generated or billed", "stream",
     req(type="generate_blog", model="gemini", query="Write a short blog post about brewing better tea at home, without images",
         blog_options={"length": "short"}),
     [("200", lambda r, s: r["http"] == 200), ("no images", lambda r, s: n_images(r) == 0 and "placeholder::" not in r["text"]),
      ("still has a table and quotes", lambda r, s: blog_tables(r) >= 1 and blog_quotes(r) >= 2)]),

    ("enhance_with_ai (stream): a long document keeps its code blocks, tables, headings and links; progress is reported", "stream",
     req(type="enhance_with_ai", model="openai", secondary_prompt="clearer and more direct", query=DOC),
     [("200", lambda r, s: r["http"] == 200), ("ran the chunked pipeline", lambda r, s: s and s.get("mode") == "longform"),
      ("one refine_progress event per passage", lambda r, s: len(r["progress"]) == s["longform"]["chunks"] >= 2),
      ("code blocks identical, in order", lambda r, s: kept(r, fences)), ("tables identical, in order", lambda r, s: kept(r, tables_of)),
      ("same headings at the same levels", lambda r, s: kept(r, heading_levels)), ("same URLs", lambda r, s: kept(r, urls_of)),
      ("length within 15% of the original", lambda r, s: 0.85 <= len(r["text"]) / len(DOC) <= 1.15),
      ("warnings only if a passage was restored", lambda r, s: (r["warnings"] or []) == [] or s["longform"]["restored"] > 0),
      ("first text within 30 s", lambda r, s: r["first_token"] is not None and r["first_token"] <= 30)]),

    ("fix_grammar (process): the same document, whole, with a warnings list", "process",
     req(type="fix_grammar", model="gemini", query=DOC),
     [("200", lambda r, s: r["http"] == 200), ("ran the chunked pipeline", lambda r, s: s and s.get("mode") == "longform"),
      ("code blocks identical", lambda r, s: kept(r, fences)), ("tables identical", lambda r, s: kept(r, tables_of)),
      ("same headings", lambda r, s: kept(r, heading_levels)), ("same URLs", lambda r, s: kept(r, urls_of)),
      ("only fixes errors: length within 5%", lambda r, s: 0.95 <= len(r["text"]) / len(DOC) <= 1.05),
      ("warnings is a list", lambda r, s: isinstance(r["json"].get("warnings"), list))]),

    ("validation: blog_options on another type -> 400", "stream",
     req(type="general", model="gemini", query="x", blog_options={"length": "short"}),
     [("HTTP 400", lambda r, s: r["http"] == 400)]),

    ("validation: unknown type -> 400", "stream", req(type="nope", model="gemini", query="x"),
     [("HTTP 400", lambda r, s: r["http"] == 400)]),

    ("autocomplete", "autocomplete", {"api_key": LICENSE, "model": "gemini", "text_before": "Thank you for your email, I will get back to you"},
     [("200", lambda r, s: r["http"] == 200)]),
]


def server_summary(client, rid):
    if not rid:
        return None
    try:
        recent = client.get(f"{BASE}/agent/metrics?limit=200", timeout=10).json()["recent"]
    except Exception:
        return None
    for _ in range(20):                       # the summary is written when the stream's finally-block runs
        for s in recent:
            if s["request_id"] == rid:
                return s
        time.sleep(0.25)
        recent = client.get(f"{BASE}/agent/metrics?limit=200", timeout=10).json()["recent"]
    return None


def main(selected):
    results = []
    with httpx.Client() as client:
        for i, (title, kind, body, checks) in enumerate(SCENARIOS, 1):
            if selected and i not in selected:
                continue
            print("\n" + "=" * 100 + f"\n#{i}  {title}\n" + "=" * 100)
            if kind == "GET /health":
                t0 = time.perf_counter(); r = client.get(f"{BASE}/health", timeout=10)
                res = {"http": r.status_code, "total": time.perf_counter() - t0, "text": r.text, "rid": r.headers.get("X-Request-ID")}
                print(f"REQUEST   GET /health\nRESPONSE  {r.status_code} {r.text}\nCLIENT    total={res['total']*1000:.0f} ms")
                srv = None
            else:
                print(f"REQUEST   POST /agent/{kind}\n          {json.dumps(redact(body), ensure_ascii=False)}")
                if kind == "stream":
                    res = stream_request(client, body)
                elif kind == "process":
                    res = process_request(client, body)
                else:
                    t0 = time.perf_counter(); r = client.post(f"{BASE}/agent/autocomplete", json=body, timeout=60)
                    res = {"http": r.status_code, "total": time.perf_counter() - t0, "text": r.text, "rid": r.headers.get("X-Request-ID")}
                body_txt = res.get("text") or res.get("body") or ""
                print(f"RESPONSE  HTTP {res['http']}  request_id={res.get('rid')}")
                print("          " + (body_txt[:700].replace("\n", "\n          ") + ("…" if len(body_txt) > 700 else "")))
                for im in res.get("images", []):
                    print(f"          image_ready @ {im['t']}s  {im['w']}x{im['h']}  {im['url']}")
                for e in res.get("events", []):
                    print(f"          EVENT {e}")
                parts = [f"total={res['total']:.2f}s"]
                for k in ("ttfb", "first_token", "done", "stream_end"):
                    if res.get(k) is not None:
                        parts.append(f"{k}={res[k]:.2f}s")
                print("CLIENT    " + "  ".join(parts))
                srv = server_summary(client, res.get("rid")) if kind in ("stream", "process") and res.get("http") == 200 else None
                if srv:
                    marks = " | ".join(f"{k} {v:.2f}s" for k, v in srv["marks"].items())
                    print(f"SERVER    total={srv['total_seconds']:.2f}s route={srv.get('route')} plan={srv.get('plan')} | {marks}")
                    for im in srv["images"]:
                        print(f"          image {im['provider']}/{im['model']} {im['status']} {im['seconds']}s "
                              f"(provider call {im.get('adapter_seconds')}s, convert+save {im.get('upload_seconds')}s)")
            ok_all = True
            for desc, fn in checks:
                try:
                    ok = bool(fn(res, srv))
                except Exception as exc:
                    ok = False; desc += f" [{type(exc).__name__}: {exc}]"
                ok_all &= ok
                print(f"CHECK     {'PASS' if ok else 'FAIL'}  {desc}")
            results.append({"n": i, "title": title, "ok": ok_all, "client_total": round(res["total"], 2),
                            "first_token": round(res["first_token"], 2) if res.get("first_token") else None,
                            "server_total": srv["total_seconds"] if srv else None, "request_id": res.get("rid")})
        credits = client.get(f"{BASE}/local/credits", timeout=10).json()

    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    print(f"{'#':>2} {'result':6} {'client s':>8} {'server s':>8} {'1st token':>9}  scenario")
    for r in results:
        print(f"{r['n']:>2} {'PASS' if r['ok'] else 'FAIL':6} {r['client_total']:>8} {str(r['server_total'] or '-'):>8} {str(r['first_token'] or '-'):>9}  {r['title']}")
    print(f"\nCredits (in-memory): balance={credits.get('balance_credits')} reserved={credits.get('reserved_credits')}  "
          f"(* DEDUCT lines in the server log are the simulated backend)")
    out = RUN_DIR / f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"results": results, "credits": credits}, indent=1))
    print(f"report: {out}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main({int(a) for a in sys.argv[1:]}))
