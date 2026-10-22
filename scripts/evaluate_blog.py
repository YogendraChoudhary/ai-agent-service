"""
Blog generation (Mode 2): a free structural check, or an opt-in live quality/latency evaluation.

  python scripts/evaluate_blog.py                       # offline: 30 briefs through the real pipeline with a scripted model
  python scripts/evaluate_blog.py --live --providers openai claude --limit 3 --images
  python scripts/evaluate_blog.py --live --outline-only --models openai:advanced gemini:basic --ids t01 t04 t12
  python scripts/evaluate_blog.py --live --providers openai --outline-model gemini:basic --ids t01 t04 t12 --images

Offline proves the code's structure rules (every post has its tables and quotes, headings match the outline, no stray
markup) on every brief, for nothing. --live makes PAID provider calls with the configured default keys and, for each
(brief, provider), records tables / quotes / images, the longest run of prose, repairs, warnings, latency, tokens and
cost, then writes the post itself as Markdown (local_run/blog_eval/) so providers can be read side by side.
Pictures (--images) are written to local_run/images; nothing is ever uploaded to S3.

--outline-only times just the outline call for each "provider:tier" in --models (the first attempt and whether it was
valid without a repair), which is how a faster outline model is chosen; --outline-model sets BLOG_OUTLINE_MODEL for a
full-post run so the effect on first content and on the finished post is measured end to end.

Targets from docs/rich_content_engine_plan.md (Phase 5): every post meets its quotas when pictures are available;
longest prose run <= 250 words in >= 95% of posts; p50 <= 40 s for 1,400 words with 3 images; first content <= 12 s.
"""

import argparse
import functools
import io
import json
import re
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings                                             # noqa: E402
from src.blog.llm import AgnoLLM, model_id_for                          # noqa: E402
from src.blog.media import GeneratedImages                              # noqa: E402
from src.blog.outline import build_outline, validate_outline            # noqa: E402
from src.blog.plan import BlogRequest, plan_blog                        # noqa: E402
from src.blog.runtime import parse_outline_model                        # noqa: E402
from src.blog.service import generate_blog_stream                       # noqa: E402
from src.blog.validate import longest_prose_run, sentence_count         # noqa: E402
from src.longform.markdown_ast import parse, prose_words, table_shape  # noqa: E402
from src.models import ADVANCED_MODEL_IDS                              # noqa: E402
from src.observability import finish_trace, start_trace                 # noqa: E402
from src.provider_errors import status_code                             # noqa: E402

RUN_DIR = ROOT / "local_run"
TARGET_PROSE_RUN = 250
TARGET_FIRST_CONTENT_SECONDS = 12
TARGET_P50_SECONDS = 40

# (id, brief, blog_options). A spread of subjects, lengths, picture requests and research cues.
TOPICS = [
    ("t01", "Write a blog post about how to build a sleep routine that survives a busy week", {}),
    ("t02", "Write a short blog post for small business owners about choosing their first project management tool", {"length": "short"}),
    ("t03", "Write a long-form guide for homeowners who want to switch their home to renewable energy", {"length": "long"}),
    ("t04", "Write a blog post for engineering managers about running effective one-on-ones", {}),
    ("t05", "Write a blog post about brewing better coffee at home, with 2 images", {}),
    ("t06", "Write a 1,000-word blog post about learning to cook from scratch", {}),
    ("t07", "Write a blog post about the case for four-day work weeks", {"tone": "persuasive"}),
    ("t08", "Write a blog post about getting started with personal budgeting", {"audience": "recent graduates"}),
    ("t09", "Write a short blog post about why code reviews matter, without images", {"length": "short"}),
    ("t10", "Write a blog post about travelling light: how to pack for two weeks in one bag", {}),
    ("t11", "Write a blog post about the latest trends in remote work in 2026", {}),
    ("t12", "Write a blog post comparing the main ways to learn a new language as an adult", {}),
    ("t13", "Write a blog post about growing herbs on a small balcony", {"images": 2}),
    ("t14", "Write a blog post about how small teams can write better documentation", {"tone": "practical"}),
    ("t15", "Write a long blog post about the basics of personal data privacy", {"length": "long", "images": 3}),
    ("t16", "Write a blog post about starting a running habit without injuries", {}),
    ("t17", "Write a short blog post about picking a first houseplant", {"length": "short", "images": 1}),
    ("t18", "Write a blog post about how to run a good customer interview", {}),
    ("t19", "Write a 2,000-word blog post about the history and future of electric cars", {}),
    ("t20", "Write a blog post about mindful use of social media", {"audience": "parents"}),
    ("t21", "Write a blog post about choosing between renting and buying a home", {}),
    ("t22", "Write a short blog post about the habits of effective note-takers", {"length": "short"}),
    ("t23", "Write a blog post about designing a home office that you will actually use", {}),
    ("t24", "Write a blog post about preparing for a first marathon", {"images": 0}),
    ("t25", "Write a blog post about how open-source projects find and keep contributors", {}),
    ("t26", "Write a blog post about teaching kids to cook", {"tone": "warm"}),
    ("t27", "Write a blog post about upcycling old furniture, text-only", {}),
    ("t28", "Write a blog post about the current state of AI coding assistants", {}),
    ("t29", "Write a blog post about planning a week-long trip to Japan on a budget", {}),
    ("t30", "Write a short blog post about why saying no is a leadership skill", {"length": "short"}),
]


def analyse(markdown: str) -> dict:
    """Facts about a finished post, read from its Markdown alone (so the same code judges offline and live runs)."""
    blocks = parse(markdown)
    tables = [table_shape(b) for b in blocks if b.kind == "table"]
    paragraphs = [b for b in blocks if b.kind == "paragraph"]
    return {
        "words": prose_words(blocks),
        "h1": sum(b.kind == "heading" and b.level == 1 for b in blocks),
        "h2": sum(b.kind == "heading" and b.level == 2 for b in blocks),
        "tables": len(tables),
        "tables_consistent": all(t and t.consistent for t in tables),
        "quotes": sum(b.kind == "blockquote" for b in blocks),
        "figures": sum(b.kind == "image" for b in blocks),
        "placeholders": len(re.findall(r"placeholder::", markdown)),
        "lists": sum(b.kind == "list" for b in blocks),
        "longest_prose_run": longest_prose_run(blocks),
        "long_paragraphs": sum(sentence_count(b.text) > 4 for b in paragraphs),
        "raw_urls": len(re.findall(r"(?<![(\[])https?://\S+", markdown)),
    }


def failures(metrics: dict, plan, *, pictures_available: bool) -> list:
    """What a post must satisfy, whoever wrote it. The offline run fails on any of these; a live run reports them."""
    out = []
    if metrics["h1"] != 1:
        out.append(f"{metrics['h1']} top-level titles")
    if metrics["h2"] != plan.sections + 1:
        out.append(f"{metrics['h2']} section headings, the plan has {plan.sections + 1}")
    if metrics["tables"] < plan.tables or not metrics["tables_consistent"]:
        out.append(f"tables: {metrics['tables']} (need {plan.tables}, all complete)")
    if metrics["quotes"] < plan.quotes:
        out.append(f"quotes: {metrics['quotes']} (need {plan.quotes})")
    if pictures_available and metrics["figures"] != plan.images:
        out.append(f"pictures: {metrics['figures']} (planned {plan.images})")
    if metrics["longest_prose_run"] > TARGET_PROSE_RUN:
        out.append(f"{metrics['longest_prose_run']} words of prose in a row")
    if not plan.use_web and metrics["raw_urls"]:
        out.append("URLs in a post that did no research")
    return out


def plan_for(brief: str, options: dict, images_available: bool = True):
    return plan_blog(BlogRequest.from_options(brief, "", options), image_available=images_available)


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------

def run_offline() -> list:
    from tests.blog.fakes import FakeImageTool, ScriptedLLM

    rows = []
    for topic_id, brief, options in TOPICS:
        request = BlogRequest.from_options(brief, "", options)
        plan = plan_blog(request)
        tool = FakeImageTool()
        chunks = list(generate_blog_stream(request, plan, ScriptedLLM(), GeneratedImages(tool)))
        text = "".join(c for c in chunks if isinstance(c, str))
        metrics = analyse(text)
        warnings = next((c["__warnings__"] for c in chunks if isinstance(c, dict) and "__warnings__" in c), [])
        rows.append({"id": topic_id, "brief": brief, "planned": {"words": plan.target_words, "sections": plan.sections,
                                                                  "tables": plan.tables, "quotes": plan.quotes, "images": plan.images},
                     "metrics": metrics, "warnings": [w["kind"] for w in warnings],
                     "failures": failures(metrics, plan, pictures_available=True)
                     + ([] if len(tool.placeholders) == plan.images else [f"{len(tool.placeholders)} pictures started"])})
    return rows


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def install_local_uploads(img_dir: Path) -> None:
    """Pictures go to disk, never to S3: this script must not write to the production bucket."""
    import src.image_gen as image_gen
    from PIL import Image
    from src.s3_service import convert_image_format, sniff_image_bytes

    img_dir.mkdir(parents=True, exist_ok=True)

    def local_upload(image_bytes, content_type=None, user_id=None, request_id=None, target_format=None):
        if target_format:
            data, ctype, ext = convert_image_format(image_bytes, target_format)
        else:
            (ctype, ext), data = sniff_image_bytes(image_bytes), image_bytes
        name = f"{request_id or 'eval'}_{uuid.uuid4().hex[:6]}.{ext}"
        (img_dir / name).write_bytes(data)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
        return {"cdn_url": f"../images/{name}", "width": width, "height": height, "format": ext, "content_type": ctype}
    image_gen.upload_image_from_bytes_detailed = local_upload


def key_resolver():
    """The platform-account key router the API builds, from the default keys (no Redis involved)."""
    from src.redis_manager import RedisManager
    from src.task_router import KeyResolver

    default_keys = {"openai_api_key": settings.default_openai_api_key, "anthropic_api_key": settings.default_anthropic_api_key,
                    "grok_api_key": settings.default_grok_api_key, "gemini_api_key": settings.default_gemini_api_key,
                    "deepseek_api_key": settings.default_deepseek_api_key}
    return KeyResolver({}, default_keys, functools.partial(RedisManager.resolve_api_key_from_user_keys, None))


def run_one_live(provider: str, key: str, topic_id: str, brief: str, options: dict, images: bool, resolver=None) -> dict:
    from src.text_processing_agent import _wait_for_images, process_user_query_stream, resolve_image_placeholders

    request_id = f"eval-blog-{provider}-{topic_id}"
    options = {**options, **({} if images else {"images": 0})}
    start_trace(request_id, "blog_evaluation", type="generate_blog", chat_model=provider)
    done_event = threading.Event()
    started, first, chunks, pending, warnings, failure = time.perf_counter(), None, [], {}, [], None
    try:
        for chunk in process_user_query_stream(
                brief, provider, key, request_id=request_id, force_agent_type="blog_writer", is_byok=False,
                image_completion_callback=done_event.set, blog_options=options, key_resolver=resolver):
            if isinstance(chunk, str):
                first = first if first is not None else time.perf_counter() - started
                chunks.append(chunk)
            elif isinstance(chunk, dict) and "__pending_images__" in chunk:
                pending = chunk["__pending_images__"]
            elif isinstance(chunk, dict) and "__warnings__" in chunk:
                warnings = chunk["__warnings__"]
        text_seconds = time.perf_counter() - started
        if any(not str(k).startswith("__") for k in pending):
            _wait_for_images(pending, done_event, float(settings.image_wait_seconds))
    except Exception as error:
        failure = error
        text_seconds = time.perf_counter() - started
    text = resolve_image_placeholders("".join(chunks), pending)
    summary = finish_trace(request_id, "error" if failure else "ok") or {}
    plan = plan_for(brief, options)
    metrics = analyse(text) if text else {}
    served_by = summary.get("model_used")
    found = failures(metrics, plan, pictures_available=images) if metrics else ["no text"]
    if served_by and not served_by.startswith(f"{provider}/"):
        found.append(f"served by {served_by}, not {provider} (a fallback answered)")      # never score the wrong model
    return {
        "provider": provider, "model": ADVANCED_MODEL_IDS[provider], "id": topic_id, "brief": brief, "options": options,
        "planned": {"words": plan.target_words, "sections": plan.sections, "tables": plan.tables, "quotes": plan.quotes,
                    "images": plan.images},
        "served_by": served_by, "outline_setting": settings.blog_outline_model, "outline_model": summary.get("outline_model"),
        "fallbacks": summary.get("fallbacks"), "llm_calls": summary.get("llm_calls"),
        "metrics": metrics, "failures": found,
        "warnings": warnings, "blog": summary.get("blog"), "first_content": round(first, 2) if first else None,
        "text_seconds": round(text_seconds, 2), "seconds": round(time.perf_counter() - started, 2),
        "images": [{"status": v.get("status"), "provider": v.get("provider"), "model": v.get("model"), "seconds": v.get("seconds")}
                   for k, v in pending.items() if not str(k).startswith("__")],
        "tokens_in": summary.get("tokens_in"), "tokens_out": summary.get("tokens_out"), "est_cost_usd": summary.get("est_cost_usd"),
        "error": str(failure) if failure else None, "content": text,
    }


class TimedLLM:
    """Wraps an LLM and records every call, so a run can be judged call by call: (stage, seconds, reply)."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []

    def __call__(self, system, messages, spec):
        started = time.perf_counter()
        reply = self.llm(system, messages, spec)
        self.calls.append((spec.stage, time.perf_counter() - started, reply))
        return reply


def outline_digest(outline) -> dict:
    """What a person needs to judge an outline at a glance."""
    return {"title": outline.title, "subtitle": outline.subtitle,
            "sections": [{"heading": sec.heading, "angle": sec.angle, "words": sec.target_words,
                          "visuals": [v.kind for v in sec.visuals]} for sec in outline.sections],
            "conclusion": outline.conclusion.heading}


def run_outline_only(models: list, topics: list, keys: dict, repeats: int = 1) -> list:
    """Time the outline call alone for each "provider:tier": first-call seconds and whether it was valid unrepaired."""
    from src.task_router import Candidate

    rows = []
    for spec in models:
        provider, advanced = parse_outline_model(spec) or (None, None)
        label = f"{provider}:{'advanced' if advanced else 'basic'}"
        if provider is None or not keys.get(provider):
            rows.append({"model": spec, "error": "unknown provider or no configured key", "first_pass_valid": False})
            continue
        for topic_id, brief, options in topics:
            for repeat in range(repeats):
                request_id = f"eval-outline-{label.replace(':', '-')}-{topic_id}-{repeat}"
                request = BlogRequest.from_options(brief, "", options)
                plan = plan_blog(request)
                start_trace(request_id, "blog_outline_evaluation", type="generate_blog", chat_model=provider)
                llm = TimedLLM(AgnoLLM([Candidate(provider, advanced)], lambda _p, k=keys[provider]: k,
                                       request_id=request_id, is_byok=False))
                started, result, failure = time.perf_counter(), None, None
                try:
                    result = build_outline(llm, request, plan, request_id)
                except Exception as error:
                    failure = error
                total = time.perf_counter() - started
                summary = finish_trace(request_id, "error" if failure else "ok") or {}
                first = llm.calls[0] if llm.calls else None
                check = validate_outline(first[2], plan) if first else None
                rows.append({
                    "model": label, "model_id": model_id_for(Candidate(provider, advanced)), "id": topic_id, "repeat": repeat,
                    "brief": brief, "first_call_seconds": round(first[1], 2) if first else None,
                    "total_seconds": round(total, 2), "calls": len(llm.calls),
                    "first_pass_valid": bool(check and check.outline is not None and not check.errors),
                    "first_pass_errors": [e.kind for e in check.errors] if check else ["no reply"],
                    "repaired": bool(result and result.repaired), "filled": result.filled if result else None,
                    "tokens_in": summary.get("tokens_in"), "tokens_out": summary.get("tokens_out"),
                    "llm_calls": summary.get("llm_calls"), "error": str(failure) if failure else None,
                    "outline": outline_digest(result.outline) if result else None,
                })
    return rows


def summarise_outlines(rows: list) -> list:
    lines = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model and r.get("first_call_seconds") is not None]
        if not mine:
            lines.append(f"{model}: no results")
            continue
        seconds = [r["first_call_seconds"] for r in mine]
        valid = sum(r["first_pass_valid"] for r in mine)
        lines.append(f"{model:18} n={len(mine)}  first call p50 {statistics.median(seconds):5.1f}s max {max(seconds):5.1f}s  "
                     f"valid unrepaired {valid}/{len(mine)}  repaired {sum(r['repaired'] for r in mine)}  "
                     f"filled {sum(bool(r['filled']) for r in mine)}  tokens out {int(statistics.mean(r['tokens_out'] or 0 for r in mine))}")
    return lines


def write_post(row: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_outline-{row['outline_setting'].replace(':', '-')}" if row.get("outline_setting") else ""
    path = out_dir / f"{row['id']}_{row['provider']}{suffix}.md"          # one file per setting: runs can be compared
    m = row["metrics"] or {}
    header = (f"<!-- {row['provider']} / {row['model']} | outline model: {row.get('outline_model') or 'same'} | brief: {row['brief']} | options: {json.dumps(row['options'])}\n"
              f"     {row['seconds']}s total, first content {row['first_content']}s | tables {m.get('tables')} quotes {m.get('quotes')} "
              f"figures {m.get('figures')} | longest prose run {m.get('longest_prose_run')} | words {m.get('words')}\n"
              f"     failures: {row['failures']} | warnings: {[w['kind'] for w in row['warnings']]} -->\n\n")
    path.write_text(header + (row["content"] or f"(no text: {row['error']})") + "\n", encoding="utf-8")
    return path


def summarise(rows: list) -> list:
    """Printable lines comparing the run against the plan's Phase 5 targets."""
    ok = [r for r in rows if r.get("metrics")]
    if not ok:
        return ["no results"]
    lines = [f"posts: {len(ok)}  with structural failures: {sum(bool(r['failures']) for r in ok)}"]
    runs = [r["metrics"]["longest_prose_run"] for r in ok]
    lines.append(f"longest prose run <= {TARGET_PROSE_RUN}: {sum(v <= TARGET_PROSE_RUN for v in runs)}/{len(runs)} "
                 f"(target >= 95%)")
    timed = [r for r in ok if r.get("seconds") is not None and r.get("first_content") is not None]
    if timed:
        firsts, totals = [r["first_content"] for r in timed], [r["seconds"] for r in timed]
        lines.append(f"first content: p50 {statistics.median(firsts):.1f}s max {max(firsts):.1f}s (target <= {TARGET_FIRST_CONTENT_SECONDS}s)")
        lines.append(f"total: p50 {statistics.median(totals):.1f}s max {max(totals):.1f}s (target p50 <= {TARGET_P50_SECONDS}s for 1,400 words)")
    cost = [r["est_cost_usd"] for r in ok if r.get("est_cost_usd") is not None]
    if cost:
        lines.append(f"estimated text cost: ${sum(cost):.3f} total, ${statistics.mean(cost):.3f} per post")
    repaired = sum((r.get("blog") or {}).get("repairs", 0) for r in ok)
    lines.append(f"repair calls: {repaired}  posts with warnings: {sum(bool(r['warnings']) for r in ok)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Make PAID provider calls using the configured default keys")
    parser.add_argument("--providers", nargs="+", choices=["openai", "claude", "gemini", "grok"], default=["openai"])
    parser.add_argument("--limit", type=int, default=0, help="Briefs per provider; 0 runs all 30")
    parser.add_argument("--ids", nargs="+", help="Only these brief ids (t01 ...)")
    parser.add_argument("--images", action="store_true", help="Also generate the pictures (paid; saved to local_run/images)")
    parser.add_argument("--outline-only", action="store_true", help="Time only the outline call, for each of --models")
    parser.add_argument("--models", nargs="+", default=["openai:advanced", "gemini:basic"], metavar="PROVIDER:TIER",
                        help="With --outline-only: the models to compare (tier basic|advanced)")
    parser.add_argument("--repeats", type=int, default=1, help="With --outline-only: calls per model and brief")
    parser.add_argument("--outline-model", metavar="PROVIDER:TIER", help="Use this model for the first outline attempt (BLOG_OUTLINE_MODEL)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 0 or args.repeats < 1:
        parser.error("--limit must be non-negative and --repeats at least 1")
    if args.outline_model and parse_outline_model(args.outline_model) is None:
        parser.error("--outline-model must look like gemini:basic")
    args.out = args.out or RUN_DIR / ("blog_outline_benchmark.json" if args.outline_only else "blog_evaluation.json")

    if not args.live:
        rows = run_offline()
        bad = [r for r in rows if r["failures"]]
        print(f"Offline: {len(rows)} briefs through the real pipeline with a scripted model, {len(bad)} with failures")
        for r in bad:
            print(f"  {r['id']}: {r['failures']}")
        return int(bool(bad))

    keys = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key,
            "gemini": settings.default_gemini_api_key, "grok": settings.default_grok_api_key}
    topics = [t for t in TOPICS if not args.ids or t[0] in args.ids][:args.limit or None]
    if args.outline_only:
        rows = run_outline_only(args.models, topics, keys, args.repeats)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        for r in rows:
            print(f"{r['model']:18} {r.get('id', '-')} #{r.get('repeat', 0)} first={r.get('first_call_seconds')}s "
                  f"valid={r['first_pass_valid']} {r.get('first_pass_errors') or ''} error={r.get('error')}")
        print("\n".join(summarise_outlines(rows)))
        print(f"Wrote {len(rows)} results to {args.out}")
        return int(any(r.get("error") for r in rows))
    if args.images:
        install_local_uploads(RUN_DIR / "images")
    if args.outline_model:
        settings.blog_outline_model = args.outline_model.strip().lower()
    resolver = key_resolver()
    rows = []
    for provider in args.providers:
        if not keys[provider]:
            rows.append({"provider": provider, "error": "No configured key", "metrics": None, "failures": ["no key"], "warnings": []})
            continue
        for topic_id, brief, options in topics:
            row = run_one_live(provider, keys[provider], topic_id, brief, options, args.images, resolver)
            rows.append(row)
            if row["content"]:
                write_post(row, RUN_DIR / "blog_eval")
            args.out.parent.mkdir(parents=True, exist_ok=True)          # saved after every post: an interrupted paid
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")      # evaluation loses no finished work
            print(f"{provider:7} {topic_id} {row['seconds']:>6}s first={row['first_content']}s outline={row.get('outline_model')} "
                  f"tables={(row['metrics'] or {}).get('tables')} quotes={(row['metrics'] or {}).get('quotes')} "
                  f"figures={(row['metrics'] or {}).get('figures')} failures={row['failures']}")
            if row["error"] and status_code(Exception(row["error"])) in (400, 401, 403, 404):
                break                                                    # a dead key or retired model: stop paying for it
    print("\n".join(summarise([r for r in rows if r.get("metrics") is not None])))
    print(f"Wrote {len(rows)} results to {args.out} and the posts to {RUN_DIR / 'blog_eval'}")
    return int(any(r.get("error") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
