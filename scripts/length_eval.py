"""Diff the length parsers over existing fixtures, or evaluate length adherence with --live (paid text calls)."""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings
from src.length_spec import measure, parse_length

PROVIDERS = ("openai", "claude", "gemini")
TARGETS = (50, 100, 150, 300, 600, 1450)
PROMPTS = {
    "article": "Write an article on growing kitchen herbs in {} words.",
    "paragraph": "Write a paragraph about a neighborhood book swap in {} words.",
    "draft": "Draft a passage about preparing a seed tray in {} words.",
    "under": "Write a note about packing for a picnic in under {} words.",
}
SOURCE = "# Growing from seed\n\nThis guide covers small containers and easy crops for first-time gardeners.\n"


def parser_diff():
    from scripts.document_actions.eval_auto import HELDOUT, ITEMS
    from scripts.evaluate_blog import TOPICS
    from src.request_plan import extract_word_count
    from tests.fixtures.code_intent_eval_cases import CASES as CODE
    from tests.fixtures.image_intent_eval_cases import CASES as IMAGE

    groups = {
        "auto": chain.from_iterable((*ITEMS.values(), *HELDOUT.values())),
        "blog": (brief for _, brief, _ in TOPICS),
        "chat": (case[1] for case in chain(CODE, IMAGE)),
    }
    prompts = {query: group for group, texts in groups.items() for query in texts}
    with patch.object(settings, "length_contract_enabled", False):
        old = {query: extract_word_count(query) for query in prompts}
    changes = []
    for query, group in prompts.items():
        spec = parse_length(query)
        if spec is None and old[query] is None:
            continue
        if spec is not None and spec.kind == "about" and spec.unit == "words" and spec.target == old[query]:
            continue
        changes.append({"group": group, "query": query, "old_words": old[query],
                        "new": asdict(spec) if spec else None})
    return {"prompts": len(prompts), "changes": changes}


def cases():
    for shape, template in PROMPTS.items():
        for target in TARGETS:
            yield shape, template.format(target)
    for target in (2, 3, 5):
        yield "sentences", f"Write an article on keeping a sketchbook in {target} sentences."


def evaluate(provider, key, shape, query, request_id):
    from src.document_context import DocumentContext, prepare_context
    from src.observability import finish_trace, start_trace
    from src.provider_errors import status_code
    from src.text_processing_agent import FREE_FORM_AGENT, process_user_query_stream

    spec = parse_length(query)
    assert spec is not None, query
    row = {"provider": provider, "shape": shape, "query": query, "spec": asdict(spec)}
    prepared = (prepare_context(DocumentContext(text=SOURCE, cursor_utf16=len(SOURCE)), query, "draft")
                if shape == "draft" else None)
    parts, warnings, replacement = [], [], None
    start_trace(request_id, "length_evaluation")
    started = time.perf_counter()
    try:
        for chunk in process_user_query_stream(
            query, provider, key, request_id=request_id, is_byok=True,
            force_agent_type=FREE_FORM_AGENT, prepared_context=prepared, image_routing=True,
        ):
            if isinstance(chunk, str):
                parts.append(chunk)
            elif "__content_replace__" in chunk:
                replacement = chunk["__content_replace__"]
            elif "__warnings__" in chunk:
                warnings.extend(chunk["__warnings__"])
    except Exception as error:
        row.update(error=type(error).__name__, error_detail=str(error)[:500], http_status=status_code(error))
    finally:
        summary = finish_trace(request_id, "error" if "error" in row else "ok")

    first = "".join(parts)
    final = first if replacement is None else replacement
    measured = measure(spec, final)
    row.update(
        model=summary.get("model_used"), first=first, final=final, first_count=measure(spec, first), final_count=measured,
        in_window=spec.contains(measured) and "error" not in row,
        # Measured from the allowed range, not the target: 113 words is a valid "under 150 words".
        within_20_percent="error" not in row and spec.window_distance(measured) <= 0.2 * spec.target,
        ceiling_miss=spec.kind == "at_most" and measured > spec.high,
        repair_attempted=summary.get("length", {}).get("repair_attempted", False),
        blog_repairs=summary.get("blog", {}).get("repairs", 0),
        seconds=round(time.perf_counter() - started, 2), warnings=warnings,
    )
    return row


def evaluate_provider(provider):
    key = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key,
           "gemini": settings.default_gemini_api_key}[provider]
    if not key:
        return [{"provider": provider, "error": "missing provider key"}]
    rows = []
    for i, (shape, query) in enumerate(cases(), 1):
        row = evaluate(provider, key, shape, query, f"length-eval-{provider}-{i}")
        rows.append(row)
        spec = row["spec"]
        result = row.get("error") or f"{row['first_count']} → {row['final_count']} {spec['unit']}"
        print(f"{provider:6} {shape:10} {spec['target']:4} {result} ({row['seconds']}s)", flush=True)
        if row.get("http_status") in (401, 403, 404):
            break
    return rows


def run_live(providers):
    settings.length_contract_enabled = True
    settings.blog_intent_shadow_mode = False
    settings.blog_image_source = "none"  # Isolate the text paths and avoid paid images.
    logging.disable(logging.CRITICAL)
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        rows = list(chain.from_iterable(pool.map(evaluate_provider, providers)))
    stats = {
        "expected": len(list(cases())) * len(providers),
        "runs": len(rows),
        "errors": sum("error" in r for r in rows),
        "in_window": sum(bool(r.get("in_window")) for r in rows),
        "within_20_percent": sum(bool(r.get("within_20_percent")) for r in rows),
        "ceiling_misses": sum(bool(r.get("ceiling_miss")) for r in rows),
        "repairs": sum(bool(r.get("repair_attempted")) for r in rows),
    }
    stats["repair_rate"] = round(stats["repairs"] / stats["runs"], 3)
    passed = (stats["errors"] == 0 and stats["runs"] == stats["expected"]
              and stats["in_window"] / stats["runs"] >= 0.95 and stats["within_20_percent"] == stats["runs"]
              and stats["ceiling_misses"] == 0)
    print(json.dumps({**stats, "passed": passed}, indent=2))
    return {"stats": stats, "passed": passed, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the configured provider keys for paid text calls")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=PROVIDERS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out.resolve() if args.out else None
    report = run_live(tuple(dict.fromkeys(args.providers))) if args.live else parser_diff()
    if not args.live:
        print(f"{report['prompts']} existing prompts. {len(report['changes'])} parser changes.")
        for row in report["changes"]:
            print(json.dumps(row, ensure_ascii=False))
    if output:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(args.live and not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
