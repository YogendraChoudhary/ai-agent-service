"""
Accuracy and latency evaluation for voice dictation. Run from the project root.

  python -m src.voice_flow.evals.run_evals                        # Gemini, the production formatter (paid, ~60 short calls)
  python -m src.voice_flow.evals.run_evals --engine rules         # the rule-based fallback only (free, offline)
  python -m src.voice_flow.evals.run_evals --engine openai:gpt-4.1-nano   # OpenAI, while Gemini is out (or to compare)
  python -m src.voice_flow.evals.run_evals --style polished       # score the Polished style (long-form cases are scored on checks)
  python -m src.voice_flow.evals.run_evals --audio --limit 12     # speak each case with macOS `say` through real Whisper + the formatter

Text mode scores the formatter against test_cases.json (exact match to `expected` or one of `alternatives`) and checks the
plan's acceptance criteria. Audio mode measures end-to-end latency; its transcripts vary with the synthetic voice, so it
prints them next to the expectation instead of scoring them.

A run in which the model did not answer is reported as such: its numbers describe the fallback, not the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import settings                                                   # noqa: E402
from src.voice_flow.formatter.engine import (                                   # noqa: E402
    Breaker, GeminiBackend, LlmError, OpenAIBackend, VoiceFormattingService, build_backend)
from src.voice_flow.formatter.post_processor import PostProcessor              # noqa: E402
from src.voice_flow.schemas import AppContext, CursorContext, effective_context   # noqa: E402

CASES_PATH = Path(__file__).with_name("test_cases.json")
MODE = "always"
STYLE: Optional[str] = None            # None = the server default; "polished" to score the Polished style
STRUCTURAL = {"numbered_list", "bullet_list", "false_trigger", "backtracking", "cursor_stitching"}
HALLUCINATION = {"question_guardrail"}
LLM_P95_BUDGET_MS = 250.0
LLM_BUDGET_MAX_WORDS = 40

_LIST_LINE = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+(\S)")


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    raw: str
    expected: str
    got: str
    source: str
    fallback_reason: Optional[str]
    llm_ms: float
    post_ms: float
    words: int
    lint: List[str]
    reasons: List[str] = field(default_factory=list)      # which long-form checks failed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def lint(text: str, app_context: str) -> List[str]:
    """Plan criterion: zero double spaces, missing lead-in colons, or uncapitalized list items."""
    problems = []
    lines = text.split("\n")
    if re.search(r"(?<=\S)[ \t]{2,}", text):
        problems.append("double space")
    if re.search(r"[ \t]+[,.;:!?](?:\s|$)", text):
        problems.append("space before punctuation")
    for i, line in enumerate(lines):
        m = _LIST_LINE.match(line)
        if not m:
            continue
        if app_context != "code_editor" and m.group(1).islower():
            problems.append(f"uncapitalized list item: {line.strip()[:30]!r}")
        prev = lines[i - 1] if i else ""
        if prev.strip() and not _LIST_LINE.match(prev) and not prev.rstrip().endswith((":", "?", "!", ",")):
            problems.append(f"lead-in without colon: {prev.strip()[:40]!r}")
        if i and not line.strip():
            problems.append("blank line in list")
    for a, b, c in zip(lines, lines[1:], lines[2:]):
        if _LIST_LINE.match(a) and not b.strip() and _LIST_LINE.match(c):
            problems.append("blank line inside list")
    return problems


def check_long(got: str, checks: Dict[str, Any]) -> List[str]:
    """Long dictations have no single right answer, so they are scored on properties that a good answer must have."""
    failed: List[str] = []
    items = re.findall(r"(?m)^\s*\d{1,3}[.)]\s", got) + re.findall(r"(?m)^\s*[-*\u2022]\s", got)
    if "list_items" in checks and len(items) != checks["list_items"]:
        failed.append(f"{len(items)} list items, the speaker counted {checks['list_items']}")
    for pattern in checks.get("must_match", []):
        if not re.search(pattern, got):
            failed.append(f"missing {pattern!r}")
    for pattern in checks.get("must_not_match", []):
        if re.search(pattern, got):
            failed.append(f"should not contain {pattern!r}")
    lines = [l for l in got.split("\n") if l.strip()]
    if checks.get("closing_after_list") and lines and re.match(r"\s*(?:\d{1,3}[.)]|[-*\u2022])\s", lines[-1]):
        failed.append("the closing sentence is inside the last list item")
    if "max_words" in checks and len(got.split()) > checks["max_words"]:
        failed.append(f"{len(got.split())} words, more than {checks['max_words']}")
    return failed


def passed(got: str, case: Dict[str, Any]) -> bool:
    if "checks" in case:
        return not check_long(got, case["checks"])
    return normalize(got) in {normalize(case["expected"]), *(normalize(a) for a in case.get("alternatives", []))}


def load_cases(path: "Path | str" = CASES_PATH, include_contexts: bool = False) -> List[Dict[str, Any]]:
    """App-specific cases (email, Slack, code, Markdown) are kept in the file but only run when those formats are on."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    if include_contexts or settings.voice_app_contexts_enabled:
        return cases
    return [c for c in cases if c.get("app_context", "general") == "general"]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(p * len(ordered) + 0.5)) - 1))]


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
class RulesOnly:
    async def generate(self, *a: Any, **k: Any) -> str:
        raise LlmError("rules engine selected")


def build_service(engine: str, mode: str = "always") -> VoiceFormattingService:
    """mode "always" scores the model itself; "smart" scores what a user gets (plain sentences skip the model)."""
    timeout = max(settings.voice_format_timeout_seconds, 8.0)
    if engine == "rules":
        return VoiceFormattingService(backend=RulesOnly(), timeout=timeout, breaker=Breaker(), mode="always")
    if engine.startswith("openai"):
        return VoiceFormattingService(backend=OpenAIBackend(model=engine.partition(":")[2] or None), timeout=timeout, breaker=Breaker(), mode=mode)
    if engine == "configured":                                   # whatever VOICE_FORMAT_PROVIDER / VOICE_FORMAT_MODEL say
        return VoiceFormattingService(backend=build_backend(), timeout=timeout, breaker=Breaker(), mode=mode)
    return VoiceFormattingService(backend=GeminiBackend(), timeout=timeout, breaker=Breaker(), mode=mode)


# ---------------------------------------------------------------------------
# Text evaluation
# ---------------------------------------------------------------------------
async def run_case(service: VoiceFormattingService, case: Dict[str, Any], style: Optional[str] = None) -> CaseResult:
    context = effective_context(case.get("app_context", "general"))
    cursor = CursorContext(preceding_text=case.get("preceding_text", ""), following_text=case.get("following_text", ""))
    result = await service.compile(case["raw"], context, cursor, case.get("custom_dictionary", []), style or STYLE)
    t = time.perf_counter()
    stitched = PostProcessor.stitch(result.formatted_text, cursor, context)
    post_ms = (time.perf_counter() - t) * 1000.0
    got = stitched.insert_text
    reasons = check_long(got, case["checks"]) if "checks" in case else []
    return CaseResult(case["id"], case["category"], passed(got, case), case["raw"], case.get("expected", ""), got, result.source,
                      result.fallback_reason, result.processing_time_ms, post_ms, len(case["raw"].split()),
                      lint(got, context.value), reasons)


def rate(rows: List[CaseResult]) -> str:
    return f"{sum(r.passed for r in rows)}/{len(rows)} ({100 * sum(r.passed for r in rows) / max(1, len(rows)):.1f}%)" if rows else "n/a"


def report(results: List[CaseResult], engine: str) -> Dict[str, Any]:
    print(f"\n=== {engine}: {len(results)} cases ===")
    for category in dict.fromkeys(r.category for r in results):
        print(f"  {category:20s} {rate([r for r in results if r.category == category])}")

    answered = [r for r in results if r.source == "llm"]
    fell_back = [r for r in results if r.source == "rules"]
    fast = [r for r in results if r.source == "fast"]
    valid = engine == "rules" or len(fell_back) <= 0.1 * len(results)
    if fast:
        print(f"\n  {len(fast)}/{len(results)} cases took the fast path (no model call)")
    if engine != "rules" and fell_back:
        reasons = sorted({r.fallback_reason or "?" for r in fell_back})
        print(f"\n  !! the model did not answer {len(fell_back)}/{len(results)} cases: {reasons[:3]}")
        if not valid:
            print("  !! these numbers describe the rule-based fallback, not the model; the criteria below are NOT valid")

    structural = [r for r in results if r.category in STRUCTURAL]
    hallucination = [r for r in results if r.category in HALLUCINATION]
    longform = [r for r in results if r.category == "longform"]
    lint_hits = [(r.id, p) for r in results for p in r.lint]
    budget = [r.llm_ms + r.post_ms for r in answered if r.words <= LLM_BUDGET_MAX_WORDS]
    p95 = percentile(budget, 0.95)
    criteria = {
        "structural >= 98%": (bool(structural) and sum(r.passed for r in structural) / len(structural) >= 0.98, rate(structural)),
        "zero hallucination (questions never answered)": (all(r.passed for r in hallucination), rate(hallucination)),
        "zero spacing/list defects": (not lint_hits, f"{len(lint_hits)} found"),
        **({"long-form checks all pass": (all(r.passed for r in longform), rate(longform))} if longform else {}),
        f"p95 model+post <= {LLM_P95_BUDGET_MS:.0f} ms (<= {LLM_BUDGET_MAX_WORDS} words)":
            (bool(budget) and p95 <= LLM_P95_BUDGET_MS, f"{p95:.0f} ms over {len(budget)} calls" if budget else "no model calls"),
    }
    print("\n  acceptance criteria" + ("" if valid else "  (NOT VALID: model unavailable)"))
    for name, (ok, detail) in criteria.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if budget:
        print(f"    model latency: p50 {statistics.median(budget):.0f} ms, p95 {p95:.0f} ms, max {max(budget):.0f} ms")

    failures = [r for r in results if not r.passed]
    for r in failures:
        print(f"\n  FAIL {r.id} [{r.category}] via {r.source}{f' ({r.fallback_reason})' if r.fallback_reason else ''}"
              f"\n    raw     : {r.raw[:160]!r}\n    expected: {r.expected!r}\n    got     : {r.got!r}"
              + (f"\n    checks  : {r.reasons}" if r.reasons else ""))
    for case_id, problem in lint_hits:
        print(f"  LINT {case_id}: {problem}")
    return {"valid": valid, "criteria": {k: v[0] for k, v in criteria.items()}, "p95_ms": p95}


async def text_eval(engine: str, cases: List[Dict[str, Any]], out_dir: Path) -> int:
    service = build_service(engine, MODE)
    results = [await run_case(service, c) for c in cases]
    summary = report(results, engine)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{engine.replace(':', '-')}.json"
    path.write_text(json.dumps({"engine": engine, "model": getattr(service.backend, "model", None), "summary": summary,
                                "results": [asdict(r) for r in results]}, indent=1))
    print(f"\n  saved {path}")
    ok = all(summary["criteria"].values()) and summary["valid"]
    return 0 if ok else (2 if not summary["valid"] else 1)


# ---------------------------------------------------------------------------
# Audio evaluation (macOS `say` -> Whisper -> formatter)
# ---------------------------------------------------------------------------
def speak(text: str, voice: str = "Samantha") -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(path), text], check=True)
        return path.read_bytes()


async def audio_eval(engine: str, cases: List[Dict[str, Any]], out_dir: Path) -> int:
    from src.voice_flow.asr.audio import decode_audio_bytes
    from src.voice_flow.asr.whisper_client import get_transcriber
    transcriber, service = get_transcriber(), build_service(engine, MODE)
    print(f"loading Whisper {transcriber.model_name} ({transcriber.compute_type}) ... {transcriber.load():.1f}s")
    rows = []
    for case in cases:
        audio = decode_audio_bytes(speak(case["raw"]))
        context = effective_context(case.get("app_context", "general"))
        cursor = CursorContext(preceding_text=case.get("preceding_text", ""), following_text=case.get("following_text", ""))
        t0 = time.perf_counter()
        asr = await transcriber.transcribe(audio, case.get("custom_dictionary", []))
        result = await service.compile(asr.text, context, cursor, case.get("custom_dictionary", []), STYLE)
        stitched = PostProcessor.stitch(result.formatted_text, cursor, context)
        total = (time.perf_counter() - t0) * 1000.0
        rows.append({"id": case["id"], "audio_ms": asr.audio_ms, "asr_ms": asr.asr_ms, "llm_ms": result.processing_time_ms,
                     "total_ms": total, "source": result.source, "heard": asr.text, "got": stitched.insert_text,
                     "expected": case.get("expected", "")})
        print(f"\n{case['id']}  {asr.audio_ms / 1000:.1f}s audio | whisper {asr.asr_ms:.0f} ms | {result.source} {result.processing_time_ms:.0f} ms"
              f" | total {total:.0f} ms\n  heard   : {asr.text!r}\n  got     : {stitched.insert_text!r}\n  expected: {case.get('expected', '(scored on checks)')!r}")
    totals = [r["total_ms"] for r in rows]
    print(f"\nend-to-end after the speaker stops (n={len(rows)}): p50 {statistics.median(totals):.0f} ms, "
          f"p95 {percentile(totals, 0.95):.0f} ms, max {max(totals):.0f} ms")
    print(f"  whisper p50 {statistics.median(r['asr_ms'] for r in rows):.0f} ms | formatter p50 {statistics.median(r['llm_ms'] for r in rows):.0f} ms"
          f" | model {sum(r['source'] == 'llm' for r in rows)}, fast {sum(r['source'] == 'fast' for r in rows)}, rules {sum(r['source'] == 'rules' for r in rows)} of {len(rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"audio_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="gemini", help="gemini | rules | openai[:model] | configured")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--category", action="append", help="only these categories (repeatable)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smart", action="store_true", help="what a user gets: plain sentences skip the model (default scores the model on every case)")
    parser.add_argument("--audio", action="store_true", help="macOS only: speak each case through real Whisper")
    parser.add_argument("--style", choices=["clean", "polished"], help="score this writing style (default: the server default)")
    parser.add_argument("--contexts", action="store_true", help="also run the email / Slack / code / Markdown cases and switch those formats on")
    parser.add_argument("--out", default=str(ROOT / "local_run" / "voice_eval"))
    args = parser.parse_args()

    if args.contexts:
        settings.voice_app_contexts_enabled = True
    cases = load_cases(args.cases, args.contexts)
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if args.limit:
        step = max(1, len(cases) // args.limit)         # spread the sample over every category
        cases = cases[::step][: args.limit]
    global MODE, STYLE
    MODE = "smart" if args.smart or args.audio else "always"
    STYLE = args.style
    runner = audio_eval if args.audio else text_eval
    return asyncio.run(runner(args.engine, cases, Path(args.out)))


if __name__ == "__main__":
    sys.exit(main())
