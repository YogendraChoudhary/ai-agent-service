"""
The golden set (src/voice_flow/evals/test_cases.json) as a regression guard for everything that needs no model.

The model-dependent cases are exercised by `python -m src.voice_flow.evals.run_evals`, which calls Gemini. Here the
rule-based fallback must keep passing every case it passed when it was written; the cases it cannot do are listed, so
a new failure is a regression and a newly passing case is not an error.
"""

import json
from collections import Counter

import pytest

from src.voice_flow import router as router_module
from src.voice_flow.evals import run_evals as ev
from src.voice_flow.formatter.engine import Breaker, LlmError, VoiceFormattingService

ALL_CASES = json.loads(ev.CASES_PATH.read_text(encoding="utf-8"))["cases"]
CASES = ev.load_cases()                         # app-format cases (email, Slack, code, Markdown) are skipped while those are off

# Needs a language model: a name or weekday replaced ("make that Alice"), homophones, dictionary mishearings,
# email layout (greeting/sign-off on their own lines), implicit bullet lists and unspoken capitalization of names.
NEEDS_MODEL = {
    "bak-02", "bak-03", "bak-04", "bak-05", "bak-06", "bak-07", "bul-04", "bul-05", "cln-01", "cln-02", "cln-04", "cln-05",
    "cln-07", "cln-08", "num-06", "num-07", "qst-03",
}


def test_dataset_shape():
    assert len(CASES) >= 50
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["category"] for c in CASES} >= ev.STRUCTURAL | ev.HALLUCINATION
    assert all(c["expected"].strip() for c in ALL_CASES if "checks" not in c)
    assert all(c["checks"] for c in ALL_CASES if c["category"] == "longform")
    counts = Counter(c["category"] for c in ALL_CASES if c["category"] != "longform")
    assert min(counts.values()) >= 8, counts          # every category has real coverage
    assert NEEDS_MODEL <= {c["id"] for c in ALL_CASES}
    assert {c["id"] for c in ALL_CASES} - {c["id"] for c in CASES} == {c["id"] for c in ALL_CASES if c.get("app_context", "general") != "general"}
    assert sum(c["category"] == "longform" for c in ALL_CASES) >= 3


@pytest.fixture(scope="module")
def rule_results():
    import asyncio
    service = ev.build_service("rules")

    async def run():
        return {c["id"]: await ev.run_case(service, c) for c in CASES}
    return asyncio.run(run())


@pytest.mark.parametrize("case", [c for c in CASES if c["id"] not in NEEDS_MODEL and c["category"] != "longform"], ids=lambda c: c["id"])
def test_rules_keep_passing_what_they_pass(case, rule_results):
    r = rule_results[case["id"]]
    assert r.passed, f"{r.id}: expected {r.expected!r}, got {r.got!r}"


def test_rules_never_produce_spacing_or_list_defects(rule_results):
    assert [(r.id, r.lint) for r in rule_results.values() if r.lint] == []


def test_rules_never_answer_a_question(rule_results):
    for r in rule_results.values():
        if r.category == "question_guardrail":
            assert r.got.strip().lower().rstrip(".?") .startswith(r.raw.split()[0].lower())


def test_scoring_accepts_alternatives_and_ignores_trailing_spaces():
    case = {"expected": "A\nB", "alternatives": ["A, B"]}
    assert ev.passed("A  \nB ", case) and ev.passed("A, B", case) and not ev.passed("a\nB", case)


def test_lint_catches_each_defect_the_plan_names():
    assert ev.lint("two  spaces", "general")
    assert ev.lint("Tasks:\n1. a\n2. b", "general")                 # uncapitalized items
    assert ev.lint("We have tasks\n1. A\n2. B", "general")          # lead-in without a colon
    assert ev.lint("1. A\n\n2. B", "general")                       # blank line inside a list
    assert not ev.lint("We have tasks:\n1. A\n2. B\n\nThanks,\nAlex", "general")
    assert not ev.lint("Steps:\n1. npm install", "code_editor")     # code context may keep lower case


async def test_a_failed_startup_probe_pauses_the_model_straight_away(monkeypatch):
    class Dead:
        model = "gemini-test"

        async def warm(self):
            raise LlmError("gemini 402 RESOURCE_EXHAUSTED", permanent=True)

        async def generate(self, *a, **k):
            raise AssertionError("must not be called while paused")

    class Whisper:
        async def aload(self):
            return 0.0

    service = VoiceFormattingService(backend=Dead(), timeout=1.0, breaker=Breaker())
    monkeypatch.setattr(router_module, "_service", service)
    monkeypatch.setattr(router_module, "get_transcriber", lambda: Whisper())
    await router_module.warm_up()
    assert not service.breaker.allow() and "402" in service.breaker.last_reason
    result = await service.compile("how do I center a div in css")
    assert result.source == "rules" and result.formatted_text == "How do I center a div in CSS?"
