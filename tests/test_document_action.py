"""document_action "auto": the decision itself. Rules first, one light-tier call when the wording is unclear, Ask when that fails."""
import time
from unittest.mock import Mock

import pytest

from config import settings
from src.document_action import CLASSIFIER_SYSTEM, LABELS, ActionDecision, decide_by_rules, decide_with_model
from src.intent_classifier import classify_label

DOC = "# Launch guide\n\n## Pricing\n\nUSD 20 per seat.\n\n## Intro\n\nHello there.\n"


@pytest.mark.parametrize("task, action", [
    # questions and advice
    ("What does the Team plan cost?", "ask"), ("Summarize the document", "ask"), ("Is the pricing consistent?", "ask"),
    ("Why is the second paragraph so long?", "ask"), ("Review the conclusion", "ask"),
    # new text
    ("add a short note about annual billing below the Pricing heading", "draft"), ("Continue writing", "draft"),
    ("add a new section about security", "draft"), ("write the next paragraph", "draft"), ("add Acme Inc. below the Contact heading", "draft"),
    ("add a note next to the Top 10 list", "draft"), ("add more detail to the Benefits paragraph", "draft"),
    ("improve the intro by adding more detail", None),            # rewrite AND new facts: unclear, never guessed
    # changes to what is there
    ("make the Introduction more concise", "edit"), ("fix the typo in the title", "edit"), ("rewrite the conclusion", "edit"),
    ("add (2024) next to the title", "edit"), ("add a phone number after the address", "edit"), ("Translate this paragraph to Spanish", "edit"),
    ("rename the Contact heading", "edit"), ("make the quote shorter", "edit"),
    # standalone pieces: only the detectors Write itself routes on, and only when the task does not point at the document
    ("Write a blog post about remote work for small teams", "write"),
    ("Write a Python function that checks if an email address is valid", "write"),
    ("Generate an image of a cat on a sofa", "write"),
    ("Make a bar chart of quarterly revenue: Q1 10, Q2 14, Q3 9, Q4 17", "write"),
    ("Write a blog post about the pricing section below", None),
    ("Write a blog post about this document", None),
    # unclear wording is left to the classifier
    ("Write a short essay about cats", None), ("How can I make the intro shorter?", None), ("please make it better", None),
    ("Write a closing sentence for the Conclusion", None), ("Anything unclear in the pricing?", None),
    ("Compose a thank-you note to a customer", None), ("", None),
])
def test_rules_decide_only_when_the_wording_points_one_way(task, action):
    decision = decide_by_rules(task, DOC)
    assert (decision.action if decision else None) == action
    if decision:
        assert decision.source == "rules" and not decision.ambiguous


def test_a_mixed_request_is_never_guessed():
    for task in ["What is wrong with the intro? Fix it.", "Is the tone formal? Rewrite it if so.", "Rewrite the intro and add a new section"]:
        assert decide_by_rules(task, DOC) is None


def test_scope_and_analysis_requests_can_only_be_ask():
    for kwargs in ({"scope": "selection"}, {"scope": "focused"}, {"analysis": "summary"}):
        decision = decide_by_rules("rewrite the intro", DOC, **kwargs)
        assert (decision.action, decision.source, decision.reason) == ("ask", "request", "scope_requires_ask")


def test_an_empty_document_has_nothing_to_edit_so_ask_explains_it():
    assert decide_by_rules("make it shorter", "  \n").action == "ask"
    assert decide_by_rules("make it shorter", "  \n").reason == "empty_document"
    assert decide_by_rules("write the next paragraph", "").action == "draft"
    assert decide_by_rules("What is this about?", "").action == "ask"


def test_an_instruction_too_long_for_edit_is_left_to_the_classifier():
    assert decide_by_rules("make the intro more concise. " * 100, DOC) is None


def test_the_decision_is_reported_in_the_documented_shape():
    assert decide_by_rules("add (2024) next to the title", DOC).event() == {
        "requested_action": "auto", "action": "edit", "source": "rules", "reason": "existing_prose", "ambiguous": False}
    assert ActionDecision("ask", "fallback", "classifier_unavailable", True).event()["ambiguous"] is True


# ---- the classifier -----------------------------------------------------------------------------------------------------------------

def test_the_classifier_answer_is_used_and_sees_a_compact_document_hint_not_the_document():
    classify = Mock(return_value="write")
    decision = decide_with_model("Write a short essay about cats", DOC, has_selection=True, classify=classify)
    assert (decision.action, decision.source, decision.ambiguous) == ("write", "model", False)
    text, system, labels = classify.call_args.args
    assert text == f"[document: {len(DOC.split())} words; text is selected]\nWrite a short essay about cats"
    assert system == CLASSIFIER_SYSTEM and labels == LABELS == ("ask", "draft", "edit", "write")
    assert "Launch guide" not in text


@pytest.mark.parametrize("failure", [Mock(return_value=None), Mock(side_effect=RuntimeError("boom")), Mock(side_effect=TimeoutError())])
def test_without_an_answer_the_request_is_a_read_only_ask_flagged_as_unsure(failure):
    decision = decide_with_model("Write a short essay about cats", DOC, classify=failure)
    assert (decision.action, decision.source, decision.reason, decision.ambiguous) == ("ask", "fallback", "classifier_unavailable", True)


def test_no_classifier_or_a_disabled_one_also_means_ask(monkeypatch):
    assert decide_with_model("x", DOC).action == "ask"
    monkeypatch.setattr(settings, "document_action_classifier_enabled", False)
    classify = Mock(return_value="edit")
    assert decide_with_model("x", DOC, classify=classify).source == "fallback"
    classify.assert_not_called()


def test_an_edit_answer_for_an_empty_document_becomes_ask():
    decision = decide_with_model("tidy this up", "", classify=Mock(return_value="edit"))
    assert (decision.action, decision.reason) == ("ask", "empty_document")


# ---- classify_label: the same key rules as the code classifier ----------------------------------------------------------------------

def test_byok_stays_on_the_users_own_provider_and_key():
    call = Mock(return_value="draft")
    assert classify_label("x" * 1500, "gemini", "user-key", system="s", labels=LABELS, is_byok=True, platform_key="platform", call=call) == "draft"
    assert call.call_args.args[:3] == ("x" * 1000, "gemini", "user-key")


def test_other_users_go_through_the_platform_claude_key_light_tier():
    call = Mock(return_value="edit")
    classify_label("hi", "gemini", "user-key", system="s", labels=LABELS, is_byok=False, platform_key="platform", call=call)
    assert call.call_args.args[1:3] == ("claude", "platform")


@pytest.mark.parametrize("reply", ["edit because it asks to change text", "", None, "Rewrite", 7])
def test_an_answer_outside_the_labels_is_no_answer(reply):
    assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, call=Mock(return_value=reply)) is None


def test_label_answers_are_normalised():
    for reply, label in [(" Draft. ", "draft"), ("ASK", "ask"), ("write\n", "write")]:
        assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, call=Mock(return_value=reply)) == label


def test_no_key_means_no_call_and_provider_errors_are_swallowed():
    call = Mock(return_value="edit")
    assert classify_label("hi", "openai", None, system="s", labels=LABELS, is_byok=True, call=call) is None
    assert classify_label("hi", "openai", "u", system="s", labels=LABELS, is_byok=False, platform_key=None, call=call) is None
    call.assert_not_called()
    assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, call=Mock(side_effect=RuntimeError())) is None


def test_the_wait_is_bounded_by_the_timeout():
    def slow(*args):
        time.sleep(0.2)
        return "edit"
    started = time.perf_counter()
    assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, timeout=0.02, call=slow) is None
    assert time.perf_counter() - started < 0.15


def test_the_label_classifier_has_its_own_slots_so_a_busy_code_classifier_cannot_starve_it():
    import src.intent_classifier as module
    held = [module._SLOTS.acquire(blocking=False) for _ in range(4)]
    try:
        assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, call=Mock(return_value="ask")) == "ask"
    finally:
        for _ in filter(None, held):
            module._SLOTS.release()


# ---- fixes from the regression audit ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", [
    "Correct me if I'm wrong, but does the bakery open in March?", "Tell me about Polish bakery traditions",
    "Write a short story about a robot who wants to rename itself",
    "Write a poem about how to fix a flat tire", "Can I polish the intro or is it fine?",
])
def test_an_edit_word_inside_a_sentence_is_not_an_instruction_to_edit(task):
    # (Known limit of wording rules: "Translate 'good morning' into French" begins with an edit verb about text that is not in the
    # document. The rules call it Edit; Edit then says it found nothing to change.)
    decision = decide_by_rules(task, DOC)
    assert decision is None or decision.action != "edit"


@pytest.mark.parametrize("task", ["make the Introduction more concise", "fix the typo in the title", "Can you make the intro shorter?",
                                  "I think the intro is long, shorten it", "please rename the Pricing heading", "add (2024) next to the title"])
def test_a_polite_or_embedded_imperative_is_still_decided_by_the_rules(task):
    assert decide_by_rules(task, DOC).action == "edit"


@pytest.mark.parametrize("task", ["Suggest names for a new sandwich", "suggest 10 slogans for a bike shop", "Brainstorm slogans for a bike shop"])
def test_generative_requests_are_left_to_the_classifier(task):
    assert decide_by_rules(task, DOC) is None


def test_suggest_about_the_document_is_still_ask():
    assert decide_by_rules("Suggest improvements to the pricing section", DOC).action == "ask"


@pytest.mark.parametrize("task", ["Add up the prices in Pricing and tell me the total", "Add the two prices together"])
def test_arithmetic_is_not_new_text(task):
    decision = decide_by_rules(task, DOC)
    assert decision is None or decision.action != "draft"


def test_the_rules_and_the_classifier_read_a_bounded_digest_with_the_start_and_the_end():
    from src.document_action import digest
    DIGEST_CHARS = settings.document_action_digest_chars
    assert digest("short") == "short"
    long = "A" * 3000 + " rewrite this " + "B" * 3000
    out = digest(long)
    assert len(out) <= DIGEST_CHARS and out.startswith("AAA") and out.endswith("BBB") and " … " in out
    seen = []
    decide_with_model("Z" * 2000 + " Write a poem " + "Y" * 2000, DOC, classify=lambda text, system, labels: seen.append(text) or "write")
    assert len(seen[0]) <= 1000 and seen[0].startswith("[document:") and "ZZZ" in seen[0] and "YYY" in seen[0]


def test_decide_by_rules_is_bounded_for_a_huge_query():
    import time
    started = time.perf_counter()
    decide_by_rules("write a blog post about " * 400, DOC)
    decide_by_rules("add " + "1" * 8000, DOC)
    assert time.perf_counter() - started < 0.5


def test_a_burst_waits_for_a_free_slot_instead_of_giving_up(monkeypatch):
    import threading
    import src.intent_classifier as module
    slots = [module._LABEL_SLOTS.acquire(timeout=2) for _ in range(module.LABEL_CONCURRENCY)]       # (an earlier test's slow call may still hold one)
    assert all(slots)
    threading.Timer(0.05, module._LABEL_SLOTS.release).start()
    try:
        assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, timeout=1.0, call=Mock(return_value="edit")) == "edit"
    finally:
        for _ in range(module.LABEL_CONCURRENCY - 1):
            module._LABEL_SLOTS.release()


def test_when_no_slot_frees_up_within_the_timeout_there_is_no_answer():
    import time
    import src.intent_classifier as module
    slots = [module._LABEL_SLOTS.acquire(timeout=2) for _ in range(module.LABEL_CONCURRENCY)]
    assert all(slots)
    try:
        started = time.perf_counter()
        assert classify_label("hi", "openai", "k", system="s", labels=LABELS, is_byok=True, timeout=0.05, call=Mock(return_value="edit")) is None
        assert time.perf_counter() - started < 0.5
    finally:
        for _ in filter(None, slots):
            module._LABEL_SLOTS.release()
