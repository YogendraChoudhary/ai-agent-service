"""Wording rules in src/document_intent.py: the cases the regression audit found, and the old behaviour they must not disturb."""
import time

import pytest

from src.document_intent import (asks_for_new_facts, asks_for_small_addition, asks_to_insert, imperative_edit, intent_signals,
                                 suggest_document_intent, without_small_additions)


# ---- "-ing" forms of new facts: a request, unless something earlier in the clause forbids it -----------------------------------------

@pytest.mark.parametrize("task", [
    "improve the intro by adding more detail", "Rewrite this by adding examples", "Expand it by giving two examples",
    "tighten the text, providing statistics where you can", "Adding more detail to the intro",
    "Add more detail to the intro", "include two examples", "Don't be vague. Add statistics",      # base forms: exactly as before
])
def test_gerund_requests_for_new_facts_are_new_facts(task):
    assert asks_for_new_facts(task)


@pytest.mark.parametrize("task", [
    "Shorten the About paragraph without adding details", "Fix the grammar in the Contact section without adding facts",
    "Rewrite the About paragraph without including statistics", "Shorten this without adding details", "Fix the typo, no adding more detail",
    "Simplify this and avoid giving more detail", "Make it shorter, avoiding adding examples", "Tighten it. Don't go adding statistics",
    "proofread everything, including references", "proofread the document including data tables",
    "add a reference number after the address", "add a reference code next to the title",
])
def test_constraints_and_scopes_are_not_requests_for_new_facts(task):
    assert not asks_for_new_facts(task)


def test_without_adding_details_is_a_plain_edit_for_the_signals_and_the_edit_gate():
    signals = intent_signals("Shorten the About paragraph without adding details")
    assert signals == {"ask": False, "draft": False, "edit": True}
    from src.document_edit.service import NEW_FACTS_MESSAGE, edit_document
    seen = []
    doc = "# Guide\n\n## About\n\nWe make tools that help teams plan their work every day.\n"
    _, result = edit_document(doc, "Shorten the About paragraph without adding details",
                              lambda s, m, spec: seen.append(spec.stage) or '{"status":"no_change","message":"Already short."}')
    assert seen == ["document_plan"] and result.message != NEW_FACTS_MESSAGE


# ---- other senses of "add" ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task, expected", [
    ("Add up the prices in Pricing and tell me the total", {}),
    ("Add the two prices together", {}),
    ("add 2 and 3", {}),
    ("Insert a link to our website on the word Contact", {"edit": True}),
    ("Add a new section about security", {"draft": True}),
    ("add three items to the list", {"draft": True}),
])
def test_arithmetic_and_links_are_not_new_content(task, expected):
    assert {k: v for k, v in intent_signals(task).items() if v} == expected


@pytest.mark.parametrize("task, expected", [
    ("Retitle my draft", {"edit": True}), ("Rename my draft", {"edit": True}), ("Add a date after the title of my draft", {"edit": True}),
    ("Rewrite my draft", {"edit": True}),
    ("Draft a summary below Pricing", {"draft": True}), ("write a draft of the conclusion", {"draft": True}), ("Continue my draft", {"draft": True}),
])
def test_my_draft_names_the_document_and_is_not_a_request_to_draft(task, expected):
    assert {k: v for k, v in intent_signals(task).items() if v} == expected


# ---- small additions: the anchor is a short noun phrase, and a second request after it still counts ------------------------------------

@pytest.mark.parametrize("task", [
    "add (2024) next to the title and add a FAQ section below Pricing", "add a phone number after the address, then add a new section about hours",
    "add (2024) next to the title but also insert a table after the intro",
])
def test_a_real_insertion_after_a_small_addition_is_still_an_insertion(task):
    assert asks_for_small_addition(task) and asks_to_insert(task)
    assert "FAQ" not in without_small_additions(task) or "FAQ" in without_small_additions(task)       # nothing swallowed past the anchor
    assert "section" in without_small_additions(task) or "table" in without_small_additions(task)


@pytest.mark.parametrize("task, small", [
    ("add 2024 to the end of the title", True), ("add (draft) at the end of the paragraph", True), ("add a date at the start of the title", True),
    ("add (2024) at the end of the document", False), ("add (2024) to the end of the section", False), ("add a note at the end of the table", False),
    ("add a phone number to the end of the page", False), ("add 2024 at the start of the list", False),
])
def test_the_end_of_a_block_is_inline_but_the_end_of_a_document_or_section_is_not(task, small):
    assert asks_for_small_addition(task) is small


# ---- an edit verb has to begin the instruction -------------------------------------------------------------------------------------

@pytest.mark.parametrize("task, expected", [
    ("make the Introduction more concise", True), ("fix the typo in the title", True), ("Can you rename the heading?", True),
    ("I think the intro is long, shorten it", True), ("please tighten the text", True), ("Look at the intro and then fix the grammar", True),
    ("add (2024) next to the title", True), ("I want you to translate the intro", True),
    ("Correct me if I'm wrong, but does the bakery open in March?", True),        # begins with the verb: the question check handles it
    ("Tell me about Polish bakery traditions", False), ("Write a short story about a robot who wants to rename itself", False),
    ("Write a poem about how to fix a flat tire", False), ("The word correct has two meanings", False),
])
def test_imperative_edit(task, expected):
    assert imperative_edit(task) is expected


# ---- cost ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [500, 2000, 8000])
def test_adversarial_queries_stay_cheap(n):
    shapes = ["add " + "1" * n, "add " + "1" * (n // 2) + " " + "1" * (n // 2), "add " * (n // 4), "add a " * (n // 6), "add" + " " * n + "x",
              "add 1 next to " * (n // 14), "add (1) after " + "the " * (n // 4), "add " + "(" * n, 'add "' + "a " * (n // 2),
              "add " + "Acme " * (n // 5), "add 1 after " + "a," * (n // 2), "add 1 after the title " + "and add 1 after the title " * (n // 26)]
    for text in shapes:
        for fn in (asks_to_insert, asks_for_small_addition, intent_signals, suggest_document_intent, asks_for_new_facts, imperative_edit):
            started = time.perf_counter()
            fn(text)
            assert time.perf_counter() - started < 0.25, (fn.__name__, text[:30], n)


# ---- round 2 of the audit ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["Add a date after the title of my document", "add a date after the heading of the page", "add (2024) adjacent to the title",
                                  "add (2024) next to the name of the report"])
def test_what_the_anchor_belongs_to_does_not_make_it_a_block(task):
    assert asks_for_small_addition(task) and not asks_to_insert(task)


@pytest.mark.parametrize("task", ["Shorten it, refraining from adding details", "Shorten it and refrain from adding details",
                                  "Tighten the text, forgoing adding examples", "Tighten it, holding off on adding statistics"])
def test_more_ways_to_say_do_not_add_are_constraints(task):
    assert not asks_for_new_facts(task)
