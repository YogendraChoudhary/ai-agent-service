"""Draft placement: the anchor block's exact text and UTF-16 offsets, resolved in code, or a clarification."""
import json

import pytest

from src.document_context import DocumentContext, prepare_context
from src.document_edit.blocks import document_blocks, utf16_length
from src.document_placement import resolve_placement
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import body, runtime  # noqa: F401  (fixture)

DOC = ("# Launch guide\n\n## Introduction\n\nThis intro explains the launch.\n\nA second intro paragraph about goals.\n\n"
       "## Pricing\n\nThe price is USD 20.\n\n| Plan | USD |\n|---|---:|\n| Team | 20 |\n\n### Discounts\n\nStudents get 50% off.\n\n"
       "## Conclusion\n\nWe recommend launching in May.\n")


def block(starts_with):
    return next(b for b in document_blocks(DOC) if b.text.startswith(starts_with))


@pytest.mark.parametrize("task, anchor, position", [
    ("Draft a summary below the Pricing heading", "## Pricing", "after"),
    ("Add a line under the \"Pricing\" heading", "## Pricing", "after"),
    ("Draft a paragraph below heading Discounts", "### Discounts", "after"),
    ("Add a line below Pricing", "## Pricing", "after"),
    ("Add a note at the end of the Pricing section", "Students get", "after"),     # subsections belong to the section
    ("write a paragraph after the Pricing section", "Students get", "after"),
    ("Add a disclaimer above the Conclusion heading", "## Conclusion", "before"),
    ("Write an intro at the top of the document", "# Launch guide", "before"),
    ("Add a closing line at the end", "We recommend", "after"),
    ("Draft a sentence after the second paragraph", "A second intro", "after"),
    ("add a summary after the last paragraph in Introduction", "A second intro", "after"),
    ("add a paragraph after the paragraph about goals", "A second intro", "after"),
    ("put a caption below the table", "| Plan", "after"),
    ("Write a tip after 'Students get'", "Students get", "after"),
    ("Write a summary after the Conclusion", "We recommend", "after"),             # bare "after X": after the section
    ("Add a line after Pricing", "Students get", "after"),
    ("Add a line following Introduction", "A second intro", "after"),
    ("Add a paragraph between Pricing and Conclusion", "## Conclusion", "before"),
    ("Add a paragraph between the Pricing and Conclusion sections", "## Conclusion", "before"),
    ("Write a transition between the intro and the conclusion that explains the timeline", "## Conclusion", "before"),
    ("Add a paragraph between Pricing and Risks", "Students get", "after"),
])
def test_named_places_resolve_to_one_top_level_block(task, anchor, position):
    placement, question = resolve_placement(DOC, task)
    expected = block(anchor)
    assert question is None and placement.position == position and placement.text == expected.text
    assert (placement.start_utf16, placement.end_utf16) == (utf16_length(DOC[:expected.start]), utf16_length(DOC[:expected.end]))


@pytest.mark.parametrize("task", ["Write the next paragraph", "write about what happens after the merger",
                                  "Explain the difference between cats and dogs",
                                  "Continue the conclusion", "Draft a paragraph on pricing before launch day"])
def test_no_named_place_means_the_cursor(task):
    assert resolve_placement(DOC, task) == (None, None)


@pytest.mark.parametrize("task, phrase", [
    ("Add text below the Risks heading", "no heading named 'Risks'"),
    ("add a paragraph after the paragraph about launch", "More than one paragraph"),
    ("Draft a line after the ninth paragraph", "There is no ninth paragraph"),
    ("Add a note after 'not in the document'", "was not found"),
    ("add a note below this paragraph", "Place the cursor"),
    ("Add a note between the Risks and Benefits sections", "couldn't find the sections"),
])
def test_unclear_or_missing_places_need_clarification(task, phrase):
    placement, question = resolve_placement(DOC, task)
    assert placement is None and phrase in question


def test_this_paragraph_uses_the_cursor_and_offsets_count_utf16():
    text = "## Émojis 😀\n\nFirst 😀 paragraph.\n\nSecond paragraph.\n"
    placement, _ = resolve_placement(text, "add a note below this paragraph", text.index("First") + 3)
    assert placement.text == "First 😀 paragraph.\n"
    assert placement.start_utf16 == utf16_length(text[:text.index("First")])
    assert placement.end_utf16 == placement.start_utf16 + utf16_length(placement.text)


def test_prepared_draft_carries_placement_and_is_focused_on_the_anchor():
    prepared = prepare_context(DocumentContext(text=DOC, cursor_utf16=0), "Draft a summary below the Pricing heading", "draft")
    assert prepared.metadata.status == "ready" and prepared.metadata.placement.text == "## Pricing\n"
    envelope = json.loads(prepared.prompt.split("\n\nUSER_TASK:")[0])
    assert envelope["focus"]["start_utf16"] == utf16_length(DOC[:block("## Pricing").end])


def test_ask_never_gets_a_placement():
    prepared = prepare_context(DocumentContext(text=DOC), "What is below the Pricing heading?", "ask")
    assert prepared.metadata.placement is None and prepared.metadata.status == "ready"


def test_stream_sends_placement_before_the_draft(runtime):  # noqa: F811
    client, state = runtime
    events = parse_sse_events(client.post("/agent/stream", json=body(
        query="Draft a short summary below the Conclusion heading", document_action="draft",
        document_context={"text": DOC})).text)
    ready = next(e for e in events if e["event"] == "document_context_ready")
    assert ready["data"]["placement"] == {"position": "after", "text": "## Conclusion\n",
        "start_utf16": utf16_length(DOC[:block("## Conclusion").start]), "end_utf16": utf16_length(DOC[:block("## Conclusion").end])}
    content = [i for i, e in enumerate(events) if e["data"].get("type") == "content"]
    assert content and events.index(ready) < content[0] and len(state["calls"]) == 1


def test_clarification_streams_no_content_and_calls_no_model(runtime):  # noqa: F811
    client, state = runtime
    events = parse_sse_events(client.post("/agent/stream", json=body(
        query="Draft a paragraph below the Risks heading", document_action="draft", document_context={"text": DOC})).text)
    ready = next(e["data"] for e in events if e["event"] == "document_context_ready")
    assert ready["status"] == "needs_clarification" and "Risks" in ready["message"] and ready["placement"] is None
    assert not any(e["data"].get("type") == "content" for e in events) and not state["calls"]
    assert events[-1]["data"]["type"] == "done"


# --- "next to" / "beside", an unnamed heading, and "top" that is only a word in a name ------------------------------------------------

LIST_DOC = ("# Report\n\n## Top 10\n\n1. Planning\n2. Reviews\n\n## Office\n\nOur office is at 12 Park Lane.\n\n"
            "## Top 10 list of risks\n\n- Delay\n- Cost\n\n## Pricing\n\nUSD 20.\n")


@pytest.mark.parametrize("task, anchor, position", [
    ("add a note next to the Pricing heading", "## Pricing", "after"),
    ("add a note next to Pricing", "## Pricing", "after"),
    ("add a line beside the Conclusion heading", "## Conclusion", "after"),
    ("Add a line alongside the Conclusion heading", "## Conclusion", "after"),
    ("add a note right next to the table", "| Plan", "after"),
    ("put a caption beside the table", "| Plan", "after"),
    ("add a note next to the paragraph about goals", "A second intro", "after"),
])
def test_next_to_and_beside_place_after_the_named_block(task, anchor, position):
    placement, question = resolve_placement(DOC, task)
    assert question is None and placement.position == position and placement.text == block(anchor).text


def test_next_to_a_list_named_by_its_heading_goes_beside_the_list_not_to_the_top_of_the_document():
    placement, question = resolve_placement(LIST_DOC, "add a note next to the Top 10 list")
    assert question is None and placement.position == "after" and placement.text.strip() == "1. Planning\n2. Reviews"


def test_next_to_a_heading_called_top_10_list_is_that_heading():
    doc = "# Report\n\n## Top 10 list\n\n1. Planning\n\n## Pricing\n\nUSD 20.\n"
    placement, question = resolve_placement(doc, "add a note next to the Top 10 list")
    assert question is None and (placement.position, placement.text) == ("after", "## Top 10 list\n")


@pytest.mark.parametrize("task", [
    "add a note at the top of the document", "add a note to the top of the document", "Add a one-line summary at the top",
    "add a note at the top, please", "put a line at the top and keep it short", "add a banner to the top of the page",
])
def test_top_still_means_the_top_of_the_document_when_it_clearly_does(task):
    placement, question = resolve_placement(DOC, task)
    assert question is None and (placement.position, placement.text) == ("before", block("# Launch guide").text)


@pytest.mark.parametrize("task", [
    "add a note next to the Top 10 list", "Add a note close to the Top 10 list", "put a line next to the top of the table",
    "add a note at the Top 10 list", "write about the Top 10 products", "add a line at the End of Year report",
])
def test_top_or_end_inside_a_name_is_not_an_edge_of_the_document(task):
    placement, _ = resolve_placement(LIST_DOC, task)
    assert placement is None or placement.text != "# Report\n"


@pytest.mark.parametrize("task, anchor, position", [
    ("add a line below the heading", "## Introduction", "after"),
    ("add a line above the heading", "## Introduction", "before"),
    ("add a line under this heading", "## Introduction", "after"),
    ("add a line next to the heading", "## Introduction", "after"),
])
def test_an_unnamed_heading_is_the_one_the_cursor_is_under(task, anchor, position):
    focus = DOC.index("A second intro")
    placement, question = resolve_placement(DOC, task, focus)
    assert question is None and (placement.position, placement.text) == (position, block(anchor).text)


def test_the_cursor_inside_a_heading_means_that_heading():
    placement, _ = resolve_placement(DOC, "add a line below the heading", DOC.index("## Pricing") + 4)
    assert placement.text == block("## Pricing").text


def test_an_unnamed_heading_without_a_cursor_is_the_only_heading_or_a_question():
    only = "## Contact\n\nReach us.\n"
    placement, question = resolve_placement(only, "add a line below the heading")
    assert question is None and placement.text == "## Contact\n"
    placement, question = resolve_placement(DOC, "add a line below the heading")
    assert placement is None and "more than one heading" in question and "Pricing heading" in question


def test_the_title_is_the_top_level_heading_wherever_the_cursor_is():
    placement, question = resolve_placement(DOC, "add a subtitle below the title", DOC.index("The price"))
    assert question is None and (placement.position, placement.text) == ("after", "# Launch guide\n")
    placement, question = resolve_placement("Some intro.\n\n## A\n\ntext\n\n## B\n\ntext\n", "add a line below the title")
    assert placement is None and "no title heading" in question
    placement, _ = resolve_placement("## A\n\ntext\n\n## B\n\ntext\n", "add a line below the title")
    assert placement.text == "## A\n"      # a document that opens with a heading has that as its title


def test_an_unnamed_section_goes_after_the_whole_section():
    placement, question = resolve_placement(DOC, "add a note after the section", DOC.index("The price"))
    assert question is None and placement.text == block("Students get").text


def test_a_cursor_before_any_heading_cannot_stand_for_the_heading():
    placement, question = resolve_placement("Intro text.\n\n## A\n\ntext\n\n## B\n\ntext\n", "add a line below the heading", 3)
    assert placement is None and "not under a heading" in question


def test_prepare_context_uses_the_cursor_for_an_unnamed_heading():
    context = DocumentContext(text=DOC, cursor_utf16=DOC.index("A second intro"))
    prepared = prepare_context(context, "Add a closing line below the heading", "draft")
    assert prepared.metadata.placement.text == block("## Introduction").text and prepared.metadata.status == "ready"


# ---- fixes from the regression audit ----------------------------------------------------------------------------------------------

EDGE_DOC = ("# Report\n\n## About\n\nWe make tools.\n\n## Top 10 list\n\n1. One\n2. Two\n\n## Pricing\n\nUSD 20.\n\n## Conclusion\n\nDone.\n")


@pytest.mark.parametrize("task, position", [
    ("write a conclusion at the end summarizing the key points", "after"), ("add a sign-off at the end thanking the reader", "after"),
    ("append a note to the end saying the document is a draft", "after"), ("put a disclaimer at the bottom stating that prices may change", "after"),
    ("add a note at the end - keep it short", "after"), ("add a note at the end (short)", "after"), ("add a note at the end, thanks", "after"),
    ("add a summary at the top highlighting the main findings", "before"), ("add a banner at the top listing the changes", "before"),
    ("add a note at the top by the author", "before"), ("add a note at the start asking readers to review", "before"),
])
def test_an_edge_followed_by_the_rest_of_the_sentence_is_still_the_edge(task, position):
    placement, question = resolve_placement(EDGE_DOC, task)
    assert question is None and placement.position == position
    assert placement.text == (block("# Launch guide").text if False else (document_blocks(EDGE_DOC)[-1].text if position == "after" else "# Report\n"))


@pytest.mark.parametrize("task", ["add a note at the Top 10 list", "add a line at the Top Picks table", "add a note to the Start Here page",
                                  "write at the End Notes section"])
def test_an_edge_word_that_starts_a_name_is_not_an_edge(task):
    placement, _ = resolve_placement(EDGE_DOC, task)
    assert placement is None or placement.text not in ("# Report\n", document_blocks(EDGE_DOC)[-1].text)


@pytest.mark.parametrize("task, anchor, position", [
    ("Add a paragraph about working alongside engineers below the Pricing heading", "## Pricing", "after"),
    ("Write a paragraph on living next to a volcano above the Conclusion heading", "## Conclusion", "before"),
    ("Add a paragraph that is beside the point after the Pricing section", "USD 20.", "after"),
    ("Draft a blurb for the bench beside the lake, below the Pricing heading", "## Pricing", "after"),
    ("add a note next to the end of the Pricing section", "USD 20.", "after"),
])
def test_a_place_word_used_as_a_topic_does_not_hide_the_real_place(task, anchor, position):
    placement, question = resolve_placement(EDGE_DOC, task)
    assert question is None and (placement.position, placement.text.strip()) == (position, anchor)


def test_a_missing_heading_after_a_topic_word_is_still_reported():
    placement, question = resolve_placement(EDGE_DOC, "Add a paragraph about working alongside engineers below the Risks heading")
    assert placement is None and "Risks" in question


def test_the_older_place_words_still_decide_with_their_first_candidate():
    # exactly the old behaviour: "under …" is read first and cannot be found
    placement, question = resolve_placement(EDGE_DOC, "add under construction below the title")
    assert placement is None and question is not None


@pytest.mark.parametrize("task", ["add a line below this heading please", "add a line below this heading in bold",
                                  "add a line under this heading (short)"])
def test_this_heading_may_be_followed_by_other_words(task):
    placement, question = resolve_placement(DOC, task, DOC.index("A second intro"))
    assert question is None and placement.text == block("## Introduction").text


@pytest.mark.parametrize("task", ["add a line below the heading please", "add a line below the heading now", "add a line below the heading (short)"])
def test_the_heading_may_be_followed_by_a_filler_word(task):
    placement, question = resolve_placement(DOC, task, DOC.index("A second intro"))
    assert question is None and placement.text == block("## Introduction").text


def test_a_heading_literally_called_title_or_section_is_found_by_its_name():
    doc = "# Report\n\n## Title\n\ntext\n\n## Section\n\ntext\n"
    assert resolve_placement(doc, "add a line below Title")[0].text == "## Title\n"
    assert resolve_placement(doc, "add a line after Section")[0].text.strip() == "text"
    assert resolve_placement(doc, "add a subtitle below the title")[0].text == "## Title\n"      # a heading with that name wins, as before
    plain = "# Report\n\n## Intro\n\ntext\n"
    assert resolve_placement(plain, "add a subtitle below the title")[0].text == "# Report\n"


def test_a_leading_bom_does_not_hide_the_title():
    placement, question = resolve_placement("﻿# Report\n\ntext\n", "add a subtitle below the title")
    assert question is None and placement.text == "# Report\n"


@pytest.mark.parametrize("task", ["add a line below the next list", "add a line below the previous table", "add a note after the selected paragraph",
                                  "add a note after the preceding paragraph"])
def test_cursor_relative_words_are_not_a_name_for_a_block(task):
    doc = "# R\n\n## One\n\n- a\n\nA paragraph that mentions the next list and the previous table and the selected paragraph.\n"
    placement, question = resolve_placement(doc, task)
    assert placement is None or "mentions the next list" not in placement.text


def test_this_title_is_the_document_title_even_with_trailing_words():
    placement, question = resolve_placement(DOC, "add a line below this title, thanks", DOC.index("A second intro"))
    assert question is None and placement.text == block("# Launch guide").text


def test_the_edge_look_back_is_bounded():
    import time
    started = time.perf_counter()
    resolve_placement(EDGE_DOC, "go to the end " * 6000)
    assert time.perf_counter() - started < 1.0


def test_the_setext_title_level_survives_odd_line_separators():
    placement, question = resolve_placement("Annual\x0bReport\n=====\n\ntext\n\n## B\n\ntext\n", "add a subtitle below the title")
    assert question is None and placement.text.startswith("Annual")
