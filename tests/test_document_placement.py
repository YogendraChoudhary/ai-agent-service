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
