"""Write mode (type general, prompt only): document-ready output, and needs_context instead of guessing a document."""
import pytest

import src.text_processing_agent as tpa
from src.document_intent import write_needs_document
from src.request_plan import RequestPlan
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import runtime  # noqa: F401  (fixture)


def write(**kw):
    return {"type": "general", "query": "Rewrite this", "model": "openai", "api_key": "endpoint-key", **kw}


@pytest.mark.parametrize("query, action", [
    ("Rewrite this", "edit"), ("please summarize the document", "ask"), ("continue my draft", "draft"),
    ("Add a conclusion to my essay", "draft"), ("fix the grammar in this paragraph", "edit"), ("Continue writing", "draft"),
    ("write the next paragraph", "draft"), ("Make the text above shorter", "edit"), ("What is my document about?", "ask"),
    ("Add 3 more items to the list", "draft"), ("Summarize the meeting notes into action items", "ask"),
    ("fix the table formatting", "edit"), ("Translate the text to French", "edit"), ("continue the story", "draft"),
])
def test_prompts_about_an_unsent_document_name_a_mode(query, action):
    result = write_needs_document(query)
    assert result["status"] == "needs_context" and result["suggested_action"] == action
    assert "wasn't included" in result["message"]


@pytest.mark.parametrize("query", [
    "Write a blog post about dogs", "Write the introduction for an article about solar power",
    "Draft an email to my landlord about the leak", "Create a table of the planets", "Make a bar chart of US GDP",
    "Rewrite this:\nThe dog ran fast.", "Explain photosynthesis", "summarize the history of Rome",
    "write a poem about this autumn", "Summarize the history of Rome", "Create a table of the planets",
    "Explain the theory of relativity", "Write a poem about the sea", "Sort the planets by size",
])
def test_self_contained_prompts_and_prompts_with_text_are_written(query):
    assert write_needs_document(query) is None


def test_stream_returns_needs_context_without_content_model_or_charge(runtime):  # noqa: F811
    client, state = runtime
    events = parse_sse_events(client.post("/agent/stream", json=write()).text)
    event = next(e["data"] for e in events if e["event"] == "needs_context")
    assert event == {"status": "needs_context", "suggested_action": "edit", "ambiguous": False,
                     "message": "This refers to your document, which wasn't included. Use Edit to change existing text."}
    assert not any(e["data"].get("type") == "content" for e in events) and events[-1]["data"]["type"] == "done"
    assert not state["calls"] and not state["charges"]


def test_process_returns_needs_context_and_no_text(runtime):  # noqa: F811
    client, state = runtime
    result = client.post("/agent/process", json=write(query="continue my draft")).json()
    assert result["markdown_response"] == "" and result["needs_context"]["suggested_action"] == "draft"
    assert not state["calls"]


def test_a_self_contained_prompt_is_answered(runtime):  # noqa: F811
    client, state = runtime
    result = client.post("/agent/process", json=write(query="Write a two-line note about Velvet Mango")).json()
    assert result["needs_context"] is None and result["markdown_response"] and len(state["calls"]) == 1


@pytest.mark.parametrize("plan", [None, RequestPlan(execution_mode="smart")])
def test_write_prompt_forbids_notes_and_questions_and_keeps_mermaid_charts(plan):
    prompt = "\n".join(tpa.build_smart_global_instructions(plan))
    assert "GOES STRAIGHT INTO THE USER'S DOCUMENT" in prompt
    assert "Never ask the user to paste" in prompt and "mermaid xychart-beta" in prompt


# 2026-11-20: "Make it more appealing to the users, add some photos and other aspects" with no document was answered by a
# generic page and three unrelated photos. "it"/"this" with nothing else in the prompt is the user's document.
USER_PROMPT = "Make it more appealing to the users, add some photos and other aspects"


@pytest.mark.parametrize("query", [
    USER_PROMPT, "make it better", "Please make it more fun for kids.", "make this more engaging", "spice it up",
    "jazz this up", "add some photos to it", "add images to this, please", "enhance it", "revise this", "update it",
    "make the blog more appealing", "make my email sound friendlier", "make this blog better", "Can you make the intro shorter?",
    "what's the weakest part of this write-up",
])
def test_improve_it_wording_without_a_document_is_stopped(query):
    result = write_needs_document(query)
    assert result is not None and "wasn't included" in result["message"]


@pytest.mark.parametrize("query", [
    "Write a story about a dragon and make it funny", "Make the best list of 10 hiking trails",
    "Write a blog post for my blog about dogs", "Add a twist to this classic recipe for brownies",
    "Make it bullets. First, review Q3. Second, finalize the plan.", "make it a pie chart: apples 30, pears 20",
    "Make a bar chart of US GDP", "Make me a logo for a bakery",
    # image follow-ups: the router edits the previous image when there is one, so the guard must not stop them
    "make it night", "make it darker and rainier", "make it look like an oil painting", "restyle this picture into watercolor",
])
def test_self_contained_and_image_follow_up_prompts_are_still_answered(query):
    assert write_needs_document(query) is None


def test_mixed_wording_names_every_mode():
    result = write_needs_document(USER_PROMPT)
    assert result["ambiguous"] is True and result["suggested_action"] == "ask"
    assert result["message"] == ("This refers to your document, which wasn't included. Use Edit to change existing text, "
                                 "Draft to add new content to it, or Ask to ask about it.")


def test_the_reported_prompt_runs_no_model_and_makes_no_photo(runtime):  # noqa: F811
    client, state = runtime
    events = parse_sse_events(client.post("/agent/stream", json=write(query=USER_PROMPT)).text)
    assert next(e["data"] for e in events if e["event"] == "needs_context")["status"] == "needs_context"
    assert not any(e["data"].get("type") == "content" for e in events)
    assert not state["calls"] and not state["charges"]
