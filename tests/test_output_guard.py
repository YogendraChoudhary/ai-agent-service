"""Write mode and Enhance output that goes straight into the document (src/output_guard.py)."""
from types import SimpleNamespace

import pytest
from agno.agent import RunEvent

import src.text_processing_agent as tpa
from src.output_guard import guard_enhance, guard_write
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import runtime  # noqa: F401  (fixture)


def write(text, query="x", size=7):
    out = list(guard_write([text[i:i + size] for i in range(0, len(text), size)], query))
    return "".join(c for c in out if isinstance(c, str)), [c["__event__"]["data"] for c in out if not isinstance(c, str)]


def enhance(text, instruction, size=7):
    return "".join(guard_enhance([text[i:i + size] for i in range(0, len(text), size)], instruction))


SIZES = [1, 3, 7, 50, 10_000]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("answer, query, action", [
    ("Please provide the list you'd like me to add items to, and I'll assist you with that.", "Add 3 more items to the list", "draft"),
    ("Sure! Please provide the meeting notes that you would like summarized into action items.", "Summarize the meeting notes", "ask"),
    ("Please provide the existing list so I can add three more items to it.", "Add 3 more items to the list", "draft"),
    ("I don't see the text you'd like me to rewrite. Could you paste it here?", "Rewrite the paragraph", "edit"),
])
def test_requests_for_input_become_needs_context(answer, query, action, size):
    text, events = write(answer, query, size)
    assert text == "" and events[0]["status"] == "needs_context" and events[0]["suggested_action"] == action
    assert events[0]["source"] == "model" and "wasn't included" in events[0]["message"]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("answer, query", [
    ("Hi team,\n\nCould you please share your availability for next week so I can schedule the review?\n\nThanks,\nSam",
     "Write an email asking the team for availability"),
    ("Could you please share your availability for next week?", "Write a one-line note asking for availability"),
    ("Dear Ann,\n\nThanks for the update.\n\nLet me know if you have any questions.\n\nBest,\nSam", "Write a reply to Ann"),
    ("Please provide feedback on the attached proposal by Friday.", "Write a reminder for reviewers"),
    ("```python\ndef f():\n\n    return 1\n```\n", "Write a Python function"),
    ("# Plan\n\nStep one.\n\nStep two.\n", "Write a plan"),
    ("Text with --- inside.\n\n---\n\nMore content.\n", "Write two paragraphs"),
])
def test_drafted_content_passes_through_unchanged(answer, query, size):
    assert write(answer, query, size) == (answer, [])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("answer, expected", [
    ("Sure! Here's a checklist:\n\n1. Laptop\n2. Accounts\n", "1. Laptop\n2. Accounts\n"),
    ("Here is your product description:\n\n# App\n\nFast.\n", "# App\n\nFast.\n"),
    ("# Product\n\nGreat app.\n\n---\n\nFeel free to adjust any sections to better fit your app!", "# Product\n\nGreat app.\n\n"),
    ("# Plan\n\nStep one.\n\nI hope this helps!", "# Plan\n\nStep one.\n\n"),
    ("# Plan\n\nStep one.\n\nLet me know if you'd like me to expand any step.\n", "# Plan\n\nStep one.\n\n"),
    ("# Card\n\n[Company name] ships fast.\n\nNote: replace the placeholders with your details.", "# Card\n\n[Company name] ships fast.\n\n"),
])
def test_prefaces_and_closing_notes_about_the_draft_are_dropped(answer, expected, size):
    assert write(answer, "Write it", size) == (expected, [])


def test_a_lone_closing_line_is_never_dropped_to_nothing():
    assert write("I hope this helps!", "Write a sign-off line") == ("I hope this helps!", [])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("answer, instruction, expected", [
    ("Add a bar chart.\n\nWe hired 12 people.", "Add a bar chart", "We hired 12 people."),
    ("Add a bar chart, Text: We hired 12 people.", "Add a bar chart", "We hired 12 people."),
    ("Make it more formal\n\nIn 2025 we hired.", "make it more formal", "In 2025 we hired."),
    ("Text: We hired 12.", "Make it formal", "We hired 12."),
    ("<text>\nWe hired 12.\n</text>", "Make it formal", "We hired 12."),
    ("We hired 12 people.", "Add a bar chart", "We hired 12 people."),
    ("Add bar charts to reports is our goal.\n\nMore.", "Add a bar chart", "Add bar charts to reports is our goal.\n\nMore."),
])
def test_enhance_drops_an_echoed_instruction_only(answer, instruction, expected, size):
    assert enhance(answer, instruction, size) == expected


def _reply(monkeypatch, text):
    def run(agent, prompt, **kwargs):
        for i in range(0, len(text), 9):
            yield SimpleNamespace(event=RunEvent.run_content, content=text[i:i + 9], metrics=None)
        yield SimpleNamespace(event=RunEvent.run_completed, content=None, metrics={"input_tokens": 10, "output_tokens": 5})
    monkeypatch.setattr(tpa.Agent, "run", run)


def test_write_stream_turns_a_request_for_input_into_needs_context(runtime, monkeypatch):  # noqa: F811
    client, state = runtime
    _reply(monkeypatch, "Please provide the list you'd like me to add items to, and I'll assist you with that.")
    events = parse_sse_events(client.post("/agent/stream", json={
        "type": "general", "query": "Please add three onboarding items to our checklist", "model": "openai",
        "api_key": "endpoint-key"}).text)
    assert not any(e["data"].get("type") == "content" for e in events)
    event = next(e["data"] for e in events if e["event"] == "needs_context")
    assert event["source"] == "model" and event["suggested_action"] == "draft"
    assert events[-1]["data"]["type"] == "done"


def test_write_stream_drops_a_closing_note(runtime, monkeypatch):  # noqa: F811
    client, _ = runtime
    _reply(monkeypatch, "# App\n\nFast and simple.\n\nFeel free to adjust any sections to better fit your app!")
    events = parse_sse_events(client.post("/agent/stream", json={
        "type": "general", "query": "Write a product description for our app", "model": "openai", "api_key": "endpoint-key"}).text)
    content = "".join(e["data"]["content"] for e in events if e["data"].get("type") == "content")
    assert content == "# App\n\nFast and simple.\n\n"


def test_enhance_stream_never_starts_with_the_instruction(runtime, monkeypatch):  # noqa: F811
    client, state = runtime
    _reply(monkeypatch, "Add a bar chart.\n\nWe hired 12 people in 2025.")
    result = client.post("/agent/process", json={
        "type": "enhance_with_ai", "query": "We hired 12 people in 2025.", "secondary_prompt": "Add a bar chart",
        "model": "openai", "api_key": "endpoint-key", "suggest_graphs": False}).json()
    assert result["markdown_response"] == "We hired 12 people in 2025."


def test_enhance_prompt_keeps_instruction_and_text_apart():
    import api
    query = api.build_full_query("enhance_with_ai", "We hired 12.", "Add a bar chart")
    assert query == "<instruction>Add a bar chart</instruction>\n<text>\nWe hired 12.\n</text>"
    assert any("never copy the instruction" in line for line in tpa.AGENT_CONFIGS["enhance"]["instructions"])
