"""Draft with exact text: "add Acme Inc. below the Contact heading" inserts "Acme Inc." and nothing else, without a model call."""
import pytest

from src.document_context import DocumentContext, prepare_context
from src.document_literal import literal_draft_text
from tests.design.test_design_api import parse_sse_events
from tests.test_document_context_api import body, runtime  # noqa: F401  (fixture)

DOC = "# Acme\n\n## About\n\nWe make tools.\n\n## Contact\n\nReach us during business hours.\n\n## Careers\n\nWe hire in spring.\n"


@pytest.mark.parametrize("task, literal", [
    ("add Acme Inc. below the Contact heading", "Acme Inc."),
    ("Add Acme Inc. below the Contact heading", "Acme Inc."),
    ("add Acme Cloud below the Products heading", "Acme Cloud"),
    ("add Acme next to the Contact heading", "Acme"),
    ("add 555-0100 below the Contact heading", "555-0100"),
    ("add hello@acme.com below the Contact heading", "hello@acme.com"),
    ("add Springfield, IL below the Contact heading", "Springfield, IL"),
    ("insert Draft v2 below the title", "Draft v2"),
    ("put $20 beside the Pricing heading", "$20"),
    ("write $20 under the Contact heading", "$20"),
    ("write hello@acme.com under the Contact heading", "hello@acme.com"),
    ("Please add 'Hiring now' above the Careers heading", "Hiring now"),
    ('add "Built to last." below the About heading', "Built to last."),
    ("add the text “See pricing” at the end of the document", "See pricing"),
    ("Can you add (2024) after the title?", "(2024)"),
])
def test_exact_text_in_the_task_is_the_whole_answer(task, literal):
    assert literal_draft_text(task) == literal


@pytest.mark.parametrize("task", [
    "add a short company bio below the Contact heading",              # asks for writing
    "add Acme Inc. bio below the Contact heading",                    # a kind of text, not a literal
    "add Acme Inc. below the Contact heading and make it bold",       # more to do than insert
    "add Acme Inc. below the Contact heading with a link to acme.com",
    "write a paragraph about Acme Inc. below the Contact heading",
    "add a summary below the Pricing heading", "add Summary below the Pricing heading", "add Conclusion after the last paragraph",
    "add 3 bullet points below the intro", "add 2 examples after the first paragraph", "add more detail below the Pricing heading",
    "Draft a summary below the Pricing heading", "add the heading Risks below the title", "add Pricing Table below the intro",
    "add Hiring now above the Careers heading",                       # plain lower-case words: left to the model and its prompt rule
    "add Acme Inc. and Beta Ltd. below the Contact heading",
    "add Acme Inc.", "write the next paragraph", "continue writing",
])
def test_anything_that_asks_for_writing_or_is_unclear_is_not_a_literal(task):
    assert literal_draft_text(task) is None


def test_prepare_context_returns_the_literal_with_its_placement_and_no_prompt():
    prepared = prepare_context(DocumentContext(text=DOC), "add Acme Inc. below the Contact heading", "draft")
    assert prepared.message == "Acme Inc." and prepared.prompt == ""
    meta = prepared.metadata
    assert (meta.status, meta.source, meta.coverage, meta.strategy) == ("ready", "literal", "none", "none")
    assert meta.placement.position == "after" and meta.placement.text == "## Contact\n"
    assert meta.billable_input_chars == 0 and meta.estimated_input_tokens == 0


def test_a_named_place_that_cannot_be_found_still_asks_instead_of_inserting():
    prepared = prepare_context(DocumentContext(text=DOC), "add Acme Inc. below the Risks heading", "draft")
    assert prepared.metadata.status == "needs_clarification" and prepared.metadata.source is None


def test_only_draft_uses_the_literal_path():
    prepared = prepare_context(DocumentContext(text=DOC), "add Acme Inc. below the Contact heading", "ask")
    assert prepared.message is None and prepared.metadata.source is None


def test_process_returns_the_literal_without_calling_a_model(runtime):
    client, state = runtime
    result = client.post("/agent/process", json=body(document_context={"text": DOC}, document_action="draft",
                                                    query="add Acme Inc. below the Contact heading")).json()
    assert result["markdown_response"] == "Acme Inc."
    assert result["document_context"]["source"] == "literal" and result["document_context"]["placement"]["position"] == "after"
    assert not state["calls"]


def test_stream_returns_the_literal_free_of_charge(runtime):
    client, state = runtime
    response = client.post("/agent/stream", json=body(document_context={"text": DOC}, document_action="draft",
                                                       query="add Acme Cloud below the Contact heading"))
    events = parse_sse_events(response.text)
    assert [e["event"] or e["data"].get("type") for e in events] == ["start", "document_context_ready", "content", "done"]
    assert events[2]["data"]["content"] == "Acme Cloud"
    assert not state["calls"] and not state["charges"] and not state["reserves"]


def test_a_task_that_asks_for_writing_still_reaches_the_model_with_the_exact_text_rule(runtime):
    client, state = runtime
    client.post("/agent/process", json=body(document_context={"text": DOC}, document_action="draft",
                                            query="add a short company bio below the Contact heading"))
    call, = state["calls"]
    prompt = "\n".join(call["instructions"])
    assert "SUPPLIED:" in prompt and "output exactly that text and nothing else" in prompt
    assert "WRITE: USER_TASK names a kind of text" in prompt


# ---- hardening found by the regression audit --------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", [
    "add Acme Inc. below the Contact heading in bold", "add Acme Inc. below the Contact heading twice",
    "add Acme Inc. below the Contact heading in capitals", "add Acme Inc. below the Contact heading, centred",
    "add 3 hashtags below the Pricing heading", "add 5 keywords below the About heading", "add 3 emojis below the title",
    'add "A" and "B" below the title', 'add "Built to last." "Since 1987." below the title',
    "add Key Takeaways below the title", "add Acme Cloud below paragraph 3", "add Acme Cloud below the Risks and Rewards, twice",
])
def test_more_to_do_after_the_place_or_a_count_of_things_is_not_a_literal(task):
    assert literal_draft_text(task) is None


@pytest.mark.parametrize("task", ["add Acme Cloud below Pricing", "add Acme Cloud below the Pricing", "add Acme Cloud below Contact.",
                                  "add Acme Cloud under the Pricing heading", "add Acme Inc. at the end of the document.", "add Acme Inc. at the end"])
def test_a_bare_heading_name_or_an_edge_is_a_complete_place(task):
    headings = ["Pricing", "Contact"]
    assert literal_draft_text(task, headings) in ("Acme Cloud", "Acme Inc.")


def test_a_bare_name_that_is_not_a_heading_of_this_document_is_not_a_place():
    assert literal_draft_text("add Acme Cloud below Nothing", ["Pricing"]) is None
    assert literal_draft_text("add Acme Cloud below Pricing in bold", ["Pricing"]) is None


def test_prepare_context_uses_the_documents_headings_for_a_bare_name():
    prepared = prepare_context(DocumentContext(text=DOC), "add Acme Cloud below Careers", "draft")
    assert (prepared.metadata.source, prepared.message) == ("literal", "Acme Cloud")
    assert prepare_context(DocumentContext(text=DOC), "add Acme Cloud below Careers in bold", "draft").metadata.source == "model"


@pytest.mark.parametrize("task", ["add" + " " * 8000 + "x", "add" + "\t" * 3000 + "x", "add " + "A " * 3000 + "below", "Add " + "Acme " * 1500,
                                  "add " + "(" * 4000, "add " + "1" * 7000, "please " * 1000 + "add x"])
def test_adversarial_requests_are_cheap(task):
    import time
    started = time.perf_counter()
    assert literal_draft_text(task) is None
    assert time.perf_counter() - started < 0.1


def test_a_long_request_is_never_a_literal():
    assert literal_draft_text("add Acme Inc. below the Contact heading" + " " * 400) is None


def test_draft_prompt_separates_supplied_text_from_text_to_write(runtime):
    client, state = runtime
    client.post("/agent/process", json=body(document_context={"text": DOC}, document_action="draft",
                                            query="add a note about Acme Cloud below the Contact heading"))
    prompt = "\n".join(state["calls"][0]["instructions"])
    assert "SUPPLIED:" in prompt and "WRITE:" in prompt
    assert "'Add a note about Acme Cloud below Pricing' asks for one or two real sentences ABOUT Acme Cloud, never just the name" in prompt


def test_source_is_set_only_when_a_draft_is_produced():
    ctx = DocumentContext(text=DOC)
    assert prepare_context(ctx, "What is this about?", "ask").metadata.source is None
    assert prepare_context(ctx, "Write the next paragraph", "draft").metadata.source == "model"
    assert prepare_context(ctx, "add Acme Inc. below the Contact heading", "draft").metadata.source == "literal"
    assert prepare_context(ctx, "add Acme Inc. below the Risks heading", "draft").metadata.source is None      # clarification
    assert prepare_context(DocumentContext(text=" "), "What is this about?", "ask").metadata.source is None    # needs_context


# ---- round 2 of the audit ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", [
    "add 5 stars in bold below the title", "add 5 Stars below the title", "add 3 Taglines below the title", "add 3 Slogans below the title",
    "add 3-5 ideas below the title", "add up to 5 slogans below the title", "add top 10 tips below the title", "add ~5 keywords below the title",
    "add 5+ ideas below the title", "add 3x variants below the title", "add 10 Downing Street below the title",
    "add Acme Inc. in capitals below the title", "add Acme Inc. twice below the title", "add Acme Inc. below the bold heading",
    "add Acme Inc. below the Contact heading, centred under the title page", "add Acme Inc. below the title\nand the quote",
    "add Related Posts below the title", "write Haiku below the title", "write Acme Inc. under the Contact heading",
    "write Coming Soon below the title",
])
def test_instructions_counts_and_kinds_of_text_are_not_literals(task):
    assert literal_draft_text(task) is None


@pytest.mark.parametrize("task, literal", [
    ("add 555-0100 below the Contact heading", "555-0100"), ("add 2024-25 below the title", "2024-25"), ("add 2024 below the title", "2024"),
    ("add v2.1 below the title", "v2.1"), ("add 3M next to the title", "3M"), ("put Q3 2026 beside the Pricing heading", "Q3 2026"),
    ("write $20 under the Contact heading", "$20"), ("write hello@acme.com under the Contact heading", "hello@acme.com"),
    ("type Acme Inc. below the title", "Acme Inc."), ("enter Acme Cloud below the title", "Acme Cloud"),
])
def test_figures_addresses_and_codes_are_literals_and_so_are_title_case_names_unless_the_verb_is_write(task, literal):
    assert literal_draft_text(task) == literal


def test_draft_prompt_applies_asked_formatting_to_supplied_text_and_flags_missing_facts(runtime):
    client, state = runtime
    client.post("/agent/process", json=body(document_context={"text": DOC}, document_action="draft",
                                            query="add a short bio for Priya Shah below the Contact heading"))
    prompt = "\n".join(state["calls"][0]["instructions"])
    assert "Use Markdown formatting only when USER_TASK asks for it" in prompt and "('in bold' -> **text**" in prompt
    assert "never inventing one" in prompt and "[bracketed placeholder] only for a specific detail the text cannot do without" in prompt
