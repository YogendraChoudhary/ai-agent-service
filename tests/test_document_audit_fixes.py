"""
Regression tests for the five document defects of the 2026-11-20 audit (each reproduced before the fix):

1. blank lines counted toward the 400-block limit (Edit 400, /apply 422), and Draft placement silently fell back to the cursor;
2. DOCX/HTML import flattened superscripts and subscripts (10^6 became "106");
3. a paragraph or list over 12,000 characters could not be edited at all, with a misleading message;
4. a literal number change the user typed ("USD 20" to "USD 25") failed as "changed protected content";
5. Ask refused "In section 2, what is the rent?" when "Section 2." is a paragraph, not a heading.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.document_context import DocumentContext, prepare_context
from src.document_edit.blocks import MAX_BLOCK_CHARS, document_blocks, edit_units
from src.document_edit.router import router as apply_router
from src.document_edit.schema import DocumentPatch
from src.document_edit.service import (IN_PLACE_INVALID_MESSAGE, apply_document_patches, edit_document, is_short_substitution)
from src.document_import.docx_conv import docx_to_markdown
from src.document_import.html_to_md import html_to_markdown
from src.document_placement import UNPLACEABLE_MESSAGE, resolve_placement
from tests.document_import.builders import make_docx, p, run


def no_model(*args):
    raise AssertionError("no model call expected")


@pytest.fixture(scope="module")
def apply_client():
    app = FastAPI()
    app.include_router(apply_router)
    return TestClient(app)


def apply_over_http(client, source, proposal):
    return client.post("/agent/documents/apply", json={
        "document_text": source, "base_sha256": proposal.base_sha256, "patches": [p.model_dump() for p in proposal.patches]})


# ---- 1. blank lines and the block limit ---------------------------------------------------------------------------------------------

def handbook(sections: int) -> str:
    return ("# Handbook\n\n" + "".join(f"## Section {i}\n\nPara one {i}.\n\nPara two.\n\nPara three.\n\nPara four.\n\n"
                                       for i in range(sections)) + "## Pricing\n\nOur price is fair.\n")


def test_blank_lines_do_not_count_toward_the_block_limit(apply_client):
    source = handbook(41)                                     # the audit's 2.6 KB document: 402 blocks, about half blank lines
    blocks = document_blocks(source)
    assert len(blocks) > 400 and sum(b.kind != "blank" for b in blocks) < 400
    text, proposal = edit_document(source, 'Change "Our price is fair." to "Our price is low."', no_model)
    assert proposal.status == "proposed" and text == source.replace("is fair.", "is low.")
    response = apply_over_http(apply_client, source, proposal)
    assert response.status_code == 200 and response.json()["document_text"] == text


def test_over_400_real_blocks_is_still_refused():
    many = "\n\n".join(f"Paragraph {i}." for i in range(401)) + "\n"
    with pytest.raises(ValueError, match="too many blocks"):
        document_blocks(many)
    assert len(document_blocks(many, max_blocks=None)) > 800


def test_draft_placement_resolves_in_documents_over_the_edit_block_limit():
    for source in (handbook(41), handbook(120)):              # 120 sections: over 400 real blocks, too many for Edit
        placement, question = resolve_placement(source, "Add a paragraph about discounts below the Pricing heading")
        assert question is None and placement.text == "## Pricing\n"
        prepared = prepare_context(DocumentContext(text=source, cursor_utf16=0),
                                   "Add a paragraph about discounts below the Pricing heading", "draft")
        assert prepared.metadata.placement is not None and prepared.metadata.placement.text == "## Pricing\n"


def test_unparseable_document_asks_where_to_put_a_named_place_instead_of_using_the_cursor():
    html = "<p>Intro</p>\n<div>Pricing</div>\n"                 # a document the block parser refuses
    assert resolve_placement(html, "Add a paragraph below the Pricing heading") == (None, UNPLACEABLE_MESSAGE)
    assert resolve_placement(html, "Add a paragraph at the end of the document") == (None, UNPLACEABLE_MESSAGE)
    assert resolve_placement(html, "Write a short closing paragraph") == (None, None)


# ---- 2. superscripts and subscripts on import ---------------------------------------------------------------------------------------

def test_html_import_keeps_superscripts_and_subscripts():
    converted = html_to_markdown("<p>10<sup>6</sup> cells, CO<sub>2</sub> and 5 m<sup>2</sup>; x<sup>n+1</sup>.</p>")
    assert converted.markdown == "10<sup>6</sup> cells, CO<sub>2</sub> and 5 m<sup>2</sup>; x<sup>n+1</sup>.\n"
    assert converted.stats["superscripts"] == 3 and converted.stats["subscripts"] == 1
    assert [w["code"] for w in converted.warnings] == ["superscript_subscript_html"] and "4 " in converted.warnings[0]["message"]


def test_superscript_text_is_escaped_and_code_and_empty_runs_stay_plain():
    md = html_to_markdown("<p>a<sup>*b*&lt;i&gt;</sup> <code>x<sup>2</sup></code> c<sup> </sup>d</p>").markdown
    assert md == "a<sup>\\*b\\*\\<i\\></sup> `x2` c d\n"


def test_docx_import_keeps_superscripts_and_subscripts():
    subscript = "<w:r><w:rPr><w:vertAlign w:val='subscript'/></w:rPr><w:t>2</w:t></w:r>"
    data = make_docx([p([run("The sample held 10"), run("6", sup=True), run(" cells and CO"), subscript, run(".")])])
    converted = docx_to_markdown(data)
    assert "10<sup>6</sup> cells and CO<sub>2</sub>." in converted.markdown


def test_imported_superscripts_are_accepted_and_protected_by_the_document_pipeline():
    source = html_to_markdown("<h1>Lab</h1><p>We grew 10<sup>6</sup> cells in CO<sub>2</sub>.</p>").markdown
    DocumentContext(text=source)                              # not taken for HTML
    paragraph = next(b for b in document_blocks(source) if b.kind == "paragraph")

    def model(system, messages, spec):
        data = json.loads(messages[0]["content"])
        if spec.stage == "document_plan":
            return json.dumps({"status": "ready", "message": "Tighten.", "targets": [
                {"block_id": paragraph.id, "request_quote": "Tighten the text", "reason": "Wordy."}]})
        assert "<sup>" not in data["TARGET"] and "[[HTML_" in data["TARGET"]        # the tags travel as locked tokens
        return data["TARGET"].replace("We grew", "We cultured")
    text, proposal = edit_document(source, "Tighten the text", model)
    assert proposal.status == "proposed" and "We cultured 10<sup>6</sup> cells in CO<sub>2</sub>." in text


# ---- 3. paragraphs and lists over 12,000 characters ---------------------------------------------------------------------------------

def meeting_notes(items: int = 160) -> str:
    return "# Meeting notes\n\n" + "".join(
        f"- Item {i}: the team have discussed the roadmap and agreed to followup on budget items next week {i}.\n" for i in range(items))


class Proofreader:
    """Plans every allowed block and fixes one grammar mistake in each, as a model would."""

    def __init__(self):
        self.targets = []

    def __call__(self, system, messages, spec):
        data = json.loads(messages[0]["content"])
        if spec.stage == "document_plan":
            return json.dumps({"status": "ready", "message": "Fixing the grammar.", "targets": [
                {"block_id": b, "request_quote": "Fix the grammar in the notes", "reason": "Grammar."} for b in data["allowed_block_ids"]]})
        assert len(data["TARGET"]) <= MAX_BLOCK_CHARS
        self.targets.append(data["TARGET"])
        return data["TARGET"].replace(" have discussed", " discussed")


def test_a_long_list_is_rewritten_as_runs_of_items(apply_client):
    source = meeting_notes()                                   # ~16 KB, one 160-item list
    model = Proofreader()
    text, proposal = edit_document(source, "Fix the grammar in the notes", model)
    assert proposal.status == "proposed" and text == source.replace(" have discussed", " discussed")
    blocks = document_blocks(source)
    big = next(b for b in blocks if b.kind == "list")
    assert len(big.text) > MAX_BLOCK_CHARS and len(model.targets) == len(proposal.patches) > 1
    assert {p.block_id for p in proposal.patches} == {big.id} and [t.block_id for t in proposal.targets] == [big.id]
    assert all(t.startswith("- Item") for t in model.targets)                    # each run starts at an item
    assert apply_document_patches(source, proposal.base_sha256, proposal.patches) == text
    response = apply_over_http(apply_client, source, proposal)
    assert response.status_code == 200 and response.json()["document_text"] == text


def test_a_plan_that_lists_a_run_twice_is_still_proposed():
    # Live 2026-11-20 (OpenAI): two mistakes in one run of items made the planner list that run twice, and the plan was refused.
    source = meeting_notes()

    def model(system, messages, spec):
        data = json.loads(messages[0]["content"])
        if spec.stage == "document_plan":
            first = data["allowed_block_ids"][0]
            return json.dumps({"status": "ready", "message": "Fixing the grammar.", "targets": [
                {"block_id": first, "request_quote": "Fix the grammar in the notes", "reason": reason} for reason in ("One.", "Two.")]})
        return data["TARGET"].replace(" have discussed", " discussed")
    text, proposal = edit_document(source, "Fix the grammar in the notes", model)
    assert proposal.status == "proposed" and len(proposal.patches) == 1 and proposal.targets[0].reason == "One."


def test_runs_cover_every_item_and_leave_blank_lines_between_them_alone():
    loose = "".join(f"{i + 1}. Item {i} " + "word " * 20 + "\n\n" for i in range(200)) + "End.\n"
    blocks = document_blocks(loose)
    runs = [u for u in edit_units(blocks) if u.parent]
    assert len(runs) > 2 and all(len(r.text) <= 4_000 and r.text.endswith(" \n") for r in runs)
    assert all(loose[r.end:r.end + 1] == "\n" for r in runs)                        # the blank line after a run is not in it
    covered = "".join(loose[r.start:r.end] for r in runs)
    assert covered.count("Item ") == 200


def test_a_long_list_run_may_not_change_numbers():
    source = meeting_notes()

    def model(system, messages, spec):
        data = json.loads(messages[0]["content"])
        if spec.stage == "document_plan":
            return Proofreader()(system, messages, spec)
        return data["TARGET"].replace("next week 7.", "next week 8.")
    text, proposal = edit_document(source, "Fix the grammar in the notes", model)
    assert proposal.status == "failed" and text == source


def test_literal_and_selection_edits_work_inside_a_long_list():
    source = meeting_notes()
    old = "agreed to followup on budget items next week 7."
    text, proposal = edit_document(source, f'Change "{old}" to "agreed to follow up on budget items next week 7."', no_model)
    assert proposal.status == "proposed" and text == source.replace(old, "agreed to follow up on budget items next week 7.")
    assert len(proposal.patches) == 1 and proposal.patches[0].before == old

    start = source.index("have discussed", source.index("Item 9:"))
    selection = {"start_utf16": start, "end_utf16": start + len("have")}
    from src.document_context import ContextRange

    def editor(system, messages, spec):
        assert json.loads(messages[0]["content"])["TARGET"] == "have"
        return json.dumps({"replacement": "has"})
    text, proposal = edit_document(source, "Fix the grammar", editor, selection=ContextRange(**selection))
    assert proposal.status == "proposed" and text == source[:start] + "has" + source[start + len("have"):]


def test_replace_everywhere_reaches_into_a_long_list():
    source = meeting_notes()
    text, proposal = edit_document(source, "Replace roadmap with plan everywhere", no_model)
    assert proposal.status == "proposed" and "roadmap" not in text and text.count(" plan ") == 160
    assert len(proposal.patches) > 1 and all(len(p.before) <= MAX_BLOCK_CHARS for p in proposal.patches)
    assert apply_document_patches(source, proposal.base_sha256, proposal.patches) == text


def test_overlapping_patches_into_a_long_list_are_rejected():
    source = meeting_notes()
    _, proposal = edit_document(source, "Replace roadmap with plan everywhere", no_model)
    first = proposal.patches[0]
    with pytest.raises(ValueError, match="Duplicate"):
        apply_document_patches(source, proposal.base_sha256, [first, first])


def test_too_long_paragraph_gets_an_accurate_message():
    source = "# Notes\n\n" + "This sentence is part of one very long paragraph. " * 260 + "\n"
    text, proposal = edit_document(source, "Fix the grammar in the notes", no_model)
    assert proposal.status == "needs_clarification" and text == source
    assert proposal.message.startswith("This paragraph is longer than 12,000 characters") and "Change X to Y" in proposal.message
    single_item = "- " + "word " * 3000 + "\n"
    _, proposal = edit_document(single_item, "Fix the grammar", no_model)
    assert proposal.message.startswith("This list is longer than 12,000 characters")


# ---- 4. literal number changes the user typed ---------------------------------------------------------------------------------------

PLAN = "# Plan 2025\n\nThe plan costs USD 20 per month and launches in 2025.\n\nCall 555-0100 for help.\n"


@pytest.mark.parametrize("instruction, old, new", [
    ('Change "USD 20" to "USD 25"', "USD 20", "USD 25"),
    ('Replace "555-0100" with "555-0199"', "555-0100", "555-0199"),
    ('Change "launches in 2025" to "launches in 2026"', "launches in 2025", "launches in 2026"),
])
def test_a_literal_number_change_is_proposed_and_applies(apply_client, instruction, old, new):
    text, proposal = edit_document(PLAN, instruction, no_model)
    assert proposal.status == "proposed" and text == PLAN.replace(old, new)
    assert apply_document_patches(PLAN, proposal.base_sha256, proposal.patches) == text
    response = apply_over_http(apply_client, PLAN, proposal)
    assert response.status_code == 200 and response.json()["document_text"] == text


def test_a_model_selection_edit_still_may_not_change_numbers():
    start = PLAN.index("launches in 2025")
    from src.document_context import ContextRange
    selection = ContextRange(start_utf16=start, end_utf16=start + len("launches in 2025"))
    text, proposal = edit_document(PLAN, "Fix the typos", lambda *a: json.dumps({"replacement": "launches in 2026"}),
                                   selection=selection)
    assert proposal.status == "failed" and proposal.message == IN_PLACE_INVALID_MESSAGE and text == PLAN


def test_apply_still_rejects_a_whole_block_patch_that_changes_a_number():
    block = next(b for b in document_blocks(PLAN) if b.text.startswith("The plan"))
    from src.document_edit.blocks import fingerprint, utf16_length
    patch = DocumentPatch(block_id=block.id, start_utf16=utf16_length(PLAN[:block.start]), end_utf16=utf16_length(PLAN[:block.end]),
                          before=block.text, after=block.text.replace("USD 20", "USD 25"), reason="forged")
    with pytest.raises(ValueError):
        apply_document_patches(PLAN, fingerprint(PLAN), [patch])


def test_short_substitution_shape():
    block = "The plan costs USD 20 per month.\n"
    start = block.index("USD 20")
    assert is_short_substitution(block, start, start + 6, "USD 20", "USD 25")
    assert not is_short_substitution(block, 0, len(block), block, block.replace("20", "25"))            # the whole block
    assert not is_short_substitution(block, start, start + 6, "USD 20", "USD 25\nmore")                # a new line
    assert not is_short_substitution(block * 20, 0, 300 + start, (block * 20)[:300 + start], "x")      # too long


# ---- 5. Ask about a "section" that is not a heading ---------------------------------------------------------------------------------

CONTRACT = "# Contract\n\nSection 1. Parties: Ann and Bob.\n\nSection 2. Rent is USD 900.\n\nSection 3. Deposit is USD 1,800.\n"
LEASE_LIST = "# Lease\n\n1. Parties. The landlord is Ann Lee.\n2. Rent. Rent is USD 900 per month.\n3. Deposit. USD 1,800.\n"


@pytest.mark.parametrize("source", [CONTRACT, LEASE_LIST])
@pytest.mark.parametrize("question", ["In section 2, what is the rent?", "What is in Appendix A?", "What does chapter 1 cover?",
                                      "What does part b say about the deposit?"])
def test_a_named_section_that_is_no_heading_is_answered_from_the_whole_small_document(source, question):
    prepared = prepare_context(DocumentContext(text=source), question, "ask")
    assert prepared.metadata.status == "ready" and prepared.metadata.coverage == "full" and prepared.metadata.strategy == "full"
    assert "USD 900" in prepared.prompt


def test_a_named_heading_still_uses_its_section_and_auto_scope_also_falls_back():
    source = "# Lease\n\n## Section 1\n\nParties.\n\n## Section 2\n\nRent is USD 900.\n"
    prepared = prepare_context(DocumentContext(text=source), "In section 2, what is the rent?", "ask")
    assert prepared.metadata.strategy == "retrieval" and "Parties." not in prepared.prompt
    prepared = prepare_context(DocumentContext(text=CONTRACT), "In section 2, what is the rent?", "ask", scope="auto")
    assert prepared.metadata.status == "ready" and prepared.metadata.coverage == "full"


def test_headings_that_use_the_label_keep_the_refusal_for_a_missing_number():
    # "## Section 40" headings: the document numbers its sections with headings, so "section 4" is one that is not there.
    prepared = prepare_context(DocumentContext(text="## Section 40: Policy\n\nOther policy.\n"), "Summarize section 4", "ask")
    assert prepared.metadata.status == "needs_context" and "No heading matches" in prepared.message
    # "part b" is not a "Parties" heading.
    prepared = prepare_context(DocumentContext(text="## Parties\n\nAnn and Bob.\n\nPart B. Deposit USD 1,800.\n"),
                               "What does part b say about the deposit?", "ask")
    assert prepared.metadata.status == "ready" and prepared.metadata.coverage == "full"


def test_a_document_too_large_for_full_context_keeps_the_refusal():
    source = "# Contract\n\n" + "".join(f"Clause {i}. The tenant keeps the garden tidy and the hallway clear.\n\n" for i in range(400))
    for scope in ("document", "auto"):
        prepared = prepare_context(DocumentContext(text=source), "In section 2, what is the rent?", "ask", scope=scope)
        assert prepared.metadata.status == "needs_context" and "No heading matches" in prepared.message
