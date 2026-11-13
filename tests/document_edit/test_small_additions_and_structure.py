"""Edit: small additions inside a line, and headings, titles and quotes. Look-alikes that must keep their old answer are pinned too."""
import json

import pytest

from src.document_edit.blocks import allowed_targets, document_blocks, fingerprint, quote_is_plain, requote, unquote
from src.document_edit.schema import DocumentPatch
from src.document_edit.service import INSERT_MESSAGE, SMALL_ADDITION_SYSTEM, apply_document_patches, edit_document, is_small_insertion
from tests.document_edit.test_service import ScriptedEditor

DOC = ("# Annual Report\n\n## Office\n\nOur office is at 12 Park Lane, Springfield.\nOpen 9 to 5.\n\n"
       "> Quality means doing it right\n> when no one is watching.\n\n## Pricing\n\nThe Team plan costs USD 20 in 2026.\n\n"
       "## Contact\n\nReach us during business hours.\n")


def block(starts_with, source=DOC):
    return next(b for b in document_blocks(source) if b.text.startswith(starts_with))


def never_called(*args, **kwargs):
    pytest.fail("No model call expected")


def run(instruction, target, reply, source=DOC, **extra):
    llm = ScriptedEditor(instruction, [target], {target.id: reply}, **extra)
    result, metadata = edit_document(source, instruction, llm)
    return result, metadata, llm


# ---- item 2: small additions inside an existing line ------------------------------------------------------------------------------

def test_a_year_next_to_the_title_is_an_edit_of_the_title():
    title = block("# Annual Report")
    result, metadata, llm = run("add (2024) next to the title", title, "# Annual Report (2024)\n")
    assert metadata.status == "proposed" and result == DOC.replace("# Annual Report\n", "# Annual Report (2024)\n", 1)
    assert [(p.before, p.after) for p in metadata.patches] == [("# Annual Report\n", "# Annual Report (2024)\n")]
    # The block writer is told it is a small addition, the stored patch can be applied later without the instruction.
    block_call, = [c for c in llm.calls if c[0].stage == "document_block"]
    assert json.loads(block_call[1][0]["content"])["TARGET_KIND"] == "heading"
    assert apply_document_patches(DOC, metadata.base_sha256, metadata.patches) == result


def test_the_block_writer_gets_the_small_addition_rules_only_for_small_additions():
    def block_system(instruction, target, reply):
        systems, scripted = [], ScriptedEditor(instruction, [target], {target.id: reply})
        edit_document(DOC, instruction, lambda system, messages, spec: (systems.append((spec.stage, system)), scripted(system, messages, spec))[1])
        return next(system for stage, system in systems if stage == "document_block")
    title, office = block("# Annual Report"), block("Our office")
    assert block_system("add (2024) next to the title", title, "# Annual Report (2024)\n").endswith(SMALL_ADDITION_SYSTEM)
    assert SMALL_ADDITION_SYSTEM not in block_system("make the Office section more concise", office, "Our office is at 12 Park Lane.\nOpen 9 to 5.\n")


def test_a_named_kind_of_detail_without_a_value_becomes_a_bracketed_placeholder():
    office = block("Our office")
    reply = "Our office is at 12 Park Lane, Springfield. [Phone number]\nOpen 9 to 5.\n"
    result, metadata, _ = run("add a phone number after the address", office, reply)
    assert metadata.status == "proposed" and "[Phone number]" in result
    assert result.replace(" [Phone number]", "") == DOC       # nothing else moved


def test_a_value_the_user_typed_is_inserted_as_written():
    office = block("Our office")
    reply = "Our office is at 12 Park Lane, Springfield, tel. 555-0100.\nOpen 9 to 5.\n"
    result, metadata, _ = run("add 555-0100 after the address", office, reply)
    assert metadata.status == "proposed" and "555-0100" in result


@pytest.mark.parametrize("instruction, reply", [
    ("add a phone number after the address", "Our office is at 12 Park Lane, Springfield, tel. 555-0100.\nOpen 9 to 5.\n"),   # invented value
    ("add (2024) next to the title", "# Annual Report (2025)\n"),                       # a number the user did not type
    ("add (2024) next to the title", "# Annual Report (2024) 2024\n"),                 # the typed number twice
])
def test_a_number_the_user_did_not_type_is_never_accepted(instruction, reply):
    target = block("# Annual Report") if "title" in instruction else block("Our office")
    result, metadata, llm = run(instruction, target, reply)
    assert metadata.status == "failed" and not metadata.patches and result == DOC
    assert [c[0].stage for c in llm.calls].count("document_repair") == 1


def test_original_numbers_must_survive_a_small_addition():
    price = block("The Team plan")
    result, metadata, _ = run("add (2024) after the price", price, "The Team plan costs USD 25 in 2026. (2024)\n")
    assert metadata.status == "failed" and result == DOC


def test_a_small_addition_cannot_open_a_new_line_or_block():
    office = block("Our office")
    result, metadata, _ = run("add a phone number after the address", office,
                              "Our office is at 12 Park Lane, Springfield.\n[Phone number]\nOpen 9 to 5.\n")
    assert metadata.status == "failed" and result == DOC


@pytest.mark.parametrize("instruction", [
    "add a short FAQ below the Pricing heading", "add a phone number below the address", "add (2024) after the table",
    "add a new section after the intro", "add a sentence after the title", "add a note next to the Top 10 list",
    "add a caption beside the image", "insert a table after the intro", "add a paragraph after the address",
    "add a phone number after the Contact section",
])
def test_new_blocks_still_go_to_draft_without_a_model_call(instruction):
    result, metadata = edit_document(DOC, instruction, never_called)
    assert (metadata.status, metadata.message) == ("needs_clarification", INSERT_MESSAGE) and result == DOC


@pytest.mark.parametrize("instruction", ["add (2024) next to the title", "add a phone number after the address",
                                         "add 2024 at the end of the title"])
def test_small_additions_are_not_refused_before_the_planner(instruction):
    seen = []
    edit_document(DOC, instruction, lambda system, messages, spec: seen.append(spec.stage) or json.dumps(
        {"status": "needs_clarification", "message": "Which line?"}))
    assert seen == ["document_plan"]


def test_apply_accepts_a_short_insertion_with_a_number_but_not_a_reworked_text_with_one():
    price = block("The Team plan")

    def patch(target, after):
        return DocumentPatch(block_id=target.id, start_utf16=target.start, end_utf16=target.end, before=target.text, after=after, reason="test")
    ok = apply_document_patches(DOC, fingerprint(DOC), [patch(price, "The Team plan costs USD 20 in 2026 (2024).\n")])
    assert "(2024)" in ok
    for after in ["The Team plan costs USD 20 in 2026 (2024), but only for teams that sign up before the end of the first quarter.\n",
                  "The Team plan costs USD 25 in 2026.\n", "The plan costs USD 20 in 2026 (2024).\n", "USD 999\n"]:
        with pytest.raises(ValueError):
            apply_document_patches(DOC, fingerprint(DOC), [patch(price, after)])


def test_is_small_insertion_is_a_pure_short_insertion():
    assert is_small_insertion("Annual Report\n", "Annual Report (2024)\n")
    assert is_small_insertion("Annual Report\n", "(2024) Annual Report\n")
    assert not is_small_insertion("Annual Report\n", "Annuel Reports\n")          # a changed letter is not a pure insertion
    assert not is_small_insertion("Annual Report\n", "Annual Report\n")
    assert not is_small_insertion("Annual Report\n", "Annual\n")
    assert not is_small_insertion("Annual Report\n", "Annual Report " + "x" * 81 + "\n")


# ---- item 3: headings, titles and quotes ------------------------------------------------------------------------------------------

def test_headings_and_quotes_are_rewritable_but_still_not_prose():
    title, quote, para = block("# Annual Report"), block("> Quality"), block("Our office")
    assert (title.editable, title.rewritable, quote.editable, quote.rewritable, para.editable, para.rewritable) == (False, True, False, True, True, True)
    table = document_blocks("| a | b |\n|---|---|\n| 1 | 2 |\n")[0]
    assert not table.rewritable


def test_a_title_can_be_shortened():
    title = block("# Annual Report")
    result, metadata, _ = run("make the title shorter", title, "# Report\n")
    assert metadata.status == "proposed" and result == DOC.replace("# Annual Report", "# Report", 1)
    assert metadata.patches[0].before == "# Annual Report\n"


def test_a_heading_named_in_the_request_can_be_renamed():
    contact = block("## Contact")
    result, metadata, _ = run("rename the Contact heading to Get in touch", contact, "## Get in touch\n")
    assert metadata.status == "proposed" and "## Get in touch\n" in result and "Reach us" in result


def test_a_heading_keeps_its_level():
    contact = block("## Contact")
    result, metadata, _ = run("rename the Contact heading to Get in touch", contact, "# Get in touch\n")
    assert metadata.status == "failed" and result == DOC


@pytest.mark.parametrize("instruction", ["make the Office section more concise", "make the introduction shorter", "fix the typos",
                                         "tighten the paragraphs, but leave the headings unchanged", "shorten Pricing without changing the title",
                                         "improve the text throughout"])
def test_a_general_request_never_offers_headings_or_quotes(instruction):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    kinds = {b.kind for b in document_blocks(DOC) if b.id in allowed}
    assert kinds <= {"paragraph", "list"}, kinds


@pytest.mark.parametrize("instruction, kinds", [
    ("make the title shorter", {"heading", "paragraph"}), ("shorten all the headings", {"heading", "paragraph"}),
    ("rename the Contact heading to Get in touch", {"heading", "paragraph"}), ("make the quote shorter", {"blockquote", "paragraph"}),
    ("fix the typo in the quote", {"blockquote", "paragraph"}),
])
def test_headings_and_quotes_are_offered_only_when_the_request_is_about_them(instruction, kinds):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    assert {b.kind for b in document_blocks(DOC) if b.id in allowed} == kinds


def test_the_title_request_offers_only_the_top_level_heading():
    allowed, _ = allowed_targets("make the title shorter", document_blocks(DOC))
    assert [b.text for b in document_blocks(DOC) if b.id in allowed and b.kind == "heading"] == ["# Annual Report\n"]


def test_a_heading_the_model_picks_for_a_general_request_is_rejected():
    title = block("# Annual Report")
    llm = ScriptedEditor("make the Office section more concise", [title], {title.id: "# Report\n"})
    result, metadata = edit_document(DOC, "make the Office section more concise", llm)
    assert metadata.status == "needs_clarification" and result == DOC and not [c for c in llm.calls if c[0].stage == "document_block"]


def test_a_quote_is_edited_as_text_and_keeps_its_markers():
    quote = block("> Quality")
    result, metadata, llm = run("make the quote shorter", quote, "Quality is doing it right\nunseen.\n")
    assert metadata.status == "proposed"
    assert metadata.patches[0].after == "> Quality is doing it right\n> unseen.\n"
    assert result == DOC.replace(quote.text, "> Quality is doing it right\n> unseen.\n")
    block_call, = [c for c in llm.calls if c[0].stage == "document_block"]
    data = json.loads(block_call[1][0]["content"])
    assert data["TARGET_KIND"] == "quote" and ">" not in data["TARGET"]
    assert apply_document_patches(DOC, metadata.base_sha256, metadata.patches) == result


def test_a_quote_the_model_returns_with_its_markers_is_accepted_too():
    quote = block("> Quality")
    result, metadata, _ = run("make the quote shorter", quote, "> Quality is doing it right\n> unseen.\n")
    assert metadata.status == "proposed" and metadata.patches[0].after == "> Quality is doing it right\n> unseen.\n"


def test_a_quote_cannot_gain_or_lose_a_line():
    quote = block("> Quality")
    result, metadata, _ = run("make the quote shorter", quote, "Quality is doing it right unseen.\n")
    assert metadata.status == "failed" and result == DOC


def test_apply_rejects_a_quote_patch_that_drops_a_marker():
    quote = block("> Quality")
    patch = DocumentPatch(block_id=quote.id, start_utf16=quote.start, end_utf16=quote.end, before=quote.text,
                          after="Quality is doing it right\n> unseen.\n", reason="forged")
    with pytest.raises(ValueError):
        apply_document_patches(DOC, fingerprint(DOC), [patch])


def test_quote_helpers_round_trip_and_reject_nested_or_mixed_quotes():
    text = "> one\n>two\n>\n> three\n"
    inner, prefixes = unquote(text)
    assert inner == "one\ntwo\n\nthree\n" and requote(inner, prefixes) == text
    assert requote("one\n", prefixes) is None
    assert quote_is_plain("> plain\n> text\n")
    assert not quote_is_plain("> > nested\n")
    assert not quote_is_plain("> ```\n> code\n> ```\n")
    assert not quote_is_plain("> | a | b |\n> |---|---|\n> | 1 | 2 |\n")


def test_requests_that_do_not_mention_quotes_leave_the_documents_quote_protected():
    # The same quote the older tests rely on stays untouchable by an ordinary rewrite.
    doc = "## Intro\n\nThis is very very useful.\n\n> Keep this exact quote.\n"
    allowed, _ = allowed_targets("Improve Intro", document_blocks(doc))
    assert {b.kind for b in document_blocks(doc) if b.id in allowed} == {"paragraph"}


# ---- fixes from the regression audit ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("instruction", [
    "Improve clarity. Leave the title alone.", "Make the text shorter, leaving the headings alone.", "Shorten the intro but keep the title.",
    "Shorten the text; keep the quote as it is", "Shorten the paragraphs; keep headings as they are.", "Rewrite the text. Don't rewrite the title.",
    "Tighten the paragraphs and skip the quote", "Shorten the text, ignoring the headings", "Rewrite the intro. Do not touch the title.",
    "Tighten the paragraphs, preserving the headings", "Shorten everything but not the title", "Make it clearer and avoid changing the quote",
])
def test_any_keep_or_leave_phrasing_keeps_headings_and_quotes_out(instruction):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    assert {b.kind for b in document_blocks(DOC) if b.id in allowed} <= {"paragraph", "list"}


@pytest.mark.parametrize("instruction, kind", [
    ("keep the quote but make the title shorter", "heading"), ("Leave the headings alone and shorten the quote", "blockquote"),
    ("Retitle the document", "heading"), ("Don't change the intro; rename the Contact heading", "heading"),
])
def test_what_is_asked_after_a_keep_clause_is_still_offered(instruction, kind):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    assert kind in {b.kind for b in document_blocks(DOC) if b.id in allowed}


def test_retitle_the_document_offers_only_the_top_level_heading():
    allowed, _ = allowed_targets("Retitle the document", document_blocks(DOC))
    assert [b.text for b in document_blocks(DOC) if b.id in allowed and b.kind == "heading"] == ["# Annual Report\n"]


@pytest.mark.parametrize("reply", ["##\n", "## \n", "## ##\n"])
def test_a_heading_cannot_be_emptied(reply):
    contact = block("## Contact")
    result, metadata, _ = run("rename the Contact heading to Get in touch", contact, reply)
    assert metadata.status == "failed" and result == DOC
    patch = DocumentPatch(block_id=contact.id, start_utf16=contact.start, end_utf16=contact.end, before=contact.text, after=reply, reason="x")
    with pytest.raises(ValueError):
        apply_document_patches(DOC, fingerprint(DOC), [patch])


def test_a_small_addition_must_leave_every_other_word_alone():
    office = block("Our office")
    # the item is there, but the line was reworded around it
    reworded = "The office can be found at 12 Park Lane, Springfield. [Phone number]\nOpen 9 to 5.\n"
    result, metadata, llm = run("add a phone number after the address", office, reworded)
    assert metadata.status == "failed" and result == DOC
    assert [c[0].stage for c in llm.calls].count("document_repair") == 1


def test_a_small_addition_may_move_punctuation_around_the_item():
    office = block("Our office")
    result, metadata, _ = run("add a phone number after the address", office,
                              "Our office is at 12 Park Lane, Springfield, [Phone number].\nOpen 9 to 5.\n")
    assert metadata.status == "proposed"


def test_an_ordinary_rewrite_is_not_held_to_the_small_addition_rule():
    office = block("Our office")
    result, metadata, _ = run("make the Office section more concise", office, "We are at 12 Park Lane, Springfield.\nOpen 9 to 5.\n")
    assert metadata.status == "proposed"


# ---- round 2 of the audit ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("instruction", [
    "Leave the title and the quote alone", "Leave the title, the headings and the quote alone", "Improve the text except for the headings and the quote",
    "Shorten the intro but keep the title and the quote", "Tighten the text without touching the title or the quote", "Do not rewrite the title",
])
def test_a_keep_clause_covers_everything_it_lists(instruction):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    assert {b.kind for b in document_blocks(DOC) if b.id in allowed} <= {"paragraph", "list"}


@pytest.mark.parametrize("instruction, kind", [
    ("Leave the title alone and shorten the quote", "blockquote"), ("Don't touch the quote, rename the Contact heading", "heading"),
    ("Don't rewrite the title; shorten the quote", "blockquote"), ("Keep the headings; rewrite the title", "heading"),
])
def test_a_request_after_a_keep_clause_is_still_asked_for(instruction, kind):
    allowed, _ = allowed_targets(instruction, document_blocks(DOC))
    assert kind in {b.kind for b in document_blocks(DOC) if b.id in allowed}


def test_a_title_that_is_kept_by_name_is_not_offered_when_the_other_headings_are_asked_for():
    allowed, _ = allowed_targets("Shorten the headings but keep the title", document_blocks(DOC))
    offered = [b.text for b in document_blocks(DOC) if b.id in allowed and b.kind == "heading"]
    assert offered and "# Annual Report\n" not in offered


def test_a_rewrite_with_a_small_addition_may_change_words_and_uses_the_typed_number():
    instruction = "Make the text shorter and add (2024) next to the title"
    title, office = block("# Annual Report"), block("Our office")
    scripted = ScriptedEditor(instruction, [title, office], {title.id: "# Annual Report (2024)\n",
                                                             office.id: "Our office is at 12 Park Lane.\nOpen 9 to 5.\n"})
    systems = []
    result, metadata = edit_document(DOC, instruction, lambda s, m, spec: (systems.append((spec.stage, s)), scripted(s, m, spec))[1])
    assert metadata.status == "proposed" and {p.after for p in metadata.patches} == {"# Annual Report (2024)\n", "Our office is at 12 Park Lane.\nOpen 9 to 5.\n"}
    block_systems = [s for stage, s in systems if stage == "document_block"]
    assert block_systems and all(SMALL_ADDITION_SYSTEM not in s and "Part of this request is a small in-line addition" in s for s in block_systems)


def test_a_rewrite_with_a_small_addition_still_refuses_a_number_nobody_typed():
    instruction = "Make the text shorter and add (2024) next to the title"
    office = block("Our office")
    result, metadata, _ = run(instruction, office, "Our office is at 12 Park Lane, since 1999.\nOpen 9 to 5.\n")
    assert metadata.status == "failed" and result == DOC


def test_the_setext_level_survives_odd_line_separators():
    from src.document_edit.blocks import _level
    title = document_blocks("Annual\x0bReport\n=====\n\ntext\n")[0]
    assert title.kind == "heading" and _level(title) == 1
