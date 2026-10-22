"""
One part of the post: code writes the heading and places the picture, the model writes the body, the lint decides
whether it needs one repair. Partly wrong is accepted with a warning (D14); empty is an error.
"""

import json
import unittest
from unittest.mock import patch

from config import settings
from src.blog import schema as s
from src.blog.draft import (
    CLOSING_ANGLE, WRITER_LENGTH_FACTOR, Unit, assemble, image_block, unit_max_tokens, unit_request, units_from_outline,
    write_unit,
)
from src.blog.media import Placed
from src.blog.plan import BlogRequest, plan_blog
from src.blog.prompts import load_example, section_system_prompt
from src.blog.sources import WebSources
from tests.blog.fakes import ScriptedLLM, filler, good_body, with_quote

BRIEF = "Write a blog post for team leads about running remote meetings that people actually want to attend."
REQUEST = BlogRequest.from_options(BRIEF, "Friendly, no jargon", {"audience": "team leads", "tone": "warm"})
PLAN = plan_blog(REQUEST)
OUTLINE = s.Outline.model_validate(load_example("standard")["output"])
PLACED = {"s1": Placed("pid-1"), "s3": Placed("pid-3"), "s4": Placed(None, "insufficient credits")}
TABLE = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"


def units():
    return {u.id: u for u in units_from_outline(OUTLINE, PLACED)}


class TestUnits(unittest.TestCase):

    def test_intro_sections_conclusion_in_order(self):
        us = units_from_outline(OUTLINE, PLACED)
        self.assertEqual([u.id for u in us], ["intro", "s1", "s2", "s3", "s4", "s5", "conclusion"])
        self.assertEqual([u.kind for u in us], ["intro"] + ["section"] * 5 + ["conclusion"])
        self.assertIsNone(us[0].heading)
        self.assertEqual(us[-1].angle, CLOSING_ANGLE)

    def test_the_picture_belongs_to_the_part_only_if_it_was_started(self):
        u = units()
        self.assertEqual((u["s1"].placeholder, u["s3"].placeholder), ("pid-1", "pid-3"))
        self.assertIsNone(u["s4"].image)            # its picture failed to start: no placeholder in the text
        self.assertIsNone(u["s2"].image)            # never had one

    def test_assignments(self):
        u = units()
        self.assertEqual(u["s2"].table.title, "Meeting formats compared")
        self.assertTrue(u["s2"].quote and u["s3"].quote)
        self.assertFalse(u["s1"].quote)
        self.assertEqual(u["conclusion"].takeaways, 3)

    def test_token_budget_is_three_per_word_with_a_floor(self):
        self.assertEqual(unit_max_tokens(units()["s3"]), 3 * 270)
        self.assertEqual(unit_max_tokens(Unit("x", "section", "h", "a", 100)), 512)


class TestRequestData(unittest.TestCase):

    def data(self, unit_id, previous=None):
        return unit_request(REQUEST, PLAN, OUTLINE, units()[unit_id], previous)

    def test_the_part_gets_the_whole_outline_and_its_own_job(self):
        d = self.data("s2", "previous angle")
        self.assertEqual(d["post"]["title"], OUTLINE.title)
        self.assertEqual([p["id"] for p in d["post"]["parts"]], ["intro", "s1", "s2", "s3", "s4", "s5", "conclusion"])
        self.assertEqual(d["write"]["id"], "s2")
        self.assertEqual(d["write"]["heading"], "Match the format to the decision")
        self.assertEqual(d["already_covered"], "previous angle")

    def test_the_writer_is_asked_for_a_little_less_than_the_plan_because_it_writes_a_little_more(self):
        for unit_id in ("intro", "s1", "s3", "conclusion"):
            with self.subTest(unit=unit_id):
                unit, asked = units()[unit_id], self.data(unit_id)["write"]["target_words"]
                self.assertEqual(asked, round(unit.target_words * WRITER_LENGTH_FACTOR))
                self.assertLess(asked, unit.target_words)
        self.assertTrue(0.9 <= WRITER_LENGTH_FACTOR < 1)                    # a calibration, not a different length

    def test_the_parts_together_still_add_up_to_the_plan_once_the_writer_overshoots(self):
        asked = sum(self.data(u.id)["write"]["target_words"] for u in units_from_outline(OUTLINE, PLACED))
        self.assertAlmostEqual(asked / WRITER_LENGTH_FACTOR, PLAN.target_words, delta=10)

    def test_table_and_quote_assignments_are_spelled_out(self):
        a = self.data("s2")["assignments"]
        self.assertEqual(a["table"], {"title": "Meeting formats compared", "columns": ["Format", "Best for", "Watch out for"],
                                      "rows_hint": 4})
        self.assertTrue(a["quote"])

    def test_empty_assignments_are_left_out(self):
        self.assertNotIn("assignments", self.data("s1"))
        self.assertNotIn("already_covered", self.data("s1"))

    def test_the_writer_is_told_a_picture_follows_only_when_one_does(self):
        self.assertTrue(self.data("s1")["image_follows_first_paragraph"])
        self.assertNotIn("image_follows_first_paragraph", self.data("s2"))
        self.assertNotIn("image_follows_first_paragraph", self.data("s4"))

    def test_request_context_and_limits(self):
        d = self.data("s1")
        self.assertEqual((d["brief"], d["style_notes"], d["audience"], d["tone"]), (BRIEF, "Friendly, no jargon", "team leads", "warm"))
        self.assertEqual(d["limits"], {"max_words_without_visual_break": PLAN.max_prose_run})
        self.assertEqual(d["web"], PLAN.use_web)

    def test_conclusion_asks_for_takeaways(self):
        self.assertEqual(self.data("conclusion")["assignments"], {"takeaways": 3})

    def test_the_data_is_json(self):
        json.dumps(self.data("s2"))


class TestAssemble(unittest.TestCase):

    BODY = f"{filler(30)}\n\n{filler(30, 4)}\n\n> {filler(10)}\n"

    def test_heading_is_written_by_code(self):
        text = assemble(units()["s2"], self.BODY)
        self.assertTrue(text.startswith("## Match the format to the decision\n\n"))
        self.assertTrue(text.endswith("\n\n"))

    def test_the_intro_has_no_heading(self):
        self.assertFalse(assemble(units()["intro"], self.BODY).startswith("#"))

    def test_the_image_block_goes_after_the_first_paragraph(self):
        text = assemble(units()["s1"], self.BODY)
        blocks = [b for b in text.split("\n\n") if b]
        self.assertEqual(blocks[0], "## Why most remote meetings lose people early")
        self.assertTrue(blocks[2].startswith("![A laptop on a home-office desk showing a video call](placeholder::pid-1)\n*"))
        self.assertEqual(blocks[1], filler(30))

    def test_image_block_is_the_placeholder_then_an_italic_caption_line(self):
        self.assertEqual(image_block(units()["s1"]),
                         "![A laptop on a home-office desk showing a video call](placeholder::pid-1)\n"
                         "*Most meeting trouble starts before the call begins*")

    def test_a_missing_caption_leaves_just_the_image_line(self):
        u = units()["s1"]
        bare = Unit(u.id, u.kind, u.heading, u.angle, u.target_words, image=u.image.model_copy(update={"caption": ""}),
                    placeholder="pid-1")
        self.assertEqual(image_block(bare), "![A laptop on a home-office desk showing a video call](placeholder::pid-1)")

    def test_a_body_with_no_paragraph_gets_the_image_first(self):
        text = assemble(units()["s1"], TABLE)
        blocks = [b for b in text.split("\n\n") if b]
        self.assertTrue(blocks[1].startswith("![A laptop"))
        self.assertTrue(blocks[2].startswith("| A | B |"))

    def test_a_part_without_a_picture_is_just_heading_and_body(self):
        text = assemble(units()["s4"], self.BODY)
        self.assertNotIn("placeholder", text)

    def test_blocks_with_blank_lines_inside_survive_intact(self):
        body = "Lead.\n\n```python\na = 1\n\nb = 2\n```\n"
        self.assertIn("a = 1\n\nb = 2", assemble(units()["s2"], body))


class Calls:
    """Wraps a ScriptedLLM so a test can see the specs the part was called with."""

    def __init__(self, **kw):
        self.llm = ScriptedLLM(**kw)

    def run(self, unit_id, sources=None, request=REQUEST, plan=PLAN):
        us = units_from_outline(OUTLINE, PLACED)
        unit = next(u for u in us if u.id == unit_id)
        previous = us[us.index(unit) - 1].angle if unit_id != "intro" else None
        return write_unit(self.llm, request, plan, OUTLINE, unit, previous, sources, "rid")


class TestWriteUnit(unittest.TestCase):

    def test_a_good_first_reply_costs_one_call(self):
        c = Calls()
        part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 1)
        self.assertFalse(part.repaired)
        self.assertEqual(part.warnings, [])
        self.assertTrue(part.text.startswith("## Match the format to the decision"))
        self.assertIn("| Format | Best for | Watch out for |", part.text)

    def test_the_call_is_warm_budgeted_and_researches_only_when_planned(self):
        c = Calls()
        c.run("s3")
        system, messages, spec = c.llm.calls[0]
        self.assertEqual(system, section_system_prompt())
        self.assertEqual((spec.stage, spec.unit, spec.temperature, spec.max_tokens, spec.use_web),
                         ("unit", "s3", 0.7, 810, False))
        researching = plan_blog(BlogRequest.from_options("Write about the latest AI trends in 2026", "", {}))
        c2 = Calls()
        c2.run("s3", plan=researching)
        self.assertTrue(c2.llm.calls[0][2].use_web)

    def test_the_picture_is_inserted_after_the_first_paragraph(self):
        part = Calls().run("s1")
        self.assertIn("![A laptop on a home-office desk showing a video call](placeholder::pid-1)", part.text)
        self.assertLess(part.text.index("## Why"), part.text.index("placeholder::pid-1"))

    def test_the_model_cannot_write_headings_or_image_markup(self):
        reply = f"## My own heading\n\n{good_body(self.request('s2'))}\n\n![x](placeholder::evil)"
        part = Calls(units={"s2": reply}).run("s2")
        self.assertEqual(part.text.count("## "), 1)
        self.assertNotIn("evil", part.text)
        self.assertIn("heading", part.fixes)

    def request(self, unit_id):
        return unit_request(REQUEST, PLAN, OUTLINE, units()[unit_id], None)

    def test_a_missing_table_is_repaired_once(self):
        no_table = f"{filler(200)}\n\n> {filler(10)}\n"
        c = Calls(units={"s2": [no_table, good_body(self.request("s2"))]})
        part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 2)
        self.assertTrue(part.repaired)
        self.assertEqual(part.warnings, [])
        stage, repair = c.llm.calls[1][2], c.llm.calls[1][1]
        self.assertEqual(stage.stage, "unit_repair")
        self.assertEqual([m["role"] for m in repair], ["user", "assistant", "user"])
        self.assertIn("1. [table_missing]", repair[-1]["content"])

    def test_the_repair_does_not_research_again(self):
        researching = plan_blog(BlogRequest.from_options("Write about the latest AI trends in 2026", "", {}))
        c = Calls(units={"s2": [f"{filler(200)}\n", good_body(self.request("s2"))]})
        c.run("s2", plan=researching)
        self.assertEqual([call[2].use_web for call in c.llm.calls], [True, False])

    def test_still_wrong_after_the_repair_is_accepted_with_a_warning(self):
        no_table = f"{filler(200)}\n\n> {filler(10)}\n"
        c = Calls(units={"s2": no_table})
        part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 2)
        (warning,) = part.warnings
        self.assertEqual((warning["kind"], warning["unit"]), ("unit_lint", "s2"))
        self.assertIn("table_missing", warning["detail"])
        self.assertIn("## Match the format", part.text)         # still delivered

    def test_a_repair_that_makes_things_worse_is_not_used(self):
        original = f"{filler(200)}\n\n> {filler(10)}\n"                 # one problem: no table
        worse = f"{filler(40)}\n"                                      # no table, no quote, far too short
        c = Calls(units={"s2": [original, worse]})
        part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 2)                  # the repair was tried...
        self.assertTrue(part.repaired)
        self.assertGreater(part.words, 150)                    # ...but the original (about 200 words) is what ships
        (warning,) = part.warnings
        self.assertEqual([k for k in ("table_missing", "quote_missing", "length") if k in warning["detail"]], ["table_missing"])

    def test_repair_can_be_switched_off(self):
        c = Calls(units={"s2": f"{filler(200)}\n"})
        with patch.object(settings, "blog_repair_enabled", False):
            part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 1)
        self.assertTrue(part.warnings)

    def test_too_much_prose_in_a_row_is_repaired(self):
        c = Calls(units={"s5": [f"{filler(120)}\n\n{filler(120, 2)}\n", good_body(self.request("s5"))]})
        part = c.run("s5")
        self.assertTrue(part.repaired)
        self.assertEqual(part.warnings, [])
        self.assertIn("[prose_run]", c.llm.calls[1][1][-1]["content"])

    def test_an_empty_part_is_an_error_not_a_hole_in_the_post(self):
        for reply in ("", "   \n", "## Only a heading\n", "![x](placeholder::y)"):
            with self.subTest(reply=reply), self.assertRaises(s.BlogError):
                Calls(units={"s2": reply}).run("s2")

    def test_a_repair_that_is_imperfect_still_beats_an_empty_reply(self):
        part = Calls(units={"s2": ["", f"{filler(60)}\n"]}).run("s2")           # text, but no table / quote / right length
        self.assertTrue(part.repaired)
        self.assertIn("table_missing", part.warnings[0]["detail"])
        self.assertGreater(part.words, 0)

    def test_empty_first_then_good_is_repaired(self):
        part = Calls(units={"s2": ["", good_body(self.request("s2"))]}).run("s2")
        self.assertTrue(part.repaired)
        self.assertEqual(part.warnings, [])

    def test_an_extra_quote_stays_when_it_is_what_keeps_the_prose_broken_up(self):
        # s5 has no quote assigned and no picture. Without the quote its two paragraphs would be one 240-word run (limit 220).
        body = f"{filler(120)}\n\n> {filler(8)}\n\n{filler(120, 3)}\n"
        c = Calls(units={"s5": body})
        part = c.run("s5")
        self.assertEqual(len(c.llm.calls), 1)                                  # no repair was needed
        self.assertNotIn("extra_quote", part.fixes)
        self.assertEqual(part.text.count("\n> "), 1)
        self.assertEqual(part.warnings, [])

    def test_an_extra_quote_stays_when_dropping_it_would_make_the_part_too_short(self):
        body = f"{filler(60)}\n\n> {filler(26, 2)}\n"                         # 86 words with it, 60 without; the window is 78-162
        c = Calls(units={"intro": body})
        part = c.run("intro")
        self.assertEqual((len(c.llm.calls), part.text.count("> ")), (1, 1))

    def test_the_repair_reason_is_recorded(self):
        c = Calls(units={"s2": [f"{filler(200)}\n\n> {filler(10)}\n", good_body(self.request("s2"))]})
        part = c.run("s2")
        self.assertEqual(part.repair_kinds, ["table_missing"])

    def test_the_writer_is_told_what_is_not_assigned(self):
        self.assertEqual(self.request("s1")["not_assigned"], ["table", "quote"])
        self.assertNotIn("not_assigned", self.request("s2"))                          # both assigned: nothing to forbid
        self.assertEqual(self.request("conclusion")["not_assigned"], ["table", "quote"])
        self.assertEqual(self.request("intro")["not_assigned"], ["table", "quote"])

    def test_an_extra_quote_goes_when_a_picture_already_breaks_the_run(self):
        body = f"{filler(120)}\n\n> {filler(8)}\n\n{filler(100, 3)}\n"                  # s1 gets its picture after paragraph one
        c = Calls(units={"s1": body})
        part = c.run("s1")
        self.assertEqual((len(c.llm.calls), part.text.count("\n> ")), (1, 0))
        self.assertIn("extra_quote", part.fixes)

    def test_a_part_keeps_only_the_pull_quote_it_was_assigned(self):
        unassigned = good_body(self.request("s1")) + f"\n\n> {filler(10)}\n\n{filler(30)}\n\n> {filler(9, 2)}\n"
        part = Calls(units={"s1": unassigned}).run("s1")
        self.assertEqual(part.text.count("\n> "), 0)
        self.assertEqual(part.fixes.count("extra_quote"), 2)
        assigned = good_body(self.request("s2")) + f"\n\n{filler(30)}\n\n> {filler(9, 2)}\n"
        part = Calls(units={"s2": assigned}).run("s2")
        self.assertEqual(part.text.count("\n> "), 1)
        self.assertEqual(part.warnings, [])

    def test_a_miscounted_table_is_fixed_in_place_without_a_repair_call(self):
        body = good_body(self.request("s2"))
        broken = body.replace("|---|---|---|", "|---|---|")                 # the 3-column table loses a delimiter cell
        c = Calls(units={"s2": broken})
        part = c.run("s2")
        self.assertEqual(len(c.llm.calls), 1)
        self.assertIn("table_delimiter", part.fixes)
        self.assertEqual(part.warnings, [])

    def test_attribution_without_a_read_page_is_removed_not_repaired(self):
        body = with_quote(good_body(self.request("s2")), "> A quote of note.\n> — Jane Doe, CEO of Acme")
        c = Calls(units={"s2": body})
        part = c.run("s2")
        self.assertNotIn("Jane Doe", part.text)
        self.assertIn("> A quote of note.", part.text)                 # the quote stays, as the author's own sentence
        self.assertIn("attribution", part.fixes)
        self.assertEqual(len(c.llm.calls), 1)

    def test_attribution_with_a_read_page_survives(self):
        body = with_quote(good_body(self.request("s2")), "> Focus wins.\n> — [Reuters](https://reuters.com/a)")
        part = Calls(units={"s2": body}).run("s2", WebSources(["https://reuters.com/a"]))
        self.assertIn("[Reuters](https://reuters.com/a)", part.text)

    def test_word_count_is_reported(self):
        part = Calls().run("s3")
        self.assertTrue(0.65 * 260 <= part.words <= 1.35 * 260)


if __name__ == "__main__":
    unittest.main()
