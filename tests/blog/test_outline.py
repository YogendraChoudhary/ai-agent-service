"""
The outline stage: examples that obey the rules they teach, validation, the single repair, the rule-based fill.
The model is a scripted fake, so no provider key is needed.
"""

import copy
import json
import re
import unittest
from unittest.mock import patch

from config import settings
from src.blog import schema as s
from src.blog.outline import (
    TitleWatcher, build_outline, finalize, image_prompt_problems, normalise, outline_max_tokens, outline_request, plan_issues,
    top_level_strings, trim_guidance, validate_outline,
)
from src.blog.plan import BlogPlan, BlogRequest, plan_blog
from src.blog.prompts import LENGTHS, _dump, load_example, outline_system_prompt, outline_user_message
from tests.blog.fakes import ScriptedLLM, good_outline

BRIEFS = {
    "short": ("Write a short blog post for small business owners about choosing their first project management tool.", {"length": "short"}),
    "standard": ("Write a blog post for team leads about running remote meetings that people actually want to attend.", {}),
    "long": ("Write a long-form guide for homeowners who want to switch their home to renewable energy, step by step.", {"length": "long"}),
}
_DIGITS = re.compile(r"\d+")


def plan_of(length="standard", **options):
    brief, opts = BRIEFS[length]
    return plan_blog(BlogRequest.from_options(brief, "", {**opts, **options}))


def request_of_plan(plan, length="standard"):
    return outline_request(BlogRequest.from_options(BRIEFS[length][0], "", BRIEFS[length][1]), plan)


def dump(outline):
    return outline.model_dump(exclude_none=True)


class TestExamples(unittest.TestCase):
    """Few-shot examples teach more strongly than rules do, so each must obey the rules it teaches."""

    def test_every_length_has_an_example_and_the_plan_in_it_is_the_real_plan(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                example = load_example(length)
                plan = plan_of(length)
                # the example shows what the model really receives: the user message drops empty fields
                self.assertEqual(example["input"], json.loads(_dump(request_of_plan(plan, length))))

    def test_examples_validate_against_their_own_plan_without_any_findings(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                check = validate_outline(load_example(length)["output"], plan_of(length))
                self.assertEqual([e.message for e in check.errors], [])
                self.assertIsNotNone(check.outline)

    def test_examples_need_no_fill(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                outline = validate_outline(load_example(length)["output"], plan_of(length)).outline
                self.assertEqual(finalize(outline, plan_of(length))[1], [])

    def test_examples_add_no_numbers_the_brief_did_not_have(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                example = load_example(length)
                prose = json.dumps({k: v for k, v in example["output"].items()
                                    if k not in ("intro", "conclusion")} | {
                    "intro": {"hook": example["output"]["intro"]["hook"]},
                    "conclusion": {"heading": example["output"]["conclusion"]["heading"]}}, ensure_ascii=False)
                prose = re.sub(r'"(?:target_words|rows_hint|takeaways)": \d+|"id": "s\d+"', "", prose)
                self.assertLessEqual(set(_DIGITS.findall(prose)), set(_DIGITS.findall(example["input"]["brief"])))

    def test_example_picture_prompts_pass_the_picture_rules(self):
        for length in LENGTHS:
            for sec in load_example(length)["output"]["sections"]:
                for v in sec["visuals"]:
                    if v["kind"] == "image":
                        with self.subTest(length=length, section=sec["id"]):
                            self.assertEqual(image_prompt_problems(v["prompt"]), [])

    def test_system_prompt_is_byte_stable_per_length_and_carries_its_example(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                a, b = outline_system_prompt(length), outline_system_prompt(length)
                self.assertIs(a, b)
                self.assertIn(load_example(length)["output"]["title"], a)
        self.assertNotEqual(outline_system_prompt("short"), outline_system_prompt("long"))

    def test_the_prompts_tell_the_model_what_the_validators_enforce(self):
        from src.blog.prompts import section_system_prompt
        self.assertIn("or at most one more", outline_system_prompt("standard"))
        self.assertIn('"not_assigned" lists what this part must NOT contain', section_system_prompt())

    def test_no_request_data_leaks_into_the_system_prompt(self):
        brief = "Write about zeppelin maintenance 7391"
        self.assertNotIn("zeppelin", outline_system_prompt("standard"))
        self.assertIn("zeppelin", outline_user_message({"brief": brief}))


class TestNormalise(unittest.TestCase):

    def test_ids_and_positions_are_overwritten_never_trusted(self):
        data = copy.deepcopy(load_example("standard")["output"])
        data["sections"][0]["id"] = "intro-1"
        data["sections"][2]["id"] = "s2"
        data["sections"][0]["visuals"][0]["position"] = "end"
        out = normalise(data)
        self.assertEqual([x["id"] for x in out["sections"]], ["s1", "s2", "s3", "s4", "s5"])
        self.assertEqual(out["sections"][0]["visuals"][0]["position"], s.IMAGE_POSITION)

    def test_strings_that_markdown_will_carry_are_made_safe(self):
        data = copy.deepcopy(load_example("standard")["output"])
        data["title"] = "# **Remote** meetings\n that work"
        data["sections"][0]["heading"] = "1. Section 1: Why meetings <b>fail</b>"
        data["sections"][0]["visuals"][0].update(alt="A [laptop] on\na desk", caption="*Starts* early\n")
        out = normalise(data)
        self.assertEqual(out["title"], "Remote meetings that work")
        self.assertEqual(out["sections"][0]["heading"], "Why meetings fail")
        image = out["sections"][0]["visuals"][0]
        self.assertEqual((image["alt"], image["caption"]), ("A laptop on a desk", "Starts early"))

    def test_headings_keep_their_content(self):
        from src.blog.text import clean_heading
        for raw, want in (("Learning C#", "Learning C#"), ("F# for beginners", "F# for beginners"), ("## Title ##", "Title"),
                          ("Step 1: Audit your calendar", "Step 1: Audit your calendar"), ("Section 2: Costs", "Costs"),
                          ("5 ways to save", "5 ways to save")):
            with self.subTest(raw=raw):
                self.assertEqual(clean_heading(raw), want)

    def test_a_wrong_shape_is_left_for_the_schema_to_report(self):
        for data in ({}, {"sections": "x"}, {"sections": [1, 2]}, {"title": 5}):
            with self.subTest(data=data):
                normalise(data)       # must not raise

    def test_guidance_for_the_writer_that_runs_long_is_cut_by_code_not_sent_back(self):
        # a post was lost to a 301-character angle: the model cannot count characters, so its repair failed too
        data = copy.deepcopy(load_example("standard")["output"])
        long_angle = "Argue that a routine must survive a bad week. " * 12
        data["sections"][1]["angle"] = long_angle
        data["intro"]["hook"] = "Open on a calendar nobody leaves energised, then promise a fix. " * 10
        check = validate_outline(data, plan_of("standard"))
        self.assertEqual([e.message for e in check.errors if e.kind == "schema"], [])
        self.assertLessEqual(len(check.outline.sections[1].angle), s.ANGLE[1])
        self.assertTrue(check.outline.sections[1].angle.endswith("week."))        # cut at a sentence, not mid-word
        self.assertLessEqual(len(check.outline.intro.hook), s.HOOK_MAX)

    def test_an_angle_as_long_as_the_longest_gpt_wrote_is_kept_whole(self):
        # gpt-5.2 wrote angles up to 335 characters (a post was lost to the old 300 cap): nothing below the cap is touched
        angle = "Show why a fixed bedtime clock fails and what to use instead. " * 5 + "Then give two examples."
        self.assertGreater(len(angle), 330)
        data = copy.deepcopy(load_example("standard")["output"])
        data["sections"][1]["angle"] = angle
        outline = validate_outline(data, plan_of("standard")).outline
        self.assertEqual(outline.sections[1].angle, angle)

    def test_trim_guidance_cuts_at_a_sentence_then_at_a_word_and_leaves_short_text_alone(self):
        self.assertEqual(trim_guidance("Short enough.", 40), "Short enough.")
        self.assertEqual(trim_guidance("First sentence here. Second sentence goes on and on.", 40), "First sentence here.")
        self.assertEqual(trim_guidance("word " * 20, 33), "word word word word word word")
        self.assertEqual(trim_guidance("x" * 50, 10), "x" * 10)                       # no space to cut at
        # a sentence end in the first half would throw away most of the text: cut at a word instead
        self.assertEqual(trim_guidance("Tiny. then a very long second sentence without any stop", 30), "Tiny. then a very long second")

    def test_what_readers_see_is_still_the_models_to_shorten(self):
        data = copy.deepcopy(load_example("standard")["output"])
        data["sections"][0]["heading"] = "A heading that goes on and on " * 5
        data["subtitle"] = "s" * (s.SUBTITLE_MAX + 5)
        fields = [e.field for e in validate_outline(data, plan_of("standard")).errors]
        self.assertTrue(any("heading" in f for f in fields) and "subtitle" in fields, fields)


class TestTitleWatcher(unittest.TestCase):
    """The title opens the post while the outline is still streaming, so it is read with care: complete, top level, cleaned."""

    DOC = ('{\n "title": "Sleep that \\"survives\\" a busy week",\n "subtitle": "A calm system \\u2014 anchors and buffers.",\n'
           ' "intro": {"target_words": 120, "hook": "x"},\n "sections": [{"id": "s1", "visuals": '
           '[{"kind": "table", "title": "Not the post title"}]}]}')

    def watch(self, *texts):
        got = []
        watcher = TitleWatcher(lambda title, subtitle: got.append((title, subtitle)))
        for text in texts:
            watcher.feed(text)
        return got, watcher

    def test_a_string_counts_only_once_it_is_closed(self):
        closed_at = self.DOC.index('week"') + 5
        for end in range(len(self.DOC) + 1):
            with self.subTest(end=end):
                self.assertEqual("title" in top_level_strings(self.DOC[:end]), end >= closed_at)

    def test_escapes_are_decoded_and_nothing_nested_is_read(self):
        found = top_level_strings(self.DOC)
        self.assertEqual(found, {"title": 'Sleep that "survives" a busy week', "subtitle": "A calm system \u2014 anchors and buffers."})

    def test_a_title_written_after_a_section_is_not_the_posts_title(self):
        self.assertEqual(top_level_strings('{"intro": {"hook": "x"}, "title": "Late"}'), {})
        self.assertEqual(top_level_strings('{"sections": [{"title": "Table"}], "title": "Late"}'), {})

    def test_null_is_a_value_and_prose_or_fences_before_the_object_are_skipped(self):
        self.assertEqual(top_level_strings('Here is the plan:\n```json\n{"title": "T", "subtitle": null, "intro": {}}'),
                         {"title": "T", "subtitle": None})
        self.assertEqual(top_level_strings('{"title": "T", "subtitle": nu'), {"title": "T"})

    def test_it_reports_once_when_both_are_written_not_before(self):
        pieces = [self.DOC[:end] for end in range(10, len(self.DOC) + 1, 10)]
        got, watcher = self.watch(*pieces)
        self.assertEqual(got, [('Sleep that "survives" a busy week', "A calm system \u2014 anchors and buffers.")])
        self.assertEqual(watcher.shown, got[0])
        self.assertEqual(self.watch(self.DOC[:self.DOC.index("subtitle") + 30])[0], [])        # title alone: wait

    def test_the_strings_are_cleaned_exactly_as_the_outline_cleans_them(self):
        raw = '{"title": "# **Remote** meetings\\n that work", "subtitle": "A *plain* line\\n", "intro": {}}'
        (title, subtitle), = self.watch(raw)[0]
        data = normalise(json.loads(raw))
        self.assertEqual((title, subtitle), (data["title"], data["subtitle"]))
        self.assertEqual((title, subtitle), ("Remote meetings that work", "A *plain* line"))

    def test_a_null_subtitle_is_a_post_without_one_and_a_missing_subtitle_waits_for_the_whole_outline(self):
        self.assertEqual(self.watch('{"title": "A fine title", "subtitle": null, "intro": {}}')[0], [("A fine title", None)])
        self.assertEqual(self.watch('{"title": "A fine title", "intro": {"hook": "x"}, "subtitle": "Late"}')[0], [])

    def test_what_the_schema_would_reject_is_never_shown(self):
        for title, subtitle in (("Hi", "ok"), ("T" * (s.TITLE_MAX + 1), "ok"), ("A fine title", "s" * (s.SUBTITLE_MAX + 1))):
            with self.subTest(title=title[:12], subtitle=subtitle[:5]):
                self.assertEqual(self.watch(json.dumps({"title": title, "subtitle": subtitle, "intro": {}}))[0], [])

    def test_a_retried_call_starts_again_from_nothing_and_cannot_report_twice(self):
        got, _ = self.watch('{"title": "Hel', '{"title": "First one", "subtitle": "x", "intro": {}}',
                            '{"title": "Second one", "subtitle": "y", "intro": {}}')
        self.assertEqual(got, [("First one", "x")])


class TestValidate(unittest.TestCase):

    def setUp(self):
        self.plan = plan_of("standard")
        self.good = load_example("standard")["output"]

    def kinds(self, data):
        return [e.kind for e in validate_outline(data, self.plan).errors]

    def test_valid_reply_is_clean(self):
        self.assertEqual(self.kinds(self.good), [])

    def test_json_in_code_fences_and_prose_is_accepted(self):
        reply = "Here you go:\n```json\n" + json.dumps(self.good) + "\n```\nHope that helps."
        self.assertEqual(self.kinds(reply), [])

    def test_not_json(self):
        for reply in ("no json here", "[1, 2]", "", "{"):
            with self.subTest(reply=reply):
                check = validate_outline(reply, self.plan)
                self.assertIsNone(check.outline)
                self.assertEqual(check.errors[0].kind, "json")

    def test_schema_errors_name_the_field(self):
        bad = copy.deepcopy(self.good)
        bad["sections"][1]["visuals"][0]["columns"] = ["Only one"]
        bad["extra_field"] = 1
        check = validate_outline(bad, self.plan)
        self.assertIsNone(check.outline)
        fields = [e.field for e in check.errors]
        self.assertTrue(any("columns" in f for f in fields), fields)
        self.assertIn("extra_field", fields)

    def test_wrong_section_count(self):
        bad = copy.deepcopy(self.good)
        bad["sections"].pop()
        self.assertIn("section_count", self.kinds(bad))

    def test_section_words_must_match_the_plan(self):
        bad = copy.deepcopy(self.good)
        for sec in bad["sections"]:
            sec["target_words"] = 100
        self.assertIn("words_total", self.kinds(bad))

    def test_missing_visuals_are_counted_per_kind(self):
        no_table = copy.deepcopy(self.good)
        no_table["sections"][1]["visuals"] = [v for v in no_table["sections"][1]["visuals"] if v["kind"] != "table"]
        self.assertIn("tables", self.kinds(no_table))
        no_quotes = copy.deepcopy(self.good)
        for sec in no_quotes["sections"]:
            sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "quote"]
        self.assertIn("quotes", self.kinds(no_quotes))

    def test_one_visual_above_the_minimum_is_fine_two_is_a_post_that_reads_like_a_spreadsheet(self):
        one_more = copy.deepcopy(self.good)
        one_more["sections"][2]["visuals"].append(copy.deepcopy(self.good["sections"][1]["visuals"][0]))   # a 2nd table
        one_more["sections"][3]["visuals"].append({"kind": "quote", "mode": "pull"})                      # a 3rd quote
        self.assertEqual(self.kinds(one_more), [])
        two_more = copy.deepcopy(one_more)
        two_more["sections"][4]["visuals"] += [copy.deepcopy(self.good["sections"][1]["visuals"][0]), {"kind": "quote", "mode": "pull"}]
        self.assertEqual(sorted(self.kinds(two_more)), ["quotes", "tables"])
        message = next(e.message for e in validate_outline(two_more, self.plan).errors if e.kind == "tables")
        self.assertIn("Between 1 and 2", message)

    def test_images_must_match_exactly(self):
        fewer = copy.deepcopy(self.good)
        fewer["sections"][0]["visuals"] = []
        more = copy.deepcopy(self.good)
        more["sections"][1]["visuals"].append(copy.deepcopy(self.good["sections"][0]["visuals"][0]))
        self.assertIn("images", self.kinds(fewer))
        self.assertIn("images", self.kinds(more))

    def test_one_of_each_visual_per_section(self):
        bad = copy.deepcopy(self.good)
        bad["sections"][2]["visuals"].append({"kind": "quote", "mode": "pull"})
        self.assertIn("duplicate_visual", self.kinds(bad))

    def test_no_two_sections_in_a_row_without_a_visual(self):
        bad = copy.deepcopy(self.good)
        bad["sections"][3]["visuals"] = []        # s4 and s5 are now both bare (the plan still has its 3 images elsewhere)
        bad["sections"][0]["visuals"].append(copy.deepcopy(self.good["sections"][3]["visuals"][0]))
        self.assertIn("visual_gap", self.kinds(bad))

    def test_picture_prompts_are_linted(self):
        bad = copy.deepcopy(self.good)
        bad["sections"][0]["visuals"][0]["prompt"] = 'A poster that says "SALE" in big letters'
        self.assertIn("image_prompt", self.kinds(bad))

    def test_a_dict_is_accepted_as_well_as_a_string(self):
        self.assertIsNotNone(validate_outline(self.good, self.plan).outline)
        self.assertIsNotNone(validate_outline(json.dumps(self.good), self.plan).outline)


class TestPicturePromptRules(unittest.TestCase):

    def test_clear_violations(self):
        for prompt in ('A sign saying "Open" on a shop door', "A poster with big letters", "A laptop showing a chart",
                       "A famous politician shaking hands", "A banner with a logo"):
            with self.subTest(prompt=prompt):
                self.assertTrue(image_prompt_problems(prompt))

    def test_negations_and_plain_scenes_are_fine(self):
        for prompt in ("A laptop on a desk, no text or logos", "A mountain lake at dawn, without any text",
                       "Hands arranging sticky notes on a whiteboard", "A bright office with plants"):
            with self.subTest(prompt=prompt):
                self.assertEqual(image_prompt_problems(prompt), [])


class TestFinalize(unittest.TestCase):
    """Rule-based fill (§4.4): a visual the plan requires is added by code, never left to chance."""

    def setUp(self):
        self.plan = plan_of("standard")
        self.good = validate_outline(load_example("standard")["output"], self.plan).outline

    def finalize(self, mutate):
        data = dump(self.good)
        mutate(data)
        outline = s.Outline.model_validate(data)
        out, notes = finalize(outline, self.plan)
        self.assertEqual(plan_issues(out, self.plan, prompts=False), [], notes)
        return dump(out), notes

    def visuals(self, data, kind):
        return [(sec["id"], v) for sec in data["sections"] for v in sec["visuals"] if v["kind"] == kind]

    def test_a_missing_table_goes_to_the_comparative_section(self):
        def drop(d):
            d["sections"][1]["visuals"] = [v for v in d["sections"][1]["visuals"] if v["kind"] != "table"]
        out, notes = self.finalize(drop)
        # s2 compares formats ("compare"), so the table lands there again
        self.assertEqual([sid for sid, _ in self.visuals(out, "table")], ["s2"])
        self.assertIn("added 1 table(s)", notes)

    def test_a_missing_table_falls_back_to_section_two(self):
        def bare(d):
            for sec in d["sections"]:
                sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "table"]
                sec["heading"] = "A heading"
                sec["angle"] = "An angle about something else entirely and nothing structural."
        out, _ = self.finalize(bare)
        self.assertEqual([sid for sid, _ in self.visuals(out, "table")], ["s2"])

    def test_missing_quotes_go_to_the_longest_sections(self):
        def drop(d):
            for sec in d["sections"]:
                sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "quote"]
            d["sections"][3]["target_words"] = 300      # now the longest
            d["sections"][2]["target_words"] = 280
        out, notes = self.finalize(drop)
        self.assertEqual(sorted(sid for sid, _ in self.visuals(out, "quote")), ["s3", "s4"])

    def test_missing_images_go_to_the_first_and_the_middle_section(self):
        def drop(d):
            for sec in d["sections"]:
                sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "image"]
        out, notes = self.finalize(drop)
        ids = [sid for sid, _ in self.visuals(out, "image")]
        self.assertEqual(len(ids), 3)
        self.assertTrue({"s1", "s3"} <= set(ids), ids)          # first and middle first, then the next free one
        self.assertTrue(all("placeholder" not in v["prompt"] for _, v in self.visuals(out, "image")))

    def test_surplus_images_are_removed_evenly(self):
        def extra(d):
            for sec in d["sections"]:
                if not any(v["kind"] == "image" for v in sec["visuals"]):
                    sec["visuals"].append({"kind": "image", "prompt": "A calm scene of a desk and a window",
                                           "alt": "A desk by a window", "caption": "", "position": "end"})
        out, notes = self.finalize(extra)
        self.assertEqual(len(self.visuals(out, "image")), 3)
        self.assertIn("removed 2 surplus image(s)", notes)
        self.assertIn("s1", [sid for sid, _ in self.visuals(out, "image")])
        self.assertIn("s5", [sid for sid, _ in self.visuals(out, "image")])

    def test_surplus_tables_and_quotes_are_trimmed_to_one_above_the_minimum_evenly(self):
        def pile_on(d):
            table = {"kind": "table", "title": "Another comparison", "columns": ["A", "B"], "rows_hint": 3}
            for sec in d["sections"]:
                if not any(v["kind"] == "table" for v in sec["visuals"]):
                    sec["visuals"].append(copy.deepcopy(table))
                if not any(v["kind"] == "quote" for v in sec["visuals"]):
                    sec["visuals"].append({"kind": "quote", "mode": "pull"})
        out, notes = self.finalize(pile_on)
        self.assertEqual((len(self.visuals(out, "table")), len(self.visuals(out, "quote"))), (2, 3))
        self.assertIn("removed 3 surplus table(s)", notes)
        self.assertIn("removed 2 surplus quote(s)", notes)

    def test_closing_gaps_never_pushes_quotes_past_the_cap(self):
        def crowd(d):
            for i, sec in enumerate(d["sections"]):                 # 3 quotes (the cap for a plan of 2), then two bare neighbours
                sec["visuals"] = [{"kind": "quote", "mode": "pull"}] if i in (0, 1, 2) else []
            d["sections"][1]["visuals"].append(copy.deepcopy(dump(self.good)["sections"][1]["visuals"][0]))   # its table
            d["sections"][0]["visuals"].append(copy.deepcopy(dump(self.good)["sections"][0]["visuals"][0]))   # an image
            d["sections"][2]["visuals"].append(copy.deepcopy(dump(self.good)["sections"][2]["visuals"][0]))   # an image
        out, _ = self.finalize(crowd)                                # finalize asserts no rule is left broken
        self.assertLessEqual(len(self.visuals(out, "quote")), self.plan.quotes + 1)

    def test_extra_sections_are_dropped(self):
        def extra(d):
            d["sections"].append({**copy.deepcopy(d["sections"][0]), "id": "s6", "heading": "One more section"})
        out, notes = self.finalize(extra)
        self.assertEqual(len(out["sections"]), 5)
        self.assertIn("dropped 1 extra section(s)", notes)

    def test_gaps_between_visuals_are_closed(self):
        def gap(d):
            d["sections"][3]["visuals"] = []      # s4 loses its image, s5 has none: a gap
            d["sections"][1]["visuals"].append(copy.deepcopy(self.good.sections[0].visuals[0].model_dump()))
        out, notes = self.finalize(gap)
        self.assertTrue(any("gaps" in n for n in notes))

    def test_section_lengths_are_rescaled_to_the_plan(self):
        def small(d):
            for sec in d["sections"]:
                sec["target_words"] = 100
        out, notes = self.finalize(small)
        total = sum(sec["target_words"] for sec in out["sections"])
        self.assertLessEqual(abs(total - self.plan.section_words), 0.15 * self.plan.section_words)
        self.assertIn("rescaled section lengths to the plan", notes)

    def test_code_owned_fields_are_always_overwritten(self):
        def mutate(d):
            d["intro"]["target_words"] = 40
            d["conclusion"]["target_words"] = 299
            d["sections"][0]["id"] = "s9"
        out, notes = self.finalize(mutate)
        self.assertEqual((out["intro"]["target_words"], out["conclusion"]["target_words"]),
                         (self.plan.intro_words, self.plan.conclusion_words))
        self.assertEqual([sec["id"] for sec in out["sections"]], ["s1", "s2", "s3", "s4", "s5"])
        self.assertEqual(notes, [])        # routine overwrites are not reported

    def test_no_images_when_the_plan_has_none(self):
        plan = plan_of("standard", images=0)
        outline = validate_outline(load_example("standard")["output"], plan).outline
        out, _ = finalize(outline, plan)
        self.assertEqual(self.visuals(dump(out), "image"), [])


class TestBuildOutline(unittest.TestCase):

    def setUp(self):
        self.plan = plan_of("standard")
        self.request = BlogRequest.from_options(BRIEFS["standard"][0], "", {})

    def build(self, llm):
        return build_outline(llm, self.request, self.plan, "rid")

    def test_a_valid_first_reply_costs_one_call(self):
        llm = ScriptedLLM()
        result = self.build(llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse(result.repaired)
        self.assertEqual((result.filled, result.warnings), ([], []))

    def test_the_call_is_cold_and_budgeted(self):
        llm = ScriptedLLM()
        self.build(llm)
        system, messages, spec = llm.calls[0]
        self.assertEqual((spec.stage, spec.temperature, spec.use_web), ("outline", 0.4, False))
        self.assertEqual(spec.max_tokens, outline_max_tokens(self.plan))
        self.assertGreaterEqual(spec.max_tokens, 2000)
        self.assertEqual(system, outline_system_prompt("standard"))
        self.assertEqual(len(messages), 1)

    def test_the_request_data_carries_the_plan_numbers(self):
        data = outline_request(self.request, self.plan)
        self.assertEqual(data["plan"], {"target_words": 1450, "sections": 5, "section_words": 1210, "intro_words": 120,
                                        "conclusion_words": 120, "images": 3, "tables_at_least": 1, "quotes_at_least": 2})
        self.assertFalse(data["web_research"])

    def test_token_budget_grows_with_sections(self):
        small, big = plan_of("short"), plan_of("long")
        self.assertLess(outline_max_tokens(small), outline_max_tokens(big))

    def test_a_rule_failure_is_repaired_once_with_numbered_errors(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        llm = ScriptedLLM(outline=[bad, good_outline({"plan": {"target_words": 1400, "images": 3}})])
        result = self.build(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result.repaired)
        _, messages, spec = llm.calls[1]
        self.assertEqual(spec.stage, "outline_repair")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertIn("1. [tables]", messages[-1]["content"])
        self.assertEqual(result.filled, [])

    def test_a_fault_code_can_fix_without_loss_costs_no_model_call(self):
        # s4's picture moves to s2: the picture count is still right, but s4 and s5 are now both bare (visual_gap only)
        outline = good_outline({"plan": {"target_words": 1400, "images": 3}})
        outline["sections"][3]["visuals"] = []
        outline["sections"][1]["visuals"].append(copy.deepcopy(load_example("standard")["output"]["sections"][3]["visuals"][0]))
        check = validate_outline(outline, self.plan)
        self.assertEqual([e.kind for e in check.errors], ["visual_gap"])
        llm = ScriptedLLM(outline=outline)
        result = self.build(llm)
        self.assertEqual(len(llm.calls), 1)                        # no repair round trip
        self.assertFalse(result.repaired)
        self.assertEqual(plan_issues(result.outline, self.plan, prompts=False), [])
        self.assertTrue(any("gaps" in n for n in result.filled))
        self.assertEqual(result.warnings, [])                      # lossless: not worth the client's attention

    def test_other_lossless_faults_are_fixed_without_asking_again(self):
        short_quotes = good_outline({"plan": {"target_words": 1400, "images": 3}})
        for sec in short_quotes["sections"]:
            sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "quote"]
        wrong_lengths = good_outline({"plan": {"target_words": 1400, "images": 3}})
        for sec in wrong_lengths["sections"]:
            sec["target_words"] = 100
        for name, reply, kind in (("quotes", short_quotes, "quotes"), ("lengths", wrong_lengths, "words_total")):
            llm = ScriptedLLM(outline=reply)
            with self.subTest(fault=name):
                self.assertEqual([e.kind for e in validate_outline(reply, self.plan).errors], [kind])
                result = self.build(llm)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(result.warnings, [])
                self.assertEqual(plan_issues(result.outline, self.plan, prompts=False), [])

    def test_a_fault_that_needs_the_model_is_still_repaired_even_next_to_a_lossless_one(self):
        outline = good_outline({"plan": {"target_words": 1400, "images": 3}})
        outline["sections"][3]["visuals"] = []
        outline["sections"][4]["visuals"] = []
        outline["sections"][1]["visuals"] = [v for v in outline["sections"][1]["visuals"] if v["kind"] != "table"]   # no table
        llm = ScriptedLLM(outline=[outline, good_outline({"plan": {"target_words": 1400, "images": 3}})])
        result = self.build(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result.repaired)

    def test_unusable_structure_is_never_treated_as_lossless(self):
        for reply in ("not json", '{"title": "x"}'):
            with self.subTest(reply=reply):
                self.assertEqual(len(self.run_calls(reply)), 2)

    def run_calls(self, reply):
        llm = ScriptedLLM(outline=reply)
        with self.assertRaises(s.BlogError):
            self.build(llm)
        return llm.calls

    def test_what_the_first_reply_got_wrong_is_traced_for_tuning(self):
        from src.observability import finish_trace, get_trace, start_trace
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        start_trace("fp1", "/agent/stream")
        try:
            build_outline(ScriptedLLM(outline=[bad, good_outline({"plan": {"target_words": 1400, "images": 3}})]),
                          self.request, self.plan, "fp1")
            first = get_trace("fp1").attrs["outline_first_pass"]
            start_trace("fp2", "/agent/stream")
            build_outline(ScriptedLLM(), self.request, self.plan, "fp2")
            clean = get_trace("fp2").attrs
        finally:
            finish_trace("fp1")
            finish_trace("fp2")
        self.assertEqual([e["kind"] for e in first], ["tables"])
        self.assertIn("table directive", first[0]["message"])
        self.assertNotIn("outline_first_pass", clean)                      # a clean first reply leaves nothing to report

    def test_still_broken_after_the_repair_is_completed_by_code_with_a_warning(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        for sec in bad["sections"]:
            sec["visuals"] = [v for v in sec["visuals"] if v["kind"] not in ("table", "quote")]
        result = self.build(ScriptedLLM(outline=bad))
        self.assertEqual(plan_issues(result.outline, self.plan, prompts=False), [])
        self.assertEqual(result.outline.count("table"), 1)
        self.assertGreaterEqual(result.outline.count("quote"), 2)
        (warning,) = result.warnings
        self.assertEqual(warning["kind"], "outline_filled")
        self.assertIn("table", warning["detail"])

    def test_unusable_structure_after_the_repair_is_an_error(self):
        llm = ScriptedLLM(outline=["not json at all", '{"title": "x"}'])
        with self.assertRaises(s.BlogError) as ctx:
            self.build(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(ctx.exception.errors)

    def test_repair_can_be_switched_off_and_the_fill_still_runs(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        llm = ScriptedLLM(outline=bad)
        with patch.object(settings, "blog_repair_enabled", False):
            result = self.build(llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.outline.count("table"), 1)

    def test_code_fenced_json_is_fine(self):
        good = good_outline({"plan": {"target_words": 1400, "images": 3}})
        llm = ScriptedLLM(outline="```json\n" + json.dumps(good) + "\n```")
        self.assertEqual(len(self.build(llm).outline.sections), 5)

    def reply_for(self, **changes):
        outline = good_outline(request_of_plan(self.plan))
        outline.update(changes)
        return outline

    def test_the_title_is_reported_before_the_reply_is_finished(self):
        reply = json.dumps(self.reply_for())
        order = []

        def llm(system, messages, spec):
            for end in range(20, len(reply) + 20, 20):
                spec.on_text(reply[:end])
                order.append("chunk")
            order.append("returned")
            return reply
        result = build_outline(llm, self.request, self.plan, "rid", on_title=lambda t, sub: order.append(("title", t, sub)))
        (marker,) = [o for o in order if isinstance(o, tuple)]
        self.assertEqual(marker[1:], (result.outline.title, result.outline.subtitle))
        self.assertLess(order.index(marker), len(order) // 4)                 # the first sentences of a long reply
        self.assertEqual(order[-1], "returned")

    def test_only_a_listener_makes_the_calls_streamed(self):
        plain = ScriptedLLM()
        self.build(plain)
        self.assertIsNone(plain.calls[0][2].on_text)
        streamed = ScriptedLLM(outline=[self.reply_for(sections=[]), self.reply_for()])
        build_outline(streamed, self.request, self.plan, "rid", on_title=lambda *a: None)
        self.assertEqual([c[2].stage for c in streamed.calls], ["outline", "outline_repair"])
        self.assertTrue(all(c[2].on_text is not None for c in streamed.calls))

    def test_the_outline_carries_the_title_the_reader_has_seen_even_after_a_repair(self):
        first = self.reply_for(title="The title the reader sees", sections=self.reply_for()["sections"][:2])   # too few sections
        second = self.reply_for(title="Another title", subtitle="Another subtitle")
        seen = []
        result = build_outline(ScriptedLLM(outline=[first, second]), self.request, self.plan, "rid",
                               on_title=lambda t, sub: seen.append((t, sub)))
        self.assertTrue(result.repaired)
        self.assertEqual(len(seen), 1)
        self.assertEqual((result.outline.title, result.outline.subtitle), seen[0])
        self.assertEqual(result.outline.title, "The title the reader sees")

    def test_without_a_report_the_outline_keeps_the_title_it_validated(self):
        first = self.reply_for(title="T" * (s.TITLE_MAX + 1), sections=self.reply_for()["sections"][:2])
        seen = []
        result = build_outline(ScriptedLLM(outline=[first, self.reply_for(title="A valid title")]), self.request, self.plan, "rid",
                               on_title=lambda t, sub: seen.append(t))
        self.assertEqual(seen, ["A valid title"])                  # the first reply's title was never fit to show
        self.assertEqual(result.outline.title, "A valid title")

    def test_an_outline_that_ends_unusable_after_its_title_was_shown_is_still_an_error(self):
        broken = self.reply_for(sections="nope")
        seen = []
        with self.assertRaises(s.BlogError):
            build_outline(ScriptedLLM(outline=[broken, broken]), self.request, self.plan, "rid", on_title=lambda t, sub: seen.append(t))
        self.assertEqual(len(seen), 1)

    def test_an_outline_for_a_plan_without_images_has_none(self):
        plan = plan_of("standard", images=0)
        llm = ScriptedLLM()
        result = build_outline(llm, self.request, plan, "rid")
        self.assertEqual(result.outline.count("image"), 0)


if __name__ == "__main__":
    unittest.main()
