"""
The gate between a drafted part and the editor: deterministic fixes (sanitize), the checks worth a repair (lint),
and the whole-post report. Includes rule D13: no attributed quote and no URL without a page that was read.
"""

import unittest

from src.blog.sources import WebSources
from src.blog.validate import article_warnings, lint, longest_prose_run, sanitize, sentence_count, trim_extra_quotes
from src.longform.markdown_ast import parse
from tests.blog.fakes import filler


class TestSanitize(unittest.TestCase):

    def test_clean_text_is_unchanged_apart_from_block_spacing(self):
        out = sanitize("One paragraph.\n\n\n\nTwo paragraph.\n")
        self.assertEqual((out.text, out.fixes), ("One paragraph.\n\nTwo paragraph.", []))

    def test_headings_are_removed_at_every_level(self):
        out = sanitize("# One\n\nText.\n\n## Two\n\n### Three\n\nMore text.")
        self.assertEqual(out.text, "Text.\n\nMore text.")
        self.assertEqual(out.fixes, ["heading"] * 3)

    def test_a_hash_inside_a_code_fence_is_not_a_heading(self):
        out = sanitize("Text.\n\n```bash\n# a comment\n```\n")
        self.assertIn("# a comment", out.text)

    def test_the_model_never_writes_image_markup(self):
        out = sanitize("Intro.\n\n![hero](placeholder::abc-123)\n\nBody placeholder::def-456 text <ImagePrompt a=\"b\">x</ImagePrompt> end.")
        self.assertEqual(out.text, "Intro.\n\nBody  text  end.")
        self.assertIn("image_markup", out.fixes)
        self.assertNotIn("placeholder", out.text)

    def test_a_reply_wrapped_in_a_code_fence_is_unwrapped(self):
        out = sanitize("```markdown\nFirst.\n\nSecond.\n```")
        self.assertEqual(out.text, "First.\n\nSecond.")
        self.assertIn("fence", out.fixes)

    def test_crlf_is_normalised(self):
        self.assertEqual(sanitize("A.\r\n\r\nB.").text, "A.\n\nB.")

    def test_empty_and_none(self):
        self.assertEqual(sanitize("").text, "")
        self.assertEqual(sanitize(None).text, "")


class TestTablesAndQuotes(unittest.TestCase):
    """Found in live output: a 4-column header with a 3-cell delimiter row is not a table to the editor."""

    LIVE_FAILURE = ("Lead.\n\n| Input that steals sleep | Soft boundary | Buffer | Planned exception |\n|---|---|---|\n"
                    "| Caffeine | Set a cut-off | Switch to decaf | Coffee with a friend |\n"
                    "| Screens | Phone out of bed | Night mode | One episode |\n")

    def kinds(self, text):
        return [b.kind for b in parse(text) if b.kind != "blank"]

    def test_a_miscounted_delimiter_row_is_rebuilt_so_the_table_is_a_table(self):
        self.assertEqual(self.kinds(self.LIVE_FAILURE), ["paragraph", "paragraph"])          # what the editor would have seen
        out = sanitize(self.LIVE_FAILURE)
        self.assertEqual(self.kinds(out.text), ["paragraph", "table"])
        self.assertIn("table_delimiter", out.fixes)
        self.assertIn("|---|---|---|---|", out.text)
        self.assertEqual(lint(out.text, expect_table=True).errors, [])

    def test_too_many_delimiter_cells_are_fixed_the_same_way(self):
        out = sanitize("| A | B |\n|---|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")
        self.assertEqual(self.kinds(out.text), ["table"])

    def test_correct_tables_and_ordinary_pipes_are_left_alone(self):
        good = "| A | B |\n|:--|--:|\n| 1 | 2 |"
        self.assertEqual(sanitize(good).text, good)
        self.assertEqual(sanitize(good).fixes, [])
        prose = "Use a | b | c to separate options.\nSecond line | with | pipes"
        self.assertEqual(sanitize(prose).text, prose)
        notdelim = "| not | a |\n| table | at all |"
        self.assertEqual(sanitize(notdelim).text, notdelim)

    def test_a_quote_that_is_only_an_attribution_disappears_and_the_real_quote_stays(self):
        self.assertEqual(sanitize("A.\n\n> — Anonymous Source\n\n> real quote\n").text, "A.\n\n> real quote")


class TestTrimExtraQuotes(unittest.TestCase):
    """Extra pull quotes go, unless a quote is what keeps a rule satisfied: then removing it would cost a repair call."""

    ALWAYS_FINE = staticmethod(lambda text: set())

    def lint_kinds(self, **kw):
        return lambda text: {e.kind for e in lint(text, **kw).errors}

    def test_extras_beyond_the_assignment_are_dropped_in_document_order_of_keeping(self):
        text = "A.\n\n> one\n\nB.\n\n> two\n\nC.\n\n> three\n\n"
        self.assertEqual(trim_extra_quotes(text, 3, self.ALWAYS_FINE), (text, 0))
        self.assertEqual(trim_extra_quotes(text, 1, self.ALWAYS_FINE), ("A.\n\n> one\n\nB.\n\nC.\n\n", 2))
        self.assertEqual(trim_extra_quotes(text, 0, self.ALWAYS_FINE), ("A.\n\nB.\n\nC.\n\n", 3))

    def test_a_quote_that_breaks_up_a_long_run_stays(self):
        text = f"{filler(120)}\n\n> {filler(8)}\n\n{filler(120, 3)}\n\n"
        errors_of = self.lint_kinds(max_prose_run=220)
        self.assertEqual(errors_of(text), set())
        self.assertEqual(trim_extra_quotes(text, 0, errors_of), (text, 0))           # 240 words in a row without it

    def test_a_quote_that_keeps_a_short_part_inside_its_length_window_stays(self):
        text = f"{filler(60)}\n\n> {filler(20, 2)}\n\n"
        errors_of = self.lint_kinds(target_words=100)                                   # 80 words with it, 60 without
        self.assertEqual(errors_of(text), set())
        self.assertEqual(trim_extra_quotes(text, 0, errors_of)[1], 0)

    def test_only_the_quotes_that_are_free_to_remove_are_removed(self):
        text = f"{filler(120)}\n\n> {filler(8)}\n\n{filler(120, 3)}\n\n> {filler(9)}\n\n{filler(30, 5)}\n\n"
        out, removed = trim_extra_quotes(text, 0, self.lint_kinds(max_prose_run=220))
        self.assertEqual(removed, 1)                                                      # the one between the long paragraphs stays
        self.assertEqual(out.count("\n> "), 1)

    def test_a_part_that_already_has_errors_is_not_blocked_from_dropping_extras(self):
        text = f"A.\n\n> one\n\n{filler(10)}\n\n"
        errors_of = lambda t: {"length"}                                                  # unrelated, unchanged by the trim
        self.assertEqual(trim_extra_quotes(text, 0, errors_of)[1], 1)

    def test_tables_are_never_touched(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        text = f"A.\n\n{table}\n\nB.\n\n"
        self.assertEqual(trim_extra_quotes(text, 0, self.ALWAYS_FINE), (text, 0))


class TestUrls(unittest.TestCase):

    def test_a_link_to_a_page_nobody_read_keeps_its_text(self):
        out = sanitize("See [the study](https://example.com/study) for details.")
        self.assertEqual(out.text, "See the study for details.")
        self.assertIn("url", out.fixes)

    def test_a_bare_url_is_removed_and_sentence_punctuation_stays(self):
        self.assertEqual(sanitize("Read https://example.com/a. Then www.example.org!").text, "Read . Then !")

    def test_urls_of_pages_that_were_read_survive(self):
        sources = WebSources(["https://www.reuters.com/world/story/"])
        out = sanitize("Per [Reuters](https://reuters.com/world/story) and https://reuters.com/world/story.", sources)
        self.assertIn("[Reuters](https://reuters.com/world/story)", out.text)
        self.assertEqual(out.fixes, [])

    def test_a_page_that_was_searched_but_not_read_does_not_count(self):
        out = sanitize("[x](https://other.example.com/p)", WebSources(["https://reuters.com/a"]))
        self.assertEqual(out.text, "x")


class TestAttribution(unittest.TestCase):
    """Rule D13: a pull quote is the author's own sentence; an attributed quote needs a page that was read."""

    def test_an_attribution_line_is_dropped_when_no_page_was_read(self):
        out = sanitize("> Meetings fail early.\n> — Jane Doe, CEO of Acme")
        self.assertEqual(out.text, "> Meetings fail early.")
        self.assertIn("attribution", out.fixes)

    def test_a_trailing_attribution_on_the_same_line_is_dropped(self):
        for line in ("> Meetings fail early. — Jane Doe", "> Meetings fail early. -- Jane Doe, CEO",
                     "> Meetings fail early. - Jane Doe"):
            with self.subTest(line=line):
                self.assertEqual(sanitize(line).text, "> Meetings fail early.")

    def test_a_dash_inside_the_sentence_is_punctuation_not_attribution(self):
        for line in ("> The goal isn't more meetings — it's better ones.",
                     "> Fewer meetings -- better ones.", "> We tried it — and it worked.",
                     "> Do less — Then do it well."):
            with self.subTest(line=line):
                self.assertEqual(sanitize(line).text, line)

    def test_a_quote_that_is_only_an_attribution_disappears(self):
        self.assertEqual(sanitize("Text.\n\n> — Anonymous Source\n\nMore.").text, "Text.\n\nMore.")

    def test_a_sourced_attribution_survives_only_with_a_read_page(self):
        text = "> Focus wins. — [Reuters](https://reuters.com/a)"
        self.assertEqual(sanitize(text, WebSources(["https://reuters.com/a"])).text, text)
        self.assertEqual(sanitize(text, WebSources()).text, "> Focus wins.")
        self.assertEqual(sanitize(text, None).text, "> Focus wins.")

    def test_multi_line_quotes_keep_their_body(self):
        out = sanitize("> First line.\n> Second line.\n> — Someone Known")
        self.assertEqual(out.text, "> First line.\n> Second line.")


class TestLint(unittest.TestCase):

    TABLE = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"

    def test_a_good_part_is_clean(self):
        text = f"## Heading\n\n{filler(100)}\n\n{self.TABLE}\n\n> {filler(12)}\n\n{filler(100)}\n"
        out = lint(text, expect_table=True, expect_quotes=1, target_words=200, max_prose_run=220)
        self.assertEqual([e.kind for e in out.errors], [])
        self.assertEqual(out.longest_run, 100)

    def test_no_text_is_an_error(self):
        self.assertEqual([e.kind for e in lint("## Heading\n\n").errors], ["empty"])

    def test_missing_table_and_quote(self):
        out = lint(f"{filler(100)}\n", expect_table=True, expect_quotes=1)
        self.assertEqual([e.kind for e in out.errors], ["table_missing", "quote_missing"])

    def test_malformed_tables(self):
        cases = {
            "ragged": "| A | B | C |\n|---|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 |",
            "one column": "| A |\n|---|\n| 1 |\n| 2 |",
            "one row": "| A | B |\n|---|---|\n| 1 | 2 |",
        }
        for name, table in cases.items():
            with self.subTest(case=name):
                out = lint(f"{filler(50)}\n\n{table}\n", expect_table=True)
                self.assertEqual([e.kind for e in out.errors], ["table_shape"])

    def test_one_good_table_is_enough(self):
        out = lint(f"{filler(50)}\n\n| A |\n|---|\n| 1 |\n\n{self.TABLE}\n", expect_table=True)
        self.assertEqual(out.errors, [])

    def test_prose_run_counts_paragraphs_until_a_visual_break(self):
        text = f"{filler(120)}\n\n{filler(120)}\n"
        out = lint(text, max_prose_run=220)
        self.assertEqual([e.kind for e in out.errors], ["prose_run"])
        self.assertEqual(out.longest_run, 240)
        for brk in (f"- {filler(5)}", f"> {filler(5)}", self.TABLE, "![a](placeholder::x)\n*c*", "## Next", "```\ncode\n```"):
            with self.subTest(break_=brk[:12]):
                broken = f"{filler(120)}\n\n{brk}\n\n{filler(120)}\n"
                self.assertEqual(lint(broken, max_prose_run=220).errors, [])

    def test_prose_run_ignores_blank_lines_between_paragraphs(self):
        self.assertEqual(longest_prose_run(parse(f"{filler(50)}\n\n\n\n{filler(50)}\n")), 100)

    def test_length_window_is_plus_or_minus_thirty_five_percent(self):
        for words, ok in ((65, True), (64, False), (135, True), (136, False), (100, True)):
            with self.subTest(words=words):
                out = lint(f"{filler(words)}\n", target_words=100)
                self.assertEqual("length" not in [e.kind for e in out.errors], ok, out.words)

    def test_minor_findings_are_notes_not_errors(self):
        long_paragraph = "One. Two. Three. Four. Five. Six words here.\n"
        many_lists = "- a\n- b\n\ntext between\n\n- c\n- d\n"
        out = lint(long_paragraph + "\n" + many_lists)
        self.assertEqual(out.errors, [])
        self.assertEqual(sorted(out.notes), ["lists", "long_paragraph"])

    def test_visuals_the_part_was_not_assigned_are_noted_for_the_trace(self):
        both = f"{filler(40)}\n\n{self.TABLE}\n\n> {filler(8)}\n\n{filler(40)}\n"
        self.assertEqual(sorted(lint(both).notes), ["unassigned_quote", "unassigned_table"])
        self.assertEqual(lint(both, expect_table=True, expect_quotes=1).notes, [])
        self.assertEqual(lint(both, expect_table=True, expect_quotes=1).errors, [])

    def test_sentence_count_ignores_abbreviations_and_decimals(self):
        cases = {"One. Two. Three.": 3, "Dr. Smith went e.g. home. Then left.": 2, "Costs 3.5 million. Really? Yes.": 3,
                 "No end": 1, "": 0}
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(sentence_count(text), want)


class TestArticleWarnings(unittest.TestCase):

    DOC = "# Title\n\n## One\n\ntext\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n> quote\n\n## Two\n\ntext\n"

    def test_a_post_that_meets_its_plan_has_no_warnings(self):
        self.assertEqual(article_warnings(self.DOC, tables=1, quotes=1, headings=["One", "Two"]), [])

    def test_shortfalls_are_reported(self):
        kinds = [w["kind"] for w in article_warnings(self.DOC, tables=2, quotes=3, headings=["One", "Two"])]
        self.assertEqual(kinds, ["quota_tables", "quota_quotes"])

    def test_headings_must_match_the_outline_in_order(self):
        warnings = article_warnings(self.DOC, tables=1, quotes=1, headings=["Two", "One"])
        self.assertEqual([w["kind"] for w in warnings], ["heading_mismatch"])

    def test_a_heading_that_ends_in_a_literal_hash_matches_itself(self):
        self.assertEqual(article_warnings("## Learning C#\n\ntext\n", tables=0, quotes=0, headings=["Learning C#"]), [])

    def test_heading_comparison_ignores_case_and_closing_hashes(self):
        doc = "## ONE ##\n\ntext\n\n## two\n\ntext\n"
        self.assertEqual(article_warnings(doc, tables=0, quotes=0, headings=["One", "Two"]), [])


if __name__ == "__main__":
    unittest.main()
