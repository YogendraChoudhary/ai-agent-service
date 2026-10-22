"""scripts/render_blog_preview.py: the review-only page. What is read from a post, what is counted, how pictures and
tables are drawn, and that the page it writes follows the Artifact contract."""

import base64
import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import render_blog_preview as rp
from src.longform.markdown_ast import parse

HEADER = ("<!-- openai / gpt-5.2 | outline model: gemini/gemini-3.5-flash-lite | brief: Write a blog post about tea | options: {}\n"
          "     22.34s total, first content 4.2s | tables 1 quotes 2 figures 3 | longest prose run 215 | words 1500\n"
          "     failures: [] | warnings: [] -->\n\n")
OLD_HEADER = ("<!-- openai / gpt-5.2 | brief: Write a blog post about tea | options: {}\n"
              "     29.77s total, first content 13.84s | tables 3 quotes 5 figures 3 | longest prose run 150 | words 1350\n"
              "     failures: [] | warnings: [] -->\n\n")

POST = ("# Title\n\n*A subtitle*\n\nOpening paragraph with a handful of words in it.\n\n"
        "## One\n\nText here.\n\n![A cup](pic.png)\n*A caption*\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
        "> A pull quote.\n\n- first\n- second\n\n"
        "## Two\n\nClosing words.\n")
BROKEN_TABLE = "| A | B | C |\n|---|---|\n| 1 | 2 | 3 |\n"     # the delimiter row has two cells, the header three


class TempDirCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name: str, text: str) -> str:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def post(self, role: str, text: str) -> rp.Post:
        return rp.load_post(role, Path(self.write(f"{role.lower()}.md", text)), {})


class TestHeader(unittest.TestCase):

    def test_reads_what_the_evaluation_script_wrote(self):
        fields, body = rp.parse_header(HEADER + "# Title\n")
        self.assertEqual(fields, {
            "model": "openai / gpt-5.2", "outline_model": "gemini/gemini-3.5-flash-lite", "brief": "Write a blog post about tea",
            "total_seconds": "22.34", "first_content_seconds": "4.2", "words": "1500"})
        self.assertEqual(body, "# Title\n")

    def test_an_older_header_has_no_outline_model(self):
        fields, _ = rp.parse_header(OLD_HEADER + "# T\n")
        self.assertIsNone(fields["outline_model"])
        self.assertEqual((fields["total_seconds"], fields["first_content_seconds"]), ("29.77", "13.84"))

    def test_a_post_without_a_header_is_left_alone(self):
        self.assertEqual(rp.parse_header("# Title\n\nText.\n"), ({}, "# Title\n\nText.\n"))


class TestCounting(unittest.TestCase):

    def test_counts_what_a_reader_sees(self):
        facts = rp.count_facts(rp.classify(parse(POST)))
        self.assertEqual({k: facts[k] for k in ("sections", "tables", "broken_tables", "quotes", "pictures", "lists")},
                         {"sections": 2, "tables": 1, "broken_tables": 0, "quotes": 1, "pictures": 1, "lists": 1})

    def test_words_are_prose_only(self):
        text = "Four words in here.\n\n| A | B |\n|---|---|\n| many words in a cell | x |\n"
        self.assertEqual(rp.count_facts(rp.classify(parse(text)))["words"], 4)

    def test_longest_run_of_text_without_a_visual_break(self):
        text = "one two three four five\n\nsix seven eight nine ten eleven twelve\n\n## H\n\nthree more words\n"
        self.assertEqual(rp.count_facts(parse(text))["longest_prose_run"], 12)

    def test_a_table_markdown_cannot_parse_is_broken_not_a_table_and_not_prose(self):
        blocks = rp.classify(parse("Intro words.\n\n" + BROKEN_TABLE))
        facts = rp.count_facts(blocks)
        self.assertEqual((facts["tables"], facts["broken_tables"], facts["words"]), (0, 1, 2))
        self.assertIn("broken", [kind for kind, _, _ in rp.rhythm_of(blocks)])

    def test_one_line_of_pipes_is_just_text(self):
        self.assertEqual(rp.count_facts(rp.classify(parse("| not | a | table |\n")))["broken_tables"], 0)


class TestRhythm(unittest.TestCase):

    def test_every_block_in_order_sized_by_the_room_it_takes(self):
        rhythm = rp.rhythm_of(rp.classify(parse(POST)))
        self.assertEqual([kind for kind, _, _ in rhythm],
                         ["heading", "text", "text", "heading", "text", "image", "table", "quote", "list", "heading", "text"])
        self.assertEqual(rhythm[0][:2], ("heading", 14))
        self.assertEqual(rhythm[3][:2], ("heading", 9))
        self.assertEqual(rhythm[1][1], 8)                                  # a short paragraph still shows
        self.assertEqual(rhythm[5][:2], ("image", 70))
        self.assertEqual(rhythm[6], ("table", 22 + 12 * 2, "Table, 2 rows"))
        self.assertGreater(rhythm[7][1], 26)


class TestRendering(TempDirCase):

    def picture(self, name: str = "pic.png", size=(2000, 1000)) -> None:
        Image.new("RGB", size, (200, 40, 40)).save(self.dir / name)

    def test_a_picture_is_resized_and_inlined(self):
        self.picture()
        out = rp.render_markdown("![A cup](pic.png)\n*A caption*\n", self.dir, {})
        data = re.search(r'src="data:image/jpeg;base64,([^"]+)"', out)
        self.assertIsNotNone(data)
        with Image.open(io.BytesIO(base64.b64decode(data.group(1)))) as shown:
            self.assertEqual((shown.format, shown.width), ("JPEG", rp.IMAGE_MAX_WIDTH))
        self.assertIn('alt="A cup"', out)
        self.assertIn("<em>A caption</em>", out)

    def test_a_picture_is_read_once_per_page(self):
        self.picture()
        with patch.object(rp.Image, "open", wraps=rp.Image.open) as opened:
            rp.render_markdown("![a](pic.png)\n\n![b](pic.png)\n", self.dir, {})
        self.assertEqual(opened.call_count, 1)

    def test_a_picture_that_cannot_be_found_is_flagged_not_dropped(self):
        out = rp.render_markdown("![Alt words](gone.jpg)\n", self.dir, {})
        self.assertIn("missing-picture", out)
        self.assertIn("picture file not found: gone.jpg", out)
        self.assertIn("Alt words", out)
        self.assertNotIn("<img", out)

    def test_a_picture_still_generating_says_so(self):
        out = rp.render_markdown("![Alt words](placeholder::abc-123)\n", self.dir, {})
        self.assertIn("the picture was still generating", out)
        self.assertNotIn("<img", out)

    def test_a_file_that_is_not_a_picture_is_missing_not_an_error(self):
        (self.dir / "notes.jpg").write_text("not an image")
        self.assertIsNone(rp.inline_image("notes.jpg", self.dir, {}))

    def test_raw_html_in_a_post_is_text_never_markup(self):
        out = rp.render_markdown("<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n", self.dir, {})
        self.assertNotIn("<script", out)
        self.assertNotIn("<img", out)
        self.assertIn("&lt;script&gt;", out)

    def test_a_real_table_is_wrapped_so_it_can_scroll(self):
        out = rp.render_markdown("| A | B |\n|---|---|\n| 1 | 2 |\n", self.dir, {})
        self.assertIn('<div class="table-wrap"><table>', out)
        self.assertNotIn("not-a-table", out)

    def test_a_table_that_does_not_render_is_flagged_and_shown_as_text(self):
        out = rp.render_markdown(BROKEN_TABLE, self.dir, {})
        self.assertIn('<p class="not-a-table">', out)
        self.assertNotIn("<table", out)

    def test_what_is_counted_as_broken_is_what_is_flagged(self):
        samples = (BROKEN_TABLE, "| A | B |\n|---|\n| **bold** | `code` |\n", "| one line |\n", "Plain text.\n",
                   "| A | B |\n|---|---|\n| 1 | 2 |\n")
        for text in samples:
            counted = any(b.kind == "broken" for b in rp.classify(parse(text)))
            self.assertEqual(counted, "not-a-table" in rp.render_markdown(text, self.dir, {}), text)


class TestGlance(TempDirCase):

    def rows(self, *posts: rp.Post) -> dict:
        return {label: values for label, values, _ in rp._glance_rows(list(posts))}

    def test_models_and_timings_come_from_the_header(self):
        rows = self.rows(self.post("Previous", OLD_HEADER + POST), self.post("Current", HEADER + POST))
        self.assertEqual(rows["Outline written by"], ["gpt-5.2", "gemini-3.5-flash-lite"])
        self.assertEqual(rows["First text on screen"], ["13.8 s", "4.2 s"])
        self.assertEqual(rows["Whole post"], ["29.8 s", "22.3 s"])

    def test_a_post_without_a_header_has_counts_but_no_timings(self):
        rows = self.rows(self.post("Post", POST))
        self.assertNotIn("Whole post", rows)
        self.assertEqual(rows["Tables"], ["1"])

    def test_the_outline_model_is_the_drafting_model_when_the_header_says_same_or_nothing(self):
        same = HEADER.replace("gemini/gemini-3.5-flash-lite", "same")
        rows = self.rows(self.post("Previous", same + POST), self.post("Current", OLD_HEADER + POST))
        self.assertEqual(rows["Outline written by"], ["gpt-5.2", "gpt-5.2"])

    def test_a_table_that_does_not_render_is_a_warning_row(self):
        previous, current = self.post("Previous", HEADER + BROKEN_TABLE), self.post("Current", HEADER + POST)
        self.assertIn(("Tables that do not render", ["1", "0"], True), rp._glance_rows([previous, current]))
        self.assertEqual(rp._glance([previous, current]).count("class='bad'"), 1)
        self.assertNotIn("Tables that do not render", self.rows(current, current))

    def test_what_changed_is_marked_on_the_current_column_only(self):
        previous = self.post("Previous", HEADER + POST)
        current = self.post("Current", HEADER + POST.replace("> A pull quote.\n\n", ""))
        glance = rp._glance([previous, current])
        self.assertIn("<th scope='row'>Pull quotes</th><td>1</td><td class='changed'>0</td>", glance)
        self.assertNotIn("</th><td class=", glance)                       # the first value of a row is the reference


class TestPage(TempDirCase):

    def compare_page(self, label: str = "Tea") -> str:
        previous = self.write("old.md", OLD_HEADER + "# Old\n\nSome words here.\n\n" + BROKEN_TABLE)
        current = self.write("new.md", HEADER + POST)
        return rp.build([[label, previous, current]], [], "Review", "What changed.")

    def test_follows_the_artifact_page_contract(self):
        page = self.compare_page()
        self.assertTrue(page.startswith('<meta charset="utf-8">'))
        self.assertIn("<title>Review</title>", page)
        self.assertIn('name="viewport"', page)
        self.assertIn('name="claude:viewport-fit-sides"', page)
        self.assertIsNone(re.search(r"<(!doctype|html|head|body)[\s>]", page, re.I))

    def test_only_fonts_are_loaded_from_elsewhere(self):
        hosts = set(re.findall(r'(?:href|src)="https?://([^/"]+)', self.compare_page()))
        self.assertEqual(hosts, {"fonts.googleapis.com", "fonts.gstatic.com"})

    def test_a_comparison_has_two_columns_and_a_toggle(self):
        page = self.compare_page()
        self.assertEqual(re.findall(r"class='col (\w+)'", page), ["previous", "current"])
        self.assertEqual(re.findall(r"data-view='(\w+)' aria-pressed", page), ["both", "previous", "current"])

    def test_tabs_exist_only_for_more_than_one_brief(self):
        one = self.compare_page()
        self.assertNotIn("class='tabs'", one)
        a, b = self.write("a.md", "# A\n\nWords.\n"), self.write("b.md", "# B\n\nWords.\n")
        two = rp.build([["First brief", a, b], ["Second brief", a, b]], [], "T", "I")
        self.assertEqual(re.findall(r"data-tab='([^']+)'", two), ["first-brief", "second-brief"])
        self.assertEqual(re.findall(r"<section class='set' id='([^']+)'", two), ["first-brief", "second-brief"])

    def test_a_single_post_has_one_column_and_no_toggle(self):
        page = rp.build([], [["Just one", self.write("solo.md", "# Solo\n\nWords.\n")]], "T", "I")
        self.assertEqual(re.findall(r"class='col (\w+)'", page), ["post"])
        self.assertNotIn("class='views'", page)

    def test_a_single_post_is_one_centred_column_and_a_comparison_is_not(self):
        solo = rp.build([], [["Just one", self.write("solo.md", "# Solo\n\nWords.\n")]], "T", "I")
        self.assertIn("data-single", re.search(r"<section class='set'[^>]*>", solo).group(0))
        self.assertNotIn("data-single", re.search(r"<section class='set'[^>]*>", self.compare_page()).group(0))

    def test_two_briefs_cannot_share_a_label(self):
        a, b = self.write("a.md", "# A\n"), self.write("b.md", "# B\n")
        with self.assertRaisesRegex(ValueError, "own label"):
            rp.build([["Same", a, b], ["same!", a, b]], [], "T", "I")

    def test_nothing_to_render_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Nothing to render"):
            rp.build([], [], "T", "I")

    def test_labels_titles_and_briefs_are_escaped(self):
        post = self.write("p.md", "<!-- openai / gpt-5.2 | brief: Write <b>bold</b> | options: {} -->\n# P\n\nWords.\n")
        page = rp.build([], [["<i>Label</i>", post]], "<u>Title</u>", "<s>Intro</s>")
        for raw in ("<i>Label</i>", "<u>Title</u>", "<s>Intro</s>", "<b>bold</b>"):
            self.assertNotIn(raw, page)
        self.assertIn("&lt;i&gt;Label&lt;/i&gt;", page)


class TestCommandLine(TempDirCase):

    def run_main(self, *argv: str):
        with patch("sys.argv", ["render_blog_preview.py", *argv]), contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            return rp.main(), out.getvalue()

    def test_writes_one_self_contained_file(self):
        old, new = self.write("old.md", OLD_HEADER + POST), self.write("new.md", HEADER + POST)
        target = self.dir / "nested" / "page.html"
        code, printed = self.run_main("--out", str(target), "--title", "My review", "--compare", "Tea", old, new)
        self.assertEqual(code, 0)
        self.assertIn("<title>My review</title>", target.read_text(encoding="utf-8"))
        self.assertIn(str(target), printed)

    def test_a_missing_file_is_a_usage_error(self):
        target = self.dir / "page.html"
        with self.assertRaises(SystemExit) as raised:
            self.run_main("--out", str(target), "--post", "Solo", str(self.dir / "nope.md"))
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(target.exists())

    def test_it_needs_something_to_render(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main("--out", str(self.dir / "page.html"))
        self.assertEqual(raised.exception.code, 2)

    def test_a_repeated_label_is_a_usage_error(self):
        a, b = self.write("a.md", "# A\n"), self.write("b.md", "# B\n")
        with self.assertRaises(SystemExit) as raised:
            self.run_main("--out", str(self.dir / "page.html"), "--compare", "Same", a, b, "--compare", "Same", a, b)
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
