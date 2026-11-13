"""
markdown_ast: every line belongs to exactly one block, kinds and counts match what a reader sees, and the
original text is never re-rendered (join(parse(text)) == text).
"""

import random
import unittest

from src.longform.markdown_ast import (
    heading_levels, heading_lines, join, list_shape, parse, prose_words, shape, table_shape, word_count,
)

DOC = """# Title

*Sub*

Intro paragraph with a [link](https://example.com/a?b=c) and `code`. Two sentences here.

## Section

First para.

![alt text](placeholder::abc-123)
*A caption line*

| Option | Cost | Best for |
|---|---|---|
| A | $1 | Small |
| B | $2 |

> A pull quote.

- one
  - nested
- two

1. first
2. second

```python
x = 1

y = 2
```

<div>html</div>

[ref]: http://example.com "Title"

Setext
======

Last para.
"""


class TestParse(unittest.TestCase):

    def test_kinds_in_document_order(self):
        self.assertEqual(shape(parse(DOC)), (
            "heading", "paragraph", "paragraph", "heading", "paragraph", "image", "table", "blockquote",
            "list", "list", "fence", "html", "other", "heading", "paragraph"))

    def test_join_is_the_identity(self):
        self.assertEqual(join(parse(DOC)), DOC)

    def test_identity_on_awkward_input(self):
        for text in ("", "\n", "\n\n\n", "no newline", "a\r\nb\r\n\r\nc", "a\rb", "    indented\n\ntext",
                     "text\n```\nunclosed", "x\x0by", "[a]: http://x\n[b]: http://y\n", "![i](u)", "\ttab",
                     "| a | b |\n|---|---|\n| 1 | 2 |"):
            with self.subTest(text=text):
                self.assertEqual(join(parse(text)), text)

    def test_identity_on_random_documents(self):
        rnd = random.Random(7)
        parts = ["# ", "## ", "> ", "- ", "1. ", "| a | b |\n", "|---|---|\n", "```\n", "text ", "\n", "\n\n",
                 "![a](b)", "    ", "<div>", "***", "[x]: y\n", "\r\n"]
        for _ in range(500):
            text = "".join(rnd.choice(parts) for _ in range(rnd.randint(1, 25)))
            self.assertEqual(join(parse(text)), text, repr(text))

    def test_line_ranges_slice_the_source(self):
        lines = DOC.splitlines(keepends=True)
        for b in parse(DOC):
            self.assertEqual(b.text, "".join(lines[b.start:b.end]))

    def test_empty_text_has_no_blocks(self):
        self.assertEqual(parse(""), [])
        self.assertEqual(parse(None), [])


class TestBlockDetails(unittest.TestCase):

    def setUp(self):
        self.blocks = parse(DOC)

    def by_kind(self, kind):
        return [b for b in self.blocks if b.kind == kind]

    def test_headings_have_levels_including_setext(self):
        self.assertEqual(heading_levels(self.blocks), (1, 2, 1))

    def test_an_image_with_a_caption_line_is_one_figure(self):
        (figure,) = self.by_kind("image")
        self.assertEqual((figure.images, figure.words), (1, 3))          # the caption's words, not the alt text

    def test_an_image_inside_a_long_paragraph_stays_a_paragraph(self):
        (b,) = parse("![i](u) " + "word " * 60 + "\n")
        self.assertEqual((b.kind, b.images), ("paragraph", 1))

    def test_two_images_are_not_one_figure(self):
        (b,) = parse("![a](x) ![b](y)\n")
        self.assertEqual((b.kind, b.images), ("paragraph", 2))

    def test_lists_know_items_order_and_depth(self):
        bullets, numbered = self.by_kind("list")
        self.assertEqual((bullets.items, bullets.ordered, bullets.depths), (2, False, (0, 1, 0)))
        self.assertEqual((numbered.items, numbered.ordered), (2, True))
        self.assertEqual(list_shape(self.blocks), ((0, 1, 0), (0, 0)))

    def test_a_code_fence_with_blank_lines_is_one_block_and_not_prose(self):
        (fence,) = self.by_kind("fence")
        self.assertEqual((fence.words, fence.text.count("\n")), (0, 5))

    def test_words_exclude_urls_and_code_blocks(self):
        intro = self.by_kind("paragraph")[1]
        self.assertEqual(intro.words, 10)       # "Intro paragraph with a link and code. Two sentences here."

    def test_prose_words_count_paragraphs_lists_and_quotes_only(self):
        self.assertEqual(prose_words(self.blocks), sum(
            b.words for b in self.blocks if b.kind in ("paragraph", "list", "blockquote")))

    def test_word_count_treats_hyphenated_and_apostrophised_words_as_one(self):
        self.assertEqual(word_count("It's a well-known, state-of-the-art fact."), 5)
        self.assertEqual(word_count(""), 0)
        self.assertEqual(word_count(None), 0)


class TestTables(unittest.TestCase):

    def shape_of(self, text):
        (block,) = [b for b in parse(text) if b.kind == "table"]
        return table_shape(block)

    def test_a_complete_table_is_consistent(self):
        s = self.shape_of("| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n")
        self.assertEqual((s.header, s.delimiter, s.rows, s.consistent), (2, 2, (2, 2), True))

    def test_a_short_or_long_row_is_inconsistent(self):
        self.assertFalse(self.shape_of("| A | B | C |\n|---|---|---|\n| 1 | 2 |\n").consistent)
        self.assertFalse(self.shape_of("| A | B |\n|---|---|\n| 1 | 2 | 3 |\n").consistent)

    def test_escaped_pipes_do_not_split_cells(self):
        self.assertEqual(self.shape_of("| A | B |\n|---|---|\n| 1 \\| x | 2 |\n").rows, (2,))

    def test_rows_without_outer_pipes(self):
        self.assertEqual(self.shape_of("A | B\n--|--\n1 | 2\n").rows, (2,))

    def test_non_tables_have_no_shape(self):
        (para,) = parse("just text\n")
        self.assertIsNone(table_shape(para))


class HeadingLinesMatchParse(unittest.TestCase):
    """heading_lines() skips the inline pass for speed; it must still find exactly the headings parse() finds."""

    FRAGMENTS = [
        "# H1\n", "## Two ##\n", "### t\n", "Setext\n=====\n", "Sub\n---\n", "```\n# in code\n```\n", "    # indented code\n",
        "> # quoted\n", "- # in list\n", "- item\n", "1. one\n", "text line\n", "\n", "#nohash\n", "####### seven\n",
        "<div>\n# in html?\n</div>\n", "| a | b |\n|---|---|\n| # | x |\n", "[ref]: http://x\n", "\r\n", "# crlf head\r\n",
        " # sep\n", "head ing\n---\n", "***\n", "~~~\n## tilde\n~~~\n", "# ![[~ [a](b\n", "  ## indented head\n",
        "#\t tab head\n", "# \n", "##\n", "\x0b# vt\n", "# émoji 🚀 ☕\n",
    ]

    @staticmethod
    def via_parse(text):
        return [(b.level, b.text.splitlines()[0]) for b in parse(text) if b.kind == "heading"]

    def test_the_sample_document(self):
        self.assertEqual(heading_lines(DOC), self.via_parse(DOC))
        self.assertTrue(heading_lines(DOC))

    def test_edge_inputs(self):
        for text in ["", "\n", "   ", "# only", "﻿# bom", "a\rb\r# cr only\r", "Title\n=====", "# a\n# b\n"]:
            self.assertEqual(heading_lines(text), self.via_parse(text), repr(text))

    def test_random_documents_from_tricky_fragments(self):
        rng = random.Random(11)
        for _ in range(600):
            text = "".join(rng.choice(self.FRAGMENTS) for _ in range(rng.randint(1, 25)))
            self.assertEqual(heading_lines(text), self.via_parse(text), repr(text))

    def test_headings_in_quotes_lists_and_code_are_not_top_level_headings(self):
        self.assertEqual(heading_lines("> # q\n\n- # l\n\n```\n# c\n```\n\n# real\n"), [(1, "# real")])

    def test_it_does_not_run_the_inline_pass(self):
        from src.longform.markdown_ast import _block_parser, _parser

        def parse_tokens(text):
            return _parser().parse(text)

        tokens = _block_parser().parse("# Title with [a link](x) and *emphasis*\n")
        inline = [t for t in tokens if t.type == "inline"]
        self.assertTrue(inline and all(not t.children for t in inline))              # raw content kept, no child tokens
        normal = [t for t in parse_tokens("# Title with [a link](x) and *emphasis*\n") if t.type == "inline"]
        self.assertTrue(normal and all(t.children for t in normal))                  # while the normal parser does build them


if __name__ == "__main__":
    unittest.main()
