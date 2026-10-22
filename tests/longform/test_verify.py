"""
verify: every way a model can damage a passage is caught by code, and what it should be allowed to do is not.
The bad outputs are scripted: no model is involved.
"""

import random
import re
import unittest

from src.longform.freeze import thaw
from src.longform.verify import Issue, repair_message, snap_blocks, unwrap, verify, with_edges
from tests.longform.fixtures import FIGURE, QUOTE, TABLE, paragraph
from tests.longform.test_chunk import chunks_of


def document(seed=1):
    rng = random.Random(seed)
    return (f"# A document title\n\n{paragraph(rng, 60)}\n\n"
            f"## First section\n\n{paragraph(rng, 70)} See [the guide](https://example.com/guide) or https://example.org/x for more.\n\n"
            f"{TABLE}\n\n{paragraph(rng, 50)}\n\n{QUOTE}\n\n"
            f"## Second section\n\n- {paragraph(rng, 12)}\n  - {paragraph(rng, 8)}\n- {paragraph(rng, 10)}\n\n{FIGURE}\n\n"
            f"{paragraph(rng, 60)}\n\n{paragraph(rng, 50)}\n\n{paragraph(rng, 45)}\n")


def chunk(text=None):
    (c,) = chunks_of(text or document())[0]
    return c


class Base(unittest.TestCase):

    def setUp(self):
        self.chunk = chunk()
        self.frozen = self.chunk.frozen

    def kinds(self, output, **kw):
        return {i.kind for i in verify(self.chunk, output, **kw).issues}

    def replace_line(self, old, new):
        self.assertIn(old, self.frozen)
        return self.frozen.replace(old, new, 1)


class TestWhatPasses(Base):

    def test_the_input_itself_passes(self):
        verdict = verify(self.chunk, self.frozen)
        self.assertTrue(verdict.ok, verdict.issues)
        self.assertEqual(verdict.text, self.frozen)
        self.assertAlmostEqual(verdict.ratio, 1.0)

    def test_reworded_prose_passes(self):
        reworded = self.frozen.replace("Most ", "Nearly all ", 3).replace(" and ", " plus ", 4)
        self.assertNotEqual(reworded, self.frozen)
        self.assertTrue(verify(self.chunk, reworded).ok)

    def test_a_reworded_link_anchor_passes_and_a_changed_destination_does_not(self):
        self.assertTrue(verify(self.chunk, self.frozen.replace("[the guide](link://1)", "[our complete guide](link://1)")).ok)
        self.assertIn("tokens_missing", self.kinds(self.frozen.replace("(link://1)", "(https://elsewhere.example)")))

    def test_the_answer_comes_back_with_the_chunks_own_edges(self):
        verdict = verify(self.chunk, "\n\n" + self.frozen.strip() + "\n\n\n\n")
        self.assertEqual(verdict.text, self.frozen)
        self.assertEqual(with_edges("core", self.chunk), "core" + self.frozen[len(self.frozen.rstrip()):])

    def test_wrappers_are_removed_in_place(self):
        for wrapped in (f"Here is the corrected text:\n\n{self.frozen}", f"```markdown\n{self.frozen.strip()}\n```",
                        f"<<<PASSAGE\n{self.frozen.strip()}\nPASSAGE>>>", f"Sure! Here's the polished version:\n{self.frozen}\n"):
            with self.subTest(wrapped=wrapped[:30]):
                self.assertEqual(verify(self.chunk, wrapped).text, self.frozen)

    def test_unwrap_leaves_a_real_first_line_alone(self):
        self.assertEqual(unwrap("Here is how it works: a simple idea.\n\nNext."), "Here is how it works: a simple idea.\n\nNext.")
        self.assertEqual(unwrap("Here is the result:\n\nBody"), "Body")
        self.assertEqual(unwrap("Here is the passage:\n\nBody", original="Here is the passage:\n\nBody"), "Here is the passage:\n\nBody")
        self.assertEqual(unwrap("```\ncode\n```"), "code")
        self.assertEqual(unwrap("```\ncode\n```", original="```\ncode\n```\n"), "```\ncode\n```")


class TestTokens(Base):

    def test_a_dropped_token_is_named(self):
        verdict = verify(self.chunk, self.replace_line("[[TABLE_1]]\n", ""))
        self.assertIn("tokens_missing", {i.kind for i in verdict.issues})
        self.assertTrue(any("[[TABLE_1]]" in i.message for i in verdict.issues))

    def test_a_duplicated_or_invented_token_is_caught(self):
        self.assertIn("tokens_extra", self.kinds(self.frozen + "\n\n[[QUOTE_1]]\n"))
        self.assertIn("tokens_extra", self.kinds(self.replace_line("[[QUOTE_1]]", "[[QUOTE_1]]\n\n[[QUOTE_9]]")))

    def test_reordered_tokens_are_caught(self):
        swapped = self.frozen.replace("[[TABLE_1]]", "[[@]]").replace("[[QUOTE_1]]", "[[TABLE_1]]").replace("[[@]]", "[[QUOTE_1]]")
        self.assertEqual(self.kinds(swapped), {"tokens_order", "order"})                 # a table where the quote was, and back

    def test_a_block_token_glued_into_a_paragraph_is_caught(self):
        glued = self.frozen.replace("\n\n[[TABLE_1]]\n\n", " [[TABLE_1]] ")
        self.assertIn("token_inline", self.kinds(glued))

    def test_an_edited_link_reference_is_caught(self):
        self.assertIn("link_form", self.kinds(self.frozen.replace("(link://1)", '(link://1 "title")')))
        self.assertIn("link_form", self.kinds(self.frozen.replace("](link://1)", "] link://1")))

    def test_a_token_that_was_inline_may_move_inside_its_sentence(self):
        text = "Start with ![icon](https://i.example/i.png) here and go on for a while with some more words around it.\n"
        c = chunk(text)
        moved = c.frozen.replace("Start with [[IMAGE_1]] here and go on", "Start here with [[IMAGE_1]] and go on")
        self.assertTrue(verify(c, moved).ok)


class TestStructure(Base):

    def test_merged_paragraphs_are_caught_for_grammar_but_one_merge_is_tolerated_elsewhere(self):
        parts = self.frozen.split("\n\n")
        prose = [i for i, p in enumerate(parts) if len(p.split()) > 40]
        i = next(i for i in prose if i + 1 in prose)                                     # two paragraphs side by side
        merged = "\n\n".join(parts[:i] + [parts[i] + " " + parts[i + 1]] + parts[i + 2:])
        self.assertEqual(self.kinds(merged, paragraphs="equal"), {"paragraphs"})
        self.assertEqual(self.kinds(merged, paragraphs="within"), set())
        many = re.sub(r"(?<=[a-z.])\n\n(?=[A-Z])", " ", self.frozen)
        self.assertIn("paragraphs", self.kinds(many, paragraphs="within"))

    def test_a_changed_heading_level_or_count_is_caught(self):
        self.assertEqual(self.kinds(self.replace_line("## First section", "### First section")), {"headings"})
        self.assertEqual(self.kinds(self.replace_line("## Second section\n\n", "")), {"headings", "order"})
        self.assertEqual(self.kinds(self.frozen + "\n## An added heading\n"), {"headings", "order"})

    def test_a_lead_in_paragraph_under_a_heading_is_caught_as_a_change_of_order(self):
        """Seen live: a tone change added "Hey there! Let's walk through ..." between a heading and its list."""
        self.assertIn("\n## Second section\n\n- ", self.frozen)
        lead_in = self.replace_line("## Second section\n\n- ", "## Second section\n\nHey there! Let's look at what comes next.\n\n- ")
        found = {i.kind: i for i in verify(self.chunk, lead_in).issues}
        self.assertEqual(set(found), {"order"})
        self.assertIn("heading > paragraph > list", found["order"].message)
        self.assertIn("lead-in paragraph", found["order"].message)

    def test_a_paragraph_dropped_before_a_list_or_a_new_one_before_the_title_is_caught(self):
        self.assertIn("order", self.kinds("A new opening line the model wrote.\n\n" + self.frozen))
        self.assertIn("order", self.kinds(self.frozen.rstrip("\n") + "\n\nHope this helps!\n") | self.kinds(self.frozen + "\n- one more\n"))

    def test_a_locked_block_moved_past_a_paragraph_is_caught(self):
        parts = self.frozen.split("\n\n")
        at = next(i for i, p in enumerate(parts) if p.strip() == "[[TABLE_1]]")
        self.assertGreater(len(parts[at + 1].split()), 20)                                 # a paragraph follows the table
        moved = "\n\n".join(parts[:at] + [parts[at + 1], parts[at]] + parts[at + 2:])
        self.assertIn("order", self.kinds(moved))

    def test_merging_and_splitting_paragraphs_does_not_move_anything(self):
        parts = self.frozen.split("\n\n")
        prose = [i for i, p in enumerate(parts) if len(p.split()) > 40]
        i = next(i for i in prose if i + 1 in prose)
        merged = "\n\n".join(parts[:i] + [parts[i] + " " + parts[i + 1]] + parts[i + 2:])
        self.assertNotIn("order", self.kinds(merged))
        words = parts[i].split(" ")
        split = "\n\n".join(parts[:i] + [" ".join(words[:20]), " ".join(words[20:])] + parts[i + 1:])
        self.assertNotIn("order", self.kinds(split))

    def test_a_changed_list_is_caught(self):
        lines = self.frozen.splitlines()
        item = next(i for i, ln in enumerate(lines) if ln.startswith("- "))
        self.assertIn("lists", self.kinds("\n".join(lines[:item] + lines[item + 1:]) + "\n"))
        flat = self.frozen.replace("  - ", "- ")
        self.assertIn("lists", self.kinds(flat))

    def test_a_list_may_change_its_markers(self):
        self.assertTrue(verify(self.chunk, self.frozen.replace("\n- ", "\n* ").replace("\n  - ", "\n  * ")).ok)

    def test_an_added_table_quote_image_or_code_block_is_caught(self):
        for added in (TABLE, "> A quote the model wrote.", "![new](https://x.example/n.png)\n*caption*", "```\ncode\n```", "<div>x</div>"):
            with self.subTest(added=added[:20]):
                self.assertIn("structure", self.kinds(self.frozen + "\n\n" + added + "\n"))

    def test_a_rule_the_model_adds_is_a_structure_change(self):
        self.assertIn("structure", self.kinds(self.replace_line("## Second section", "---\n\n## Second section")))

    def test_unfrozen_quotes_must_keep_their_count(self):
        text = "Intro words here for the passage.\n\n> A quote the editor wrote.\n\nOutro words here for the passage.\n"
        (c,), _ = chunks_of(text, freeze_quotes=False)
        self.assertTrue(verify(c, c.frozen.replace("editor wrote", "author wrote")).ok)
        self.assertIn("structure", {i.kind for i in verify(c, c.frozen.replace("> A quote the editor wrote.", "A quote.")).issues})


class TestLength(Base):

    def trimmed(self, share):
        """The prose paragraphs cut to `share` of their words."""
        out = []
        for part in self.frozen.split("\n\n"):
            words = part.split(" ")
            plain = len(words) > 30 and not part.startswith(("[[", "|", ">", "-", "#", "![", "*"))
            out.append(" ".join(words[:max(1, int(len(words) * share))]) if plain else part)
        return "\n\n".join(out)

    def test_forty_percent_of_the_words_is_caught_for_every_operation(self):
        for ratio in ((0.85, 1.25), (0.6, 1.1)):
            self.assertIn("length", self.kinds(self.trimmed(0.4), ratio=ratio))

    def test_the_windows_differ_by_operation(self):
        shorter = self.trimmed(0.75)
        self.assertIn("length", self.kinds(shorter, ratio=(0.85, 1.25)))                 # grammar / enhance / tone
        self.assertNotIn("length", self.kinds(shorter, ratio=(0.6, 1.1)))                # simplify may shrink

    def test_longer_than_the_window_is_caught(self):
        longer = self.frozen.replace(". ", ". Furthermore, this additional sentence says something extra here. ")
        self.assertIn("length", self.kinds(longer))

    def test_a_short_passage_may_move_a_few_words(self):
        c = chunk("Just four short words.\n")
        self.assertTrue(verify(c, "Just five short plain words.\n").ok)
        self.assertIn("length", {i.kind for i in verify(c, "Just five short plain words and then ten more words after those.\n").issues})

    def test_the_message_says_what_the_window_was(self):
        message = next(i.message for i in verify(self.chunk, self.trimmed(0.4)).issues if i.kind == "length")
        self.assertRegex(message, r"has \d+ words; the passage has \d+ and the answer must have between \d+ and \d+")


class TestRepairMessage(unittest.TestCase):

    def test_numbered_errors_and_one_instruction(self):
        text = repair_message([Issue("tokens_missing", "[[TABLE_2]] is missing."), Issue("length", "412 words; the passage has 980.")])
        self.assertIn("1. [tokens_missing] [[TABLE_2]] is missing.", text)
        self.assertIn("2. [length] 412 words; the passage has 980.", text)
        self.assertTrue(text.rstrip().endswith("Output only the passage."))


if __name__ == "__main__":
    unittest.main()


class TestTokensGoBackToTheirOwnLine(Base):
    """The whitespace around a block token is not the model's: a stray space changes the bytes of the table that comes back."""

    def token_lines(self):
        return [ln for ln in self.frozen.split("\n") if re.fullmatch(r"\[\[(?:TABLE|QUOTE|IMAGE)_\d+\]\]", ln.strip())]

    def test_trailing_spaces_after_a_token_are_taken_off(self):
        for tail in ("  ", " ", "\t", "   \t "):
            with self.subTest(tail=repr(tail)):
                output = self.frozen
                for line in self.token_lines():
                    output = output.replace(line + "\n", line + tail + "\n")
                self.assertNotEqual(output, self.frozen)
                verdict = verify(self.chunk, output)
                self.assertTrue(verdict.ok, verdict.issues)
                self.assertEqual(verdict.text, self.frozen)

    def test_leading_spaces_are_replaced_by_the_indent_the_token_had(self):
        output = self.frozen
        for line in self.token_lines():
            output = output.replace("\n" + line + "\n", "\n   " + line + "\n")
        self.assertEqual(verify(self.chunk, output).text, self.frozen)

    def test_a_dropped_blank_line_is_put_back_so_nothing_is_glued_to_a_quote_or_a_table(self):
        output = re.sub(r"\n\n(?=\[\[(?:TABLE|QUOTE)_\d+\]\])|(?<=\[\[TABLE_1\]\]|\[\[QUOTE_1\]\])\n\n", "\n", self.frozen)
        self.assertNotEqual(output, self.frozen)
        verdict = verify(self.chunk, output)
        self.assertTrue(verdict.ok, verdict.issues)
        self.assertEqual(verdict.text, self.frozen)

    def test_the_thawed_document_is_then_the_original_to_the_byte(self):
        text = document()
        (c,), vault = chunks_of(text)
        glued = c.frozen
        for line in self.token_lines():
            glued = glued.replace(line + "\n", line + "  \n")
        self.assertEqual(thaw(verify(c, glued).text, vault), text)

    def test_extra_blank_lines_and_tokens_in_a_sentence_are_not_touched(self):
        chunk_text = "Intro with an icon ![i](https://i.example/i.png) inside.\n\n" + TABLE + "\n\nOutro.\n"
        (c,), _ = chunks_of(chunk_text)
        inline = c.frozen
        self.assertEqual(snap_blocks(inline, c), inline)
        spaced = inline.replace("\n\n[[TABLE_1]]\n\n", "\n\n\n[[TABLE_1]]\n\n\n")
        self.assertIn("\n\n\n[[TABLE_1]]\n\n\n", spaced)
        self.assertEqual(snap_blocks(spaced, c), spaced)                                   # nothing is removed
        moved_inline = inline.replace("Intro with an icon [[IMAGE_1]] inside.", "Intro with an icon\n[[IMAGE_1]]\ninside.")
        self.assertEqual(snap_blocks(moved_inline, c), moved_inline)                       # an inline token has no line to go back to

    def test_a_token_inside_a_list_item_keeps_its_indent(self):
        text = "- first item\n  ```sh\n  echo hi\n  ```\n- second item\n" + "word " * 700 + "\n"
        (c,), _ = chunks_of(text)
        self.assertIn("  [[CODE_1]]", c.frozen)
        answer = c.frozen.replace("  [[CODE_1]]\n", "[[CODE_1]]   \n")
        self.assertEqual(verify(c, answer).text, c.frozen)

    def test_a_chunk_without_block_tokens_is_returned_as_it_is(self):
        (c,), _ = chunks_of("Just a paragraph of plain words here.\n\nAnd another one right after it.\n")
        answer = "Just  a paragraph.   \n\nAnd another.\n"
        self.assertEqual(snap_blocks(answer.strip("\n"), c), answer.strip("\n"))
