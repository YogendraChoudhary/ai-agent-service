"""
chunk: whole blocks only, about 1,000 words each, cut before headings, with the structure each chunk expects of its answer.
"""

import random
import unittest

from src.longform.chunk import HIGH, LOW, analyse, chunk_blocks
from src.longform.freeze import freeze_blocks, token_list, words_of
from src.longform.markdown_ast import parse
from tests.longform.fixtures import QUOTE, TABLE, acceptance_document, edge_documents, paragraph


def chunks_of(text, target=1000, first=0, **kw):
    frozen, vault = freeze_blocks(parse(text), **kw)
    return chunk_blocks(frozen, target, first), vault


def sectioned(sections=8, words=400, seed=1):
    rng = random.Random(seed)
    return "# Title\n\n" + "".join(f"## Section {i}\n\n{paragraph(rng, words // 2)}\n\n{paragraph(rng, words // 2)}\n\n" for i in range(sections))


class TestCutting(unittest.TestCase):

    def test_the_chunks_are_the_document_in_pieces(self):
        for name, text in {"acceptance": acceptance_document(), **edge_documents()}.items():
            with self.subTest(document=name):
                chunks, _ = chunks_of(text)
                self.assertEqual("".join(c.original for c in chunks), text)

    def test_every_chunk_ends_where_a_block_ends(self):
        text = acceptance_document()
        chunks, _ = chunks_of(text)
        offsets = set()
        position = 0
        for block in parse(text):
            position += len(block.text)
            offsets.add(position)
        position = 0
        for chunk in chunks:
            position += len(chunk.original)
            self.assertIn(position, offsets, f"chunk {chunk.index} ends inside a block")

    def test_a_five_thousand_word_document_is_a_handful_of_chunks_inside_the_window(self):
        chunks, _ = chunks_of(acceptance_document())
        self.assertTrue(4 <= len(chunks) <= 9, len(chunks))
        for chunk in chunks[:-1]:
            self.assertGreaterEqual(chunk.structure.words, 1000 * LOW * 0.6)
            self.assertLessEqual(chunk.structure.words, 1000 * HIGH)
        self.assertEqual(sum(c.structure.words for c in chunks), words_of("".join(c.frozen for c in chunks)))

    def test_a_section_heading_starts_a_chunk_once_the_current_one_has_600_words(self):
        chunks, _ = chunks_of(sectioned(sections=8, words=400))                         # 8 sections of ~400 words
        self.assertTrue(all(c.frozen.lstrip().startswith("#") for c in chunks), [c.frozen[:20] for c in chunks])
        self.assertTrue(all(abs(c.structure.words - 800) < 120 for c in chunks[:-1]), [c.structure.words for c in chunks])

    def test_short_sections_stay_together_until_the_target_then_wait_for_a_heading(self):
        chunks, _ = chunks_of(sectioned(sections=10, words=200))
        self.assertTrue(all(c.frozen.lstrip().startswith("#") for c in chunks))
        self.assertTrue(all(c.structure.words >= 600 for c in chunks[:-1]))

    def test_a_document_without_headings_is_cut_between_paragraphs_near_the_target(self):
        rng = random.Random(2)
        text = "".join(paragraph(rng, 150) + "\n\n" for _ in range(20))                  # 3,000 words, no headings
        chunks, _ = chunks_of(text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(900 <= c.structure.words <= 1400 for c in chunks), [c.structure.words for c in chunks])

    def test_one_giant_paragraph_is_a_chunk_by_itself(self):
        rng = random.Random(3)
        text = f"Short opening paragraph here.\n\n{paragraph(rng, 2000)}\n\nA closing paragraph, also short.\n"
        chunks, _ = chunks_of(text)
        self.assertEqual([c.structure.words > 1500 for c in chunks], [False, True, False][:len(chunks)] if len(chunks) == 3 else [True])
        self.assertEqual(len(chunks), 3)

    def test_a_sliver_at_the_end_joins_the_chunk_before_it(self):
        rng = random.Random(4)
        text = "".join(paragraph(rng, 300) + "\n\n" for _ in range(3)) + "# Tail\n\nJust a few closing words here.\n"
        chunks, _ = chunks_of(text, target=1000)
        self.assertEqual(len(chunks), 1)

    def test_a_small_document_is_one_chunk_and_an_empty_one_is_none(self):
        self.assertEqual(len(chunks_of("# T\n\nOne paragraph of text.\n")[0]), 1)
        self.assertEqual(chunks_of("")[0], [])

    def test_the_target_is_a_setting(self):
        text = sectioned(sections=8, words=400)
        self.assertGreater(len(chunks_of(text, target=500)[0]), len(chunks_of(text, target=1000)[0]))


class TestWhatEachChunkExpects(unittest.TestCase):

    def test_tokens_run_through_the_chunks_in_document_order(self):
        text = acceptance_document()
        chunks, vault = chunks_of(text)
        every = [t for c in chunks for t in c.tokens]
        whole = token_list("".join(c.frozen for c in chunks))
        self.assertEqual(every, whole)
        self.assertEqual(len([t for t in every if t.startswith("[[TABLE")]), 4)
        blocks = [t for t in every if t.startswith("[[")]
        self.assertEqual(len(blocks), len(set(blocks)))                                  # each token appears once in the document

    def test_structure_is_measured_on_the_frozen_text(self):
        text = "# One\n\nFirst para here.\n\n## Two\n\n- a\n  - b\n- c\n\n" + TABLE + "\n\nLast para of the chunk.\n"
        (chunk,), _ = chunks_of(text)
        self.assertEqual(chunk.structure.headings, (1, 2))
        self.assertEqual(chunk.structure.lists, ((0, 1, 0),))
        self.assertEqual(chunk.structure.paragraphs, 3)                                  # the table token is a paragraph too
        self.assertEqual(chunk.tokens, ("[[TABLE_1]]",))
        self.assertEqual(chunk.own_line, ("[[TABLE_1]]",))
        self.assertEqual(analyse(chunk.frozen), chunk.structure)

    def test_unfrozen_quotes_are_counted_as_structure(self):
        (chunk,), _ = chunks_of("Intro text.\n\n> A quote.\n\nOutro text.\n", freeze_quotes=False)
        self.assertEqual(dict(chunk.structure.kinds), {"blockquote": 1})
        self.assertEqual(chunk.tokens, ())

    def test_context_is_the_end_of_the_previous_chunk_and_the_next_heading(self):
        chunks, _ = chunks_of(sectioned(sections=6, words=400))
        self.assertEqual(chunks[0].before, "")
        self.assertTrue(chunks[1].before and len(chunks[1].before.split()) <= 60)
        self.assertTrue(chunks[0].after.startswith("## Section"))
        self.assertIsNone(chunks[-1].after)

    def test_a_chunk_of_only_tokens_has_nothing_to_refine(self):
        rng = random.Random(5)
        text = f"{paragraph(rng, 700)}\n\n## Data\n\n{TABLE}\n\n{TABLE}\n\n> quote\n"
        chunks, _ = chunks_of(text, target=400)
        self.assertEqual([c.refinable for c in chunks][-1], chunks[-1].structure.words > 0)
        only_tokens, _ = chunks_of(f"{TABLE}\n\n> quote\n")
        self.assertEqual(len(only_tokens), 1)
        self.assertFalse(only_tokens[0].refinable)


if __name__ == "__main__":
    unittest.main()


class TestASmallerFirstChunk(unittest.TestCase):
    """A chunk reaches the reader only when it is verified, so a short first one puts text on screen sooner."""

    def sizes(self, text, first, target=1000):
        return [c.structure.words for c in chunks_of(text, target, first)[0]]

    def test_the_first_chunk_aims_at_its_own_size_and_the_rest_are_unchanged(self):
        text = acceptance_document()
        plain, small = self.sizes(text, 0), self.sizes(text, 300)
        self.assertLess(small[0], plain[0])
        self.assertLessEqual(small[0], 300 * HIGH)
        self.assertGreaterEqual(small[0], 300 * LOW * 0.5)
        self.assertEqual(sum(small), sum(plain))                                          # nothing gained or lost
        self.assertTrue(all(w <= 1000 * HIGH for w in small[1:-1]), small)

    def test_the_document_is_still_all_there_in_order(self):
        for name, text in {"acceptance": acceptance_document(), "sections": sectioned(), **edge_documents()}.items():
            with self.subTest(document=name):
                chunks, _ = chunks_of(text, first=300)
                self.assertEqual("".join(c.original for c in chunks), text)
                self.assertEqual([t for c in chunks for t in c.tokens], token_list("".join(c.frozen for c in chunks)))

    def test_a_first_size_that_is_not_smaller_than_the_target_changes_nothing(self):
        text = acceptance_document()
        for first in (0, 1000, 1500, 5000):
            with self.subTest(first=first):
                self.assertEqual(self.sizes(text, first), self.sizes(text, 0))

    def test_a_chunk_still_ends_where_a_block_ends(self):
        text = acceptance_document()
        chunks, _ = chunks_of(text, first=250)
        offsets, position = set(), 0
        for block in parse(text):
            position += len(block.text)
            offsets.add(position)
        position = 0
        for chunk in chunks:
            position += len(chunk.original)
            self.assertIn(position, offsets)

    def test_a_document_barely_over_the_first_size_becomes_two_chunks_not_a_sliver(self):
        rng = random.Random(4)
        text = f"{paragraph(rng, 250)}\n\n{paragraph(rng, 250)}\n\n{paragraph(rng, 250)}\n"
        sizes = self.sizes(text, 300)
        self.assertEqual(len(sizes), 2)
        self.assertEqual(self.sizes(text, 0), [sum(sizes)])

    def test_the_context_of_the_second_chunk_is_the_end_of_the_small_first_one(self):
        chunks, _ = chunks_of(acceptance_document(), first=300)
        self.assertEqual(chunks[0].before, "")
        self.assertTrue(chunks[1].before)
        self.assertEqual(chunks[1].before.split()[-1], " ".join(chunks[0].frozen.split()).split()[-1])

    def test_a_tiny_first_size_cannot_make_a_chunk_of_nothing(self):
        for first in (1, 10, 49):
            with self.subTest(first=first):
                chunks, _ = chunks_of(acceptance_document(), first=first)
                self.assertTrue(all(c.original for c in chunks))


class TestTheSequenceOfBlocks(unittest.TestCase):

    def test_a_run_of_paragraphs_counts_once(self):
        text = "# Title\n\nOne.\n\nTwo.\n\nThree.\n\n- a\n- b\n\nFour.\n\n## Next\n\nFive.\n"
        self.assertEqual(analyse(text).sequence, ("heading", "paragraph", "list", "paragraph", "heading", "paragraph"))

    def test_a_locked_block_is_its_own_kind_in_the_sequence(self):
        (c,), _ = chunks_of(f"Intro words.\n\n{TABLE}\n\nOutro words.\n\n{QUOTE}\n\n```\ncode\n```\n")
        self.assertEqual(c.structure.sequence, ("paragraph", "table", "paragraph", "quote", "code"))

    def test_a_lead_in_changes_it_and_a_merge_does_not(self):
        base = analyse("## H\n\n- a\n- b\n\nEnd.\n").sequence
        self.assertNotEqual(analyse("## H\n\nHello there.\n\n- a\n- b\n\nEnd.\n").sequence, base)
        self.assertEqual(analyse("## H\n\n- a\n- b\n\nEnd. And more.\n").sequence, base)
