"""
freeze / thaw: what the model must not touch is out of its sight, and comes back byte for byte.
The property that matters: thaw(freeze(doc)) is the identity on every document.
"""

import random
import unittest

from src.longform.freeze import (
    TOKEN_RE, Vault, collides, freeze, own_line_tokens, strip_tokens, thaw, token_list, words_of,
)
from src.longform.markdown_ast import normalise, parse, restyle
from tests.longform.fixtures import (
    FENCE, FIGURE, HTML, NESTED_QUOTE, QUOTE, TABLE, acceptance_document, edge_documents, paragraph,
)


class TestRoundTrip(unittest.TestCase):

    def test_every_fixture_comes_back_identical(self):
        documents = {"acceptance": acceptance_document(), **edge_documents()}
        for name, text in documents.items():
            with self.subTest(document=name):
                frozen, vault = freeze(text)
                self.assertEqual(thaw(frozen, vault), text)

    def test_random_documents_built_from_every_kind_of_block_come_back_identical(self):
        rng = random.Random(11)
        pieces = [TABLE, QUOTE, NESTED_QUOTE, FIGURE, FENCE, HTML, "---", "# A heading", "## Another heading",
                  "- one\n- two\n  - nested [l](https://a.example/x)", "1. first\n2. second",
                  "Text with [a link](https://x.example \"T\"), a bare https://y.example/z. and ![i](https://i.example/a.png).",
                  "A plain paragraph with **bold** and `code` and an <https://auto.example> link.", "[ref]: https://r.example",
                  "Press <kbd>Ctrl</kbd>+<kbd>C</kbd>.<br>Mail <a@b.example> or c.d+e@f.example.org, a < b and <i>c</i> > d.",
                  "<!-- start --> text <img src=\"a.png\" alt='x y' /> and x<y and z>w and `<span>`."]
        for n in range(150):
            doc = "\n\n".join(rng.choice(pieces) for _ in range(rng.randint(1, 12))) + rng.choice(["\n", "\n\n", ""])
            for freeze_quotes in (True, False):
                with self.subTest(case=n, freeze_quotes=freeze_quotes):
                    frozen, vault = freeze(doc, freeze_quotes=freeze_quotes)
                    self.assertEqual(thaw(frozen, vault), doc)

    def test_a_model_that_returns_the_frozen_text_unchanged_gives_the_document_back(self):
        text = acceptance_document()
        frozen, vault = freeze(text)
        answer = "\n".join(line for line in frozen.split("\n"))
        self.assertEqual(thaw(answer, vault), text)

    def test_crlf_and_bom_survive_normalise_freeze_thaw_restyle(self):
        text = "﻿# T\r\n\r\n| a | b |\r\n|---|---|\r\n| 1 | 2 |\r\n\r\nPara.\r\n"
        lf, style = normalise(text)
        self.assertNotIn("\r", lf)
        frozen, vault = freeze(lf)
        self.assertEqual(restyle(thaw(frozen, vault), style, first=True), text)


class TestWhatIsFrozen(unittest.TestCase):

    def freeze(self, text, **kw):
        frozen, vault = freeze(text, **kw)
        return frozen, vault

    def test_the_acceptance_document_has_everything_the_plan_lists(self):
        frozen, vault = self.freeze(acceptance_document())
        counts = vault.summary()
        self.assertEqual((counts["TABLE"], counts["QUOTE"], counts["IMAGE"], counts["CODE"]), (4, 6, 6, 2))
        self.assertEqual(counts["LINK"] + counts["URL"], 10)                            # ten links of every kind
        for kind in ("TABLE", "QUOTE", "CODE"):                                          # none of it is left in the text
            for token, original in vault.blocks.items():
                if token.startswith(f"[[{kind}_"):
                    self.assertNotIn(original, frozen, token)
        self.assertNotIn("|---|", frozen)
        self.assertNotIn("https://", frozen)

    def test_each_block_becomes_one_token_on_its_own_line(self):
        text = f"Intro.\n\n{TABLE}\n\n{QUOTE}\n\n{FIGURE}\n\n{FENCE}\n\n{HTML}\n\n---\n\nEnd.\n"
        frozen, vault = self.freeze(text)
        self.assertEqual(token_list(frozen), ["[[TABLE_1]]", "[[QUOTE_1]]", "[[IMAGE_1]]", "[[CODE_1]]", "[[HTML_1]]", "[[HTML_2]]"])
        self.assertEqual(own_line_tokens(frozen), token_list(frozen))
        self.assertEqual(frozen, "Intro.\n\n[[TABLE_1]]\n\n[[QUOTE_1]]\n\n[[IMAGE_1]]\n\n[[CODE_1]]\n\n[[HTML_1]]\n\n[[HTML_2]]\n\nEnd.\n")
        self.assertEqual(vault.blocks["[[IMAGE_1]]"], FIGURE)                             # the caption goes with its picture

    def test_the_text_between_stays_in_place(self):
        frozen, _ = self.freeze("# Title\n\nFirst paragraph.\n\n- a\n- b\n\nLast one.\n")
        self.assertEqual(frozen, "# Title\n\nFirst paragraph.\n\n- a\n- b\n\nLast one.\n")

    def test_links_keep_their_anchor_and_lose_their_destination(self):
        frozen, vault = self.freeze('Read [the guide](https://a.example/g "Title") and [more][ref].\n\n[ref]: https://b.example\n')
        self.assertIn("[the guide](link://1)", frozen)
        self.assertIn("[more](link://2)", frozen)
        self.assertEqual(vault.links, {1: ("(", 'https://a.example/g "Title"'), 2: ("[", "ref")})
        self.assertEqual(thaw(frozen, vault), 'Read [the guide](https://a.example/g "Title") and [more][ref].\n\n[ref]: https://b.example\n')

    def test_a_linked_image_freezes_both_the_image_and_the_destination(self):
        frozen, vault = self.freeze("[![alt](https://i.example/a.png)](https://t.example)\n")
        self.assertEqual(frozen, "[[[IMAGE_1]]](link://1)\n")
        self.assertEqual(token_list(frozen), ["[[IMAGE_1]]", "link://1"])

    def test_bare_urls_and_autolinks_become_tokens_without_their_punctuation(self):
        frozen, vault = self.freeze("See https://a.example/x, then (https://b.example/y) or www.c.example/z. Also <https://d.example>.\n")
        self.assertEqual(frozen, "See [[URL_1]], then ([[URL_2]]) or [[URL_3]]. Also [[URL_4]].\n")
        self.assertEqual(vault.blocks["[[URL_1]]"], "https://a.example/x")
        self.assertEqual(vault.blocks["[[URL_4]]"], "<https://d.example>")

    def test_inline_html_tags_and_comments_become_tokens_in_place(self):
        frozen, vault = self.freeze("Press <kbd>Ctrl</kbd>+<kbd>C</kbd>.<br>Next <img src=\"a.png\" alt='x y' /> <!-- note --> end.\n")
        self.assertEqual(frozen, "Press [[HTML_1]]Ctrl[[HTML_2]]+[[HTML_3]]C[[HTML_4]].[[HTML_5]]Next [[HTML_6]] [[HTML_7]] end.\n")
        self.assertEqual(vault.blocks["[[HTML_6]]"], "<img src=\"a.png\" alt='x y' />")
        self.assertEqual(vault.blocks["[[HTML_7]]"], "<!-- note -->")
        self.assertEqual(own_line_tokens(frozen), [])                                    # inline: they may sit in a sentence
        self.assertEqual(words_of(frozen), 5)                                            # Press Ctrl C Next end: the tags are not words

    def test_text_that_is_not_a_tag_stays_text(self):
        for text in ("a < b and c > d.\n", "3 <5 and 7> 2\n", "<3 hearts\n", "if a <- b then\n", "x -> y <-> z\n"):
            with self.subTest(text=text):
                self.assertEqual(self.freeze(text)[0], text)

    def test_a_tag_with_attributes_across_lines_is_one_token(self):
        frozen, _ = self.freeze('Look <a href="https://q.example/x?a=1&b=2"\n   target="_blank">here</a> now.\n')
        self.assertEqual(frozen, "Look [[HTML_1]]here[[HTML_2]] now.\n")

    def test_email_addresses_are_tokens_and_handles_are_not(self):
        frozen, vault = self.freeze("Mail <alice@example.com> or bob.smith+tag@mail.example.co.uk, not @handle or x@y.\n")
        self.assertEqual(frozen, "Mail [[URL_1]] or [[URL_2]], not @handle or x@y.\n")
        self.assertEqual(vault.blocks["[[URL_1]]"], "<alice@example.com>")

    def test_an_inline_scan_of_pathological_input_stays_fast(self):
        import time
        for text in ("<a" + " b" * 5000 + "\n", "<a " + " " * 20000 + "x\n", "<!--" + "-" * 20000 + "\n", "<" * 20000 + "\n",
                     "[" * 3000 + "\n", "a@" + "b." * 5000 + "\n"):
            started = time.perf_counter()
            frozen, vault = self.freeze(text)
            self.assertLess(time.perf_counter() - started, 1.0)
            self.assertEqual(thaw(frozen, vault), text)

    def test_an_inline_image_is_frozen_inside_its_sentence(self):
        frozen, _ = self.freeze("Before ![icon](https://i.example/i.png \"T\") after.\n")
        self.assertEqual(frozen, "Before [[IMAGE_1]] after.\n")
        self.assertEqual(own_line_tokens(frozen), [])

    def test_code_inside_a_list_item_is_frozen_with_its_indent(self):
        frozen, vault = self.freeze("- item\n  ```sh\n  echo hi\n  ```\n- next\n")
        self.assertEqual(frozen, "- item\n  [[CODE_1]]\n- next\n")
        self.assertEqual(vault.blocks["[[CODE_1]]"], "```sh\n  echo hi\n  ```")

    def test_quotes_can_be_left_for_polishing_and_are_then_prose(self):
        frozen, vault = self.freeze("Text.\n\n> A quote with [a link](https://q.example).\n", freeze_quotes=False)
        self.assertEqual(frozen, "Text.\n\n> A quote with [a link](link://1).\n")
        self.assertNotIn("[[QUOTE", frozen)

    def test_link_reference_definitions_are_opaque(self):
        frozen, vault = self.freeze("Text [x][r].\n\n[r]: https://r.example \"Title\"\n")
        self.assertEqual(token_list(frozen), ["link://1", "[[HTML_1]]"])

    def test_numbering_runs_through_the_whole_document(self):
        frozen, _ = self.freeze(f"{TABLE}\n\nText.\n\n{TABLE}\n\n{TABLE}\n")
        self.assertEqual(token_list(frozen), ["[[TABLE_1]]", "[[TABLE_2]]", "[[TABLE_3]]"])

    def test_words_are_counted_without_the_tokens(self):
        frozen, _ = self.freeze("Three plain words.\n\n" + TABLE + "\n\nSee [link text](https://x.example) now.\n")
        self.assertEqual(words_of(frozen), 3 + 4)
        self.assertEqual(strip_tokens("a [[TABLE_1]] b [x](link://3)").split(), ["a", "b", "[x]"])


class TestThaw(unittest.TestCase):

    def test_an_unknown_token_is_left_as_written(self):
        self.assertEqual(thaw("a [[TABLE_9]] b [x](link://7)", Vault()), "a [[TABLE_9]] b [x](link://7)")

    def test_tokens_are_replaced_once_and_what_comes_back_is_not_scanned_again(self):
        vault = Vault()
        token = vault.block("QUOTE", "> literally [[TABLE_1]] inside")
        self.assertEqual(thaw(f"x\n{token}\n", vault), "x\n> literally [[TABLE_1]] inside\n")

    def test_the_final_line_break_belongs_to_the_text_around_the_token(self):
        frozen, vault = freeze("a\n\n> q\n> r")
        self.assertEqual(frozen, "a\n\n[[QUOTE_1]]")
        self.assertEqual(thaw(frozen, vault), "a\n\n> q\n> r")


class TestCollision(unittest.TestCase):

    def test_a_document_that_already_looks_like_it_has_tokens_is_flagged(self):
        for text in ("see [[TABLE_1]]", "[[URL_12]]", "a link://3 b"):
            with self.subTest(text=text):
                self.assertTrue(collides(text))
        for text in ("[[wiki link]]", "[[TABLE]]", "plain", "TABLE_1", "[x](https://link.example)"):
            with self.subTest(text=text):
                self.assertFalse(collides(text))

    def test_the_token_pattern_matches_only_real_tokens(self):
        self.assertEqual(TOKEN_RE.findall("[[TABLE_1]] [[QUOTE_22]] [[nope_1]] [[IMAGE_x]] [[HTML_3]]"),
                         ["[[TABLE_1]]", "[[QUOTE_22]]", "[[HTML_3]]"])


class TestParsingAssumptions(unittest.TestCase):
    """The blocks markdown-it reports are what freeze works from: these are the cases the plan's fixtures name."""

    def test_a_giant_paragraph_is_one_block(self):
        rng = random.Random(1)
        self.assertEqual(len(parse(paragraph(rng, 2000) + "\n")), 1)

    def test_nothing_is_lost_between_blocks(self):
        for name, text in edge_documents().items():
            with self.subTest(document=name):
                self.assertEqual("".join(b.text for b in parse(text)), text)


if __name__ == "__main__":
    unittest.main()
