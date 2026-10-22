"""
The refinement pipeline end to end with a scripted model: the document comes back with every table, quote, image, code block
and link byte for byte in place, a chunk the model damages is restored as written, and nothing is lost on any path.
"""

import random
import re
import time
import unittest

from src.longform.markdown_ast import parse, word_count
from src.longform.prompts import OPERATIONS
from src.longform.refine import RefineConfig, qualifies, refine_stream
from src.observability import finish_trace, get_trace, start_trace
from tests.longform.fakes import (
    ScriptedRefiner, add_a_lead_in, add_a_table, drop_a_token, fix_on_repair, glue_to_the_blocks, merge_paragraphs, passage_of,
    polish, space_the_tokens, summarise, wrap,
)
from tests.longform.fixtures import QUOTE, TABLE, acceptance_document, edge_documents, paragraph

GRAMMAR, TONE, ENHANCE, SIMPLIFY = (OPERATIONS[n] for n in ("grammar", "tone", "enhance", "simplify"))
PROTECTED = ("table", "blockquote", "image", "fence", "html")
URL = re.compile(r"https?://[^\s)>\]\"']+")


def run(text, llm=None, operation=GRAMMAR, instruction=None, config=None, request_id=None):
    llm = llm or ScriptedRefiner()
    out = list(refine_stream(text, instruction, operation, llm, config or RefineConfig(), request_id=request_id))
    content = [x for x in out if isinstance(x, str)]
    events = [x["__event__"] for x in out if isinstance(x, dict) and "__event__" in x]
    warnings = next((x["__warnings__"] for x in out if isinstance(x, dict) and "__warnings__" in x), [])
    return "".join(content), content, events, warnings, llm


def protected(text):
    return [b.text for b in parse(text) if b.kind in PROTECTED]


def inline_urls(text):
    return sorted(URL.findall(text))


class TestTheDocumentComesBack(unittest.TestCase):

    def test_an_identity_model_returns_every_fixture_byte_for_byte(self):
        for name, text in {"acceptance": acceptance_document(), **edge_documents()}.items():
            if not qualifies(text) or not text.strip():
                continue
            with self.subTest(document=name):
                self.assertEqual(run(text)[0], text)

    def test_polished_prose_leaves_every_protected_block_and_url_untouched(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, ScriptedRefiner(polish))
        self.assertNotEqual(out, text)                                                       # the prose was changed
        self.assertEqual(protected(out), protected(text))                                    # byte for byte, in place
        self.assertEqual(inline_urls(out), inline_urls(text))
        self.assertEqual([b.kind for b in parse(out) if b.kind != "blank"], [b.kind for b in parse(text) if b.kind != "blank"])
        self.assertEqual(warnings, [])
        self.assertEqual(len(llm.calls), len(content))                                       # one call per chunk, no repairs

    def test_the_model_never_sees_what_is_frozen(self):
        seen = ScriptedRefiner()
        run(acceptance_document(), seen)
        sent = "\n".join(m[1][0]["content"] for m in seen.calls)
        for forbidden in ("|---|", "https://", "http://", "```", "> A pull quote", "<div"):
            self.assertNotIn(forbidden, sent)
        self.assertRegex(sent, r"\[\[TABLE_1\]\]")
        self.assertRegex(sent, r"\[anchor|\]\(link://\d+\)")

    def test_chunks_are_emitted_in_document_order_even_when_a_later_one_finishes_first(self):
        text = acceptance_document()
        out, content, *_ = run(text, ScriptedRefiner(polish, delays={"c1": 0.3}))
        self.assertEqual(protected(out), protected(text))
        self.assertTrue(content[0].lstrip().startswith("# A long working document"))

    def test_progress_follows_each_chunk(self):
        _, content, events, *_ = run(acceptance_document())
        self.assertEqual(events, [{"type": "refine_progress", "chunk": i, "of": len(content)} for i in range(1, len(content) + 1)])

    def test_the_stream_interleaves_chunk_then_progress(self):
        out = list(refine_stream(acceptance_document(), None, GRAMMAR, ScriptedRefiner()))
        kinds = ["text" if isinstance(x, str) else "progress" for x in out]
        self.assertEqual(kinds, ["text", "progress"] * (len(kinds) // 2))

    def test_line_endings_and_the_byte_order_mark_are_put_back(self):
        text = "﻿" + acceptance_document().replace("\n", "\r\n")
        out, content, *_ = run(text, ScriptedRefiner(polish))
        self.assertTrue(out.startswith("﻿# A long working document\r\n"))
        self.assertEqual(out.count("﻿"), 1)
        self.assertNotRegex(out, r"(?<!\r)\n")                                               # every line break is CRLF
        self.assertEqual(protected(out.replace("\r\n", "\n").lstrip("﻿")), protected(acceptance_document()))


class TestWhenTheModelMisbehaves(unittest.TestCase):

    def test_a_chunk_that_loses_a_token_is_repaired_when_the_model_can_fix_it(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, ScriptedRefiner(polish, {"c2": fix_on_repair(drop_a_token)}))
        self.assertEqual(warnings, [])
        self.assertEqual(protected(out), protected(text))
        self.assertEqual(sorted(s for s, u in llm.stages() if u == "c2"), ["chunk", "chunk_repair"])
        self.assertEqual(sum(1 for s, _ in llm.stages() if s == "chunk_repair"), 1)

    def test_a_chunk_that_stays_wrong_is_restored_exactly_and_reported(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, ScriptedRefiner(polish, {"c3": drop_a_token}))
        chunks = list(refine_stream(text, None, GRAMMAR, ScriptedRefiner(), config=RefineConfig()))
        self.assertEqual([w["chunk"] for w in warnings], [3])
        self.assertEqual(warnings[0]["kind"], "chunk_restored")
        self.assertIn("tokens_missing", warnings[0]["detail"])
        self.assertEqual(protected(out), protected(text))
        self.assertEqual(len(content), len(events))
        original_3 = "".join(c for c in [x for x in chunks if isinstance(x, str)][2:3])
        self.assertIn(original_3, out)                                                       # chunk 3 is the input's own text
        self.assertGreater(out.count("Nearly all") + out.count(" plus "), 0)                 # the others were refined

    def test_a_model_that_summarises_cannot_shrink_the_document(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, ScriptedRefiner(summarise))
        self.assertEqual(out, text)                                                          # every chunk restored
        self.assertEqual(len(warnings), len(content))
        self.assertEqual(word_count(out), word_count(text))

    def test_each_kind_of_damage_ends_in_a_restored_chunk_never_a_damaged_document(self):
        text = acceptance_document()
        for name, behaviour in (("merged paragraphs", merge_paragraphs), ("added table", add_a_table), ("summary", summarise),
                                ("dropped token", drop_a_token)):
            with self.subTest(damage=name):
                out, _, _, warnings, _ = run(text, ScriptedRefiner(behaviour), operation=GRAMMAR)
                self.assertEqual(protected(out), protected(text))
                self.assertEqual(word_count(out), word_count(text))
                self.assertTrue(warnings)

    def test_wrappers_are_removed_without_a_repair(self):
        text = acceptance_document()
        out, _, _, warnings, llm = run(text, ScriptedRefiner(wrap))
        self.assertEqual(out, text)
        self.assertEqual(warnings, [])
        self.assertEqual({s for s, _ in llm.stages()}, {"chunk"})

    def test_with_repair_switched_off_a_bad_chunk_is_restored_at_once(self):
        text = acceptance_document()
        out, _, _, warnings, llm = run(text, ScriptedRefiner(drop_a_token), config=RefineConfig(repair=False))
        self.assertEqual(out, text)
        self.assertTrue(warnings)
        self.assertEqual({s for s, _ in llm.stages()}, {"chunk"})

    def test_a_provider_error_is_raised_where_the_chunk_is_and_nothing_after_it_is_spent(self):
        def down(passage, spec):
            raise RuntimeError("503 overloaded")
        llm = ScriptedRefiner(polish, {"c2": down})
        got = []
        with self.assertRaisesRegex(RuntimeError, "503"):
            for item in refine_stream(acceptance_document(), None, GRAMMAR, llm, RefineConfig(parallel=1)):
                got.append(item)
        self.assertEqual(len([x for x in got if isinstance(x, str)]), 1)                     # chunk 1 had already gone out
        time.sleep(0.1)
        self.assertEqual({u for s, u in llm.stages()}, {"c1", "c2"})                         # c3.. never started

    def test_a_chunk_with_nothing_to_refine_costs_no_call(self):
        only_tables = f"{TABLE}\n\n{QUOTE}\n\n{TABLE}\n"
        out, content, events, warnings, llm = run(only_tables)
        self.assertEqual((out, llm.calls, warnings), (only_tables, [], []))
        self.assertEqual(events, [{"type": "refine_progress", "chunk": 1, "of": 1}])

    def test_tokens_in_front_of_a_giant_paragraph_are_a_passthrough_chunk_of_their_own(self):
        text = f"{TABLE}\n\n{paragraph(random.Random(1), 2000)}\n"
        out, content, events, warnings, llm = run(text, ScriptedRefiner(polish))
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], f"{TABLE}\n\n")
        self.assertEqual([u for _, u in llm.stages()], ["c2"])                               # only the paragraph was sent
        self.assertEqual(protected(out), protected(text))


class TestWhatTheModelIsAsked(unittest.TestCase):

    def test_system_prompt_is_the_operations_and_user_message_is_data(self):
        llm = ScriptedRefiner()
        run(acceptance_document(), llm, operation=TONE, instruction="friendly and warm")
        system, messages, spec = llm.calls[0]
        self.assertIn("Your task is to rewrite text in a different tone or style.", system)
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertIn("TONE: friendly and warm", content)
        self.assertTrue(content.rstrip().endswith("PASSAGE>>>"))
        self.assertNotIn("friendly and warm", system)                                         # request data stays out of the prompt
        self.assertEqual((spec.stage, spec.unit, spec.temperature), ("chunk", "c1", TONE.temperature))
        self.assertGreater(spec.max_tokens, 1500)

    def test_every_chunk_gets_the_same_system_prompt(self):
        llm = ScriptedRefiner()
        run(acceptance_document(), llm, operation=ENHANCE)
        self.assertEqual(len({c[0] for c in llm.calls}), 1)

    def test_the_repair_is_the_same_conversation_plus_the_numbered_errors_at_temperature_zero(self):
        llm = ScriptedRefiner(drop_a_token)
        run(acceptance_document(), llm)
        system, messages, spec = next(c for c in llm.calls if c[2].stage == "chunk_repair")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertIn("1. [tokens_missing]", messages[-1]["content"])
        self.assertEqual(spec.temperature, 0.0)

    def test_context_lines_show_the_neighbouring_text_but_are_marked_as_context(self):
        llm = ScriptedRefiner()
        run(acceptance_document(), llm)
        content = llm.calls[1][1][0]["content"]
        self.assertIn("CONTEXT, for continuity only (never output it):", content)
        self.assertIn("The passage before this one ends:", content)

    def test_the_ratio_window_can_be_overridden_by_json(self):
        text = acceptance_document()
        shorter = lambda passage, spec: passage.replace(" holds.", ".").replace("; and then it", "")
        strict = RefineConfig(word_ratio='{"grammar": [0.999, 1.001]}')
        out, _, _, warnings, _ = run(text, ScriptedRefiner(shorter), config=strict)
        self.assertTrue(warnings)
        self.assertEqual(out, text)


class TestQualifies(unittest.TestCase):

    def words(self, n):
        return " ".join(["word"] * n) + "."

    def test_the_word_boundary(self):
        self.assertFalse(qualifies(self.words(599)))
        self.assertTrue(qualifies(self.words(600)))
        self.assertTrue(qualifies(self.words(100), min_words=50))

    def test_a_short_text_with_something_to_protect_qualifies(self):
        for name, text in {"table": TABLE, "quote": "Some words.\n\n> quoted\n", "figure": "![a](https://x.example/a.png)\n*c*\n",
                           "inline image": "Text ![a](https://x.example/a.png) text.\n", "code": "Text.\n\n```\ncode\n```\n",
                           "html block": "Text.\n\n<div>x</div>\n"}.items():
            with self.subTest(case=name):
                self.assertTrue(qualifies(text))

    def test_a_short_plain_text_keeps_the_single_call(self):
        for text in ("Fix teh grammar please.", "Two paragraphs.\n\nWith a [link](https://x.example) and https://y.example.\n",
                     "- a\n- b\n\n# Heading\n", ""):
            with self.subTest(text=text[:20]):
                self.assertFalse(qualifies(text))

    def test_html_input_and_token_lookalikes_are_left_to_the_single_call(self):
        self.assertFalse(qualifies("<p>" + self.words(900) + "</p>"))
        self.assertFalse(qualifies(self.words(900) + " [[TABLE_1]]"))
        self.assertFalse(qualifies(self.words(900) + " link://3"))
        self.assertFalse(qualifies(None))


class TestTrace(unittest.TestCase):

    def test_what_happened_is_in_the_trace(self):
        start_trace("lf-trace", "/agent/stream")
        try:
            run(acceptance_document(), ScriptedRefiner(polish, {"c2": drop_a_token}), request_id="lf-trace")
            attrs, marks = get_trace("lf-trace").attrs, get_trace("lf-trace").marks
        finally:
            finish_trace("lf-trace")
        info = attrs["longform"]
        self.assertEqual(attrs["mode"], "longform")
        self.assertEqual((info["operation"], info["chunks"], info["restored"]), ("grammar", 6, 1))
        self.assertEqual(info["repaired"], 1)
        self.assertIn("tokens_missing", info["repair_reasons"]["c2"])
        self.assertEqual(info["frozen"]["TABLE"], 4)
        self.assertEqual(info["words_in"] > info["words_out"] * 0.9, True)
        self.assertAlmostEqual(info["word_ratio"], info["words_out"] / info["words_in"], places=2)
        self.assertTrue({"refine_planned", "first_chunk"} <= set(marks))


if __name__ == "__main__":
    unittest.main()


class TestASmallerFirstChunk(unittest.TestCase):

    def test_the_first_text_is_the_small_chunk_and_the_document_is_unchanged(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, config=RefineConfig(first_chunk_words=300))
        self.assertEqual(out, text)
        self.assertLess(word_count(content[0]), 450)
        self.assertGreater(word_count(content[1]), word_count(content[0]))
        self.assertEqual([e["chunk"] for e in events], list(range(1, len(content) + 1)))
        self.assertEqual(warnings, [])

    def test_the_small_chunk_is_the_first_call_and_every_chunk_still_gets_the_same_prompt(self):
        llm = ScriptedRefiner()
        run(acceptance_document(), llm, config=RefineConfig(first_chunk_words=300))
        first = next(c for c in llm.calls if c[2].unit == "c1")
        self.assertLess(word_count(passage_of(first[1][0]["content"])), 450)
        self.assertEqual(len({c[0] for c in llm.calls}), 1)

    def test_a_polished_document_keeps_its_protected_blocks_with_a_small_first_chunk(self):
        text = acceptance_document()
        out = run(text, ScriptedRefiner(polish), config=RefineConfig(first_chunk_words=250))[0]
        self.assertEqual(protected(out), protected(text))
        self.assertEqual(inline_urls(out), inline_urls(text))

    def test_the_first_chunk_is_off_unless_asked_for(self):
        self.assertEqual(RefineConfig().first_chunk_words, 0)
        text = acceptance_document()
        self.assertEqual(len(run(text)[1]), 6)


class TestWhitespaceAroundBlocks(unittest.TestCase):
    """Seen live: two trailing spaces after a table token came back as two trailing spaces on the table's last row."""

    def test_trailing_spaces_after_tokens_never_reach_the_document(self):
        text = acceptance_document()
        out, content, events, warnings, llm = run(text, ScriptedRefiner(space_the_tokens), operation=TONE)
        self.assertEqual(out, text)
        self.assertEqual((warnings, [c[2].stage for c in llm.calls].count("chunk_repair")), ([], 0))      # not even a repair call

    def test_dropped_blank_lines_around_blocks_are_put_back(self):
        text = acceptance_document()
        out, _, _, warnings, llm = run(text, ScriptedRefiner(glue_to_the_blocks), operation=ENHANCE)
        self.assertEqual(out, text)
        self.assertEqual((warnings, [c[2].stage for c in llm.calls].count("chunk_repair")), ([], 0))

    def test_polished_prose_with_both_habits_keeps_every_protected_block_to_the_byte(self):
        text = acceptance_document()
        both = ScriptedRefiner(lambda p, s: glue_to_the_blocks(space_the_tokens(polish(p, s), s), s))
        out = run(text, both)[0]
        self.assertNotEqual(out, text)
        self.assertEqual(protected(out), protected(text))


class TestNothingIsAddedBetweenTheBlocks(unittest.TestCase):

    @staticmethod
    def steps():
        """Headings straight over lists, as in a how-to: the place a friendly model likes to add a lead-in."""
        rng = random.Random(3)
        parts = ["# A how-to", ""]
        for i in range(8):
            parts += [f"## Step {i + 1}", "", f"- {paragraph(rng, 12)}", f"- {paragraph(rng, 10)}", "", paragraph(rng, 130), ""]
        return "\n".join(parts)

    def test_a_lead_in_paragraph_under_a_heading_is_repaired_away(self):
        text = self.steps()
        self.assertRegex(text, r"(?m)^#{1,6} [^\n]*\n\n[-*] ")
        out, content, events, warnings, llm = run(text, ScriptedRefiner(fix_on_repair(add_a_lead_in)), operation=TONE)
        self.assertEqual(out, text)
        self.assertEqual(warnings, [])
        repairs = [c for c in llm.calls if c[2].stage == "chunk_repair"]
        self.assertTrue(repairs)
        self.assertIn("[order]", repairs[0][1][-1]["content"])

    def test_a_model_that_keeps_adding_it_gets_the_chunk_back_as_written(self):
        text = self.steps()
        out, _, _, warnings, _ = run(text, ScriptedRefiner(add_a_lead_in), operation=TONE)
        self.assertEqual(out, text)
        self.assertTrue(warnings)
        self.assertTrue(all("order" in w["detail"] for w in warnings))
