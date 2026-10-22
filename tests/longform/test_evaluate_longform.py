"""
scripts/evaluate_longform.py: the free offline run, the structure check that judges a result independently of the pipeline, the
spelling slips the grammar runs are scored on, the corpora, the summary against the plan's targets, and the live path's outputs.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.text_processing_agent as tpa
from config import settings
from scripts import evaluate_longform as ev
from src.longform.markdown_ast import parse, prose_words, word_count
from src.longform.refine import qualifies
from tests.longform.fakes import FakeAgents, ScriptedRefiner
from tests.longform.fixtures import TABLE, acceptance_document

DOC = acceptance_document(2500)
PROTECTED = ev.PROTECTED


def protected(text):
    return [b.text for b in parse(text) if b.kind in PROTECTED]


class TestOffline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = ev.run_offline()

    def test_ten_documents_four_operations_two_models_all_meet_the_rules(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual({(r["id"], r["op"], r["model"]): r["failures"] for r in self.rows if r["failures"]}, {})

    def test_a_well_behaved_model_restores_nothing_and_fixes_every_slip(self):
        good = [r for r in self.rows if r["model"] == "well-behaved"]
        self.assertEqual((sum(r["restored"] for r in good), sum(r["typos"]["left"] for r in good)), (0, 0))
        self.assertGreater(sum(r["typos"]["injected"] for r in good), 100)

    def test_a_damaging_model_cannot_make_the_net_leak(self):
        bad = [r for r in self.rows if r["model"] == "damaging"]
        self.assertTrue(all(r["expected_bad"] and r["loss"] == [] for r in bad))
        self.assertTrue(all(r["restored"] >= 1 for r in bad if r["chunks"] >= 3))
        self.assertGreater(sum(r["restored"] for r in bad), 30)

    def test_the_run_is_deterministic(self):
        again = ev.run_offline(ev.builtin_corpus()[:2])
        first = [r for r in self.rows if r["id"] in ("d01", "d02")]
        key = lambda r: (r["id"], r["op"], r["model"])
        self.assertEqual([(key(r), r["content"]) for r in sorted(again, key=key)], [(key(r), r["content"]) for r in sorted(first, key=key)])

    def test_exit_code_reflects_failures(self):
        with patch("sys.argv", ["evaluate_longform.py", "--limit", "1"]):
            self.assertEqual(ev.main(), 0)
        bad = [dict(r, failures=["structural loss: x"]) for r in self.rows[:2]]
        with patch("sys.argv", ["evaluate_longform.py", "--limit", "1"]), patch.object(ev, "run_offline", return_value=bad):
            self.assertEqual(ev.main(), 1)

    def test_the_offline_run_judges_the_scripted_models_not_just_the_pipeline(self):
        doc = ev.builtin_corpus()[5]
        with patch.object(ev, "scripted", lambda op, typos, damaging: ScriptedRefiner()):          # a model that changes nothing
            good, bad = ev.run_offline([doc])[:2]
        self.assertTrue(any("spelling slips not fixed" in f for f in good["failures"]), good["failures"])
        self.assertGreaterEqual(bad["chunks"], 3)
        self.assertTrue(any("went unnoticed" in f for f in bad["failures"]), bad["failures"])

    def test_a_word_ratio_outside_the_targets_window_is_reported_even_if_the_pipeline_accepted_it(self):
        doc = ev.builtin_corpus()[2]
        shrinking = ScriptedRefiner(lambda passage, spec: ev.trim(passage))                       # about 0.93: verify allows 0.85
        with patch.object(ev, "scripted", lambda op, typos, damaging: shrinking), patch.object(ev, "TARGET_RATIO", (0.97, 1.03)):
            good = ev.run_offline([doc])[0]
        self.assertEqual((good["op"], good["restored"]), ("grammar", 0))
        self.assertLess(good["ratio"], 0.97)
        self.assertTrue(any(f.startswith("word ratio") for f in good["failures"]), good["failures"])

    def test_a_model_that_ruins_every_chunk_gives_the_document_back_as_it_was_sent(self):
        doc = ev.builtin_corpus()[2]
        with patch.object(ev, "scripted", return_value=ScriptedRefiner(lambda passage, spec: "Nothing to see.")):
            row = ev.run_one_offline(doc, "grammar", False)
        self.assertEqual(row["content"], doc.text)
        self.assertEqual((row["loss"], row["restored"]), ([], row["chunks"]))


class TestTheBuiltinCorpus(unittest.TestCase):

    def test_ten_documents_from_800_to_5000_words_that_all_qualify(self):
        docs = ev.builtin_corpus()
        self.assertEqual([d.id for d in docs], [f"d{n:02d}" for n in range(1, 11)])
        for doc, (_, target) in zip(docs, ev.BUILTIN):
            with self.subTest(doc=doc.id):
                self.assertLess(abs(doc.words - target) / target, 0.15)
                self.assertTrue(qualifies(doc.text))

    def test_every_kind_of_locked_element_is_in_the_big_ones(self):
        big = ev.builtin_corpus()[-1].original
        kinds = {b.kind for b in parse(big)}
        self.assertGreaterEqual(kinds, {"table", "blockquote", "image", "fence", "list", "heading", "paragraph"})
        self.assertIn("[our policy][pol]", big)
        self.assertIn("https://example.net/plain.", big)

    def test_each_document_has_spelling_slips_and_they_are_not_words_of_the_original(self):
        for doc in ev.builtin_corpus():
            with self.subTest(doc=doc.id):
                self.assertGreater(len(doc.typos), 2)
                self.assertNotEqual(doc.text, doc.original)
                words = set(re.findall(r"[A-Za-z]+", doc.original))
                self.assertFalse(set(doc.typos) & words)

    def test_the_typo_rate_can_be_turned_off(self):
        self.assertTrue(all(not d.typos and d.text == d.original for d in ev.builtin_corpus(0)))


class TestStructureLoss(unittest.TestCase):

    def test_an_identical_document_lost_nothing(self):
        self.assertEqual(ev.structure_loss(DOC, DOC), [])

    def test_reworded_prose_and_merged_paragraphs_are_not_a_loss(self):
        reworded = DOC.replace("Most ", "Nearly all ").replace(" and ", " plus ")
        self.assertNotEqual(reworded, DOC)
        self.assertEqual(ev.structure_loss(DOC, reworded), [])
        text = "Intro paragraph one.\n\nSecond paragraph follows here.\n\n## Heading\n\nThird paragraph.\n\nFourth paragraph.\n"
        merged = "Intro paragraph one. Second paragraph follows here.\n\n## Heading\n\nThird paragraph. Fourth paragraph.\n"
        self.assertEqual(ev.structure_loss(text, merged), [])

    def test_a_paragraph_pulled_into_a_quote_is_a_loss(self):
        text = "Intro.\n\n> A quote.\n\nAfter the quote.\n"
        pulled = "Intro.\n\n> A quote.\nAfter the quote.\n"                      # markdown's lazy continuation: it is the quote now
        self.assertTrue(any(f.startswith("protected blocks") for f in ev.structure_loss(text, pulled)))

    def test_a_changed_table_quote_code_or_picture_is_a_loss(self):
        for name, old, new in (("table", "| Plan A | $10 |", "| Plan A | $11 |"), ("quote", "> A pull quote", "> A pulled quote"),
                               ("code", "print(x)", "print(y)"), ("picture", "Desk 0 with", "Chair 0 with")):
            with self.subTest(block=name):
                self.assertIn(old, DOC)
                found = ev.structure_loss(DOC, DOC.replace(old, new, 1))
                self.assertTrue(any(f.startswith("protected blocks") for f in found), found)

    def test_a_dropped_block_names_where_the_difference_starts(self):
        without_table = DOC.replace(TABLE + "\n\n", "", 1)
        found = ev.structure_loss(DOC, without_table)
        self.assertTrue(any("kept, first difference at #" in f for f in found), found)

    def test_a_changed_url_is_a_loss_but_punctuation_after_a_bare_url_is_not(self):
        self.assertTrue(any(f.startswith("urls changed") for f in
                            ev.structure_loss(DOC, DOC.replace("https://example.com/guide", "https://example.com/guides"))))
        self.assertIn("https://example.net/plain.", DOC)
        self.assertEqual(ev.structure_loss(DOC, DOC.replace("https://example.net/plain.", "https://example.net/plain,")), [])
        self.assertEqual(ev.structure_loss(DOC, DOC.replace("https://example.net/plain.", "https://example.net/plain")), [])

    def test_inline_html_tags_must_all_come_back(self):
        text = "Press <kbd>Ctrl</kbd> and then <kbd>C</kbd>.<br>Next line <img src=\"a.png\"> end.\n"
        self.assertEqual(ev.structure_loss(text, text.replace("Next line", "The following line")), [])
        for broken in (text.replace("<br>", " "), text.replace("</kbd> and", " and"), text.replace('<img src="a.png">', '<img src="b.png">')):
            self.assertIn("inline HTML tags changed", ev.structure_loss(text, broken))

    def test_heading_levels_must_hold_but_the_words_may_change(self):
        self.assertEqual(ev.structure_loss(DOC, DOC.replace("## Section 2:", "## Part 2:")), [])
        found = ev.structure_loss(DOC, DOC.replace("## Section 2:", "### Section 2:"))
        self.assertTrue(any(f.startswith("headings") for f in found), found)
        self.assertTrue(any(f.startswith("headings") for f in ev.structure_loss(DOC, DOC.replace("## Section 3:", "Section 3:"))))

    def test_a_list_must_keep_its_items_and_nesting(self):
        self.assertIn("\n  - ", DOC)
        flat = DOC.replace("\n  - ", "\n- ")
        self.assertIn("a list changed shape", ev.structure_loss(DOC, flat))
        lines = DOC.split("\n")
        first = next(i for i, l in enumerate(lines) if l.startswith("- "))
        self.assertIn("a list changed shape", ev.structure_loss(DOC, "\n".join(lines[:first] + lines[first + 1:])))

    def test_a_block_moved_past_another_is_a_loss(self):
        text = f"Intro paragraph.\n\n{TABLE}\n\n## A heading\n\nBody text.\n"
        moved = f"Intro paragraph.\n\n## A heading\n\n{TABLE}\n\nBody text.\n"
        found = ev.structure_loss(text, moved)
        self.assertTrue(any("order" in f for f in found), found)

    def test_quotes_may_change_when_they_are_not_frozen(self):
        edited = DOC.replace("> A pull quote", "> A pulled quote")
        self.assertTrue(ev.structure_loss(DOC, edited))
        self.assertEqual(ev.structure_loss(DOC, edited, freeze_quotes=False), [])

    def test_empty_output_is_a_loss(self):
        row = ev.row_of(ev.builtin_corpus()[0], "grammar", "", "", {"mode": "longform"}, provider="p", model="m",
                        first_chunk=None, seconds=1.0, warnings=[])
        self.assertEqual(row["loss"], ["no text"])
        self.assertTrue(row["failures"])


class TestSpellingSlips(unittest.TestCase):

    def test_they_are_deterministic_and_the_seed_matters(self):
        a, b = ev.inject_typos(DOC, 100, 1), ev.inject_typos(DOC, 100, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], ev.inject_typos(DOC, 100, 2)[0])

    def test_about_one_per_n_prose_words(self):
        text, typos = ev.inject_typos(DOC, 100, 3)
        expected = prose_words(parse(DOC)) // 100
        self.assertLessEqual(len(typos), expected)                          # a repeated word's identical slip is not counted twice
        self.assertGreaterEqual(len(typos), 0.8 * expected)
        self.assertEqual(ev.inject_typos(DOC, 0, 3), (DOC, {}))
        self.assertEqual(ev.inject_typos("Short text here.\n", 100, 3), ("Short text here.\n", {}))

    def test_nothing_but_prose_words_changes(self):
        text, typos = ev.inject_typos(DOC, 40, 5)
        self.assertGreater(len(typos), 10)
        self.assertEqual(protected(text), protected(DOC))                                      # tables, quotes, code, pictures
        self.assertEqual(ev.structure_loss(DOC, text), [])                                     # urls, headings, lists, order
        self.assertEqual([b.text for b in parse(text) if b.kind == "heading"], [b.text for b in parse(DOC) if b.kind == "heading"])
        self.assertEqual(len(parse(text)), len(parse(DOC)))

    def test_code_inside_a_list_inline_code_and_link_targets_are_never_touched(self):
        text = ("Intro paragraph with `inline command arguments` and a [link to documentation](https://example.com/documentation) "
                "and some ordinary prose words that may take a slip.\n\n"
                "- first item has ordinary prose to misspell\n"
                "  ```python\n  response = client.generate_content(prompt)\n  ```\n"
                "- second item has `another command` and more ordinary prose\n")
        for seed in range(30):
            slipped, typos = ev.inject_typos(text, 5, seed)
            self.assertIn("response = client.generate_content(prompt)", slipped)
            self.assertIn("`inline command arguments`", slipped)
            self.assertIn("`another command`", slipped)
            self.assertIn("(https://example.com/documentation)", slipped)
            self.assertEqual(protected(slipped), protected(text))

    def test_the_slips_are_not_dictionary_words_by_construction(self):
        """A drop ("shared" -> "shred") often makes another word, which a proofreader cannot call wrong: only swaps and doublings."""
        _, typos = ev.inject_typos(DOC, 20, 11)
        self.assertGreater(len(typos), 10)
        for slip, word in typos.items():
            doubled = len(slip) == len(word) + 1
            swapped = len(slip) == len(word) and sorted(slip) == sorted(word)
            self.assertTrue(doubled or swapped, (slip, word))

    def test_correcting_every_slip_gives_back_the_original_exactly(self):
        text, typos = ev.inject_typos(DOC, 40, 5)
        self.assertEqual(ev.fixer(typos)(text), DOC)

    def test_slips_still_in_the_output_are_counted_as_whole_words(self):
        typos = {"recieve": "receive", "teh": "the"}
        self.assertEqual(ev.typos_left("We recieve teh mail.", typos), 2)
        self.assertEqual(ev.typos_left("We receive the mail.", typos), 0)
        self.assertEqual(ev.typos_left("Unrecieved is not recieve.", typos), 1)           # inside another word does not count

    def test_each_slip_is_a_real_change_of_the_same_word(self):
        _, typos = ev.inject_typos(DOC, 20, 9)
        for slip, word in typos.items():
            self.assertNotEqual(slip, word)
            self.assertTrue(slip.isalpha() and slip.islower())
            self.assertLessEqual(abs(len(slip) - len(word)), 1)


class TestSlicingDocuments(unittest.TestCase):

    TEXT = "".join(f"## Part {n}\n\n{'word ' * 400}\n\n" for n in range(1, 9))

    def test_sections_are_cut_before_h1_and_h2_headings_only(self):
        text = "# Title\n\nIntro.\n\n## One\n\nText.\n\n### Deeper\n\nMore.\n\n## Two\n\nEnd.\n"
        parts = ev.sections_of(text)
        self.assertEqual([p.split("\n")[0] for p in parts], ["# Title", "## One", "## Two"])
        self.assertEqual("".join(parts), text)

    def test_a_hash_inside_a_code_fence_is_not_a_heading(self):
        text = "## One\n\n```sh\n# a comment\n## not a heading\n```\n\nAfter.\n\n## Two\n\nEnd.\n"
        self.assertEqual(len(ev.sections_of(text)), 2)

    def test_parts_hold_whole_sections_and_join_back_to_the_text(self):
        parts = ev.slice_document(self.TEXT, max_words=1000, min_words=300)
        self.assertEqual("".join(parts), self.TEXT)
        self.assertGreater(len(parts), 2)
        for part in parts:
            self.assertTrue(part.startswith("## Part"))
            self.assertLessEqual(word_count(part), 1000 + 400)                # the last section may overshoot only by a short tail

    def test_a_short_tail_joins_the_part_before(self):
        text = "## A\n\n" + "word " * 900 + "\n\n## B\n\n" + "word " * 100 + "\n"
        self.assertEqual(len(ev.slice_document(text, max_words=1000, min_words=300)), 1)
        self.assertEqual(len(ev.slice_document(text, max_words=1000, min_words=50)), 2)

    def test_a_giant_section_is_a_part_of_its_own(self):
        text = "## A\n\n" + "word " * 3000 + "\n\n## B\n\nshort\n"
        parts = ev.slice_document(text, max_words=1000, min_words=0)
        self.assertEqual(len(parts), 2)
        self.assertEqual("".join(parts), text)


class TestTheRealCorpora(unittest.TestCase):

    def test_the_repositorys_own_documents_are_a_corpus(self):
        docs = ev.docs_corpus()
        self.assertGreaterEqual(len(docs), 5)
        self.assertEqual(len({d.id for d in docs}), len(docs))
        by_file = {}
        for d in docs:
            by_file.setdefault(d.source, []).append(d.original)
        for source, parts in by_file.items():
            self.assertEqual("".join(parts), (ev.ROOT / source).read_text(encoding="utf-8"), source)

    def test_posts_are_one_per_brief_alone_and_three_together_with_the_header_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blog_eval").mkdir()
            for brief in ("t01", "t02", "t03"):
                post = f"<!-- header for {brief}\n     second line -->\n\n# Post {brief}\n\n" + "word " * 850 + "\n"
                (root / "blog_eval" / f"{brief}_openai.md").write_text(post)
                (root / "blog_eval" / f"{brief}_openai_run2.md").write_text(post.replace("word", "text"))     # same brief again
            with patch.object(ev, "RUN_DIR", root):
                docs = ev.posts_corpus()
        self.assertEqual([d.id for d in docs], ["p01", "p02", "p03", "p04"])
        self.assertTrue(all("<!--" not in d.original for d in docs))
        self.assertEqual([d.original.count("# Post") for d in docs], [1, 1, 1, 3])

    def test_files_are_documents_too(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.md"
            path.write_text("# Notes\n\n" + "careful planning " * 200 + "\n")
            (doc,) = ev.files_corpus([str(path)])
        self.assertEqual((doc.id, doc.title, doc.original.startswith("# Notes")), ("f01", "notes.md", True))


class TestSummary(unittest.TestCase):

    def row(self, **kw):
        base = {"provider": "openai", "model": "gpt", "op": "grammar", "pipeline": True, "chunks": 5, "repaired": 1, "restored": 0,
                "ratio": 1.0, "loss": [], "words": 5000, "seconds": 30.0, "first_chunk": 9.0, "repair_reasons": {},
                "typos": {"injected": 10, "left": 1}, "est_cost_usd": 0.1, "failures": []}
        base.update(kw)
        return base

    def test_the_summary_compares_against_the_plans_targets(self):
        rows = [self.row(), self.row(seconds=50.0, ratio=1.2, restored=1, repair_reasons={"c2": ["tokens_missing", "length"]}),
                self.row(op="tone", ratio=1.4, loss=["a list changed shape"])]
        text = "\n".join(ev.summarise(rows))
        self.assertIn("structural loss: 1 runs (target 0)", text)
        self.assertIn("word ratio 0.9-1.1 for enhance/grammar: 1/2", text)
        self.assertIn("restored chunks: 1/15 = 6.7% (target <= 5%)", text)
        self.assertIn("p50 40.0s", text)
        self.assertIn("spelling slips left in the output: 3/30", text)
        self.assertIn("tokens_missing 1, length 1", text)
        self.assertIn("$0.200", text)

    def test_runs_scripted_to_misbehave_are_left_out_of_the_targets(self):
        rows = [self.row(), self.row(expected_bad=True, restored=4, chunks=5, model="damaging")]
        text = "\n".join(ev.summarise(rows))
        self.assertIn("restored chunks: 0/5 = 0.0%", text)
        self.assertIn("scripted to misbehave: 1 runs, 4 of 5 chunks restored as written, structural loss in 0 runs", text)

    def test_runs_answered_by_a_fallback_are_not_counted_for_the_provider_asked_for(self):
        rows = [self.row(), self.row(provider="claude", model="claude-sonnet-5", seconds=99.0, restored=3, loss=["x"],
                                     served_by="openai/gpt-4o-mini",
                                     failures=["served by openai/gpt-4o-mini, not claude (a fallback answered)"])]
        text = "\n".join(ev.summarise(rows))
        self.assertIn("NOT COUNTED: 1 runs of ['claude'] were answered by ['openai/gpt-4o-mini']", text)
        self.assertIn("restored chunks: 0/5 = 0.0%", text)                                  # its restores are not in the targets
        self.assertNotIn("claude-sonnet-5", text.split("NOT COUNTED")[1].split("\n", 1)[1])   # and it has no line of its own
        only_fallback = [rows[1]]
        self.assertTrue(ev.summarise(only_fallback)[0].startswith("no results: 1 runs were answered by a fallback"))

    def test_nothing_to_summarise(self):
        self.assertEqual(ev.summarise([]), ["no results"])
        self.assertEqual(ev.summarise([{"pipeline": False}]), ["no results"])

    def test_percentile_is_nearest_rank(self):
        self.assertEqual([ev.percentile([1, 2, 3, 4, 5], q) for q in (0, 0.5, 0.95, 1)], [1, 3, 5, 5])
        self.assertEqual(ev.percentile([7], 0.95), 7)


class TestLive(unittest.TestCase):
    """The paid path, with the model call faked at the agent factory so the real routing, hook and pipeline run."""

    def setUp(self):
        self.agents = FakeAgents()
        for p in (patch.object(tpa, "create_agent", self.agents), patch.object(settings, "longform_refine_enabled", False),
                  patch.object(settings, "longform_parallel", 3), patch.object(settings, "longform_chunk_words", 1000),
                  patch.object(settings, "task_routing_enabled", False), patch.object(settings, "provider_fallback_enabled", False),
                  patch.object(settings, "default_openai_api_key", "dummy"), patch.object(settings, "default_anthropic_api_key", None),
                  patch.object(settings, "default_gemini_api_key", None), patch.object(settings, "default_grok_api_key", None),
                  patch.object(settings, "default_deepseek_api_key", None)):
            p.start()
            self.addCleanup(p.stop)
        self.doc = ev.builtin_corpus()[3]

    def run_main(self, *argv):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "results.json"
            with patch.object(ev, "RUN_DIR", Path(directory)), \
                    patch("sys.argv", ["evaluate_longform.py", "--live", "--providers", "openai", "--corpus", "builtin",
                                       "--out", str(out), *argv]):
                code = ev.main()
            rows = json.loads(out.read_text())
            files = sorted(p.name for p in (Path(directory) / "longform_eval").glob("*") if p.is_file())
            inputs = sorted(p.name for p in (Path(directory) / "longform_eval" / "inputs").glob("*"))
            refined = (Path(directory) / "longform_eval" / f"{rows[0]['id']}_{rows[0]['op']}_openai.md").read_text()
        return code, rows, files, inputs, refined

    def test_a_live_run_records_the_metrics_and_writes_the_document_and_a_diff(self):
        fix = ev.fixer(self.doc.typos)
        self.agents.refiner.behaviour = lambda passage, spec: fix(passage)
        code, rows, files, inputs, refined = self.run_main("--ids", "d04", "--ops", "grammar")
        self.assertEqual(code, 0)
        (row,) = rows
        self.assertEqual((row["id"], row["op"], row["provider"], row["pipeline"]), ("d04", "grammar", "openai", True))
        self.assertEqual((row["failures"], row["loss"], row["restored"], row["repaired"]), ([], [], 0, 0))
        self.assertEqual(row["typos"], {"injected": len(self.doc.typos), "left": 0})
        self.assertEqual(row["ratio"], 1.0)
        self.assertTrue(row["served_by"].startswith("openai/"))
        self.assertIsNotNone(row["first_chunk"])
        self.assertNotIn("content", row)                                   # documents live in their own files, not in the results
        self.assertEqual(files, ["d04_grammar_openai.diff", "d04_grammar_openai.md"])
        self.assertEqual(inputs, ["d04.md"])
        self.assertEqual(refined, self.doc.original)

    def test_the_run_turns_the_pipeline_on_for_itself_and_leaves_the_setting_as_it_was(self):
        self.assertFalse(settings.longform_refine_enabled)
        self.run_main("--ids", "d01", "--ops", "grammar")
        self.assertFalse(settings.longform_refine_enabled)
        self.assertEqual(set(self.agents.agent_types()), {"longform_refiner"})

    def test_parallel_and_chunk_size_apply_to_the_run_only(self):
        seen = {}
        original = ev.run_one_live

        def spy(*a, **k):
            seen.update(parallel=settings.longform_parallel, chunk=settings.longform_chunk_words)
            return original(*a, **k)
        with patch.object(ev, "run_one_live", spy):
            self.run_main("--ids", "d01", "--ops", "grammar", "--parallel", "2", "--chunk-words", "500")
        self.assertEqual(seen, {"parallel": 2, "chunk": 500})
        self.assertEqual((settings.longform_parallel, settings.longform_chunk_words), (3, 1000))

    def test_the_tone_and_the_instruction_reach_the_model(self):
        self.run_main("--ids", "d01", "--ops", "tone", "enhance", "--tone", "stern", "--instruction", "make it vivid")
        seen = "\n".join(m["content"] for _, messages, _ in self.agents.refiner.calls for m in messages)
        self.assertIn("TONE: stern", seen)
        self.assertIn("INSTRUCTION: make it vivid", seen)

    def test_ids_ops_and_limit_select_the_runs(self):
        _, rows, _, _, _ = self.run_main("--ids", "d01", "d02", "--ops", "grammar", "simplify")
        self.assertEqual([(r["id"], r["op"]) for r in rows], [("d01", "grammar"), ("d01", "simplify"), ("d02", "grammar"), ("d02", "simplify")])
        _, rows, _, _, _ = self.run_main("--limit", "1", "--ops", "grammar")
        self.assertEqual([r["id"] for r in rows], ["d01"])

    def test_a_model_that_damages_a_chunk_is_a_restore_in_the_row_not_a_loss(self):
        self.agents.refiner.per_unit = {}
        self.agents.refiner.behaviour = lambda passage, spec: "Nothing to see."
        code, rows, _, _, refined = self.run_main("--ids", "d02", "--ops", "enhance")
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["loss"], [])
        self.assertEqual(rows[0]["restored"], rows[0]["chunks"] - rows[0]["passthrough"])
        self.assertEqual(refined, ev.builtin_corpus()[1].text)                     # restored: as it was sent

    def test_debug_logs_what_the_model_first_answered_when_a_passage_needed_a_repair(self):
        import logging
        logger = logging.getLogger("src.longform")
        self.addCleanup(logger.setLevel, logger.level)
        self.agents.refiner.behaviour = lambda passage, spec: "Nothing to see here." if spec.stage == "chunk" else passage
        with patch("logging.basicConfig"), self.assertLogs("src.longform.refine", level="DEBUG") as logged:
            code, rows, _, _, _ = self.run_main("--ids", "d01", "--ops", "grammar", "--debug")
        self.assertEqual((code, rows[0]["restored"]), (0, 0))                               # the repair put it right
        self.assertTrue(any("first answer (4 words" in line and "Nothing to see here." in line for line in logged.output))

    def test_a_provider_error_is_an_error_row_and_the_exit_code_says_so(self):
        self.agents.fail = {"openai": RuntimeError("503 overloaded")}
        code, rows, _, _, _ = self.run_main("--ids", "d01", "--ops", "grammar")
        self.assertEqual(code, 1)
        self.assertIn("503", rows[0]["error"])
        self.assertTrue(any("error: " in f for f in rows[0]["failures"]))

    def test_a_dead_key_stops_the_paid_run(self):
        self.agents.fail = {"openai": RuntimeError("401 invalid api key")}
        _, rows, _, _, _ = self.run_main("--ids", "d01", "d02", "--ops", "grammar", "enhance")
        self.assertEqual(len(rows), 1)

    def test_a_provider_whose_calls_are_answered_by_a_fallback_is_not_run_to_the_end(self):
        real = ev.finish_trace
        with patch.object(ev, "finish_trace", lambda *a, **k: {**(real(*a, **k) or {}), "model_used": "gemini/some-model"}):
            code, rows, _, _, _ = self.run_main("--ids", "d01", "d02", "--ops", "grammar", "enhance")
        self.assertEqual(len(rows), 1)
        self.assertIn("served by gemini/some-model, not openai (a fallback answered)", rows[0]["failures"])

    def test_a_row_answered_by_a_fallback_is_not_scored_as_the_requested_model(self):
        real = ev.finish_trace
        with patch.object(settings, "longform_refine_enabled", True):
            honest = ev.run_one_live("openai", "k", self.doc, "grammar", "")
            with patch.object(ev, "finish_trace", lambda *a, **k: {**(real(*a, **k) or {}), "model_used": "gemini/some-model"}):
                swapped = ev.run_one_live("openai", "k", self.doc, "grammar", "")
        self.assertEqual(honest["failures"], [])
        self.assertIn("served by gemini/some-model, not openai (a fallback answered)", swapped["failures"])

    def test_the_single_call_path_is_a_failure_the_row_names(self):
        row = ev.run_one_live("openai", "k", self.doc, "grammar", "")                     # the flag is off: the single call ran
        self.assertFalse(row["pipeline"])
        self.assertIn("the chunked path did not run", row["failures"])

    def test_a_missing_key_is_a_row_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ev, "RUN_DIR", Path(directory)), \
                patch.object(settings, "default_gemini_api_key", None), \
                patch("sys.argv", ["evaluate_longform.py", "--live", "--providers", "gemini", "--corpus", "builtin", "--limit", "1",
                                   "--out", str(Path(directory) / "r.json")]):
            self.assertEqual(ev.main(), 1)

    def test_bad_flags_are_rejected(self):
        for argv in (["--limit", "-1"], ["--typo-every", "-5"], ["--parallel", "-1"], ["--ids", "zz99"], ["--ops", "translate"]):
            with self.subTest(argv=argv), patch("sys.argv", ["evaluate_longform.py", *argv]), self.assertRaises(SystemExit):
                ev.main()

    def test_documents_that_would_keep_the_single_call_are_skipped_not_paid_for(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain.md"
            plain.write_text("# Notes\n\n" + "careful planning work " * 300 + "\n")
            tricky = Path(directory) / "tricky.md"
            tricky.write_text("# Notes\n\n" + "careful planning work " * 300 + "\nAn example token: [[TABLE_2]] in the text.\n")
            with patch.object(ev, "RUN_DIR", Path(directory)), patch("builtins.print") as printed, \
                    patch("sys.argv", ["evaluate_longform.py", "--live", "--providers", "openai", "--files", str(tricky), str(plain),
                                       "--ops", "grammar", "--out", str(Path(directory) / "r.json")]):
                self.assertEqual(ev.main(), 0)
            rows = json.loads((Path(directory) / "r.json").read_text())
        self.assertEqual([r["title"] for r in rows], ["plain.md"])
        said = "\n".join(str(c[0][0]) for c in printed.call_args_list)
        self.assertIn("Skipped", said)
        self.assertIn("f01 tricky.md", said)

    def test_list_prints_the_corpus_and_stops(self):
        with patch("sys.argv", ["evaluate_longform.py", "--list", "--limit", "2"]), patch("builtins.print") as printed:
            self.assertEqual(ev.main(), 0)
        self.assertEqual(printed.call_count, 2)
        self.assertIn("d01", printed.call_args_list[0][0][0])


if __name__ == "__main__":
    unittest.main()
