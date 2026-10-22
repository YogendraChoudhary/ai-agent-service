"""scripts/evaluate_blog.py: the free structural check over 30 briefs, the failure rules, and the live path's outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.text_processing_agent as tpa
from config import settings
from scripts import evaluate_blog as ev
from src.blog.media import GeneratedImages
from src.blog.plan import BlogRequest, plan_blog
from src.blog.service import generate_blog_stream
from src.observability import trace_set
from tests.blog.fakes import FakeAgentFactory, FakeImageTool, ScriptedLLM, good_outline

GOOD = ("# Title\n\n*Sub*\n\n## One\n\nText.\n\n![a](x.jpg)\n*c*\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n> q1\n\n> q2\n\n"
        "## Two\n\nText.\n")


class TestOffline(unittest.TestCase):

    def test_thirty_briefs_all_meet_the_structure_rules(self):
        rows = ev.run_offline()
        self.assertEqual(len(rows), 30)
        self.assertEqual({r["id"]: r["failures"] for r in rows if r["failures"]}, {})

    def test_the_corpus_is_varied(self):
        briefs = [b for _, b, _ in ev.TOPICS]
        self.assertEqual(len(set(briefs)), 30)
        self.assertEqual(len({i for i, _, _ in ev.TOPICS}), 30)
        plans = [ev.plan_for(b, o) for _, b, o in ev.TOPICS]
        self.assertGreaterEqual({p.length for p in plans}, {"short", "standard", "long"})
        self.assertTrue(any(p.use_web for p in plans))
        self.assertTrue(any(p.images == 0 for p in plans))

    def test_exit_code_reflects_failures(self):
        with patch("sys.argv", ["evaluate_blog.py"]):
            self.assertEqual(ev.main(), 0)
        with patch("sys.argv", ["evaluate_blog.py"]), patch.object(ev, "run_offline", return_value=[
                {"id": "t99", "failures": ["tables: 0"]}]):
            self.assertEqual(ev.main(), 1)


class TestAnalyseAndFailures(unittest.TestCase):

    def test_analyse(self):
        m = ev.analyse(GOOD)
        self.assertEqual((m["h1"], m["h2"], m["tables"], m["quotes"], m["figures"]), (1, 2, 1, 2, 1))
        self.assertTrue(m["tables_consistent"])
        self.assertEqual(m["raw_urls"], 0)

    def test_every_defect_is_named(self):
        plan = ev.plan_for("Write a blog post about tea", {})
        defective = "# T\n\n# Second title\n\n" + "## A\n\n" + "word " * 300 + "\n\nhttps://example.com\n"
        found = " | ".join(ev.failures(ev.analyse(defective), plan, pictures_available=True))
        for fragment in ("top-level titles", "section headings", "tables", "quotes", "pictures", "words of prose in a row", "URLs"):
            self.assertIn(fragment, found)

    def test_a_post_that_researched_may_have_urls(self):
        plan = ev.plan_for("Write a blog post about the latest AI trends in 2026", {"images": 0})
        self.assertTrue(plan.use_web)
        found = ev.failures(ev.analyse(GOOD + "\nSee [x](https://example.com/a).\n"), plan, pictures_available=False)
        self.assertFalse(any("URLs" in f for f in found))

    def test_a_ragged_table_fails_even_if_present(self):
        plan = ev.plan_for("Write a blog post about tea", {})
        ragged = GOOD.replace("| 1 | 2 |", "| 1 |")
        self.assertTrue(any("tables" in f for f in ev.failures(ev.analyse(ragged), plan, pictures_available=False)))

    def test_pictures_are_only_required_when_available(self):
        plan = ev.plan_for("Write a blog post about tea", {})
        found = ev.failures(ev.analyse("# T\n\n## A\n\ntext\n"), plan, pictures_available=False)
        self.assertFalse(any("pictures" in f for f in found))


class TestSummary(unittest.TestCase):

    def test_the_summary_compares_against_the_plans_targets(self):
        rows = [{"metrics": {"longest_prose_run": 200}, "failures": [], "warnings": [], "seconds": 30.0, "first_content": 8.0,
                 "est_cost_usd": 0.1, "blog": {"repairs": 1}},
                {"metrics": {"longest_prose_run": 300}, "failures": ["x"], "warnings": [{"kind": "k"}], "seconds": 50.0,
                 "first_content": 14.0, "est_cost_usd": 0.3, "blog": {"repairs": 0}}]
        text = "\n".join(ev.summarise(rows))
        self.assertIn("posts: 2  with structural failures: 1", text)
        self.assertIn("1/2", text)
        self.assertIn("p50 40.0s", text)
        self.assertIn("$0.400 total", text)
        self.assertIn("repair calls: 1  posts with warnings: 1", text)

    def test_nothing_to_summarise(self):
        self.assertEqual(ev.summarise([]), ["no results"])


class TestLive(unittest.TestCase):
    """The paid path, with the stream faked by the real pipeline over a scripted model."""

    def fake_stream(self, brief, provider, key, **kw):
        request = BlogRequest.from_options(brief, "", kw["blog_options"])
        tool = FakeImageTool()
        yield from (c for c in generate_blog_stream(request, plan_blog(request), ScriptedLLM(), GeneratedImages(tool)))
        yield {"__pending_images__": {
            pid: {"status": "done", "cdn_url": f"../images/{i}.jpg", "provider": "gemini", "model": "m", "seconds": 1.0}
            for i, pid in enumerate(tool.placeholders)}}

    def run_main(self, *argv):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "results.json"
            with patch.object(ev, "RUN_DIR", Path(directory)), patch.object(settings, "default_openai_api_key", "dummy"), \
                    patch("src.text_processing_agent.process_user_query_stream", self.fake_stream), \
                    patch("sys.argv", ["evaluate_blog.py", "--live", "--providers", "openai", "--out", str(out), *argv]):
                code = ev.main()
            rows = json.loads(out.read_text())
            posts = sorted(p.name for p in (Path(directory) / "blog_eval").glob("*.md"))
            text = (Path(directory) / "blog_eval" / posts[0]).read_text() if posts else ""
        return code, rows, posts, text

    def test_a_live_run_records_metrics_and_writes_the_post_for_review(self):
        code, rows, posts, text = self.run_main("--limit", "1", "--images")
        self.assertEqual(code, 0)
        self.assertEqual(posts, ["t01_openai.md"])
        (row,) = rows
        self.assertEqual(row["failures"], [])
        self.assertEqual((row["metrics"]["tables"], row["metrics"]["quotes"], row["metrics"]["figures"]), (1, 2, 3))
        self.assertEqual(len(row["images"]), 3)
        self.assertIsNotNone(row["first_content"])
        self.assertTrue(text.startswith("<!-- openai /"))
        self.assertNotIn("placeholder::", text)                       # pictures resolved to their (local) files
        self.assertIn("](../images/", text)

    def test_without_images_the_plan_asks_for_none(self):
        _, rows, _, text = self.run_main("--ids", "t01")
        self.assertEqual(rows[0]["options"]["images"], 0)

    def test_ids_select_briefs(self):
        _, rows, posts, _ = self.run_main("--ids", "t02", "t03")
        self.assertEqual([r["id"] for r in rows], ["t02", "t03"])

    def test_a_missing_key_is_a_row_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ev, "RUN_DIR", Path(directory)), \
                patch.object(settings, "default_gemini_api_key", None), \
                patch("sys.argv", ["evaluate_blog.py", "--live", "--providers", "gemini", "--out", str(Path(directory) / "r.json")]):
            self.assertEqual(ev.main(), 1)

    def test_negative_limits_are_rejected(self):
        with patch("sys.argv", ["evaluate_blog.py", "--limit", "-1"]), self.assertRaises(SystemExit):
            ev.main()


class TestOutlineOnly(unittest.TestCase):
    """The benchmark that picks the outline model: timing per call, and whether the first reply needed repair."""

    KEYS = {"openai": "o", "gemini": "g"}
    TOPICS = [("t01", "Write a blog post about sleep routines", {})]

    def run_models(self, factory, models=("openai:advanced", "gemini:basic"), repeats=1):
        with patch.object(tpa, "create_agent", factory), patch("time.sleep"):
            return ev.run_outline_only(list(models), self.TOPICS, self.KEYS, repeats)

    def test_each_model_is_timed_and_judged_on_its_first_reply(self):
        factory = FakeAgentFactory()
        rows = self.run_models(factory)
        self.assertEqual([(r["model"], r["id"], r["first_pass_valid"], r["repaired"], r["calls"]) for r in rows],
                         [("openai:advanced", "t01", True, False, 1), ("gemini:basic", "t01", True, False, 1)])
        self.assertEqual(factory.providers(), ["openai", "gemini"])
        self.assertEqual([c[3]["plan"].advanced_model for c in factory.created], [True, False])
        for r in rows:
            self.assertIsNotNone(r["first_call_seconds"])
            self.assertEqual(len(r["outline"]["sections"]), 5)
            self.assertEqual(r["outline"]["sections"][1]["visuals"], ["table", "quote"])
        self.assertEqual(rows[1]["model_id"], ev.model_id_for(__import__("src.task_router", fromlist=["x"]).Candidate("gemini", False)))

    def test_a_fast_model_that_needs_a_repair_is_not_counted_as_valid(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        good = good_outline({"plan": {"target_words": 1400, "images": 3}})
        (row,) = self.run_models(FakeAgentFactory(ScriptedLLM(outline=[bad, good])), models=["gemini:basic"])
        self.assertFalse(row["first_pass_valid"])
        self.assertEqual(row["first_pass_errors"], ["tables"])
        self.assertTrue(row["repaired"])
        self.assertEqual(row["calls"], 2)

    def test_unusable_replies_are_an_error_row_not_a_crash(self):
        (row,) = self.run_models(FakeAgentFactory(ScriptedLLM(outline="not json")), models=["openai:advanced"])
        self.assertFalse(row["first_pass_valid"])
        self.assertEqual(row["first_pass_errors"], ["json"])
        self.assertIn("usable outline", row["error"])
        self.assertIsNone(row["outline"])

    def test_unknown_models_and_missing_keys_are_rows_too(self):
        rows = self.run_models(FakeAgentFactory(), models=["nonsense", "grok:basic"])
        self.assertEqual([r["error"] for r in rows], ["unknown provider or no configured key"] * 2)

    def test_repeats(self):
        rows = self.run_models(FakeAgentFactory(), models=["openai:advanced"], repeats=3)
        self.assertEqual([r["repeat"] for r in rows], [0, 1, 2])

    def test_summary_compares_models(self):
        rows = [{"model": "a:advanced", "first_call_seconds": 13.0, "first_pass_valid": True, "repaired": False,
                 "filled": [], "tokens_out": 950},
                {"model": "a:advanced", "first_call_seconds": 15.0, "first_pass_valid": True, "repaired": False,
                 "filled": [], "tokens_out": 950},
                {"model": "b:basic", "first_call_seconds": 4.0, "first_pass_valid": False, "repaired": True,
                 "filled": ["added 1 table(s)"], "tokens_out": 900},
                {"model": "c:basic", "error": "no key", "first_call_seconds": None, "first_pass_valid": False}]
        lines = ev.summarise_outlines(rows)
        self.assertIn("p50  14.0s", lines[0])
        self.assertIn("valid unrepaired 2/2", lines[0])
        self.assertIn("valid unrepaired 0/1", lines[1])
        self.assertIn("repaired 1  filled 1", lines[1])
        self.assertEqual(lines[2], "c:basic: no results")

    def test_the_cli_writes_the_benchmark_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ev, "RUN_DIR", Path(directory)), \
                patch.object(settings, "default_openai_api_key", "dummy"), patch.object(tpa, "create_agent", FakeAgentFactory()), \
                patch("sys.argv", ["evaluate_blog.py", "--live", "--outline-only", "--models", "openai:advanced", "--ids", "t01"]):
            self.assertEqual(ev.main(), 0)
            rows = json.loads((Path(directory) / "blog_outline_benchmark.json").read_text())
        self.assertEqual((rows[0]["model"], rows[0]["first_pass_valid"]), ("openai:advanced", True))

    def test_bad_flags_are_rejected(self):
        for argv in (["--outline-model", "gemini:fast"], ["--outline-model", "nope"], ["--repeats", "0"]):
            with self.subTest(argv=argv), patch("sys.argv", ["evaluate_blog.py", *argv]), self.assertRaises(SystemExit):
                ev.main()


class TestLiveOutlineModel(TestLive):
    """Full posts with BLOG_OUTLINE_MODEL set from the command line, and the served-by honesty check."""

    def test_the_flag_sets_the_setting_for_the_run(self):
        seen = {}

        def stream(brief, provider, key, **kw):
            seen["outline_model"] = settings.blog_outline_model
            seen["resolver"] = kw["key_resolver"]
            return self.fake_stream(brief, provider, key, **kw)
        with patch.object(settings, "blog_outline_model", ""), patch("src.text_processing_agent.process_user_query_stream", stream), \
                tempfile.TemporaryDirectory() as directory, patch.object(ev, "RUN_DIR", Path(directory)), \
                patch.object(settings, "default_openai_api_key", "dummy"), patch.object(settings, "default_gemini_api_key", None), \
                patch.object(settings, "default_anthropic_api_key", None), patch.object(settings, "default_grok_api_key", None), \
                patch.object(settings, "default_deepseek_api_key", None), \
                patch("sys.argv", ["evaluate_blog.py", "--live", "--providers", "openai", "--ids", "t01", "--outline-model", "Gemini:Basic",
                                   "--out", str(Path(directory) / "r.json")]):
            ev.main()
            settings.blog_outline_model = ""
        self.assertEqual(seen["outline_model"], "gemini:basic")
        self.assertEqual(seen["resolver"].available_providers, {"openai"})          # only providers that have a platform key

    def test_posts_from_different_outline_models_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = {"id": "t01", "provider": "openai", "model": "m", "brief": "b", "options": {}, "metrics": {}, "seconds": 1,
                     "first_content": 1, "failures": [], "warnings": [], "content": "x", "error": None}
            a = ev.write_post(plain, Path(directory))
            b = ev.write_post({**plain, "outline_setting": "gemini:basic", "outline_model": "gemini/lite"}, Path(directory))
        self.assertEqual((a.name, b.name), ("t01_openai.md", "t01_openai_outline-gemini-basic.md"))

    def test_a_row_answered_by_a_fallback_is_not_scored_as_the_requested_model(self):
        def stream(brief, provider, key, **kw):
            trace_set(kw["request_id"], model_used="openai/gpt-5.2", outline_model="gemini/lite")
            return self.fake_stream(brief, provider, key, **kw)
        with patch("src.text_processing_agent.process_user_query_stream", stream):
            row = ev.run_one_live("claude", "k", "t01", "Write a blog post about tea", {}, True)
        self.assertTrue(any("served by openai/gpt-5.2, not claude" in f for f in row["failures"]))
        self.assertEqual(row["outline_model"], "gemini/lite")

    def test_a_row_answered_by_the_requested_model_has_no_such_failure(self):
        def stream(brief, provider, key, **kw):
            trace_set(kw["request_id"], model_used="openai/gpt-5.2")
            return self.fake_stream(brief, provider, key, **kw)
        with patch("src.text_processing_agent.process_user_query_stream", stream):
            row = ev.run_one_live("openai", "k", "t01", "Write a blog post about tea", {}, True)
        self.assertEqual(row["failures"], [])


if __name__ == "__main__":
    unittest.main()
