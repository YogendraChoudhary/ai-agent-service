"""
The Mode 2 pipeline end to end with a scripted model: event order, ordered emission under parallel drafting,
pictures started right after the outline, the title that opens the post while the outline is still streaming, fail-fast,
cancellation, warnings and trace fields.
"""

import threading
import time
import unittest
from unittest.mock import patch

from config import settings
from src.blog import schema as s
from src.blog.media import GeneratedImages
from src.blog.plan import BlogRequest, plan_blog
from src.blog.prompts import load_example
from src.blog.service import PICTURES_STARTED, generate_blog_stream, outline_event, title_block, title_text
from src.blog.sources import WebSources
from src.longform.markdown_ast import parse
from src.observability import finish_trace, get_trace, start_trace
from tests.blog.fakes import FakeImageTool, ScriptedLLM, filler, good_body, good_outline, with_quote

BRIEF = "Write a blog post for team leads about running remote meetings that people actually want to attend."
REQUEST = BlogRequest.from_options(BRIEF, "", {})
PLAN = plan_blog(REQUEST)


def run(llm=None, tool=None, plan=PLAN, request=REQUEST, **kw):
    """(events, text chunks, other dict chunks: the warnings) from one run."""
    llm = llm or ScriptedLLM()
    media = GeneratedImages(tool) if tool is not None else None
    chunks = list(generate_blog_stream(request, plan, llm, media, **kw))
    events = [c["__event__"] for c in chunks if isinstance(c, dict) and "__event__" in c]
    text = [c for c in chunks if isinstance(c, str)]
    other = [c for c in chunks if isinstance(c, dict) and "__event__" not in c and "__pictures_started__" not in c]
    return events, text, other


class TestHappyPath(unittest.TestCase):

    def test_the_stream_is_outline_event_then_title_then_parts_in_order(self):
        events, text, other = run(tool=FakeImageTool())
        self.assertEqual([e["type"] for e in events], ["outline"])
        self.assertEqual(len(text), 1 + 7)                                     # title block + intro, 5 sections, conclusion
        self.assertTrue(text[0].startswith("# Remote meetings people actually want to join\n\n*"))
        headings = [t.splitlines()[0] for t in text[2:]]
        self.assertEqual(headings, ["## Why most remote meetings lose people early", "## Match the format to the decision",
                                    "## Design the agenda around outcomes", "## Make participation fair across time zones",
                                    "## Close the loop after the call", "## Make every meeting earn its place"])
        self.assertEqual(other, [])                                            # clean run: no warnings chunk

    def test_the_outline_event_gives_the_editor_a_skeleton(self):
        (event,) = run(tool=FakeImageTool())[0]
        self.assertEqual(event["title"], "Remote meetings people actually want to join")
        self.assertEqual([x["id"] for x in event["sections"]], ["s1", "s2", "s3", "s4", "s5", "conclusion"])
        self.assertEqual(event["sections"][-1]["heading"], "Make every meeting earn its place")

    def test_the_post_meets_every_quota_of_the_plan(self):
        _, text, _ = run(tool=FakeImageTool())
        blocks = parse("".join(text))
        self.assertGreaterEqual(sum(b.kind == "table" for b in blocks), PLAN.tables)
        self.assertGreaterEqual(sum(b.kind == "blockquote" for b in blocks), PLAN.quotes)
        self.assertEqual(sum(b.kind == "image" for b in blocks), PLAN.images)

    def test_the_document_ends_with_one_newline(self):
        document = "".join(run(tool=FakeImageTool())[1])
        self.assertTrue(document.endswith("\n") and not document.endswith("\n\n"))

    def test_every_picture_has_a_unique_placeholder_and_a_caption_line(self):
        tool = FakeImageTool()
        document = "".join(run(tool=tool)[1])
        for pid in tool.placeholders:
            self.assertEqual(document.count(f"placeholder::{pid}"), 1)
        self.assertEqual(document.count("placeholder::"), 3)
        for block in (b for b in parse(document) if b.kind == "image"):
            self.assertEqual(len(block.text.strip().splitlines()), 2)
            self.assertTrue(block.text.strip().splitlines()[1].startswith("*"))

    def test_without_pictures_there_are_no_placeholders_and_no_warnings(self):
        plan = plan_blog(BlogRequest.from_options("Write a blog post about tea without images", "", {}))
        events, text, other = run(plan=plan, request=BlogRequest.from_options("Write a blog post about tea without images", "", {}))
        self.assertNotIn("placeholder::", "".join(text))
        self.assertEqual(other, [])

    def test_title_block(self):
        outline = s.Outline.model_validate(load_example("standard")["output"])
        self.assertEqual(title_block(outline), f"# {outline.title}\n\n*{outline.subtitle}*\n\n")
        outline.subtitle = None
        self.assertEqual(title_block(outline), f"# {outline.title}\n\n")
        self.assertEqual(outline_event(outline)["sections"][0], {"id": "s1", "heading": outline.sections[0].heading})


class TestOrderAndConcurrency(unittest.TestCase):

    def test_parts_are_emitted_in_order_even_when_later_ones_finish_first(self):
        def slow(req, spec):
            time.sleep(0.25)
            return good_body(req)
        llm = ScriptedLLM(units={"intro": slow, "s1": slow})
        _, text, _ = run(llm)
        self.assertEqual([t.splitlines()[0] for t in text[2:4]],
                         ["## Why most remote meetings lose people early", "## Match the format to the decision"])
        # the later parts really were drafted while the slow ones were still running
        starts = {c[2].unit: i for i, c in enumerate(llm.calls) if c[2].stage == "unit"}
        self.assertLess(starts["s2"], 4)

    def test_parallelism_is_bounded_by_the_setting(self):
        for limit in (1, 2):
            live, peak, lock = 0, 0, threading.Lock()

            def tracked(req, spec):
                nonlocal live, peak
                with lock:
                    live += 1
                    peak = max(peak, live)
                time.sleep(0.03)
                with lock:
                    live -= 1
                return good_body(req)
            llm = ScriptedLLM(units={u: tracked for u in ("intro", "s1", "s2", "s3", "s4", "s5", "conclusion")})
            with self.subTest(limit=limit), patch.object(settings, "blog_section_parallel", limit):
                run(llm)
                self.assertEqual(peak, limit)

    def test_the_title_opens_the_post_then_the_outline_event_then_the_pictures_before_any_part_is_drafted(self):
        tool, llm = FakeImageTool(), ScriptedLLM()
        gen = generate_blog_stream(REQUEST, PLAN, llm, GeneratedImages(tool))
        title = next(gen)                                    # written by the model; the rest of the outline is still coming
        self.assertTrue(title.startswith("# "))
        first = next(gen)
        self.assertEqual(first["__event__"]["type"], "outline")
        self.assertEqual(tool.prompts, [])                   # a client that leaves now never starts (or pays for) a picture
        self.assertEqual(next(gen), PICTURES_STARTED)        # pictures start as soon as the stream moves on...
        self.assertEqual(len(tool.prompts), 3)
        self.assertEqual([c[2].stage for c in llm.calls], ["outline"])      # ...before a single part is written
        gen.close()

    def test_with_the_early_title_off_the_outline_event_comes_first_and_the_title_follows_the_pictures(self):
        tool, llm = FakeImageTool(), ScriptedLLM()
        with patch.object(settings, "blog_stream_title", False):
            gen = generate_blog_stream(REQUEST, PLAN, llm, GeneratedImages(tool))
            first = next(gen)
            self.assertEqual(first["__event__"]["type"], "outline")
            self.assertEqual(tool.prompts, [])
            self.assertTrue(next(gen).startswith("# "))
            self.assertEqual(len(tool.prompts), 3)
            self.assertTrue(all(c[2].on_text is None for c in llm.calls))       # nothing was streamed
            gen.close()

    def test_the_first_part_is_not_held_back_by_slow_later_ones(self):
        gate = threading.Event()

        def late(req, spec):
            gate.wait(2)
            return good_body(req)
        llm = ScriptedLLM(units={u: late for u in ("s1", "s2", "s3", "s4", "s5", "conclusion")})
        gen = generate_blog_stream(REQUEST, PLAN, llm, None)
        next(gen), next(gen)                                 # outline event, title
        intro = next(gen)                                    # the intro needs nothing else to be ready
        self.assertFalse(intro.startswith("#"))
        gate.set()
        gen.close()



class TestEarlyTitle(unittest.TestCase):
    """The title opens the post while the outline is still being written. Nothing else about the post may change."""

    @staticmethod
    def document(**kw):
        with patch.object(settings, "blog_stream_title", kw.pop("early")):
            return "".join(c for c in generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), kw.pop("media", None)) if isinstance(c, str))

    def test_the_title_is_the_first_chunk_and_is_the_outlines_own_title_block(self):
        chunks = list(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), None))
        outline = s.Outline.model_validate(load_example("standard")["output"])
        self.assertEqual(chunks[0], title_block(outline))
        self.assertEqual(chunks[1]["__event__"]["type"], "outline")

    def test_the_post_is_the_same_word_for_word_with_the_title_early_or_not(self):
        early, late = self.document(early=True), self.document(early=False)
        self.assertEqual(early, late)
        self.assertTrue(early.startswith("# Remote meetings people actually want to join\n\n*"))

    def test_the_title_block_is_one_function(self):
        self.assertEqual(title_text("A title", "A subtitle"), "# A title\n\n*A subtitle*\n\n")
        self.assertEqual(title_text("A title", None), "# A title\n\n")

    def test_a_repair_cannot_change_the_title_the_reader_has_seen(self):
        first = good_outline({"plan": {"target_words": 1450, "images": 3}})
        first["title"] = "The title the reader sees"
        first["sections"] = first["sections"][:2]                       # too few sections: the first reply is unusable
        second = good_outline({"plan": {"target_words": 1450, "images": 3}})
        second.update(title="Another title", subtitle="Another subtitle")
        llm = ScriptedLLM(outline=[first, second])
        events, text, _ = run(llm)
        self.assertTrue(text[0].startswith("# The title the reader sees\n\n*"))
        self.assertEqual(events[0]["title"], "The title the reader sees")
        self.assertNotIn("Another", "".join(text))
        self.assertEqual([c[2].stage for c in llm.calls if c[2].stage.startswith("outline")], ["outline", "outline_repair"])

    def test_a_title_the_schema_would_reject_is_never_shown(self):
        bad = good_outline({"plan": {"target_words": 1450, "images": 3}})
        bad["title"] = "T" * 130
        llm = ScriptedLLM(outline=[bad, good_outline({"plan": {"target_words": 1450, "images": 3}})])
        text = "".join(c for c in generate_blog_stream(REQUEST, PLAN, llm, None) if isinstance(c, str))
        self.assertNotIn("T" * 130, text)
        self.assertTrue(text.startswith("# Remote meetings people actually want to join"))      # the repaired reply's own title

    def test_an_outline_that_fails_after_its_title_raises_in_the_stream(self):
        broken = good_outline({"plan": {"target_words": 1450, "images": 3}})
        broken["sections"] = "nope"                                     # a fine title and subtitle, an unusable structure
        tool = FakeImageTool()
        gen = generate_blog_stream(REQUEST, PLAN, ScriptedLLM(outline=[broken, broken]), GeneratedImages(tool))
        self.assertTrue(next(gen).startswith("# Remote meetings"))
        with self.assertRaises(s.BlogError):
            next(gen)
        self.assertEqual(tool.prompts, [])                              # nothing billed for a post that never got an outline

    def test_a_model_error_during_the_outline_is_raised_in_the_stream(self):
        def down(req, spec):
            raise RuntimeError("503 overloaded")
        with self.assertRaisesRegex(RuntimeError, "503"):
            list(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(outline=down), None))

    def test_a_client_that_leaves_after_the_title_costs_no_picture(self):
        tool = FakeImageTool()
        gen = generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), GeneratedImages(tool))
        next(gen)
        gen.close()
        self.assertEqual(tool.prompts, [])

    def test_the_pictures_marker_is_sent_only_when_pictures_started_after_the_title(self):
        def kinds(chunks):
            return ["text" if isinstance(c, str) else next(iter(c)) for c in chunks]
        started = kinds(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), GeneratedImages(FakeImageTool())))
        self.assertEqual(started[:3], ["text", "__event__", "__pictures_started__"])
        self.assertNotIn("__pictures_started__", kinds(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), None)))
        failing = GeneratedImages(FakeImageTool(fail=lambda prompt: "ERROR: no credits"))
        self.assertNotIn("__pictures_started__", kinds(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), failing)))
        with patch.object(settings, "blog_stream_title", False):
            off = kinds(generate_blog_stream(REQUEST, PLAN, ScriptedLLM(), GeneratedImages(FakeImageTool())))
        self.assertEqual(off[:2], ["__event__", "text"])
        self.assertNotIn("__pictures_started__", off)


class TestFailure(unittest.TestCase):

    def test_a_provider_error_in_a_part_stops_the_post_and_cancels_the_rest(self):
        def down(req, spec):
            raise RuntimeError("503 overloaded")
        llm = ScriptedLLM(units={"s1": down})
        with patch.object(settings, "blog_section_parallel", 1):
            with self.assertRaisesRegex(RuntimeError, "503"):
                run(llm)
        time.sleep(0.1)
        called = {c[2].unit for c in llm.calls if c[2].stage == "unit"}
        self.assertEqual(called, {"intro", "s1"})            # s2.. never started

    def test_an_empty_part_is_a_blog_error(self):
        with self.assertRaises(s.BlogError):
            run(ScriptedLLM(units={"s3": ["", ""]}))

    def test_an_unusable_outline_fails_before_any_picture_or_text(self):
        tool, llm = FakeImageTool(), ScriptedLLM(outline="nope")
        with self.assertRaises(s.BlogError):
            run(llm, tool)
        self.assertEqual(tool.prompts, [])                   # nothing billed for a post that never started
        self.assertEqual([c[2].stage for c in llm.calls], ["outline", "outline_repair"])

    def test_closing_the_stream_cancels_parts_that_have_not_started(self):
        llm = ScriptedLLM(units={u: (lambda req, spec: (time.sleep(0.05), good_body(req))[1])
                                 for u in ("intro", "s1", "s2", "s3", "s4", "s5", "conclusion")})
        with patch.object(settings, "blog_section_parallel", 1):
            gen = generate_blog_stream(REQUEST, PLAN, llm, None)
            next(gen), next(gen), next(gen)                  # outline event, title, intro
            gen.close()
            time.sleep(0.4)
        self.assertLess(len([c for c in llm.calls if c[2].stage == "unit"]), 7)


class TestWarnings(unittest.TestCase):

    def warnings(self, *a, **kw):
        other = run(*a, **kw)[2]
        return other[0]["__warnings__"] if other else []

    def test_a_picture_that_could_not_start_is_reported_and_not_referenced(self):
        tool = FakeImageTool(fail=lambda p: "Image generation skipped: insufficient credits" if "world map" in p else None)
        events, text, other = run(tool=tool)
        (w,) = [w for w in other[0]["__warnings__"] if w["kind"] == "image_skipped"]
        self.assertEqual((w["unit"], "insufficient credits" in w["detail"]), ("s4", True))
        self.assertEqual("".join(text).count("placeholder::"), 2)

    def test_every_picture_failing_still_delivers_the_post(self):
        tool = FakeImageTool(fail=lambda p: "ERROR: image generation is not available for this account.")
        _, text, other = run(tool=tool)
        self.assertEqual("".join(text).count("placeholder::"), 0)
        self.assertEqual(sum(w["kind"] == "image_skipped" for w in other[0]["__warnings__"]), 3)

    def test_asking_for_more_pictures_than_the_post_gets_is_explained(self):
        request = BlogRequest.from_options("Write a blog post about tea with 9 images", "", {})
        plan = plan_blog(request)
        ws = self.warnings(plan=plan, request=request, tool=FakeImageTool())
        (w,) = [w for w in ws if w["kind"] == "images_limited"]
        self.assertIn("9 images were asked for", w["detail"])

    def test_no_provider_for_pictures_the_user_asked_for_is_explained(self):
        request = BlogRequest.from_options("Write a blog post about tea with 2 images", "", {})
        plan = plan_blog(request, image_available=False)
        ws = self.warnings(plan=plan, request=request)
        self.assertEqual([w["kind"] for w in ws], ["images_limited"])
        self.assertIn("no usable image provider", ws[0]["detail"])

    def test_a_part_that_stays_wrong_after_its_repair_is_a_warning_not_a_failure(self):
        llm = ScriptedLLM(units={"s2": f"{filler(240)}\n"})            # never writes its table or quote
        _, text, other = run(llm, FakeImageTool())
        (w,) = [w for w in other[0]["__warnings__"] if w["kind"] == "unit_lint"]
        self.assertEqual(w["unit"], "s2")
        self.assertIn("## Match the format to the decision", "".join(text))
        kinds = {x["kind"] for x in other[0]["__warnings__"]}
        self.assertTrue({"quota_tables", "quota_quotes"} <= kinds)     # and the whole-post check notices

    def test_a_filled_outline_is_reported(self):
        outline = good_outline({"plan": {"target_words": 1400, "images": 3}})
        for sec in outline["sections"]:
            sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != "table"]
        ws = self.warnings(ScriptedLLM(outline=outline), FakeImageTool())
        self.assertEqual([w["kind"] for w in ws], ["outline_filled"])

    def test_a_quote_attributed_to_a_real_person_is_cleaned_without_a_warning(self):
        body = lambda req, spec: with_quote(good_body(req), "> A quote of note.\n> — Jane Doe, CEO of Acme")
        _, text, other = run(ScriptedLLM(units={"s3": body}), FakeImageTool())
        self.assertNotIn("Jane Doe", "".join(text))
        self.assertIn("> A quote of note.", "".join(text))
        self.assertEqual(other, [])

    def test_a_page_that_was_read_allows_its_attribution(self):
        body = lambda req, spec: with_quote(good_body(req), "> Focus wins.\n> — [Reuters](https://reuters.com/a)")
        _, text, _ = run(ScriptedLLM(units={"s3": body}), FakeImageTool(), sources=WebSources(["https://reuters.com/a"]))
        self.assertIn("[Reuters](https://reuters.com/a)", "".join(text))


class TestTrace(unittest.TestCase):

    def test_the_trace_records_what_was_built(self):
        start_trace("blogtrace", "/agent/stream")
        try:
            run(tool=FakeImageTool(), request_id="blogtrace")
            attrs = get_trace("blogtrace").attrs
            marks = get_trace("blogtrace").marks
        finally:
            finish_trace("blogtrace")
        self.assertEqual(attrs["mode"], "blog")
        blog = attrs["blog"]
        self.assertEqual((blog["sections"], blog["tables"], blog["quotes"], blog["images"]), (5, 1, 2, 3))
        self.assertEqual((blog["repairs"], blog["warnings"]), (0, 0))
        self.assertIn("outline_seconds", blog)
        self.assertIn("draft_seconds", blog)
        self.assertLessEqual(blog["title_seconds"], blog["outline_seconds"])         # shown while the outline was still streaming
        self.assertGreater(blog["words"], 800)
        self.assertEqual(attrs["blog_plan"]["words"], 1450)
        self.assertTrue({"title_streamed", "outline_done", "first_unit", "draft_done"} <= set(marks))

    def test_without_the_early_title_the_trace_says_there_was_none(self):
        start_trace("blogtrace2", "/agent/stream")
        try:
            with patch.object(settings, "blog_stream_title", False):
                run(tool=FakeImageTool(), request_id="blogtrace2")
            attrs, marks = get_trace("blogtrace2").attrs, get_trace("blogtrace2").marks
        finally:
            finish_trace("blogtrace2")
        self.assertIsNone(attrs["blog"]["title_seconds"])
        self.assertNotIn("title_streamed", marks)

    def test_repairs_and_fixes_are_counted(self):
        bad = good_outline({"plan": {"target_words": 1400, "images": 3}})
        bad["sections"][1]["visuals"] = [v for v in bad["sections"][1]["visuals"] if v["kind"] != "table"]
        llm = ScriptedLLM(outline=[bad, good_outline({"plan": {"target_words": 1400, "images": 3}})],
                          units={"s2": [f"## H\n\n{filler(200)}\n\n> {filler(10)}\n", None]})
        start_trace("blogtrace2", "/agent/stream")
        try:
            run(llm, FakeImageTool(), request_id="blogtrace2")
            blog = get_trace("blogtrace2").attrs["blog"]
        finally:
            finish_trace("blogtrace2")
        self.assertEqual(blog["repairs"], 2)                  # the outline and s2
        self.assertEqual(blog["fixes"].get("heading"), 1)


if __name__ == "__main__":
    unittest.main()
