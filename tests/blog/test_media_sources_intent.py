"""Pictures (the app's image tool behind a small interface), the record of pages read, and the blog intent regex."""

import unittest
from unittest.mock import patch

from config import settings
from src.blog import schema as s
from src.blog.intent import detect_blog_intent
from src.blog.media import IMAGE_STYLE_SUFFIX, GeneratedImages, Placed, get_source, start_images
from src.blog.plan import BRIEF_MAX_CHARS
from src.blog.prompts import load_example
from src.blog.sources import WebSources, normalise_url
from tests.blog.fakes import FakeImageTool

OUTLINE = s.Outline.model_validate(load_example("standard")["output"])


class TestGeneratedImages(unittest.TestCase):

    def test_the_prompt_gets_the_house_rules_appended_by_code(self):
        tool = FakeImageTool()
        placed = GeneratedImages(tool).fulfil(OUTLINE.sections[0].of_kind("image")[0])
        self.assertEqual(tool.prompts[0], OUTLINE.sections[0].of_kind("image")[0].prompt + ". " + IMAGE_STYLE_SUFFIX)
        self.assertEqual(placed.placeholder_id, tool.placeholders[0])

    def test_an_error_string_becomes_a_reason_not_a_dead_placeholder(self):
        for message, reason in (("Image generation skipped: insufficient credits", "insufficient credits"),
                                ("ERROR: image generation is not available for this account. Write without images.",
                                 "image generation is not available for this account. Write without images.")):
            with self.subTest(message=message):
                tool = FakeImageTool(fail=lambda _p, m=message: m)
                placed = GeneratedImages(tool).fulfil(OUTLINE.sections[0].of_kind("image")[0])
                self.assertEqual((placed.placeholder_id, placed.reason), (None, reason))

    def test_a_tool_that_raises_never_takes_the_post_down(self):
        def boom(_prompt):
            raise RuntimeError("provider exploded")
        placed = GeneratedImages(boom).fulfil(OUTLINE.sections[0].of_kind("image")[0])
        self.assertEqual((placed.placeholder_id, placed.reason), (None, "provider exploded"))

    def test_start_images_is_keyed_by_section_in_outline_order(self):
        tool = FakeImageTool()
        placed = start_images(GeneratedImages(tool), OUTLINE)
        self.assertEqual(list(placed), ["s1", "s3", "s4"])
        self.assertEqual([p.placeholder_id for p in placed.values()], tool.placeholders)

    def test_no_source_means_no_pictures_with_a_reason(self):
        placed = start_images(None, OUTLINE)
        self.assertEqual({p.placeholder_id for p in placed.values()}, {None})
        self.assertEqual({p.reason for p in placed.values()}, {"no image backend"})

    def test_backends(self):
        tool = FakeImageTool()
        self.assertIsInstance(get_source("generate", tool), GeneratedImages)
        self.assertIsInstance(get_source(" Generate ", tool), GeneratedImages)
        for name in ("none", "", "stock", "diagram"):
            with self.subTest(name=name):
                self.assertIsNone(get_source(name, tool))
        self.assertIsNone(get_source("generate", None))


class TestWebSources(unittest.TestCase):

    def test_two_spellings_of_a_page_are_one_page(self):
        sources = WebSources(["https://www.Example.com/path/to/page/#section"])
        for url in ("http://example.com/path/to/page", "example.com/path/to/page/", "https://example.com/path/to/page.",
                    "https://example.com/path/to/page)"):
            with self.subTest(url=url):
                self.assertTrue(sources.allows(url))

    def test_a_different_page_or_query_is_a_different_page(self):
        sources = WebSources(["https://example.com/a?id=1"])
        self.assertFalse(sources.allows("https://example.com/a?id=2"))
        self.assertFalse(sources.allows("https://example.com/b"))
        self.assertTrue(sources.allows("https://example.com/a?id=1"))

    def test_truthiness_and_thread_safe_adds(self):
        import threading
        sources = WebSources()
        self.assertFalse(sources)
        threads = [threading.Thread(target=sources.add, args=(f"https://x.test/{i}",)) for i in range(40)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(sources), 40)
        self.assertTrue(sources)

    def test_normalise(self):
        self.assertEqual(normalise_url("HTTPS://WWW.Foo.com/Bar/"), "foo.com/Bar")


class TestBlogIntent(unittest.TestCase):

    def kind(self, text):
        return detect_blog_intent(text)["kind"]

    def test_explicit_requests_at_the_start_match(self):
        for text in ("Write a blog post about remote work", "write a blog about tea", "Draft an article on solar power",
                     "Please write me a guide covering onboarding", "can you create a long-form piece about sleep",
                     "write a detailed blog post about React hooks", "I want you to write a short article on tea",
                     "Write an in-depth guide about budgeting"):
            with self.subTest(text=text):
                self.assertEqual(self.kind(text), "blog")

    def test_mentions_and_other_work_do_not_match(self):
        for text in ("Fix my blog post about tea", "summarise this article about tea", "my article on solar needs a rewrite",
                     "Rewrite the blog post about tea", "Write a poem about tea", "Write a function that parses tea",
                     "What is a blog post?", "I read a blog post about tea yesterday", "", "Write a blog"):
            with self.subTest(text=text):
                self.assertEqual(self.kind(text), "other")

    def test_a_long_pasted_document_is_never_a_brief(self):
        self.assertEqual(self.kind("Write a blog post about tea. " + "x" * BRIEF_MAX_CHARS), "other")

    def test_the_decision_names_its_rule(self):
        self.assertEqual(detect_blog_intent("Write a blog post about tea")["rule"], "anchored_blog")
        self.assertEqual(detect_blog_intent("hello")["rule"], "no_blog_cue")

    def test_ships_off(self):
        self.assertFalse(settings.blog_intent_regex_enabled)
        self.assertFalse(settings.blog_intent_shadow_mode)
        self.assertFalse(settings.blog_ai_enabled)


if __name__ == "__main__":
    unittest.main()
