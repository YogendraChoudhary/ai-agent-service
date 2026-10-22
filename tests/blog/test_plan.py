"""
plan_blog: every row of the plan's table (§4.3). Code decides length, sections, visual quotas, images and web
access from the user's own words and options, before the model writes anything.
"""

import unittest
from unittest.mock import patch

from config import settings
from src.blog.plan import (
    BRIEF_MAX_CHARS, LENGTH_WORDS, MAX_WORDS, MIN_WORDS, BlogRequest, length_bucket, plan_blog, section_count,
)
from src.request_plan import extract_word_count


def plan_for(brief="Write a blog post about remote team meetings", options=None, **kw):
    return plan_blog(BlogRequest.from_options(brief, "", options), **kw)


class TestLength(unittest.TestCase):

    def test_default_is_standard(self):
        self.assertEqual(plan_for().target_words, LENGTH_WORDS["standard"])

    def test_option_picks_the_bucket(self):
        for name, words in LENGTH_WORDS.items():
            with self.subTest(length=name):
                self.assertEqual(plan_for(options={"length": name}).target_words, words)

    def test_the_briefs_own_number_wins_over_the_option(self):
        p = plan_for("Write a 2,500-word guide to solar panels", {"length": "short"})
        self.assertEqual((p.target_words, p.length), (2500, "long"))

    def test_absurd_lengths_are_clamped_and_the_reason_says_so(self):
        for brief, want in (("Write a 90000 word novel", MAX_WORDS), ("Write a 120 word note", MIN_WORDS)):
            with self.subTest(brief=brief):
                p = plan_for(brief)
                self.assertEqual(p.target_words, want)
                self.assertIn("clamped", p.reasons["length"])

    def test_extract_word_count(self):
        self.assertEqual(extract_word_count("a 1,500-word post"), 1500)
        self.assertEqual(extract_word_count("about 800 words please"), 800)
        self.assertIsNone(extract_word_count("a short post"))
        self.assertIsNone(extract_word_count(None))


class TestSectionsAndQuotas(unittest.TestCase):

    def test_section_count_by_length(self):
        for words, sections in ((800, 4), (1000, 4), (1001, 5), (1400, 5), (1600, 5), (1601, 6), (2200, 6), (2201, 7)):
            with self.subTest(words=words):
                self.assertEqual(section_count(words), sections)

    def test_buckets(self):
        self.assertEqual([length_bucket(w) for w in (800, 1000, 1400, 1600, 2200)],
                         ["short", "short", "standard", "standard", "long"])

    def test_visual_quotas_grow_from_two_thousand_words(self):
        small, big = plan_for(options={"length": "standard"}), plan_for(options={"length": "long"})
        self.assertEqual((small.tables, small.quotes), (1, 2))
        self.assertEqual((big.tables, big.quotes), (2, 3))

    def test_section_words_are_what_is_left_after_intro_and_conclusion(self):
        for name in LENGTH_WORDS:
            p = plan_for(options={"length": name})
            with self.subTest(length=name):
                self.assertEqual(p.section_words + p.intro_words + p.conclusion_words, p.target_words)
                self.assertTrue(80 <= p.intro_words <= 180)

    def test_prose_run_comes_from_config(self):
        self.assertEqual(plan_for().max_prose_run, settings.blog_max_prose_run)
        with patch.object(settings, "blog_max_prose_run", 150):
            self.assertEqual(plan_for().max_prose_run, 150)


class TestImages(unittest.TestCase):

    def test_auto_is_one_plus_one_per_seven_hundred_words(self):
        self.assertEqual([plan_for(options={"length": n}).images for n in ("short", "standard", "long")], [2, 3, 4])

    def test_the_briefs_count_wins_over_the_option(self):
        p = plan_for("Write a blog post about tea with 2 images", {"images": 4})
        self.assertEqual((p.images, p.images_asked), (2, 2))

    def test_the_option_wins_over_auto(self):
        p = plan_for(options={"images": 1})
        self.assertEqual((p.images, p.images_asked), (1, 1))

    def test_zero_is_a_valid_option(self):
        self.assertEqual(plan_for(options={"images": 0}).images, 0)

    def test_the_brief_can_ask_for_none(self):
        for brief in ("Write a blog post about tea without images", "Write a text-only post about tea",
                      "Write about tea, no pictures please", "write a post on tea with no images"):
            with self.subTest(brief=brief):
                self.assertEqual(plan_for(brief, {"images": 3}).images, 0)

    def test_capped_by_config_and_by_section_count(self):
        p = plan_for("Write a blog post about tea with 9 images")
        self.assertEqual((p.images, p.images_asked), (min(settings.image_max_per_request, p.sections), 9))
        with patch.object(settings, "image_max_per_request", 2):
            self.assertEqual(plan_for("Write a blog post about tea with 4 images").images, 2)

    def test_no_provider_means_no_images_and_the_reason(self):
        p = plan_for(image_available=False)
        self.assertEqual(p.images, 0)
        self.assertIn("no usable image provider", p.reasons["images"])

    def test_only_the_generate_backend_exists_today(self):
        for source in ("none", "stock", ""):
            with self.subTest(source=source), patch.object(settings, "blog_image_source", source):
                self.assertEqual(plan_for().images, 0)

    def test_asking_is_recorded_only_when_the_user_asked(self):
        self.assertIsNone(plan_for().images_asked)


class TestWeb(unittest.TestCase):

    def test_web_follows_the_briefs_freshness_cues(self):
        self.assertTrue(plan_for("Write a blog post about the latest AI trends in 2026").use_web)
        self.assertFalse(plan_for("Write a blog post about running better meetings").use_web)

    def test_kill_switch_restores_always_on(self):
        with patch.object(settings, "web_tools_gate_enabled", False):
            self.assertTrue(plan_for().use_web)


class TestRequest(unittest.TestCase):

    def test_options_are_normalised(self):
        r = BlogRequest.from_options("  topic  ", " notes ", {"length": "long", "images": "auto", "tone": "warm",
                                                                "audience": "founders"})
        self.assertEqual((r.brief, r.style_notes, r.length, r.images, r.tone, r.audience),
                         ("topic", "notes", "long", None, "warm", "founders"))

    def test_missing_options_and_notes(self):
        r = BlogRequest.from_options("topic", None, None)
        self.assertEqual((r.style_notes, r.length, r.images), ("", None, None))

    def test_a_bool_is_not_an_image_count(self):
        self.assertIsNone(BlogRequest.from_options("t", "", {"images": True}).images)

    def test_brief_limit_is_one_number(self):
        self.assertEqual(BRIEF_MAX_CHARS, 5000)


if __name__ == "__main__":
    unittest.main()
