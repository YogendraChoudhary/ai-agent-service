"""The LONGFORM_* settings: what ships, what the environment can change, what is rejected, and how they reach the pipeline."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from config import Settings, settings
from src.longform.refine import RefineConfig
from src.longform.runtime import refine_config


def env(**values):
    return patch.dict(os.environ, {k: str(v) for k, v in values.items()})


def clean_env():
    cleaned = {k: v for k, v in os.environ.items() if not k.startswith("LONGFORM_")}
    return patch.dict(os.environ, cleaned, clear=True)


class TestDefaults(unittest.TestCase):

    def test_the_pipeline_ships_off_and_with_the_plans_numbers(self):
        with clean_env():
            s = Settings(_env_file=None)
        self.assertEqual(
            (s.longform_refine_enabled, s.longform_refine_min_words, s.longform_chunk_words, s.longform_parallel,
             s.longform_freeze_quotes, s.longform_repair_enabled, s.longform_word_ratio),
            (False, 600, 1000, 5, True, True, ""))
        self.assertEqual((s.longform_first_chunk_words, s.longform_call_timeout_seconds), (300, 150.0))

    def test_the_environment_turns_it_on(self):
        for raw, want in (("true", True), ("1", True), ("false", False), ("0", False)):
            with self.subTest(raw=raw), env(LONGFORM_REFINE_ENABLED=raw):
                self.assertEqual(Settings(_env_file=None).longform_refine_enabled, want)

    def test_the_numbers_are_bounded(self):
        for name, bad in (("LONGFORM_PARALLEL", 0), ("LONGFORM_PARALLEL", 9), ("LONGFORM_CHUNK_WORDS", 50),
                          ("LONGFORM_CHUNK_WORDS", 5000), ("LONGFORM_REFINE_MIN_WORDS", 0), ("LONGFORM_FIRST_CHUNK_WORDS", -1),
                          ("LONGFORM_FIRST_CHUNK_WORDS", 4000), ("LONGFORM_CALL_TIMEOUT_SECONDS", -1),
                          ("LONGFORM_CALL_TIMEOUT_SECONDS", 1000)):
            with self.subTest(name=name, value=bad), env(**{name: bad}):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None)
        with env(LONGFORM_PARALLEL=8, LONGFORM_CHUNK_WORDS=3000, LONGFORM_REFINE_MIN_WORDS=1, LONGFORM_FIRST_CHUNK_WORDS=0):
            s = Settings(_env_file=None)
            self.assertEqual((s.longform_parallel, s.longform_chunk_words, s.longform_refine_min_words, s.longform_first_chunk_words),
                             (8, 3000, 1, 0))


class TestTheWordRatioOverride(unittest.TestCase):

    def parse(self, value):
        with env(LONGFORM_WORD_RATIO=value):
            return Settings(_env_file=None).longform_word_ratio

    def test_valid_overrides_are_kept_as_written(self):
        for raw in ('{"grammar": [0.9, 1.1]}', '{"simplify": [0.5, 1], "tone": [0.8, 1.3]}', '{"enhance": [1, 1]}',
                    ' {"grammar": [0.9, 1.1]} '):
            with self.subTest(raw=raw):
                self.assertEqual(self.parse(raw), raw.strip())

    def test_anything_else_means_the_per_operation_windows(self):
        for raw in ("not json", "[0.9, 1.1]", '{"grammar": 1}', '{"grammar": [0.9]}', '{"grammar": [0.9, 1.1, 1.3]}',
                    '{"grammar": [1.1, 0.9]}', '{"grammar": [0, 1.1]}', '{"grammar": [0.9, 0.95]}', '{"grammar": ["a", "b"]}',
                    '{"grammar": [true, true]}', '{"translate": [0.9, 1.1]}', '{"grammar": [0.9, 1.1], "bogus": [0.9, 1.1]}'):
            with self.subTest(raw=raw):
                self.assertEqual(self.parse(raw), "")

    def test_empty_is_empty(self):
        self.assertEqual(self.parse(""), "")
        self.assertEqual(self.parse("   "), "")


class TestTheConfigThePipelineGets(unittest.TestCase):

    def test_the_settings_become_the_pipelines_knobs(self):
        with patch.object(settings, "longform_chunk_words", 700), patch.object(settings, "longform_parallel", 2), \
                patch.object(settings, "longform_freeze_quotes", False), patch.object(settings, "longform_repair_enabled", False), \
                patch.object(settings, "longform_refine_min_words", 250), patch.object(settings, "longform_first_chunk_words", 200), \
                patch.object(settings, "longform_word_ratio", '{"grammar": [0.9, 1.1]}'):
            self.assertEqual(refine_config(), RefineConfig(chunk_words=700, first_chunk_words=200, parallel=2, freeze_quotes=False,
                                                           repair=False, min_words=250, word_ratio='{"grammar": [0.9, 1.1]}'))

    def test_an_empty_ratio_override_is_no_override(self):
        with patch.object(settings, "longform_word_ratio", ""):
            self.assertIsNone(refine_config().word_ratio)

    def test_the_defaults_are_the_dataclass_defaults_except_the_small_first_chunk(self):
        """RefineConfig() is the plan's uniform chunking; the setting ships the small first chunk on (0 turns it off)."""
        with clean_env():
            s = Settings(_env_file=None)
        with patch.object(settings, "longform_chunk_words", s.longform_chunk_words), \
                patch.object(settings, "longform_parallel", s.longform_parallel), \
                patch.object(settings, "longform_freeze_quotes", s.longform_freeze_quotes), \
                patch.object(settings, "longform_repair_enabled", s.longform_repair_enabled), \
                patch.object(settings, "longform_refine_min_words", s.longform_refine_min_words), \
                patch.object(settings, "longform_first_chunk_words", s.longform_first_chunk_words), \
                patch.object(settings, "longform_word_ratio", s.longform_word_ratio):
            self.assertEqual(refine_config(), RefineConfig(first_chunk_words=300))
            self.assertEqual(RefineConfig().first_chunk_words, 0)


if __name__ == "__main__":
    unittest.main()
