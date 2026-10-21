"""
Few-shot examples teach the model more strongly than its rules, so each one must itself obey the rules:
it validates, and every fact in its reply comes from its request.
"""

import unittest

from src.email_template.categories import ALL_CATEGORIES, enabled_categories, example_path, load_example
from src.email_template.prompts import system_prompt
from src.email_template.tone import iter_text, numbers_in
from src.email_template.validate import Context, validate


def _context(category_id, example):
    inp = example["input"]
    return Context(
        category=ALL_CATEGORIES[category_id], routing="selected",
        tone=ALL_CATEGORIES[category_id].default_tone, language=inp["language"],
        allowed_merge_tags=frozenset(inp["allowed_merge_tags"]), link_keys=frozenset(inp["links"]),
        asset_keys=frozenset(inp["assets"]), image_slots=tuple(inp["image_slots"]),
        source_texts=(inp["brief"], inp.get("audience", "")), sign_off=inp.get("sign_off"),
    )


class TestExamples(unittest.TestCase):
    def test_every_enabled_category_has_an_example(self):
        for cid, cat in ALL_CATEGORIES.items():
            if cat.enabled:
                self.assertTrue(example_path(cid).exists(), f"{cid} is enabled but has no example")

    def test_examples_validate_cleanly(self):
        for cid in enabled_categories():
            with self.subTest(category=cid):
                example = load_example(cid)
                result = validate(example["output"], _context(cid, example))
                self.assertTrue(result.ok, [e.as_dict() for e in result.errors])
                self.assertEqual([w.kind for w in result.warnings], [], "examples should not raise warnings")

    def test_examples_add_no_numbers(self):
        for cid in enabled_categories():
            with self.subTest(category=cid):
                example = load_example(cid)
                inp = example["input"]
                out_numbers = numbers_in(text for _, _, text in iter_text(example["output"]))
                self.assertLessEqual(out_numbers, numbers_in([inp["brief"], inp.get("audience", "")]))

    def test_system_prompt_is_stable_and_includes_example(self):
        for cid, cat in enabled_categories().items():
            with self.subTest(category=cid):
                p1, p2 = system_prompt(cat, cat.default_tone), system_prompt(cat, cat.default_tone)
                self.assertEqual(p1, p2)
                self.assertIn("# Example", p1)
                self.assertIn("[[fill:", p1)


if __name__ == "__main__":
    unittest.main()
