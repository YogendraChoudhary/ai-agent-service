"""
Prompt / image memory for follow-ups ("make it night", "regenerate").

Covers storage scoping (per document), bounded prompt chaining, parameter inheritance, and
the Redis client lookup.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import src.image_gen as ig
from src.intent import detect_image_intent
from src.image_gen import (
    ImageParams,
    save_image_context,
    get_previous_image_context,
    compose_followup_prompt,
)


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)


class TestPromptReuse(unittest.TestCase):

    def setUp(self):
        self.redis = FakeRedis()
        patcher = patch("src.image_gen._get_redis_client", return_value=self.redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _save(self, prompt="draw a mosque at dusk", doc=None, **params):
        save_image_context(
            user_id="u1",
            final_prompt=prompt,
            params=ImageParams.from_prompt_and_overrides(prompt, explicit_params=params),
            cdn_url="https://cdn.example.com/u1/ai-images/a.webp",
            provider="gemini",
            document_id=doc,
            width=1536,
            height=1024,
        )

    def test_two_turn_flow_inherits_params_and_prompt(self):
        self._save(aspect_ratio="16:9", format="webp")
        ctx = get_previous_image_context("u1")
        self.assertEqual(ctx["cdn_url"], "https://cdn.example.com/u1/ai-images/a.webp")

        intent = detect_image_intent("make it night", has_previous_image=True)
        self.assertEqual((intent["kind"], intent["followup_mode"]), ("edit", "delta"))

        final, base, mods = compose_followup_prompt(ctx, "make it night", "delta")
        self.assertEqual(final, "draw a mosque at dusk. Modifications: make it night")
        self.assertEqual(base, "draw a mosque at dusk")

        params = ImageParams.from_prompt_and_overrides("make it night", inherited_params=ctx["params"])
        self.assertEqual((params.aspect_ratio, params.format), ("16:9", "webp"))

    def test_redo_reuses_prompt_verbatim(self):
        self._save()
        ctx = get_previous_image_context("u1")
        final, _, _ = compose_followup_prompt(ctx, "regenerate", "redo")
        self.assertEqual(final, "draw a mosque at dusk")

    def test_prompt_regex_beats_inherited_but_explicit_beats_both(self):
        inherited = {"aspect_ratio": "16:9", "quality": "low"}
        p = ImageParams.from_prompt_and_overrides("make it a vertical 9:16 story", inherited_params=inherited)
        self.assertEqual(p.aspect_ratio, "9:16")      # prompt overrides inherited
        self.assertEqual(p.quality, "low")            # untouched setting still inherited
        p2 = ImageParams.from_prompt_and_overrides(
            "make it a vertical 9:16 story", explicit_params={"aspect_ratio": "1:1"}, inherited_params=inherited
        )
        self.assertEqual(p2.aspect_ratio, "1:1")      # explicit wins

    def test_chained_edits_stay_bounded(self):
        self._save()
        ctx = get_previous_image_context("u1")
        base = ctx["base_prompt"]
        mods = []
        for n in range(6):
            final, base, mods = compose_followup_prompt(
                {"final_prompt": "x", "base_prompt": base, "modifications": mods}, f"change {n}", "delta"
            )
        self.assertEqual(len(mods), ig.MAX_MODIFICATIONS)
        self.assertEqual(mods, ["change 3", "change 4", "change 5"])
        self.assertEqual(final.count("Modifications:"), 1)
        self.assertTrue(final.startswith("draw a mosque at dusk. Modifications:"))

    def test_document_scoping_never_leaks_across_documents(self):
        self._save("draw a red barn", doc="docA")
        self._save("draw a blue boat", doc="docB")

        self.assertEqual(get_previous_image_context("u1", "docA")["final_prompt"], "draw a red barn")
        self.assertEqual(get_previous_image_context("u1", "docB")["final_prompt"], "draw a blue boat")
        # A document with no image of its own must NOT fall back to another document's image.
        self.assertIsNone(get_previous_image_context("u1", "docC"))
        # Clients that send no document id still get the per-user entry (most recent save).
        self.assertEqual(get_previous_image_context("u1")["final_prompt"], "draw a blue boat")

    def test_no_user_no_memory(self):
        save_image_context("", "p", ImageParams(), "u", "gemini")
        self.assertEqual(self.redis.data, {})
        self.assertIsNone(get_previous_image_context("", None))


class TestRedisClientLookup(unittest.TestCase):

    def setUp(self):
        ig._REDIS_RETRY_AFTER = 0.0
        self.addCleanup(setattr, ig, "_REDIS_RETRY_AFTER", 0.0)

    def test_uses_process_singleton_not_a_new_manager(self):
        manager = MagicMock()
        with patch("src.redis_manager.get_redis_manager", return_value=manager) as getter:
            self.assertIs(ig._get_redis_client(), manager.redis_client)
            self.assertIs(ig._get_redis_client(), manager.redis_client)
        self.assertEqual(getter.call_count, 2)  # singleton accessor, never RedisManager()

    def test_failure_triggers_cooldown_instead_of_retrying_every_request(self):
        with patch("src.redis_manager.get_redis_manager", side_effect=RuntimeError("down")) as getter:
            self.assertIsNone(ig._get_redis_client())
            self.assertIsNone(ig._get_redis_client())
            self.assertIsNone(ig._get_redis_client())
        self.assertEqual(getter.call_count, 1)


if __name__ == "__main__":
    unittest.main()
