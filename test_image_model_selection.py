"""
Image model selection: the provider/model comes from the user's own words (regex) or an explicit
API field. The chat LLM only writes the picture prompt and never decides which model draws it.
Gemini 2.5 is the default.
"""

import threading
import time
import unittest
from unittest.mock import patch

import src.image_gen as ig
from src.image_gen import (
    ImageParams,
    detect_image_model_request,
    generate_image_unified,
    sanitize_image_params,
    strip_image_model_cue,
)
from src.models import IMAGE_MODEL_IDS, get_image_model_fallbacks, get_image_model_id


class TestDefaults(unittest.TestCase):

    def test_gemini_default_is_2_5(self):
        self.assertEqual(get_image_model_id("gemini"), "gemini-2.5-flash-image")
        self.assertEqual(IMAGE_MODEL_IDS["gemini"], "gemini-2.5-flash-image")

    def test_gemini_3_1_is_still_a_fallback(self):
        self.assertIn("gemini-3.1-flash-image", get_image_model_fallbacks("gemini"))
        self.assertNotIn("gemini-2.5-flash-image", get_image_model_fallbacks("gemini"))

    def test_no_cue_means_no_provider_or_model_in_params(self):
        p = ImageParams.from_prompt_and_overrides("a cat on a sofa")
        self.assertIsNone(p.provider)
        self.assertIsNone(p.model)


class TestRegexSelector(unittest.TestCase):

    CASES = [
        ("a cat using grok", ("grok", None)),
        ("draw a mountain with gemini", ("gemini", None)),
        ("a logo via openai", ("openai", None)),
        ("generate a car using gpt", ("openai", None)),
        ("using chatgpt, make a poster", ("openai", None)),
        ("using xai", ("grok", None)),
        ("a skyline using grok imagine", ("grok", "grok-imagine-image")),
        ("a skyline using grok 2", ("grok", "grok-imagine-image-2.0")),
        ("using gemini 2.5, draw a cat", ("gemini", "gemini-2.5-flash-image")),
        ("a poster with nano banana", ("gemini", "gemini-2.5-flash-image")),
        ("a portrait using gemini 3.1", ("gemini", "gemini-3.1-flash-image")),
        ("an image with gpt-image-1", ("openai", "gpt-image-1")),
        ("use gpt-image-1-mini for a banner", ("openai", "gpt-image-1-mini")),
        ("a dall-e style painting", ("openai", None)),
        ("generate a castle in gemini model", ("gemini", None)),
        ("a castle, grok model please", ("grok", None)),
        ("USING GROK a red car", ("grok", None)),
    ]

    NO_MATCH = [
        "a cat on a sofa",
        "a poster with the Gemini zodiac sign",
        "the gemini twins constellation",
        "a robot called Grok",
        "a phone showing Google Maps",
        "a photo with google",
        "a gpt-powered robot",
        "ask chatgpt about it",
        "an openai logo sticker",
        "",
    ]

    def test_requested_models_are_detected(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(detect_image_model_request(text), want)

    def test_brand_words_alone_do_not_switch_models(self):
        for text in self.NO_MATCH:
            with self.subTest(text=text):
                self.assertEqual(detect_image_model_request(text), (None, None))

    def test_first_mention_wins(self):
        self.assertEqual(detect_image_model_request("using grok, not gemini")[0], "grok")

    def test_none_input_is_safe(self):
        self.assertEqual(detect_image_model_request(None), (None, None))

    def test_cue_is_removed_from_the_picture_prompt(self):
        self.assertEqual(strip_image_model_cue("a cat using grok"), "a cat")
        self.assertEqual(strip_image_model_cue("using gemini 2.5, draw a cat"), "draw a cat")
        self.assertEqual(strip_image_model_cue("a castle, grok model please"), "a castle, please")

    def test_bare_model_names_and_non_cues_are_left_alone(self):
        self.assertEqual(strip_image_model_cue("a dall-e style painting"), "a dall-e style painting")
        self.assertEqual(strip_image_model_cue("a robot called Grok"), "a robot called Grok")
        self.assertEqual(strip_image_model_cue("using xai"), "using xai")  # nothing would be left


class TestImageParamsSelection(unittest.TestCase):

    def test_user_text_selects_provider_and_model(self):
        p = ImageParams.from_prompt_and_overrides("a cat using gemini 3.1")
        self.assertEqual((p.provider, p.model), ("gemini", "gemini-3.1-flash-image"))

    def test_explicit_api_field_beats_user_text(self):
        p = ImageParams.from_prompt_and_overrides("a cat using grok", {"provider": "openai"})
        self.assertEqual(p.provider, "openai")
        self.assertIsNone(p.model)

    def test_empty_selection_text_blocks_cues_in_an_llm_prompt(self):
        p = ImageParams.from_prompt_and_overrides("a cat, rendered using grok", selection_text="")
        self.assertIsNone(p.provider)

    def test_selection_text_is_what_counts_not_the_prompt(self):
        p = ImageParams.from_prompt_and_overrides("a cat", selection_text="draw a cat using grok")
        self.assertEqual(p.provider, "grok")

    def test_inherited_model_is_dropped_when_provider_changes(self):
        inherited = {"provider": "gemini", "model": "gemini-2.5-flash-image"}
        p = ImageParams.from_prompt_and_overrides("make it night using grok", inherited_params=inherited)
        self.assertEqual((p.provider, p.model), ("grok", None))

    def test_inherited_choice_persists_for_follow_ups(self):
        inherited = {"provider": "grok", "model": None}
        p = ImageParams.from_prompt_and_overrides("make it night", inherited_params=inherited)
        self.assertEqual(p.provider, "grok")

    def test_explicit_model_infers_its_provider(self):
        p = ImageParams.from_prompt_and_overrides("a cat", {"model": "gemini-3.1-flash-image"})
        self.assertEqual(p.provider, "gemini")

    def test_api_rejects_model_of_another_provider(self):
        with self.assertRaises(ValueError):
            sanitize_image_params({"provider": "gemini", "model": "gpt-image-1"})
        self.assertEqual(
            sanitize_image_params({"provider": "gemini", "model": "gemini-2.5-flash-image"})["model"],
            "gemini-2.5-flash-image",
        )


def _fake(provider_fn_name, captured):
    def fn(api_key, prompt, params, ref, rid, model_id=None):
        captured.update(provider=provider_fn_name, api_key=api_key, model_id=model_id, prompt=prompt)
        return b"x", "image/jpeg", model_id or "default-model"
    return fn


class TestGenerateRoutes(unittest.TestCase):
    """generate_image_unified end to end with the provider calls and upload mocked."""

    def setUp(self):
        self.cap = {}
        up = {"cdn_url": "https://cdn/x.jpg", "width": 1, "height": 1, "format": "jpeg", "content_type": "image/jpeg"}
        self.patches = [
            patch.object(ig, "_call_gemini_with_fallback", _fake("gemini", self.cap)),
            patch.object(ig, "_call_grok_with_fallback", _fake("grok", self.cap)),
            patch.object(ig, "_call_openai_with_fallback", _fake("openai", self.cap)),
            patch.object(ig, "upload_image_from_bytes_detailed", lambda **k: up),
            patch.object(ig.settings, "default_gemini_api_key", "plat-gemini"),
            patch.object(ig.settings, "default_grok_api_key", "plat-grok"),
            patch.object(ig.settings, "default_openai_api_key", "plat-openai"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_default_is_gemini_even_when_chat_model_is_openai(self):
        res = generate_image_unified("a cat", "openai", "user-openai-key")
        self.assertEqual((res["provider"], self.cap["api_key"]), ("gemini", "plat-gemini"))
        self.assertIsNone(self.cap["model_id"])  # adapter then uses the 2.5 default

    def test_user_words_pick_grok(self):
        res = generate_image_unified("a cat", "openai", "user-openai-key", selection_text="a cat using grok")
        self.assertEqual((res["provider"], self.cap["api_key"]), ("grok", "plat-grok"))

    def test_user_words_pick_exact_model(self):
        generate_image_unified("a cat", "claude", "k", selection_text="a cat using gemini 3.1")
        self.assertEqual((self.cap["provider"], self.cap["model_id"]), ("gemini", "gemini-3.1-flash-image"))

    def test_user_words_pick_openai(self):
        res = generate_image_unified("a cat", "gemini", "k", selection_text="a cat using gpt")
        self.assertEqual(res["provider"], "openai")

    def test_llm_written_prompt_cannot_pick_the_model(self):
        # What the LLM wrote mentions grok; what the user said does not. Empty selection_text = LLM prompt.
        res = generate_image_unified("a cat, in the style of an image made using grok", "openai", "k", selection_text="")
        self.assertEqual(res["provider"], "gemini")

    def test_user_text_still_wins_over_the_llm_prompt(self):
        res = generate_image_unified("a cat", "openai", "k", selection_text="draw a cat with gpt")
        self.assertEqual(res["provider"], "openai")

    def test_byok_user_keeps_own_provider_and_never_gets_platform_key(self):
        res = generate_image_unified("a cat", "openai", "user-openai-key", is_byok=True)
        self.assertEqual((res["provider"], self.cap["api_key"]), ("openai", "user-openai-key"))

    def test_byok_user_asking_for_a_provider_without_a_key_is_refused(self):
        with self.assertRaises(ValueError):
            generate_image_unified("a cat", "openai", "user-openai-key", is_byok=True, selection_text="using grok")

    def test_model_of_another_provider_is_never_sent(self):
        generate_image_unified("a cat", "gemini", "k", model_id="gpt-image-1")
        self.assertIsNone(self.cap["model_id"])

    def test_explicit_api_provider_and_model(self):
        generate_image_unified(
            "a cat using grok", "openai", "k",
            explicit_params={"provider": "gemini", "model": "gemini-3.1-flash-image"},
            selection_text="a cat using grok",
        )
        self.assertEqual((self.cap["provider"], self.cap["model_id"]), ("gemini", "gemini-3.1-flash-image"))


class TestAgentToolDoesNotLetLlmChoose(unittest.TestCase):
    """The LLM-facing tool passes the USER's query as the selection text and honours BYOK."""

    def _run_tool(self, factory, **kw):
        import src.text_processing_agent as tpa
        seen = {}

        def fake_generate(**kwargs):
            seen.update(kwargs)
            return {"cdn_url": "https://cdn/x.jpg", "width": 1, "height": 1, "format": "jpeg"}

        pending = {}
        tool = getattr(tpa, factory)(
            api_key="k", pending_images=pending, is_byok=kw.get("is_byok", False),
            image_ctx={"document_id": None, "explicit_params": None, "user_query": kw["user_query"]},
        )
        with patch("src.image_gen.generate_image_unified", fake_generate):
            tool("a cat, drawn using grok")  # the LLM's own wording
            deadline = time.time() + 5
            while time.time() < deadline and not all(
                isinstance(v, dict) and v.get("status") != "generating" for v in pending.values()
            ):
                time.sleep(0.02)
        return seen

    def test_all_three_tools_pass_user_query_not_llm_prompt(self):
        for factory in ("create_image_tool", "create_grok_image_tool", "create_gemini_image_tool"):
            with self.subTest(factory=factory):
                seen = self._run_tool(factory, user_query="make me a cat picture")
                self.assertEqual(seen["selection_text"], "make me a cat picture")
                self.assertIn("using grok", seen["prompt"])

    def test_byok_flag_is_forwarded(self):
        for factory in ("create_image_tool", "create_grok_image_tool", "create_gemini_image_tool"):
            with self.subTest(factory=factory):
                self.assertTrue(self._run_tool(factory, user_query="x", is_byok=True)["is_byok"])

    def test_missing_context_blocks_cues(self):
        import src.text_processing_agent as tpa
        seen = {}
        pending = {}
        tool = tpa.create_image_tool(api_key="k", pending_images=pending, image_ctx=None)
        with patch("src.image_gen.generate_image_unified",
                   lambda **kw: seen.update(kw) or {"cdn_url": "u", "width": 1, "height": 1, "format": "jpeg"}):
            tool("a cat using grok")
            time.sleep(0.3)
        self.assertEqual(seen["selection_text"], "")


if __name__ == "__main__":
    unittest.main()
