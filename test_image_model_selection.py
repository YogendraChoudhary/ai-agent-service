"""
Image model selection: the provider/model comes from the user's own words (regex) or an explicit
API field. The chat LLM only writes the picture prompt and never decides which model draws it.
Gemini 2.5 is the default.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

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

    def test_gemini_default_is_the_lite_model(self):
        self.assertEqual(get_image_model_id("gemini"), "gemini-3.1-flash-lite-image")
        self.assertEqual(IMAGE_MODEL_IDS["gemini"], "gemini-3.1-flash-lite-image")

    def test_older_gemini_models_remain_fallbacks_in_order(self):
        fb = get_image_model_fallbacks("gemini")
        self.assertEqual(fb[:2], ["gemini-2.5-flash-image", "gemini-3.1-flash-image"])
        self.assertNotIn("gemini-3.1-flash-lite-image", fb)

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
        ("a portrait using gemini 3.1 lite", ("gemini", "gemini-3.1-flash-lite-image")),
        ("a portrait with gemini lite", ("gemini", "gemini-3.1-flash-lite-image")),
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


class TestGeminiSizeRouting(unittest.TestCase):
    """The default lite model only draws 1K; larger requests go to the model that really delivers them."""

    def test_default_1k_stays_on_lite(self):
        from src.models import gemini_model_for_resolution
        self.assertEqual(gemini_model_for_resolution(None, "1K"), "gemini-3.1-flash-lite-image")

    def test_2k_and_4k_route_to_the_hires_model_when_no_model_was_named(self):
        from src.models import gemini_model_for_resolution
        for res in ("2K", "4K"):
            self.assertEqual(gemini_model_for_resolution(None, res), "gemini-3.1-flash-image")

    def test_a_model_the_user_named_is_kept(self):
        from src.models import gemini_model_for_resolution
        self.assertEqual(gemini_model_for_resolution("gemini-3.1-flash-lite-image", "4K"), "gemini-3.1-flash-lite-image")

    def test_unsupported_size_is_clamped_never_sent(self):
        from src.models import gemini_effective_resolution
        self.assertEqual(gemini_effective_resolution("gemini-3.1-flash-lite-image", "2K"), "1K")
        self.assertEqual(gemini_effective_resolution("gemini-2.5-flash-image", "4K"), "1K")   # silently 1024px otherwise
        self.assertEqual(gemini_effective_resolution("gemini-3.1-flash-image", "4K"), "4K")
        self.assertEqual(gemini_effective_resolution("some-future-model", "4K"), "4K")        # unknown: pass through

    def test_credits_are_priced_for_the_model_that_will_run(self):
        from src.image_gen import plan_image_target
        with patch.object(ig.settings, "default_gemini_api_key", "plat"):
            self.assertEqual(plan_image_target("openai", "k", False, None, None, "a cat"),
                             ("gemini", "gemini-3.1-flash-lite-image"))
            self.assertEqual(plan_image_target("openai", "k", False, None, None, "a poster in 4k"),
                             ("gemini", "gemini-3.1-flash-image"))

    def _run_adapter(self, params, model_id=None, fail_first_with=None):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (4, 4)).save(buf, "PNG")
        part = MagicMock(); part.inline_data.data = buf.getvalue(); part.inline_data.mime_type = "image/png"
        resp = MagicMock(); resp.candidates = [MagicMock()]; resp.candidates[0].content.parts = [part]
        calls = []

        def gen(model, contents, config):
            calls.append((model, config.image_config.image_size if config.image_config else None,
                          config.image_config.aspect_ratio if config.image_config else None))
            if fail_first_with and len(calls) == 1:
                raise Exception(fail_first_with)
            return resp
        client = MagicMock(); client.models.generate_content.side_effect = gen
        with patch("google.genai.Client", return_value=client):
            out = ig._call_gemini_with_fallback("k", "a cat", params, None, None, model_id=model_id)
        return out, calls

    def test_adapter_sends_1k_to_lite(self):
        (_, _, model), calls = self._run_adapter(ImageParams(resolution="1K", aspect_ratio="16:9"))
        self.assertEqual((model, calls), ("gemini-3.1-flash-lite-image", [("gemini-3.1-flash-lite-image", "1K", "16:9")]))

    def test_adapter_routes_2k_to_the_hires_model(self):
        (_, _, model), calls = self._run_adapter(ImageParams(resolution="2K"))
        self.assertEqual(calls, [("gemini-3.1-flash-image", "2K", "1:1")])

    def test_explicit_lite_with_2k_is_clamped_to_1k_not_a_400(self):
        _, calls = self._run_adapter(ImageParams(resolution="2K"), model_id="gemini-3.1-flash-lite-image")
        self.assertEqual(calls, [("gemini-3.1-flash-lite-image", "1K", "1:1")])

    def test_a_size_rejection_retries_with_the_aspect_ratio_kept(self):
        _, calls = self._run_adapter(
            ImageParams(resolution="1K", aspect_ratio="9:16"),
            fail_first_with="400 INVALID_ARGUMENT. Image size 1K is not supported for this model",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ("gemini-3.1-flash-lite-image", None, "9:16"))   # same model, no size, aspect kept

    def test_hires_fallbacks_never_downgrade_the_size(self):
        called = []
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (4, 4)).save(buf, "PNG")
        part = MagicMock(); part.inline_data.data = buf.getvalue(); part.inline_data.mime_type = "image/png"
        resp = MagicMock(); resp.candidates = [MagicMock()]; resp.candidates[0].content.parts = [part]

        def gen(model, contents, config):
            called.append(model)
            if len(called) == 1:
                raise Exception("404 model not found")
            return resp
        client = MagicMock(); client.models.generate_content.side_effect = gen
        with patch("google.genai.Client", return_value=client):
            ig._call_gemini_with_fallback("k", "a cat", ImageParams(resolution="4K"), None, None)
        self.assertEqual(called, ["gemini-3.1-flash-image", "gemini-3.1-flash-image-preview"])   # no lite / 2.5 in between


class TestGeminiFallbackOnUnavailability(unittest.TestCase):
    """Default lite; when it is not available the next model (2.5) answers."""

    def _run(self, first_error):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (4, 4)).save(buf, "PNG")
        part = MagicMock(); part.inline_data.data = buf.getvalue(); part.inline_data.mime_type = "image/png"
        resp = MagicMock(); resp.candidates = [MagicMock()]; resp.candidates[0].content.parts = [part]
        called = []

        def gen(model, contents, config):
            called.append(model)
            if len(called) == 1:
                raise Exception(first_error)
            return resp
        client = MagicMock(); client.models.generate_content.side_effect = gen
        with patch("google.genai.Client", return_value=client):
            try:
                _, _, used = ig._call_gemini_with_fallback("k", "a cat", ImageParams(resolution="1K"), None, None)
            except Exception as exc:
                return called, exc
        return called, used

    def test_model_gone_falls_back_to_2_5(self):
        called, used = self._run("404 NOT_FOUND. models/gemini-3.1-flash-lite-image is not found")
        self.assertEqual((called, used), (["gemini-3.1-flash-lite-image", "gemini-2.5-flash-image"], "gemini-2.5-flash-image"))

    def test_real_google_404_text_goes_straight_to_the_fallback(self):
        # Google's 404 says "not supported for generateContent"; that must not trigger a same-model size retry
        called, used = self._run(
            "404 NOT_FOUND. models/gemini-3.1-flash-lite-image is not found for API version v1beta, "
            "or is not supported for generateContent."
        )
        self.assertEqual((called, used), (["gemini-3.1-flash-lite-image", "gemini-2.5-flash-image"], "gemini-2.5-flash-image"))

    def test_model_deprecated_falls_back(self):
        called, used = self._run("The model gemini-3.1-flash-lite-image has been deprecated")
        self.assertEqual(used, "gemini-2.5-flash-image")

    def test_overloaded_503_falls_back(self):
        called, used = self._run("503 UNAVAILABLE. The model is overloaded. Please try again later.")
        self.assertEqual(used, "gemini-2.5-flash-image")

    def test_quota_429_falls_back(self):
        called, used = self._run("429 RESOURCE_EXHAUSTED. Quota exceeded for this model")
        self.assertEqual(used, "gemini-2.5-flash-image")

    def test_safety_block_does_not_fall_back(self):
        called, exc = self._run("Gemini image blocked by safety filters: block_reason=SAFETY")
        self.assertEqual(called, ["gemini-3.1-flash-lite-image"])
        self.assertIsInstance(exc, Exception)

    def test_bad_request_does_not_fall_back(self):
        called, exc = self._run("400 INVALID_ARGUMENT. Prompt must not be empty")
        self.assertEqual(called, ["gemini-3.1-flash-lite-image"])

    def test_other_providers_keep_the_narrow_rule(self):
        self.assertFalse(ig._is_model_not_found_error(Exception("503 UNAVAILABLE overloaded")))
        self.assertTrue(ig._is_model_unavailable_error(Exception("503 UNAVAILABLE overloaded")))


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
