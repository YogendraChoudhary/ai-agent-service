"""
Regression tests for the image adapter fixes: params, OpenAI payloads, Grok JSON edits,
provider resolution, and upload metadata.
"""

import base64
import io
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

import src.image_gen as ig
from src.image_gen import (
    ImageParams,
    sanitize_image_params,
    resolve_image_provider,
    generate_image_unified,
    _call_openai_with_fallback,
    _call_grok_with_fallback,
    _call_gemini_with_fallback,
    _is_model_not_found_error,
)
from src.s3_service import get_image_dimensions, upload_image_from_bytes_detailed


def img_bytes(fmt="PNG", size=(32, 32)):
    buf = io.BytesIO()
    im = Image.new("RGBA" if fmt.upper() == "PNG" else "RGB", size, (200, 80, 40))
    im.save(buf, format=fmt.upper())
    return buf.getvalue()


PNG = img_bytes("PNG")
JPEG = img_bytes("JPEG")
WEBP = img_bytes("WEBP")
PNG_B64 = base64.b64encode(PNG).decode()


class TestParamRules(unittest.TestCase):

    def test_style_words_do_not_set_orientation_or_quality(self):
        for prompt in (
            "a children's story illustration of a fox",
            "wide-angle shot of a mountain, cinematic and photorealistic",
            "a detailed masterpiece of a castle",
            "make a quick sketch of a cat",
        ):
            p = ImageParams.from_prompt_and_overrides(prompt)
            self.assertEqual((p.aspect_ratio, p.quality), ("1:1", "medium"), prompt)

    def test_explicit_requests_are_honoured(self):
        cases = {
            "a landscape image of mountains": ("aspect_ratio", "16:9"),
            "draw it in portrait": ("aspect_ratio", "9:16"),
            "an instagram story background": ("aspect_ratio", "9:16"),
            "a widescreen wallpaper": ("aspect_ratio", "16:9"),
            "high quality render of a car": ("quality", "high"),
            "a draft of the layout": ("quality", "low"),
            "a poster in 4k": ("resolution", "4K"),
        }
        for prompt, (field, want) in cases.items():
            self.assertEqual(getattr(ImageParams.from_prompt_and_overrides(prompt), field), want, prompt)

    def test_transparent_background_forces_png_even_over_explicit_format(self):
        p = ImageParams.from_prompt_and_overrides("a logo", {"format": "webp", "transparent_bg": True})
        self.assertEqual((p.format, p.transparent_bg), ("png", True))

    def test_sanitize_accepts_valid_and_rejects_bad_input(self):
        self.assertEqual(
            sanitize_image_params({"aspect_ratio": "16:9", "quality": "HIGH", "format": "jpg", "resolution": "2k"}),
            {"aspect_ratio": "16:9", "quality": "high", "format": "jpeg", "resolution": "2K"},
        )
        self.assertEqual(sanitize_image_params(None), {})
        for bad in ({"aspect_ratio": "7:3"}, {"quality": "ultra"}, {"format": "gif"}, {"resolution": "8K"},
                    {"unknown": 1}, "nope"):
            with self.assertRaises(ValueError, msg=str(bad)):
                sanitize_image_params(bad)


class TestOpenAIPayload(unittest.TestCase):

    def _client(self):
        client = MagicMock()
        item = MagicMock(b64_json=PNG_B64, url=None)
        client.images.generate.return_value = MagicMock(data=[item])
        client.images.edit.return_value = MagicMock(data=[item])
        return client

    def test_quality_levels_map_one_to_one(self):
        client = self._client()
        with patch("openai.OpenAI", return_value=client):
            for q in ("low", "medium", "high"):
                _call_openai_with_fallback("k", "a cat", ImageParams(quality=q))
                self.assertEqual(client.images.generate.call_args[1]["quality"], q)

    def test_format_and_transparency_are_requested_natively(self):
        client = self._client()
        with patch("openai.OpenAI", return_value=client):
            _call_openai_with_fallback("k", "a cat", ImageParams(format="webp"))
            kwargs = client.images.generate.call_args[1]
            self.assertEqual(kwargs["output_format"], "webp")
            self.assertNotIn("background", kwargs)

            _call_openai_with_fallback("k", "a logo", ImageParams(format="png", transparent_bg=True))
            kwargs = client.images.generate.call_args[1]
            self.assertEqual((kwargs["output_format"], kwargs["background"]), ("png", "transparent"))

    def test_edit_path_sends_image_quality_format_and_fidelity(self):
        client = self._client()
        with patch("openai.OpenAI", return_value=client):
            _call_openai_with_fallback("k", "add vibe", ImageParams(quality="high"), reference_image_bytes=JPEG)
        client.images.generate.assert_not_called()
        kwargs = client.images.edit.call_args[1]
        self.assertEqual(kwargs["quality"], "high")
        self.assertEqual(kwargs["output_format"], "jpeg")
        self.assertEqual(kwargs["input_fidelity"], "high")
        self.assertTrue(kwargs["image"].name.endswith(".jpg"))

    def test_rejected_tuning_param_retries_once_with_bare_request(self):
        client = self._client()
        good = client.images.generate.return_value
        client.images.generate.side_effect = [Exception("Unknown parameter: 'output_format'."), good]
        with patch("openai.OpenAI", return_value=client):
            _, mime, _ = _call_openai_with_fallback("k", "a cat", ImageParams(format="webp"))
        self.assertEqual(client.images.generate.call_count, 2)
        self.assertNotIn("output_format", client.images.generate.call_args[1])
        self.assertEqual(mime, "image/png")  # bare request returns the API default

    def test_unrelated_errors_are_not_swallowed(self):
        client = self._client()
        client.images.generate.side_effect = Exception("Rate limit exceeded")
        with patch("openai.OpenAI", return_value=client):
            with self.assertRaises(Exception):
                _call_openai_with_fallback("k", "a cat", ImageParams())
        self.assertEqual(client.images.generate.call_count, 1)

    def test_model_not_found_detection_is_narrow(self):
        self.assertTrue(_is_model_not_found_error(Exception("The model 'x' does not exist")))
        self.assertTrue(_is_model_not_found_error(Exception("model_not_found")))
        self.assertFalse(_is_model_not_found_error(Exception("invalid_value for parameter size")))
        self.assertFalse(_is_model_not_found_error(Exception("HTTP 404 page not found")))


class TestGrokEdit(unittest.TestCase):

    def _response(self, status=200, body=None):
        r = MagicMock(status_code=status, text="boom")
        r.json.return_value = body if body is not None else {"data": [{"url": "https://x.ai/out.jpg"}]}
        return r

    def test_edit_uses_json_endpoint_with_data_uri(self):
        dl = MagicMock(content=JPEG, headers={"Content-Type": "image/jpeg"})
        dl.raise_for_status = MagicMock()
        with patch("requests.post", return_value=self._response()) as post, patch("requests.get", return_value=dl):
            raw, mime, model = _call_grok_with_fallback("key", "pencil sketch", ImageParams(), PNG)

        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.x.ai/v1/images/edits")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        body = kwargs["json"]
        self.assertEqual(body["model"], "grok-imagine-image-2.0")
        self.assertEqual(body["prompt"], "pencil sketch")
        self.assertEqual(body["image"]["type"], "image_url")
        self.assertTrue(body["image"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("files", kwargs)  # JSON body, not multipart
        self.assertEqual(raw, JPEG)

    def test_webp_reference_is_transcoded_to_png(self):
        dl = MagicMock(content=JPEG, headers={"Content-Type": "image/jpeg"})
        dl.raise_for_status = MagicMock()
        with patch("requests.post", return_value=self._response()) as post, patch("requests.get", return_value=dl):
            _call_grok_with_fallback("key", "p", ImageParams(), WEBP)
        uri = post.call_args[1]["json"]["image"]["url"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertTrue(base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG"))

    def test_http_error_surfaces_status(self):
        with patch("requests.post", return_value=self._response(status=415)):
            with self.assertRaises(Exception) as ctx:
                _call_grok_with_fallback("key", "p", ImageParams(), PNG)
        self.assertIn("415", str(ctx.exception))

    def test_failed_grok_edit_never_silently_generates_from_text(self):
        with patch.object(ig.settings, "default_gemini_api_key", None), \
             patch("requests.post", return_value=self._response(status=500)), \
             patch("src.image_gen.upload_image_from_bytes_detailed") as upload:
            with self.assertRaises(ValueError) as ctx:
                generate_image_unified("sketch it", "grok", "key", reference_image_bytes=PNG, user_id="u")
        self.assertIn("no fallback", str(ctx.exception))
        upload.assert_not_called()

    def test_failed_grok_edit_falls_back_to_gemini_when_platform_key_exists(self):
        up = {"cdn_url": "https://cdn/x.jpg", "width": 1, "height": 1, "format": "jpg", "content_type": "image/jpeg"}
        with patch.object(ig.settings, "default_gemini_api_key", "gem-key"), \
             patch("requests.post", return_value=self._response(status=500)), \
             patch("src.image_gen._call_gemini_with_fallback", return_value=(PNG, "image/png", "gem")) as gem, \
             patch("src.image_gen.upload_image_from_bytes_detailed", return_value=up), \
             patch("src.image_gen.save_image_context"):
            res = generate_image_unified("sketch it", "grok", "key", reference_image_bytes=PNG, user_id="u")
        self.assertEqual(res["provider"], "gemini")
        self.assertEqual(gem.call_args[0][0], "gem-key")
        self.assertEqual(gem.call_args[0][3], PNG)  # reference image still passed along

    def test_byok_never_falls_back_to_platform_key(self):
        with patch.object(ig.settings, "default_gemini_api_key", "gem-key"), \
             patch("requests.post", return_value=self._response(status=500)):
            with self.assertRaises(ValueError):
                generate_image_unified("p", "grok", "key", reference_image_bytes=PNG, user_id="u", is_byok=True)


class TestGeminiPayload(unittest.TestCase):

    def test_reference_mime_comes_from_bytes_and_config_has_ratio_and_size(self):
        part = MagicMock()
        part.inline_data.data = PNG
        part.inline_data.mime_type = "image/png"
        response = MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])
        client = MagicMock()
        client.models.generate_content.return_value = response

        with patch("google.genai.Client", return_value=client):
            _call_gemini_with_fallback(
                "k", "refine", ImageParams(aspect_ratio="16:9", resolution="2K"), reference_image_bytes=JPEG
            )
        kwargs = client.models.generate_content.call_args[1]
        self.assertEqual(kwargs["contents"][0].inline_data.mime_type, "image/jpeg")
        self.assertEqual(kwargs["contents"][1], "refine")
        self.assertEqual(kwargs["config"].image_config.aspect_ratio, "16:9")
        self.assertEqual(kwargs["config"].image_config.image_size, "2K")


class TestProviderResolution(unittest.TestCase):

    def test_native_providers_need_a_key(self):
        # With no platform key the chat provider's own key is used (BYOK / no platform key)
        with patch.object(ig.settings, "default_gemini_api_key", None):
            self.assertEqual(resolve_image_provider("openai", "k"), ("openai", "k"))
            self.assertIsNone(resolve_image_provider("gemini", None))

    def test_default_image_provider_is_gemini_whatever_the_chat_model(self):
        with patch.object(ig.settings, "default_gemini_api_key", "plat"):
            self.assertEqual(resolve_image_provider("openai", "user-key"), ("gemini", "plat"))
            self.assertEqual(resolve_image_provider("grok", "user-key"), ("gemini", "plat"))
            self.assertEqual(resolve_image_provider("gemini", "user-key"), ("gemini", "user-key"))

    def test_byok_never_uses_platform_key_for_the_default(self):
        with patch.object(ig.settings, "default_gemini_api_key", "plat"):
            self.assertEqual(resolve_image_provider("openai", "user-key", is_byok=True), ("openai", "user-key"))

    def test_claude_and_deepseek_use_platform_fallback_only_without_byok(self):
        with patch.object(ig.settings, "default_gemini_api_key", "plat"):
            self.assertEqual(resolve_image_provider("claude", "user-key"), ("gemini", "plat"))
            self.assertEqual(resolve_image_provider("deepseek", None), ("gemini", "plat"))
            self.assertIsNone(resolve_image_provider("claude", "user-key", is_byok=True))
        with patch.object(ig.settings, "default_gemini_api_key", None):
            self.assertIsNone(resolve_image_provider("claude", "user-key"))


class TestUploadMetadata(unittest.TestCase):

    def test_unreadable_image_reports_unknown_dimensions(self):
        self.assertEqual(get_image_dimensions(b"not an image"), (None, None))

    @patch("src.s3_service._get_s3_client")
    @patch("src.s3_service._check_storage_quota")
    @patch("src.s3_service._track_file")
    def test_type_and_extension_follow_the_bytes_not_the_caller(self, _t, _q, s3):
        client = MagicMock()
        s3.return_value = client
        res = upload_image_from_bytes_detailed(PNG, content_type="image/webp", user_id="u")
        self.assertEqual((res["content_type"], res["format"]), ("image/png", "png"))
        put = client.put_object.call_args[1]
        self.assertEqual(put["ContentType"], "image/png")
        self.assertTrue(put["Key"].endswith(".png"))


class TestUnifiedFlow(unittest.TestCase):

    def test_target_format_follows_params_and_context_is_saved_per_document(self):
        up = {"cdn_url": "https://cdn/x.webp", "width": 10, "height": 20, "format": "webp",
              "content_type": "image/webp"}
        with patch("src.image_gen._call_openai_with_fallback", return_value=(PNG, "image/png", "gpt-image-1")), \
             patch("src.image_gen.upload_image_from_bytes_detailed", return_value=up) as upload, \
             patch("src.image_gen.save_image_context") as save:
            res = generate_image_unified(
                "make it night", "openai", "k", explicit_params={"format": "webp"}, user_id="u1",
                document_id="docA", inherited_params={"aspect_ratio": "16:9"},
                context_extra={"base_prompt": "a mosque", "modifications": ["make it night"]},
            )
        self.assertEqual(upload.call_args[1]["target_format"], "webp")
        self.assertEqual(res["params"]["aspect_ratio"], "16:9")
        kw = save.call_args[1]
        self.assertEqual((kw["document_id"], kw["base_prompt"], kw["modifications"]),
                         ("docA", "a mosque", ["make it night"]))
        self.assertEqual((res["width"], res["height"]), (10, 20))


if __name__ == "__main__":
    unittest.main()
