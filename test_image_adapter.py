"""
Unit tests for Unified Image Adapter (Phase 1).
Tests mocked SDK payloads for OpenAI, Gemini, Grok, parameter overriding,
fallback model retry, and magic-byte format sniffing & conversion.
"""

import io
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.image_gen import (
    ImageParams,
    generate_image_unified,
    _call_gemini_with_fallback,
    _call_openai_with_fallback,
    _call_grok_with_fallback,
)
from src.s3_service import (
    sniff_image_bytes,
    convert_image_format,
    get_image_dimensions,
    upload_image_from_bytes_detailed,
)


class TestImageAdapterAndS3(unittest.TestCase):

    def _create_test_image_bytes(self, fmt: str = "PNG") -> bytes:
        img = Image.new("RGBA", (100, 100), (255, 100, 50, 255))
        buf = io.BytesIO()
        if fmt.upper() in ("JPG", "JPEG"):
            img.convert("RGB").save(buf, format="JPEG")
        else:
            img.save(buf, format=fmt.upper())
        return buf.getvalue()

    # -------------------------------------------------------------
    # 1. Parameter Extraction & Precedence
    # -------------------------------------------------------------
    def test_image_params_defaults(self):
        p = ImageParams.from_prompt_and_overrides("A cat sitting on a chair")
        self.assertEqual(p.aspect_ratio, "1:1")
        self.assertEqual(p.quality, "medium")
        self.assertEqual(p.resolution, "1K")
        self.assertEqual(p.format, "jpeg")
        self.assertFalse(p.transparent_bg)

    def test_image_params_format_variants(self):
        # Default is jpeg
        p_def = ImageParams.from_prompt_and_overrides("A sunset over mountains")
        self.assertEqual(p_def.format, "jpeg")

        # Explicit png in prompt
        p_png = ImageParams.from_prompt_and_overrides("A modern company logo in PNG format")
        self.assertEqual(p_png.format, "png")

        # Transparency forces png
        p_trans = ImageParams.from_prompt_and_overrides("An app icon with transparent background")
        self.assertEqual(p_trans.format, "png")
        self.assertTrue(p_trans.transparent_bg)

        # WebP in prompt
        p_webp = ImageParams.from_prompt_and_overrides("A landscape photo in webp format")
        self.assertEqual(p_webp.format, "webp")

    def test_image_params_prompt_regex_overrides(self):
        p = ImageParams.from_prompt_and_overrides(
            "Wide 16:9 cinematic landscape of mountains, ultra-detailed 4k quality, transparent background"
        )
        self.assertEqual(p.aspect_ratio, "16:9")
        self.assertEqual(p.quality, "high")
        self.assertEqual(p.resolution, "4K")
        self.assertEqual(p.format, "png")
        self.assertTrue(p.transparent_bg)

    def test_image_params_explicit_overrides_precedence(self):
        # Prompt says 16:9 and 4k, but explicit dict forces 9:16 portrait and 1K webp
        explicit = {
            "aspect_ratio": "9:16",
            "resolution": "1K",
            "quality": "low",
            "format": "webp",
        }
        p = ImageParams.from_prompt_and_overrides(
            "Wide 16:9 cinematic landscape in 4k quality",
            explicit_params=explicit,
        )
        self.assertEqual(p.aspect_ratio, "9:16")
        self.assertEqual(p.resolution, "1K")
        self.assertEqual(p.quality, "low")
        self.assertEqual(p.format, "webp")

    # -------------------------------------------------------------
    # 2. Magic-byte Sniffing & Pillow Format Conversion
    # -------------------------------------------------------------
    def test_sniff_magic_bytes(self):
        png_bytes = self._create_test_image_bytes("PNG")
        jpg_bytes = self._create_test_image_bytes("JPEG")
        webp_bytes = self._create_test_image_bytes("WEBP")

        ct_png, ext_png = sniff_image_bytes(png_bytes)
        self.assertEqual(ct_png, "image/png")
        self.assertEqual(ext_png, "png")

        ct_jpg, ext_jpg = sniff_image_bytes(jpg_bytes)
        self.assertEqual(ct_jpg, "image/jpeg")
        self.assertEqual(ext_jpg, "jpg")

        ct_webp, ext_webp = sniff_image_bytes(webp_bytes)
        self.assertEqual(ct_webp, "image/webp")
        self.assertEqual(ext_webp, "webp")

    def test_pillow_format_conversion(self):
        png_bytes = self._create_test_image_bytes("PNG")

        # Convert PNG -> WebP
        webp_bytes, ct_webp, ext_webp = convert_image_format(png_bytes, "webp")
        self.assertEqual(ct_webp, "image/webp")
        self.assertEqual(ext_webp, "webp")
        self.assertTrue(webp_bytes.startswith(b"RIFF"))

        # Convert PNG -> JPEG
        jpg_bytes, ct_jpg, ext_jpg = convert_image_format(png_bytes, "jpeg")
        self.assertEqual(ct_jpg, "image/jpeg")
        self.assertEqual(ext_jpg, "jpg")
        self.assertTrue(jpg_bytes.startswith(b"\xff\xd8\xff"))

    @patch("src.s3_service._get_s3_client")
    @patch("src.s3_service._check_storage_quota")
    @patch("src.s3_service._track_file")
    def test_s3_upload_sniffed_extensions(self, mock_track, mock_quota, mock_s3):
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        # Upload PNG
        png_res = upload_image_from_bytes_detailed(self._create_test_image_bytes("PNG"), user_id="u123")
        self.assertEqual(png_res["format"], "png")
        self.assertEqual(png_res["content_type"], "image/png")
        self.assertTrue(mock_client.put_object.call_args[1]["Key"].endswith(".png"))

        # Upload JPEG
        jpg_res = upload_image_from_bytes_detailed(self._create_test_image_bytes("JPEG"), user_id="u123")
        self.assertEqual(jpg_res["format"], "jpg")
        self.assertEqual(jpg_res["content_type"], "image/jpeg")
        self.assertTrue(mock_client.put_object.call_args[1]["Key"].endswith(".jpg"))

        # Upload WebP
        webp_res = upload_image_from_bytes_detailed(self._create_test_image_bytes("WEBP"), user_id="u123")
        self.assertEqual(webp_res["format"], "webp")
        self.assertEqual(webp_res["content_type"], "image/webp")
        self.assertTrue(mock_client.put_object.call_args[1]["Key"].endswith(".webp"))

    # -------------------------------------------------------------
    # 3. Mocked SDK Payload Assertions
    # -------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_openai_payload_mapping(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        mock_data.url = None
        mock_resp = MagicMock()
        mock_resp.data = [mock_data]
        mock_client.images.generate.return_value = mock_resp

        # 1. Default params
        params_default = ImageParams()
        _call_openai_with_fallback("fake-key", "A cute kitten", params_default)
        call_kwargs = mock_client.images.generate.call_args[1]
        self.assertEqual(call_kwargs["size"], "1024x1024")
        self.assertEqual(call_kwargs["quality"], "medium")
        self.assertEqual(call_kwargs["output_format"], "jpeg")

        # 2. Overridden params (16:9 wide high-quality)
        params_wide = ImageParams(aspect_ratio="16:9", quality="high", resolution="2K")
        _call_openai_with_fallback("fake-key", "A wide sunset", params_wide)
        call_kwargs_wide = mock_client.images.generate.call_args[1]
        self.assertEqual(call_kwargs_wide["size"], "1536x1024")
        self.assertEqual(call_kwargs_wide["quality"], "high")

    @patch("openai.OpenAI")
    def test_grok_payload_mapping(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.url = "https://mock.x.ai/image.jpg"
        mock_resp = MagicMock()
        mock_resp.data = [mock_data]
        mock_client.images.generate.return_value = mock_resp

        with patch("requests.get") as mock_get:
            mock_http_resp = MagicMock()
            mock_http_resp.content = self._create_test_image_bytes("JPEG")
            mock_http_resp.headers = {"Content-Type": "image/jpeg"}
            mock_get.return_value = mock_http_resp

            params = ImageParams()
            raw, mime, model = _call_grok_with_fallback("fake-key", "A futuristic rover", params)
            self.assertEqual(mock_openai_cls.call_args[1]["base_url"], "https://api.x.ai/v1")
            self.assertEqual(mock_client.images.generate.call_args[1]["prompt"], "A futuristic rover")
            self.assertEqual(model, "grok-imagine-image-2.0")

    @patch("openai.OpenAI")
    def test_model_not_found_fallback_retry(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock first call fails with model not found, second succeeds
        mock_data = MagicMock()
        mock_data.url = "https://mock.x.ai/image.jpg"
        mock_resp = MagicMock()
        mock_resp.data = [mock_data]

        mock_client.images.generate.side_effect = [
            Exception("Error code 400: The model 'grok-imagine-image-2.0' does not exist"),
            mock_resp,
        ]

        with patch("requests.get") as mock_get:
            mock_http_resp = MagicMock()
            mock_http_resp.content = self._create_test_image_bytes("JPEG")
            mock_http_resp.headers = {"Content-Type": "image/jpeg"}
            mock_get.return_value = mock_http_resp

            params = ImageParams()
            raw, mime, model = _call_grok_with_fallback("fake-key", "Fallback test", params)
            # Model should have retried with fallback grok-imagine-image
            self.assertEqual(model, "grok-imagine-image")
            self.assertEqual(mock_client.images.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
