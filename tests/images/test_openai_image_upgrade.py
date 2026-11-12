"""
OpenAI image models after gpt-image-1 was withdrawn (2026-11-11: "The model 'gpt-image-1' does not exist"). The default is now
gpt-image-2.5-flare; params are sent only where the model takes them (probed live), and a rejected param is dropped alone.
"""
import base64
import unittest
from unittest.mock import MagicMock, patch

import src.models as M
from src.image_gen import ImageParams, _call_openai_with_fallback, _is_optional_param_error
from src.models import openai_image_supports_input_fidelity

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 32


def client_with(*results):
    client = MagicMock()
    item = MagicMock(b64_json=PNG, url=None)
    ok = MagicMock(data=[item])
    sequence = [ok if r is None else r for r in results]
    client.images.generate.side_effect = sequence
    client.images.edit.side_effect = sequence
    return client


class TestDefaultsAndChain(unittest.TestCase):

    def test_the_default_and_fallbacks_are_models_that_exist(self):
        self.assertEqual(M.IMAGE_MODEL_IDS["openai"], "gpt-image-2.5-flare")
        self.assertEqual(M.IMAGE_MODEL_FALLBACKS["openai"], ["gpt-image-2.6-flare", "gpt-image-1.5"])
        self.assertNotIn("gpt-image-1", [M.IMAGE_MODEL_IDS["openai"], *M.IMAGE_MODEL_FALLBACKS["openai"]])

    def test_a_withdrawn_model_moves_on_to_the_default_then_the_fallbacks(self):
        gone = Exception("The model 'gpt-image-1' does not exist.")
        client = client_with(gone, None)
        with patch("openai.OpenAI", return_value=client):
            _, _, used = _call_openai_with_fallback("k", "a cat", ImageParams(), model_id="gpt-image-1")
        models = [c[1]["model"] for c in client.images.generate.call_args_list]
        self.assertEqual(models, ["gpt-image-1", "gpt-image-2.5-flare"])       # the default next, not the 3.3x dearer 2.6-flare
        self.assertEqual(used, "gpt-image-2.5-flare")

    def test_the_whole_chain_is_tried_once_each_then_the_last_error_is_raised(self):
        gone = Exception("The model 'x' does not exist.")
        client = client_with(gone, gone, gone, gone)
        with patch("openai.OpenAI", return_value=client), self.assertRaises(Exception):
            _call_openai_with_fallback("k", "a cat", ImageParams(), model_id="gpt-image-1")
        models = [c[1]["model"] for c in client.images.generate.call_args_list]
        self.assertEqual(models, ["gpt-image-1", "gpt-image-2.5-flare", "gpt-image-2.6-flare", "gpt-image-1.5"])


class TestInputFidelity(unittest.TestCase):

    def test_only_gpt_image_1_5_takes_it(self):
        self.assertTrue(openai_image_supports_input_fidelity("gpt-image-1.5"))
        for model_id in ("gpt-image-1-mini", "gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2.6-flare", "", None):
            with self.subTest(model=model_id):
                self.assertFalse(openai_image_supports_input_fidelity(model_id))

    def test_edit_requests_send_it_only_where_it_is_accepted(self):
        for model_id, expected in (("gpt-image-1.5", True), ("gpt-image-2.5-flare", False), ("gpt-image-2.6-sunburst", False)):
            with self.subTest(model=model_id):
                client = client_with(None)
                with patch("openai.OpenAI", return_value=client):
                    _call_openai_with_fallback("k", "make it blue", ImageParams(), reference_image_bytes=JPEG, model_id=model_id)
                self.assertEqual("input_fidelity" in client.images.edit.call_args[1], expected)


class TestRejectedParams(unittest.TestCase):

    def test_only_the_named_param_is_dropped_the_rest_keep_working(self):
        # gpt-image-2 answers this to background=transparent
        client = client_with(Exception("Transparent background is not supported for this model."), None)
        with patch("openai.OpenAI", return_value=client):
            _, mime, _ = _call_openai_with_fallback("k", "a logo", ImageParams(format="png", quality="high", transparent_bg=True), model_id="gpt-image-2")
        retry = client.images.generate.call_args_list[1][1]
        self.assertNotIn("background", retry)
        self.assertEqual((retry["quality"], retry["output_format"]), ("high", "png"))     # the old code dropped these too
        self.assertEqual(mime, "image/png")

    def test_a_dropped_output_format_means_the_api_default_png(self):
        client = client_with(Exception("Unknown parameter: 'output_format'."), None)
        with patch("openai.OpenAI", return_value=client):
            _, mime, _ = _call_openai_with_fallback("k", "a cat", ImageParams(format="webp", quality="low"))
        self.assertEqual(client.images.generate.call_args_list[1][1]["quality"], "low")
        self.assertEqual(mime, "image/png")

    def test_two_rejected_params_are_dropped_one_after_the_other(self):
        client = client_with(Exception("Unknown parameter: 'output_format'."), Exception("Unsupported value for background"), None)
        with patch("openai.OpenAI", return_value=client):
            _call_openai_with_fallback("k", "a logo", ImageParams(format="png", transparent_bg=True, quality="low"))
        last = client.images.generate.call_args_list[2][1]
        self.assertEqual({k for k in last if k in ("quality", "output_format", "background")}, {"quality"})

    def test_an_error_that_blames_something_we_did_not_send_is_raised_not_retried(self):
        client = client_with(Exception("Unknown parameter: 'input_fidelity'."))
        with patch("openai.OpenAI", return_value=client), self.assertRaises(Exception):
            _call_openai_with_fallback("k", "a cat", ImageParams(), model_id="gpt-image-2.5-flare")
        self.assertEqual(client.images.generate.call_count, 1)

    def test_the_real_api_wording_for_input_fidelity_is_recognised(self):
        self.assertTrue(_is_optional_param_error(Exception("The model 'gpt-image-2' does not support the 'input_fidelity' parameter.")))


if __name__ == "__main__":
    unittest.main()
