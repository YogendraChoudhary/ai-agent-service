"""
Routing tests for process_user_query_stream: which requests may reach the image adapter,
what they send it, and what the editor receives.

The adapter, reference download and agent factory are stubbed; no network is used.
"""

import re
import time
import unittest
from unittest.mock import patch

import src.image_gen as ig
import src.text_processing_agent as tpa
from src.text_processing_agent import process_user_query_stream

PLACEHOLDER = re.compile(r"^!\[[^\]]*\]\(placeholder::[0-9a-f-]{36}\)$")


class AgentPathReached(Exception):
    """Raised by the stubbed agent factory so tests can tell the agent path was chosen."""

    def __init__(self, kwargs):
        super().__init__("agent path")
        self.kwargs = kwargs


def _fake_create_agent(*args, **kwargs):
    raise AgentPathReached(kwargs)


class RouteTestCase(unittest.TestCase):

    def setUp(self):
        self.calls = []
        self.prev_ctx = None

        def fake_adapter(**kw):
            self.calls.append(kw)
            return {"cdn_url": "https://cdn.test/u/new.jpg", "width": 8, "height": 6, "format": "jpg"}

        patches = [
            patch.object(tpa, "create_agent", _fake_create_agent),
            patch("src.image_gen.generate_image_unified", fake_adapter),
            patch("src.image_gen.get_previous_image_context", lambda *a, **k: self.prev_ctx),
            patch.object(ig.settings, "default_gemini_api_key", "platform-gem"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_stream(self, query, **kwargs):
        kwargs.setdefault("model_provider", "openai")
        kwargs.setdefault("api_key", "user-key")
        kwargs.setdefault("user_id", "u1")
        kwargs.setdefault("image_routing", True)
        return list(process_user_query_stream(query, **kwargs))

    def wait_for_adapter(self, pending, timeout=2.0):
        deadline = time.time() + timeout
        entry = next(v for k, v in pending.items() if not k.startswith("__"))
        while entry["status"] == "generating" and time.time() < deadline:
            time.sleep(0.01)
        return entry

    def assertAgentPath(self, query, **kwargs):
        with self.assertRaises(AgentPathReached) as ctx:
            self.run_stream(query, **kwargs)
        self.assertEqual(self.calls, [], f"adapter must not run for {query!r}")
        return ctx.exception.kwargs


class TestGating(RouteTestCase):

    def test_text_operations_never_reach_the_image_adapter(self):
        # api.py sets image_routing only for type == "general"
        for q in ("Fix grammar: Please create a logo for our new brand",
                  "create a logo for our new brand",
                  "draw a mosque at dusk"):
            self.assertAgentPath(q, image_routing=False)

    def test_general_chat_text_requests_stay_on_the_agent(self):
        for q in ("write an article about AI with 2 images", "make it more formal",
                  "change the tone to friendly", "Expand on the graphics discussion in my essay"):
            self.assertAgentPath(q)

    def test_agent_path_receives_document_and_params(self):
        kw = self.assertAgentPath("write about cats", document_id="docA", image_params={"quality": "high"})
        self.assertEqual(kw["image_ctx"], {"document_id": "docA", "explicit_params": {"quality": "high"}})

    def test_force_agent_type_bypasses_image_routing(self):
        with self.assertRaises(Exception):
            self.run_stream("draw a mosque at dusk", force_agent_type="text_extraction")
        self.assertEqual(self.calls, [])

    def test_router_can_be_switched_off(self):
        with patch.object(ig.settings, "image_router_regex_enabled", False):
            self.assertAgentPath("draw a mosque at dusk")


class TestDirectRoute(RouteTestCase):

    def test_generate_streams_markdown_placeholder_then_pending_sentinel(self):
        out = self.run_stream("draw a mosque at dusk", document_id="docA", image_params={"quality": "high"})
        self.assertEqual(len(out), 2)
        self.assertRegex(out[0], PLACEHOLDER)
        pid = out[0].split("placeholder::")[1].rstrip(")")
        entry = self.wait_for_adapter(out[1]["__pending_images__"])
        self.assertEqual(entry["status"], "done")
        self.assertIn(pid, out[1]["__pending_images__"])

        call = self.calls[0]
        self.assertEqual(call["prompt"], "draw a mosque at dusk")
        self.assertEqual(call["document_id"], "docA")
        self.assertEqual(call["explicit_params"], {"quality": "high"})
        self.assertIsNone(call["reference_image_bytes"])
        self.assertEqual((entry["width"], entry["height"], entry["format"]), (8, 6, "jpg"))

    def test_placeholder_alt_text_is_markdown_safe(self):
        out = self.run_stream("draw a [weird] (dragon)\nwith wings")
        self.assertRegex(out[0], PLACEHOLDER)
        self.assertNotIn("\n", out[0])

    def test_attached_reference_edit_downloads_it_in_the_background(self):
        with patch.object(tpa, "_fetch_reference_bytes", return_value=b"REF") as fetch:
            out = self.run_stream("refine the image with more indepth and taj-mahal vibe",
                                  image_urls=["https://cdn.test/u/ref.png"])
            self.wait_for_adapter(out[1]["__pending_images__"])
        self.assertEqual(fetch.call_args[0][0], "https://cdn.test/u/ref.png")
        self.assertEqual(self.calls[0]["reference_image_bytes"], b"REF")

    def test_edit_fails_loudly_when_reference_cannot_be_loaded(self):
        with patch.object(tpa, "_fetch_reference_bytes", side_effect=OSError("404")):
            out = self.run_stream("refine this with golden lighting", image_urls=["https://cdn.test/u/ref.png"])
            entry = self.wait_for_adapter(out[1]["__pending_images__"])
        self.assertEqual(entry["status"], "failed")
        self.assertIn("reference image", entry["error"])
        self.assertEqual(self.calls, [])  # never generated from text alone

    def test_followup_without_stored_image_goes_to_agent(self):
        self.prev_ctx = None
        for q in ("make it night", "regenerate", "refine the image with more indepth and taj-mahal vibe"):
            self.assertAgentPath(q)

    def test_delta_followup_reuses_prompt_params_image_and_document(self):
        self.prev_ctx = {
            "final_prompt": "draw a mosque at dusk", "base_prompt": "draw a mosque at dusk",
            "modifications": [], "params": {"aspect_ratio": "16:9", "format": "webp"},
            "cdn_url": "https://cdn.test/u/prev.webp",
        }
        with patch.object(tpa, "_fetch_reference_bytes", return_value=b"PREV") as fetch:
            out = self.run_stream("make it night", document_id="docA")
            self.wait_for_adapter(out[1]["__pending_images__"])
        call = self.calls[0]
        self.assertEqual(call["prompt"], "draw a mosque at dusk. Modifications: make it night")
        self.assertEqual(call["inherited_params"], {"aspect_ratio": "16:9", "format": "webp"})
        self.assertEqual(call["reference_image_bytes"], b"PREV")
        self.assertEqual(call["document_id"], "docA")
        self.assertEqual(call["context_extra"]["modifications"], ["make it night"])
        self.assertEqual(fetch.call_args[0][0], "https://cdn.test/u/prev.webp")

    def test_redo_reuses_prompt_without_a_reference_image(self):
        self.prev_ctx = {"final_prompt": "draw a mosque at dusk", "base_prompt": "draw a mosque at dusk",
                         "modifications": [], "params": {}, "cdn_url": "https://cdn.test/u/prev.webp"}
        with patch.object(tpa, "_fetch_reference_bytes") as fetch:
            out = self.run_stream("regenerate")
            self.wait_for_adapter(out[1]["__pending_images__"])
        fetch.assert_not_called()
        self.assertEqual(self.calls[0]["prompt"], "draw a mosque at dusk")
        self.assertIsNone(self.calls[0]["reference_image_bytes"])


class TestProviderAvailability(RouteTestCase):

    def test_claude_uses_platform_key_but_never_for_byok(self):
        out = self.run_stream("draw a mosque at dusk", model_provider="claude")
        self.wait_for_adapter(out[1]["__pending_images__"])
        self.assertEqual(len(self.calls), 1)

        self.calls.clear()
        self.assertAgentPath("draw a mosque at dusk", model_provider="claude", is_byok=True)

    def test_claude_without_platform_key_keeps_previous_behaviour(self):
        with patch.object(ig.settings, "default_gemini_api_key", None):
            self.assertAgentPath("draw a mosque at dusk", model_provider="claude")


if __name__ == "__main__":
    unittest.main()
