"""
Mode 8 on the real routing path: process_user_query_stream with task routing on and claude, gemini and openai all
keyed. Only the network edges are stubbed (the provider call, the screenshot download, the image adapter and the LLM
code classifier), so these tests fail if the design gate loses to the image router or the code classifier, if a
design request is not routed down the design_canvas fallback tree (claude -> gemini -> openai), if the screenshot
skips the capped download or is sent under the wrong media type, or if /agent/abort does not stop a design.
"""

from __future__ import annotations

import io
import json
import threading
import time

import pytest
from PIL import Image

import src.text_processing_agent as tpa
from config import settings
from src.task_router import KeyResolver

VALID_AST = {
    "block_type": "eddyter_design_canvas",
    "version": "1.0",
    "design_id": "dsn_route",
    "title": "Card",
    "theme_palette": ["#000000", "#FFFFFF"],
    "root": {
        "node_type": "container",
        "id": "root",
        "style": {"background_color": "#000000"},
        "children": [{"node_type": "text", "id": "title", "tag": "h1", "spans": [{"text": "Hi", "color": "#FFFFFF"}]}],
    },
}
KEYS = {"anthropic_api_key": "k-claude", "openai_api_key": "k-openai", "gemini_api_key": "k-gemini"}
REFERENCE_URL = "https://cdn.example.test/ref.png"
REAL_FETCH = tpa._fetch_reference_bytes


def _image_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 200, 140)).save(buf, format=fmt)
    return buf.getvalue()


def _resolver() -> KeyResolver:
    return KeyResolver(None, KEYS, lambda user_keys, model_provider, default_keys:
                       default_keys.get("anthropic_api_key" if model_provider == "claude" else f"{model_provider}_api_key"))


@pytest.fixture
def edges(monkeypatch):
    """Routing switches on; records every provider call, classifier call and image-adapter call."""
    for name, value in {
        "task_routing_enabled": True, "provider_fallback_enabled": True, "design_canvas_enabled": True,
        "image_router_regex_enabled": True, "image_router_shadow_mode": False,
        "code_intent_regex_enabled": True, "code_intent_classifier_enabled": True, "code_intent_shadow_mode": False,
    }.items():
        monkeypatch.setattr(settings, name, value, raising=False)

    calls = {"providers": [], "images": [], "fetched": [], "classifier": [], "image_adapter": [], "failing": set(),
             "screenshot": _image_bytes("PNG")}

    def provider_call(messages, image_bytes, temperature, provider, key, request_id=""):
        calls["providers"].append(provider)
        calls["images"].append(image_bytes)
        if provider in calls["failing"]:
            raise RuntimeError(f"{provider} is down")
        return json.dumps(VALID_AST)

    def classifier(text, *args, **kwargs):
        calls["classifier"].append(text)
        return "code"

    def image_adapter(**kwargs):
        calls["image_adapter"].append(kwargs.get("prompt"))
        return {"cdn_url": "https://cdn.example.test/out.png"}

    def fetch(url, request_id=None):
        calls["fetched"].append(url)
        return calls["screenshot"]

    monkeypatch.setattr(tpa, "_run_design_json_completion", provider_call)
    monkeypatch.setattr("src.intent_classifier.classify_intent", classifier)
    monkeypatch.setattr("src.image_gen.image_route_available", lambda *a, **k: True)
    monkeypatch.setattr("src.image_gen.generate_image_unified", image_adapter)
    monkeypatch.setattr(tpa, "_fetch_reference_bytes", fetch)  # screenshots stay offline (image route too, on a regression)
    return calls


def _stream(query, *, operation_type="general", model="openai", model_source="default",
            image_urls=None, existing_design_ast=None, first_only=False, cancel_event=None):
    gen = tpa.process_user_query_stream(
        query,
        model_provider=model,
        api_key=None,
        request_id="route_test",
        image_urls=image_urls,
        force_agent_type=tpa.agent_for_request_type(operation_type),
        image_routing=(operation_type == "general"),
        model_source=model_source,
        key_resolver=_resolver(),
        report_model=True,
        existing_design_ast=existing_design_ast,
        operation_type=operation_type,
        cancel_event=cancel_event,
    )
    return [next(gen)] if first_only else list(gen)


def _events(chunks):
    return {c["__event__"]["event"]: c["__event__"]["data"] for c in chunks
            if isinstance(c, dict) and isinstance(c.get("__event__"), dict) and "event" in c["__event__"]}


def _model_used(chunks):
    return next(c["__model_used__"] for c in chunks if isinstance(c, dict) and "__model_used__" in c)


# ===========================================================================
# Gap 1: the design_canvas fallback tree is the one used
# ===========================================================================

class TestDesignFallbackTree:
    def test_default_route_starts_on_claude(self, edges):
        chunks = _stream("Make an onboarding card UI")
        assert edges["providers"] == ["claude"]
        assert _events(chunks)["init"]["provider"] == "claude"
        assert _events(chunks)["design_ready"]["design_id"] == "dsn_route"

    def test_falls_back_claude_then_gemini_then_openai(self, edges):
        edges["failing"].update({"claude", "gemini"})
        chunks = _stream("Make an onboarding card UI")
        assert edges["providers"] == ["claude", "gemini", "openai"]
        assert _model_used(chunks) == ("openai", "gpt-5.2")  # the provider that answered, not the first tried

    def test_users_own_model_choice_still_goes_first(self, edges):
        edges["failing"].add("openai")
        _stream("Make an onboarding card UI", model="openai", model_source="user")
        assert edges["providers"] == ["openai", "claude"]


# ===========================================================================
# Gap 2: the design gate runs before the image router and the code classifier
# ===========================================================================

class TestDesignGateRunsFirst:
    def test_banner_image_wording_is_a_design_not_a_raster(self, edges):
        chunks = _stream("Design a hero section with a banner image")
        assert "design_ready" in _events(chunks)
        assert edges["image_adapter"] == []
        assert not any(isinstance(c, str) and "placeholder::" in c for c in chunks)

    def test_same_wording_goes_to_the_image_route_when_design_is_off(self, edges, monkeypatch):
        # Control: proves the stubs above would catch the image router winning.
        monkeypatch.setattr(settings, "design_canvas_enabled", False)
        first = _stream("Design a hero section with a banner image", first_only=True)[0]
        assert isinstance(first, str) and "placeholder::" in first

    def test_screenshot_plus_ui_wording_is_a_design_not_an_image_edit(self, edges):
        chunks = _stream("Make a feature card UI for my blog", image_urls=[REFERENCE_URL])
        assert "design_ready" in _events(chunks)
        assert edges["image_adapter"] == []

    def test_component_wording_never_calls_the_llm_classifier(self, edges):
        chunks = _stream("Create a pricing card UI component")
        assert "design_ready" in _events(chunks)
        assert edges["classifier"] == []

    def test_same_wording_reaches_the_classifier_when_design_is_off(self, edges, monkeypatch):
        # Control: the classifier stub is wired, so the assertion above is meaningful. The stream is stopped at
        # resolve_route (after classification, before any model call).
        class _Stop(Exception):
            pass

        def stop(*args, **kwargs):
            raise _Stop

        monkeypatch.setattr(settings, "design_canvas_enabled", False)
        monkeypatch.setattr("src.task_router.resolve_route", stop)
        with pytest.raises(_Stop):
            _stream("Create a pricing card UI component")
        assert edges["classifier"] == ["Create a pricing card UI component"]


# ===========================================================================
# Gap 3: explicit design operation types reach the engine
# ===========================================================================

class TestExplicitDesignTypes:
    def test_generate_design_type_forces_design_without_keywords(self, edges, monkeypatch):
        captured = {}
        real = tpa._run_design_json_completion

        def capture(messages, *args, **kwargs):
            captured["prompt"] = messages[-1]["content"]
            return real(messages, *args, **kwargs)

        monkeypatch.setattr(tpa, "_run_design_json_completion", capture)
        chunks = _stream("something calm for the pricing page", operation_type="generate_design")
        assert "design_ready" in _events(chunks)
        assert "TASK: Generate a new editable UI design" in captured["prompt"]

    def test_edit_design_type_edits_the_given_ast(self, edges, monkeypatch):
        captured = {}
        real = tpa._run_design_json_completion

        def capture(messages, *args, **kwargs):
            captured["prompt"] = messages[-1]["content"]
            return real(messages, *args, **kwargs)

        monkeypatch.setattr(tpa, "_run_design_json_completion", capture)
        chunks = _stream("greener please", operation_type="edit_design", existing_design_ast=VALID_AST)
        assert "design_ready" in _events(chunks)
        assert "<existing_design_ast>" in captured["prompt"]
        assert "TASK: Edit the existing DesignDocumentAST" in captured["prompt"]

    def test_design_types_map_to_the_free_form_agent(self):
        assert tpa.agent_for_request_type("generate_design") == tpa.FREE_FORM_AGENT
        assert tpa.agent_for_request_type("edit_design") == tpa.FREE_FORM_AGENT


# ===========================================================================
# Gap 5: the design runner's OpenAI call uses the gpt-5 output-limit parameter
# ===========================================================================

def test_openai_design_call_uses_max_completion_tokens(monkeypatch):
    sent = {}

    class _Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            msg = type("M", (), {"content": "{}"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(tpa, "_get_cached_openai_client", lambda *a, **k: client)
    tpa._run_design_json_completion([{"role": "user", "content": "x"}], None, 0.2, "openai", "k")
    assert sent["model"] == "gpt-5.2"
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] >= 4096


# ===========================================================================
# The screenshot: capped download, readable format, real media type
# ===========================================================================

class TestReferenceScreenshot:
    def test_fetched_with_the_capped_helper_and_sent_to_the_model(self, edges):
        chunks = _stream("Recreate this UI as an editable design", image_urls=[REFERENCE_URL])
        assert "design_ready" in _events(chunks)
        assert edges["fetched"] == [REFERENCE_URL]
        assert edges["images"][0] == edges["screenshot"]  # first attempt carries the screenshot

    def test_download_over_the_cap_fails_before_any_model_call(self, edges, monkeypatch):
        class _Response:
            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                while True:
                    yield b"x" * chunk_size  # an endless body: only the cap can stop it

        monkeypatch.setattr(tpa, "_fetch_reference_bytes", REAL_FETCH)
        monkeypatch.setattr(tpa, "_MAX_REFERENCE_BYTES", 200 * 1024)
        monkeypatch.setattr("requests.get", lambda *a, **k: _Response())
        with pytest.raises(ValueError, match="Could not load the reference image: Reference image is too large"):
            _stream("Recreate this UI as an editable design", image_urls=[REFERENCE_URL])
        assert edges["providers"] == []

    def test_unreadable_screenshot_fails_clearly(self, edges):
        edges["screenshot"] = b"<html>not an image</html>"
        with pytest.raises(ValueError, match="Could not load the reference image: not a readable image"):
            _stream("Recreate this UI as an editable design", image_urls=[REFERENCE_URL])
        assert edges["providers"] == []

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_formats_every_provider_accepts_are_sent_as_they_are(self, edges, fmt):
        edges["screenshot"] = _image_bytes(fmt)
        _stream("Recreate this UI as an editable design", image_urls=[REFERENCE_URL])
        assert edges["images"][0] == edges["screenshot"]

    @pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
    def test_other_formats_are_re_encoded_as_png(self, edges, fmt):
        edges["screenshot"] = _image_bytes(fmt)
        _stream("Recreate this UI as an editable design", image_urls=[REFERENCE_URL])
        assert edges["images"][0].startswith(b"\x89PNG\r\n\x1a\n")

    def test_oversized_images_are_downscaled_to_max_dimension(self):
        import io
        from PIL import Image
        from src.text_processing_agent import _design_reference_image, MAX_DESIGN_IMAGE_DIMENSION

        # Create a 3000x2000 image
        img = Image.new("RGB", (3000, 2000), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        processed = _design_reference_image(raw_bytes)
        with Image.open(io.BytesIO(processed)) as out_img:
            w, h = out_img.size
            assert max(w, h) == MAX_DESIGN_IMAGE_DIMENSION
            assert w == 1500 and h == 1000


class TestTruncationDetection:
    def test_claude_truncation_raises(self, monkeypatch):
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return type("Resp", (), {"stop_reason": "max_tokens", "content": [type("C", (), {"text": "{"})()]})()
        monkeypatch.setattr(tpa, "_get_cached_anthropic_client", lambda *a, **k: _Client())
        with pytest.raises(ValueError, match="Design output truncated"):
            tpa._run_design_json_completion([{"role": "user", "content": "hi"}], None, 0.2, "claude", "key")

    def test_openai_truncation_raises(self, monkeypatch):
        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return type("Resp", (), {"choices": [type("Choice", (), {"finish_reason": "length", "message": type("M", (), {"content": "{"})()})]})()
        monkeypatch.setattr(tpa, "_get_cached_openai_client", lambda *a, **k: _Client())
        with pytest.raises(ValueError, match="Design output truncated"):
            tpa._run_design_json_completion([{"role": "user", "content": "hi"}], None, 0.2, "openai", "key")

    def test_gemini_truncation_raises(self, monkeypatch):
        class _Client:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    return type("Resp", (), {"candidates": [type("Cand", (), {"finish_reason": "MAX_TOKENS"})()], "text": "{"})()
        monkeypatch.setattr(tpa, "_get_cached_gemini_client", lambda *a, **k: _Client())
        with pytest.raises(ValueError, match="Design output truncated"):
            tpa._run_design_json_completion([{"role": "user", "content": "hi"}], None, 0.2, "gemini", "key")


class _Recorder:
    """A stand-in SDK client: records the request and returns an empty JSON object."""

    def __init__(self):
        self.sent = {}

    def anthropic(self):
        rec = self

        class _Messages:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                return type("R", (), {"content": [type("B", (), {"text": "{}"})()]})()

        return type("Client", (), {"messages": _Messages()})()

    def openai(self):
        rec = self

        class _Completions:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                msg = type("M", (), {"content": "{}"})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        return type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()

    def gemini(self):
        rec = self

        class _Models:
            def generate_content(self, **kwargs):
                rec.sent.update(kwargs)
                return type("R", (), {"text": "{}"})()

        return type("Client", (), {"models": _Models()})()


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
class TestMediaTypeSentToEachProvider:
    def _call(self, provider, fmt, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(tpa, "_get_cached_anthropic_client", lambda *a, **k: rec.anthropic())
        monkeypatch.setattr(tpa, "_get_cached_openai_client", lambda *a, **k: rec.openai())
        monkeypatch.setattr(tpa, "_get_cached_gemini_client", lambda *a, **k: rec.gemini())
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "recreate"}]
        tpa._run_design_json_completion(messages, _image_bytes(fmt), 0.2, provider, "k")
        return rec.sent

    def test_claude(self, fmt, mime, monkeypatch):
        sent = self._call("claude", fmt, monkeypatch)
        assert sent["messages"][0]["content"][0]["source"]["media_type"] == mime
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_openai(self, fmt, mime, monkeypatch):
        sent = self._call("openai", fmt, monkeypatch)
        assert sent["messages"][1]["content"][1]["image_url"]["url"].startswith(f"data:{mime};base64,")

    def test_gemini(self, fmt, mime, monkeypatch):
        sent = self._call("gemini", fmt, monkeypatch)
        assert sent["contents"][0].parts[0].inline_data.mime_type == mime


# ===========================================================================
# /agent/abort stops a design that is still generating
# ===========================================================================

class TestAbort:
    def test_abort_mid_call_returns_promptly_and_makes_no_further_calls(self, edges, monkeypatch):
        release = threading.Event()
        started = threading.Event()

        def slow_provider(messages, image_bytes, temperature, provider, key, request_id=""):
            edges["providers"].append(provider)
            started.set()
            release.wait(10)  # a model call that would take 10 s
            return "{not json"  # and would need a repair attempt

        monkeypatch.setattr(tpa, "_run_design_json_completion", slow_provider)
        cancel = threading.Event()
        threading.Thread(target=lambda: started.wait(5) and cancel.set(), daemon=True).start()
        try:
            t0 = time.perf_counter()
            chunks = _stream("Make an onboarding card UI", cancel_event=cancel)
            elapsed = time.perf_counter() - t0
        finally:
            release.set()
        time.sleep(0.2)  # let the abandoned call return: its answer must not start a repair attempt

        assert elapsed < 2.0
        assert "design_ready" not in _events(chunks)
        assert "init" in _events(chunks)
        assert edges["providers"] == ["claude"]

    def test_abort_before_generation_makes_no_model_call(self, edges):
        cancel = threading.Event()
        cancel.set()
        chunks = _stream("Make an onboarding card UI", cancel_event=cancel)
        assert "design_ready" not in _events(chunks)
        assert edges["providers"] == []

    def test_no_abort_still_delivers(self, edges):
        chunks = _stream("Make an onboarding card UI", cancel_event=threading.Event())
        assert "design_ready" in _events(chunks)


# ===========================================================================
# Live run 2026-10-27, scenario 5: a follow-up worded "Switch this to..." went to ordinary chat
# ===========================================================================

def test_switch_wording_with_a_design_attached_is_routed_as_an_edit(edges, monkeypatch):
    captured = {}
    real = tpa._run_design_json_completion

    def capture(messages, *args, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return real(messages, *args, **kwargs)

    monkeypatch.setattr(tpa, "_run_design_json_completion", capture)
    chunks = _stream("Switch this to a light theme: white background and dark text. Keep the mint accents.",
                     existing_design_ast=VALID_AST)
    assert "design_ready" in _events(chunks)
    assert "TASK: Edit the existing DesignDocumentAST" in captured["prompt"]
    assert edges["classifier"] == []
