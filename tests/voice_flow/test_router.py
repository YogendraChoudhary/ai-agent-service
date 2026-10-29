import asyncio
import io
import json
import threading
import time
import wave
from typing import Sequence

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from src.voice_flow import router as router_module
from src.voice_flow.asr.whisper_client import AsrResult
from src.voice_flow.formatter.engine import Breaker, LlmError, VoiceFormattingService
from tests.voice_flow.test_engine import FakeBackend

RATE = 16000


class FakeTranscriber:
    model_name, compute_type, _model = "fake-whisper", "float32", object()

    def __init__(self, text="we have two tasks mark it point 1 update the schema and point to deploy the service"):
        self.text, self.calls, self.languages = text, [], []
        self.go = threading.Event()
        self.go.set()                                          # clear() it to make the speech model slow

    async def transcribe(self, audio: np.ndarray, custom_vocab: Sequence[str] = (), language=None) -> AsrResult:
        self.calls.append((audio.size, tuple(custom_vocab)))
        self.languages.append(language)
        while not self.go.is_set():
            await asyncio.sleep(0.01)
        return AsrResult(self.text if audio.size else "", audio.size / RATE * 1000, 12.0, "en")


def wav(seconds=1.0, rate=RATE, amp=0.1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    samples = (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(samples)
    return buf.getvalue()


def dictate(client, audio=None, context=None, headers=None):
    """POST a recording the way a browser does: multipart with an `audio` file and a `context` JSON text part."""
    files = {"audio": ("dictation.wav", wav(1.0) if audio is None else audio, "audio/wav")}
    data = {} if context is None else {"context": context if isinstance(context, str) else json.dumps(context)}
    return client.post("/v1/voice/dictate", files=files, data=data, headers=headers or {})


@pytest.fixture
def setup(monkeypatch):
    backend = FakeBackend("We have two tasks:\n1. Update the schema\n2. Deploy the service")
    service = VoiceFormattingService(backend=backend, timeout=1.0, breaker=Breaker(), mode="always")
    transcriber = FakeTranscriber()
    monkeypatch.setattr(router_module, "_service", service)
    monkeypatch.setattr(router_module, "get_transcriber", lambda: transcriber)
    monkeypatch.setattr(router_module, "_gate", None)
    monkeypatch.setattr(settings, "voice_flow_token", "")
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app), backend, transcriber, service


class TestDictate:
    def test_audio_in_text_out(self, setup):
        client, _, transcriber, _ = setup
        r = dictate(client, wav(1.0), {"custom_dictionary": ["Redis"]})
        body = r.json()
        assert r.status_code == 200 and body["source"] == "llm"
        assert body["raw_text"].startswith("we have two tasks")
        assert body["formatted_text"] == "We have two tasks:\n1. Update the schema\n2. Deploy the service"
        t = body["timings"]
        assert t["audio_ms"] == pytest.approx(1000, abs=5) and t["asr_ms"] == 12.0 and t["total_ms"] >= t["post_ms"] >= 0
        assert transcriber.calls == [(RATE, ("Redis",))]

    def test_the_context_part_is_optional(self, setup):
        client, *_ = setup
        assert dictate(client).status_code == 200

    def test_caret_context_reaches_the_model_and_the_stitching(self, setup):
        client, backend, transcriber, _ = setup
        transcriber.text = "um the logs and the metrics"
        backend.reply = "the logs and the metrics."
        r = dictate(client, wav(0.5), {"cursor_context": {"preceding_text": "Please check", "following_text": " tomorrow"}})
        body = r.json()
        assert body["leading_space_required"] is True and body["formatted_text"] == " the logs and the metrics"
        sent = backend.calls[0][1][-1]["content"]
        assert "<preceding_text>Please check</preceding_text>" in sent and "<following_text> tomorrow</following_text>" in sent

    def test_style_language_and_dictionary_reach_the_pipeline(self, setup):
        client, backend, transcriber, _ = setup
        dictate(client, wav(0.5), {"custom_dictionary": ["Redis", "Kubernetes"], "language": "en", "style": "polished",
                                   "cursor_context": {"preceding_text": "Please check ", "following_text": " tomorrow"}})
        assert transcriber.calls == [(RATE // 2, ("Redis", "Kubernetes"))] and transcriber.languages == ["en"]
        system, turns = backend.calls[0][0], backend.calls[0][1]
        assert "Polished style" in system
        assert "<preceding_text>Please check </preceding_text>" in turns[-1]["content"]
        assert "<dictionary>Redis, Kubernetes</dictionary>" in turns[-1]["content"]

    def test_unicode_context_survives_the_upload(self, setup):
        client, backend, *_ = setup
        dictate(client, wav(0.5), {"cursor_context": {"preceding_text": "Привет, мир — café "}, "custom_dictionary": ["Zoë"]})
        content = backend.calls[0][1][-1]["content"]
        assert "Привет, мир — café" in content and "Zoë" in content

    def test_utterances_of_any_sample_rate_are_resampled(self, setup):
        client, _, transcriber, _ = setup
        assert dictate(client, wav(1.0, rate=48000)).status_code == 200
        assert transcriber.calls[0][0] == pytest.approx(RATE, abs=200)       # swr trims/pads a few samples at the edges

    def test_a_recording_too_short_to_be_speech_never_reaches_whisper(self, setup):
        client, _, transcriber, _ = setup
        body = dictate(client, wav(0.02)).json()
        assert body["raw_text"] == "" and body["formatted_text"] == "" and body["fallback_reason"] == "no speech detected"
        assert transcriber.calls == []

    def test_nothing_recognised_says_so(self, setup):
        client, _, transcriber, _ = setup
        transcriber.text = ""
        r = dictate(client, wav(1.0))
        assert r.status_code == 200 and r.json()["formatted_text"] == "" and r.json()["fallback_reason"] == "no speech detected"
        assert len(transcriber.calls) == 1


class TestTheFormatterBeingDown:
    def test_model_down_still_returns_text_and_says_why(self, setup):
        client, backend, transcriber, _ = setup
        transcriber.text = "how do I center a div in css"
        backend.error = LlmError("gemini 402 RESOURCE_EXHAUSTED", permanent=True)
        body = dictate(client).json()
        assert body["source"] == "rules" and body["formatted_text"] == "How do I center a div in CSS?"
        assert "402" in body["fallback_reason"]

    def test_a_formatter_crash_does_not_fail_the_request(self, setup):
        client, backend, *_ = setup
        backend.error = RuntimeError("boom")                       # contained by the service: rules answer
        r = dictate(client)
        assert r.status_code == 200 and r.json()["source"] == "rules" and r.json()["formatted_text"].startswith("We have two")


class TestRefusals:
    def test_no_audio_part(self, setup):
        client, *_ = setup
        assert client.post("/v1/voice/dictate", data={"context": "{}"}).status_code == 400

    def test_empty_and_undecodable_recordings(self, setup):
        client, _, transcriber, _ = setup
        assert dictate(client, b"").status_code == 400
        assert dictate(client, b"definitely not audio").status_code == 400
        assert transcriber.calls == [] and router_module._in_flight == 0

    @pytest.mark.parametrize("context", [
        "not json", "[]", '{"style": "fancy"}', '{"app_context": "nope"}', '{"language": "english-us!"}',
        '{"cursor_context": {"preceding_text": 5}}', "x" * (router_module.CONTEXT_MAX_CHARS + 1)])
    def test_a_bad_context_is_a_400_before_any_work(self, setup, context):
        client, _, transcriber, _ = setup
        assert dictate(client, context=context).status_code == 400
        assert transcriber.calls == []

    def test_too_long(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_max_utterance_seconds", 5)
        assert dictate(client, wav(6.0)).status_code == 413
        assert router_module._in_flight == 0                       # a refused request frees its slot

    def test_too_big_a_file(self, setup, monkeypatch):
        client, _, transcriber, _ = setup
        monkeypatch.setattr(router_module, "AUDIO_MAX_BYTES", 1000)
        assert dictate(client, wav(1.0)).status_code == 413
        assert transcriber.calls == []

    def test_the_token_is_required_when_configured(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_flow_token", "s3cret")
        assert dictate(client).status_code == 401
        assert dictate(client, headers={"x-voice-token": "wrong"}).status_code == 401
        assert dictate(client, headers={"x-voice-token": "s3cret"}).status_code == 200

    def test_a_busy_server_answers_429_and_recovers(self, setup, monkeypatch):
        client, _, transcriber, _ = setup
        monkeypatch.setattr(router_module, "MAX_DICTATIONS_IN_FLIGHT", 1)
        transcriber.go.clear()                                      # the speech model is stuck on the first recording
        first = {}
        worker = threading.Thread(target=lambda: first.update(r=dictate(client)))
        worker.start()
        deadline = time.time() + 5
        while not transcriber.calls and time.time() < deadline:
            time.sleep(0.01)
        refused = dictate(client)
        assert refused.status_code == 429 and refused.headers["retry-after"] == "2"
        transcriber.go.set()
        worker.join(5)
        assert first["r"].status_code == 200 and router_module._in_flight == 0
        assert dictate(client).status_code == 200

    def test_limits_are_what_the_review_asked_for(self):
        assert router_module.MIN_UTTERANCE_SECONDS == 0.05 and router_module.MAX_DICTATIONS_IN_FLIGHT == 8
        assert router_module.AUDIO_MAX_BYTES == 10 * 1024 * 1024

    def test_the_default_limit_is_long_enough_for_a_real_dictation(self):
        from config import Settings
        assert Settings().voice_max_utterance_seconds >= 600


def test_status(setup):
    client, _, _, service = setup
    status = client.get("/v1/voice/status").json()
    assert status["whisper_model"] == "fake-whisper" and status["format_model_available"] is True
    service.breaker.failure(LlmError("gemini 402", permanent=True))
    status = client.get("/v1/voice/status").json()
    assert status["format_model_available"] is False and status["last_format_error"] == "gemini 402"


def test_the_api_serves_only_dictate_and_status(setup):
    client, *_ = setup
    assert sorted(r.path for r in router_module.router.routes) == ["/v1/voice/dictate", "/v1/voice/status"]
    for path in ("/v1/voice/format", "/v1/voice/stream", "/v1/voice/demo"):
        assert client.post(path, json={}).status_code == 404 and client.get(path).status_code == 404, path
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/voice/stream"):
            pass
    # the mic test page lives in scripts/voice_demo.py only
