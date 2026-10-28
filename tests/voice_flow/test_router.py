import base64
import io
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

    async def transcribe(self, audio: np.ndarray, custom_vocab: Sequence[str] = (), language=None) -> AsrResult:
        self.calls.append((audio.size, tuple(custom_vocab)))
        self.languages.append(language)
        return AsrResult(self.text if audio.size else "", audio.size / RATE * 1000, 12.0, "en")


def pcm(seconds=1.0, amp=0.1) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def silence(seconds) -> bytes:
    return bytes(int(seconds * RATE) * 2)


def wav_b64(seconds=1.0) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(pcm(seconds))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def setup(monkeypatch):
    backend = FakeBackend("We have two tasks:\n1. Update the schema\n2. Deploy the service")
    service = VoiceFormattingService(backend=backend, timeout=1.0, breaker=Breaker(), mode="always")
    transcriber = FakeTranscriber()
    monkeypatch.setattr(router_module, "_service", service)
    monkeypatch.setattr(router_module, "get_transcriber", lambda: transcriber)
    monkeypatch.setattr(settings, "voice_flow_token", "")
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app), backend, transcriber, service


START = {"type": "start", "sample_rate": RATE, "mode": "push_to_talk", "app_context": "general"}


class TestFormatEndpoint:
    def test_formats_and_stitches(self, setup):
        client, backend, *_ = setup
        backend.reply = "that the Redis pool ran out."
        r = client.post("/v1/voice/format", json={
            "raw_text": "um that the redis pool ran out", "app_context": "slack_chat",
            "cursor_context": {"preceding_text": "The root cause is"}})
        body = r.json()
        assert r.status_code == 200 and body["source"] == "llm"
        assert body["formatted_text"] == " that the Redis pool ran out." and body["leading_space_required"] is True

    def test_model_down_still_returns_text_and_says_why(self, setup):
        client, backend, *_ = setup
        backend.error = LlmError("gemini 402 RESOURCE_EXHAUSTED", permanent=True)
        body = client.post("/v1/voice/format", json={"raw_text": "how do I center a div in css"}).json()
        assert body["source"] == "rules" and body["formatted_text"] == "How do I center a div in CSS?"
        assert "402" in body["fallback_reason"]

    def test_input_limits(self, setup):
        client, *_ = setup
        assert client.post("/v1/voice/format", json={"raw_text": "x" * 8001}).status_code == 422
        assert client.post("/v1/voice/format", json={"raw_text": "hi", "app_context": "nope"}).status_code == 422

    def test_token_is_required_when_configured(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_flow_token", "s3cret")
        assert client.post("/v1/voice/format", json={"raw_text": "hi"}).status_code == 401
        assert client.post("/v1/voice/format", json={"raw_text": "hi"}, headers={"x-voice-token": "s3cret"}).status_code == 200


class TestDictateEndpoint:
    def test_audio_in_text_out(self, setup):
        client, _, transcriber, _ = setup
        r = client.post("/v1/voice/dictate", json={"audio_b64": wav_b64(1.0), "custom_dictionary": ["Redis"]})
        body = r.json()
        assert r.status_code == 200 and body["formatted_text"].startswith("We have two tasks:")
        assert body["audio_ms"] == pytest.approx(1000, abs=5) and body["asr_latency_ms"] == 12.0
        assert transcriber.calls == [(RATE, ("Redis",))]

    def test_bad_input(self, setup):
        client, *_ = setup
        assert client.post("/v1/voice/dictate", json={"audio_b64": "!!!not base64"}).status_code == 400
        assert client.post("/v1/voice/dictate", json={"audio_b64": base64.b64encode(b"definitely not audio").decode()}).status_code == 400

    def test_too_long(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_max_utterance_seconds", 5)
        assert client.post("/v1/voice/dictate", json={"audio_b64": wav_b64(6.0)}).status_code == 413


def test_status_and_demo(setup):
    client, _, _, service = setup
    status = client.get("/v1/voice/status").json()
    assert status["whisper_model"] == "fake-whisper" and status["format_model_available"] is True
    service.breaker.failure(LlmError("gemini 402", permanent=True))
    status = client.get("/v1/voice/status").json()
    assert status["format_model_available"] is False and status["last_format_error"] == "gemini 402"
    page = client.get("/v1/voice/demo")
    assert page.status_code == 200 and "Voice Flow test" in page.text


class TestStream:
    def test_push_to_talk_round_trip(self, setup):
        client, backend, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "custom_dictionary": ["Redis"]})
            assert ws.receive_json() == {"type": "ready"}
            for _ in range(5):
                ws.send_bytes(pcm(0.2))
            ws.send_json({"type": "end"})
            final = ws.receive_json()
        assert final["type"] == "final" and final["source"] == "llm"
        assert final["raw_text"].startswith("we have two tasks")
        assert final["formatted_text"] == "We have two tasks:\n1. Update the schema\n2. Deploy the service"
        t = final["timings"]
        assert t["audio_ms"] == pytest.approx(1000, abs=5) and t["asr_ms"] == 12.0 and t["total_ms"] >= t["post_ms"] >= 0
        assert transcriber.calls == [(RATE, ("Redis",))]

    def test_caret_context_reaches_the_model_and_the_stitching(self, setup):
        client, backend, transcriber, _ = setup
        transcriber.text = "um the logs and the metrics"
        backend.reply = "the logs and the metrics."
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "cursor_context": {"preceding_text": "Please check", "following_text": " tomorrow"}})
            ws.receive_json()
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            final = ws.receive_json()
        assert final["leading_space_required"] is True and final["formatted_text"] == " the logs and the metrics"
        sent = backend.calls[0][1][-1]["content"]
        assert "<preceding_text>Please check</preceding_text>" in sent and "<following_text> tomorrow</following_text>" in sent

    def test_utterances_come_back_in_order_on_one_connection(self, setup):
        client, _, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            for seconds in (0.5, 1.0, 1.5):
                ws.send_bytes(pcm(seconds))
                ws.send_json({"type": "end"})
            audio_ms = [ws.receive_json()["timings"]["audio_ms"] for _ in range(3)]
        assert [round(a) for a in audio_ms] == [500, 1000, 1500]

    def test_end_with_no_audio_still_answers(self, setup):
        client, *_ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_json({"type": "end"})
            final = ws.receive_json()
        assert final["raw_text"] == "" and final["fallback_reason"] == "no speech detected"

    def test_hands_free_mode_ends_the_utterance_on_a_pause_the_client_asked_for(self, setup):
        client, *_ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "mode": "auto", "silence_ms": 800})
            ws.receive_json()
            ws.send_bytes(silence(0.3))
            ws.send_bytes(pcm(1.0))
            assert ws.receive_json() == {"type": "speech_start"}
            ws.send_bytes(silence(1.2))
            final = ws.receive_json()
        assert final["type"] == "final" and final["raw_text"]

    def test_a_dictation_with_thinking_pauses_is_one_result_not_many(self, setup):
        """A 2-minute monologue arrived as nine separately formatted pieces when 0.8 s of silence ended an utterance."""
        client, _, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "mode": "auto"})                 # the server default: several seconds, not 0.8
            ws.receive_json()
            for _ in range(4):                                      # four stretches of speech, each followed by a 2.5 s pause
                ws.send_bytes(pcm(1.0))
                ws.send_bytes(silence(2.5))
            ws.send_json({"type": "end"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "final":
                    break
        assert [f["type"] for f in frames if f["type"] == "final"] == ["final"]
        final = frames[-1]
        assert final["timings"]["audio_ms"] == pytest.approx(4 * 3500, abs=20)         # all of it, pauses included
        assert len(transcriber.calls) == 1 and transcriber.calls[0][0] == 4 * 3.5 * RATE   # one recognition pass over everything

    def test_silence_zero_means_only_the_client_can_finish(self, setup):
        client, _, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "mode": "auto", "silence_ms": 0})
            ws.receive_json()
            ws.send_bytes(pcm(0.5))
            assert ws.receive_json() == {"type": "speech_start"}
            ws.send_bytes(silence(30.0))                            # half a minute of quiet: still the same dictation
            assert transcriber.calls == []
            ws.send_json({"type": "end"})
            final = ws.receive_json()
        assert final["type"] == "final" and len(transcriber.calls) == 1

    def test_invalid_silence_is_rejected_without_closing(self, setup):
        client, *_ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "mode": "auto", "silence_ms": 100})
            assert ws.receive_json()["code"] == "bad_start"
            ws.send_json({**START, "mode": "auto", "silence_ms": 4000})
            assert ws.receive_json() == {"type": "ready"}

    def test_reaching_the_length_limit_says_so(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_max_utterance_seconds", 5)
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            for _ in range(7):
                ws.send_bytes(pcm(1.0))
            frames = [ws.receive_json(), ws.receive_json()]
        assert frames[0]["type"] == "notice" and frames[0]["code"] == "max_duration"
        assert frames[1]["type"] == "final"

    def test_the_default_limit_is_long_enough_for_a_real_dictation(self):
        from config import Settings
        s = Settings()
        assert s.voice_max_utterance_seconds >= 600 and s.voice_endpoint_silence_ms >= 5000

    def test_cancel_drops_the_audio(self, setup):
        client, _, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "cancel"})
            assert ws.receive_json() == {"type": "cancelled"}
            ws.send_json({"type": "end"})
            assert ws.receive_json()["raw_text"] == ""
        assert transcriber.calls == []                       # nothing was left to recognise, and Whisper was not asked

    def test_protocol_errors_keep_the_connection_open(self, setup):
        client, *_ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_bytes(pcm(0.1))
            assert ws.receive_json()["code"] == "no_session"
            ws.send_text("not json")
            assert ws.receive_json()["code"] == "bad_frame"
            ws.send_json({"type": "start", "sample_rate": 5})
            assert ws.receive_json()["code"] == "bad_start"
            ws.send_json({"type": "bogus"})
            assert ws.receive_json()["code"] == "unexpected_frame"
            ws.send_json(START)
            assert ws.receive_json() == {"type": "ready"}

    def test_wrong_token_closes_the_socket(self, setup, monkeypatch):
        client, *_ = setup
        monkeypatch.setattr(settings, "voice_flow_token", "s3cret")
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "token": "wrong"})
            assert ws.receive_json()["code"] == "unauthorized"

    def test_other_sample_rates_are_resampled(self, setup):
        client, _, transcriber, _ = setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "sample_rate": 48000})
            ws.receive_json()
            ws.send_bytes(bytes(48000 * 2))                       # 1 s at 48 kHz
            ws.send_json({"type": "end"})
            ws.receive_json()
        assert transcriber.calls[0][0] == pytest.approx(RATE, abs=200)       # swr trims/pads a few samples at the edges

    def test_a_formatter_crash_does_not_kill_the_session(self, setup, monkeypatch):
        client, backend, *_ = setup
        backend.error = RuntimeError("boom")                       # contained by the service: rules answer
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            final = ws.receive_json()
        assert final["source"] == "rules" and final["formatted_text"].startswith("We have two")
