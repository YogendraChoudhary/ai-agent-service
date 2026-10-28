import numpy as np
import pytest

from src.voice_flow.asr.audio import pcm16_to_float, resample_linear, rms_dbfs
from src.voice_flow.asr.vad import SPEECH_START, UTTERANCE_END, EndpointDetector
from src.voice_flow.asr.vocabulary import MAX_TERMS_CHARS, STRUCTURAL_SEED, build_initial_prompt, is_prompt_echo

RATE = 16000


def tone(seconds: float, amp: float = 0.1, hz: float = 220.0) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype("<i2").tobytes()


def quiet(seconds: float, amp: float = 0.0005) -> bytes:
    rng = np.random.default_rng(1)
    return (rng.normal(0, amp, int(seconds * RATE)) * 32767).astype("<i2").tobytes()


def feed_all(detector: EndpointDetector, audio: bytes, chunk: int = 3200):
    events = []
    for i in range(0, len(audio), chunk):
        events += detector.feed(audio[i: i + chunk])
    return events


class TestAudio:
    def test_pcm_roundtrip_and_odd_byte(self):
        x = pcm16_to_float(np.array([0, 16384, -32768], dtype="<i2").tobytes() + b"\x01")
        assert x.dtype == np.float32 and x.tolist() == [0.0, 0.5, -1.0]

    def test_resample_length_and_identity(self):
        x = np.linspace(-1, 1, 48000, dtype=np.float32)
        assert resample_linear(x, 48000, 16000).size == 16000
        assert resample_linear(x, 16000, 16000) is x

    def test_loudness(self):
        assert rms_dbfs(np.zeros(100, dtype=np.float32)) == -120.0
        assert rms_dbfs(np.full(100, 0.1, dtype=np.float32)) == pytest.approx(-20.0, abs=0.1)


class TestEndpointDetector:
    def test_silence_produces_no_events(self):
        assert feed_all(EndpointDetector(RATE, 800), quiet(3)) == []

    def test_speech_then_pause_is_one_utterance(self):
        d = EndpointDetector(RATE, 800)
        events = feed_all(d, quiet(0.3) + tone(1.0) + quiet(1.2))
        assert events == [SPEECH_START, UTTERANCE_END]

    def test_a_short_pause_inside_speech_does_not_end_it(self):
        d = EndpointDetector(RATE, 800)
        events = feed_all(d, tone(0.6) + quiet(0.4) + tone(0.6) + quiet(1.2))
        assert events == [SPEECH_START, UTTERANCE_END]

    def test_a_click_is_not_an_utterance(self):
        d = EndpointDetector(RATE, 800)
        assert UTTERANCE_END not in feed_all(d, quiet(0.3) + tone(0.1) + quiet(1.5))

    def test_two_utterances(self):
        d = EndpointDetector(RATE, 600)
        events = feed_all(d, tone(0.8) + quiet(1.0) + tone(0.8) + quiet(1.0))
        assert events == [SPEECH_START, UTTERANCE_END, SPEECH_START, UTTERANCE_END]

    def test_odd_sized_chunks_give_the_same_events(self):
        audio = quiet(0.3) + tone(1.0) + quiet(1.2)
        assert feed_all(EndpointDetector(RATE, 800), audio, chunk=997 * 2) == [SPEECH_START, UTTERANCE_END]

    def test_silence_zero_never_ends_the_utterance(self):
        d = EndpointDetector(RATE, 0)
        assert feed_all(d, tone(0.6) + quiet(60.0)) == [SPEECH_START]

    def test_long_thinking_pauses_stay_inside_one_utterance(self):
        d = EndpointDetector(RATE, 6000)
        assert feed_all(d, tone(1.0) + quiet(3.0) + tone(1.0) + quiet(4.0) + tone(1.0) + quiet(7.0)) == [SPEECH_START, UTTERANCE_END]

    def test_background_hum_does_not_count_as_speech(self):
        d = EndpointDetector(RATE, 800)
        assert feed_all(d, tone(3.0, amp=0.003)) == []                       # ~ -50 dBFS


class TestVocabulary:
    def test_seed_carries_the_structural_cues(self):
        prompt = build_initial_prompt()
        assert prompt == STRUCTURAL_SEED and "point 1" in prompt and "bullets" in prompt

    def test_custom_terms_are_appended_and_capped(self):
        assert build_initial_prompt(["Kubernetes", "Sarah Okafor"]).endswith("Kubernetes, Sarah Okafor.")
        long = build_initial_prompt([f"term{i:03d}word" for i in range(200)])
        assert len(long) <= len(STRUCTURAL_SEED) + MAX_TERMS_CHARS + 4

    def test_prompt_echo(self):
        prompt = build_initial_prompt()
        assert is_prompt_echo("Mark it point 1, point 2.", prompt)
        assert not is_prompt_echo("Mark it point 1, update the schema", prompt)
        assert not is_prompt_echo("", prompt) and not is_prompt_echo("Bullets", prompt)     # one word may be real speech
