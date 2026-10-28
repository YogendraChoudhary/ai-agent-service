"""
Local speech recognition with faster-whisper (CTranslate2 Whisper).

One model per process, loaded and warmed at startup. Decodes run on a single worker thread: Whisper already uses every core
it is given, so two decodes at once would each take twice as long, and the event loop stays free for audio frames.

Settings that matter for dictation latency (measured, Apple M4, 5 s clip): `small` float32 675 ms, int8 860-940 ms; `base`
250-370 ms; `distil-large-v3` over 4 s. Whisper encodes a padded 30 s window whatever the clip length, so latency is nearly
flat in utterance length.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from config import settings
from src.voice_flow.asr.audio import WHISPER_RATE, rms_dbfs
from src.voice_flow.asr.vocabulary import build_initial_prompt, is_hallucination, is_prompt_echo

logger = logging.getLogger(__name__)

MIN_AUDIO_S = 0.15
SILENCE_DBFS = -58.0                    # a clip quieter than this overall holds no speech worth decoding
_QUIET_DBFS = -42.0                     # a known silence-hallucination is dropped only when the clip is also this quiet


@dataclass(frozen=True)
class AsrResult:
    text: str
    audio_ms: float
    asr_ms: float
    language: str = ""


class Transcriber(Protocol):
    async def transcribe(self, audio: np.ndarray, custom_vocab: Sequence[str] = (), language: Optional[str] = None) -> AsrResult: ...


class WhisperTranscriber:
    def __init__(self, model: Optional[str] = None, compute_type: Optional[str] = None,
                 cpu_threads: Optional[int] = None, language: Optional[str] = None):
        self.model_name = model or settings.voice_whisper_model
        self.compute_type = compute_type or settings.voice_whisper_compute_type
        self.cpu_threads = settings.voice_whisper_threads if cpu_threads is None else cpu_threads
        self.language = (settings.voice_whisper_language if language is None else language) or None
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> float:
        """Load the model and run one decode so the first dictation does not pay for kernel set-up. Returns seconds."""
        if self._model is not None:
            return 0.0
        from faster_whisper import WhisperModel
        t0 = time.perf_counter()
        self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type, cpu_threads=self.cpu_threads)
        noise = np.random.default_rng(0).normal(0.0, 0.01, 2 * WHISPER_RATE).astype(np.float32)
        self._decode(noise, build_initial_prompt(language=self.language), self.language)   # the exact path dictation takes, so Silero VAD loads now
        elapsed = time.perf_counter() - t0
        logger.info("voice_flow: Whisper %s (%s, %d threads) ready in %.1fs", self.model_name, self.compute_type,
                    self.cpu_threads, elapsed)
        return elapsed

    async def aload(self) -> float:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.load)

    def _decode(self, audio: np.ndarray, prompt: str, language: Optional[str]) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            audio.astype(np.float32, copy=False),
            language=language,
            beam_size=1,                              # greedy: the formatting model repairs what extra beams would
            temperature=0.0,                          # no fallback re-decodes, which would double the latency
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 2000, "speech_pad_ms": 300},
            no_speech_threshold=0.6,
        )
        return " ".join(s.text.strip() for s in segments).strip(), info.language

    # -- decoding ----------------------------------------------------------
    def transcribe_sync(self, audio: np.ndarray, custom_vocab: Sequence[str] = (), language: Optional[str] = None) -> AsrResult:
        t0 = time.perf_counter()
        wanted = language or self.language                       # a per-request language beats the server setting
        audio_ms = audio.size / WHISPER_RATE * 1000.0
        if audio.size < MIN_AUDIO_S * WHISPER_RATE or rms_dbfs(audio) < SILENCE_DBFS:
            return AsrResult("", audio_ms, (time.perf_counter() - t0) * 1000.0, wanted or "")
        self.load()
        prompt = build_initial_prompt(custom_vocab, wanted)
        text, detected = self._decode(audio, prompt, wanted)
        if is_prompt_echo(text, prompt) or (is_hallucination(text, detected or wanted) and rms_dbfs(audio) < _QUIET_DBFS):
            text = ""
        return AsrResult(text, audio_ms, (time.perf_counter() - t0) * 1000.0, detected)

    async def transcribe(self, audio: np.ndarray, custom_vocab: Sequence[str] = (), language: Optional[str] = None) -> AsrResult:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self.transcribe_sync, audio, tuple(custom_vocab), language)


_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber
