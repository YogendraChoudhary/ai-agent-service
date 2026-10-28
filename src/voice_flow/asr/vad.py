"""
Utterance end detection for the hands-free ("auto") WebSocket mode.

An energy gate with an adaptive noise floor: cheap enough to run on every audio frame and deterministic enough to test.
Push-to-talk does not need it (the client says when the utterance is over); Whisper's own Silero filter still trims
silence inside the clip before decoding.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.voice_flow.asr.audio import pcm16_to_float, rms_dbfs

SPEECH_START = "speech_start"
UTTERANCE_END = "utterance_end"

FRAME_MS = 30
MIN_SPEECH_DB = -45.0          # quieter than this is never speech
SPEECH_OVER_FLOOR_DB = 12.0    # speech must rise this far above the room
FLOOR_MIN_DB, FLOOR_MAX_DB = -70.0, -40.0
START_FRAMES = 3               # 90 ms of voiced frames before "speech_start"


class EndpointDetector:
    def __init__(self, sample_rate: int = 16000, silence_ms: int = 800, min_speech_ms: int = 200):
        self.frame_len = sample_rate * FRAME_MS // 1000
        self.silence_frames = max(1, silence_ms // FRAME_MS) if silence_ms > 0 else 10 ** 9      # 0: never; the client says when
        self.min_speech_frames = max(START_FRAMES, min_speech_ms // FRAME_MS)
        self.reset()

    def reset(self) -> None:
        self._pending = b""
        self._floor_db = -60.0
        self._voiced_run = 0
        self._speech_frames = 0
        self._silent_run = 0
        self.speaking = False

    def feed(self, pcm16: bytes) -> List[str]:
        """Consume PCM16 bytes of any length; return the events they completed, in order."""
        events: List[str] = []
        data = self._pending + pcm16
        step = self.frame_len * 2
        usable = len(data) - (len(data) % step)
        self._pending = data[usable:]
        for offset in range(0, usable, step):
            event = self._frame(rms_dbfs(pcm16_to_float(data[offset: offset + step])))
            if event:
                events.append(event)
        return events

    def _frame(self, level_db: float) -> str | None:
        voiced = level_db > max(MIN_SPEECH_DB, self._floor_db + SPEECH_OVER_FLOOR_DB)
        if not voiced and not self.speaking:
            self._floor_db = float(np.clip(0.95 * self._floor_db + 0.05 * level_db, FLOOR_MIN_DB, FLOOR_MAX_DB))

        if voiced:
            self._voiced_run += 1
            self._silent_run = 0
            if self.speaking:
                self._speech_frames += 1
            elif self._voiced_run >= START_FRAMES:
                self.speaking = True
                self._speech_frames = self._voiced_run
                return SPEECH_START
            return None

        self._voiced_run = 0
        if not self.speaking:
            return None
        self._silent_run += 1
        if self._silent_run >= self.silence_frames:
            enough = self._speech_frames >= self.min_speech_frames
            self.speaking = False
            self._speech_frames = 0
            self._silent_run = 0
            return UTTERANCE_END if enough else None
        return None
