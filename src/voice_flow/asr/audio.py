"""Small audio helpers: PCM16 <-> float32, resampling, loudness, and decoding any container Whisper's loader (PyAV) reads."""

from __future__ import annotations

import io
import math

import numpy as np

WHISPER_RATE = 16000


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono PCM -> float32 in [-1, 1]. A trailing odd byte is dropped."""
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data, dtype="<i2", count=usable // 2).astype(np.float32) / 32768.0


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int = WHISPER_RATE) -> np.ndarray:
    """Cheap linear resampling, with aliasing: only the fallback for resample_audio."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    n_out = int(round(audio.size * dst_rate / src_rate))
    return np.interp(np.linspace(0, audio.size - 1, n_out), np.arange(audio.size), audio).astype(np.float32)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int = WHISPER_RATE) -> np.ndarray:
    """Anti-aliased resampling (libswresample through PyAV, already a dependency; no scipy needed).

    Linear interpolation folds everything above the new Nyquist back into the speech band, which smears the sibilants (s, sh, f, th)
    Whisper relies on. Browsers are asked for 16 kHz, so this is the path for the ones that ignore the request."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    try:
        import av
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = src_rate
        resampler = av.AudioResampler(format="s16", layout="mono", rate=dst_rate)
        parts = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
        if parts:
            return np.concatenate(parts).astype(np.float32) / 32768.0
    except Exception:                                                      # noqa: BLE001 - never lose a dictation over a filter
        pass
    return resample_linear(audio, src_rate, dst_rate)


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-9 else -120.0


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """wav / webm (opus) / m4a / ogg ... -> mono float32 at 16 kHz.

    Uses PyAV directly: faster-whisper 1.2.1's own `decode_audio` passes an argument PyAV 19 rejects.
    """
    import av
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=WHISPER_RATE)
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))     # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0
