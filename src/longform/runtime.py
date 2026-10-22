"""
Binds the refinement pipeline to this app: the provider candidates and keys of the request, the model call with its fallback
rules, and the settings. refine.py stays free of all of it.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator, Optional, Sequence, Union

from config import settings

from .llm import AgnoLLM
from .prompts import OPERATIONS
from .refine import RefineConfig, refine_stream

logger = logging.getLogger(__name__)


def refine_config() -> RefineConfig:
    """The pipeline's knobs, from the LONGFORM_* settings."""
    return RefineConfig(
        chunk_words=int(settings.longform_chunk_words), first_chunk_words=int(settings.longform_first_chunk_words),
        parallel=int(settings.longform_parallel),
        freeze_quotes=bool(settings.longform_freeze_quotes), repair=bool(settings.longform_repair_enabled),
        min_words=int(settings.longform_refine_min_words), word_ratio=settings.longform_word_ratio or None)


def stream_refine(
    text: str,
    instruction: Optional[str],
    agent_type: str,
    *,
    candidates: Sequence,
    key_for: Callable[[str], Optional[str]],
    is_byok: bool,
    user_id: Optional[str],
    request_id: Optional[str],
    report_model: bool = False,
) -> Iterator[Union[str, dict]]:
    """
    The refinement pipeline for one request. `candidates` are the providers the single call would have tried (the operation's
    tier already applied); a chunk's call moves to the next one on the same retryable errors, exactly as the main loop does.
    """
    llm = AgnoLLM(candidates, key_for, request_id=request_id, user_id=user_id, is_byok=is_byok, agent_type="longform_refiner",
                  timeout=float(settings.longform_call_timeout_seconds) or None)
    reported = False
    for chunk in refine_stream(text, instruction, OPERATIONS[agent_type], llm, refine_config(), request_id=request_id):
        if report_model and not reported and isinstance(chunk, str):
            reported = True
            model = llm.used or llm.drafting_model()               # the model that served the first chunk
            if model is not None:
                yield {"__model_used__": model}
        yield chunk
