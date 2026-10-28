"""
Biasing Whisper towards the words dictation depends on, per language.

The plan boosts structural cues acoustically (Deepgram `keywords`). Whisper has no per-word boosts; what it has is
`initial_prompt`, text the decoder is conditioned on as if it had just heard it. Measured on this machine, with the cue
phrase in the prompt "mark it point 1" decodes correctly on `base` and `small`; without it the same audio came out as
"market.1" / "Market Point 1". `hotwords` did not help, so it is not used.

The seed phrases are English. For any other language they are left out, because English cue words in the prompt would
pull a non-English transcription towards English. Whisper reads at most 224 prompt tokens, so the seed stays short and
custom terms are capped.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Optional, Sequence

STRUCTURAL_SEEDS: Dict[str, str] = {
    "en": "Mark it point 1, point 2. Make it bullets. New paragraph.",
}
STRUCTURAL_SEED = STRUCTURAL_SEEDS["en"]
MAX_TERMS = 40
MAX_TERMS_CHARS = 320

# What Whisper writes for silence and room noise, per language. English is always checked as well.
HALLUCINATIONS_BY_LANG: Dict[str, FrozenSet[str]] = {
    "en": frozenset({"you", "thank you", "thanks", "thanks for watching", "thank you for watching", "bye", "so"}),
    "hi": frozenset({"धन्यवाद", "सब्सक्राइब करें"}),
    "es": frozenset({"gracias", "subtítulos por la comunidad"}),
}


def _base_language(language: Optional[str]) -> str:
    return (language or "en").replace("_", "-").split("-")[0].lower()


def get_structural_seed(language: Optional[str]) -> str:
    """The English cue phrases for English (or an unspecified language); nothing for any other language."""
    if not language:
        return STRUCTURAL_SEEDS["en"]
    return STRUCTURAL_SEEDS.get(_base_language(language), "")


def build_initial_prompt(custom_vocab: Sequence[str] = (), language: Optional[str] = None) -> str:
    terms, used = [], 0
    for term in custom_vocab:
        term = " ".join(term.split())
        if term and used + len(term) + 2 <= MAX_TERMS_CHARS and len(terms) < MAX_TERMS:
            terms.append(term)
            used += len(term) + 2
    seed = get_structural_seed(language)
    listed = (", ".join(terms) + ".") if terms else ""
    return " ".join(part for part in (seed, listed) if part)


def is_hallucination(text: str, language: Optional[str] = "en") -> bool:
    cleaned = text.strip().lower().rstrip(".!,?")
    active = HALLUCINATIONS_BY_LANG.get(_base_language(language), frozenset()) | HALLUCINATIONS_BY_LANG["en"]
    return cleaned in active


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def is_prompt_echo(text: str, prompt: str) -> bool:
    """On near-silence Whisper sometimes writes out its own prompt. Such a transcript is not speech."""
    t, p = _norm(text), _norm(prompt)
    return bool(t) and bool(p) and t in p and len(t.split()) >= 2
