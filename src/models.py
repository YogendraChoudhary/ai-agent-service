"""
Single source of truth for every AI model ID the agent uses.

Model IDs rot. Providers retire them on their own schedule, and requests start
failing with no change on our side — in Sept 2026 both `grok-3` and
`gemini-2.0-flash-lite` had already been withdrawn while still pinned in code,
silently breaking Grok chat and Gemini autocomplete in production.

Keeping every ID in one module means the next deprecation is a one-line edit
rather than a hunt across get_model(), the autocomplete path, the image tools
and the .env files.

House rules:
  * Never pin a dated snapshot (`claude-sonnet-4-5-20250929`,
    `grok-4.20-0309-...`). Dated IDs are the first to be retired and they hide
    how stale a pin has become. Use the rolling ID.
  * Never point a high-frequency path at a flagship model — see
    AUTOCOMPLETE_MODEL_IDS.
  * When bumping anything here, re-check it against the provider's live model
    list, not its docs. The two disagree.

All IDs below were verified against each provider's live /models endpoint on
2026-09-08.
"""

# ============================================
# TEXT CHAT
# ============================================

# Capable tier — the main editor AI actions (write, enhance, summarise, ...).
ADVANCED_MODEL_IDS = {
    "openai": "gpt-5.2",
    "claude": "claude-opus-5",
    "grok": "grok-4.6",
    "deepseek": "deepseek-reasoner",
    "gemini": "gemini-3.8-flash",
}

# Cheaper tier — used when an agent config sets use_advanced=False.
# Tasks like grammar, table, simplify, etc. need fast non-reasoning models
# with low TTFT (<1s) and deterministic temperature control.
BASIC_MODEL_IDS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-sonnet-5",
    "grok": "grok-4.20-0309-non-reasoning",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.5-flash-lite",
}

# Inline autocomplete fires on almost every keystroke, and the user is charged a
# flat AUTOCOMPLETE_CREDITS_SCALED rate regardless of which model serves it — so
# the model's cost comes straight out of our margin. Always the cheapest usable
# model per provider, never the flagship. (This path previously ran on
# claude-opus-5 at 5x the price of claude-haiku-4-5, for no benefit.)
AUTOCOMPLETE_MODEL_IDS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5",
    "grok": "grok-4.20-0309-non-reasoning",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.5-flash-lite",
}

# deepseek-reasoner (R1) does not support function/tool calling, so any agent
# that needs tools has to fall back to deepseek-chat.
DEEPSEEK_TOOL_MODEL_ID = "deepseek-chat"


import os

# ============================================
# IMAGE GENERATION
# ============================================

# Environment-overridable model IDs so deprecation/rename does not require deployment.
# Verified on 2026-10-16 via image_smoke_test.py.
# Gemini default is 2.5: in the 2026-10-19 live benchmark it had a ~9.5s median (7/7 runs under
# 10s) against ~12s for gemini-3.1-flash-image (0/7 under 10s).
IMAGE_MODEL_IDS = {
    "openai": os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-1"),
    "grok": os.getenv("IMAGE_MODEL_GROK", "grok-imagine-image-2.0"),
    "gemini": os.getenv("IMAGE_MODEL_GEMINI", "gemini-2.5-flash-image"),
}

IMAGE_MODEL_FALLBACKS = {
    "gemini": ["gemini-3.1-flash-image", "gemini-3.1-flash-image-preview"],
    "grok": ["grok-imagine-image"],
    "openai": ["gpt-image-1"],
}


def get_image_model_id(provider: str) -> str:
    """Get the active image generation model ID for a provider."""
    provider_clean = (provider or "").lower().strip()
    return IMAGE_MODEL_IDS.get(provider_clean, os.getenv("IMAGE_MODEL_DEFAULT", "gemini-2.5-flash-image"))


def get_image_model_fallbacks(provider: str) -> list[str]:
    """Get fallback model IDs in order of preference."""
    provider_clean = (provider or "").lower().strip()
    return list(IMAGE_MODEL_FALLBACKS.get(provider_clean, []))


# ============================================
# VISION (IMAGE INPUT)
# ============================================

# Providers whose ADVANCED_MODEL_IDS entry accepts image input, for the
# reference-image feature in the AI chat.
#
# DeepSeek is excluded: both deepseek-reasoner and deepseek-chat are text-only.
# Grok is included as of the grok-4.x upgrade — grok-3 could not read images,
# which is why it was previously gated out; grok-4.6 and grok-4.3 both can.
VISION_CAPABLE_PROVIDERS = {"openai", "claude", "gemini", "grok"}


def supports_vision(model_provider: str) -> bool:
    """Whether this provider's model can accept image input."""
    return (model_provider or "").lower() in VISION_CAPABLE_PROVIDERS
