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
BASIC_MODEL_IDS = {
    "openai": "gpt-5-mini",
    "claude": "claude-sonnet-5",
    "grok": "grok-4.3",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.5-flash-lite",
}

# Inline autocomplete fires on almost every keystroke, and the user is charged a
# flat AUTOCOMPLETE_CREDITS_SCALED rate regardless of which model serves it — so
# the model's cost comes straight out of our margin. Always the cheapest usable
# model per provider, never the flagship. (This path previously ran on
# claude-opus-5 at 5x the price of claude-haiku-4-5, for no benefit.)
#
# Grok note: grok-4.20-0309-non-reasoning is ~40% cheaper per reply than
# grok-4.3 because it skips reasoning tokens, but it is a dated snapshot. Switch
# only if autocomplete latency or spend becomes a problem.
# OpenAI note: this path must stay on a NON-reasoning model. Unlike the chat
# path (where agno normalises parameters), get_autocomplete_suggestion() calls
# client.chat.completions.create(..., max_tokens=100) directly, and the GPT-5
# family rejects `max_tokens` outright ("Unsupported parameter ... use
# max_completion_tokens"). Moving this to gpt-5-mini returns an empty
# suggestion on every keystroke. gpt-4o-mini is the correct model here: cheap,
# non-reasoning, and still served by the API (it was retired from ChatGPT only).
# Revisit only alongside a change to that call site.
AUTOCOMPLETE_MODEL_IDS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5",
    "grok": "grok-4.3",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.5-flash-lite",
}

# deepseek-reasoner (R1) does not support function/tool calling, so any agent
# that needs tools has to fall back to deepseek-chat.
DEEPSEEK_TOOL_MODEL_ID = "deepseek-chat"


# ============================================
# IMAGE GENERATION
# ============================================

IMAGE_MODEL_IDS = {
    "openai": "gpt-image-1",
    "grok": "grok-imagine-image",
    # Gemini has no GA image model yet; this preview ID is the current one.
    # Re-check when Google promotes a stable replacement.
    "gemini": "gemini-3.1-flash-image-preview",
}


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
