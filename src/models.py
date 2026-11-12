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
BLOCKED_MODELS = frozenset({'claude-fable-5-1'})


def ensure_allowed_model(model_id: str) -> str:
    if model_id in BLOCKED_MODELS:
        raise ValueError(f'Model {model_id} is blocked')
    return model_id


# Prices are supplied by deployment after verification; unknown prices remain unknown, never zero.
TEXT_PRICES_PER_MTOK = {}

# Capable tier — the main editor AI actions (write, enhance, summarise, ...).
ADVANCED_MODEL_IDS = {
    # gpt-6.1-terra since 2026-11-12: runs with reasoning_effort "none" (settings.openai_advanced_effort); at the default effort its
    # p95 first token was 8.9 s. Rejects temperature and top_p unless effort is "none", and max_tokens always (see
    # openai_reasoning_family / openai_effort_params).
    "openai": "gpt-6.1-terra",
    # claude-opus-5-5 since 2026-11-11 (served; 1M context, 128K output). Thinking cannot be turned off on it, effort is set from
    # settings.claude_advanced_effort (claude_effort_params), and it takes no temperature (claude_accepts_temperature).
    "claude": "claude-opus-5-5",
    "grok": "grok-4.6",
    "deepseek": "deepseek-reasoner",
    "gemini": "gemini-3.8-flash",
}

# Agents that stay on a given model while the rest of the provider's advanced tier moves on. Chosen by the user on 2026-11-12 after
# a measured comparison (docs/MODEL_UPGRADE_BENCHMARK.md, section 14): gpt-6.1-terra wrote blog posts ~5% shorter and plainer (fewer
# tables and quotes) and code answers with fewer of the requested usage examples and comments than gpt-5.2, which the user prefers.
AGENT_MODEL_PINS = {
    ("openai", "blog_writer"): "gpt-5.2",
    ("openai", "code_writer"): "gpt-5.2",
    ("openai", "code_explainer"): "gpt-5.2",
}


def advanced_model_id(provider: str, agent_type: str = None):
    """The advanced-tier model for this provider, or the one this agent is pinned to."""
    return AGENT_MODEL_PINS.get((provider, agent_type)) or ADVANCED_MODEL_IDS.get(provider)


# Cheaper tier — used when an agent config sets use_advanced=False.
# Tasks like grammar, table, simplify, etc. need fast non-reasoning models
# with low TTFT (<1s) and deterministic temperature control.
BASIC_MODEL_IDS = {
    "openai": "gpt-4o-mini",
    # claude-sonnet-5-5 since 2026-11-12 (same price as Sonnet 5, cheaper cache reads). Takes no temperature; it thinks by default, but
    # not on short edits (measured: no thinking blocks at any effort), so the effort dial is left to settings.claude_basic_effort.
    "claude": "claude-sonnet-5-5",
    "grok": "grok-4.20-0309-non-reasoning",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.5-flash-lite",
}

# ============================================
# OUTPUT TOKEN LIMITS
# ============================================
# agno's Claude defaults to 8,192 output tokens and the other providers to their own limits, so a long
# answer was cut off silently. Callers that know how much text they expect pass max_tokens to get_model().
# That number is the VISIBLE output wanted; output_token_budget() turns it into the provider limit.
#
# OUTPUT_TOKEN_CEILING and REASONING_HEADROOM_TOKENS are conservative and NOT yet checked against the
# providers' live model cards (open item O7 of docs/rich_content_engine_plan.md): verify before raising them.
OUTPUT_TOKEN_CEILING = 16_384
REASONING_HEADROOM_TOKENS = 4_096


def reasoning_headroom(model_id: str) -> int:
    """
    Extra tokens for models whose output limit also counts hidden reasoning tokens (OpenAI gpt-5 / o-series,
    Grok 4 reasoning, Claude Opus 5.5 and Sonnet 5.5, which think by default and bill thinking as output). Without it a short,
    well-sized answer can be cut off before the first visible word.
    """
    mid = (model_id or "").lower()
    reasons = (mid.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4", "claude-opus-5-5", "claude-sonnet-5-5"))
               or (mid.startswith("grok-4") and "non-reasoning" not in mid))
    return REASONING_HEADROOM_TOKENS if reasons else 0


def output_token_budget(model_id: str, visible_tokens: int) -> int:
    """The provider output limit for `visible_tokens` of visible text from this model."""
    return min(int(visible_tokens), OUTPUT_TOKEN_CEILING) + reasoning_headroom(model_id)


# OpenAI models that take max_completion_tokens (not max_tokens) and reject a custom temperature / top_p: the gpt-5 and gpt-6
# families and the o-series. Probed live 2026-11-12: gpt-5.2 and every gpt-6.x model answer 400 to max_tokens; the gpt-6.x ones
# also reject temperature and top_p unless reasoning_effort is "none".
OPENAI_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
# Models that take reasoning_effort "none" (gpt-6.1-sol does not; gpt-5, gpt-5-mini and the o-series have no "none" either).
OPENAI_EFFORT_NONE_PREFIXES = ("gpt-6", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6")


def openai_reasoning_family(model_id: str) -> bool:
    return (model_id or "").lower().startswith(OPENAI_REASONING_PREFIXES)


def openai_effort_params(model_id: str, effort: str, needs_tools: bool = False) -> dict:
    """
    `reasoning_effort` for the OpenAI models that have a "none" level. Defaults matter: gpt-6.1-terra left at its default effort
    spent hidden reasoning tokens (8.9 s p95 to the first token, measured), and on the Chat Completions API its function tools are
    rejected at any effort but "none" (calls with tools therefore always get "none"). gpt-6.1-sol cannot do "none" and gets "low".
    """
    mid = (model_id or "").lower()
    if not mid.startswith(OPENAI_EFFORT_NONE_PREFIXES) or not effort:
        return {}
    if needs_tools:
        effort = "none"
    if effort == "none" and mid.startswith("gpt-6.1-sol"):
        effort = "low"
    return {"reasoning_effort": effort}


# Where a provider's design canvas runs on a different model than its general advanced tier. gpt-6.1-sol at effort low drafted a
# design in a median 9.8 s (6 runs); gpt-6.1-terra took 19 s and gpt-5.2 16 s (docs/MODEL_UPGRADE_BENCHMARK.md, section 4.5).
DESIGN_MODEL_IDS = {"openai": "gpt-6.1-sol"}
OPENAI_DESIGN_EFFORT = "low"


def design_model_id(provider: str, default=None):
    """The model the design canvas runs on for this provider: its own entry if it has one, else the advanced tier's."""
    return DESIGN_MODEL_IDS.get(provider) or ADVANCED_MODEL_IDS.get(provider, default)


# Claude models that still accept `temperature`. Every newer model rejects it with
# `400: temperature is deprecated for this model`, so a request that sends it never succeeds.
# Probed live on 2026-11-11 with temperature=0.3:
#   accepted: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6, claude-opus-4-5
#   rejected: claude-opus-4-7, claude-opus-4-8, claude-opus-5, claude-opus-5-5,
#             claude-sonnet-5, claude-sonnet-5-5, claude-haiku-5-5
# An allow-list, not a deny-list: a model ID that is not listed here (any new release) is sent no
# temperature, which is what every Claude model released since Opus 4.7 expects.
CLAUDE_TEMPERATURE_PREFIXES = (
    "claude-3", "claude-haiku-4", "claude-sonnet-4",
    "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6",
)


def claude_accepts_temperature(model_id: str) -> bool:
    """Whether this Claude model ID takes a `temperature` parameter (see CLAUDE_TEMPERATURE_PREFIXES)."""
    return (model_id or "").lower().startswith(CLAUDE_TEMPERATURE_PREFIXES)


# Claude models that take `output_config.effort` (GA on Opus 4.6+ / Sonnet 4.6+; errors on Haiku 4.5 and Sonnet 4.5).
CLAUDE_EFFORT_PREFIXES = ("claude-opus-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
                          "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable", "claude-haiku-5")


def claude_effort_params(model_id: str, effort: str) -> dict:
    """`output_config` for the thinking-depth dial on Claude models that have it; nothing for the ones that error on it."""
    if (model_id or "").lower().startswith(CLAUDE_EFFORT_PREFIXES) and effort:
        return {"effort": effort}
    return {}


def claude_light_request_params(model_id: str) -> dict:
    """
    Extra request fields for Claude's light tier (autocomplete, the code-intent classifier, the voice formatter).

    Haiku 5 and newer think by default, and thinking tokens come out of max_tokens: a 100-token autocomplete or a
    one-word classifier answer can be spent before any text. These calls want the answer, not reasoning, so thinking is
    turned off and effort set to the lowest level (Haiku 5.5 accepts `thinking: disabled` at effort low/medium/high).
    Haiku 4.5 and older reject `output_config.effort`, so they get nothing extra.
    """
    mid = (model_id or "").lower()
    if mid.startswith("claude-haiku-") and not mid.startswith(("claude-haiku-3", "claude-haiku-4")):
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    return {}


# Inline autocomplete fires on almost every keystroke, and the user is charged a
# flat AUTOCOMPLETE_CREDITS_SCALED rate regardless of which model serves it — so
# the model's cost comes straight out of our margin. Always the cheapest usable
# model per provider, never the flagship. (This path previously ran on
# claude-opus-5 at 5x the price of claude-haiku-4-5, for no benefit.)
# Claude moved from claude-haiku-4-5 to claude-haiku-5-5 on 2026-11-11 (served by /v1/models/claude-haiku-5-5; list price
# $0.10 / $0.50 per MTok vs $1 / $5). Its requests need claude_light_request_params() and no temperature.
AUTOCOMPLETE_MODEL_IDS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-5-5",
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
# Gemini default is gemini-3.1-flash-lite-image (2026-10-19 live benchmark, same prompts side by side):
# 3.5-4.7s vs 8.4-10.2s for gemini-2.5-flash-image, image quality equal or better, $0.0336/image vs $0.039.
# Google marks gemini-2.5-flash-image deprecated and recommends the lite model. Lite only does 1K, so
# 2K/4K requests are routed to GEMINI_HIRES_MODEL (see gemini_model_for_resolution).
IMAGE_MODEL_IDS = {
    # gpt-image-1 no longer exists on the API ("The model 'gpt-image-1' does not exist", 2026-11-11). gpt-image-2.5-flare: ~8-12 s, ~$0.013
    # per medium 1024x1024 image, text renders correctly (docs/MODEL_UPGRADE_BENCHMARK.md, section 4.6).
    "openai": os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-2.5-flare"),
    "grok": os.getenv("IMAGE_MODEL_GROK", "grok-imagine-image-2.0"),
    "gemini": os.getenv("IMAGE_MODEL_GEMINI", "gemini-3.1-flash-lite-image"),
}

IMAGE_MODEL_FALLBACKS = {
    "gemini": ["gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview"],
    "grok": ["grok-imagine-image"],
    "openai": ["gpt-image-2.6-flare", "gpt-image-1.5"],
}

GEMINI_HIRES_MODEL = os.getenv("IMAGE_MODEL_GEMINI_HIRES", "gemini-3.1-flash-image")

# Sizes each Gemini image model really delivers (probed live 2026-10-19). Unknown models are not listed
# and are passed through unchanged. gemini-2.5-flash-image accepts "2K"/"4K" but returns 1024x1024, so it is
# listed as 1K only; the lite model rejects anything but 1K with a 400.
GEMINI_IMAGE_SIZES = {
    "gemini-3.1-flash-lite-image": ("1K",),
    "gemini-2.5-flash-image": ("1K",),
    "gemini-3.1-flash-image": ("512", "1K", "2K", "4K"),
}


def gemini_model_for_resolution(model_override, resolution: str) -> str:
    """
    The Gemini model that will draw this request. With no model named by the user, a 2K/4K request goes to
    GEMINI_HIRES_MODEL (the default lite model only does 1K). A model the user named is kept as is.
    """
    model = model_override or IMAGE_MODEL_IDS["gemini"]
    sizes = GEMINI_IMAGE_SIZES.get(model)
    if model_override or sizes is None or resolution in sizes:
        return model
    hires_sizes = GEMINI_IMAGE_SIZES.get(GEMINI_HIRES_MODEL)
    return GEMINI_HIRES_MODEL if hires_sizes is None or resolution in hires_sizes else model


def gemini_effective_resolution(model: str, resolution: str) -> str:
    """The size to send for this model: the requested one if it is supported, else 1K (never a 400)."""
    sizes = GEMINI_IMAGE_SIZES.get(model)
    return resolution if sizes is None or resolution in sizes else "1K"


def openai_image_supports_input_fidelity(model_id: str) -> bool:
    """Only gpt-image-1.5 takes `input_fidelity` (probed live: gpt-image-1-mini rejects "high"; gpt-image-2 and newer reject the parameter)."""
    return (model_id or "").lower().startswith("gpt-image-1.5")


def get_image_model_id(provider: str) -> str:
    """Get the active image generation model ID for a provider."""
    provider_clean = (provider or "").lower().strip()
    return IMAGE_MODEL_IDS.get(provider_clean, os.getenv("IMAGE_MODEL_DEFAULT", "gemini-3.1-flash-lite-image"))


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


# ============================================
# ENVIRONMENT OVERRIDES (instant rollback without a deploy)
# ============================================
# MODEL_<TIER>_<PROVIDER>=<model id>, with TIER one of ADVANCED, BASIC, AUTOCOMPLETE, DESIGN and PROVIDER one of OPENAI, CLAUDE, GROK,
# DEEPSEEK, GEMINI. Example: MODEL_ADVANCED_OPENAI=gpt-5.2 puts OpenAI's advanced tier back on gpt-5.2 (restart the service). The request
# helpers above (temperature, effort, thinking, token limits) follow the model ID, so a rollback sends the request that model expects.
# Images already had IMAGE_MODEL_<PROVIDER>. docs/ROLLOUT_RUNBOOK.md lists the rollback value for every change of 2026-11.
MODEL_TIERS = {"ADVANCED": ADVANCED_MODEL_IDS, "BASIC": BASIC_MODEL_IDS, "AUTOCOMPLETE": AUTOCOMPLETE_MODEL_IDS, "DESIGN": DESIGN_MODEL_IDS}
MODEL_PROVIDERS = ("openai", "claude", "grok", "deepseek", "gemini")


def apply_model_env_overrides(env=None) -> dict:
    """Apply MODEL_<TIER>_<PROVIDER> overrides in place; returns what was changed, e.g. {"ADVANCED/openai": "gpt-5.2"}."""
    env = os.environ if env is None else env
    changed = {}
    for tier, table in MODEL_TIERS.items():
        for provider in MODEL_PROVIDERS:
            value = (env.get(f"MODEL_{tier}_{provider.upper()}") or "").strip()
            if value and table.get(provider) != value:
                table[provider] = value
                changed[f"{tier}/{provider}"] = value
    return changed


MODEL_ENV_OVERRIDES = apply_model_env_overrides()

