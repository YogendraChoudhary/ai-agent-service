# Rollout runbook: model upgrade of 2026-11

What changed, how to check it before a deploy, how to roll one piece back without a deploy, and what to watch afterwards.
Measurements behind every choice: `docs/MODEL_UPGRADE_BENCHMARK.md`.

## 1. Before deploying

1. `python scripts/preflight_models.py` must exit 0. It makes free lookups only: every model the app can call (all tiers, pins, design, image default and fallbacks) must exist, and each platform key (Anthropic, OpenAI, Gemini) must be accepted. A dead key or a withdrawn model is exactly what stayed invisible before (a dead Anthropic key behind silent fallback for weeks; `gpt-image-1` withdrawn). It does **not** see a depleted balance (that only fails when a call is generated).
2. Deployment settings (none are required, all have the measured defaults):

| Setting | Default | Meaning |
|---|---|---|
| `CLAUDE_ADVANCED_EFFORT` | `low` | Opus 5.5 thinking depth (low | medium | high | xhigh | max). `low` was as good as `medium`, faster and cheaper |
| `CLAUDE_BASIC_EFFORT` | empty | Sonnet 5.5; empty = model default (low changed nothing) |
| `OPENAI_ADVANCED_EFFORT` | `none` | gpt-6.1-terra; tool-using agents always run at `none` |
| `MODEL_PREFLIGHT_ON_START` | `true` | startup check in a background thread; logs `ROLLOUT_ALERT preflight ...` |
| `AGENT_BACKEND_SECRET` | unset | set it: it is also what lets a monitor read `/agent/health/providers` (header `X-Agent-Backend-Secret`) |
| `TEXT_PRICES_PER_MTOK` | unset | add `claude-opus-5-5` [4, 20], `claude-sonnet-5-5` [2, 10], `claude-haiku-5-5` [0.10, 0.50], `gpt-6.1-terra` [1, 5], `gpt-6.1-sol` [2, 10] (list prices of 2026-11; unset stays "unknown", never zero) |
| `IMAGE_CREDITS_BY_MODEL` | unset | the old `openai/gpt-image-1` key matches nothing now; add `openai/gpt-image-2.5-flare` etc. if you price per model (unset = flat 5 credits) |

3. Delete or update the stale copy of the old dead key in the parent folder's `.env` (`/Agent/.env`); the app reads `agent/.env`.
4. Confirm the new Anthropic key (`sk-ant-usr-...`, a different type from the earlier `sk-ant-api03-...`) is meant for server use.

## 2. Canary

Deploy the new build to one instance (or the share of traffic you choose), compare it with the old fleet for 30-60 minutes, then ramp. The model choices are in code defaults, so the instance split *is* the traffic split. Compare, per feature: the `alerts` list, failures by kind and fallbacks in `GET /agent/health/providers`, p50/p95 in `/agent/metrics` or `REQUEST_SUMMARY` logs, and cost per request (needs the prices above).

## 3. Rolling one change back without a deploy

Set the variable and restart. The request helpers follow the model ID (temperature, effort, thinking, token limits), so a rolled-back model gets the request it expects. `GET /agent/health/providers` lists `model_overrides` while any are active, and startup logs a warning.

| Change | Roll back with |
|---|---|
| Claude advanced (Opus 5.5) | `MODEL_ADVANCED_CLAUDE=claude-opus-5` |
| Claude basic (Sonnet 5.5) | `MODEL_BASIC_CLAUDE=claude-sonnet-5` |
| Claude light tier: autocomplete, classifier, voice (Haiku 5.5) | `MODEL_AUTOCOMPLETE_CLAUDE=claude-haiku-4-5` |
| OpenAI advanced (gpt-6.1-terra) | `MODEL_ADVANCED_OPENAI=gpt-5.2` |
| OpenAI design canvas (gpt-6.1-sol) | `MODEL_DESIGN_OPENAI=gpt-5.2` |
| OpenAI image default | `IMAGE_MODEL_OPENAI=gpt-image-2.6-flare` (`gpt-image-1` no longer exists; there is nothing to roll back to) |

Other models follow the same pattern: `MODEL_<ADVANCED|BASIC|AUTOCOMPLETE|DESIGN>_<OPENAI|CLAUDE|GROK|DEEPSEEK|GEMINI>`. Not env-switchable (a code edit): the blog/code pins to gpt-5.2 (`AGENT_MODEL_PINS` in `src/models.py`). The temperature fix needs no rollback; Claude models that still take temperature still get it.

## 4. What the health view tells you

`GET /agent/health/providers` (with `X-Agent-Backend-Secret`, or when `METRICS_ENDPOINT_ENABLED=true`): the last 15 minutes per provider/model: calls, ok, failures by kind, a short scrubbed error sample (never keys or content), fallbacks, the startup check, and `alerts`. `status` is `alert` (any critical), `warning` or `ok`. Each critical alert is also logged once an hour as `ROLLOUT_ALERT <meaning> provider=... model=...`; alarm on that string if you use log-based alerting. Counters are per process and in memory: with several instances, query each one or alarm on the log line.

| Alert | Severity | Means | First action |
|---|---|---|---|
| `key_rejected` | critical | platform key rejected or cannot spend (401/403/402); requests silently fall back to another provider | fix the key or balance; the fallback is hiding it |
| `model_missing` | critical | the model no longer exists | set the `MODEL_*` / `IMAGE_MODEL_*` override to a live model |
| `request_rejected` | critical | 3 or more 400s in the window, or over 2%: our request shape is wrong for this model (a parameter it no longer takes) | read `sample`; roll the model back (section 3) while fixing |
| `preflight` | critical | the startup check found a missing model or rejected key | as above |
| `truncated` | warning | over 5% of answers cut at max_tokens (20+ calls) | raise the cap or lower the effort |
| `refused` | warning | 3 or more refusals | look at the prompts; Haiku 5.5 has no server-side fallback |
| `empty` | warning | over 5% empty answers | |
| `high_failure_rate` | warning | over 20% of calls fail for no sharper reason (rate limits, 5xx) | check provider status |

A user's own (BYOK) key failing is counted but never alerts.

## 5. Expected numbers (from the measurements; alarm if a p50 doubles)

Claude: autocomplete median ~1.0-1.2 s; classifier ~1.1 s; quick edits ~1.8 s; long-document grammar ~11 s per document; code generation ~14 s median (was 61 s); design canvas ~11 s (was 25 s). OpenAI: enhance ~1.6 s; design canvas on Sol ~11 s; blog (gpt-5.2, pinned) ~41 s; code (gpt-5.2, pinned) ~13 s.

## 6. Known gaps

The preflight cannot see a depleted balance or per-key model access (a user's own key may lack a model). Truncation and refusals are counted for design and direct calls, not inside the streaming agno path (an empty answer there is counted as `empty`). Gemini, Grok and DeepSeek keys are only checked for Gemini.
