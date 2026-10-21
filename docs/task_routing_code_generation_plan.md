# Task Routing, Provider Fallback & Code Generation — Implementation Plan

*Status: implemented locally behind disabled rollout flags; verification notes below. Date: 2026-10-21.*

Implementation notes:
- API contract, task routing, code intent/classifier, code agents, validation/repair, provider fallback,
  actual-provider billing, usage traces, and evaluation tooling are implemented. See `API_DOCUMENTATION.md`.
- Claude IDs remain unchanged by explicit user decision after the live `/models` check returned 401.
  The Fable guard is implemented. The new-model latency benchmark awaits verified IDs and a working key.
- All new rollout flags remain off, including streaming repair until frontend `content_replace` support.
- Price estimates require verified `TEXT_PRICES_PER_MTOK` configuration; unavailable prices are null.
- The intent corpus has 195 cases. The paid quality runner has 30 prompts per language group and saves
  results for manual side-by-side review. Production shadow observation and full paid quality evaluation
  remain rollout work, not completed by unit tests.
- Follow-up fixes enforce own-key lookup before classification, move ambiguous words to classification,
  return 422 for invalid non-streaming code, disable nested SDK retries in the managed retry loop,
  distinguish JSX text from code and reject incomplete HTML, and evaluate repaired replacements.
- Further regressions cover credit-rejected output (including repairs), shared search limits across
  provider attempts, and prose requests such as code reviews, specifications, and explanations.
- Concurrent classifier/answer usage updates are atomic so neither call's tokens or estimated cost is lost.

## 1. Goal

Give users output good enough that they never leave Eddyter to use another AI tool, starting with a new
**code generation** feature. Behind it, add:

- a **task → model routing table** used only when the user has not picked a model,
- a **fallback tree** so a 429 or outage moves to the next model instead of failing.

Everything that already works stays as is: image generation flow, image defaults, quick-edit tiers, email.

## 2. Locked decisions

| # | Decision |
|---|---|
| D1 | Frontend always sends `model` (default `gemini`) **plus** a new `model_source: "default" \| "user"`. |
| D2 | `model_source: "user"` → the user's model always wins, for every task. Fallbacks only on failure. |
| D3 | `model_source: "default"` + code task → Claude Opus 5.5. Any other task → Gemini (current behaviour, current tier logic). |
| D4 | Missing `model_source` is treated as `"user"`, so nothing changes until the frontend ships the flag. |
| D5 | Image generation: no change. Gemini default; higher quality only when the user asks; a named image model is honoured. |
| D6 | Fallback tree in config. Code: Claude Opus 5.5 → GPT-5.2 → Gemini 3.8 Flash. Other: chosen provider → GPT-5.2 → Claude Opus 5.5. Same tier as the original request. |
| D7 | Fallback on 429 / 5xx / overloaded / timeout only. One same-model retry on 429 (short wait) before switching. Never on auth or bad-request errors. Only before the first token is streamed. Logged, never shown to the user. |
| D8 | BYOK users: candidates filtered to providers they have keys for. Platform keys never touch BYOK text. |
| D9 | Code entry points: new toolbar type `generate_code` **and** regex detection inside `general`. Mixed requests (blog post with a code sample) stay with the writing agent. |
| D10 | Intent: regex first, start-anchored, fail-safe. A light LLM classifier (Haiku 5.5 on platform key; BYOK → their cheapest model on their key) runs only when regex is unsure. Output `code` \| `other`; ~1.5s timeout; any failure → `other`. |
| D11 | Code output: fenced Markdown blocks with a language tag, **code only**, no prose. Multiple files → one fence each, file name as a first-line comment. |
| D12 | New `explain_code` type for the "Explain" button. |
| D13 | Code agent has web search + page reading, with the same per-request budgets as the general agent. |
| D14 | Validation: Python (`ast`), JSON (`json`), JS/TS/HTML/CSS structural balance check. One repair pass on failure. Node.js-based JS/TS check deferred until logs justify it. |
| D15 | `/agent/process`: accepts `generate_code` / `explain_code` but never switches provider (backend owns key + billing). Default routing + cross-provider fallback only on `/agent/stream`. |
| D16 | Credits: current per-character text price, kept configurable. Log model, tokens, estimated cost per request so pricing can be decided with data. |
| D17 | Fable is blocked in code (`BLOCKED_MODELS`), whatever config says. |

## 3. Current state (what we build on)

- `api.py` — `model` is required and validated against `SUPPORTED_MODELS`; the API key is resolved for that
  one provider in `async_validate_and_get_api_key()` **before** the agent runs.
- `src/text_processing_agent.py` — `AGENT_FOR_REQUEST_TYPE` maps `type` → agent; `AGENT_CONFIGS` sets
  tier/tools/temperature; `get_model()` maps (provider, tier) → model ID; `process_user_query_stream()` runs
  one agent with one provider and has no retry.
- `src/request_plan.py` — regex planner for `general` (images, web, tier). No LLM in routing.
- `src/intent.py` — start-anchored, fail-safe regex image intent. The template for code intent.
- `src/models.py` — single source of model IDs. Claude IDs are behind (`claude-opus-5`, `claude-sonnet-5`,
  `claude-haiku-4-5`).
- `src/email_template/service.py` — validate → one repair pass pattern to reuse.
- `src/observability.py` — per-request trace (route, plan, chat model, images, latency).
- Billing (`billed_provider_model`) bills text against the **requested** provider, model `None`.

## 4. Request lifecycle after the change (`/agent/stream`)

```
request (type, query, model, model_source)
  │
  ├─ 1. task = task_for_request(type, query)            # src/task_router.py
  │      generate_code / explain_code → "code"
  │      general → image intent (existing, unchanged) → code intent (regex → classifier if unsure) → "content"
  │      other typed actions → "content"
  │
  ├─ 2. candidates = resolve_route(task, model, model_source, available_providers, tier)
  │      user pick first, or task default; then fallback tree; BYOK-filtered; Fable removed
  │
  ├─ 3. keys resolved lazily per candidate (existing resolve_api_key_from_user_keys)
  │
  ├─ 4. run candidate #1 → on retryable error before first token → retry once (429) → next candidate
  │
  ├─ 5. stream chunks; code task → validate at end → one repair if broken (see §6.5)
  │
  └─ 6. billing + trace use the provider/model that actually answered
```

## 5. Phases

Each phase ships behind its own flag (all default off except where noted) and leaves current behaviour
unchanged when its flag is off.

### Phase 0 — Model IDs + Fable guard  *(small, ship first)*

Files: `src/models.py`, tests.

- Update to `claude-opus-5-5` (advanced), `claude-sonnet-5-5` (basic), `claude-haiku-5-5` (autocomplete).
  Verify each against Anthropic's live `/models` list first (house rule in `models.py`).
- Add `BLOCKED_MODELS = {"claude-fable-5-1"}` and a guard in `get_model()` that raises if a resolved ID is
  blocked. Unit test proves no tier table or env override can reach it.
- Re-run the latency benchmark for Claude basic tier (Sonnet 5 → 5.5) to confirm quick edits stay < 1s TTFT.

### Phase 1 — API contract

Files: `api.py`, `docs/API_DOCUMENTATION.md`.

- `StreamRequest`, `ProcessRequest`, `EmailGenerateRequest`: add
  `model_source: Literal["default", "user"] = "user"` (D4).
- Add `generate_code` and `explain_code` to `VALID_AGENT_TYPES` and `AGENT_FOR_REQUEST_TYPE`.
- `TYPE_PREFIX_MAP` entries for the two new types (the `/agent/process` prefix map must cover all types).
- Document the contract for the frontend (§7) and backend (§8).

### Phase 2 — Routing table + resolver

New file: `src/task_router.py`. Config: `config.py`.

```python
TASK_DEFAULTS   = {"code": "claude", "content": None}   # None = keep the request's model (gemini today)
FALLBACK_TREES  = {
    "code":    ["claude", "openai", "gemini"],
    "content": ["<requested>", "openai", "claude"],
}
```

- Candidates are **(provider, tier)**, not raw IDs, so `models.py` stays the only place IDs live.
  Tier = the tier the request would already get (agent config / request plan).
- `resolve_route(task, model, model_source, available_providers, tier) -> list[Candidate]`:
  1. first = user's model if `model_source == "user"`, else `TASK_DEFAULTS[task]` or the request's model;
  2. append the task's tree, de-duplicated;
  3. drop providers with no usable key (BYOK: user keys only; platform: user keys or `DEFAULT_API_KEYS`);
  4. drop DeepSeek when the request has reference images (no vision) — existing `supports_vision()`;
  5. drop anything resolving into `BLOCKED_MODELS`.
- Env overrides: `TASK_ROUTE_DEFAULTS` and `TASK_FALLBACK_TREES` (JSON), validated at startup; an invalid
  value is logged and ignored, never crashes the app.
- Flag: `TASK_ROUTING_ENABLED` (off → candidates = [requested model], i.e. today).

`api.py` change: key resolution stops being "one key up front". `async_validate_and_get_api_key()` still
validates the licence and returns `user_keys`; a small `KeyResolver` (wrapping
`resolve_api_key_from_user_keys`) is passed into the stream function so each candidate's key is looked up
only when that candidate runs. The upfront "No API key configured" 403 now fires only when **no** candidate
has a key.

### Phase 3 — Code intent detection

New files: `src/code_intent.py`, `src/intent_classifier.py`, `tests/fixtures/code_intent_eval_cases.py`.

**Regex (`detect_code_intent(text) -> {"kind": "code"|"other"|"unsure", "rule": str}`)**, same rules as
`src/intent.py`:

- **Strong → `code`:** request starts (after polite filler) with a build verb + code noun
  ("write a function…", "create a Python script…", "generate a React component…", "SQL query to…",
  "regex for…", "bash command to…", "fix this code", "convert this to TypeScript"), or the request contains a
  fenced block plus an action verb ("fix", "refactor", "optimize", "add types to").
- **Weak → `unsure`:** code vocabulary or a language name appears, but not start-anchored
  ("I need something in Python that…", "can you help with my API endpoint").
- **None → `other`:** no code vocabulary. The classifier is **not** called — no added latency for the
  vast majority of requests.
- **Mixed-content guard → `other`:** long-form nouns lead the request ("blog post / article / tutorial
  … with a code example") — D9.
- Only runs for `type == "general"`, after the existing image intent (image wins if it matched).

**Classifier (`classify_intent(text) -> "code" | "other"`)** — only on `unsure`:

- Model: `claude-haiku-5-5` on the platform key; BYOK → `AUTOCOMPLETE_MODEL_IDS[their provider]` on their key.
- `max_tokens` ~5, temperature 0, prompt asks for exactly one word; anything else → `other`.
- Timeout `CODE_CLASSIFIER_TIMEOUT_SECONDS` (1.5); timeout or error → `other` (safe: the general agent can
  still write code, just via the content route).
- Input truncated to the first ~1,000 characters.
- Flags: `CODE_INTENT_REGEX_ENABLED`, `CODE_INTENT_CLASSIFIER_ENABLED`, `CODE_INTENT_SHADOW_MODE`
  (log the decision, don't act on it — run this first in production for a week).

### Phase 4 — Code agents

File: `src/text_processing_agent.py`.

- `AGENT_CONFIGS["code_writer"]`: advanced tier, temperature 0.2, web tools only (no image tool).
  Instructions cover: detect language from the request (default to the language of any pasted code);
  output only fenced blocks with a language tag; complete, runnable code, no placeholders like
  `// ... rest of code`; consistent 4-space (Python) / 2-space (JS/TS/HTML/CSS) indentation, never tabs
  mixed with spaces; multiple files → one fence each with the file name as a first-line comment;
  no prose before or after.
- `AGENT_CONFIGS["code_explainer"]`: advanced tier, temperature 0.3, no tools, normal Markdown prose with
  snippets allowed. Routed with the `code` task tree.
- `create_agent()` today treats "has a plan" as "is smart_global" (it swaps in smart_global instructions).
  Refactor so the tool grant comes from a plan object but instructions come from the agent's own config;
  the code agent gets `RequestPlan(use_web=True, image_count=0, advanced_model=True)`. This also guarantees
  the plan-less "full tool set" branch (which adds image tools) is never used for code.
- Keep the code agent's system prompt byte-stable (no datetime) for prompt caching.

### Phase 5 — Code validation + repair

New file: `src/code_validate.py`.

- `extract_blocks(text) -> list[(lang, code)]`; missing fence or tag is itself a validation error.
- Checks: Python `ast.parse` (catches `IndentationError` / `TabError`); JSON `json.loads`;
  JS/TS/HTML/CSS: balanced `() [] {}`, quotes, template literals, comments; HTML open/close tag balance
  (void elements excluded). Unknown languages: fence check only.
- **Streaming conflict (needs frontend sign-off):** the answer streams before we can validate it.
  Plan: stream as normal; validate when the stream ends; if broken, run one non-streaming repair
  (original reply + error list, reusing the email repair pattern) and send a new SSE event
  `{"type": "content_replace", "content": "<full corrected answer>"}` before `done`.
  Until the frontend handles `content_replace`, `CODE_REPAIR_STREAM_EVENT=false` → we validate and log only.
- `/agent/process` (non-streaming) can repair in place with no new event.
- Flags: `CODE_VALIDATION_ENABLED`, `CODE_REPAIR_ENABLED`, `CODE_REPAIR_STREAM_EVENT`.

### Phase 6 — Provider fallback

Files: `src/text_processing_agent.py`, `api.py`, new `src/provider_errors.py`.

- `is_retryable(err)`: 429, 5xx, "overloaded", timeouts, connection resets → yes; 401/403/400/safety → no.
  Must cover each SDK's error shape **and** agno's `RunEvent.run_error` content string.
- In `process_user_query_stream()`, wrap create-agent + run in a loop over candidates:
  - error before the first chunk is yielded and retryable → (429: one same-model retry, honouring
    `Retry-After` up to 2s) → next candidate;
  - error after the first chunk → raise as today (never mix two models' output);
  - all candidates fail → raise the last error (existing error SSE).
- Yield a sentinel `{"__model_used__": (provider, model_id)}` so `api.py` bills and traces the model that
  actually answered (`billed_provider_model` gets the real provider + model for text too).
- `/agent/process`: candidate list is only the requested provider (D15); the same-model 429 retry still
  applies.
- Flag: `PROVIDER_FALLBACK_ENABLED`.

### Phase 7 — Observability + cost

Files: `src/observability.py`, `src/models.py`.

- New trace fields: `task`, `model_source`, `route_reason`, `intent_rule`, `classifier_used`,
  `classifier_ms`, `classifier_result`, `candidates`, `model_used`, `fallbacks` (list of
  `{from, to, error}`), `validation` (`ok` / `repaired` / `failed` + error kinds), `tokens_in`, `tokens_out`,
  `est_cost_usd`.
- `TEXT_PRICES_PER_MTOK` table in `models.py` (env-overridable) for the cost estimate.
- `latency_stats()` grouped by task and model_used, so code vs content and Opus vs others are comparable.

### Phase 8 — Tests + evaluation

- Unit tests, following the existing `tests/` layout:
  `tests/routing/test_task_router.py` (defaults, user pick wins, BYOK filtering, vision filter, Fable blocked, bad env JSON),
  `tests/code/test_code_intent.py`, `tests/routing/test_intent_classifier.py` (mocked; timeout/garbage → `other`),
  `tests/code/test_code_validate.py` (indentation errors, unbalanced braces, missing fence, multi-file),
  `tests/routing/test_provider_fallback.py` (429 → retry → next; error after first token is not retried; non-retryable
  stops), `tests/routing/test_type_routing.py` updated for the new types.
- `tests/fixtures/code_intent_eval_cases.py`: ≥150 labelled prompts (code / not code / mixed / tricky mentions), same
  shape as `tests/fixtures/image_intent_eval_cases.py`. Target: 0 false positives on the "not code" set.
- Quality eval: 30–50 real prompts per language (Python, JS/TS, HTML/CSS, SQL), run through Opus 5.5,
  GPT-5.2, Gemini 3.8 Flash; record validation pass rate, latency, cost; manual side-by-side review.
- Benchmark: TTFT for `general` must not regress (classifier never runs on requests without code cues).

## 6. Rollout order

1. Phase 0 (model IDs, Fable guard) — independent, ship now.
2. Phases 1 + 2 with `TASK_ROUTING_ENABLED=false` — contract live, behaviour unchanged.
3. Phase 4 + 5 — `generate_code` / `explain_code` toolbar actions usable (user's model or Claude default).
4. Phase 6 — fallback on, watch `fallbacks` in logs.
5. Phase 3 in shadow mode for a week → review misroutes → turn on regex → then classifier.
6. Phase 7 data → credits pricing decision for code.

## 7. Frontend handoff

- Send `model_source: "user"` once the user changes the dropdown, otherwise `"default"`.
- New request types: `generate_code` (toolbar "Generate code"), `explain_code` ("Explain" button; send the
  code, optionally with the user's question, in `query`).
- Code answers arrive as normal `content` chunks containing fenced Markdown (` ```python … ``` `).
  Render fences as code blocks.
- New optional SSE event `content_replace` (only after a repair): replace the streamed answer with
  `content`. Needs frontend support before `CODE_REPAIR_STREAM_EVENT` is turned on.

## 8. Backend handoff (`/agent/process`)

- Accepts `generate_code` / `explain_code` and `model_source`, but always uses the provider whose key the
  backend sent. No cross-provider fallback here.
- Optional later: if the backend wants defaults/fallback on this endpoint too, it must either send keys for
  several providers or let the agent resolve them, and read the `model_used` the agent returns for billing.

## 9. Open items

| Item | Owner | Blocks |
|---|---|---|
| Credits price for code (Opus 5.5 costs more per request) | Product | Nothing — current price kept, data from Phase 7 |
| Confirm no customer contract limits which provider sees their text | Business | Phase 6 rollout to those customers |
| Frontend support for `content_replace` | Frontend | Turning on `CODE_REPAIR_STREAM_EVENT` |
| Node.js on the server for real JS/TS checks | Ops | Nothing — revisit after Phase 5 logs |
| Live check of new Claude IDs | Dev | Phase 0 |
