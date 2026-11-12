# Model upgrade review: verified parameters, before/after latency and quality

*Run date: 2026-11-11. Environment: Python 3.12, macOS, live APIs. OpenAI and Claude both measured live. The Anthropic key in `.env` was dead (401) for the first part of the session; after it was replaced, Claude was run (§4.7-4.10, §5).*

This checks the pasted "Model Upgrade Plan" against the real APIs and against our code. Nothing in the plan was taken on trust: every OpenAI parameter claim below was sent to the API and the answer recorded, and the Claude behaviours we depend on were run live. This first part changed no production code. The changes made afterwards are §10-§16, and the live check run before committing is §17. The measurement scripts and raw result files named below are **not in the repository**: the scripts stay on the machine that ran them and the results in `local_run/model_upgrade/` (gitignored). [MODEL_UPGRADE_EVAL_SCRIPTS.md](MODEL_UPGRADE_EVAL_SCRIPTS.md) says what each script did, what it found and how to rebuild it.

Decision columns are left for you. The numbers are samples, not guarantees (caveats in §9).

---

## 1. Headlines

1. **`gpt-image-1` no longer exists on our OpenAI account** (`400: The model 'gpt-image-1' does not exist`). It is the default OpenAI image model (`src/models.py:120`) *and* the only fallback (`src/models.py:128`), so OpenAI image generation (used when a user names OpenAI or brings an OpenAI-only key; Gemini is the default image provider) is already failing today, independent of any upgrade.
2. **A model-ID-only swap breaks our OpenAI text code.** `gpt-6.1-terra`, `gpt-6.1-sol` and `gpt-6-luna` all fail through `get_model()`, autocomplete, the intent classifier and the design canvas (400s: `temperature` not default, `max_tokens` not supported). Autocomplete and the classifier swallow the error, so they would fail *silently*. `gpt-5.6-luna` happens to work as-is only because its ID still starts with `gpt-5`.
3. **Effort must be set explicitly on every GPT-6.x call.** Left at the default, `gpt-6.1-terra` has a 8.9 s p95 time-to-first-token and 16.9 s p95 total; with `reasoning_effort="none"` it is 1.2 s and 10.0 s. An unset-effort `gpt-6-luna` with a 100-token cap returned nothing usable on 7 of 30 autocomplete calls and 60 of 60 classifier calls with a 5-token cap.
4. **gpt-6.1-terra is faster, better-rated and cheaper than gpt-5.2 for writing** (effort=none: median total 2.9 s vs 5.3 s, judge 8.0 vs 6.9, $0.0013 vs $0.0043 per call). Its first token is ~0.2 s *slower*, and "make longer" produced much shorter expansions (72-86 words vs 266).
5. **gpt-6-luna is not faster than gpt-4o-mini; it is ~0.2-0.26 s slower** on every short call, about equal in quality, ~30% cheaper. The decision is cost vs. keystroke latency.
6. **Sampling params come back only with `reasoning_effort="none"`.** Confirmed live: `temperature=0.3` and `top_p=0.9` are accepted on Terra/Luna/GPT-6 Sol at effort `none` and rejected (even `temperature=0`) otherwise. GPT-6.1 Sol has no `none`, so it never takes them.
7. **For design canvas, `gpt-6.1-sol` effort=low beat everything** (median 9.8 s vs 15.9 s today, no repair rounds, 33% fewer tokens). Only 6 runs per config.
8. **Claude: `temperature` is already rejected on the models we run today.** Confirmed live: `claude-opus-5`, `claude-sonnet-5` and `claude-haiku-5-5` return `400: \`temperature\` is deprecated for this model.` on every call (10/10, 16/16, 60/60 in these runs). agno sends `temperature` on every Claude call, so the Claude text path cannot work as configured; it was invisible because the key was dead and provider fallback answers with OpenAI. `claude-haiku-4-5` still accepts it.
9. **Opus 5.5 is the largest measured win in the review.** Design canvas, 6 runs per config: median **25.0 s on Opus 5 -> 10.7 s (effort low) / 12.4 s (medium)**, no repair rounds (Opus 5: 2 of 6), and **4.8 s with fast mode** (2x price). Writing: judged better than Opus 5 in every config (8.8-9.0 vs 7.2-7.7), cost per call -7% (low) to +26% (high) because it thinks more.
10. **Haiku 5.5 as the classifier fixes a real weakness; as autocomplete it is ~0.2 s slower.** Today's Haiku 4.5 classifier often answers `code` and then starts writing code (12 of 60 replies), which our exact-match check discards: strict accuracy 0.38 vs 0.95 for Haiku 5.5 with `thinking:{type:"disabled"}` + effort low. Autocomplete: 0.83 -> 1.03 s median, quality about equal, -85% cost.
11. **The guide's feared Haiku 5.5 pitfall did not occur here:** with default thinking and `max_tokens` of 5 or 100, there were no thinking blocks and no `max_tokens` stops in 80 calls. Still a risk on harder prompts; the fix (thinking off + effort low) cost nothing.

---

## 2. The pasted doc, claim by claim

**OpenAI** (all rows "live" = sent to the API today; probe output in `local_run/model_upgrade/probe_openai.json`, `probe_images.json`)

| Claim in the doc | Verdict | Evidence |
|---|---|---|
| `gpt-6.1-terra`, `gpt-6-luna`, `gpt-5.6-luna`, `gpt-6.1-sol` exist | **Verified (live)** | `/v1/models`; also `gpt-6-sol`, `gpt-6-astra`, `gpt-5.6-terra/-sol`, `gpt-5.5` |
| Terra tool calling needs the Responses API | **Partly wrong** | Chat Completions tools work if `reasoning_effort="none"`; at default effort: *"Function tools with reasoning_effort are not supported ... use /v1/responses or set reasoning_effort to 'none'"*. Both routes tested through agno |
| Effort levels `none, low, medium, high, xhigh, max` | **Partly wrong** | Chat Completions: `none/low/medium/high/xhigh` only; **`max` works only on the Responses API**; `minimal` rejected everywhere. gpt-5.2 has no `max` either |
| GPT-6.1 Sol has no `none`; Terra, GPT-6 Sol, Luna do | **Verified (live)** | `gpt-6.1-sol`: *"Supported values are: low, medium, high, xhigh"* |
| `temperature`/`top_p` work again only at effort `none` | **Confirmed** (doc said "reported, unconfirmed") | OK at `none` on terra, luna, 5.6-luna, gpt-6-sol (Chat and Responses). Rejected otherwise: *"Only the default (1) value is supported"*, `top_p` *"not supported"* |
| Use `max_output_tokens` / `max_completion_tokens`, not `max_tokens` | **Verified (live)** | `max_tokens` is a 400 on gpt-5.2 and every 5.6/6.x model; still fine on gpt-4o-mini |
| `reasoning.mode: "pro"`, `reasoning.context` | **Verified (live)** | Accepted on terra/sol/luna (Responses). gpt-5.2: `mode` rejected; `context` accepts only `auto`/`current_turn` |
| `prompt_cache_options.ttl` / `mode:"explicit"`, replaces `prompt_cache_retention` | **Partly wrong** | The option exists only on the new models (gpt-5.2 / 4o-mini: *"not supported on this model"*); `ttl` accepts only **`30m`** (`1h` rejected). **`prompt_cache_retention="24h"` is still accepted everywhere**, not replaced |
| `text.verbosity` low/medium/high | **Verified (live)** | Not on gpt-4o-mini (medium only) |
| `safety_identifier` | **Verified (live)** | Accepted by every model |
| Fast mode: `service_tier` `priority` and `fast` both work, 2x cost | **Verified (live + pricing page)** | Both accepted; API echoes `fast` on GPT-6.x, `priority` on older models. Pricing page: 2x on gpt-5.2/6.x, ~1.67x on gpt-4o-mini |
| Flex is 50% cheaper | **Inconsistent** | Accepted on terra and gpt-5.2; **429 "Flex does not have sufficient resources"** on gpt-6.1-sol and gpt-6-luna during the test |
| Predicted Outputs unsupported on Terra, "verify for Luna" | **Verified (live)** | Rejected on terra, luna, sol, gpt-5.2. Not used by us |
| (not in the doc) | **New finding** | `logprobs` and `stop` are rejected on gpt-5.2 and all GPT-6.x; `response_format: json_schema` works on all; `json_object` works (needs the word "JSON" in the prompt) |
| Terra price $1 / $0.05 cached / $5 | **Verified** | OpenAI pricing page |
| Luna "roughly $0.20 per 1M input" | **Wrong / conflated** | `gpt-6-luna` is **$0.10 / $0.50**; `gpt-5.6-luna` is $0.20 / $1.20. gpt-4o-mini is $0.15 / $0.60; gpt-5.2 $1.75 / $14 |
| `gpt-image-1-mini` retires Dec 1, 2026 | **Not verifiable here** | Still works today. `gpt-image-1` is already gone (see headline 1) |
| `gpt-image-2.5-flare` / `-sunburst` | **Verified (live)** | Both work. **`gpt-image-2.6-flare` / `-sunburst` also exist and work; the doc does not mention them** |
| `quality` adds `xhigh` | **Verified (live)** | Accepted on 2.5-flare: 3,122 output tokens, 32 s (vs 439 tokens, ~8-12 s at medium). `max` not tested (cost) |
| Any `WxH`, multiples of 16 | **Verified (live)** | `1280x720` accepted on gpt-image-2 and newer; rejected on 1.5 and mini (error lists only the three fixed sizes) |
| Do not send `input_fidelity` on gpt-image-2+ | **Verified (live)** | 400 on 2, 2.5, 2.6. Also: mini rejects `high` (accepts `low`); 1.5 accepts both |
| `background:"transparent"` "fully supported" | **Wrong for gpt-image-2** | *"Transparent background is not supported for this model."* Works on 1.5, mini, 2.5, 2.6 |
| `response_format` unsupported | **Verified (live)** | *"Unknown parameter"* on every image model |
| Sunburst may need ~360 s timeouts | **Not seen** | At medium, sunburst took 14-28 s; slowest call anywhere was 42.5 s (gpt-image-2) |

**Claude** (live results in `local_run/model_upgrade/claude_bench.json`, `claude_design.json`; items marked "guide" were not tested)

| Claim in the doc | Verdict | Evidence |
|---|---|---|
| Model IDs `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` (no date) | **Verified (live)** | Served; 1M context, 128K max output (Haiku 4.5: 200K / 64K). `claude-haiku-4-5` is an alias that resolves to `claude-haiku-4-5-20251001` (`/v1/models` lists only the dated name) |
| `temperature`/`top_p`/`top_k` rejected | **Verified (live) and wider than the doc says** | `temperature` -> `400 "\`temperature\` is deprecated for this model"` on **Opus 5 and Sonnet 5 as well** as Haiku 5.5. Haiku 4.5 still accepts it. Not tested: `top_p`/`top_k`. The guide adds that API version `2026-10-21` rejects all three on every model; our `anthropic<1.0` pin keeps us on `2023-06-01` |
| Haiku 5.5: thinking on by default, `disabled` allowed at effort <= high | **Verified (live)** | `thinking:{type:"disabled"}` + `effort:"low"` accepted; default thinking produced no thinking blocks on short prompts (see headline 11) |
| Sonnet 5.5 `between_tools` | **Verified (live)** | Accepted (16/16 calls) |
| Opus 5.5 effort `low`/`medium`/`high` | **Verified (live)** | All accepted. Opus 5.5 `speed:"fast"` accepted with no beta header, latency halved |
| Response can start with a `thinking` block | **Verified (live)** | First block was thinking in 9/10 Opus 5 and 8/10 Opus 5.5 (medium) writing calls, 2/16 Sonnet 5 calls. Our `content[0].text` reads (autocomplete, design) would break |
| Haiku 5.5 +30% tokens | **Verified (live)** | Same autocomplete prompt: 418 input tokens on Haiku 4.5 vs 584 on Haiku 5.5 (+40%). 584 is above the 512-token cache minimum, so our existing `cache_control` on that prompt can start to work |
| Prefill, `budget_tokens`, forced `tool_choice`, refusals, history-editing check, computer use | Guide only | Not exercised by our flows in this run |
| Haiku refusals | **Doc incomplete (guide)** | Haiku 5.5 can return `stop_reason:"refusal"` and has **no server-side fallback**; the doc mentions refusals only for Sonnet. None occurred in ~550 calls |
| Opus 5.5 thinking blocks readable by Fable 5.1 / Mythos 5.1 only | **Out of date (guide)** | Also Fable 5.2 / Mythos 5.2 |
| Pricing "not verified" | Guide | Opus 5.5 $4/$20 (Opus 5: $5/$25); Sonnet 5.5 $2/$10, cache reads $0.10; Haiku 5.5 $0.10/$0.50 up to 100K-token prompts, $0.50/$2.50 above (Haiku 4.5: $1/$5). Fast mode on Opus 5.5 is 2x |


---

## 3. What breaks in our code on a model-ID-only swap (OpenAI: reproduced live)

A local script (`smoke_integration.py`, see [MODEL_UPGRADE_EVAL_SCRIPTS.md](MODEL_UPGRADE_EVAL_SCRIPTS.md)) patched `src/models.py` in memory and ran our real functions.

| Our code | Swap to | Result | Why |
|---|---|---|---|
| `get_model()` advanced, no cap ([text_processing_agent.py:386](../src/text_processing_agent.py)) | gpt-6.1-terra / gpt-6.1-sol | **400** | `temperature=0.7` is passed because the `startswith("gpt-5")` branch (L462) does not match `gpt-6` |
| `get_model()` with `max_tokens` ([L359-369](../src/text_processing_agent.py)) | gpt-6.1-terra / sol | **400** | `_output_limit_kwargs` sends `max_tokens` (only `gpt-5*` gets `max_completion_tokens`); `reasoning_headroom()` ([models.py:80](../src/models.py)) also skips `gpt-6` |
| `get_model()` basic, temperature 0.3 | gpt-6-luna | **400** | same `startswith("gpt-5")` miss. `gpt-5.6-luna` works as-is |
| `get_autocomplete_suggestion()` ([L3473](../src/text_processing_agent.py)) | gpt-6-luna | **returns "" silently** | `max_tokens=100, temperature=0.3` (L3524-3525) rejected; the `except` returns "" |
| `intent_classifier._call()` ([intent_classifier.py:21-25](../src/intent_classifier.py)) | gpt-6-luna | **returns "" -> classified `other`** | `temperature=0`, `max_tokens=5` |
| `_run_design_json_completion()` OpenAI branch ([L3410-3417](../src/text_processing_agent.py)) | gpt-6.1-terra | **400** | `max_tokens` (and `temperature`) |
| Voice formatter OpenAI backend ([engine.py:165](../src/voice_flow/formatter/engine.py)) | gpt-6-luna | works, one wasted call | learns the temperature 400 at run time; `_KNOWN_REASONING_PREFIXES` misses `gpt-6` |
| Tool-using agents (`code_writer`, `smart_global`, `blog_writer`) on Terra | gpt-6.1-terra | **400** unless effort is `none` or the agent uses `OpenAIResponses` | Chat Completions + tools at default effort |
| Image: `_call_openai_with_fallback` ([image_gen.py:726-727](../src/image_gen.py)) | any `gpt-image-2+` edit | **400 on `input_fidelity`, then a "bare" retry that also drops `quality` and `output_format`** | `_is_optional_param_error` (L374-385) retries without *all* tuning params, so a transparency request on gpt-image-2 silently loses format and quality too |

Confirmed fixes (each run live and OK): `OpenAIChat(id="gpt-6.1-terra", max_completion_tokens=N)` with no temperature; `OpenAIChat(id="gpt-6-luna", temperature=0.3, reasoning_effort="none")`; `OpenAIChat(..., reasoning_effort="none")` or `OpenAIResponses(...)` for tool calls. agno's `OpenAIChat` already has the fields `reasoning_effort`, `verbosity`, `service_tier`, `max_completion_tokens`, `extra_body`.

**Claude side (offline check, then confirmed live):** agno builds `{'max_tokens': N, 'temperature': T}` for `claude-opus-5`, `-opus-5-5`, `-sonnet-5-5`, `-haiku-4-5`, `-haiku-5-5` alike ([get_model L468-477](../src/text_processing_agent.py); agno omits `temperature` only when it is `None`). Live, that request is a 400 on Opus 5, Sonnet 5 and Haiku 5.5 (see §2). Direct calls also send `temperature` and read `response.content[0].text`: autocomplete ([L3529-3546](../src/text_processing_agent.py)), design canvas ([L3375-3389](../src/text_processing_agent.py)); a thinking block can come first on Opus and Sonnet, so those reads must become "first block of type text". Classifier: [intent_classifier.py:21-25](../src/intent_classifier.py).

---

## 4. Before / after, OpenAI (live)

Latency is wall-clock from this machine; all four groups ran concurrently, so read differences, not absolutes. "Judge" = blind pairwise, gpt-5.5, both A/B orders, 1-10; "W/T/L" = candidate wins / ties / losses vs today's model (a win needs to win or tie in both orders).

### 4.1 Long-form writing, enhance, make-longer, blog section, email  (gpt-5.2 today -> gpt-6.1-terra / sol)

5 tasks x 2 runs, streamed.

| Config | TTFT p50 / p95 | Total p50 / p95 | Out tok (reasoning) | $/call | Judge: today -> cand | W/T/L |
|---|---|---|---|---|---|---|
| **gpt-5.2 (today)** | 0.85 / 0.92 s | 5.34 / 8.79 s | 276 (0) | $0.0043 | - | - |
| terra, default effort | 1.31 / **8.92 s** | 3.02 / **16.92 s** | 366 (149) | $0.0021 | 7.3 -> 8.6 | 3/1/1 |
| **terra, effort=none** | 1.07 / 1.23 s | **2.85** / 9.98 s | 212 (0) | **$0.0013** | 6.9 -> 8.0 | 4/1/0 |
| terra, effort=low | 1.32 / 2.43 s | 3.16 / 9.15 s | 227 (16) | $0.0014 | 7.0 -> 8.4 | 4/0/1 |
| sol, effort=low | 1.46 / 1.69 s | 6.86 / 15.18 s | 238 (1) | $0.0029 | 7.2 -> 8.8 | 4/0/1 |

What to know: terra wins mostly on instruction-following (kept a single paragraph where gpt-5.2 split it, fewer invented specifics in the vendor email, one bullet list where asked). **Regression:** `make_longer` came back at 72-86 words (gpt-5.2: 266; sol: 147); the judge preferred gpt-5.2 in 3 of 4 pairs for that task. Terra's first token is ~0.2 s later than gpt-5.2's.

### 4.2 Quick edits: grammar, simplify, shorter, summarize  (gpt-4o-mini today -> luna / terra)

8 tasks x 3 runs, streamed.

| Config | TTFT p50 | Total p50 / p95 | $/call | Judge: today -> cand | W/T/L | Grammar vs gold (sim / exact of 9) |
|---|---|---|---|---|---|---|
| **gpt-4o-mini (today)** | 0.64 s | **0.90** / 1.59 s | $0.000062 | - | - | 0.985 / 5 |
| gpt-6-luna, effort=none | 0.83 s | 1.10 / 1.34 s | $0.000044 | 8.6 -> 9.1 | 4/3/1 | 0.990 / 6 |
| gpt-6-luna, effort=low | 0.83 s | 1.09 / 1.51 s | $0.000047 | 8.7 -> 9.3 | 4/3/1 | 0.994 / 7 |
| gpt-5.6-luna, effort=none | 0.76 s | 1.00 / 1.65 s | $0.000095 | 8.6 -> 9.1 | 5/2/1 | 0.990 / 6 |
| gpt-6.1-terra, effort=none | 1.07 s | 1.34 / 1.78 s | $0.000429 | 8.6 -> 9.6 | 5/3/0 | 0.989 / 6 |

Quality is equal or slightly better on every candidate; latency is 0.1-0.45 s worse than gpt-4o-mini. gpt-6-luna is the cheapest; gpt-5.6-luna costs ~1.5x gpt-4o-mini per call.

### 4.3 Inline autocomplete  (today: `max_tokens=100, temperature=0.3`, non-streaming)

10 contexts x 3 runs, through the app's own `_postprocess_autocomplete`.

| Config | Total p50 / p95 | Usable | Judge as emitted: today -> cand (W/T/L) | Judge, spacing normalised | Glued word (no leading space) |
|---|---|---|---|---|---|
| **gpt-4o-mini (today)** | **0.80** / 0.91 s | 30/30 | - | - | **50%** |
| gpt-4o-mini + `service_tier=priority` | 0.75 / 0.85 s | 30/30 | - | - | 60% |
| gpt-6-luna, effort=none | 1.06 / 1.48 s | 30/30 | 7.5 -> 5.5 (1/3/6) | 8.4 -> 8.0 (4/2/4) | 70% |
| gpt-6-luna, effort=none + `priority` | 0.90 / 1.13 s | 30/30 | - | - | 67% |
| gpt-5.6-luna, effort=none | 0.96 / 1.28 s | 30/30 | 7.5 -> 6.5 (2/2/6) | 8.5 -> 8.0 (3/2/5) | 57% |
| gpt-6-luna, effort **unset** | 1.60 / 2.05 s | **23/30** (6 hit the 100-token cap, 1 error) | - | - | - |

Content quality is comparable; luna loses mostly on the join: it forgets the leading space more often (70% vs 50%). **Pre-existing issue found on the way:** half of *today's* gpt-4o-mini suggestions start without the required space after a complete word ("...next step is to" + "schedule..."), and `_postprocess_autocomplete` only repairs a *capitalised* glue. If the frontend does not repair it, users see this now.

### 4.4 Code-intent classifier  (today: `max_tokens=5, temperature=0`)

60 labelled queries (30 code / 30 other, from `tests/fixtures/code_intent_eval_cases.py`), 4 in flight like production.

| Config | Accuracy | Recall on real code requests | Total p50 / p95 |
|---|---|---|---|
| **gpt-4o-mini (today)** | 0.88 | **100%**; but calls 23% of the "other" queries `code` | **0.70** / 0.91 s |
| gpt-6-luna, effort=none | 0.87 | 73%; 0% false `code` | 0.91 / 1.43 s |
| gpt-5.6-luna, effort=none | 0.85 | 73%; 3% false `code` | 0.86 / 1.11 s |
| gpt-6-luna, effort unset, cap 5 | **0.00** (60/60 empty) | 0% | 1.03 / 1.58 s |

Same overall accuracy, opposite error profile: 4o-mini over-calls "code", luna under-calls it. Production only sends regex-"unsure" queries here, so this sample is a proxy, not the production mix.

### 4.5 Mode 8 design canvas, through the real harness (draft, validate, repair, auto-heal)

3 prompts x 2 runs each = 6 runs per config.

| Config | Succeeded | Total p50 / p95 | Runs needing a repair round | Output tokens (reasoning) |
|---|---|---|---|---|
| **gpt-5.2 (today)** | 6/6 | 15.9 / 31.7 s | 2 of 6 | 1,538 (0) |
| terra, effort=none | 6/6 | 19.2 / 27.5 s | 2 of 6 | 1,613 (0) |
| terra, effort=low | 6/6 | 21.9 / 26.8 s | 2 of 6 | 1,568 (24) |
| terra, default effort | 6/6 | 33.3 / 40.1 s | 2 of 6 | 2,437 (290) |
| **sol, effort=low** | 6/6 | **9.8 / 17.1 s** | **0 of 6** | 1,031 (0) |

Terra is not faster than gpt-5.2 here and drifts slower at default effort. Sol is the standout, but n=6 per config. Sol lists at $2/$10 per 1M tokens (gpt-5.2: $1.75/$14), and it wrote 33% fewer tokens.

### 4.6 Images (medium quality, 1024x1024, 2 prompts per model, one run each)

Eyeball sheets: `local_run/model_upgrade/sheet_poster.jpg`, `sheet_product.jpg`, `images.html`.

| Model | Latency (poster / product) | Output tokens | ~$/image (list price) | What I saw |
|---|---|---|---|---|
| gpt-image-1 | **does not exist** | - | - | - |
| gpt-image-1-mini | 12.3 / 12.8 s | 1,056 | $0.008 | Headline cropped at the top edge; flat dark lighthouse |
| gpt-image-1.5 | 11.3 / 11.6 s | ~1,370 | $0.044 | Good; text OK |
| gpt-image-2 | 42.5 / 31.5 s | 1,756 | $0.053 | Rich detail, slowest by 3x |
| **gpt-image-2.5-flare** | 12.5 / 8.3 s | **439** | **$0.013** | Clean, text fully visible |
| gpt-image-2.5-sunburst | 15.7 / 14.1 s | 439 | $0.013 | Painterly, text OK |
| gpt-image-2.6-flare | **6.6 / 6.0 s** | 1,449 | $0.043 | Fast, strong layout, text OK |
| gpt-image-2.6-sunburst | 28.0 / 21.5 s | 1,449 | $0.043 | Richest detail, slow |

All new models rendered the poster text correctly; product shots are hard to tell apart. 2.5 spends 3.3x fewer output tokens than 2.6 for a similar look (one prompt pair, subjective).

---

### 4.7 Claude: writing, enhance, make-longer, blog section, email  (Opus 5 today -> Opus 5.5)

Same 5 tasks x 2 runs, streamed. "Today" (Opus 5 with the app's temperature) is a 400 on all 10 calls, so the quality yardstick is **Opus 5 without temperature** (effort at its default, high). Judge as in §4.

| Config | TTFT p50 | Total p50 / p95 | Out tok | $/call (list) | Judge: Opus 5 -> cand | W/T/L |
|---|---|---|---|---|---|---|
| Opus 5 **as the app sends it** (temperature) | **400 x 10** | - | - | - | - | - |
| Opus 5, no temperature (effort high) | 1.69 s | 7.43 / 16.64 s | 508 | $0.0149 | - | - |
| Opus 5.5, effort=low | 1.27 s | 6.93 / 13.13 s | 603 | $0.0138 | 7.5 -> 8.8 | 4/1/0 |
| Opus 5.5, effort=medium | 2.16 s | 9.52 / 14.62 s | 779 | $0.0173 | 7.7 -> 9.0 | 3/0/2 |
| Opus 5.5, effort=high | 2.43 s | 10.66 / 16.44 s | 846 | $0.0186 | 7.2 -> 9.0 | 4/1/0 |
| **Opus 5.5, medium + fast mode** | **1.14 s** | **4.36 / 5.40 s** | 774 | $0.0344 (2x) | 7.2 -> 8.8 | 5/0/0 |

Effort is a real lever on Opus 5.5 for long output: low is cheapest and fastest, high adds ~3.7 s and +35% cost over low for no extra judge score. Fast mode cut median total 54% against medium at 2x price. No "make longer" regression (362-408 words vs Opus 5's 450).

### 4.8 Claude: quick edits (Sonnet 5 today -> Sonnet 5.5, or Haiku 5.5)

8 tasks x 2 runs, streamed; yardstick = Sonnet 5 without temperature.

| Config | TTFT p50 | Total p50 / p95 | $/call | Judge: Sonnet 5 -> cand | W/T/L |
|---|---|---|---|---|---|
| Sonnet 5 as the app sends it (temperature) | **400 x 16** | - | - | - | - |
| Sonnet 5, no temperature | 0.63 s | **0.99** / 3.42 s | $0.00158 | - | - |
| Sonnet 5.5, effort=low | 0.77 s | 1.23 / 2.15 s | $0.00150 | 8.4 -> 9.3 | 3/5/0 |
| Sonnet 5.5, effort=medium | 0.76 s | 1.19 / 2.07 s | $0.00150 | 8.3 -> 8.9 | 3/4/1 |
| Sonnet 5.5, default (high) | 0.80 s | 1.23 / 2.02 s | $0.00147 | 8.3 -> 8.9 | 3/5/0 |
| Sonnet 5.5, `between_tools` | 1.13 s | 1.51 / 1.92 s | $0.00149 | 8.4 -> 9.0 | 4/2/2 |
| Haiku 5.5, effort=low | 0.72 s | **0.97** / 1.67 s | **$0.000074** | 8.8 -> 8.6 | 1/4/3 |

On these short edits effort made no measurable difference (1.19-1.23 s; the model did not think), so tuning effort only matters for long or hard work. Sonnet 5.5 is ~0.24 s slower at the median but has a tighter tail (p95 2.0 s vs 3.4 s). Haiku 5.5 matches Sonnet 5's speed at 1/20 of the cost for a small quality dip (1 win, 3 losses in 8).

### 4.9 Claude: autocomplete and the code-intent classifier  (Haiku 4.5 today -> Haiku 5.5)

Autocomplete: 10 contexts x 2 runs, non-streaming (today: `max_tokens=100, temperature=0.3`).

| Config | Total p50 / p95 | $/call | Judge, spacing normalised: today -> cand (W/T/L) | Glued word (no leading space) |
|---|---|---|---|---|
| **Haiku 4.5 (today)** | **0.83** / 0.96 s | $0.00051 | - | **80%** |
| Haiku 5.5, today's params | **400 x 20** | - | - | - |
| Haiku 5.5, temperature removed, defaults | 1.03 / 1.25 s | $0.000074 | 8.5 -> 7.8 (3/4/3) | 85% |
| Haiku 5.5, thinking disabled + effort low | 1.03 / 2.01 s | $0.000072 | 8.5 -> 8.3 (3/3/4) | 85% |
| Haiku 5.5, effort low, cap 400 | 1.04 / 1.32 s | $0.000076 | 8.4 -> 7.7 (3/3/4) | 80% |

Haiku 5.5 is ~0.2 s slower, about equal in quality, and 85% cheaper per call. **Pre-existing:** today's Haiku 4.5 omits the required leading space on 80% of suggestions (OpenAI gpt-4o-mini: 50%, §4.3).

Classifier: 60 labelled queries (30 code / 30 other), 4 in flight. *Strict* = exact `code`/`other` (what [intent_classifier.py](../src/intent_classifier.py) accepts); *lenient* = first word.

| Config | Strict acc | Lenient acc | Code recall (strict) | False `code` on "other" | Total p50 / p95 | $/call |
|---|---|---|---|---|---|---|
| **Haiku 4.5 (today)** | **0.38** | 0.55 | 73% | 80% | **0.67** / 0.74 s | $0.000058 |
| Haiku 5.5, today's params | 0.00 (400 x 60) | - | - | - | - | - |
| Haiku 5.5, temp removed, cap 5, default thinking | 0.82 | 0.82 | 100% | 37% | 0.85 / 1.08 s | $0.000007 |
| **Haiku 5.5, thinking disabled + effort low** | **0.95** | 0.95 | 100% | 10% | 0.87 / 1.24 s | $0.000007 |

Why today's score is low: Haiku 4.5 ignores "one word only" on code-like queries and answers `code` followed by the start of the code (12 of 60 replies hit the 5-token cap, e.g. "code\n\n```html"); the exact-match check turns those into `other`. The sample includes hard negatives ("write a code review"); production sends only regex-"unsure" queries to the LLM, so this is a proxy.

### 4.10 Claude: Mode 8 design canvas on Opus, real harness

3 prompts x 2 runs = 6 per config. Today's call is Opus 5 + `temperature` (400 live); yardstick = Opus 5 without temperature at today's 8,192-token cap.

| Config | Succeeded | Total p50 / p95 | Repair rounds | Output tokens (incl. thinking) |
|---|---|---|---|---|
| Opus 5, cap 8,192, no temperature | 6/6 | 25.0 / 55.5 s | 2 of 6 | 3,314 |
| **Opus 5.5, effort low, cap 16,000** | 6/6 | **10.7 / 10.9 s** | 0 of 6 | 1,358 |
| Opus 5.5, effort medium, cap 16,000 | 6/6 | 12.4 / 16.0 s | 0 of 6 | 1,613 |
| Opus 5.5, effort medium, **cap 8,192 (today's)** | 6/6 | 11.2 / 18.6 s | 0 of 6 | 1,703 |
| **Opus 5.5, medium + fast mode** | 6/6 | **4.8 / 6.7 s** | 0 of 6 | 1,659 |

Opus 5.5 wrote half the tokens Opus 5 did and was 2-5x faster at the median, with no truncation at the existing 8,192 cap in these runs. Compare with OpenAI on the same harness (§4.5): gpt-5.2 15.9 s, gpt-6.1-sol effort low 9.8 s.

## 5. Claude: summary and what the guide still says

**Measured live** (after the key was replaced): everything in §4.7-4.10 and the Claude rows of §2. **Not exercised:** tool-calling loops, forced `tool_choice`, prefill removal effects, refusals, vision, history-editing/thinking-block replay, computer use. Claude quality was judged by `gpt-5.5` (an OpenAI model) against the same-provider baseline, same as the OpenAI sections.

**What has to change in our code for Claude** (the doc agrees except where §2 notes):

- Remove `temperature` on every Claude request: `get_model` Claude branch (pass `None`), autocomplete, design, classifier. This was a bug **on Opus 5 and Sonnet 5**, not only an upgrade step. **Fixed 2026-11-11** with `claude_accepts_temperature()` in `src/models.py`, applied at the five Claude call sites (§10 below).
- Autocomplete and classifier on Haiku 5.5: send `thinking:{type:"disabled"}` with `output_config:{effort:"low"}`; read the first `text` block, not `content[0]`.
- Opus 5.5: choose effort per use (low for design canvas and drafting looked best here; medium/high only when a task measures better), raise `max_tokens` where thinking is on, handle `stop_reason:"refusal"`.
- Sonnet 5.5 quick edits: effort made no difference on short tasks; `between_tools` was slower at the median. Keep the default.
- Keep conversation history append-only if thinking blocks are replayed (guide; not tested).
- API version `2026-10-21` removes sampling params on every model; our `anthropic<1.0` + `agno==2.8.5` pins exist because newer SDKs dropped `temperature` ([requirements.txt](../requirements.txt)). Plan the agno upgrade (or `temperature=None`) before bumping.
- Pricing (guide; matches the per-call costs measured above): Haiku 5.5 is 10x cheaper per token than 4.5 (+40% tokens for our prompt); Opus 5.5 is 20% cheaper per token than Opus 5 but spent more thinking tokens at medium/high.

---

## 6. Parameters and features we do not use that could help

Ranked by expected value for this codebase. "Measured" = I tested the effect; "accepted" = the API took it, benefit untested.

| # | Feature | Where it helps | Status |
|---|---|---|---|
| 1 | **OpenAI `reasoning_effort`** (never set today) | The main latency/cost lever on GPT-6.x: terra p95 TTFT 8.9 s -> 1.2 s, $0.0021 -> $0.0013 per call | **Measured** |
| 2 | **OpenAI `service_tier: "priority"/"fast"`** | -15% median on luna autocomplete (1.06 -> 0.90 s), -6% on 4o-mini; earlier memory: 15.0 -> 9.9 s on the blog outline. 2x price on 6.x/5.2 | **Measured**; the API confirms the tier was honoured |
| 3 | **Claude `output_config.effort`** + Haiku `thinking:{type:"disabled"}` | Opus 5.5 design canvas 25.0 s -> 10.7 s (low); classifier accuracy 0.82 -> 0.95 with thinking off; on Sonnet 5.5 short edits it changed nothing | **Measured** |
| 4 | **Claude fast mode** (`speed:"fast"`, Opus 5.5, 2x price) | Design canvas 12.4 s -> 4.8 s median; drafting 9.5 s -> 4.4 s; accepted with no beta header | **Measured** |
| 5 | **Structured outputs** (`response_format: json_schema` on OpenAI, `output_config.format` on Claude) | Design canvas: 2 of 6 runs on gpt-5.2 and terra needed a repair round (a second full call, +10-17 s in these runs). A strict schema could remove it; `src/design_canvas/prompts.py` already has `_without_titles`, which looks like schema prep. Whether strict mode accepts our recursive AST schema is not tested | `json_schema` **accepted** on all models |
| 6 | **`safety_identifier`** (OpenAI) | Per-end-user abuse attribution on the platform key at zero cost | **Accepted** |
| 7 | **Prompt caching knobs** (`prompt_cache_key`; Claude auto-cache) | Cached input is 10-20x cheaper (terra $0.05 vs $1; luna $0.01 vs $0.10). OpenAI caches only prompts over 1,024 tokens: our autocomplete prompt is ~410 (no). On Claude, the same prompt is 584 tokens on Haiku 5.5, above its 512-token minimum, so the `cache_control` we already send can start to take effect (it never could on Haiku 4.5, minimum 4,096). Blog, long-form and design prompts are the other candidates | **Accepted / token counts measured**; saving not measured |
| 8 | **Native image sizes** (gpt-image-2+: any `WxH`, multiples of 16) | We map 16:9 *and* 4:3 to `1536x1024` (3:2) ([image_gen.py:335-341](../src/image_gen.py)); `1280x720` etc. are accepted | **Accepted** |
| 9 | **`verbosity`** (OpenAI, replaces "be concise" in prompts) | Output length control on 5.2/6.x (blog/email/make-longer); not on gpt-4o-mini | **Accepted**; effect on quality untested |
| 10 | **Claude Models API** (`max_tokens`, `max_input_tokens` per model) | Replaces our unverified `OUTPUT_TOKEN_CEILING = 16,384` (open item O7, [models.py:70](../src/models.py)): live values are 128K output / 1M context for Opus/Sonnet/Haiku 5.x, 64K / 200K for Haiku 4.5. (OpenAI's list endpoint exposes no limits.) | **Measured** |
| 11 | **`reasoning.mode:"pro"`, `reasoning.context`, effort `max`** (Responses API) | Hard, high-value tasks only | **Accepted** on new models; no use case yet |
| 12 | Image `quality:"xhigh"`, `moderation:"low"`, `output_format:"webp"` | Premium quality tier (3,122 tokens ~ $0.09/image on flare) | **Accepted** |

Not usable on GPT-6.x: `stop`, `logprobs`, Predicted Outputs, `top_p`/`temperature` unless effort is `none`.

---

## 7. Decision sheet (yours to fill)

| Change | Latency | Quality | Cost | Code work | Keep / discard |
|---|---|---|---|---|---|
| gpt-5.2 -> gpt-6.1-terra, effort=none (writing, enhance, email, blog) | median -47%, first token +0.2 s | 8.0 vs 6.9 (4/1/0); `make_longer` shorter | -69% | `get_model`, `_output_limit_kwargs`, tools need effort `none` or Responses API | |
| gpt-5.2 -> gpt-6.1-sol, effort=low (design canvas) | median -38%, p95 -46% | no repairs in 6 runs | 33% fewer tokens; $2/$10 list vs gpt-5.2 $1.75/$14 (2x terra) | same | |
| gpt-4o-mini -> gpt-6-luna, effort=none (quick edits, autocomplete, classifier) | +0.2-0.26 s median | ~equal; classifier error profile flips | -30% per call | autocomplete, classifier, `get_model` | |
| gpt-4o-mini -> gpt-5.6-luna, effort=none | +0.1-0.16 s | ~equal | +50% per call | none (works as-is) | |
| OpenAI images: gpt-image-1 -> 2.5-flare / 2.6-flare / other | see §4.6 | all fine; mini crops text | see §4.6 | model IDs, `input_fidelity`, transparency, error handling | **required either way** |
| **Opus 5 -> 5.5, effort low** (design canvas) | median 25.0 -> 10.7 s, p95 55.5 -> 10.9 s | no repair rounds (Opus 5: 2 of 6), half the tokens | not priced per run; tokens halved at -20% price | drop `temperature`, read first `text` block, raise cap if effort rises | |
| Opus 5.5 + **fast mode** (design canvas, drafting) | design 12.4 -> 4.8 s; drafting 9.5 -> 4.4 s | same as above | 2x price | add `speed:"fast"` | |
| Opus 5 -> 5.5 effort low / medium (writing, enhance, email) | median 7.4 -> 6.9 s (low), 9.5 s (medium) | judge 8.8-9.0 vs 7.5 | -7% (low) to +16% (medium) per call | drop `temperature`, block parsing | |
| Sonnet 5 -> 5.5 (quick edits) | median +0.24 s, p95 3.4 -> 2.0 s | equal or slightly better (3/5/0) | flat | drop `temperature` only; effort changed nothing | |
| Haiku 5.5 effort low as the quick-edit tier instead of Sonnet | 0.97 s (= Sonnet 5 today) | slightly lower (1/4/3) | -95% per call | new Claude basic-tier ID + settings | |
| Haiku 4.5 -> 5.5 (autocomplete) | median +0.2 s (0.83 -> 1.03) | about equal (3/3/4) | -85% per call | drop `temperature`, thinking disabled + effort low, block parsing | |
| Haiku 4.5 -> 5.5 (code-intent classifier) | median +0.2 s | strict accuracy 0.38 -> 0.95 | -88% per call | same | |
| **Remove `temperature` from every Claude request** | none | none | none | `get_model`, autocomplete, design, classifier | **Done 2026-11-11** (see §10) |

---

## 8. Reproduce

The comparison scripts are not committed (they hard-code this round's model IDs, cost money and need keys); [MODEL_UPGRADE_EVAL_SCRIPTS.md](MODEL_UPGRADE_EVAL_SCRIPTS.md) lists each one, its cost and what it found, and gives the recipe for rebuilding them. What can be run from the repository:

```bash
cd agent
.venv/bin/python scripts/preflight_models.py     # free: every key and every configured model ID (see ROLLOUT_RUNBOOK.md)
.venv/bin/python -m pytest tests -q              # the behaviours found in this review are pinned by tests
```

Raw results from the original runs: `local_run/model_upgrade/*.json` on the machine that ran them (gitignored).

---

## 9. Caveats

- Small samples: OpenAI: 5 writing tasks x 2 runs, 8 quick-edit tasks x 3, 10 autocomplete contexts x 3, 60 classifier queries, 6 design runs per config, 2 prompts x 7 image models x 1 run. Claude: the same tasks with 2 runs for writing, quick edits and autocomplete, 60 classifier queries, 6 design runs per config. p95 values over 10-30 calls are indicative only; Claude autocomplete p95 of 2.01 s for one config is a single slow call.
- Latency is one machine, one network, four benchmark processes running at once; absolute numbers move, ordering is more trustworthy than gaps under ~0.2 s.
- Quality judge is a single model (`gpt-5.5`, an OpenAI model, which may favour OpenAI-style output). For OpenAI sections every candidate is OpenAI; for Claude sections the judge is a different vendor, which avoids self-preference but means Claude and OpenAI quality scores are **not comparable across tables** (each is relative to its own baseline). Prompts were mine, not production traffic. Grammar gold sentences are mine. Image quality is my own look at two prompts.
- The classifier sample includes clear code requests; production routes only regex-"unsure" queries to the LLM.
- Prices are OpenAI's list prices from its pricing page (fetched 2026-11-11) and Anthropic's from the migration guide (fast mode priced at 2x; thinking tokens are included in the output-token counts); The pricing page also notes promotional pricing for GPT-5.6 Sol until 2026-11-30; the gpt-6.1-sol figures above are standard.
- Not tested: Fine-tuning, Realtime, Batch, 16-image edits, image `max` quality, Responses-API streaming latency, vision inputs, tool-heavy agents end to end on Terra; for Claude: tool loops, forced `tool_choice`, prefill, refusals, thinking-block replay, vision, the effect of `cache_control` on Haiku 5.5, and whether `speed:"fast"` was honoured (inferred from latency; `usage.speed` was not recorded).

---

## 10. Change made: Claude `temperature` fix (2026-11-11)

The only production change so far. Model IDs, prompts and tiers are untouched.

- `src/models.py`: `claude_accepts_temperature(model_id)`, an **allow-list** (`claude-3*`, `claude-haiku-4*`, `claude-sonnet-4*`, `claude-opus-4-0/4-1/4-5/4-6`). Probed live with `temperature=0.3`: accepted by haiku-4-5, sonnet-4-6, opus-4-6, opus-4-5; rejected (400) by opus-4-7, opus-4-8, opus-5, opus-5-5, sonnet-5, sonnet-5-5, haiku-5-5. Unlisted or future IDs get no temperature.
- Applied at all five Claude request sites: `get_model` Claude branch (agno gets `temperature=None`, and the model cache key ignores temperature for these models), `_run_design_json_completion`, `get_autocomplete_suggestion`, `intent_classifier._call`, voice `ClaudeBackend`. Haiku 4.5 paths keep their `temperature` (0.3 autocomplete, 0 classifier and voice) until Haiku is upgraded, at which point they stop sending it automatically.
- Tests: `tests/core/test_claude_temperature.py` (11 tests). With the old behaviour restored, 12 of its checks fail, so each site is covered. Full suite: 2,544 passed.
- Live, before/after: agno `Claude(claude-opus-5 / claude-sonnet-5, temperature=0.2)` -> 400; `get_model("claude")` advanced (Opus 5) and basic (Sonnet 5), the design completion, autocomplete and the voice backend -> OK.
- Not fixed here (separate, part of the Haiku 5.5 step): today's Haiku 4.5 classifier answers `Code\n\n```python`; the request now succeeds but our exact-match check still turns that reply into `other` (§4.9).

---

## 11. Change made: Haiku 4.5 -> Haiku 5.5 on the light tier (2026-11-11)

Scope: `AUTOCOMPLETE_MODEL_IDS["claude"]` only, which serves inline autocomplete, the code-intent classifier and the voice formatter's Claude backend. The advanced (Opus 5) and basic (Sonnet 5) tiers are unchanged.

- `src/models.py`: the ID is now `claude-haiku-5-5`; new `claude_light_request_params(model_id)` returns `thinking: {"type":"disabled"}` + `output_config: {"effort":"low"}` for Haiku 5 and newer (Haiku 4.5 and older reject `effort`, so they get nothing).
- Autocomplete ([text_processing_agent.py](../src/text_processing_agent.py)): sends those fields, no temperature, and reads the reply **by block type** (a thinking block can come first; a refusal has no text and now yields no suggestion instead of an exception that was swallowed).
- Classifier ([intent_classifier.py](../src/intent_classifier.py)): same fields on the agno model, `max_tokens` 8 for Claude (5 left for other providers); measured config was 8.
- Voice Claude backend ([engine.py](../src/voice_flow/formatter/engine.py)): same fields.
- `agent.md` autocomplete-tier table updated.
- Tests: `tests/core/test_claude_light_tier.py` (13 tests: request shape per call site, Haiku 4.5 requests unchanged, thinking-first and empty replies). Full suite: 2,557 passed.
- Live, through the real functions with the new key: autocomplete 20 calls, median 1.18 s, p95 2.40 s, 0 empty (benchmark config: 1.03 / 2.01 s; Haiku 4.5 was 0.83 s). Classifier 60 labelled queries via `intent_classifier._call`: strict accuracy **0.97** (Haiku 4.5: 0.38), code recall 1.00, false `code` 7%, median 1.09 s. Voice backend OK.

Deployment notes: if you track cost, add `claude-haiku-5-5` to `TEXT_PRICES_PER_MTOK` (`[0.10, 0.50]` per Anthropic's guide for prompts up to 100K tokens); an unconfigured price stays "unknown", never zero. Rollback is one line (`claude-haiku-4-5` in `AUTOCOMPLETE_MODEL_IDS`); the helpers keep working for either ID.

Found, not changed (belongs with the Opus step): `_run_design_json_completion` reads `resp.content[0].text`. Opus 5 put a thinking block first in 9 of 10 drafting calls, so a real design request can raise `AttributeError` on a thinking block; only the trivial test prompts skipped thinking.

---

## 12. Change made: Opus 5 -> Opus 5.5 on Claude's advanced tier (2026-11-11)

Scope: `ADVANCED_MODEL_IDS["claude"]` (write, enhance, make-longer, email, blog, code, document and tool-using agents, and the design canvas). Sonnet 5 (basic tier), the light tier (§11) and all other providers are unchanged.

- `src/models.py`: ID is `claude-opus-5-5`; `claude_effort_params()` adds `output_config.effort` for Claude models that have it (Haiku 4.5 and Sonnet 4.5 error on it); `reasoning_headroom()` gives Opus 5.5 +4,096 tokens, because it always thinks and thinking counts against `max_tokens`.
- `config.py` / `.env.example`: new `CLAUDE_ADVANCED_EFFORT` (low | medium | high | xhigh | max), default **low**; invalid values fall back to low. Thinking is never switched off (a 400 on this model).
- `get_model`: passes the effort to agno's `output_config` (the model cache keys on it); no temperature (§10).
- Design canvas (`_run_design_json_completion`): effort, thinking headroom in the cap (8,192 -> 12,288), reads the reply by block type (fixes the `content[0].text` bug found in §11), and a refusal is now an explicit error instead of an empty design.
- Not changed: `thinking.display` stays default (omitted); the app only forwards `run_content` events, so no thinking text reaches users. Forced `tool_choice` is not used anywhere. Fast mode is not enabled (a cost decision, see below).
- Tests: `tests/core/test_claude_opus_tier.py` (13) plus two existing assertions updated for the headroom (`test_output_limits.py`, `test_claude_temperature.py`). Full suite: 2,570 passed.

**Why `low`** (new measurements, real code pipeline `process_user_query_stream` / `code_writer` with validate + repair, 8 prompts = 2 per language; judge `gpt-5.5`, both orders, baseline Opus 5 at its default effort):

| Config | Succeeded | Format-validation failures | Total p50 / max | Output tokens (mean) | Judge vs Opus 5 | W/T/L |
|---|---|---|---|---|---|---|
| **Opus 5 (before)** | 8/8 | 1 | 61.1 / 158.2 s | 6,719 | 6.7 | - |
| **Opus 5.5, effort low** | 8/8 | 0 | **14.4 / 30.3 s** | **2,105** | 8.2 | 5/3/0 |
| Opus 5.5, effort medium | 8/8 | 0 | 18.4 / 37.4 s | 2,626 | 8.3 | 6/1/1 |
| Opus 5.5, effort high (first run, no judge) | 8/8 | 2 | 28.7 / 136.6 s | 7,529 | - | - |

Code: median -76% and -69% tokens versus Opus 5; low and medium are indistinguishable in quality, high was slower and twice broke the "only fenced code blocks" rule. Together with the writing (§4.7) and design (§4.10) results, low was never worse than medium and is the cheapest.

Live checks through the production code: a multi-tool agno loop on Opus 5.5 (4 tool calls, completed; no thinking-block 400s on this account); the design canvas through `_run_design_json_completion` and the real harness, 6 runs: median **10.8 s** (Opus 5: 25.0 s), max 23.0 s, 6/6 succeeded; 2 of 6 needed a repair round (the login-form prompt both times), where the standalone benchmark in §4.10 had 0 of 6 and Opus 5 had 2 of 6.

Not done on purpose: `speed:"fast"` (design canvas 4.8 s median but 2x price) is a cost choice; the one-line way in is `extra_body={"speed": "fast"}` on the direct call. Between tool calls the model's progress text now arrives as (empty) thinking blocks, so a long tool loop can look silent in the UI until the answer streams; `thinking.display: "updates"` (beta) would restore short progress notes. Rollback: set `ADVANCED_MODEL_IDS["claude"]` back to `claude-opus-5` (the helpers keep working; `CLAUDE_ADVANCED_EFFORT` then maps to Opus 5's own dial). A `stop_reason:"refusal"` in the agno streaming path is not specially handled (it surfaces as an empty answer).

---

## 13. Change made: Sonnet 5 -> Sonnet 5.5 on Claude's basic tier, plus a JSON-fence fix (2026-11-12)

Scope: `BASIC_MODEL_IDS["claude"]` (grammar, simplify, summarize, tone, list, table, make-shorter, emojify, translation, long-document passages). Advanced and light tiers are unchanged.

- `src/models.py`: the ID is `claude-sonnet-5-5`; `reasoning_headroom()` now covers it (+4,096 tokens, since thinking counts against `max_tokens`). No temperature (§10) and no `thinking` field (`disabled` is a 400 on this model).
- `config.py` / `.env.example`: new optional `CLAUDE_BASIC_EFFORT` (low | medium | high | xhigh | max), **empty by default = the model's own default**. `get_model` applies it to the basic tier; the advanced tier keeps its own `CLAUDE_ADVANCED_EFFORT` (§12). The classifier, which builds its model from the basic tier and swaps in Haiku, does not inherit it (tested).
- Tests: `tests/core/test_claude_sonnet_tier.py` (11) and three existing assertions updated for the headroom. Full suite: 2,592 passed.

**Why the effort dial stays empty** (live, real code paths, Sonnet 5 "before" now works because `temperature` is no longer sent):

| Check | Sonnet 5 (before) | Sonnet 5.5, default effort | Sonnet 5.5, effort low |
|---|---|---|---|
| Quick edits through `process_user_query_stream`, 8 tasks x 2 runs: total p50 / p95 | 2.00 / 3.81 s | **1.81 / 2.71 s** | 1.80 / 3.13 s |
| Same: first token p50 | 1.60 s | 1.27 s | 1.22 s |
| Long-document `grammar` (2 docs, 3-4 passages each): total per doc | **47.3 s, 39.4 s** | **12.4 s, 10.3 s** | 13.0 s, 11.3 s |
| Same: first text | 8.5 s | 3.7 s | 4.0 s |
| Same: passages that needed a repair round | 4 of 7 | 0 | 0 |
| Same: structural loss, word ratio, spelling slips left | 0, 1.00, 0/22 | 0, 1.00, 0/22 | 0, 1.00, 0/22 |

Sonnet 5.5 is as fast or faster on quick edits (the earlier direct-API benchmark in §4.8 had it ~0.24 s slower: those gaps are inside run-to-run noise) and about 4x faster on long-document grammar with no repair rounds. Low effort changed nothing measurable in either path, so there is no reason to override the default. In the same long-document runs the `enhance` operation, which is on the advanced tier, took 19.2 / 22.2 / 19.0 s p50 on Opus 5.5 (unchanged by this step).

**Found while testing, fixed:** Opus 5.5 sometimes wraps a JSON answer in a ```` ```json ```` fence. The chart suggestion (`src/charts/extract.py`) parses the raw reply, so 2 of 12 live calls failed with a `ValidationError` (the user would see "graph suggestion unavailable"); the key-point highlighter (`src/key_points.py`) would have spent a second repair call each time. New `src/json_reply.py` (`strip_json_fence`) unwraps a reply only when the whole reply is exactly one fenced block, and both parsers use it (the design canvas and the email template already handled fences). `tests/test_json_reply.py` (13 tests). Live after the fix: 12 of 12 chart suggestions succeed.

Rollback: set `BASIC_MODEL_IDS["claude"]` back to `claude-sonnet-5`.

---

## 14. Change made: OpenAI (2026-11-12)

What was done, after you chose "Terra, but pin blog and code" and "Keep gpt-4o-mini" for the basic tier.

**1. Images (required: `gpt-image-1` is gone).** `src/models.py` / `src/image_gen.py`
- Default OpenAI image model `gpt-image-2.5-flare` (env `IMAGE_MODEL_OPENAI` still overrides); fallbacks `gpt-image-2.6-flare`, then `gpt-image-1.5`. The chain is the model asked for, the default, then the fallbacks, so a user who names the dead `gpt-image-1` lands on the default, not the dearer 2.6.
- `input_fidelity` is sent only to `gpt-image-1.5`, the one model that takes it (probed: mini rejects `high`, gpt-image-2 and newer reject the parameter).
- A rejected tuning parameter is now dropped **alone** (the one the error names). Before, one rejection resent the request without `quality` and `output_format` too. An error that blames a parameter we did not send is raised, not retried. The detector also learned the real wording ("does not support").
- Live through `_call_openai_with_fallback`: default model low-quality jpeg 8.9 s; transparent PNG returns RGBA; naming `gpt-image-1` falls through to the default; webp, and an edit with a reference image, work; `gpt-image-2` + transparent drops only `background` (PNG kept).
- Tests: `tests/images/test_openai_image_upgrade.py` (10) and one updated assertion.

**2. Advanced tier: `gpt-6.1-terra`**, with these rules found by probing:
- `src/models.py`: `ADVANCED_MODEL_IDS["openai"] = "gpt-6.1-terra"`; `openai_reasoning_family()` (gpt-5.x, gpt-6.x, o-series: `max_completion_tokens`, no custom temperature) replaces four `startswith("gpt-5")` checks, in `get_model`, `_output_limit_kwargs`, `reasoning_headroom` and the voice engine. gpt-6.x answered 400 to `max_tokens`, and to temperature and top_p unless effort is `none`.
- `OPENAI_ADVANCED_EFFORT` (default `none`; none | low | medium | high | xhigh). Effort is always sent on gpt-6.x because terra at its default effort had an 8.9 s p95 first token. Tool-using agents are forced to `none` (Chat Completions rejects function tools at any other effort). gpt-6.1-sol has no `none` and gets `low`. The model cache keys on model and effort.
- Rollback to gpt-5.2 sends exactly the request it always got (no effort field).
- Live through `get_model` and agno: plain call and a function-tool call both complete (`reasoning_effort: none`, `max_completion_tokens: 6144`).

**3. Pins (your choice).** `AGENT_MODEL_PINS` keeps `blog_writer`, `code_writer` and `code_explainer` on `gpt-5.2`; `get_model` takes an `agent_type` and `create_agent` passes it. Checked unpatched through the real pipelines: `code_writer` -> gpt-5.2, `enhance` and `email_template` -> gpt-6.1-terra, `grammar` -> gpt-4o-mini; Claude is unaffected. `WRITER_LENGTH_FACTOR` stays 0.96 (calibrated for gpt-5.2, which still writes the blog).

**4. Design canvas on `gpt-6.1-sol`, effort low** (`DESIGN_MODEL_IDS`, `design_model_id()`): through the production function and the real harness, 6 runs: median **11.3 s**, max 19.1 s, 0 repair rounds (gpt-5.2: median 15.9 s, p95 31.7 s). No temperature is sent (a 400 on gpt-6.x). The model label reported to the client also follows this lookup.

**5. Basic tier unchanged** (`gpt-4o-mini`), as chosen.

**The measurements behind the choice** (real production code, unpatched `process_user_query_stream`; judge `gpt-5.5`, both orders):

| Path | gpt-5.2 | gpt-6.1-terra (effort none) | Note |
|---|---|---|---|
| Blog, 4 briefs: total p50 | 41.4 s | **23.6 s** | first text 1.8 s for both; 0 structural failures both |
| Blog length / plan (mean) | 1.07 (0.96-1.22) | 0.95 (0.90-0.97) | terra ~5% under the plan; 800-word plan: 1.22 vs 0.97 |
| Blog structure | tables 2,1,2,1; quotes 2,3,4,3 | tables 1,1,2,1; quotes 2,2,3,3 | plainer, fewer tables and quotes |
| Code, 8 prompts: total p50 / max | 12.9 / 59.2 s | **3.5 / 22.7 s** | answers 3,632 vs 842 characters (p50) |
| Code quality (judge) | 7.50 | 7.75 (1 win, 5 ties, 2 losses) | but a usage example in 4 of 8 answers vs 7 of 8, and 1 vs 15 explanatory comment lines |
| Writing / enhance (§4.1) | 5.3 s, judge 6.9 | **2.9 s, judge 8.0** | "make longer" expansions much shorter |

Terra is faster and cheaper everywhere, but on the blog and on code it drops things you had said you like, hence the pins. Everything else in the advanced tier (enhance, email, documents, charts, smart global, etc.) moves.

**Deployment notes.** Per-model price tables keyed by model ID need the new IDs: `IMAGE_CREDITS_BY_MODEL` (`openai/gpt-image-2.5-flare`, `openai/gpt-image-2.6-flare`, `openai/gpt-image-1.5`; an unset model costs the flat 5 credits, and the old `openai/gpt-image-1` key no longer matches anything) and `TEXT_PRICES_PER_MTOK` (`gpt-6.1-terra` $1/$5, `gpt-6.1-sol` $2/$10 per OpenAI's page of 2026-11-11; unconfigured stays "unknown"). Not done: native aspect-ratio sizes for gpt-image-2 and newer (16:9 and 4:3 still map to 1536x1024), user-named cues for the new image models (a user typing "gpt-image-2.5" gets the default), fast tier / `service_tier`, `safety_identifier`, prompt cache keys (§6).

---

## 15. Rollout safeguards (2026-11-12)

Built after you chose "Rollout safeguards". The full operating guide is `docs/ROLLOUT_RUNBOOK.md`; this records what exists and how it was checked.

**Why:** the first Claude call of this review found a dead key that had been hidden for weeks by silent fallback, and the review then found two request-shape bugs (`temperature`, `content[0].text`) and a withdrawn image model. None of them would have shown up as an error.

1. **Provider health** (`src/provider_health.py`, `GET /agent/health/providers`): a rolling 15-minute window per provider/model of calls, ok, failures by kind (key rejected, model missing, request rejected, rate limited, server error, truncated, refused, empty) and fallbacks, fed from the text stream and its fallback loop, autocomplete, the classifier, the design canvas and the OpenAI image call. Alerts: `key_rejected` and `model_missing` (critical on the first), `request_rejected` (3 or more, or over 2%), `truncated`/`empty` (over 5% at 20+ calls), `refused` (3 or more), `high_failure_rate` (over 20%). Each critical alert is logged once an hour as `ROLLOUT_ALERT ...`. A user's own (BYOK) key failing is counted, never alerted. Nothing stored but counts and a scrubbed 160-character snippet (keys masked). The endpoint is hidden unless the backend secret header is sent or the metrics flag is on.
2. **Model preflight** (`src/model_preflight.py`, `scripts/preflight_models.py`, `MODEL_PREFLIGHT_ON_START`, default on): free lookups of every configured model (tiers, autocomplete, design, pins, image default and fallbacks) and of each platform key (Anthropic, OpenAI, Gemini), at startup in a background thread and as a pre-deploy gate (exit code). Live result today: all keys accepted, every configured model present. **Replayed the incidents against the live APIs:** an invalid Anthropic key gives `key rejected`; pointing the image default back at `gpt-image-1` gives `model_missing`. That replay found a flaw in my first version: OpenAI's `models.retrieve` still answers 200 for the withdrawn `gpt-image-1`, so the OpenAI check uses the account's model list instead.
3. **Instant rollback** (`MODEL_<ADVANCED|BASIC|AUTOCOMPLETE|DESIGN>_<PROVIDER>` env overrides in `src/models.py`): a restart, no deploy, and the request helpers follow the model ID, so a rolled-back model gets the request it expects (tested for Opus 5, gpt-5.2, Haiku 4.5). Active overrides are warned about at startup and listed in the health endpoint. The runbook has the rollback value for every change.
4. **Canary:** no request-level traffic split was built. The model choices are code defaults, so the instance split is the traffic split; the runbook says what to compare.

Tests: `tests/core/test_provider_health.py` (25), `tests/core/test_model_preflight.py` (17); `tests/conftest.py` resets the counters between tests and keeps the startup check off. Full suite: 2,665 tests (before this step: 2,623).

Known gaps (also in the runbook): the preflight cannot see a depleted balance or a user key's model access; truncation and refusals are not visible inside the streaming agno path; counters are per process (several instances: query each or alarm on the log line); the new `sk-ant-usr-` key type should be confirmed as server-appropriate.

---

## 16. Autocomplete spacing fix (2026-11-12)

The defect found in §4.3: a suggestion after a finished word often started without the space it needed ("...next step is to" + "schedule" shown as "toschedule"). Models do not reliably emit a leading space: it is an invisible token that chat APIs and models habitually drop. A second bug was found while reading the post-processor: `_strip_echo` removed the start of a *new* word that began like the last word ("...to the" + " theory" became "the ory", "...is in" + " interesting" became "teresting"; 5 of 7 synthetic cases).

**What changed** (`src/autocomplete_join.py`, `AUTOCOMPLETE_SYSTEM_PROMPT` and `_postprocess_autocomplete` in `src/text_processing_agent.py`):
- The model now puts its whole answer in `<new>...</new>` (a new word: the system adds the space) or `<cont>...</cont>` (continues straight on: finishes the word being typed, attaches punctuation, or the text already ends in a space). The model's judgment ("is the last word finished?") was fine; its whitespace output was the problem. Text outside the tag is discarded, so an explanation cannot reach the document. An untagged reply still works as before, and a leaked explanation in an untagged reply ("The text ends mid-word ...") is dropped.
- Deterministic guards for joins the model may still miss: a space after `, ; : ! ? ) ]` and after a sentence-ending `.` before a capital (not decimals, `1,000`, `10:30` or domain names), and never a double space after a space or newline.
- The echo strip now only strips an echo that is glued on, never a word separated by a space, and never a tagged answer.
- Applies to every provider that uses this path (OpenAI, Claude, Gemini, Grok, DeepSeek). The extra output is about ten tokens; latency did not change measurably.

**Measured through the production function** (labelled contexts: complete word, mid-word fragment that is not itself a word, after a space, after `,` or `.`; 74 contexts x 3 runs; "correct" = the join is right after post-processing):

| Provider / model | Old overall | New overall | Complete word | Mid-word | After `,` `.` | Median latency |
|---|---|---|---|---|---|---|
| OpenAI gpt-4o-mini | 75% | **97%** | 38% -> 92% | 100% -> 100% | 8% -> 100% | 0.92 -> 0.88 s |
| Claude Haiku 5.5 | 67% | **99%** | 14% -> 96% | 100% -> 100% | 0% -> 100% | 1.28 -> 1.23 s |
| Gemini 3.5 flash-lite | 100% | 100% | 100% -> 100% | 100% -> 100% | 100% -> 100% | 1.20 -> 1.16 s |
| Grok 4.20 non-reasoning (1 run) | - | - | 5/24 -> 24/24 | 38/38 -> 37/38 | 0/4 -> 4/4 | - |

No empty replies and no leaked explanations in the final version.

**Held-out check** (12 new sentences, 35 contexts x 3 runs; written after tuning, never used to tune): OpenAI 76% -> **98%**, Claude 70% -> **100%**, Gemini 100% -> 100%. The first corpus was used while tuning (three prompt examples were added after seeing which contexts failed), so its numbers are a development score; the held-out numbers are the unbiased ones.

**Content did not get worse** (blind pairwise judge, `gpt-5.5`, both orders, 62 continuations per provider, whitespace ignored): OpenAI 8.50 -> 8.48 (18 wins / 20 ties / 24 losses), Claude 8.54 -> 8.25 (13/24/25, a small dip), Gemini 8.57 -> 8.52 (16/27/19).

**What was tried, and why it ended here** (each measured, same corpus): a visible-space symbol with a long rule list worked (OpenAI 96%) but made Claude write its reasoning into the suggestion in 4-7% of calls; a shorter version of it lost the effect (OpenAI 47%, Claude 12%); `<w>/<c>` tags fixed Claude but made gpt-4o-mini treat half of the mid-word fragments as finished words (50%); `<new>/<cont>` with explicit fragment guidance and a few complete-word examples balanced both.

**Not measured:** DeepSeek (no key; it falls back to the old reading if it ignores the tags), real keystroke traffic (the mix of mid-word and complete-word pauses), contexts with text after the cursor, non-English and code or Markdown contexts, and the frontend (if it already repaired spacing client-side, the visible gain is smaller). Fixed here for good: the server-side defects. Rollback is a code revert of the prompt and `_postprocess_autocomplete`; there is no env switch.

Tests: `tests/core/test_autocomplete_join.py` (20; with the new code removed piece by piece, each piece makes tests fail). The measurement script (`autocomplete_spacing_eval.py`, about $0.05 per repeat) is local only; see [MODEL_UPGRADE_EVAL_SCRIPTS.md](MODEL_UPGRADE_EVAL_SCRIPTS.md). Full suite: 2,685 tests.


---

## 17. Final live check before committing (2026-11-12)

The question: with everything in this change in place, do the new models work through the real code, and what do latency and cost look like next to what production ran before?

**Method.** Every flow goes through the function the API calls (`process_user_query_stream`, `get_autocomplete_suggestion`, the code-intent classifier, the design canvas harness, the OpenAI image call). Two configurations ran **at the same time** (same network, same minutes), three repeats each (two blog posts):
- **before** = the model IDs production ran until this change, set with the `MODEL_<TIER>_<PROVIDER>` env overrides (which is also the rollback path), with correctly shaped requests. For Claude this is the old models at their best: at the committed HEAD, Opus 5 and Sonnet 5 answered 400 to `temperature` (§10), so the real "before" for Claude was fallback to another provider.
- **after** = the shipped defaults.

Usage comes from the provider's response (request trace for pipeline calls, a wrapper on the SDK client for the direct autocomplete calls). No call used a fallback provider. The scripts are local (see [MODEL_UPGRADE_EVAL_SCRIPTS.md](MODEL_UPGRADE_EVAL_SCRIPTS.md)); raw results are in `local_run/model_upgrade/final_*.json`.

**Prices used (USD per million tokens, input / output), from the providers' pricing pages read on 2026-11-12:** Claude Opus 5.5 4/20, Opus 5 5/25, Sonnet 5.5 and Sonnet 5 2/10, Haiku 5.5 0.10/0.50 (prompts up to 100K tokens), Haiku 4.5 1/5; OpenAI gpt-6.1-terra 1/5, gpt-6.1-sol 2/10, gpt-5.2 1.75/14, gpt-4o-mini 0.15/0.60; gpt-image-2.5-flare 5 text input / 30 image output; Gemini 3.8 Flash 0.75/3.75 and 3.5 Flash-Lite 0.30/2.50. Caching is not credited, so costs are an upper bound. Reasoning and thinking tokens are inside the output counts the APIs report.

### Results (medians; cost is the mean per call, shown per 1,000 calls)

| Flow | Provider | Models, before -> after | Succeeded (before / after) | Total time p50, before -> after | First text p50 | $ per 1,000, before -> after |
|---|---|---|---|---|---|---|
| Autocomplete | Claude | Haiku 4.5 -> **5.5** | 30/30, 30/30 | 0.85 -> **1.26 s** (p95 1.25 -> 1.99) | - | $0.80 -> **$0.11** (-86%) |
| Autocomplete | OpenAI | gpt-4o-mini (unchanged) | 30/30, 30/30 | 0.88 -> 0.80 s | - | $0.11 -> $0.11 |
| Autocomplete | Gemini | 3.5 Flash-Lite (unchanged) | 30/30, 30/30 | 1.09 -> 1.15 s | - | $0.24 -> $0.24 |
| Code-intent classifier | Claude | Haiku 4.5 -> **5.5** | 20/20, 20/20 | 0.99 -> **1.41 s** | - | $0.06 -> $0.01; **correct answers 40% -> 90%** |
| Quick edits (grammar, simplify, shorter, summarize) | Claude | Sonnet 5 -> **5.5** | 24/24, 24/24 | 2.10 -> 2.11 s | 1.55 -> 1.34 s | $1.83 -> $1.60 (-12%) |
| Writing (enhance x2, make longer) | Claude | Opus 5 -> **5.5** | 9/9, 9/9 | 4.90 -> **4.07 s** (p95 17.1 -> 9.2) | 2.18 -> 1.80 s | $11.93 -> **$7.69** (-36%) |
| Writing | OpenAI | gpt-5.2 -> **terra** | 9/9, 9/9 | 3.26 -> **2.62 s** | 1.04 -> 1.25 s | $2.77 -> **$0.99** (-64%) |
| Code | Claude | Opus 5 -> **5.5** | 6/6, 6/6 | **121.4 -> 16.2 s** (p95 193 -> 20.9) | **36.7 -> 1.7 s** | **$287.47 -> $42.27** (-85%) |
| Code | OpenAI | gpt-5.2 (pinned, unchanged) | 6/6, 6/6 | 27.7 -> 22.7 s | 1.28 -> 1.05 s | $31.24 -> $29.82 |
| Design canvas | Claude | Opus 5 -> **5.5** | 6/6, 6/6 | 24.2 -> **16.8 s** (p95 48.1 -> 28.2) | - | $120.33 -> **$84.28** (-30%) |
| Design canvas | OpenAI | gpt-5.2 -> **Sol (low)** | **5/6**, 6/6 | 17.3 -> **11.6 s** (p95 30.4 -> 13.0) | - | $35.83 -> **$15.95** (-55%) |
| Blog (about 1,650 words; 2 posts) | Claude | Opus 5 -> **5.5** | 2/2, 2/2 | 54.1 -> **29.8 s** | 9.5 -> 4.5 s | $189.79 -> **$120.78** (-36%) |
| Blog | OpenAI | gpt-5.2 (pinned, unchanged) | 2/2, 2/2 | 43.8 -> 42.1 s | 1.73 -> 1.88 s | $89.24 -> $87.97 |
| Image (medium, 1024x1024) | OpenAI | `gpt-image-1` is withdrawn -> **gpt-image-2.5-flare** | -, 6/6 | - -> 10.6 s (p95 13.3) | - | **$13.36 per 1,000 images** (about $0.0134 each; 439 output tokens) |

Gemini (unchanged models, writing and code) is in `local_run/model_upgrade/final_*.json`; it served as a noise check, see below.

### Reading it correctly

- **Cost is solid; small latency differences are not.** Token counts and prices are exact. For latency, the models that did **not** change show the run-to-run noise: Gemini code went 14.5 -> 22.7 s (+56%) and OpenAI code 27.7 -> 22.7 s (-18%) with nothing changed, because answer length varies. So treat gaps under about 25% on 6 samples as "about the same". The large ones (Claude code, Claude blog, OpenAI design) are far outside that.
- **Claude code is the big win** (121 s -> 16 s, and the first text arrives after 1.7 s instead of 36.7 s) mostly because Opus 5 at its default effort spent about 11,000 output tokens (mostly thinking) per answer against 2,100. By language: Python 61.5 s -> 11.0 s, TypeScript 178.7 s -> 20.5 s. **All three Opus 5 TypeScript calls ended at exactly 16,384 output tokens, the output cap**, with 72-90 s before any text; the run counted them as answered because text arrived, but the stop reason was not recorded, so those answers may be cut off (one was 263 words against about 700 for Opus 5.5). The "before" cost is therefore capped, not a full-length answer.
- **The old Haiku 4.5 classifier answered "code" to every "other" query** (0 of 10 correct on those, 8 of 10 on the code queries because two replies carried a trailing code fence); Haiku 5.5 got 8 of 10 and 10 of 10. In production only queries the regex router is unsure about reach the classifier.
- **Claude autocomplete and the classifier are about 0.4 s slower** on Haiku 5.5 (controls moved less than 10%, so this is real), in exchange for 86-89% lower cost and, for the classifier, 40% -> 90% correct. The same prompt is also about 40% more input tokens on Haiku 5.5 (1,010 vs 719), already counted in the cost.
- **Design canvas on OpenAI**: gpt-5.2 failed one of six requests (schema validation after the maximum number of repair rounds); Sol succeeded in 6/6 with a 13 s worst case against 56 s.
- **Fallbacks: none.** Every call was answered by the provider and model it asked for.

### What got shorter (a quality trade-off, not measured by a judge in this run)

| Task | Words, before -> after |
|---|---|
| Enhance (paragraph / email), Claude | 105 -> 100 / 129 -> 123 (about the same) |
| Make longer, Claude | 419 -> 315 (-25%; the target range for that prompt is 200-400, still inside it) |
| Make longer, OpenAI | 113 -> 66 (-42%; the known terra trade-off from §14) |
| Code, Claude (Python CSV prompt / TypeScript prompt) | 1,446 -> 387 / 554 -> 710 |
| Code, Claude, from the saved answers of the earlier effort eval | median 12,758 -> 4,682 characters; explanatory comment lines per answer about 41 -> 13; usage example in 5/8 vs 10/16 answers (same rate) |

Opus 5.5 at effort low writes much more compact code, with fewer explanatory comments, than Opus 5 at its default effort. The judge scored it higher (§12: 8.2 vs 6.7), but the judge does not see "fewer comments". If fuller, more commented code is wanted, the lever is the code prompt, not a higher effort (high was slower and twice broke the "only code blocks" rule). Blog length is unchanged (Claude 1,648 -> 1,663 words, OpenAI 1,764 -> 1,836).

### A bug this run found, now fixed

The first smoke run showed the OpenAI blog writer reported as `gpt-6.1-terra` in the cost record, the trace and the `__model_used__` sent to the client, while the pinned `gpt-5.2` actually wrote it (`src/longform/llm.py` `model_id_for` ignored the pin). Cost attribution and the model label shown to users were wrong. Fixed: `model_id_for` takes the agent type; 3 call sites updated; two new tests in `tests/core/test_openai_tier.py`; 7 tests fail with the fix removed; 6 existing blog assertions that had the wrong label written in were corrected. The full runs above show `gpt-5.2` in the blog cost rows.

### What this means for the bill

- Per 1,000 requests of the same kind, the advanced Claude flows cost 30-85% less, OpenAI writing 64% less and OpenAI design 55% less; the pinned OpenAI blog and code are unchanged; Claude autocomplete is 86% cheaper, OpenAI and Gemini autocomplete are unchanged.
- Nothing got more expensive per call. The only price step coming is outside this change: **Gemini 3.8 Flash doubles on 2027-01-01** ($0.75/$3.75 -> $1.50/$7.50 per million), which doubles its rows above.
- This run cost about **$6.25 at list prices** (old config $3.73, new config $1.72, smoke test $0.79).
- The app's own cost estimate (`TEXT_PRICES_PER_MTOK`) is empty in `src/models.py`, so it still reports "unknown" for every model; set it in `.env` with the prices above if you want per-request cost in the logs.

### Not covered

No quality judge in this run (§4 and §12-§16 have it), no concurrent load, no Grok or DeepSeek, no real traffic mix (to turn per-1,000 numbers into a monthly figure, weight them with the request mix from `REQUEST_LOG_FILE`), no prompt-cache discounts, no fast mode or `service_tier` (a 2x price choice, §6), and the streaming UI was not exercised. Samples are small (6 per cell, 2 blog posts, 30 autocomplete calls).
