# Document context live provider results

Verified on 3 November 2026 using synthetic documents and the configured model IDs. OpenAI, Gemini and Grok passed all Ask/Draft checks. An intermittent Gemini Edit failure exposed a JSON-formatting compatibility issue; after the fix, all 18 repeated Edit requests passed across those three providers. Claude could not be tested because its configured credential returned HTTP 401.

## Accuracy and preservation

Each provider answered four document questions and generated two drafts. All **18 Ask/Draft cases passed**. The checks covered the invented codename Juniper Lantern, lead Mira Vale, cohort size 36 and duration 21 days; an unspecified renewal price; differing cohort counts in separate sections; and a quoted instruction attack. Draft checks required two new sentences, correct names, no new figures, no repetition of the existing after-cursor text, and correct full/partial context metadata. Responses were also read to check their factual fit and whether new passages worked as suggested continuations.

The first mixed run passed 23 of 24 cases. Gemini's explicit Edit returned the original source with `status="failed"`; a focused repetition reproduced this behavior. Captured provider output showed valid planner JSON wrapped in a Markdown code fence. The edit service previously passed that entire string directly to `model_validate_json()`, which rejected the wrapper before any rewriting.

The fix reuses the existing JSON parser already used by email and blog flows, then applies the same strict `EditPlan` schema, allowed-target checks and instruction-quote checks. It does not relax replacement, protected-content, source-hash or patch-offset validation. Regression tests accept fenced JSON and continue rejecting unknown schema fields before rewriting.

After the fix, each provider completed three explicit Edit and three legacy `document_text` Edit cases: **18 of 18 passed**. Gemini wrapped five of its six post-fix plans in fences; all five were handled successfully. Every result changed only the Introduction, removed the duplicated words, preserved all untargeted source regions exactly, replayed through `/agent/documents/apply`, and rejected an intervening source change with HTTP 409.

The evidence retains the two pre-fix Gemini failures. Across the initial run, diagnostic follow-ups and post-fix repetitions, 47 live cases were attempted and 45 passed; both failures occurred before the parser fix and preserved the source unchanged. These small synthetic samples establish the tested behaviors, not general model accuracy or resistance to arbitrary injection attacks.

## Measured latency

Times below are observed medians in seconds, shown as **first visible content / complete response**. Each provider has four Ask samples, two Draft samples and six post-fix Edit samples. Ask/Draft values come from the initial mixed run; Edit values come from the repeated verification after the fix. Model reasoning, cold connections and provider variability are included. These are small samples rather than production p95 measurements.

| Provider and configured model | Ask | Draft | Edit |
| --- | --- | --- | --- |
| OpenAI `gpt-5.2` | 1.48 / 2.12 | 1.36 / 1.90 | 4.02 / 4.02 |
| Gemini `gemini-3.8-flash` | 2.17 / 2.23 | 3.21 / 3.49 | 5.86 / 5.86 |
| Grok `grok-4.6` | 2.79 / 3.16 | 16.68 / 16.70 | 13.77 / 13.77 |
| Claude `claude-opus-5` | Blocked by HTTP 401 | Blocked | Blocked |

Edit emits a single complete verified snapshot, so its first-content time is close to completion time. Header arrival and metadata/progress events were not counted as first answer text. Grok was substantially slower on drafting and editing in this sample; no latency optimization or provider substitution was applied to conceal that result.

## Test path and evidence

The evaluator starts the actual FastAPI application on an ephemeral loopback port and makes real HTTP/SSE requests using the real provider SDKs. Only license/key resolution is replaced with a local synthetic-license resolver; provider calls are not mocked. Provider fallback is disabled to attribute results to the requested model. Redis, production licensing, storage and billing webhooks are not contacted. Real model-provider usage is recorded from request traces.

Provider model-list probes accepted the configured OpenAI, Grok and Gemini model IDs. Gemini reported an input-token limit of 1,048,576 for its configured model. OpenAI/Grok model-list responses did not provide a comparable capacity field; this run does not certify their maximum context limits. The conservative context allocation remains unchanged.

- [Initial requests, answers, checks and timings](../local_run/document_context_live/report.json)
- [First Gemini diagnostic follow-up](../local_run/document_context_live_gemini_followup/report.json)
- [Reproduction with the captured fenced planner reply](../local_run/document_context_live_gemini_repeats/report.json)
- [Post-fix repeated explicit and legacy Edit verification](../local_run/document_context_live_edit_verified/report.json)
- [Reproducible evaluator](../scripts/evaluate_document_context.py)

The reports contain only synthetic source text, model replies, timings and usage. Provider credentials are not written to them. Full regression verification after the fix and enabled-default change passed **2,074 tests and 1,661 subtests**.

## Enabled configuration and production state

`DOCUMENT_CONTEXT_ENABLED` now defaults to **true**, and the locally resolved configuration is enabled. An explicit `false` environment override still acts as the existing operator kill switch; a deployed service should retain `DOCUMENT_CONTEXT_ENABLED=true` if its environment specifies this setting.

No production hostname, URL or deployment connection was available in this checkout, so the deployed flag and running revision were not read or changed. Enabling the release configuration is not evidence that a production service has reloaded it. Production traffic latency, real browser/editor integration and upstream billing behavior were not measured by this loopback/provider evaluation.

To repeat the live matrix, run `.venv/bin/python scripts/evaluate_document_context.py`. For the edit verification, run `.venv/bin/python scripts/evaluate_document_context.py --providers openai grok gemini --cases edit_explicit edit_legacy --repeats 3 --output local_run/document_context_live_edit_verified`. These commands use the configured provider keys and incur model-provider usage.

## Backend changes 5–7 live check

A follow-up run on 3 November 2026 used the same real HTTP/SSE loopback harness and configured OpenAI
`gpt-5.2`, with provider fallback disabled. All **five checks passed**. These are individual observations,
not medians or production percentiles.

| Synthetic case | First content / complete (seconds) | Verified behavior |
| --- | --- | --- |
| Scoped large-document Ask | 1.115 / 1.247 | Retrieved the distant cobalt inventory, answered 91 units and cited `[r0]` |
| Full-source summary | 6.736 / 6.737 | Correct Juniper Lantern and Mira Vale, exact `[e0]`/`[e1]` quotes, all source chunks read |
| Full-source contradictions | 6.943 / 6.944 | Detected 36 versus 48 for the same pilot period across distant sections, exact citations |
| Advisory intent | 0.941 / 2.206 | “What would you rewrite?” stayed Ask with no patch proposal |
| Explicit style-aware Edit | 5.423 / 5.425 | Removed target repetition only, preserved other source, verified apply and stale-source 409 |

The summary used five successful stages, 10,703 dynamic input characters and 818 raw generated output
characters. Contradiction analysis used five stages, 11,197 input characters and 916 output characters.
Both completed extraction, synthesis and original-passage verification without rejected findings.
Answers and references were inspected in addition to automated checks. This narrow synthetic run does
not establish exhaustive analysis, general style quality, new-stage compatibility with other providers
or production latency.

Evidence: [backend requests, answers, checks and timings](../local_run/document_context_backend_live/report.json).
Repeat with `.venv/bin/python scripts/evaluate_document_context.py --providers openai --cases ask_retrieval analysis_summary analysis_contradictions ask_intent_advisory edit_explicit --output local_run/document_context_backend_live`.
Production configuration/revision and browser/upstream integration were not contacted; the enabled
release default remains unchanged.

Final backend regression verification passed **2,144 tests and 1,662 subtests**; the focused backend/edit run passed **142 tests**.

## Gemini and Grok backend verification

A further live run on 3 November 2026 exercised configured Gemini `gemini-3.8-flash` and Grok
`grok-4.6` through the real HTTP/SSE application and provider SDKs, with fallback disabled.
Each provider completed two repetitions of seven cases: **28 responses verified**. No backend or
provider configuration change was needed. The only code changes in this follow-up improved the synthetic
evaluator and its tests.

Accuracy and preservation checks:

- Scoped retrieval found the distant cobalt inventory and correctly answered 91 units with a valid `[r0]` citation.
- Full-source summaries correctly identified Juniper Lantern and Mira Vale. Every cited quote and UTF-16 range matched the exact source.
- Same-period contradictions correctly paired 36 and 48 participants from distant sections. Each summary/positive contradiction completed three extraction calls, synthesis and original-passage verification.
- The negative control placed 48 participants in a separate future pilot period. All four responses correctly returned `no_findings`, full coverage and no invented citations. Each used three extraction calls and synthesis; verification was unnecessary with no candidates.
- All four dedicated style rewrites turned the 20-word passive Introduction into short, direct imperatives. Examples: Gemini “Prepare for the session with care.” and Grok “Get ready for the session with care.” Their actual writer calls received two untargeted style samples. Human review confirmed the intended meaning and style fit.
- All **12 edit requests** (dedicated style, explicit repetition removal and legacy `document_text`) changed only the Introduction, preserved all other source exactly, replayed successfully through `/agent/documents/apply`, and rejected a stale source with HTTP 409. Legacy writer calls received zero style samples; new explicit Edit calls received samples.

The initial keyword-based evaluator scored 21/24 main cases as passing. Three accurate contradiction
answers used “one source, but … another” or “both 36 … and 48” without an explicit “conflict” keyword,
so the scorer incorrectly rejected them. The scorer now accepts these direct comparisons, with regression
tests for the observed wording. Saved answers were inspected and regraded: **24/24 main cases passed**;
all **4/4 negative controls** passed on their initial run. The raw reports remain unchanged, and the
reviewed report retains each original score/check set. No provider failure or incorrect answer was relabeled.

Times below are medians of two observations per case and provider, in seconds as **first visible
content / complete response**. The complete-response ranges across all cases were 2.736–12.159 seconds
for Gemini and 2.253–29.964 seconds for Grok. Two repetitions do not establish production p95 latency.

| Case | Gemini | Grok |
| --- | --- | --- |
| Scoped retrieval | 2.995 / 2.999 | 2.454 / 2.463 |
| Full-source summary | 10.595 / 10.596 | 22.123 / 22.124 |
| Same-period contradiction | 11.524 / 11.525 | 28.579 / 28.581 |
| Separate-period negative control | 8.466 / 8.467 | 25.078 / 25.078 |
| Dedicated style-aware Edit | 8.027 / 8.027 | 18.614 / 18.615 |
| Explicit repetition-removal Edit | 6.321 / 6.322 | 13.100 / 13.101 |
| Legacy document_text Edit | 5.724 / 5.726 | 10.284 / 10.285 |

Analysis waits for evidence verification before emitting findings; edit waits for patch validation.
Their first-content and completion times are therefore nearly identical. These results show substantially
slower Grok analysis/editing in this sample; no faster model or provider substitution was used.

Evidence and repeat commands:

- [Main raw requests, answers, checks and timings](../local_run/document_context_backend_gemini_grok/report.json)
- [Reviewed main results, preserving original scores](../local_run/document_context_backend_gemini_grok/reviewed_report.json)
- [Separate-period negative-control results](../local_run/document_context_backend_gemini_grok_negative/report.json)

```sh
.venv/bin/python scripts/evaluate_document_context.py --providers gemini grok --cases ask_retrieval analysis_summary analysis_contradictions edit_style edit_explicit edit_legacy --repeats 2 --output local_run/document_context_backend_gemini_grok
.venv/bin/python scripts/evaluate_document_context.py --providers gemini grok --cases analysis_separate_periods --repeats 2 --output local_run/document_context_backend_gemini_grok_negative
```

The evaluator/source-range grading and backend/edit regression run passed **154 tests**. The feature's
configured default and locally resolved `DOCUMENT_CONTEXT_ENABLED` remain **true**. Production was not
contacted or reconfigured: no deployment connection is available here. This is synthetic provider
verification, not a measurement of real editor behavior, production licensing/billing, exhaustive
semantic accuracy or deployed latency.
