# Model upgrade: the measurement scripts, and why they are not in the repository

The November 2026 model upgrade (report: [MODEL_UPGRADE_BENCHMARK.md](MODEL_UPGRADE_BENCHMARK.md), rollout: [ROLLOUT_RUNBOOK.md](ROLLOUT_RUNBOOK.md))
was measured with 18 one-off Python scripts (about 2,300 lines). They were deliberately **not committed**. This page records what each one did, what it found
and how to rebuild the useful ones, so the knowledge outlives the files.

Where they are now: `scripts/model_upgrade/` on the machine that ran them (untracked) and their raw results in `local_run/model_upgrade/`
(gitignored, about 13 MB, answers included). Nothing in `src/`, `tests/`, `api.py` or `config.py` imports them.

## Why they are not committed

| Reason | Detail |
|---|---|
| They are tied to this round | Model IDs, prices and corpora are written into them (gpt-5.2 vs terra, Opus 5 vs 5.5, ...). The next model change makes them wrong, not just old. |
| They cost money and need keys | Each run spends real API credit (cents to a few dollars) with the platform keys. They cannot run in CI and should not run by accident. |
| Their data is not in the repo | `local_run/` is gitignored, so a committed script would reproduce numbers nobody can compare against. |
| What lasts is kept elsewhere | The method and every number: `MODEL_UPGRADE_BENCHMARK.md`. The things the app runs: `src/provider_health.py`, `src/model_preflight.py`, `scripts/preflight_models.py`, and 154 new tests (in 10 new test files) that pin each behaviour found (temperature, effort, thinking blocks, pins, fenced JSON, spacing). |

Rule used: commit what the app executes or what an operator runs in production; keep comparison harnesses local and document them.

## What each script did

Cost is the approximate API spend of one full run.

### Do the parameters work? (answers "is the pasted guide right?")

| Script | What it did | Cost | Outcome (report section) |
|---|---|---|---|
| `probe_openai.py` | Sent one minimal request per model and request shape (a few hundred tiny calls) and recorded accept/reject. | cents | Per-model parameter matrix: gpt-6.x rejects `max_tokens`; temperature only at effort `none`; `max` effort is Responses-only; function tools need effort `none` (§2, §3). |
| `probe_images.py` | The same for the image models (~90 low-quality images). | $1-2 | `gpt-image-1` is gone; `input_fidelity` only on 1.5; gpt-image-2 rejects transparent backgrounds; flexible sizes on 2 and newer (§2, §14). |
| `smoke_integration.py` | Swapped only the model ID in `src/models.py` in memory and ran the app's real functions. | cents | Reproduced what breaks on an ID-only swap: 400s on `max_tokens` / `temperature`, tools on Chat Completions (§3). |

### Before / after on the OpenAI side

| Script | What it did | Cost | Outcome |
|---|---|---|---|
| `bench_run.py` | Groups `writing`, `fast`, `autocomplete`, `intent` against several models, with latency, first token and usage. | $0.2-0.5 | §4.1-4.4 |
| `bench_design.py` | Design canvas through the real harness (draft, validate, repair, auto-heal), several models and efforts. | $0.3 | §4.5: Sol at effort low is faster than gpt-5.2. |
| `bench_images.py` | Seven image models, two prompts, medium quality 1024x1024, plus an HTML contact sheet. | $0.5 | §4.6 |
| `openai_code_eval.py` | The real code pipeline (`code_writer`, with validate and repair), 8 prompts, gpt-5.2 vs terra. | $0.3 | **Terra was 12.9 s -> 3.5 s median but answers were 3,632 -> 842 characters, usage examples in 4 of 8 answers instead of 7 of 8, and 1 explanatory comment line instead of 15. That is why code and blog stay pinned to gpt-5.2 (§14).** |

### Before / after on the Claude side

| Script | What it did | Cost | Outcome |
|---|---|---|---|
| `claude_suite.py` | Opus 5 -> 5.5, Sonnet 5 -> 5.5, Haiku 4.5 -> 5.5 on the same corpus and judge as OpenAI. Has free `--dry-run` (prints every request body) and `--preflight` (key and model IDs). | $2-4 | §4.7-4.9, §5 |
| `claude_extra.py` | Classifier strict vs lenient scoring, autocomplete judge, design canvas on Opus. | ~$1 | §4.9, §4.10 |
| `opus_effort_eval.py` | The real code pipeline with Opus 5.5 at effort low / medium / high, 8 prompts. | ~$1 | **Low: 14.4 s median, 2,105 output tokens, 0 format failures. Medium 18.4 s, no quality gain. High 28.7 s, 7,529 tokens, broke the "only code blocks" rule twice. Default `CLAUDE_ADVANCED_EFFORT=low` (§12).** |
| `sonnet_quick_edit_eval.py` | Real pipeline quick edits (grammar, simplify, shorter, summarize), Sonnet 5 vs 5.5 vs 5.5 at low effort. | ~$0.1 | §13 |

### Judging, analysis and shared code

| Script | What it did |
|---|---|
| `judge.py` | Blind pairwise quality judge (`gpt-5.5`), every pair in both orders, a candidate "wins" only if it wins or ties in both. Baseline is always today's model. |
| `judge_autocomplete_norm.py` | Re-judged autocomplete with join spacing normalised, and counted glued words ("toschedule"). |
| `analyze.py` | Turned the raw JSON into the report tables. No API calls. |
| `bench_common.py` | Shared helper for the benchmarks: the OpenAI call wrapper (first-token time, usage), the test corpus (writing, quick edits, autocomplete contexts, classifier queries) and the configs compared. |

### Fixes found along the way

| Script | What it did | Cost | Outcome |
|---|---|---|---|
| `autocomplete_spacing_eval.py` | 74 labelled contexts x OpenAI / Claude / Gemini x old vs new prompt, through `get_autocomplete_suggestion`. | ~$0.05 per repeat | Correct join: OpenAI 75% -> 97%, Claude 67% -> 99% (held-out 76% -> 98%, 70% -> 100%) (§16). |

### The final live run (evidence before committing)

| Script | What it did | Outcome |
|---|---|---|
| `final_live_run.py` | Ran every changed flow through the real production functions twice, once with the old model IDs (the `MODEL_<TIER>_<PROVIDER>` env overrides, which is also the rollback path) and once with the shipped defaults. Records first-text time, total time, tokens and the model that served each call. | §17. It also caught a real bug: blog and long-form calls were reported (cost record, trace, `__model_used__`) as `gpt-6.1-terra` while the pinned `gpt-5.2` wrote them. Fixed in `src/longform/llm.py` with tests. |
| `final_report.py` | Turned those results into tables and priced them from the providers' pricing pages. | §17 |

## How to repeat this next time (the recipe)

1. **Free first.** `python scripts/preflight_models.py`: key accepted, every configured ID exists.
2. **Probe, do not assume.** One minimal request per model and per parameter you send or might send. A cell is "ok" only if the API answered 200.
3. **ID-only swap.** Change only the model ID in memory and run the real functions; every 400 is a code change you need.
4. **Before/after on the real pipeline**, not on raw API calls: `process_user_query_stream`, `get_autocomplete_suggestion`, the design harness. Set the old IDs with `MODEL_ADVANCED_OPENAI=...` style env vars in one process and use the defaults in another, run both at the same time (same network, same minute), at least three repeats, and read usage from the provider response.
5. **Quality by blind pairwise judge**, both orders, a judge from a different vendor where you can, and look at the answers yourself for length, structure and anything the judge cannot see (usage examples, comments, tables).
6. **Price from the pricing page**, per call and per 1,000 calls, counting reasoning and thinking tokens as output. Remember that a new tokenizer changes the token count of the same text (Claude 4.7 and later: about 30% more).
7. **Write the findings into the report and the tests**; the scripts can be thrown away.

If you only rebuild one thing, rebuild step 4: it is what exposed the model-label bug that no unit test had.

## If you need the files

They are not in git history. They are on the disk of the machine that ran them, in `scripts/model_upgrade/`. To keep them out of `git status` without committing or ignoring them for everyone, add `scripts/model_upgrade/` to `.git/info/exclude` (local to that clone).
