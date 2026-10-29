# Voice flow: improvement checklist

Sources: your repetition / "points could be better" feedback, the review plan (phases 1-6, review issue numbers in brackets), and the
scope cut "remove email, Slack, code and Markdown formats from the prompt for now".
`[x]` done and tested, `[-]` deferred on purpose. Verified with 373 voice tests, 1767 project tests, and the live Gemini eval
(53 cases, 100%; your long dictation 10/10 Clean and 5/5 Polished on repeated runs).

## A. Scope cut
- [x] A1 Email, Slack, code and Markdown formats are out of the prompt, few-shots, gate, post-processing, page and evals. They are kept behind `VOICE_APP_CONTEXTS` (default off); while off every `app_context` is treated as `general`, so clients can keep sending it.

## B. Repetition and better points
- [x] B1 Prompt rules: false starts, one lead-in only, restated sentences, filler sentence openers (keeping the rest of the sentence), greetings kept
- [x] B2 "Polished" style (short label + one tight sentence per point): `style` in the `context` part of the upload, `VOICE_DEFAULT_STYLE`, selector on the page
- [x] B3 Claim-retention check: Polished must keep hyphenated terms and numbers, Clean must keep acronyms and hyphenated terms; one repair that names what was lost, then Clean
- [x] B4 Long-form eval cases (`lf-01`..`lf-04`) from your dictation: item count = spoken cues, no stacked lead-in, key terms kept, closing line outside the list

## C. Phase 1: one shared registry
- [x] C1 `formatter/patterns.py`: acronyms, proper nouns, days and months (generated, so March/August exist), question starters, spoken commands, greeting and sign-off [3.3 4.1 4.4 6.1 6.2]
- [x] C2 rules.py uses it; "when does...", "where is...", "do you...", "have you..." end in "?", while "Do the dishes" stays a statement [4.3]
- [x] C3 Homophone numbers ("to", "too", "for") count only after a list marker and only as the next number; `step` / `item` markers [4.2]
- [x] C4 Stutter guard: no collapsing across commas or intentional repeats ("Yes, yes", "very very") [4.5]
- [x] C5 vocabulary.py is language-aware: structural seed and hallucination list per language; `language` is a per-request setting [8.1 8.2]

## D. Phase 2: schemas
- [x] D1 CursorContext: 150 / 150 characters, longer windows auto-trimmed, limits documented [9.2]
- [x] D2 `ContextUpdateFrame`: send only what changed (cursor, dictionary, language, style) [7.2]
- [-] D3 AppContext docstrings are written; the `code_editor` split [1.4 9.1] is deferred with the app formats

## E. Phase 3: prompt
- [x] E1 Capitalization and numbers are rules, not a product list; `<dictionary>` overrides everything [1.1 1.6]
- [x] E2 `following_text` example; `neutralize` on every turn; backend-neutral roles with Gemini / OpenAI adapters [2.2 2.3 2.4]
- [-] E3 Per-context rule rewrite and the Markdown / Slack examples [1.2 1.3 1.5 2.1]: written, kept in the file, off with the app formats

## F. Phase 4: gate, post-processor, engine
- [x] F1 Gate: `MAX_FAST_WORDS` explained; shared spoken-command pattern (colon, brackets, slash ...) [3.2 3.3]
- [x] F2 `_close_sentence` never closes list items, headings or code [6.3]
- [x] F3 Backend-neutral names in the circuit-breaker logs [5.1]
- [x] F4 `check_output`: homophone and contraction tolerant, chatbot markers, higher thresholds, no novelty check on short outputs [5.2]
- [x] F5 `_max_output_tokens` floor of 512 [5.3]
- [x] F6 OpenAI: a model that rejects `temperature` is learned from the 400 and remembered [5.4]
- [-] F7 Slack and code period rules [3.1 6.3 context part]: deferred with the app formats. Sign-off title-casing is done (patterns + post-processor) [6.1]

## G. Phase 5: router and audio
- [x] G1 `_verify_bearer_token` (constant-time, UTF-8) [7.1]
- [x] G2 An empty or bounced utterance never reaches Whisper; at most 8 dictations are in flight and the next gets 429 [7.3 7.4] (was a per-connection queue answering `rate_limited` until the WebSocket was replaced by `POST /dictate`)
- [x] G3 Anti-aliased resampling [8.3]

## H. Verification
- [x] H1 `tests/voice_flow/test_regressions.py`: every in-scope row of the review's matrix, plus the new behaviour
- [x] H2 Whole project suite green
- [x] H3 Live Gemini eval, long-dictation runs, docs and memory

## Where I deviated from the plan, and why
- **Do / have / will / shall as question starters:** only when a subject follows ("do you", "have we"). Added blindly, "Do the dishes before dinner" became a question. `Won't` in the plan's set was capitalised and could never match.
- **Months:** generated from `calendar`, but "may" is capitalised only where a month is meant ("in may", "may 15"); otherwise every "it may rain" would become "May".
- **Proper nouns:** `bun`, `linear` and `remix` left out of the list: they are ordinary words.
- **Sign-off pattern:** the comma stays required. With it optional, "that is the best idea" read as the sign-off "Best, idea".
- **`_close_sentence`:** kept the existing following-text rule. The plan's version would drop the period before a *new* capitalised sentence ("the logs. Then do X").
- **Resampling:** PyAV (already installed for decoding) instead of scipy: same anti-aliasing, no new dependency. Measured: a 12 kHz tone folded to 4 kHz at 0.351 amplitude with linear interpolation, 0.0000 now.
- **Empty utterances:** answered without calling Whisper, but still through the queue so results keep the order of the "end" frames. The queue accepts 9 (1 working + 8 waiting) and refuses the 10th, not the 9th.
- **Cursor window:** following text is now 150 characters as in the plan (it was 50).

## Added because the live runs asked for it
- **Count the cues in code** (skipping a number is allowed: 1, 2, 3, 5), tell the model `<list_items>N</list_items>`, verify the count, and retry once with the exact correction. Before: your long dictation passed about half the runs.
- "point number one" is one cue (the word "point" used to be left behind in the text) and the copula after a cue ("is", "would be") is dropped in the fallback.
- A lead-in nobody said ("Here are the tasks:") is removed in code and forbidden in the prompt; the model path lowercases after a mid-sentence caret, as the rules path always did.
- The model's stray `</output>` and `<list_items>` tags are stripped.
- Polished repair names exactly what was lost; Clean also checks acronyms, hyphenated terms and an opening greeting.
