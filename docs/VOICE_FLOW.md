# Voice dictation (src/voice_flow)

Speak, release a key, get formatted text at the caret: lists, self-corrections, fillers, capitalization and spacing handled.
Local Whisper for speech, a low-end Gemini model for formatting, code for the last mile. Ships **off** (`VOICE_FLOW_ENABLED`).

```
mic -> PCM16 16 kHz -> faster-whisper (local, initial_prompt carries the list cues)
    -> gate: a plain sentence takes the FAST PATH (rules only, no model call); cues/corrections/fillers/email/code go on
    -> Gemini flash-lite (XML-wrapped transcript, 9 worked examples, caret context)
    -> guards (empty / preamble / too long / dropped content / answered instead of formatting / list contradicts the cues / echoed context)
    -> PostProcessor (spacing, list tidy, lead-in colon, final period, caret stitching)   model down or rejected: rule_format()
```

## Try it with your voice

```bash
cd /Users/codecords/Desktop/Codebase/Agent/agent && .venv/bin/python scripts/voice_demo.py
```

It loads Whisper, checks the formatter, and opens **http://127.0.0.1:8000/v1/voice/demo** in your default browser. Options: `--gpt` (format with
GPT instead of Gemini), `--whisper base` (faster, less accurate), `--always-model` (no fast path), `--no-open`, `--port N`. Ctrl+C stops it.
(The desktop app has the same thing as the `voice-demo` launch config. The `agent-api-local` config runs the whole API with voice enabled.)

1. Use a normal browser (Chrome, Edge, Safari, Firefox) at `127.0.0.1` or `localhost`. The Claude desktop app's built-in browser pane blocks microphones, and other `http://` addresses are refused by browsers.
2. Click **Enable microphone** and choose Allow (once).
3. Click in the text box where text should go, then hold the mic button (or the **Ctrl** key), speak, and release.
4. The page shows what Whisper heard, the exact characters inserted, and the Whisper / model / total milliseconds. The page also lists sentences to try and what each should produce.

Gemini is the default formatter (`DEFAULT_GEMINI_API_KEY`, or `VOICE_GEMINI_API_KEY` for a different key). Without a working key, dictation still
runs on the rule-based formatter and the page says so. The startup output and `GET /v1/voice/status` show which formatter is live.

## Settings (config.py, all env vars)

| Variable | Default | Notes |
|---|---|---|
| `VOICE_FLOW_ENABLED` | `false` | mounts `/v1/voice/*` and warms the models at startup |
| `VOICE_WHISPER_MODEL` | `small` | `base` is ~2x faster and less accurate on real speech; `distil-large-v3` is 4 s+ on CPU |
| `VOICE_WHISPER_COMPUTE_TYPE` / `_THREADS` | `float32` / `8` | measured best on an Apple M4; use `int8` on x86 servers |
| `VOICE_DEFAULT_STYLE` | `clean` | `clean` keeps your wording (fillers, false starts and repeats removed); `polished` also tightens each list point to "Label: one sentence". Per request: `style`. Polished costs an extra model call when it has to repair a dropped claim |
| `VOICE_APP_CONTEXTS` | `false` | email / Slack / code / Markdown formats. Off: every `app_context` is treated as `general` and the prompt has no rules for them |
| `VOICE_LLM_MODE` | `smart` | `smart`: plain sentences skip the model (about 1 s saved); `always`: every utterance goes to the model |
| `VOICE_FORMAT_PROVIDER` | `gemini` | `gemini` or `openai` |
| `VOICE_FORMAT_MODEL` | provider's low-end model (`gemini-3.5-flash-lite` / `gpt-4o-mini`) | `src/models.py` `AUTOCOMPLETE_MODEL_IDS` |
| `VOICE_GEMINI_API_KEY` | falls back to `DEFAULT_GEMINI_API_KEY` | OpenAI uses `DEFAULT_OPENAI_API_KEY` |
| `VOICE_FORMAT_TIMEOUT_SECONDS` | `4` | on timeout the rule-based result is used |
| `VOICE_ENDPOINT_SILENCE_MS` | `6000` | hands-free mode: this much silence finishes the dictation. The page lets you pick 3 / 6 / 12 s or never (click Stop). 800 ms split a 2-minute monologue into nine separately formatted pieces |
| `VOICE_MAX_UTTERANCE_SECONDS` | `600` | memory guard, not a chunk size; if reached, the client gets a `notice` frame before the result |
| `VOICE_FLOW_TOKEN` | empty | non-empty: clients send it (`token` in the start frame, `X-Voice-Token` on REST) |

## Interfaces

* `POST /v1/voice/format` raw transcript + `cursor_context` -> `formatted_text` (already contains the leading/trailing space the caret needs).
* `POST /v1/voice/dictate` base64 audio (wav, webm/opus, m4a ...) -> the same, with Whisper timing.
* `WS /v1/voice/stream` see the docstring of `src/voice_flow/router.py`. Push-to-talk (`start`, binary PCM16 frames, `end`) or hands-free (`mode: "auto"`, `silence_ms`: how much silence finishes it, 0 = only `end`). **One dictation gives one `final` frame, however long it is and however many pauses it holds**: Whisper decodes the whole recording and the formatter sees the whole text once, so lists and context are never cut in two.

## Measured (Apple M4, 16 GB; synthetic `say` speech, so real voices will be harder)

| Stage | Result |
|---|---|
| Whisper `small`, float32 | 490-1050 ms, nearly flat in utterance length (`base`: 170-450 ms) |
| Gemini `gemini-3.5-flash-lite` | **1.0-1.3 s per call, and that is the API's floor**: a one-word prompt takes 1.05 s, the full prompt 1.16 s. Prompt size, thinking level and the other lite models (`3.1-flash-lite`, `flash-lite-latest`) change it by about 100 ms |
| Post-processor | 0.01-0.05 ms |
| **End to end after you stop speaking** (20 spoken cases, smart mode) | fast path **485-620 ms** (6 of 20); model path **1.5-2.4 s** (14 of 20); p50 1654 ms, p95 2425 ms |

Accuracy, 61 golden cases (`python -m src.voice_flow.evals.run_evals`, model scored on every case): **Gemini 61/61**;
structural 43/43, questions never answered 8/8, zero spacing/list defects, p95 model latency 1329 ms (target 250 ms: not met).
That set was used to tune the prompt, so treat it as a development score. Other results on a 20-case sample: `gpt-4o-mini` 100% structural,
`gpt-5.4-mini` 87%, `gpt-4.1-mini` and `gpt-5.4-nano` 60%, `gpt-4.1-nano` unavailable (404). Rule-based fallback alone: 44/61.

The plan's targets (formatter p95 < 250 ms, end to end < 450 ms) cannot be met with any cloud model called per utterance: the
call alone is about a second. The fast path is what keeps ordinary dictation near Whisper's own time.

## How a long dictation with spoken points is handled

Code counts the spoken cues ("point number 1 ... point 2 ... point 3 ... point 5": four items, a skipped number is allowed), tells the model the
number, and checks the answer: the wrong count gets one retry with the exact correction. An invented lead-in ("Here are the tasks:") is removed,
a counted list returned as bullets is renumbered, and the model's answer is checked for dropped acronyms, hyphenated terms and an opening greeting
(one repair, then the plain text). Measured on the 255-word test dictation, Gemini, repeated runs: Clean 10/10 and 8/8, Polished 5/5. Before these
checks the same text passed about half the runs.

## Known limits

* Hands-free endpointing is an energy gate (fans and keyboards can trip it); push-to-talk does not use it. A long dictation is decoded in one go, so the result arrives after you finish: about 5 s of Whisper plus 2 s of model for 90 s of speech (measured), nothing appears while you are talking.
* Utterances with list/correction/filler cues or an email greeting, and all code/Markdown contexts, pay the model's ~1 s. The gate errs towards the model ("wait", "point", "first" always go to it).
* Gemini often omits the final period and sometimes leaves a counted list unformatted when text precedes the caret; the post-processor closes sentences and the list guard sends such answers to the rules.
* No interim transcripts: Whisper is decoded once per utterance, because an interim decode would delay the final one.
* The rule-based fallback cannot resolve a non-numeric self-correction ("send it to Bob, make that Alice") and leaves it as spoken.
* The mic test page needs a browser microphone permission; it was exercised here only through its WebSocket path.
