# Frontend Handoff — Streaming Protocol, Voice Dictation, Design Canvas (+ deltas for the rest)

For: the Eddyter frontend engineer(s).
Backend: this agent service (`api.py`, `src/voice_flow`, `src/design_canvas`).
Verified: 2026-10-29 against HEAD `df292a7`, by reading the code paths named below.
Licence and billing are implemented in `src/voice_flow/access.py` and covered by `tests/voice_flow/test_access.py`, including production API wiring.
**Not exercised for this document:** live provider calls, a real browser microphone, the production deployment.
Where something is a recommendation rather than a fact in the code, it says so.

---

## 0. Read this first

### 0.1 Is there already a frontend guide?

Partly. There is no single guide, and two features have none at all.

| Feature | Existing document | Frontend-ready? | Covered here |
|---|---|---|---|
| Email templates | [`EMAIL_API_HANDOFF.md`](EMAIL_API_HANDOFF.md) | **Yes**, complete. Re-checked: its 15 block types and request fields still match the code | §8 (three small deltas) |
| Blog (`generate_blog`) | [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) § Visually rich blog | Contract yes, UI guidance no | §7 |
| Long-form refinement | `API_DOCUMENTATION.md` § Long-document refinement | Contract yes, UI guidance no | §6 |
| Code generation / task routing | `API_DOCUMENTATION.md` § Code generation and task routing | Contract yes, but scattered | §5 |
| **Voice dictation** | `VOICE_FLOW.md` (operator/tuning doc) and the docstring of `src/voice_flow/router.py` | **No**: nothing says what a browser must build | **§3** |
| **Design canvas** | **None.** `API_DOCUMENTATION.md` has two table rows. `gemini-code-1792761204580.md` is a pre-implementation spec that points at a `docs/design_canvas_plan.md` which does not exist | **No** | **§4** |
| SSE stream protocol | `agent.md` §11.2 | **Wrong**: do not use it (see §2.4) | **§2** |

`EMAIL_API_HANDOFF.md` is the model for what a handoff should be. This document follows it.

### 0.2 Corrections to the feature summary you were given

The summary table you received is right about *which features* need frontend work, but several specifics are wrong and
would cost days. Verified against the code:

| Claim in the summary | Reality |
|---|---|
| Voice: WebSocket at `/agent/voice/stream` | **`POST /v1/voice/dictate`.** Voice is a separate router, not under `/agent` (`router.py`), and it has no WebSocket: one multipart upload per dictation |
| Voice: "Opus/PCM audio streaming" | **No streaming.** Upload the finished recording once. `MediaRecorder` output (WebM/Opus, or MP4/AAC on Safari) is accepted as is, so there is no AudioWorklet or PCM conversion to build (§3.5) |
| Design: "listen for `event: design_ready` on `/agent/stream`" | Correct, but only the design path uses *named* SSE events (`init`, `design_ready`, `complete`). Everything else is unnamed `data: {"type": …}`. The parser must handle both (§2.3) |
| Long-form: "NO frontend work" | Small but real: `refine_progress` events, `done.warnings`, and **the document must be sent as Markdown or the pipeline silently does not run** (§6) |
| Task routing / code gen: "NO frontend work" | The editor must send `model_source`, must send the specific `type` rather than `general`, and must handle `content_replace` **before** the backend enables streaming repair (§5) |
| Email: "UI to trigger email generation" | Yes, but the browser must **not** call `/agent/email/*` directly; those routes take a raw provider key (§1, §8) |
| Blog: "minor" | Correct. Note it ships **disabled** (HTTP 503 until `BLOG_AI_ENABLED=true`) |

---

## 1. Architecture, auth, environments

| Surface | Called by | Credential in `api_key` | Billing |
|---|---|---|---|
| `POST /agent/stream`, `POST /agent/abort`, `POST /agent/autocomplete` | The editor (browser) | Eddyter **licence key**, validated against Redis | Agent service reserves and settles credits |
| `POST /agent/process`, `/agent/email/*` | **Eddyter backend only** | Provider key the backend resolved | Backend bills |
| `POST /v1/voice/dictate` | The editor (browser) | Eddyter **licence key** in `Authorization: Bearer <key>`; optional `X-Voice-Token` too | Existing text price by formatted output characters; BYOK plans pay no credits |
| `GET /v1/voice/status` | The editor (browser) | None | None |

- CORS is open: any origin, `GET/POST/OPTIONS`, all headers allowed and exposed (so `X-Request-ID` is readable from JS).
- Every HTTP response carries `X-Request-ID` and `X-Response-Time-Ms` (for a stream: time to first byte). Log the id and
  put it in bug reports.
- Locally, `scripts/run_local_server.py` accepts any licence key, keeps credits in memory (1000) and needs no Redis.

### Feature flags (code defaults in `config.py`)

If a feature is off, its UI entry point should be hidden or fail gracefully. These are the **code** defaults; the
production environment may override them. Confirm with whoever deploys.

| Feature | Env var | Code default | When off |
|---|---|---|---|
| Design canvas | `DESIGN_CANVAS_ENABLED` | **on** | No `design_ready` ever arrives; the request is answered as ordinary chat text (§4.2) |
| Voice | `VOICE_FLOW_ENABLED` | **on** | `/v1/voice/*` returns 404 |
| Email | `EMAIL_AI_ENABLED` | on | 503 on all three routes |
| Blog | `BLOG_AI_ENABLED` | **off** | `generate_blog` → HTTP 503 `{"error": "AI blog generation is disabled"}` |
| Long-form | `LONGFORM_REFINE_ENABLED` | **off** | The four text types run as one call, as before |
| Streaming code repair | `CODE_REPAIR_STREAM_EVENT` | **off** | `content_replace` is never sent |
| Task routing | `TASK_ROUTING_ENABLED` | on | Provider chosen as submitted |

---

## 2. The streaming protocol (`POST /agent/stream`)

Everything except voice and email goes through this one endpoint.

### 2.1 Request

`Content-Type: application/json`. Unknown fields are ignored on this endpoint. All strings are UTF-8.

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | One of: `fix_grammar` `simplify` `complete` `emojify` `translate` `make_longer` `make_shorter` `enhance_with_ai` `change_tone` `general` `generate_email_template` `extract_image_text` `generate_code` `explain_code` `generate_blog` `generate_design` `edit_design`. **Send the specific type.** `general` is the free-form chat agent and re-routes by wording |
| `query` | string | yes | ≤ 100,000 chars. The text to operate on, the prompt, or the blog brief (≤ 5,000 for `generate_blog`) |
| `model` | string | yes | `openai` `claude` `grok` `deepseek` `gemini` (case-insensitive) |
| `model_source` | `"default"` \| `"user"` | no (`"user"`) | **Send `"default"` until the user has changed the model dropdown, then `"user"`.** `"default"` lets the backend pick the best model per task (§5) |
| `api_key` | string | yes | The Eddyter licence key |
| `secondary_prompt` | string | no | Tone for `change_tone`, instruction for `enhance_with_ai`, style notes for `generate_blog` |
| `original_prompt` | string | no | Saved to the user's prompt history on success |
| `image_urls` | string[] | no | ≤ 3. **Each must start with the server's `CDN_URL`**: upload through the editor first, then pass the CDN URL. Anything else is HTTP 400 "Reference images must be uploaded through the editor before use" |
| `document_id` | string | no | `[A-Za-z0-9_.:-]`, ≤ 128. Scopes "the previous image" memory so a follow-up in one document never edits another's picture. **Send it on every request** |
| `image_provider` / `image_model` / `image_params` | | no | Image settings; see `API_DOCUMENTATION.md` § Image Generation |
| `blog_options` | object | `generate_blog` only | `{length?: short\|standard\|long, images?: 0..10\|"auto", tone?: string, audience?: string}`. Any other type → 400. Unknown keys → 400 |
| `existing_design_ast` | object \| JSON string | no | The design being edited (§4) |

### 2.2 Response

`200 text/event-stream`. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `X-Request-ID`.
Frames are separated by a blank line (`\n\n`); each `data:` line is one line of JSON.

**`POST` means the browser `EventSource` cannot be used.** Use `fetch` + a stream reader (reference parser in §2.3).

Failures *before* the stream starts are plain JSON with an HTTP status, **not** SSE:

| Status | Body | Meaning / UX |
|---|---|---|
| 400 | `{"error": "<field>: <reason>"}` (+ `"code": "VISION_NOT_SUPPORTED"` when the chosen model cannot read images) | Fix the request; the message is safe to show |
| 402 | `{"error": "Insufficient credits", "code": "INSUFFICIENT_CREDITS", "available": n, "needed": n}` (values are ×10 scaled: `30` = 3.0 credits) | Offer top-up. Also 402 with a plain `error` string when a managed plan is past due |
| 403 | `{"error": "...", "code"?: "..."}` | Licence invalid |
| 503 | `{"error": "..."}` | Licence service down, or `generate_blog` while disabled |
| 500 | `{"error": "..."}` | Retry once; report with `X-Request-ID` |

### 2.3 Frames

Two kinds on the same stream:

* **Unnamed**: `data: {"type": "<kind>", …}`: almost everything.
* **Named**: `event: <name>` + `data: {…}` with **no `type` field**: only `init`, `design_ready`, `complete` (design path).

So dispatch on `event ?? payload.type`.

| Kind | Payload | When | What the editor does |
|---|---|---|---|
| `start` | `{type, request_id}` | First frame, always | **Keep `request_id`**; it is what `/agent/abort` needs |
| `content` | `{type, content}` | Text chunk | Append Markdown at the insertion point |
| `content_replace` | `{type, content}` | Streaming code repair (flag off by default) | **Replace everything streamed so far** with `content` (§5) |
| `outline` | `{type, title, sections: [{id, heading}]}` | `generate_blog`, once the structure is valid | Draw a skeleton / progress list (§7) |
| `refine_progress` | `{type, chunk, of}` | Long-form refinement, after each passage | Progress "2 of 6" (§6) |
| `init` *(named)* | `{request_id, model, provider, mode: "design_canvas"}` | Design only, immediately | Show a "designing…" state. **`model`/`provider` are the first candidate, not necessarily the one that answered**; do not display them as fact |
| `design_ready` *(named)* | The full `DesignDocumentAST` (§4.4) | Design only, when finished | Render the design |
| `complete` *(named)* | `{total_tokens, elapsed_seconds, mode, warnings: [], attempts, auto_healed}` | Design only, after `design_ready` | Informational |
| `done` | `{type, duration, pending_images, warnings?}` | Text finished | Finish the text. `warnings` only when non-empty: `[{kind, detail, unit?, chunk?}]`. **More frames follow if `pending_images > 0`** |
| `image_ready` | `{type, placeholder_id, url, prompt, width?, height?, format?}` | A background image finished | Replace `placeholder::<placeholder_id>` in the content with `url` |
| `image_failed` | `{type, placeholder_id, error}` | Image failed or timed out (120 s) | Remove the placeholder or show a retry |
| `stream_end` | `{type}` | **Only** when `pending_images > 0`, after the last image | Done |
| `credit_limit_reached` | `{type, output_chars}` | Credits ran out mid-stream | Output is truncated; a `done` follows. Offer top-up |
| `error` | `{type, error}` | Provider/pipeline failure | Show `error`. **No `done` follows**; a design/blog also produces no result |

**End of stream** = the connection closes. `stream_end` is not always sent, so do not wait for it.
Typical order for a text request: `start → content… → done`. With images: `… → done → image_ready… → stream_end`.

Observed order for a design (from `local_run/design_live/run_20261027_170043.json`):
`start → init → (≈10–47 s of silence) → design_ready → complete → done`.

Reference parser (TypeScript sketch; written against the server's framing, not compiled inside Eddyter):

```ts
export async function streamAgent(
  body: Record<string, unknown>,
  { signal, onFrame }: { signal?: AbortSignal; onFrame: (kind: string, data: any) => void },
) {
  const res = await fetch(`${AGENT_BASE}/agent/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
  });
  if (!res.ok) throw Object.assign(new Error((await res.json()).error), { status: res.status, requestId: res.headers.get('X-Request-ID') });

  const reader = res.body!.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    for (let i; (i = buf.indexOf('\n\n')) >= 0; ) {
      const raw = buf.slice(0, i); buf = buf.slice(i + 2);
      let event: string | undefined; const data: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      const payload = JSON.parse(data.join('\n'));
      onFrame(event ?? payload.type, payload);
    }
  }
}
```

### 2.4 Do not use `agent.md` §11.2

That section documents `event: token`, `event: placeholder` and a `content_replace` with `old_content`/`new_content`.
**The server emits none of these.** The real frames are the table above. (`agent.md` §11.1 and the topology diagram are fine.)

### 2.5 Cancelling

`POST /agent/abort` with `{"request_id": "<from start>", "api_key": "<licence key>"}` → `200 {"success": true}`
(401 if the licence is invalid). Also abort the `fetch` with an `AbortController`.

For design requests the abort matters more than usual: the stream is silent while the model works, so the server can
only notice an abort through this call. Observed: abort during a design ends the stream in about 0.1 s with a `done` and
no `design_ready`. Call `/agent/abort` *and* abort the fetch.

### 2.6 Timeouts and proxies

* Designs took **9.5–47 s** (median ≈ 24 s, 8 designs) in the live run of 2026-10-27, before the reliability commit
  `df292a7`. The spec's "2.5–6 s" was never met. Set client and proxy idle timeouts to **≥ 90 s** and disable response buffering.
* An image can add up to 120 s after `done`. Keep the connection open until it closes.

---

## 3. Voice dictation

Hold a key or button (or click start, click stop), speak, release: formatted text appears at the caret (lists,
self-corrections, fillers, capitalisation and spacing handled). Pipeline: `VOICE_FLOW.md`. **This section is what the
browser must build.**

```
mic ─► MediaRecorder ─► user stops ─► ONE multipart upload: POST /v1/voice/dictate ─► (server) Whisper ─► formatter ─► JSON
                                                                                                                       │
                       editor.insertText(result.formatted_text) at the remembered caret ◄──────────────────────────────┘
```

### 3.1 What you must know before building

1. **One request per dictation, and nothing appears while the user speaks.** The recording is uploaded when it ends and
   the server answers once. There are no interim transcripts or live captions. Design the UI around
   "listening → processing → inserted".
2. **The client decides when the recording ends** (hold-to-talk, or click start / click stop). The server does **not**
   stop on a pause. If the product wants "stops by itself after a pause", the client needs its own voice-activity
   detection; none is provided (§9, Q9).
3. **Production requires the user's licence key** in `Authorization: Bearer <key>`; `/status` stays public. Also send
   `X-Voice-Token` if configured. Managed plans pay the existing text price: **1 credit per started 1,000 formatted output
   characters, minimum 1**. Context and raw transcript are not billed as input. The server reserves credits before work,
   tops up against the final output if needed, and settles through the existing text webhook. No speech or failed requests
   cost nothing. BYOK plans pay no credits and use their own Gemini, OpenAI, Claude, Grok or DeepSeek light-model key;
   without a supported personal key they receive `400 NO_API_KEY` before transcription. Platform keys are never used for
   BYOK. Local Whisper still transcribes accepted recordings. Managed plans with `use_own_keys` remain billable.
4. **Microphone needs a secure context** (HTTPS in production, or `localhost`/`127.0.0.1`), and the agent URL must be
   HTTPS when the page is (no mixed content). The Claude desktop app's built-in browser pane blocks microphones; test in
   Chrome, Safari or Firefox.
5. **Measured with real speech files on the local server** (Apple M4, macOS `say` voice, so real voices will be slower),
   server time per request: a plain sentence ≈ 0.6–0.7 s ("fast path", no model call); a sentence with list cues
   ≈ 1.8–2.5 s (a 6-second webm/opus recording was 2.5 s, the same speech as wav 1.8 s). On top of that comes the upload:
   that 6-second test recording was 60 KB, so on a normal connection it is small. Earlier measurement: 90 s of speech ≈ 7 s.
   A busy shared server will be slower, because the speech model decodes one recording at a time.

### 3.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/voice/dictate` | The whole feature: one recording in, formatted text out |
| GET | `/v1/voice/status` | `{whisper_model, whisper_loaded, format_provider, format_model, format_model_available, last_format_error, default_style, app_contexts_enabled}`. Use it for a health/"ready" indicator. `format_model_available: false` still works (rule-based formatting), only less polished |

There is no WebSocket, demo, format or other voice endpoint. A working reference client is `src/voice_flow/static/demo.html`
(~300 lines), served only by the local script `scripts/voice_demo.py`.

### 3.3 The request

`multipart/form-data`, built with `FormData`. **Do not set `Content-Type` yourself**; the browser adds the boundary.

| Part / header | Required | Notes |
|---|---|---|
| `audio` (file part) | yes | The recording, any container the server's PyAV reads: `webm`/Opus from Chrome, Edge and Firefox, `mp4`/AAC from Safari, `ogg`, `wav`. The file name's extension is not used. At most **10 MB** and **600 s** of audio |
| `context` (text part) | no | JSON text, all fields optional (below). Put it in the body, never in the URL or headers: it contains the user's text |
| `Authorization` (header) | yes, production | `Bearer <Eddyter licence key>` |
| `X-Voice-Token` (header) | if the server sets one | The shared secret |

`context` fields:

| Field | Meaning |
|---|---|
| `cursor_context` | `{preceding_text, following_text}`: up to 150 characters each side of the caret (longer is trimmed). Drives spacing, capitalisation and list continuation |
| `custom_dictionary` | Names and jargon to spell correctly: ≤ 100 terms, ≤ 60 characters each |
| `style` | `"clean"` keeps the speaker's wording, `"polished"` tightens list points. Omit = server default |
| `language` | Whisper language, e.g. `"en"`. Omit = server default (`en`) |
| `app_context` | Accepted but **ignored** unless the server enables `VOICE_APP_CONTEXTS` (default off): treated as `"general"` |

```ts
const form = new FormData();
form.append('audio', blob, `dictation.${ext}`);                 // ext: webm | m4a | ogg
form.append('context', JSON.stringify({ style: 'clean', cursor_context: { preceding_text, following_text }, custom_dictionary }));
const res = await fetch(`${AGENT_BASE}/v1/voice/dictate`, {
  method: 'POST', body: form, signal,
  headers: { Authorization: `Bearer ${licenceKey}`, ...(token ? { 'X-Voice-Token': token } : {}) },
});
const result = await res.json();                                 // res.ok: DictateResponse; otherwise { error: string }
```

### 3.4 The response

`200 application/json`:

```jsonc
{ "raw_text": "…what Whisper heard…",
  "formatted_text": " the onboarding flow.",           // EXACTLY what to insert; "" = nothing was heard
  "leading_space_required": true, "trailing_space_required": false,
  "source": "llm",                                     // "llm" | "fast" | "rules": all three are valid, insertable results
  "fallback_reason": null,                             // why, when the model was not used; "no speech detected" when empty
  "timings": { "audio_ms": 2100, "asr_ms": 640, "llm_ms": 1100, "post_ms": 0.03, "total_ms": 1750 } }   // server time; the upload is not counted
```

* `formatted_text` already begins with a space when `leading_space_required` and ends with one when
  `trailing_space_required`; the flags only say why. **Do not add spacing yourself.**
* **`formatted_text === ""` means nothing was heard.** Insert nothing; show "Didn't catch that".
* `source: "rules"` means the formatting model was down, too slow or rejected; the text is still usable.

Errors (production returns `{"error": "<message>", "code"?: "..."}`; the message is safe to show):

| Status | Meaning | UX |
|---|---|---|
| 400 | Bad recording or context, or `NO_API_KEY` for a personal-key account | Fix the request, or prompt for a supported personal key |
| 401 | Missing licence key, or wrong/missing shared token | Restore the session or check token configuration |
| 402 | `INSUFFICIENT_CREDITS` (`available` and `needed` are ×10 scaled), or managed plan past due | Offer top-up or payment update; keep the recording |
| 403 | Invalid or expired licence | Restore the licence |
| 503 | Licence, personal-key lookup or credit service unavailable | Keep the recording and offer Retry |
| 413 | Over 10 MB, or over 600 s of audio | "Recording too long". Set `audioBitsPerSecond` (below) so 10 minutes stays far under 10 MB |
| 429 | 8 dictations already being processed (header `Retry-After: 2`) | Retry once after the delay, then show "busy" |
| 404 | Voice is switched off on this server (`VOICE_FLOW_ENABLED`) | Hide the microphone button |
| 5xx / network error / timeout | | **Keep the recording** and offer Retry; do not make the user speak again |

Use a generous client timeout (≥ 60 s) for long dictations.

### 3.5 Recording (what `demo.html` does)

Read `src/voice_flow/static/demo.html`, the `<script>` part; it is a working implementation. In outline:

1. `navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}})`,
   once, after a user gesture.
2. Pick the first supported type: `audio/webm;codecs=opus`, `audio/webm`, `audio/mp4` (Safari), `audio/ogg;codecs=opus`
   (`MediaRecorder.isTypeSupported`). Create a `MediaRecorder`; **recommended: pass `audioBitsPerSecond: 32000`** (Opus at
   that rate is a normal speech setting; not tested here) so a 10-minute dictation is about 2.4 MB.
3. Start on key-down / first click, collect `ondataavailable` chunks. **A `MediaRecorder` has no pre-roll**, so a user who
   speaks at the instant of key-down may lose the first syllable. Either accept it, or start recording a beat earlier.
4. On release wait **~180 ms (post-roll)** before `recorder.stop()`, so the last word is not clipped. Build one `Blob`
   from the chunks and upload it.
5. **Drop holds shorter than ~250 ms** without uploading (a key bounce). The server also answers a recording under 50 ms with
   an empty result and never calls Whisper.
6. Uploads may overlap (the user can start the next dictation while the last is processing). The server handles one at a
   time and accepts up to 8 in flight; **show results in the order recorded** (the demo chains the promises).
7. Optional level meter: an `AnalyserNode` on the same stream.

### 3.6 Putting the text in the editor

Eddyter is built on Lexical. The sketch below uses standard Lexical calls; it has **not** been tried inside Eddyter.

* **Build `cursor_context`** from the editor selection when the user *starts* speaking: up to 150 characters before and
  after the caret. The caret's own paragraph is enough.
* **Remember the insertion point at the moment the recording ends**, not when the answer arrives. The answer comes 0.6–7 s
  later and the user may have moved the caret. Recommended: a FIFO queue of `{selection snapshot}`, one per upload;
  results are shown in order, each at its own snapshot.
* **Insert** `formatted_text` verbatim: `editor.update(() => { const s = $getSelection(); if ($isRangeSelection(s)) s.insertText(text); })`.
  One undo step per dictation.
* Lists arrive as ordinary text (`1. …\n2. …` or `- …`). Decide whether the editor converts them to list nodes (the same
  question the blog Markdown raises, §7).

### 3.7 UX states to implement

Mic permission (prompt / denied / no device) · server ready (`/v1/voice/status`) · idle · listening (timer, level meter) ·
processing (one per upload in flight) · inserted · "didn't catch that" · formatter degraded (`source: "rules"`, optional
quiet hint) · busy (429) · upload failed with Retry · feature off (404). Settings: style (clean/polished), language,
personal dictionary.

### 3.8 Acceptance checklist

- [ ] Hold-to-talk inserts correct text at the caret in the middle of a sentence, at the start of a paragraph, and after a list item
- [ ] Click-to-start / click-to-stop works for a 2-minute dictation and yields one insertion
- [ ] A tap shorter than ~250 ms uploads nothing and shows no error
- [ ] Two dictations in quick succession insert in order, each at its own caret
- [ ] The caret moving during processing does not misplace the text
- [ ] Mic denied / unplugged / revoked mid-session shows a clear message
- [ ] Works in Chrome (webm/opus), Safari (mp4/aac) and Firefox
- [ ] A failed upload (offline, 5xx, 429) keeps the recording and offers Retry
- [ ] Works from an HTTPS page against an HTTPS agent URL
- [ ] One undo reverses one dictation

---

## 4. Design canvas ("Mode 8")

The AI returns an **editable component tree** (JSON), not an image and not HTML. The editor renders it and the user can
click any part to edit it. Everything below is what the server actually emits (checked in `src/design_canvas/schema.py`,
`validate.py`, `text_processing_agent.py` and a real run).

The backend spec assumes the design lives in a Lexical `HtmlCanvasNode`. The server does not care: it only produces JSON.
How the block is stored and rendered inside Eddyter is the frontend's design decision.

### 4.1 Three ways to call it

Send `type: "generate_design"` for a new design and `type: "edit_design"` to change one. **The sub-mode is decided by
what the request carries, not by which of the two types you picked:**

| You send | Mode | Notes |
|---|---|---|
| `query` only | `create_from_prompt` | e.g. "Design an onboarding card for a meditation app…" |
| `query` + `image_urls: [<CDN url>]` | `create_from_vision` | Recreates the screenshot as an editable design. Only `image_urls[0]` is used |
| `query` + `existing_design_ast` | `edit_existing_ast` | An edit, **even if `type` is `generate_design`** |

* `type: "general"` also works (a regex on the wording decides) but is not deterministic. **Don't rely on it.** If the
  user has a design selected and types edit-like wording ("make the button green", "more padding"), the backend treats it
  as an edit, but the explicit types are safer.
* **Screenshot rules:** upload to the CDN first (§2.1); the server downloads it. PNG/JPEG/WebP as-is, other formats are
  re-encoded; ≤ 20 MB; large images are downscaled. The model must support vision (otherwise 400 `VISION_NOT_SUPPORTED`).
* **Send `existing_design_ast` only while a design is selected/being edited.** Send the full current AST: an object, or a
  JSON string.
* **There is no "edited node" field.** An AI edit is free text plus the whole AST. If the user selected one node,
  mention it in the `query` ("Change the CTA button label to…"). (Helpers `freeze_subtree` / `splice_subtree` exist in
  `patch.py` but are not wired into the API.)
* If `DESIGN_CANVAS_ENABLED` is off, the same request returns ordinary chat text and **no design events**. Handle
  "asked for a design, got `content`" without breaking.

### 4.2 What happens

```
start → init → [silent while the model works: 9.5–47 s observed] → design_ready → complete → done
```

* Show a working state at `init`. Nothing else arrives until `design_ready`.
* The server validates, self-repairs (up to 2 retries by default) and deterministically fixes the AST before sending, so
  **`design_ready` is always schema-valid.** `complete.auto_healed: true` means the server had to fix contrast, ids,
  depth or SVG attributes in code. `attempts` is the number of model calls. Neither needs UI.
* Failure (model unusable, output never valid) → an `error` frame and **no** `design_ready`.
* Billing, platform keys only: reserved up front (**3.0 credits** for a new design, **0.5** for an edit; 402
  `INSUFFICIENT_CREDITS` if short), charged only when `design_ready` is delivered, released on error. BYOK users are not charged.
  Show the cost before the user submits if the product wants that.

### 4.3 Endpoint parity

`POST /agent/process` (backend-to-backend) runs the same pipeline and returns the AST in a top-level **`design`** key of
its JSON response. The browser uses `/agent/stream`.

### 4.4 The `design_ready` payload

Real output of the server's own model (`DesignDocumentAST.model_dump()`), trimmed to three nodes. **Optional values are
`null`, not omitted. Every default is filled in**, so the renderer needs no default table.

```json
{
  "block_type": "eddyter_design_canvas",
  "version": "1.0",
  "design_id": "dsn_onboarding_01",
  "title": "Onboarding card",
  "theme_palette": ["#0B0B0F", "#34F5A6", "#FFFFFF"],
  "root": {
    "node_type": "container", "id": "root_card",
    "layout": "flex_col", "align_items": "center", "justify_content": "center", "gap_px": 16,
    "style": { "background_color": "#0B0B0F", "padding": {"top":24,"right":24,"bottom":24,"left":24},
               "border_radius": 24, "border_color": null, "border_width": 0, "max_width": null, "width_percent": null },
    "children": [
      { "node_type": "vector_illustration", "id": "hero_art", "view_box": "0 0 200 180", "width_px": 180, "height_px": 160, "editable": true,
        "layers": [ { "layer_id": "blob", "label": "Glow", "svg_element": "circle",
                      "attributes": {"cx": 100, "cy": 90, "r": 60}, "fill_color": "#34F5A6",
                      "stroke_color": null, "stroke_width": 0.0, "opacity": 0.9 } ] },
      { "node_type": "text", "id": "title_text", "tag": "h1", "align": "center", "font_size_px": 28, "line_height": 1.5, "editable": true,
        "style": { "background_color": null, "padding": {"top":0,"right":0,"bottom":0,"left":0}, "border_radius": 0,
                   "border_color": null, "border_width": 0, "max_width": null, "width_percent": null },
        "spans": [ {"text":"Ask ","color":"#FFFFFF","font_weight":"bold","italic":false,"underline":false},
                   {"text":"anything","color":"#34F5A6","font_weight":"bold","italic":false,"underline":false} ] },
      { "node_type": "button", "id": "cta_btn", "label": "Get started", "text_color": "#0B0B0F", "font_size_px": 15,
        "font_weight": "semibold", "editable": true,
        "style": { "background_color": "#34F5A6", "border_radius": 999, "…": "same style object as above" } }
    ]
  }
}
```

**Eight real, full designs** (from prompts, a screenshot, and edits; between them 35 text, 29 container, 9 vector,
9 button, 8 divider and 7 icon-button nodes) are in `local_run/design_live/run_20261027_170043.json` under `results[n].design`. Use
them as fixtures; `local_run/design_live/report.html` shows them side by side with timings.

TypeScript types (mirror of `schema.py`):

```ts
type Hex = string;                                   // #RGB #RGBA #RRGGBB #RRGGBBAA
type Weight = 'normal' | 'medium' | 'semibold' | 'bold';
interface Spacing { top: number; right: number; bottom: number; left: number }          // px, 0–256
interface NodeStyle { background_color: Hex | null; padding: Spacing; border_radius: number; border_color: Hex | null;
                      border_width: number; max_width: number | null; width_percent: number | null }   // radius ≤999, width ≤16, max_width 40–1440, width% 10–100

interface DesignDocument { block_type: 'eddyter_design_canvas'; version: '1.0'; design_id: string; title: string;
                           theme_palette: Hex[] /* 2–10 */; root: ContainerNode }
type DesignNode = ContainerNode | TextNode | DividerNode | VectorNode | IconButtonNode | ButtonNode;

interface ContainerNode { node_type: 'container'; id: string; layout: 'flex_col' | 'flex_row' | 'grid';
  align_items: 'start' | 'center' | 'end' | 'stretch';
  justify_content: 'start' | 'center' | 'end' | 'space_between' | 'space_around';
  gap_px: number; style: NodeStyle; children: DesignNode[] /* 1–25 */ }                // NO `editable` field
interface TextSpan { text: string; color: Hex; font_weight: Weight; italic: boolean; underline: boolean }
interface TextNode { node_type: 'text'; id: string; tag: 'h1' | 'h2' | 'h3' | 'p' | 'span' | 'caption'; spans: TextSpan[] /* ≥1 */;
  align: 'left' | 'center' | 'right'; font_size_px: number /* 10–96 */; line_height: number /* 1–2.5 */; style: NodeStyle; editable: boolean }
interface DividerNode { node_type: 'divider'; id: string; color: Hex; thickness_px: number; width_percent: number; margin_y_px: number; editable: boolean }
interface SvgLayer { layer_id: string; label: string; svg_element: 'path' | 'circle' | 'rect' | 'ellipse';
  attributes: Record<string, string | number>; fill_color: Hex; stroke_color: Hex | null; stroke_width: number; opacity: number }
interface VectorNode { node_type: 'vector_illustration'; id: string; view_box: string /* "0 0 200 180" */;
  width_px: number; height_px: number; layers: SvgLayer[] /* 1–24 */; editable: boolean }
interface IconButtonNode { node_type: 'icon_button'; id: string;
  icon_name: 'arrow_left' | 'arrow_right' | 'chevron_left' | 'chevron_right' | 'plus' | 'close' | 'check' | 'sparkles' | 'search' | 'send';
  color: Hex; size_px: number /* 12–64 */; aria_label: string; style: NodeStyle; editable: boolean }
interface ButtonNode { node_type: 'button'; id: string; label: string; text_color: Hex; font_size_px: number /* 12–32 */;
  font_weight: Weight; style: NodeStyle; editable: boolean }
```

Guarantees the server enforces (rely on them): every `id` / `layer_id` is unique and matches `^[a-z0-9_]{2,40}$` · tree
depth ≤ 6 · ≤ 25 children per container · text-vs-background contrast ≥ 3:1 · SVG `attributes` contain only these keys:
`d cx cy r rx ry x y x1 y1 x2 y2 width height transform points` (no event handlers, hrefs or `javascript:`/`data:` values).
On an edit, ids of untouched nodes are preserved, so a selection by id survives an AI edit.

### 4.5 Rendering guide (recommendation)

| Node | Render as |
|---|---|
| `container` | `div`. `flex_col`/`flex_row` → `display:flex` with column/row; `grid` → `display:grid`. **The schema has no column count for `grid`**: decide a rule (ask the backend team, §9 Q4). `align_items`: start/center/end → `flex-start`/`center`/`flex-end`, `stretch`. `justify_content`: `space_between` → `space-between`, `space_around` → `space-around`. `gap_px` → `gap`. `style` → background, padding (px), `border-radius`, `border: {width}px solid {color}`, `max-width` (px), `width_percent` (%) |
| `text` | The element named by `tag` (`caption` → `<small>`/`<figcaption>`-like). One `<span style="color;font-weight;font-style;text-decoration">` per span. `align` → `text-align`, `font_size_px`, `line_height` |
| `divider` | A block with `border-top: {thickness_px}px solid {color}`, `width: {width_percent}%`, vertical margin `margin_y_px` |
| `vector_illustration` | `<svg viewBox width height>`; one element per layer (`svg_element`), `attributes` set as attributes, plus `fill`, `stroke`, `stroke-width`, `opacity`; tag it `data-layer-id` so each layer is selectable |
| `icon_button` | `<button aria-label>` with your own icon set keyed by the 10 `icon_name`s, at `size_px` and `color` |
| `button` | `<button>` with `label`. **There is no link/action field**: the design is static; hooking buttons up is a product decision |

Safety (the AST comes from a model): never use `innerHTML`; create elements and set properties. For SVG, set only the
whitelisted attributes above even though the server already filtered them.

### 4.6 Click-to-edit (frontend-owned)

* Put `data-node-id` (and `data-layer-id`) on rendered elements; a click selects by id.
* **Direct edits** (typing in a text node, picking a colour, resizing) mutate the AST locally; there is no server call.
  The block's stored content *is* the AST (`block_type: "eddyter_design_canvas"`, `version: "1.0"`). Persist the whole thing.
  Respect `editable: false` (containers have no `editable` field; treat them as structural).
* **AI edits**: POST `type: "edit_design"`, `query: "<instruction>"`, `existing_design_ast: <current AST>`. The
  `design_ready` that comes back is a **complete replacement document**, not a patch: swap it in whole and keep the
  selection by id. The `design_id` stayed the same in all four edits of the live run, but nothing in the schema
  guarantees it, so key your stored block on your own id, not on `design_id`.
* Store the version string; future AST versions will change `version`.

### 4.7 Acceptance checklist

- [ ] `generate_design` from a prompt shows a working state at `init` and renders at `design_ready`
- [ ] Screenshot upload → CDN URL → `image_urls` → renders (and a non-image/oversize file is refused cleanly)
- [ ] All six node types render, including nested containers, multi-span (two-colour) headings and multi-layer SVG
- [ ] Every node and SVG layer is selectable by id; direct edits persist after reload
- [ ] AI edit with `existing_design_ast` replaces the design and keeps the selection; ids of untouched nodes are unchanged
- [ ] `null` optional values do not break rendering
- [ ] Stop button calls `/agent/abort` and returns the UI to idle with no design inserted
- [ ] `error` frame, 402 `INSUFFICIENT_CREDITS` and "asked for a design, got text" are all handled
- [ ] A 45-second generation does not time out in the browser or the proxy
- [ ] No `innerHTML`/`dangerouslySetInnerHTML` on AST-derived strings

---

## 5. Code generation and task routing

Small, but not zero:

1. **Send `model_source`** (`"default"` until the user touches the model dropdown, then `"user"`). Omitted = `"user"`, which
   disables the backend's per-task model choice (Claude for code, GPT-5.2 for `make_longer` and email text, etc.).
2. **Send the specific `type`** (`fix_grammar`, `generate_code`, …), never `general` for an operation that has its own type.
3. `generate_code` / `explain_code` return Markdown with tagged code fences. Render fences as code blocks.
4. **`content_replace`** (§2.3). The backend can validate generated code and repair it once; the repaired answer arrives as
   `content_replace` and **replaces the whole streamed answer**. It is off (`CODE_REPAIR_STREAM_EVENT=false`) *until the
   frontend supports it*: tell the backend when it ships. Until then streaming only logs validation errors.
5. `/agent/process` (backend) returns HTTP 422 `{"error", "issues"}` for code that fails validation. That is the Node
   backend's concern, but it may surface it.

---

## 6. Long-document refinement (`fix_grammar`, `simplify`, `change_tone`, `enhance_with_ai`)

Off by default (`LONGFORM_REFINE_ENABLED`). Frontend work is small but easy to miss:

* **Send the document as Markdown.** If the editor sends HTML, the pipeline does not run and the old single-call path is used:
  no error, no `refine_progress`, no structure guarantees. Which format the editor sends is a frontend fact I cannot see;
  please confirm (§9 Q5).
* Text arrives **passage by passage** (about 1,000 words each; the first is ~300 words), each already verified and in order.
  Append each `content` chunk; nothing is ever retracted. Show progress from `refine_progress {chunk, of}`.
* `done.warnings` may contain `{kind: "chunk_restored", chunk, detail}`: that passage was **kept exactly as the user wrote it**
  because the AI broke its structure. Show a non-blocking notice ("Passage 3 of 6 left unchanged").
* Tables, quotes, images and code blocks are guaranteed untouched byte-for-byte.
* Documents ≥ 600 words or containing a table/quote/image/code block qualify; short plain text still uses the fast single call.

---

## 7. Blog generation (`generate_blog`)

Off by default: until `BLOG_AI_ENABLED=true`, HTTP 503. Full contract: `API_DOCUMENTATION.md` § Visually rich blog generation.

* Send `type: "generate_blog"`, `query` = topic/brief (3–5,000 chars), optional `blog_options`, `document_id`.
* Order of frames: `start → content (title block) → outline → content… → done → image_ready/image_failed… → stream_end`.
  Use `outline` to draw a skeleton immediately.
* Content is **GitHub-flavoured Markdown**: `# Title`, an italic subtitle, `##` sections, pipe **tables**, `>` **pull quotes**,
  and images as `![alt](placeholder::<uuid>)` followed by an italic caption line `*caption*`. The editor must import GFM
  tables, blockquotes and image-plus-caption (open item O2 in `rich_content_engine_plan.md`), and replace each
  `placeholder::<uuid>` when `image_ready` carries the same `placeholder_id`.
* Pictures arrive **after** `done`. A post can finish text in ~30 s and pictures 20+ s later; do not mark the action
  complete on `done` if `pending_images > 0`.
* `done.warnings[].kind`: `unit_lint` `outline_filled` `outline_rules` `quota_tables` `quota_quotes` `heading_mismatch`
  `image_skipped` `images_limited`. Show as an info panel, not an error.
* Cost: text is billed per character, plus per picture (reserved after the outline).

---

## 8. Email templates: deltas to `EMAIL_API_HANDOFF.md`

Follow that document; it is accurate (15 block types, all request fields and categories re-checked against the code).
Three additions:

1. **Browser → Eddyter backend → agent.** The routes take a raw provider key and have no auth of their own. The frontend
   calls an Eddyter-backend route that adds the key. **That proxy route is not in this repository**; confirm it exists (§9 Q2).
2. The request also accepts optional `model_source` (`"default"|"user"`, same meaning as §2.1; email always uses the submitted
   provider, so it has no effect today). It was missing from that document's §3.2 table.
3. `GET /agent/email/categories` returns only the *enabled* categories: 7 today (`reminder_deadline`, `policy_notice`,
   `invitation`, `product_launch`, `feature_update`, `announcement`, `general`), matching that document. More are defined
   in the code but switched off; never hard-code the list.

`type: "generate_email_template"` also exists on `/agent/stream`. It is a chat-style agent (`email_template` in
`text_processing_agent.py`), **not** the template-JSON API, and is not covered here. Use `/agent/email/generate` for
block-based emails.

---

## 9. Open questions (need an owner before launch)

| # | Question | Who | Why it matters |
|---|---|---|---|
| Q1 — resolved | Voice authentication and pricing | Backend | Built: `Authorization: Bearer <licence key>`, existing text billing by formatted output, no credits for BYOK plans. BYOK needs a supported personal light-model key and never uses platform provider keys. Local Whisper still runs on the API server |
| Q2 | Does an Eddyter-backend proxy for `/agent/email/*` exist? | Backend | The frontend has nothing to call otherwise |
| Q3 | Production values of `VOICE_FLOW_ENABLED`, `DESIGN_CANVAS_ENABLED`, `BLOG_AI_ENABLED`, `LONGFORM_REFINE_ENABLED`, `CDN_URL`; voice defaults on in code | Backend / ops | Decides which UI is live |
| Q4 | `container.layout == "grid"` has no column count in the schema. How should it render? | Backend + Frontend | Today the AST cannot express it; either add `columns` to the schema or define a rule (e.g. auto-fit) |
| Q5 | What does the editor send for the four long-form types: Markdown or HTML? | Frontend | HTML silently bypasses the pipeline (§6) |
| Q6 | Where does a design block live in Lexical, and are buttons meant to be functional? | Frontend + Product | The AST has no link/action field |
| Q7 | When may `CODE_REPAIR_STREAM_EVENT` be turned on? | Frontend → Backend | Needs `content_replace` shipped first (§5) |
| Q8 | Do you want live voice captions? | Product | Not possible with the current server: one decode per upload, no interim transcripts |
| Q9 | Does dictation need to stop by itself after a pause? | Product + Frontend | It used to (server-side, in hands-free mode); it was dropped with the WebSocket. Now the client ends the recording (hold, or click start / click stop). Auto-stop would need a client-side voice-activity detector |

---

## 10. Trying it locally

```bash
cd agent && .venv/bin/python scripts/run_local_server.py      # http://127.0.0.1:8000, any licence key works
```

(Desktop launch config: `agent-api-local`. It turns on voice, blog and long-form refinement.)

Design, no code needed:

```bash
curl -N http://127.0.0.1:8000/agent/stream -H 'Content-Type: application/json' -d '{
  "type": "generate_design", "model": "openai", "model_source": "default", "api_key": "local",
  "query": "Design an onboarding card UI for a meditation app: dark background, two-tone headline, a Start button"}'
```

You should see `start`, `init`, a long pause, `design_ready`, `complete`, `done`.

Voice, in a normal browser (not the desktop app's pane): `.venv/bin/python scripts/voice_demo.py` opens
`http://127.0.0.1:8000/`. Read its source as the reference client. Without a browser:

```bash
curl -s http://127.0.0.1:8000/v1/voice/dictate -F 'audio=@recording.webm' -F 'context={"style":"clean"}'
```

| Asset | Path |
|---|---|
| Voice reference client | `src/voice_flow/static/demo.html` |
| Voice endpoint contract (code of record) | `src/voice_flow/router.py`, `src/voice_flow/schemas.py` |
| Design schema (code of record) | `src/design_canvas/schema.py` |
| Real design ASTs + event timings | `local_run/design_live/run_20261027_170043.json`, `report.html` |
| Live design checks (paid, ~10 model calls) | `scripts/local_design_requests.py` |
| Email previews | `local_run/email_previews/all/gallery.html` |
