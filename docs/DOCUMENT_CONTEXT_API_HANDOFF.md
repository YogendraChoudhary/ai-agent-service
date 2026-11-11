# Document context API and editor integration

The Python backend supports document-aware Ask, Draft and explicit Edit on both `/agent/process` and `/agent/stream`. Existing `document_text` requests retain their exact patch behavior. The new path is enabled by default (`DOCUMENT_CONTEXT_ENABLED=true`). An explicit environment override of `false` disables new-path requests with HTTP 503 and code `DOCUMENT_CONTEXT_DISABLED`.

## Requests and actions

Use `type="general"`. Put the actual user task in `query` and reference material in `document_context.text`. `document_action` is `ask`, `draft` or `edit`; omission defaults to Ask when context is present. `document_action` without context is invalid.

```json
{
  "type": "general",
  "query": "What are the main arguments in my draft?",
  "model": "openai",
  "model_source": "user",
  "api_key": "<endpoint credential>",
  "document_action": "ask",
  "document_context": {
    "text": "# Proposal\n\nOur onboarding process needs fewer steps.\n",
    "selection": {"start_utf16": 12, "end_utf16": 26},
    "cursor_utf16": 26
  }
}
```

Use the existing credential contract: `/agent/process` takes the provider key resolved by the application backend; `/agent/stream` takes the user's license key. The new fields do not change authentication, provider eligibility or BYOK rules.

| Action | Result | Editor handling |
| --- | --- | --- |
| Ask | Grounded answer in Markdown | Display in chat |
| Draft | Only the suggested new passage | Preview and offer Insert |
| Edit | Existing verified document patch proposal | Review and accept through the existing apply flow |

Both endpoints accept these additional fields:

- `document_context.text`: exact plain text or Markdown, at most 100,000 Python characters. HTML documents, NUL and invalid Unicode are rejected. Valid emoji joiners, BOM and line endings are preserved. Empty text is permitted for Draft; Ask returns an explanation without a model call.
- `document_context.selection`: optional `start_utf16` and `end_utf16`, inclusive start and exclusive end, in the exact text serialization. A nonempty selection wins over the cursor. A collapsed selection is ignored.
- `document_context.cursor_utf16`: optional insertion position; defaults to the end of the source. Integer coordinates must be within the source and must not split a Unicode surrogate pair. Booleans and string coordinates are rejected.
- `query`: nonblank, at most 8,000 characters for ordinary Ask/Draft or 2,000 for Edit. Analysis additionally limits the JSON-serialized instruction to 2,000 UTF-8 bytes.
- `document_scope`: optional `document` (default), `selection` or `focused`. Non-document scopes are Ask-only. Selection requires a nonempty captured selection and cannot retrieve outside it. Focused scope searches the supplied document, preferring captured focus when present.
- `document_analysis`: optional `summary` or `contradictions`, only for Ask with document scope. This explicitly requests bounded, multi-call full-source analysis.
- `suggest_document_intent`: optional strict boolean, default `false`. Returns advisory action metadata; it never switches the requested action. All these options require `document_context`.

The nested context/range schemas reject unknown fields. Do not send both `document_text` and `document_context`. Secondary prompts, design AST, image inputs/settings and blog options conflict with document-context requests. Existing typed operations such as `complete`, `fix_grammar` and `generate_blog` do not accept ambient context.

Explicit Edit runs the existing edit validation against the complete exact source, including its 400-block limit. The block writer also receives up to three read-only style samples from untargeted prose, preferring the same section. Samples exclude protected/excluded areas and fit a total 2,000 serialized UTF-8 bytes. They cannot become additional patch targets. Legacy `document_text` requests receive no new style samples. Supported changes, protected content, atomic validation and stale-source HTTP 409 behavior are unchanged. See [the edit handoff](DOCUMENT_EDIT_API_HANDOFF.md).

## Context limits and scope

Default Ask provides the complete source or returns `needs_context` without running a model. It never silently substitutes partial retrieval for a whole-document task. To answer a local question against a larger source, explicitly set `document_scope="focused"` or `"selection"`. For global analysis, explicitly set `document_analysis="summary"` or `"contradictions"`.

Scoped Ask uses exact source chunks and a request-local in-memory SQLite FTS5 index with BM25 ranking. It supports Unicode terms and CJK bigrams; it is lexical retrieval, not embedding similarity. Deployed Python must include FTS5. Unavailable search, no matching passages or an unpackable selection returns `needs_context` without a model call. Captured focus may supply a passage even without a lexical match. Retrieved references have server-assigned `r0`, `r1`, … identifiers, exact quotes, heading paths and UTF-16 ranges. Partial answers must remain scoped to those passages; retrieved evidence does not establish whole-document coverage.

Draft uses complete context when it fits. Otherwise it includes a bounded heading outline plus the active section or a window around the captured focus. With no focus, it continues from the end. Its metadata reports `coverage="partial"`, and warnings explain that limitation.

The context envelope has a 24,000 UTF-8-byte ceiling and a 6,000 estimated-token allocation. The current dependency-free estimator deliberately counts each UTF-8 byte as one estimated token, including serialization and metadata. This is conservative and usually binds before the byte ceiling: it is **not** 6,000 measured provider tokens or 24,000 characters of prose. Non-English/emoji-heavy text can fit fewer characters. The task is bounded separately; output defaults use the existing rounding/reasoning-budget helper, starting from 2,000 visible tokens for Ask and 1,000 for Draft.

No provider token-count request is added to the hot path. Before production enablement, validate the configured model IDs/capacities and synthetic grounding/latency behavior. Increase the packing allowance only after a provider-calibrated counter/estimate is tested for every eligible fallback. When reported output usage reaches the configured model output cap, the response includes `output_limit_reached` with a warning that the text may be incomplete. The installed agent SDK does not reliably expose every provider finish reason, so missing usage cannot establish that an answer was complete.

## Full-source analysis

For example, add these fields to the normal request and credential fields:

```json
{
  "type": "general",
  "query": "Find inconsistent cohort sizes for the same pilot period.",
  "document_action": "ask",
  "document_scope": "document",
  "document_analysis": "contradictions",
  "document_context": {"text": "<exact complete Markdown source>"}
}
```

The backend reads every source chunk, extracts at most two short task-relevant claims per chunk, synthesizes up to eight candidate findings across all extracted evidence, then verifies candidates against original source quotes and surrounding text. Quote membership and returned UTF-16 offsets are checked deterministically. Contradictions require two distinct source quotes. Only findings judged supported by verification are returned. Differing dates, units, entities, scopes or explicit revisions are not sufficient by themselves to establish a contradiction.

Bounds are explicit: up to 32 chunks of 4,000 UTF-8 bytes each, two extraction calls in parallel, at most `chunk_count + 2` logical calls and twice that many provider attempts, a 180-second overall deadline, and at most 24,000 UTF-8 bytes of dynamic input per call. Extraction/reduction/verification replies are limited to 2,000/6,000/2,000 UTF-8 bytes respectively. Oversized documents return `needs_context` before model work. Malformed evidence, an unverified finding, a failed required stage or an exceeded budget cannot publish completed full coverage. The existing per-request credit cap can reject an otherwise valid analysis before any model work.

Analysis starts with `coverage="pending"` and `strategy="analysis"`. Streaming emits `document_analysis_progress` with completed/total extraction counts, then `document_analysis_ready` only after the complete successful pipeline. Its data is also the process response's `document_analysis`:

```json
{
  "mode": "contradictions",
  "status": "ready",
  "context": {"coverage": "full", "strategy": "analysis", "references": []},
  "findings": [{"text": "Conflicting cohort sizes: 36 and 48.", "evidence_ids": ["e0", "e1"]}],
  "billing": {"input_chars": 11197, "output_chars": 916, "stage_calls": 5, "provider_attempts": 5},
  "rejected_findings": 0
}
```

This is a field illustration: actual context contains the complete metadata and exact cited `e0`, `e1`, … references. Final text uses those identifiers. Process `document_context` contains the final context; streams must replace the initial pending metadata with `document_analysis_ready.context`. A `no_findings` result means no supported findings were established, not proof that no contradictions exist. Full coverage means all chunks were read, not exhaustive semantic understanding: short extraction and model-based verification can miss or misinterpret facts. The response includes an `analysis_scope` warning stating this limit.

## Response metadata and events

`/agent/process` returns the existing response fields plus nullable `document_context`, `document_analysis` and `document_intent`. Ask/Draft metadata includes:

```json
{
  "action": "ask",
  "response_kind": "chat_answer",
  "status": "ready",
  "source_sha256": "<SHA-256 of exact complete UTF-8 source>",
  "coverage": "full",
  "strategy": "full",
  "source_chars": 80,
  "included_source_chars": 80,
  "context_chars": 380,
  "included_ranges": [{"start_utf16": 0, "end_utf16": 80}],
  "estimated_input_tokens": 425,
  "billable_input_chars": 414
}
```

The numeric values above illustrate field meanings. Use actual returned values. `source_chars` and `included_source_chars` count Python characters; `included_ranges` use UTF-16. `estimated_input_tokens` estimates the dynamic model message, not system instructions, and is separate from actual provider usage in request traces. Scoped retrieval and completed analysis return cited source quotes in `references`; ordinary full/focused drafting context metadata has no quotes. References are response data for the authorized caller and are excluded from application traces/logs. Metadata also carries `scope`, `analysis_mode`, `billable_output_chars`, `reserved_input_chars`, `reserved_output_chars` and `max_stage_calls`; unused fields have zero/null defaults. Coverage can be `full`, `partial`, `none` or `pending`; strategy can be `full`, `focused`, `none`, `retrieval`, `selection` or `analysis`.

Ask/Draft streams use:

```text
data: {"type":"start","request_id":"..."}

event: document_context_ready
data: {"action":"ask", ...}

data: {"type":"content","content":"..."}

data: {"type":"done","duration":0.1,"pending_images":0}
```

Handle named events and ordinary `data.type` events. Process and stream metadata have identical meanings. Partial context and output-budget warnings appear in process `warnings` or streaming `done.warnings`. A canceled new-path stream marks `done.aborted=true`; discard its incomplete suggestion. A model failure emits an error instead of successful completion and does not restart after visible text.

For empty or oversized default Ask, or unavailable scoped/analysis context, metadata has `status="needs_context"`, `coverage="none"`, zero billable input and no model prompt. The content is a deterministic explanation. Both endpoints return a normal completion with that explanation. Do not present it as a model analysis or offer Insert.

### Draft placement

When a Draft task names a place, `document_context_ready` carries `placement`, resolved in code (no model call) against
the same top-level blocks Edit uses; the draft is written for that place, and the context is centred there:

```json
"placement": {"position": "after", "text": "## Pricing\n", "start_utf16": 120, "end_utf16": 131}
```

`text` is the exact text of one top-level block of the snapshot you sent (the same rule as Edit's `before`), with its
UTF-16 offsets. Insert the draft after or before that block. Agreed meanings:

| Wording | Anchor |
|---|---|
| "below/under/after the X heading", "below heading X", "below X" | `after` the heading block |
| "at the end of section X", "after the X section", "after X", "following X" | `after` the last block of that section, subsections included |
| "between X and Y" | `before` the heading Y (or `after` section X when only X exists) |
| "at the start of section X" | `after` the heading block |
| "above/before the X heading", "before the X section" | `before` the heading block |
| "at the end / bottom (of the document)" | `after` the last block |
| "at the top / start of the document" | `before` the first block |
| "after the second / last paragraph (in X)", "after paragraph 3" | that paragraph |
| "after the paragraph about refunds", "below the table", "after '<quoted words>'" | the one block that matches |
| "below this paragraph / section" | the block (or section) at the cursor |

Locations are understood in English only; other languages fall back to the cursor. No named place: `placement` is absent; insert at the cursor as before. A named place that is missing or matches more
than one block gives `status="needs_clarification"` with the question in `message`, **no `content` frames** and no
model call or charge (for example "There is no heading named 'Risks'. Name an existing heading, or place the cursor
where the draft should go."). Metadata now also carries `message` for `needs_context`.

When `suggest_document_intent=true`, `document_intent_suggested` precedes action events in a stream; process returns `document_intent`. It reports `suggested_action`, `requested_action`, `effective_action`, `advisory=true`, `ambiguous` and a reason code. Suggestions inspect only the task, with limited multilingual cues. Explicit action always wins; omission still executes Ask. Requests to place new content ("add … below/after/above …", "insert … after …", "put … above …", "add a new section/paragraph/table") suggest Draft with reason `new_content`. Mixed/unclear tasks suggest Ask. This heuristic is not a trained multilingual classifier or permission to apply edits.

Edit continues to emit `document_edit_ready`, one verified document snapshot and `done`; its process result uses `document_edit`, with `document_context=null`. Legacy event order is unchanged.

## Billing and privacy

Billing transport now uses the [persistent delivery queue](BILLING_DELIVERY.md). Failed handoffs retain their
credit holds. Payment retries remain disabled unless receiver-side duplicate protection is confirmed. Both
explicit and legacy document edits canceled before any provider call release their reservation and report an abort.

Streaming reserves and settles using `billable_input_chars`: the user query plus the exact serialized context envelope sent to the model. Its query framing and static system rules follow existing static-prompt accounting. Output charges count emitted generated text, not metadata or warnings. Edit keeps its previous full-source accounting. BYOK calls do not use platform keys or credits.

Analysis reserves the complete bounded plan before calling a provider, using `reserved_input_chars` and `reserved_output_chars`. On success it settles the actual dynamic inputs and raw generated outputs of the extraction/reduction/verification stages from `document_analysis.billing`. Rendered answer text and result metadata are not billed a second time. Provider retries/fallback retain the existing accounting policy; the billed character fields cover successful logical-stage payloads/replies, and provider usage remains separately recorded. Incomplete canceled analysis releases its reservation without a successful text deduction; errors retain the existing failure policy.

`/agent/process` still delegates text billing to the upstream application backend. The caller must read `document_context.billable_input_chars` for new Ask/Draft requests instead of counting only `query`. On `status="needs_context"`, charge neither input nor the deterministic explanation. For ordinary Ask/Draft success with `status="ready"`, add the actual generated output length under the existing character-credit policy. For analysis, use `document_analysis.billing.input_chars` and `.output_chars`, which include intermediate work, instead of charging just the rendered answer. Match the streaming policy by not charging incomplete/error analysis. Do not attach a second text deduction inside the agent process endpoint. The upstream caller implementation is outside this checkout.

No-model results skip reservations/settlement. A new-path request canceled before a provider call releases any reservation without a text charge. Errors release reservations using the existing failure policy. An in-flight provider request cannot always be canceled; stopping a stream does not guarantee the provider stops computing.

Ask/Draft/analysis have no web, image, storage or write tools. Document source, task and generated quotes are excluded from the agent's application request/response logs even with body logging enabled, and the new path does not save `original_prompt` to prompt history. Provider errors are returned without raw detail. Do not enable SDK HTTP body debugging for private documents. Data still goes to the chosen eligible model provider; documents are not persisted or shared across requests by this feature.

## Editor integration checklist

1. Serialize the current authorized document and capture focus/revision together when Send is pressed. Send a fresh snapshot on each turn; general chat history is not added by this release.
2. Map editor positions into the exact Markdown/plain-text serialization. DOM and ProseMirror positions are not Markdown UTF-16 offsets. Omit uncertain focus positions instead of guessing them.
3. Default the assistant to Ask, and expose Draft/Edit explicitly. Show an attached-document indicator and let the user omit context for a request.
4. Require `document_context_ready`/process context metadata before presenting a new Ask/Draft response as document-aware. An older server may ignore unknown top-level fields. If metadata is absent, fail the new feature visibly and disable insertion. Require patch metadata for Edit.
5. Render Ask in chat. Preview Draft and Insert only after confirming its exact source hash and captured anchor still match. If the document changed, regenerate or let the user deliberately choose a new insertion point.
6. Render returned references against their exact source snapshot and safely display analysis progress. Keep intent suggestions advisory; require an explicit user action to change modes. Review Edit proposals and retain the existing hash/before-text/offset checks and normal save flow. Accepted insertion/edit changes should be one undoable transaction.
7. Bind responses to request and document snapshots; discard stale results after switching documents or aborting. Preserve original selected-text behavior for typed `complete`, which returns original text plus completion.
8. Sanitize Markdown/HTML rendering. Restrict remote images and unsafe links so generated markup cannot make unsolicited requests containing private source text.
9. Verify keyboard access, selection/focus restoration, rich nodes, undo, save/reload and concurrent typing in the real application before rollout.

Run `.venv/bin/python scripts/demo_document_context.py` for a provider-free process/stream example. The script checks source-aware Ask, Draft, named stream metadata and the no-model size-limit response. Run `.venv/bin/python -m pytest tests/test_document_context*.py tests/test_document_backend*.py tests/document_edit` for the new boundary/integration tests and legacy edit regressions.

## Verification

Verified on 3 November 2026: the full repository suite passed **2,144 tests and 1,662 subtests** after the final failure/cancellation checks. The context/backend/edit suites cover routing parity, exact source ranges, multilingual retrieval, all-chunk coverage, malformed evidence/verification, bounded calls and accounting, privacy, cancellation and unchanged legacy patch behavior. The provider-free API demo also passed during the first release.

[Live provider results](DOCUMENT_CONTEXT_LIVE_RESULTS.md) records first-release Ask/Draft and repeated Edit checks on OpenAI, Gemini and Grok, plus the new backend's five successful OpenAI checks and 28 verified Gemini/Grok responses, including dedicated style edits and a separate-period negative control. The latest targeted evaluator/backend/edit run passed 154 tests. Observed timings and the three corrected evaluator false failures are documented in that report. Claude's configured credential returned HTTP 401 in the earlier run. The enabled release default is retained. Production revision/flag state, real editor integration and the upstream application billing implementation remain unverified because their deployment/source is outside this checkout.
