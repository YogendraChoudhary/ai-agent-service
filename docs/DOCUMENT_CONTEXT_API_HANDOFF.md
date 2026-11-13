# Document context API and editor integration

The Python backend supports document-aware Ask, Draft and explicit Edit on both `/agent/process` and `/agent/stream`. Existing `document_text` requests retain their exact patch behavior. The new path is enabled by default (`DOCUMENT_CONTEXT_ENABLED=true`). An explicit environment override of `false` disables new-path requests with HTTP 503 and code `DOCUMENT_CONTEXT_DISABLED`.

## Requests and actions

Use `type="general"`. Put the actual user task in `query` and reference material in `document_context.text`. `document_action` is `ask`, `draft`, `edit` or `auto` (the agent picks the mode and reports it; see [Auto mode](#auto-mode-document_action-auto)); omission defaults to Ask when context is present. `document_action` without context is invalid.

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
- `document_scope`: optional `document` (default), `selection`, `focused` or `auto`. Non-document scopes are Ask-only. Selection requires a nonempty captured selection and cannot retrieve outside it. Focused scope searches the supplied document, preferring captured focus when present.
- `document_analysis`: optional `summary` or `contradictions`, only for Ask with document scope. This explicitly requests bounded, multi-call full-source analysis.
- `suggest_document_intent`: optional strict boolean, default `false`. Returns advisory action metadata; it never switches the requested action. All these options require `document_context`.

The nested context/range schemas reject unknown fields. Do not send both `document_text` and `document_context`. Secondary prompts, design AST, image inputs/settings and blog options conflict with document-context requests. Existing typed operations such as `complete`, `fix_grammar` and `generate_blog` do not accept ambient context.

Explicit Edit runs the existing edit validation against the complete exact source, including its 400-block limit. The block writer also receives up to three read-only style samples from untargeted prose, preferring the same section. Samples exclude protected/excluded areas and fit a total 2,000 serialized UTF-8 bytes. They cannot become additional patch targets. Legacy `document_text` requests receive no new style samples. Supported changes, protected content, atomic validation and stale-source HTTP 409 behavior are unchanged. See [the edit handoff](DOCUMENT_EDIT_API_HANDOFF.md).

## Context limits and scope

Default Ask resolves numbered sections and quoted section titles first. For other questions it provides the complete source or returns `needs_context` without running a model. To answer a local question against a larger source, set `document_scope="auto"`, `"focused"` or `"selection"`. For global analysis, set `document_analysis="summary"` or `"contradictions"`. A deployment can make `auto` the behaviour of the default scope with `DOCUMENT_LARGE_ASK=auto`. See [Long-document Ask](#long-document-ask).

Scoped Ask uses exact source chunks and a request-local in-memory SQLite FTS5 index with BM25 ranking. It supports Unicode terms and CJK bigrams; it is lexical retrieval, not embedding similarity. Deployed Python must include FTS5. Unavailable search, no matching passages or an unpackable selection returns `needs_context` without a model call. Captured focus may supply a passage even without a lexical match. Retrieved references have server-assigned `r0`, `r1`, … identifiers, exact quotes, heading paths and UTF-16 ranges. Partial answers must remain scoped to those passages; retrieved evidence does not establish whole-document coverage.

Draft uses complete context when it fits. Otherwise it includes a bounded heading outline plus the active section or a window around the captured focus. With no focus, it continues from the end. Its metadata reports `coverage="partial"`, and warnings explain that limitation.

The context envelope has a 24,000 UTF-8-byte ceiling and a 6,000 estimated-token allocation for a whole document (scoped Ask has its own allowance, `DOCUMENT_RETRIEVAL_BUDGET_BYTES`). The current dependency-free estimator deliberately counts each UTF-8 byte as one estimated token, including serialization and metadata. This is conservative and usually binds before the byte ceiling: it is **not** 6,000 measured provider tokens or 24,000 characters of prose. Non-English/emoji-heavy text can fit fewer characters. The task is bounded separately; output defaults use the existing rounding/reasoning-budget helper, starting from 2,000 visible tokens for Ask and 1,000 for Draft.

No provider token-count request is added to the hot path. Before production enablement, validate the configured model IDs/capacities and synthetic grounding/latency behavior. Increase the packing allowance only after a provider-calibrated counter/estimate is tested for every eligible fallback. When reported output usage reaches the configured model output cap, the response includes `output_limit_reached` with a warning that the text may be incomplete. The installed agent SDK does not reliably expose every provider finish reason, so missing usage cannot establish that an answer was complete.

## Long-document Ask

When Ask specifies one section by number or quoted title, such as “Summarize section 4” or `Summarize section "Risk Factors"`, the backend resolves the heading before retrieval. It sends only excerpts from that section, including any subsections, up to the existing retrieval budget. A missing or ambiguous heading returns `needs_context` with no model call. This applies even when the document fits in full or the caller omits the scope. With explicit `selection` scope, Ask uses the overlap with the named section. A selection outside that section returns `needs_context`.

For other Ask requests, the scope behaves as follows.

| What you send | Document fits | Document does not fit, question is about a part | Document does not fit, question needs the whole document |
|---|---|---|---|
| default scope (`DOCUMENT_LARGE_ASK=refuse`, default) | read whole | `needs_context`, no model call (as before) | `needs_context`, no model call (as before) |
| `document_scope: "auto"` | read whole (`strategy: "full"`) | answered from the matching passages (`strategy: "retrieval"`, `coverage: "partial"`, `scope: "focused"`, `references` r0, r1, ...) | `needs_context`, no model call, advice below |
| default scope with `DOCUMENT_LARGE_ASK=auto` | same as `auto` | same as `auto` | same as `auto` |

`auto` is Ask-only: with `draft`, `edit` or `document_analysis` the request is `400`; with `document_action: "auto"` it runs as Ask (`source: "request"`). The resolved choice is in the metadata (`strategy`, `scope`, `coverage`), and a partial answer always carries the `partial_document_context` warning.

**Which questions need the whole document** (`src/document_question.py`, rules only, no model, English wording): summaries, overviews, "main points / key takeaways / central arguments", "what is this paper about", enumerations of everything ("list all", "extract every", "how many times", "count"), contradictions and inconsistencies, reviews and feedback ("proofread", "critique", "is this too long"), comparing sections. Those cannot be answered from the passages that happen to match some words, and an answer built from a sample reads like a review of the whole, so they are refused at once with *"This question needs the whole document, which is too large to read in one request. Ask about a specific part (name a section or select the text), or use the document summary or contradiction analysis, which reads every part; no document analysis was run."* (nothing billed). Suggested UI: when `status="needs_context"` for an `auto` request, offer the Summarize / Find contradictions actions (`document_analysis`). Anything else is treated as a question about a part, so wording in another language gets a partial answer with the warning rather than a refusal. Checked against 39 whole-document and 27 part phrasings written for the rules (all classified as intended; optimistic, since the author wrote both) and the 39 benchmark questions below (none flagged).

**Retrieval changes** (`src/document_retrieval.py`; they apply to `focused` and `selection` too):
- Porter stemming ("selling" finds "sell", "layers" finds "layer").
- The context sent is bounded by `DOCUMENT_RETRIEVAL_BUDGET_BYTES` (default `10000`) instead of a fixed 6,000, still under the 24,000-byte ceiling. **This is also the most a scoped Ask costs in input**: `billable_input_chars` (read it, as for any Ask) is about 10,000 for a long-document answer against about 6,000 before.
- With `auto`, a bare cursor no longer puts its passage first (a cursor says where the user is typing, not what the question is about); an explicit selection still does.

**Measured.** 39 questions over 7 real documents imported with the import endpoint (the ResNet and Transformer papers as PDFs, the GNU "free software" essay, Python's `venv`, class tutorial, `argparse` and `asyncio` task docs; 26-88 KB, i.e. 4-15 times the old limit): 24 about one fact, 10 worded differently from the text, 5 needing two passages. Share of questions whose answer-bearing passage was in the context sent:

| Configuration | Passage found |
|---|---|
| before (6,000 bytes, no stemming) | 72% (28/39) |
| stemming only, 6,000 | 79% |
| 8,000 | 90% |
| **10,000 (default)** | **95% (37/39); facts 23/24, two-passage 5/5, reworded 9/10** |
| 11,000 / 12,000 / 24,000 | 97% / 97% / 97% (no further gain) |

End to end with a real model (the same 39 questions through `/agent/process`, key facts in the documents changed so a model cannot answer from memory; answer checked by pattern): OpenAI 36/39 before, 37/39 after; Claude 37/39 before, 38/39 after. **That is one question each, within the noise of 39 questions**: most facts are stated in more than one place, so the old 3-passage context often still contained one of them. Read the table above as "the context contains the evidence more often", not as "answers are 20 points better". What the change reliably buys is that a long document can be asked about at all, at the cost shown. The remaining miss in both providers' runs is a question whose wording shares no word with its answer passage ("why did the extremely deep CIFAR network perform worse" against "We argue that this is because of overfitting"): retrieval is lexical (BM25 on stemmed words), not semantic, and a model-assisted query rewrite or embeddings would be the next step for that kind of gap.

**Live behaviour** (58 KB paper, OpenAI): a part question gave a correct answer in 1.7-1.9 s with 7-8 references and `billable_input_chars` about 9,900; "Summarize this paper" was refused in 0.0 s with nothing billed; a 3 KB document was read whole; a question matching nothing was explained without a model call.

**Whole-document questions: the analysis route, and a fix.** `document_analysis: "summary" | "contradictions"` reads every chunk, and was the only route for those questions. On every real document tried it failed with `500` ("The document request could not be completed"), with the old code as well: the extraction stage allows a 120-byte quote and models cannot count bytes, and one over-long quote anywhere aborted the whole analysis. `src/document_analysis.py` now repairs *length* overflows only (a quote or statement cut at a word boundary, at most two claims per chunk, at most eight findings, four distinct evidence ids): a prefix of an exact quote is still an exact quote, and the exact-membership check still runs. Wrong content still fails closed (an invented quote, an unknown evidence id, missing verification decisions), as the existing tests require. Live, summary analysis went from `500` to `200` on the three documents tried: 58 KB paper 28 s, 18 calls, 73,700 input chars; 39 KB paper 19 s, 13 calls, 50,500; 88 KB reference page 35 s, 26 calls, 110,700 (billing as documented under *Billing and privacy*: it is a very different cost from a 10,000-character scoped Ask, which is why `auto` does not start it by itself).

**Limits.** Retrieval is lexical; a question whose words are not in the passage can miss it, and a partial answer is only as good as the passages (the model is told to say when the excerpts do not contain the answer, and to cite `[r0]`...). A question needing two distant facts worked 5/5 in the benchmark but depends on both ranking in the top passages. `needs_context` for a whole-document question is deliberate, not a failure. The whole-document wording rules are English only. Documents over 100,000 characters are still refused at upload/request validation.

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

When the document starts with YAML or TOML front matter, "at the top", "at the beginning" and "at the start" place the draft just below the front matter (front matter has to stay the first thing in the file); if the front matter is the only block, the draft goes after it.

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
| "next to / beside / alongside X" (X as in the rows above) | `after` that block |
| "below/above the heading", "under this heading", "next to the heading" (no name) | the heading the cursor is under; the only heading when the document has one; otherwise a question |
| "below/above the title", "after the document title" | the document's top-level heading (a document that opens with a heading has that as its title) |
| "next to the Top 10 list", "below the Pricing table" | the one block of that kind under the heading of that name, or the one whose text has the name |

"Top", "start", "end" and "bottom" mean an edge of the document or section when the sentence goes on after them ("at the end
summarizing the key points", "at the top, please") but not when they begin a name written with a capital letter or a digit, and not after
"next to"/"close to" unless "of …" follows. In a name they do not: "next to the Top 10 list" is read as the list, not as the top
of the document, and "close to the End of Year report" is not the end of anything.

Locations are understood in English only; other languages fall back to the cursor. No named place: `placement` is absent; insert at the cursor as before. A named place that is missing or matches more
than one block gives `status="needs_clarification"` with the question in `message`, **no `content` frames** and no
model call or charge (for example "There is no heading named 'Risks'. Name an existing heading, or place the cursor
where the draft should go."). Metadata now also carries `message` for `needs_context`.

### Draft with exact text

When the task supplies the words to insert, Draft returns exactly those words and nothing else ("add Acme Inc. below the Contact
heading" inserts `Acme Inc.`, not a paragraph about Acme). Recognised in code, without a model call, when the text is quoted
(`add "Built to last." below the About heading`) or is a short item with a digit, `@`, brackets or a currency sign, or Title Case words
(`Acme Cloud`, `555-0100`, `hello@acme.com`, `(2024)`, `Springfield, IL`, `Draft v2`). Such an answer arrives as one `content` frame,
costs nothing and has `document_context_ready.source = "literal"` (`"model"` when a model writes the draft; `null` for Ask and whenever no
draft is produced, i.e. `needs_context` and `needs_clarification`; the field is new and additive). The words must be followed by nothing but the
place ("… below the Contact heading", "… below Pricing" when Pricing is a heading, "… at the end of the document"): "… in bold",
"… twice", a count of things ("add 3 hashtags", "add 5 Stars", "up to 5 slogans", "top 10 tips"), an instruction before the place ("add 5 stars in bold below …"), two quoted items, or a place that is anything but one anchor noun at the end of the line are not literals. With the verb "write" only a quote, a figure, an address, a bracket or a currency sign counts ("write Haiku below …" asks for a haiku). Such a request goes to the model, which applies asked formatting to the supplied text only ("in bold" -> `**text**`, "as a heading" -> `# text`) and otherwise returns it plain. Anything that
asks for writing ("add a short bio …", "add a summary …"), names more to do ("… and make it bold") or is not clearly a literal goes to
the model as before, whose Draft prompt carries the same rule in both directions: text the task *supplies* is the whole answer (no
description, no prefix such as `Phone:`, no markup, no placeholder), and text it names a kind of ("add a note about Acme Cloud …") or a
topic for is *written*, never replaced by the bare name, from the document's facts and the task's only: plain wording where that suffices, and a short `[bracketed placeholder]` only for a specific detail it cannot do without (a person's role, a date, a figure).

## Auto mode (`document_action: "auto"`)

Send `document_action: "auto"` (with `document_context`, like any action) and the agent decides whether the task is **Ask**, **Draft**,
**Edit** or **Write**, runs that mode, and says which one it chose. Everything the chosen mode returns is exactly what an explicit request for
it returns; the only addition is the report below. Explicit `ask`/`draft`/`edit` and the default (Ask) behave as before and send no report.

**How it decides**, cheapest first:

1. *The request.* A `document_scope` other than `document` or a `document_analysis` can only be served by Ask: `source: "request"`.
2. *The wording (no model, instant).* The rules read at most 900 characters (a long task is judged on its first and last ~450). They answer only when the wording points one way, never on a mix, and Edit only when an edit verb *begins* an instruction ("Polish bakery traditions", "correct me if I'm wrong, but …" and a story about a robot who "wants to rename itself" are not edits). "Suggest names for …" is generative and goes to step 3: a question or request for advice
   is Ask; new text ("add … below …", "continue writing", "add a new section") is Draft; a change to existing text ("make … shorter",
   "fix …", "rename the heading", "add (2024) next to the title") is Edit; a standalone piece (blog post, code, image, chart, design, the
   things Write itself routes on) that does not point at the document is Write. An empty document cannot be edited, so an Edit-looking task
   there is answered by Ask's "There is no document content…".
3. *One light-tier model call* when the rules cannot tell (40-55% of the tasks in our sets): Claude Haiku 5.5, or the request's own provider
   light model for a BYOK request (`gpt-4o-mini`, the Grok non-reasoning model, Gemini Flash-Lite). Its input is the task (the same 900-character digest) and a one-line hint such as `[document: 120 words; no selection]`; the document itself is not sent. Same key rules as
   the code-intent classifier: platform users go through the platform's Claude key, BYOK users stay on their own key and provider. It is
   not charged to the user (traced as usage like the code classifier). A burst waits up to the timeout for one of 16 classifier slots
   rather than being turned into Ask at once. Timeout `DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS` (default 4 s,
   longer than the code classifier's because a wrong mode costs the user more than a slower right one);
   `DOCUMENT_ACTION_CLASSIFIER_ENABLED=false` skips it (see [Configuration](#configuration-document-actions)).
4. *Ask.* If that call fails, times out or answers something else, the task runs as Ask (it only reads, so a wrong guess costs a re-ask,
   not a change to reject) and the report says `ambiguous: true`, `source: "fallback"`.

**The report.** Stream: a named event, sent before anything the chosen mode sends (right after `start`):

```text
event: document_action_resolved
data: {"requested_action":"auto","action":"draft","source":"rules","reason":"new_content","ambiguous":false}
```

`/agent/process` returns the same object as `document_action_resolved` (`null` for every request that did not send `auto`).

| Field | Values |
|---|---|
| `action` | `ask`, `draft`, `edit`, `write`: the mode that ran |
| `source` | `rules`, `model`, `request` (scope/analysis), `fallback` (no answer; ran as Ask) |
| `reason` | `question_or_advice`, `new_content`, `existing_prose`, `standalone_task`, `scope_requires_ask`, `empty_document`, `model_choice`, `classifier_unavailable`, `mixed_or_unclear_task` |
| `ambiguous` | `true` when it fell back to Ask because it could not tell. Suggested UI: show the answer, with a "Use Draft/Edit instead" offer |

After the report the stream is the chosen mode's own: Ask/Draft send `document_context_ready` then content; Edit sends
`document_edit_ready`, one verified snapshot (buffer it, as for explicit Edit) and `done`; Write sends its usual frames
(`needs_context` is never produced by Auto). Handle each by `action`, and do not insert anything before the report has arrived.

Notes for the editor:

- **Write** runs the plain prompt: the document is **not** sent to it and `document_context` in the response is `null`. Use this for
  "write a blog post…", code, images, charts and designs; they come back as they do from the Write tab.
- **Edit** gets the checks an explicit Edit gets, so an unparseable or over-400-block document is HTTP 400 (`too many blocks…`) and an
  instruction over 2,000 characters is HTTP 400, before any model work. Auto accepts a `query` of up to 8,000 characters otherwise.
- `suggest_document_intent` is ignored with `auto`: the report replaces the advisory suggestion (no `document_intent_suggested`).
- Billing is by the mode that ran. A literal Draft and clarifications cost nothing; the decision call is not billed.
- Measured (82 sample tasks plus 40 held-out phrasings written after the prompt, `scripts/document_actions/eval_auto.py`): the rules decided
  45-60% of tasks, with 100% agreement with the intended mode; end to end the right mode was chosen 100% (Claude, Grok), 97.5-100% (Gemini)
  and 95-99% (OpenAI `gpt-4o-mini`; its misses were Draft-or-Write borderline tasks such as a job description). The model call took a
  median of 0.8-1.4 s (p95 1.2-2.3 s) on top of the chosen mode; rule-decided tasks add no time.

### Configuration (document actions)

Every number the Auto, exact-text Draft and in-line Edit features added lives in one block of `config.py` and can be overridden by an
environment variable (listed, commented out, in `.env.example`). Leave a variable unset for the default; a blank value, or a value outside
the range, stops the service at start-up with a clear error.

| Variable | Default | Range | What it does |
|---|---|---|---|
| `DOCUMENT_ACTION_CLASSIFIER_ENABLED` | `true` | true/false | `false`: wording the rules cannot settle resolves to Ask, with no model call |
| `DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS` | `4.0` | 0-30 | longest wait for the Auto decision call (and for a free slot) |
| `DOCUMENT_ACTION_CLASSIFIER_CONCURRENCY` | `16` | 1-64 | decisions that can wait on the provider at once; read at start-up |
| `DOCUMENT_ACTION_DIGEST_CHARS` | `900` | 200-940 | how much of a task Auto reads (first and last half); also bounds the work done before the licence check |
| `DOCUMENT_LITERAL_MAX_REQUEST_CHARS` | `400` | 50-2000 | a Draft request longer than this is never treated as exact text |
| `DOCUMENT_EDIT_SMALL_INSERTION_CHARS` | `80` | 1-500 | `/agent/documents/apply` accepts new numbers only in a pure insertion up to this long |
| `DOCUMENT_EDIT_SMALL_ADDITION_MAX_WORDS` | `12` | 1-100 | most words an in-line Edit addition may add to a block |

### Configuration (long-document Ask)

| Variable | Default | Range | What it does |
|---|---|---|---|
| `DOCUMENT_RETRIEVAL_BUDGET_BYTES` | `10000` | 4000-24000 | Most UTF-8 bytes of context a scoped Ask (`focused` or `auto`) sends to the model. Credits are priced by characters sent, so this is also the most a scoped Ask can cost in input. |
| `DOCUMENT_LARGE_ASK` | `refuse` | `refuse` / `auto` | What an Ask with the default scope does with a document too large to send whole. `refuse` (the documented contract): `needs_context`, no model call. `auto`: act as `document_scope="auto"`. |


Not configurable on purpose: the 2,000-character Edit instruction limit (the existing Edit contract) and Gemini's 10-second minimum request
deadline (a provider fact).

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
