# Editing document sections without a selection

The Python backend can locate paragraphs from a user's prompt and return a reviewable editing proposal for the
current complete document. No cursor or selection is required. Untargeted text is copied from the original source;
the model never returns or rewrites those regions. The frontend still owns uploads, document access, review,
acceptance, saving and undo.

## Request contract

Use the existing authenticated `/agent/process` or `/agent/stream`. Add `document_text` and put the instruction in
`query`. Supported types are `general` and `enhance_with_ai`. Existing calls without `document_text` are unchanged;
this field explicitly enables document editing and takes priority over image, code and design intent routing.

```json
{
  "type": "general",
  "query": "Make Introduction more engaging and remove repetition in Benefits. Leave Pricing and Conclusion unchanged.",
  "document_text": "# Guide\n\n## Introduction\n\nThe original opening.\n\n## Benefits\n\nThe original benefits.\n\n## Pricing\n\nUSD 20 per month.\n\n## Conclusion\n\nThe original conclusion.\n",
  "model": "openai",
  "api_key": "<existing endpoint credential>"
}
```

- `document_text`: exact current plain text or Markdown, 1–100,000 characters, at most 400 parsed blocks.
- `query`: editing instruction, 1–2,000 characters. Omit `secondary_prompt` in this mode.
- Headings help but are not required: “remove repetition from the paragraph about onboarding” can be located by meaning.
- No feature flag is required: only requests that explicitly include `document_text` use this path.
- Reject incompatible request types, HTML documents, empty input, invalid Unicode, reference images, image settings
  and `existing_design_ast` with HTTP 400. File/PDF/DOCX upload parsing is not implemented by this feature; extract
  the document to Markdown/plain text through the application's import flow first.
- Keep existing authentication: `/agent/stream` takes a licence key; server-to-server `/agent/process` takes the
  provider key already resolved by the application backend. Never expose provider keys in the editor.
- `document_id` still does not retrieve a document. Always send the actual authorized current snapshot.

## What the backend does

1. Parse exact source ranges and headings using the existing Markdown parser. Whitespace, tables, images, code, raw HTML
   and reference definitions are protected. Paragraph/list blocks up to 12,000 characters can be targeted. Oversize
   paragraphs stay protected rather than being truncated. **Headings, titles and plain-text quotes can be targeted too, but
   only when the instruction is about them** ("make the title shorter", "rename the Contact heading to Get in touch",
   "shorten the quote"): "the title" is the document's top-level heading; "all the headings" offers every heading; words
   inside a clause that says what to leave alone ("leave/keep/preserve/skip/ignore …", "do not/never/avoid …", "without/except …") do not count. The clause runs to the next "but/then", to an "and" or
   comma that starts a new instruction, or to the end of the sentence: "leave the title and the quote alone" protects both, "keep the quote but shorten the title" offers the title,
   and a title that is kept by name is not offered even when "the headings" are asked for. A general request ("make the intro
   shorter") never offers the heading above it. A heading stays one heading line at the same level and cannot be emptied (a changed level or an empty heading is
   refused). A quote is rewritten as its own text and gets its `>` markers back line for line, so its line count cannot
   change; quotes with nesting, code, tables or images inside stay protected.
2. Send the full block catalog and request to the model, which selects at most 12 target blocks and explains why.
   Its supporting request quotes must be literal excerpts of the user's instruction. IDs must exist and be editable.
3. Enforce literal named-heading scope and explicit exclusions independently of the model. Repeated section headings
   require a unique parent or an explicit request for all occurrences. Unsupported or ambiguous requests return a
   clarification with no edits. A broad request can identify weak paragraphs and propose improvements for review.
4. Refine only selected blocks. Existing freeze/verify machinery protects links and inline elements, paragraph/list
   structure and locked content. Numeric sequences and common currency/percent tokens must remain exact.
   Line-ending sequences and a leading BOM are preserved. One repair is attempted for an invalid replacement.
5. Validate all replacements and replay them against the original snapshot. Any failed replacement discards the
   entire proposal; already computed replacements are never partially emitted. All untargeted slices remain exact.

Target selection by meaning is a model judgment; it is not a guarantee of perfect intent or factual understanding.
Named-section guards and byte preservation are deterministic. Review is required before acceptance. This mode
polishes existing prose, not arbitrary fact changes, table edits, heading renames, new sections, charts or external
research. Worded quantities and broader factual meaning still need editorial review; the numeric guard checks
digit sequences and common currency symbols/codes, not every possible semantic fact.

The model uses the request's eligible provider candidates and existing fallback/BYOK restrictions. Tools are
disabled. Each call uses `LONGFORM_CALL_TIMEOUT_SECONDS`; a proposal makes one planning call plus one call per
selected block, and at most one repair per block. Calls are sequential and request state is local, not global.
Abort prevents further calls after observed cancellation; an already running provider call may finish, but its
result is discarded. Existing usage accounting includes planning/refinement calls.

## Response contract

`/agent/process` adds `document_edit` to its existing response. `markdown_response` and `html_response` contain
the complete verified proposed document, or the exact original for clarification, no change or failure. Here
`html_response` retains the existing API naming; the content is Markdown/plain text, not rendered HTML.

```json
{
  "success": true,
  "markdown_response": "<complete proposed document>",
  "html_response": "<same complete proposed document>",
  "document_edit": {
    "status": "proposed",
    "base_sha256": "<SHA-256 of original UTF-8 document>",
    "result_sha256": "<SHA-256 of proposed UTF-8 document>",
    "message": "The requested paragraphs can be improved.",
    "targets": [
      {"block_id":"b4_012345abcdef","request_quote":"Make Introduction more engaging","reason":"The opening needs a clearer hook."}
    ],
    "patches": [
      {
        "block_id":"b4_012345abcdef",
        "start_utf16":28,
        "end_utf16":50,
        "before":"<exact original block>",
        "after":"<verified replacement block>",
        "reason":"The opening needs a clearer hook."
      }
    ]
  }
}
```

The example offsets and fingerprints are illustrative; use the real values returned by the API.

| Status | Meaning | Frontend action |
|---|---|---|
| `proposed` | One or more verified replacements are ready | Highlight changes and offer acceptance |
| `needs_clarification` | The intended target/action could not be identified safely | Display `message`; request a more specific prompt |
| `needs_clarification` (new facts) | The request wants details, examples, facts, statistics or sources the text does not contain ("add more detail to the Benefits paragraph", "include two examples", "elaborate on pricing"). Edit may not invent facts, so no model call is made | `message` is `Edit only rewrites what is already there and can't add new details, examples or facts. Use Draft to add them.` Offer Draft. Style requests ("add more warmth", "make it longer") are still rewrites |
| `needs_clarification` (restructure) | The request deletes, moves, merges or splits whole sections, paragraphs, tables or lists ("delete the Conclusion section", "move the table below the intro"). No model call | `message` is `Edit only rewrites existing text; it can't delete, move, merge or split sections, paragraphs or tables. Make that change directly in the document.` |
| `proposed` (small addition) | The request adds a short item **inside an existing line**: a bracketed or quoted literal, a figure, or a named kind of detail, placed next to, before or after a word or phrase ("add (2024) next to the title", "add a phone number after the address", "add Inc. after Acme"). It goes through the planner and the block writer like any rewrite; the patch replaces that one block (a heading, paragraph, list or quote). The writer inserts exactly the text the user wrote, on the same line, and nothing else; this is checked in code: every other word of the block must still be there in order (punctuation may move) and at most 12 words may be added, or the writer is sent back once and then the proposal is `failed`. A request that also rewrites ("make the text shorter and add (2024) next to the title") is not held to that: its writers may change words, may use the numbers the user typed, and are told which part is the in-line addition. If the user names a kind of detail without a value, the patch contains a bracketed placeholder such as `[Phone number]` (never an invented value) | Review and accept as usual. Highlight placeholders so the user fills them |
| `needs_clarification` (insert) | The request asks to add or insert new content ("add X below the heading", "insert a table after …", "add a new section"). Edit only rewrites existing blocks, so no model call is made. In-line changes such as "add a comma after 'However'" or "add bold to the key phrase" and the small additions above are rewrites and still go to the planner. Anything placed *below/under/above* a block, or after a section, table, list, paragraph or image, is new content ("add a phone number below the address", "add (2024) after the table") | `message` is exactly `Edit only rewrites existing text; use Draft to add new content.` Offer to switch to Draft |
| `no_change` | Nothing was rewritten: the text already fits, or the request needs material Edit may not invent (for example "expand the conclusion" when there is nothing more to say without new facts) | Keep the document unchanged; show `message` ("No changes were proposed: … Use Draft to add new content.") |
| `failed` | A provider/validation failure prevented a safe proposal | Keep the document unchanged; show retry |

Non-proposed results have no patches and identical original/result hashes. HTTP `success: true` means the request
completed; always read `document_edit.status` to decide whether there is an editable proposal. Public failure
messages do not expose provider errors or keys. Existing non-document requests return `document_edit: null`.

### Streaming

The document is buffered and validated on the backend. Successful request sequences are:

```text
data: {"type":"start","request_id":"..."}

event: document_edit_ready
data: {"status":"proposed","base_sha256":"...","result_sha256":"...","message":"...","targets":[...],"patches":[...]}

data: {"type":"content","content":"<one complete verified snapshot>"}

data: {"type":"done",...}
```

Clarification/no-change/failed proposals use the same sequence, with the exact original snapshot. Parse named
`event:` frames as well as existing `payload.type` messages. In this mode **buffer content; never insert it at
the caret or append it to the document**. Wait for `done`, verify hashes and show a proposal. Do not apply patches
after an error, abort, disconnect, `credit_limit_reached` or incomplete snapshot, even if metadata already arrived.

Platform credit checks include both instruction and complete document context. An Edit is charged only when it
returns `proposed`: input is the instruction plus the document, output is the text of the rewritten blocks (the
patches' `after`), not the whole snapshot the stream carries. `no_change`, `needs_clarification` and `failed` charge
nothing. Upfront reservation also estimates
one document of output; insufficient balance or an excessive per-request credit reservation returns HTTP 402
before model work. `REQUEST_CREDIT_LIMIT` asks for a smaller document. BYOK restrictions remain unchanged.

## Numbers in small additions

Edit never changes or removes a number, and it never adds one the user did not type. For a small addition the numbers in the
instruction are allowed to appear once each ("add (2024) next to the title" may add `2024`); an invented number
(`555-0100` after "add a phone number") or a different year fails the check, gets one repair attempt, and then returns `failed`.
`/agent/documents/apply` has no instruction to compare with, so it accepts new numbers only in a **pure short insertion**: the
original block with up to 80 characters put in and nothing removed or reworded. A patch that rewrites text still must keep every
number, as before.

## Applying accepted edits

The pure Python endpoint `POST /agent/documents/apply` makes no model calls and stores nothing:

```json
{
  "document_text": "<current exact document snapshot>",
  "base_sha256": "<proposal's base_sha256>",
  "patches": ["<actual patch objects returned by the proposal>"]
}
```

The placeholder array above represents objects, not strings. Response:
`{"document_text":"<patched document>","result_sha256":"<new hash>"}`.

- HTTP 409: the supplied current document differs from the proposal's base; request a new proposal.
- HTTP 400: invalid IDs, changed `before`, forged offsets, duplicate targets, unsafe replacements, protected
  content changes or invalid input. Nothing is persisted or partially applied.
- IDs are **snapshot-bound**, not permanent editor node IDs. Positions are zero-based UTF-16 code-unit offsets,
  compatible with JavaScript string slicing even when text contains emoji. Do not treat them as UTF-8 byte offsets.
- Every patch replaces a complete editable source block. Blank lines and other blocks are copied untouched.
- The endpoint validates input snapshots, not user authorization. The application backend must enforce document
  ownership and use its normal save/version checks. Hashes are stale-snapshot checks, not signatures.

## Frontend changes

1. Import an uploaded document through the existing application flow. Serialize its authorized full current
   content to stable Markdown/plain text and capture the current editor/document version.
2. For an instruction to edit existing prose without a selection, send `document_text` plus the instruction as
   `query`; send no cursor or selection fields on this legacy edit path. Questions and new drafting use the separate
   [document-context contract](DOCUMENT_CONTEXT_API_HANDOFF.md), which is enabled by default. New
   `document_action="edit"` requests also use this same patch flow. Keep legacy selection-based operations unchanged.
3. Buffer document-mode output. Show targeted paragraph highlights, before/after text and reasons. Display
   clarifications in the chat. Provide Accept and Dismiss; proposals do not save automatically.
4. Before acceptance, confirm the current serialized document still matches `base_sha256`. Call `/apply` through
   the backend or perform equivalent whole-snapshot/offset/`before` checks locally. Apply replacements from last
   offset to first if working directly on the original string, then check `result_sha256`.
5. Map source ranges to editor nodes. Apply accepted replacements as one normal undo step and save through the
   application's existing document APIs. Preserve unrelated formatting/node metadata. Discard proposals whose
   source changed; never apply an old result over newer typing. Keep editable graph/design nodes out of prose
   serialization or represent them as protected blocks; do not regenerate their datasets during text refinement.
6. Confirm no-selection edits, explicit exclusions, missing targets, duplicate headings, emoji, reload/undo and
   concurrent typing. Backend exact Markdown preservation does not by itself prove that the editor retains rich
   formatting when it converts text back into nodes; that frontend integration must be tested separately.

## Reproduction and evidence

- `.venv/bin/python -m pytest tests/document_edit`: targeting, protected content, scope, atomic failures, stale
  application, exact line endings, Unicode/emoji offsets, provider fallback/BYOK, HTTP/SSE and credit checks.
- `.venv/bin/python scripts/demo_document_edit.py`: scripted, provider-free production-flow demonstration.
- `.venv/bin/python scripts/demo_document_edit.py --live`: real configured OpenAI calls on synthetic documents.
  Redis, licence calls, storage and external webhooks are disabled in the demo.
- Results: `local_run/document_edit_demo/{scripted,live}/example.html`, `receipts.json`, `proof.json`. The proof file
  compares every untargeted output region independently and records before/after SHA-256 hashes. The demo also
  replays accepted patches and verifies a changed source is rejected with HTTP 409.

These examples verify specific inputs and invariants; they are not a statistical benchmark of intent accuracy.

## Style reference for explicit context Edit

New `document_context` + `document_action="edit"` requests give the existing block writer up to three
read-only untargeted prose samples (2,000 serialized UTF-8 bytes total), preferring the same section and
excluding protected/excluded sections. Samples guide tone and sentence style only. Target selection,
source hashes, protected structures, numeric checks, exact replay and apply behavior are unchanged.
Legacy `document_text` requests keep their existing prompts without the additional style samples.
