# Highlight key points text action

Add a **Highlight key points** action alongside Correct grammar and the other text tools. It takes one paragraph
or a short text portion and returns exact key-point excerpts plus ranges for highlighting in place. It does not
summarize, reword, correct grammar or change any original text. The frontend may also show the excerpts as a list.

## Request

Use the existing `/agent/process` or `/agent/stream` authentication and provider fields:

```json
{
  "type": "highlight_key_points",
  "query": "The launch is scheduled for 15 November. Testing must finish by 10 November. The approved budget is USD 20,000. This note is for internal planning.",
  "model": "openai",
  "api_key": "<existing endpoint credential>"
}
```

Send the exact plain text of the selected portion/current paragraph, at most 10,000 characters and not blank.
Do not send `secondary_prompt`, `document_text`, design data or image settings/reference images. These are HTTP
400 errors. No new feature flag is needed. This explicit type always uses the dedicated highlighter, even if
the passage contains instructions such as “fix grammar” or “draw an image”. Tools are disabled and the model
uses the basic/fast tier, like grammar correction, with the existing provider eligibility/fallback rules.

`/agent/stream` takes the existing licence key. Server-to-server `/agent/process` takes the provider key resolved
by the application backend. Do not expose provider keys in the frontend.

## Response

Existing `markdown_response` and `html_response` contain the **exact unchanged input**. The additive
`highlighting` field carries the actual formatting operation:

```json
{
  "success": true,
  "markdown_response": "<exact original text>",
  "html_response": "<exact original text>",
  "highlighting": {
    "source_sha256": "<SHA-256 of the original UTF-8 text>",
    "key_points": ["<exact source excerpt>"],
    "spans": [
      {"start_utf16": 0, "end_utf16": 20, "text": "<actual matching source text>"}
    ],
    "highlighted_html": "<span style=\"white-space:pre-wrap\">Original text with <mark>key phrases</mark>.</span>"
  }
}
```

Values above are illustrative. Real spans are source ordered, unique, non-overlapping and at whole-word
boundaries. Offsets are zero-based UTF-16 code units relative to `query` (not the complete editor document),
compatible with JavaScript string slicing and emoji. `key_points` contains the same excerpts in source order.

The model selects up to five meaningful phrases, normally well below half the passage; Python validates
every excerpt against the source. Very short passages can consist entirely of key facts, so there is no
arbitrary coverage cutoff that would discard essential details. Repeated excerpts require a longer
unique phrase instead of guessing an occurrence. No meaningful points is a successful empty list/spans with
the unchanged source, not a forced highlight. The model's judgment of importance is not a guaranteed complete
editorial summary; the excerpts and the original context are retained for review.

The HTML preview escapes all input and adds only server-owned `<span>`/`<mark>` elements. It is a standalone
plain-text preview, not an editor document replacement. `html_response` retains its existing API meaning and
does not contain this preview. Other request types return `highlighting: null`.

## Streaming

The server validates the complete selection before emitting formatting or text:

```text
data: {"type":"start","request_id":"..."}

event: highlight_ready
data: {"source_sha256":"...","key_points":[...],"spans":[...],"highlighted_html":"..."}

data: {"type":"content","content":"<exact unchanged input>"}

data: {"type":"done",...}
```

Handle named `event:` frames as well as `payload.type`. Do not append the content at the caret or insert a second
copy of the paragraph. Buffer the result, wait for successful completion, verify the source and apply formatting
to the original captured range. Abort/disconnect/error/credit-limit outcomes must not apply partial formatting.

Invalid model selections get one repair. If both attempts fail validation, process returns HTTP 422 and stream
returns `error` without highlights, content or `done`. Provider failures retain the normal API error behavior.
Existing text credit accounting applies; there are no image or design charges.

## Frontend changes

1. Add the toolbar/menu action and send `type: highlight_key_points` with the selected text or current paragraph.
   Capture the editor range/node IDs and current text when the action starts; cursor movement must not redirect
   the result to a different paragraph.
2. Consume `highlighting` / `highlight_ready`. Confirm that the captured range still contains the exact source
   text and matches its fingerprint. Discard stale results after typing; never locate the range by a global text
   search, since identical text can occur elsewhere in the document.
3. Map each relative UTF-16 span to that captured editor range and apply the editor's highlight/background-color
   format. Preserve original wording, links, styles and node metadata; do not replace the paragraph with HTML
   or Markdown. Group the formatting changes as one undo step and save through the usual document mechanism.
4. Optionally render `key_points` as a short list. For a separate preview, use `highlighted_html` or construct
   text and `<mark>` nodes from the spans; never interpret the original input as HTML.
5. Show “No key points found” for empty spans. Keep existing content/formatting on failure. Test stale ranges,
   moved cursors, emoji, line breaks, repeated phrases, existing formatting and undo/save/reload.

## Reproduction

Run `.venv/bin/python -m pytest tests/test_key_points.py` for validation, HTML safety, Unicode/range integrity,
HTTP/SSE, repair limits, cancellation, routing and error cases.

Run `.venv/bin/python scripts/demo_key_points.py` for a provider-free production-flow example or add `--live`
to call the configured OpenAI provider on three synthetic text portions. The demo disables Redis, storage and webhooks
and writes a before/after preview plus redacted receipts under `local_run/key_points_demo/{scripted,live}/`.

The Python backend is implemented. Frontend toolbar integration, editor formatting and persistence must be
implemented and tested in the frontend repository, which is not available in this workspace.

Ponytail review checked the routing map and both callers, the fast-tier/no-tools configuration, exact excerpt
validation, bounded repair/cancellation, source-preserving responses and HTML escaping. Regression cases cover
dense short passages, contractions, quoted words, combining accents and emoji offsets. No unresolved backend
review findings remain. Live examples demonstrate specific selections, not a statistical importance-ranking benchmark.
