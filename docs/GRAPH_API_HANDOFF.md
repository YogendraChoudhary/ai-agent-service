# Editable graph suggestions after Enhance

The Python implementation improves the document as before, then optionally returns one graph suggestion derived from
the **original input**. It never inserts graph markup into the prose. The frontend must implement preview, Insert,
Dismiss and a persistent editable graph block. No frontend repository was available for this change.

## Enablement and compatibility

- Set `CHART_SUGGESTIONS_ENABLED=true` when the frontend can consume the contract below. It defaults to `false` to
  avoid extra model calls and invisible suggestions in existing clients. The local server runner enables it.
- Optional `CHART_EXTRACTION_TIMEOUT_SECONDS=25` bounds each extraction provider call. Existing provider fallback
  rules and eligible request keys are reused; fallback can make the total suggestion time longer.
- `suggest_graphs` is an optional boolean on `/agent/process` and `/agent/stream`, default `true`. `false` opts out.
  Suggestions run only for `type: "enhance_with_ai"`, with the feature enabled and no reference images.
- Send plain text or Markdown, including Markdown tables. Source evidence refers to that exact original string.
- At most one series with 2–24 distinct categories is returned. No extraction call is made without at least two
  numeric candidates. No suitable comparable series means no suggestion. No external research or image generation.
- The extraction is an additional text-model call, with usage recorded separately as `graph_model_used`,
  `graph_llm_calls` (stage `graph_extract`) and `graph_fallbacks`. Existing text credit settlement is unchanged;
  there are no image/design credit charges.
- Graph edits and previews are deterministic Python operations with no AI provider calls or generated-image costs.

## Enhance request

```json
{
  "type": "enhance_with_ai",
  "query": "Revenue was USD 12,000 in January and USD 18,000 in February.",
  "model": "openai",
  "api_key": "<existing endpoint credential>",
  "suggest_graphs": true
}
```

Keep the existing authentication distinction: `/agent/stream` takes the licence key; the server-to-server
`/agent/process` takes the provider key already resolved by the application's backend. Do not expose provider keys
in the editor. There is no new authentication mechanism in this change.

## Stream contract

Ordinary typed `start` and `content` SSE messages are unchanged. After the prose, before the ordinary `done`, the
server may emit one **named** event. Parse `event:` as well as the JSON `type` field used by existing messages.

```text
event: graph_suggestion
data: {"status":"suggested","reason":"Comparable values were found in your original content. Review before inserting.","chart":{...},"preview_svg":"<svg ...>...</svg>"}
```

On extraction failure, invalid evidence or timeout, the prose still succeeds:

```text
event: graph_suggestion_unavailable
data: {"code":"graph_suggestion_unavailable","message":"Your text was enhanced, but a verified graph suggestion could not be produced."}
```

No suitable data emits neither event. Always clear any optional graph loading indicator on `done` or terminal
`error`. The suggestion is additive; do not append its JSON or SVG to the text-content stream. Suggestions also work
after the protected long-document path; extraction sees the original tables, not their locked placeholders. Abort
prevents new extraction where observed; a provider call already in flight can finish, but its result is discarded.

## Non-streaming response

Existing fields remain. Two additive fields are always present:

```json
{
  "success": true,
  "markdown_response": "The enhanced document, without an inserted graph.",
  "html_response": "The enhanced document, without an inserted graph.",
  "warnings": [],
  "design": null,
  "graph_suggestions": [{"status":"suggested","reason":"...","chart":{},"preview_svg":"<svg ...>...</svg>"}],
  "graph_suggestion_error": null
}
```

An empty array is normal. Failure uses an empty array plus the same public error object as the SSE failure event.

## Canonical editable block

Persist **the complete `chart` JSON**, not only the SVG, a PNG or a generated design-canvas AST. Example:

```json
{
  "block_type": "eddyter_chart",
  "version": "1.0",
  "chart_id": "chart_0123456789abcdef",
  "title": "Monthly revenue",
  "source_sha256": "<64-character SHA-256 of the exact original input>",
  "data": {
    "unit": "USD",
    "points": [
      {"id":"p1","label":"January","value":"12000","source":{"excerpt":"USD 12,000 in January","label":"January","value_text":"12,000"}},
      {"id":"p2","label":"February","value":"18000","source":{"excerpt":"USD 18,000 in February","label":"February","value_text":"18,000"}}
    ]
  },
  "presentation": {"chart_type":"bar","color":"#2563EB","point_colors":{}}
}
```

- Supported types: `bar`, `line`, `pie`. Default is `bar`; the frontend/user chooses another type.
- Values are returned as **decimal strings**. Keep them as strings in application state and saved documents.
  Do not round, convert to percentages, re-sort or write JavaScript floating-point results back into the dataset
  when changing appearance. If a frontend plotting library needs numbers, convert a throwaway display copy only.
- Supported values: finite numbers within ±1,000,000,000,000,000 and at most six decimal places. Source parsing
  accepts decimal points, standard three-digit comma groups, signs and explicit thousand/million/billion words.
  A value such as `1.5 million` with unit `USD` becomes the exact quantity `1500000`; the raw source text is retained.
  Unsupported or ambiguous formats (e.g. decimal commas, Indian comma grouping, k/bn, accounting negatives,
  ranges, uncertain values) do not produce a suggestion. Explicit magnitude suffixes combined with a magnitude
  in the unit are rejected to prevent double scaling. Bare values under an exact shared magnitude unit retain
  that unit, e.g. `12` with `USD millions`.
- IDs and point order remain stable across edits. Source excerpts and `source_sha256` remain the **initial
  provenance**, even after a deliberate data edit; they do not claim manually edited values still match the source.
- `source_sha256` identifies the input; it is not a signature or authorization mechanism. These endpoints are
  stateless and operate on the snapshot supplied by the caller. Normal document access and persistence permissions
  remain the application's responsibility.
- A graph can be numerically grounded yet be a poor editorial choice. Show the source evidence in the preview
  and require Insert; do not automatically insert or infer that every series is appropriate for a pie chart.

## Scoped edits

`POST /agent/charts/edit` accepts the current complete chart plus **exactly one** of `patch` or `instruction`.
All requests/responses are JSON. The response is `{ "chart": { ...updated document... }, "preview_svg": "<svg...>" }`.

Type-only edit:

```json
{"chart": "<complete chart object, not a string>", "patch": {"chart_type":"line"}}
```

Color-only edit:

```json
{"chart": "<complete chart object, not a string>", "patch": {"color":"#FF0000"}}
```

The placeholders above stand for the actual chart object returned by Enhance, never a JSON-encoded string.

| Patch field | Effect | Everything else |
|---|---|---|
| `chart_type` | Sets bar/line/pie; recalculates preview from the same data | Preserved |
| `color` | Changes the base mark/line color | Preserved, including existing per-point overrides |
| `point_colors: {"p2":"#FF0000"}` | Merges only these point-color overrides | Other points and overrides preserved |
| `values: {"p2":"19000.25"}` | Explicitly changes only these data values | Other values, labels, order, unit, colors, provenance preserved |
| `labels: {"p2":"February revised"}` | Explicitly changes only these category labels | Preserved |
| `title` | Changes only the title | Preserved |
| `unit` | Changes only the unit label; performs **no conversion** | Preserved |

Combine fields only when the user explicitly requested those changes. Omitted fields are not edits. Nulls, empty
patches, unknown fields, unknown point IDs, duplicate labels and invalid colors/values are rejected with HTTP 400.
Whole requests apply atomically. A pie graph requires non-negative values and a positive total; an incompatible
conversion returns 400 instead of taking absolute values, dropping points or changing the numbers.

The optional natural-language `instruction` is deliberately limited and makes **no model calls**:

```json
{"chart": "<complete chart object>", "instruction":"change graph type to line and change color to red"}
```

Supported examples: `change graph type to line`, `make it a pie chart`, `change color to #00FF00`,
`change the graph color to red`. Clauses can be joined by `and`, `then`, comma or semicolon. Named colors:
red, blue, green, orange, purple, black, gray/grey. Unknown, contradictory or partially supported requests are
rejected in full. Use explicit patches for value edits, specific-point colors, titles and arbitrary UI controls.
Do not route edits of an existing graph through Enhance, general chat, `generate_design` or `edit_design`:
those flows can regenerate content. This endpoint is the consistency boundary.

`POST /agent/charts/preview` takes `{ "chart": <complete chart object> }` and returns the same view response
without changing any field. Both chart endpoints are pure local transforms: they need no provider key or licence,
store nothing, and cannot load another user's graph. Apply existing application routing/access controls when
proxying them; neither endpoint replaces authorization for the document being edited or saved.

## Frontend implementation checklist

1. Register a versioned editor graph node carrying the complete `eddyter_chart` document, included in existing
   document save/load, copy/paste, undo/redo and export. Keep graph nodes out of text sent for prose rewriting.
2. After Enhance, show an optional preview with title, chart, an accessible data table and expandable source excerpts.
   Provide **Insert graph** and **Dismiss**. Empty suggestions leave the editor unchanged. A graph failure must not
   discard the enhanced writing. Hide optional loading when the stream ends.
3. Insert only on the user's action, at a chosen document position. Dismiss is client-only. There is no accept/save
   endpoint because the application already owns document persistence. Track accepted/dismissed suggestions per
   document/source to prevent repeated insertion. Discard pending suggestions if the source selection changes.
4. Add a graph-type selector, base color picker, optional per-point colors and a value/label editor. Send only the
   corresponding patch fields. Use stable point IDs, not array positions, display text, colors or pixel coordinates.
5. Serialize edits per graph or reject stale responses against the submitted snapshot. This stateless API cannot
   detect a newer client document; an old response must never overwrite a more recent edit. Commit successful edits
   as normal editor undo steps. On HTTP 400, retain the original node and display the error.
6. Render `preview_svg` as an SVG image/Blob or implement rendering from `chart.data` + `chart.presentation`. Never
   execute model output as HTML. The backend SVG is escaped and computed by Python; it contains no scripts,
   external references or model-authored geometry. Backend previews include an accessible title and full data
   description; long visual category labels may be shortened, while canonical labels stay intact. Keep a full
   accessible data table in the UI and exports. An SVG preview itself is not the editable data model.
7. Verify: bar → line → pie → bar retains the entire data object byte-for-byte; color-only retains all data and other
   presentation fields; one value edit changes only that point's value; invalid pie conversion leaves the node
   unchanged; reload and undo preserve exact decimals, source evidence, stable IDs and per-point colors.

## Verification and local demonstration

Run `.venv/bin/python -m pytest tests/charts` from the `agent` folder. These tests replace only provider calls;
they exercise grounding, real HTTP and SSE integration, long-document tables, atomic edits, negative values,
exact decimal round trips, safe SVG and failure/abort paths.

Run `.venv/bin/python scripts/demo_editable_graph.py` for the provider-free Python demo. It runs the real extraction
validator on explicit sample evidence, the real Enhance HTTP flow with a scripted model, and real edit/preview
endpoints. It writes JSON receipts and a standalone HTML page showing the suggestion and edited variants under
`local_run/editable_graph_demo/`. It is a deterministic demonstration, not a live-model accuracy benchmark.

For a real local-provider check, run `.venv/bin/python scripts/demo_editable_graph.py --live`. This makes real
Enhance/extraction calls using the configured OpenAI key, while blocking Redis, webhooks and storage side effects.
Responses vary with the model; extraction may legitimately decline. No frontend integration or deployment is
performed by either demo.
