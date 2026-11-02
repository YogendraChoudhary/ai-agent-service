# Highlight key points means a concise summary

The **Highlight key points** text action produces a new, concise takeaway summary: one lead sentence followed
by supporting bullet points. It can paraphrase and shorten the source while preserving its meaning, figures,
units, categories, comparisons, thresholds, conditions and qualifications. It does not bold/color the source
or return formatting ranges.

## Request

Use the existing `/agent/process` or `/agent/stream` fields and authentication:

```json
{
  "type": "highlight_key_points",
  "query": "<paragraph or short text portion>",
  "model": "openai",
  "api_key": "<existing endpoint credential>"
}
```

The text limit is 10,000 characters; blank input is rejected. Omit `secondary_prompt`, `document_text`, design
and image inputs. These incompatible fields return HTTP 400. The dedicated action uses the fast/basic tier,
no tools and the existing provider/BYOK/fallback rules. There is no new feature flag.

`/agent/stream` takes the existing licence key. Server-to-server `/agent/process` takes the provider key already
resolved by the application backend. Never expose provider keys in the editor.

## Output format

For the supplied rich-text-editor survey passage, the intended shape is:

```markdown
73% of respondents rate rich text editors as extremely or very important.

- Importance is 71% among organisations with 100 or fewer employees.
- The figure falls to 68% among organisations with up to 1,000 employees.
- Importance rises to 80% among enterprises with more than 1,000 employees.
- As organisations scale, reliable content tools become increasingly essential across teams and products.
- Choosing an editor becomes a strategic rather than purely technical decision for larger organisations.
```

The model may phrase a faithful summary differently. Python supplies the layout: one lead paragraph and up to
six distinct bullet-point sentences. An input containing only one fact can return a lead alone. The lead is not
repeated as a bullet, and important conclusions from later paragraphs are included. There are no headings,
labels, introductory commentary or colored/inline highlights. If no meaningful takeaways exist, the text result
is `No key points found.` and the structured lead/points are empty.

## Process response

Both `markdown_response` and the existing `html_response` field contain the **new summary Markdown**, not the
original paragraph and not rendered HTML. The additive `highlighting` field is now structured summary metadata:

```json
{
  "success": true,
  "markdown_response": "<lead sentence>\n\n- <supporting point>\n- <supporting point>",
  "html_response": "<same summary Markdown>",
  "highlighting": {
    "source_sha256": "<SHA-256 of original UTF-8 input>",
    "lead": "<lead sentence>",
    "key_points": ["<first supporting point>", "<second supporting point>"],
    "summary_markdown": "<same summary Markdown>",
    "evidence": ["<source excerpt for lead>", "<source excerpt for first point>", "<source excerpt for second point>"]
  }
}
```

`key_points` contains bullets only. `evidence` is ordered lead-first, followed by bullet evidence. Every quote is
an actual source slice; whitespace-only normalization (such as NBSP from rich-text imports) is matched back to
the original slice. **There are no `spans`, `start_utf16`, `end_utf16` or `highlighted_html` fields.** Other actions
return `highlighting: null`.

Each generated point must have source evidence. Python rejects unsupported numeric values/percentages/currencies,
changed or omitted recognized numerical bounds, duplicate sentences and broad percentage evidence mixing
several groups. Examples: 71% cannot cite the enterprise excerpt with 80%, and `more than 1,000` cannot become
`up to 1,000` or simply `1,000`. Numbers may retain their value with equivalent thousand-separator formatting.
The model is instructed to retain category associations and qualitative implications too; exact factual and
editorial meaning still requires review. These checks are not a general semantic proof of every paraphrase.

One repair is attempted on invalid model output. Failure after repair returns HTTP 422 for process, with no
partial summary. Provider failures retain normal API error behavior. No image/design charges are introduced.

## Streaming

```text
data: {"type":"start","request_id":"..."}

event: highlight_ready
data: {"source_sha256":"...","lead":"...","key_points":[...],"summary_markdown":"...","evidence":[...]}

data: {"type":"content","content":"<complete validated summary Markdown>"}

data: {"type":"done",...}
```

Parse named events as well as `payload.type`. `content` is the actual summary to present or insert, like other
text-generation actions. Metadata and content describe the same output; do not insert both. The server buffers
and validates the complete summary before emitting it. On repeated validation failure, stream emits `error`
without metadata/content/done. Abort prevents further work after observed cancellation; an in-flight provider
call can finish but its result is discarded.

## Frontend integration

1. Offer **Highlight key points** beside Correct grammar. Send the selected portion or current paragraph using
   the dedicated type. Capture the original editor range/version so a moved cursor cannot redirect the result.
2. Render `markdown_response` or streamed `content` as the lead paragraph plus a bulleted list. Offer a preview
   or replace the original range through the existing text-operation flow, preserving undo and normal save behavior.
3. Do not apply `<mark>`, bold formatting or character ranges. The operation rewrites the portion into a summary.
4. Before replacing, confirm that the captured range still contains the original text/fingerprint. Discard stale
   results rather than overwriting newer typing. A no-points result should display a message and leave input intact.
5. Optionally use `highlighting.lead` and `key_points` for a separate summary panel; evidence can support review.
   Treat metadata strings as text, not executable HTML. Python escapes model text in the generated Markdown.
6. Test lead/list rendering, preserved percentages/group sizes/qualifiers, multi-paragraph coverage, no-points,
   stale selection, errors, undo, saving and reloading.

## Verification

`tests/fixtures/key_points_summary.json` holds the supplied survey passage, including NBSP, and its expected
lead/five-bullet result. Run `.venv/bin/python -m pytest tests/test_key_points.py` for this regression, numerical
fidelity and threshold checks, paraphrasing, layout, repair/cancellation and real HTTP/SSE integration.

Run `.venv/bin/python scripts/demo_key_points.py` for the provider-free API example or add `--live` to summarize
that same passage using the configured OpenAI key. Redis, licences, storage and webhooks are disabled in this demo.
Receipts and the input/summary preview are under `local_run/key_points_summary/{scripted,live}/`.

The Python implementation and handoff are complete; frontend editor integration and deployment are outside this
workspace. Ponytail review checks the callers, validation, output format and removal of the old range-formatting contract.
