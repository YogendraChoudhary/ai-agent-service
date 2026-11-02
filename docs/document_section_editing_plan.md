# Work item: edit document sections without a selection

Status: complete for the Python backend and frontend handoff.

Implement the Python side of prompt-directed editing of an entire supplied Markdown/plain-text document.
The user supplies an instruction; Python identifies blocks, asks the model to select relevant blocks, edits only
those blocks, validates the replacements and returns reviewable patches. All other source text must stay exact.

Scope: existing process/stream authentication and provider routing; explicit document context; source-bound block
IDs and document fingerprints; ambiguous-target responses; protected headings/tables/code/quotes/images/links;
atomic validation, stale-document rejection, regression tests, a reproducible demo and frontend handoff.

No frontend, file-upload parsing, document storage or deployment is included. The frontend must extract uploaded
files to text, send the current snapshot and apply accepted patches through the existing document save flow.

Validation: deterministic targeting/adversarial tests, HTTP/SSE integration and legacy regression suites; a real
provider demonstration with complete redacted receipts; Ponytail review of the implementation and connected callers.

Delivered: `src/document_edit`, document context on existing process/stream requests, `/agent/documents/apply`,
`docs/DOCUMENT_EDIT_API_HANDOFF.md`, `docs/DOCUMENT_EDIT_REVIEW.md` and `scripts/demo_document_edit.py`.

Verified on 2 November 2026: 69 new feature tests; the full suite passed with 1,946 tests and 1,656 subtests.
Two real-model editing prompts selected the expected paragraphs and preserved all untargeted regions exactly.
The ambiguity case returned clarification without edits; accepted patches replayed successfully and a stale
snapshot returned HTTP 409. The evidence index is `local_run/document_edit_evidence.json`; before/after pages,
redacted receipts and per-region hashes are under `local_run/document_edit_demo/live/`.
