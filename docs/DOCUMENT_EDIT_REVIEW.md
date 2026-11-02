What this change does: The Python backend finds the paragraphs requested by a prompt, edits only those paragraphs
and returns source-bound patches for review. Every untargeted source region stays exact, and stale proposals are
rejected before application.

Looks good. Ship.

Ponytail review covered the new document parser, scope checks, planner/rewriter validation, patch replay, all
callers through `process_user_query_stream` and `process_user_query_with_images`, provider eligibility, API request
validation, response cleanup, streaming, credit reservation/settlement and the frontend handoff. Expected use is
concurrent requests in the existing FastAPI service. This is not a production traffic load benchmark.

The review verified these concrete boundaries:

- Model selection cannot target an unknown/protected block, bypass a recognized named section or its exclusion,
  or supply a request quote absent from the user's instruction. Duplicate headings require disambiguation.
- “Improve Pricing and FAQ” permits both named targets; “Improve FAQ under Pricing” treats Pricing as context
  and permits only its FAQ. Bold, linked and inline-code headings retain their visible names for exclusions.
  These boundary cases have direct regression tests in `tests/document_edit/test_service.py`.
- Failed or malformed planning, provider failure after an earlier successful replacement, invalid numeric facts,
  changed currencies/URLs and broken structure leave the complete original unchanged. There are no partial writes.
- Snapshot hashes, IDs, exact before-text and UTF-16 offsets are checked before application. The server copies
  untouched slices rather than asking the model to reproduce them. Emoji, BOM, LF, CRLF and CR are tested.
- The explicit document-context path bypasses image/code/design routing and flat design billing. Credit checks
  count the full document, reject insufficient balances before model work and enforce the per-request cap before
  reserving the estimated original-plus-output amount. BYOK never gains platform provider candidates.
- Models have no web/image tools, and document text remains data in prompts. No database, file upload, document
  persistence, mutable global document state or new dependency was introduced. The apply endpoint operates only
  on supplied snapshots; the application still owns document access and saving.
- Tests exercise 32 concurrent independent proposal computations, keeping their snapshots isolated. Request
  size, parsed blocks, target count, per-block size, response shape, retries and provider-call timeouts are bounded.
- Live evidence checks the intended target IDs as well as independently comparing all untargeted output slices.
  The saved receipts include named sections, semantic paragraph selection without headings, ambiguity, accepted
  patch replay and HTTP 409 for a stale document.

No unresolved must-fix or should-fix findings remain in this backend change. Semantic target selection and
editorial correctness remain model judgments; the frontend must present proposals for review. Deterministic
source preservation does not establish perfect intent recognition on arbitrary documents.

Reproduce with `.venv/bin/python -m pytest tests/document_edit` and `scripts/demo_document_edit.py`; use `--live`
only for a real-provider check. The full-suite log is `local_run/document_edit_regression.log`. Demo receipts,
before/after documents and region hashes are under `local_run/document_edit_demo/{scripted,live}/`.

Verification result: 69 new feature tests; full suite 1,946 tests and 1,656 subtests passed. The machine-readable
evidence index is `local_run/document_edit_evidence.json`, including expected/actual target IDs and code hashes.

Verdict: Ship.

Not checked: frontend upload/import, editor node conversion, acceptance/save/undo/reload, production deployment or
production traffic. Two live successful editing prompts plus an ambiguity case are evidence for those examples,
not an exhaustive model-accuracy benchmark.
