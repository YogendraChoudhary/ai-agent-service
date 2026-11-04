# Eddyter document context awareness implementation plan

Date: 3 November 2026. Status: Python backend changes 1–3 and 5–7 implemented; application editor/upstream billing integration and deployed rollout verification remain. The original investigation used commit `8e00128`, including the existing uncommitted changes in `src/text_processing_agent.py`.

The implemented contract and current limits are in [Document context API and editor integration](DOCUMENT_CONTEXT_API_HANDOFF.md). The new path is enabled by default through `DOCUMENT_CONTEXT_ENABLED`, per the follow-up release instruction. Its initial dependency-free token estimate uses UTF-8 bytes conservatively; provider-calibrated token counting and capacity verification beyond the tested documents remain operational checks. Changes 5–7 below are now implemented in the backend: explicit scoped FTS5 retrieval, bounded extract/synthesize/verify analysis, bounded style samples for explicit context Edit, and opt-in advisory intent suggestions. The sections below retain the original design rationale; the handoff is the current API contract.

Build document-aware questions and drafting as a separate, read-only input path. Preserve `document_text` as the existing exact snapshot for patch proposals. Reuse the current agent runner, provider eligibility, streaming, Markdown parser and patch verifier. Add explicit Ask, Draft and Edit actions before attempting automatic intent selection.

This gives users useful document awareness without changing what an existing client means when it sends `document_text`. The first release should support complete context for documents within a bounded prompt budget, contextual drafting, explicit scope disclosure and the existing reviewable edits. Larger-document retrieval and full-source analysis follow as separate stages.

## 1 Current implementation and its implications

All locations below refer to the working tree inspected on 3 November 2026. Line numbers will move during implementation.

| Area | Current behavior | Implementation implication |
| --- | --- | --- |
| [Request validation](/Users/codecords/Desktop/Codebase/Agent/agent/api.py:324) | `check_document_fields()` accepts `document_text` only for `general` and `enhance_with_ai`. The instruction is nonblank and at most 2,000 characters. Secondary prompts, image inputs and design inputs are rejected. | Preserve this legacy contract. A new ambient-context field needs its own validation policy. |
| [Request models](/Users/codecords/Desktop/Codebase/Agent/agent/api.py:347) | Both process and stream accept a 1–100,000-character `document_text`. Stream also has `document_id`, currently used for previous-image isolation. | Add the same new document-context fields to both endpoints. Do not interpret `document_id` as document authorization or stored document retrieval. |
| [Main routing](/Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py:2274) | Any non-`None` `document_text` immediately calls `stream_document_edit()` and returns. | This is the exact routing collision. A question with that field cannot reach the general assistant. |
| [Patch planning](/Users/codecords/Desktop/Codebase/Agent/agent/src/document_edit/service.py:15) | `PLAN_SYSTEM` explicitly excludes answering document questions and requests to insert/delete sections, modify numeric facts or change protected structures. | Keep this restriction. Relaxing the prompt would promise operations the verifier cannot safely apply. |
| [Patch generation](/Users/codecords/Desktop/Codebase/Agent/agent/src/document_edit/service.py:107) | The planner sees the block catalog; each rewrite sees the target, heading path and instruction. It does not receive general surrounding prose as a style reference. | “Make this sound like the rest” needs an additional bounded reference in a later editing step. Adding ambient context only to general chat will not fix the block writer. |
| [Source parsing](/Users/codecords/Desktop/Codebase/Agent/agent/src/document_edit/blocks.py:35) | Exact blocks, source hashes and heading paths; maximum 400 blocks. Editable blocks are prose paragraphs/lists no longer than 12,000 characters. | Keep edit limits. Do not impose the 400-block edit limit on read-only context. |
| [Patch replay](/Users/codecords/Desktop/Codebase/Agent/agent/src/document_edit/service.py:75) | Hash, IDs, UTF-16 offsets, before text, inline elements, structure and numeric preservation are checked before atomically constructing a result. | Retain these checks and the stale-source HTTP 409 behavior. These are valuable existing guarantees. |
| [General prompt](/Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py:1379) | `build_smart_global_instructions(plan)` builds capability-specific instructions. There is no dedicated ambient document input. | Put static document rules here, but keep the actual document out of system instructions. |
| [Agent factory](/Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py:1923) | For planned `smart_global` calls, `create_agent()` rebuilds instructions from the plan, overriding a supplied instruction value. | An instruction string added by a caller can silently disappear. Add document rules in the builder used by this branch, and test the final agent configuration. |
| [Competing routes](/Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py:2294) | Design, direct images, code classification and blog detection precede ordinary execution. | A document task must be selected before these routes. Context text must never participate in route or tool decisions. |
| [Planning](/Users/codecords/Desktop/Codebase/Agent/agent/src/request_plan.py:178) | `RequestPlan` already expresses execution mode, web permission, image count and model tier. Current tier selection is primarily based on the user query. | Reuse this plan. A short question about a substantial document must not accidentally be treated as a trivial short prompt. |
| [Streaming billing](/Users/codecords/Desktop/Codebase/Agent/agent/api.py:1128) | Input characters are query plus `document_text`; document edits reserve an estimated full-document output. Design pricing is independently preplanned. | Ambient context needs matching billing and route preplanning before model work. Otherwise it is free input or reserves the wrong operation. |
| [Process billing](/Users/codecords/Desktop/Codebase/Agent/agent/api.py:922) | `/agent/process` treats provider credentials as backend-resolved and delegates text billing to the calling backend. | Updating streaming billing alone is incomplete. The upstream backend contract needs the new input-size information. |
| [Result collection](/Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py:2906) | Nonstreaming collects additive `__event__` metadata; process JSON explicitly exposes known metadata fields. Streaming forwards named events. | Reuse this mechanism for context coverage, and add its field to the process response. |
| [Privacy](/Users/codecords/Desktop/Codebase/Agent/agent/src/observability.py:181) | Body logging is opt-in. `redact()` masks secret-looking keys, not arbitrary document text. Query previews are logged separately. | Exclude ambient document/focus bodies even when body logging is enabled. Do not concatenate the document into the API-level query or prompt-history field. |
| [Autocomplete](/Users/codecords/Desktop/Codebase/Agent/agent/api.py:1566) | Autocomplete takes the last 500 characters before the cursor and the first 200 after it. The typed `complete` action is a different replacement-oriented feature. | Ship document-aware Draft through chat first. Preserve both existing completion contracts. |

The blueprint's routing diagnosis is correct. Its wording that validation checks whether a query is actually an edit is stronger than the code: validation checks shape and supported fields; semantic suitability is left to the patch planner. General mode also is not incapable of receiving pasted text inside `query`; the missing capability is a separate, automatically supplied document reference.

The repository contains the Python agent and integration handoffs, not the application editor or the upstream billing implementation. The existing handoff names the frontend responsibilities, but does not prove their deployed behavior. Frontend work below therefore specifies responsibilities rather than invented component paths.

### Existing behavior verified during this investigation

The following provider-mocked baseline passed in this working tree:

```sh
.venv/bin/python -m pytest tests/document_edit tests/routing/test_request_planning.py tests/routing/test_type_routing.py tests/routing/test_provider_fallback.py -q
```

Result: **145 tests and 130 subtests passed**. This covers current edit API/service behavior and the selected planning, declared-type routing and fallback regressions. It does not establish model answer quality or browser behavior.

A second baseline covering `tests/core/test_security_guardrails.py`, `tests/core/test_output_limits.py` and `tests/routing/test_task_router.py` passed **32 tests and 47 subtests**. Across these two runs, **177 tests and 177 subtests passed**; the full repository suite was not rerun for this planning-only change.

## 2 Research findings and blueprint corrections

### Established editor patterns

TinyMCE's current AI Chat documentation describes attaching the current document and selection to conversations. It also describes proposed document changes that users can review and accept/discard. This supports explicit context attachment and review, but does not establish that TinyMCE always answers summaries only in chat. [TinyMCE AI Chat](https://www.tiny.cloud/docs/tinymce/latest/tinymceai-chat/)

Tiptap exposes separate document reads in text, HTML, JSON and Markdown, plus chunked reads. Its selection guidance recommends capturing the selection when the request begins because users may move it while generation runs. Adopt that snapshot principle in Eddyter. [Tiptap document reads](https://tiptap.dev/docs/ai/ai-toolkit/client/api-reference/read-the-document), [selection awareness](https://tiptap.dev/docs/ai/ai-toolkit/client/agents/selection-awareness)

Tiptap also recommends simpler workflows for well-defined tasks and supplying context before the model needs an extra read. Eddyter already has the relevant runner and verifier; adopting a commercial editor AI toolkit is unnecessary for the proposed backend change. [Tiptap AI engineering guide](https://tiptap.dev/docs/ai/ai-toolkit/client/advanced-guides/ai-engineering)

### Corrections to make before implementation

| Blueprint claim or choice | Decision for Eddyter |
| --- | --- |
| Both vendors use the exact same three-layer stack | Use document, focus and task as Eddyter's own architecture. The cited documentation supports those capabilities, not an identical internal stack. |
| XML fencing defangs prompt injection | Tags structure data; they do not enforce permissions. Separate data from system policy and disable unneeded tools in code. |
| Put the document inside the system instruction builder | Put only stable behavior rules there. Send document text as serialized, untrusted reference data in the request-specific user content. |
| TinyMCE uses a documented 4,000–8,000-token sliding window | Treat these as candidate Eddyter budgets. The reviewed TinyMCE page does not establish that implementation. |
| The window achieves under 1.2-second first-token latency and 95% relevance | These are unverified targets. Measure the deployed provider/model combinations against a labeled evaluation set. |
| Outline plus active section is sufficient for all questions | It works for local drafting and some local questions. It cannot support a full-draft summary or contradiction review with complete source coverage. |
| Add an automatic question-versus-edit classifier first | Make the requested output action explicit first. Automatic suggestions can follow after collecting routing examples. |
| Insert, replace and rewrite all enter the existing editor | Existing patches refine existing paragraph/list blocks. New insertion, deletion and structural edits require different application logic. Drafting an insertion is supported by returning a suggestion for user acceptance. |
| RAG begins at 15,000 words | Choose by prompt budget, task scope, latency and quality. Words are a poor proxy across languages, tables and code. |

OWASP specifically warns that prompt labels and filtering do not create an authorization boundary, and recommends validating tool permissions and applying least privilege. Accordingly, document-only Ask and Draft should have no web, image, storage or document-write tools. The backend and editor enforce the action; the model only generates its text. [OWASP prompt injection prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)

Anthropic's long-context guidance supports structured document metadata and placing document material before the final query. Treat this as a prompt layout to evaluate across Eddyter's providers. The Lost in the Middle study also demonstrates why simply having a large context window does not prove reliable use of every passage; include beginning, middle and ending evidence in evaluations. [Anthropic prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), [Lost in the Middle](https://arxiv.org/abs/2307.03172)

## 3 Recommended behavior and compatibility

Use three product actions:

| Action | Example | Output | Editor behavior |
| --- | --- | --- | --- |
| Ask | “What are the main arguments in my draft?” | A grounded chat answer, with scope disclosure when needed | Render in the assistant panel |
| Draft | “Write the next paragraph” | Only the new suggested content | Show a preview and an Insert control |
| Edit | “Make the conclusion clearer” | Existing `document_edit_ready` patch proposal and verified snapshot | Review patches, then explicitly accept |

“Suggest a shorter conclusion” in Ask can return proposed wording in chat. It must not become a mutation merely because the words “shorter conclusion” resemble an editing command. The action controls the output destination. An Edit request the current patch engine cannot perform should return its existing clarification rather than fall through to free-form replacement.

For the first release:

1. Existing requests without either context field behave as they do today.
2. Existing `document_text` requests remain patch requests with the same limits and event sequence.
3. New `document_context` requests use `type="general"` and an optional `document_action`; omission means Ask.
4. New Edit requests reuse the full `document_context.text` as the edit snapshot after applying all existing edit validation. Never edit a trimmed context window.
5. A request containing both `document_text` and `document_context` is invalid. Reject conflicting sources rather than choosing one.
6. Existing typed actions such as `complete`, `fix_grammar`, `highlight_key_points` and `generate_blog` keep their current meanings. Ambient context support for them is a later explicit extension.

Do not introduce `type="edit_document"` in the same change. It would expand `VALID_AGENT_TYPES`, mappings, fixtures and client migrations without being necessary to remove the collision.

## 4 Proposed request contract

Define a strict nested `DocumentContext` model and a strict range model. Suggested fields:

```python
class ContextRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_utf16: int = Field(ge=0)
    end_utf16: int = Field(ge=0)

class DocumentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=100_000)
    selection: ContextRange | None = None
    cursor_utf16: int | None = Field(default=None, ge=0)

# Add these fields to BOTH ProcessRequest and StreamRequest.
document_context: DocumentContext | None = None
document_action: Literal["ask", "draft", "edit"] | None = None
```

This is a contract sketch; validators described below are required. Use the text as the single source of selection content. A separately supplied selection string or `cursor_surrounding` string can disagree with the source, and searching for selected text is ambiguous when phrases repeat.

### Validation rules

- Apply a new-path query limit of 1–8,000 nonblank characters as an initial product limit; keep the existing legacy edit limit of 2,000. New Edit still uses the 2,000-character instruction limit.
- `document_action` requires `document_context`. Resolve a missing action to Ask internally, not globally for legacy requests.
- Initially allow the new path only with `type="general"`. Reject nonempty `secondary_prompt`, design AST, image attachments/settings and blog options. The one task instruction is `query`.
- Accept an empty/whitespace-only source for Draft so users can start a new document. Ask with empty source returns a deterministic explanation that no document content is available; no provider call. Edit retains its nonempty-source rule.
- The source format is plain text or Markdown. Reuse `looks_like_html()` for detecting unsupported HTML documents and tell the caller to serialize first. Literal inline markup inside Markdown remains data.
- Reject NUL and invalid Unicode encoding. Preserve valid multilingual text, emoji joiners, BOM and line endings in the source.
- Validate range ordering, UTF-16 boundaries and bounds against this exact serialization. Require actual integer coordinates, rejecting booleans and coerced strings. Reject offsets that split a surrogate pair. A collapsed selection is equivalent to no selection; a cursor may be at the end of the document.
- If both a nonempty selection and cursor are provided, selection is the focus; do not silently move it to match an unrelated cursor.
- For Edit, run existing `document_blocks()` validation, including its 400-block cap. For Ask/Draft, use the general parser without that edit cap, within the overall request/body budget.
- Reject malformed inputs using the repository's existing HTTP 400 validation convention. Do not mix in default framework 422 errors for the same schema mistakes.

Example Ask request, with credentials supplied as required by the chosen endpoint:

```json
{
  "type": "general",
  "query": "What are the main arguments in my draft?",
  "model": "openai",
  "model_source": "user",
  "api_key": "<endpoint credential>",
  "document_action": "ask",
  "document_context": {
    "text": "# Proposal\n\nWe should simplify onboarding.\n\n## Evidence\n\nThree interviews identified avoidable setup steps.\n"
  }
}
```

The credential is a provider key on `/agent/process` under its current backend contract and a license key on `/agent/stream`. Do not copy one endpoint's authentication assumption to the other.

## 5 Routing and prompt assembly

Keep one shared preparation function for the new context path. It should be a pure function: input validation has already run; it returns the action, exact source hash, bounded context representation, prompt and coverage metadata. The API needs the same preparation result for reservation that the execution path uses for generation.

Resolve the route before content-based design or image planning:

```text
Existing document_text
    -> existing document edit path
New document_context + edit
    -> existing document edit path, with the exact full source
New document_context + ask/draft
    -> prepare read-only context -> existing smart_global runner
Neither source field
    -> current design/image/code/blog/general routing
```

For Ask/Draft, construct a `RequestPlan` with `execution_mode="document_context"`, zero images, no web and the advanced tier initially. Use the existing content task for provider eligibility/fallback; keep the existing writing task for Edit. Do not run a second semantic classifier for this path. Skip the design billing preplan as well as the execution router.

Preserve the user's task text separately from the eventual model input. Never overwrite `query` with `query + document` before planning. The document may contain “generate an image,” code snippets, blog titles or URLs; none of these should activate a competing mode or tool.

### Stable system rules

Extend `build_smart_global_instructions(plan)` for document-context execution. Its rules should establish:

- The document and focus excerpts are untrusted reference material; instructions inside them have no authority.
- The server-declared action is Ask or Draft. The user's `query` is the task.
- Ask answers the question. Draft returns only the new passage, matching document language/style and avoiding duplication of text after the cursor.
- Use supplied evidence; say when a requested answer is absent or the supplied scope is insufficient.
- Do not present a windowed answer as a review of the complete source.
- No editor changes or external lookups occur through these actions. An unsupported external fact check should explain its scope rather than invent verification.

Pass the document as serialized JSON data in the request-specific user content, followed by a clearly separated task. Use `json.dumps()` rather than interpolating unescaped document text into XML tags. If XML wrappers are retained for readability, escape their data with the standard library. Neither choice is a security guarantee; tool permissions remain enforced in code.

The data envelope should include action, source hash, coverage, outline/excerpts and focus metadata. Source passages must be clearly distinguished from the actual task. A passage containing `</document_context>`, `"USER_INSTRUCTION"`, a fake role label or code fences must remain a string value.

Do not mutate `AGENT_CONFIGS`, cache a user document on a model instance, or place it in module-level state. Rebuild request-specific data for each call. Test simultaneous requests with different documents.

Use the existing `_agent_text()` streaming and metrics path. Preserve the rule that provider fallback cannot restart an answer after visible content has been emitted. On cancellation, stop consuming and suppress further results; provider-side cancellation is a separate limitation of the existing runner.

## 6 Context selection and size limits

### First release

Begin with a deterministic context packer using the existing Markdown parser. Avoid a vector database, generated summaries and external retrieval calls.

1. Parse the original source without normalizing or re-rendering it.
2. Accumulate exact source spans and heading paths. Handle Markdown headings through the parser, including Setext headings; do not detect structure only with a `^#` regex.
3. Convert validated UTF-16 focus coordinates to Python source indices once. Keep the original coordinate system in returned metadata.
4. If the complete document fits the context allocation, include it and mark coverage `full`. Selection/cursor are focus hints, not permission to ignore the rest of a whole-document question.
5. For Ask, if the complete document does not fit, return `needs_context` before provider generation. Ask the user to narrow to a section or use the later full-source analysis path. Do not silently use the first few thousand characters.
6. For Draft, if the complete document does not fit, pack a bounded outline plus the active section, nearby prose and enough after-cursor text to avoid repetition. With no focus, continue from the end of the document. Mark coverage `partial`.
7. Split an oversized section at paragraph boundaries, then a safe character boundary if one block alone is oversized. Bound the outline too. Tables/code excerpts must be labeled as excerpts when incomplete.
8. Deduplicate overlapping source spans and restore document order. Preserve provenance for every included excerpt.

A full-source hash always describes the exact supplied source, even if only some of it is included. `full` means all source text was provided in that model call, not that the model understood it perfectly. An outline alone is never full coverage.

### Budget policy

The proposed starting document allocation is approximately **6,000 input tokens**, with explicit space outside it for system rules, task text, framing, output and model reasoning. It is a tuning target, not a fact about TinyMCE or a universal token ratio.

Keep a separate **24,000 UTF-8-byte serialized-context ceiling** as a simple transport/packing guard. Count escaped serialization and metadata too. This byte limit is not a substitute for token counting. If the allocation or the byte ceiling is exceeded, reduce the permitted Draft context or reject a whole-document Ask.

Use the installed provider SDK's token counter where available and appropriate. Claude documents preflight token counting as an estimate, and Gemini exposes token counting and model limits. Keep counter timeouts bounded; do not add a counter network call to every tiny request merely for a dashboard number. [Claude token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting), [Gemini tokens](https://ai.google.dev/gemini-api/docs/tokens)

For providers without a supported exact local counter, start with a deliberately conservative, tested estimate and record it as estimated. Before production enablement, verify the configured model's input/output limits and SDK behavior; this repository's output-limit comments already identify unverified provider limits. Do not promise context-window safety based only on `len(text) / 4`.

Use this constraint for every eligible candidate, including fallbacks:

```text
system + task + serialized context + output/reasoning allowance + safety margin
    <= verified candidate context window
```

Prefer one pack that fits all eligible fallback candidates. If a candidate cannot accept it, exclude that candidate or reprepare and update coverage metadata before any visible content; do not silently reduce evidence during fallback. Test token-heavy non-English text, code, long URLs and emoji.

Initial visible-output defaults can be 2,000 tokens for Ask and 1,000 for Draft, routed through the existing output-budget helper. Treat these as tunable defaults and handle truncation explicitly. Do not alter the legacy document-edit budgets.

## 7 Response and streaming contract

For new Ask/Draft requests, emit one additive `document_context_ready` event before content. Add a corresponding nullable `document_context` field to `ProcessResponse` and its actual JSON response construction.

Suggested metadata shape:

```text
action: ask | draft
response_kind: chat_answer | draft
status: ready | needs_context
source_sha256: hash of the complete exact source
coverage: full | partial | none
strategy: full | focused | none
source_chars: number of characters in the complete source
included_source_chars: unique source characters included
context_chars: characters of the serialized context envelope
included_ranges: source ranges in UTF-16 coordinates
estimated_input_tokens: optional estimate, explicitly named as such
```

For an empty source or oversized Ask, emit `status=needs_context`, no document text to the provider, a deterministic user-facing explanation, and normal completion. Do not reserve or charge a model generation that never ran. Coverage metadata and a chat explanation are preferable to returning a misleading “successful” document analysis.

Expected new stream:

```text
start
document_context_ready
content ...
done
```

Keep the existing legacy Edit sequence unchanged:

```text
start
document_edit_ready
content containing one verified complete snapshot
done
```

New explicit Edit uses that same edit contract. A client already knows it requested Edit and receives existing patch metadata; a second context event is unnecessary.

Never place a large context metadata object inside a plain content token. The frontend must not render it as prose or count it as output text. Process JSON and streaming metadata must describe the same source/action/coverage. Populate warnings consistently through the existing warning mechanism when Draft used partial context.

## 8 Billing and privacy changes required for release

### Billing

Prepare context before reserving credits. For the new one-call Ask/Draft path, define billable input characters as user query plus the serialized document-context envelope actually sent. Count duplicated focus excerpts only if actually repeated in the prompt; preferably avoid repeating them. Keep static system-prompt accounting consistent with the existing product policy.

Use this identical value for initial reservation, incremental output reservation, final settlement, failed-call reporting and the upstream process handoff. Retain exact legacy Edit accounting. Do not estimate a full-document output for Ask/Draft.

The project uses character-based customer credits, while providers meter tokens. Preserve that distinction. Keep actual provider input/output usage in `record_text_usage()` for cost analysis, and leave unknown cost unknown when no verified price exists. Do not silently change the backend credit calculator to token billing.

For `/agent/process`, return the same preparation metadata, including context character count, and update the calling backend to calculate the text charge from the selected input. Do not attach a second text-credit deduction inside this endpoint. The caller's `query` length alone is no longer a complete billing basis.

Test insufficient credits and per-request ceilings before model calls, zero-model early exits, BYOK eligibility, disconnect/cancellation, provider failures and settlement exactly once. A context-limit response should be discoverable before an expensive reservation.

### Privacy and isolation

Exclude the new context text and focus bodies from request/response body logs even when `LOG_REQUEST_BODIES=true`. Keep metadata such as lengths, strategy and coverage. Check existing query previews and error messages too; a provider exception must not echo the prepared prompt.

Do not reuse the current web sanitizer on exact editor text. It removes zero-width joiners and rewrites directive phrases, which can alter valid languages, emoji, quotations or security documentation. Keep source preservation and prompt-injection defenses separate. An escaped data copy may differ in serialization while the original source and its hash remain untouched.

Keep documents request-scoped. No document persistence or cross-request cache is needed in the first release. `document_id` alone is neither an access grant nor a cache namespace. If persistent retrieval is introduced later, the authenticated tenant/user plus document identity and revision must scope every read and cache entry.

The frontend Markdown renderer must sanitize generated content. Review links and remote image loading so model output cannot cause a browser request carrying document content to an attacker-controlled host. Disabling model web tools alone does not prevent output-rendering exfiltration.

## 9 Frontend integration

The editor integration is necessary for users to experience ambient awareness. Backend support alone will not supply a document snapshot.

1. Capture the active document, cursor and selection together when Send is pressed. Serialize once to the supported Markdown/plain-text representation. Send the exact same snapshot through the request and retain it for result handling.
2. Map editor positions to that serialization. ProseMirror/Tiptap positions, DOM offsets and Markdown UTF-16 offsets are different coordinate spaces. Use an explicit mapping or omit uncertain focus information; never guess an offset.
3. Show a simple document-context attachment indicator with a way to omit the source for a new request. Avoid implying that toggling it off erases already sent content from a provider or a retained conversation.
4. Use Ask by default. Offer Draft and Edit explicitly; a request action does not require an extra modal confirmation merely to generate a proposal.
5. Stream Ask into chat. Preview Draft with Insert. Buffer Edit's verified snapshot and show the existing patch review.
6. Before inserting a Draft, verify the captured source revision/hash and anchor still match. If typing changed the source, require regeneration or a deliberate insertion at the user's newly chosen position. Do not silently insert at an outdated offset.
7. Apply accepted Edit proposals through the existing hash-checked apply flow. Commit accepted changes as one undoable editor transaction, preserving rich nodes and the application's save behavior.
8. On document/tab changes or abort, stop rendering the old response into the new document. Bind each result to its request and source snapshot.
9. Test screen-reader labels, keyboard operation, focus restoration, selection preservation, undo, save/reload, simultaneous typing, and two documents open together.

Require context metadata before presenting a new-path response as document-aware. If an older backend ignores the new fields and never supplies that metadata, show an unsupported-feature result and disable Insert/Accept for that response. For new Edit, require the existing patch metadata. This also protects mixed-version deployment and rollback windows.

Preserve the typed `complete` contract: current local changes explicitly return original selected text plus completion because the frontend replaces the selection. Draft is an insertion suggestion with a different response destination. Mixing these is a direct data-loss risk.

Conversation history is a separate gap. The current general request does not expose a general chat-history contract. The initial feature can answer each request against a fresh snapshot; it should not promise that “make that shorter” remembers an earlier chat answer. Add bounded conversation turns only when needed, with the current source authoritative and history isolated by document and authenticated session.

## 10 Implementation sequence

The first four changes deliver the initial product. Later stages broaden capability without weakening its guarantees.

### Change 1 Define the contract and regression fixtures

Files: `api.py`, proposed `src/document_context.py`, proposed `tests/test_document_context.py`, `tests/document_edit/test_api.py`.

Implement strict nested models, action resolution, source exclusivity, supported-field validation and UTF-16 conversion. Keep the existing legacy validator behavior. Add small Markdown/plain-text fixtures covering empty sources, Unicode, repeated selections, headings and protected content.

Acceptance: both endpoints reject conflicting/malformed inputs before provider calls; old document-edit requests retain their existing responses and event sequence. Do not change the shared request base classes or reorganize the large API file merely to add these fields.

### Change 2 Prepare bounded context and metadata

Files: `src/document_context.py`, reuse `src/longform/markdown_ast.py`; add only the necessary model-limit metadata in `src/models.py` and an enable/disable field in `config.py`.

Implement pure preparation: exact spans, outline, full/focused packing, budget checks, escaped data envelope and deterministic no-model outcomes. Store the selected pack on request-local state for the endpoint to pass onward. Document initial budget constants; make them configurable only if deployment tuning requires it.

Acceptance: deterministic packing; no source mutation; bounded serialized size; oversized whole-document Ask never masquerades as a full answer; Draft reports partial coverage. Verify configured model capacities before enabling their new path.

### Change 3 Connect the runner and accounting

Files: `api.py`, `src/text_processing_agent.py`, `src/request_plan.py` only where the existing plan needs to recognize the new execution mode, `src/observability.py`, proposed API integration tests.

Thread prepared context through both API calls, `process_user_query_with_images()` and `process_user_query_stream()`. Keep new parameters optional so direct scripts and existing tests remain compatible. Add static document rules at the final prompt-building location. Set tools and route deterministically; reuse the candidate loop and metrics. Add context event/process metadata and consistent input-character accounting. Exclude document data from logs. Coordinate the process caller's billing change.

Acceptance: spy on the actual prompt passed to `agent.run()` and the actual constructed agent's tools/instructions. Assert no design/image/code/blog path runs, no document-edit call occurs for Ask/Draft, and new Edit invokes the same patch engine. Test both endpoints and provider fallback/BYOK.

Audit every caller of the touched runner signatures, including `process_user_query()`, evaluation/benchmark scripts and monkeypatched factories. `create_agent()` also serves `src/longform/llm.py`, so avoid changing its shared override semantics incidentally.

### Change 4 Integrate the editor and release documentation

Files: application editor/request client outside this repository; `docs/API_DOCUMENTATION.md`, `docs/FRONTEND_HANDOFF.md`, `docs/DOCUMENT_EDIT_API_HANDOFF.md`, `README.md`, and a proposed provider-free context demo.

Update the existing handoff instruction that currently tells every no-selection prompt to send `document_text`. That instruction would preserve the original problem even with a correct backend. Document Ask/Draft/Edit request examples, coverage events, exact serialization, credential differences, billing metadata and result application.

Acceptance: a browser demonstration answers a question using source-only facts, drafts a continuation without repeating existing text, proposes an edit without overwriting the document, and safely handles an intervening user edit. Include keyboard and undo/reload checks.

### Change 5 Add local questions for large documents

Implemented: `document_scope=document|selection|focused`, request-local FTS5/BM25, source-faithful bounded chunks, exact references, Unicode/CJK queries and no-model failure handling. Whole-document default behavior is unchanged.

Introduce an explicit task scope such as `document_scope="selection" | "document"` only when exposing this capability; default remains document. This avoids guessing that a present selection changes “summarize the whole draft” into a selection summary.

For selection/local tasks, retrieve source passages using selection/cursor and heading matches first. If evaluation shows a lexical search is needed, use a request-local in-memory SQLite FTS5 index after verifying support in the deployed Python SQLite build. FTS5 provides BM25 ranking without an external service. Escape/construct search terms safely and parameterize database queries. [SQLite FTS5](https://www.sqlite.org/fts5.html)

Chunk on actual Markdown section/paragraph boundaries, split large blocks, retain heading paths and source ranges, and include adjacent context when needed. Retrieve within a byte/token budget rather than assuming exactly three chunks always suffice. Return a scoped answer with verified source references. If retrieval finds no adequate evidence, say so; do not fabricate it from general knowledge.

Acceptance: local questions can find a relevant passage near the beginning, middle or end; repeated headings are disambiguated; a whole-document task never enters this partial retrieval strategy. Heading-free documents and non-English queries have explicit tests.

### Change 6 Add full-source summary and contradiction analysis

Implemented: explicit `document_analysis=summary|contradictions`, all-chunk extraction, synthesis and original-quote verification; pending-to-full coverage, progress, bounded calls/attempts/deadline and staged reserve/actual settlement. Evidence verification is model-based and non-exhaustive. Current bounds and billing fields are specified in the API handoff.

First evaluate whether the whole supplied document fits a supported model at acceptable cost and quality. A 100,000-character API cap does not itself require a vector store. Use a larger complete prompt where verified capacity and product budgets allow it.

When one-call full-source analysis is inadequate, add a bounded multi-stage workflow using existing `AgnoLLM`, `CallSpec` and the concurrency patterns in `src/longform/ordered_stream.py`:

1. Build sections/chunks covering every source span exactly once, with deliberate overlap only where provenance records it.
2. Extract per-section arguments, claims, figures, qualifications and evidence ranges. Validate quoted evidence against the source.
3. Combine section findings to form a summary or candidate contradictory pairs. Include comparisons between distant chunks, not just within each chunk.
4. Reopen original source passages for each candidate contradiction and check entities, time periods, units and qualifications before reporting it.
5. Show progress and state the analysis scope. Source coverage is not a guarantee that every possible contradiction was found.

Reserve a maximum number of calls, total input/output budget, bounded concurrency and a deadline before launching work. Check cancellation before each stage. Existing helpers cannot cancel every in-flight provider request; account for that cost.

Do not launch this workflow under the one-call credit formula. Estimate map inputs, reducer inputs, intermediate output and final output, then settle according to an explicit matching backend policy. Repeated source/intermediate content costs money even if users see one final answer. Preserve a single accountable reservation/settlement lifecycle and test failures between stages.

Acceptance: a planted inconsistency across the first and last sections is detected with valid quotes; differing dates/units are not falsely labeled a contradiction; a failed chunk prevents a full-coverage success claim; cancellation stops unscheduled work and releases unused reservation.

### Change 7 Add style-aware edits and optional intent suggestions

Implemented: explicit context Edit supplies up to three untargeted prose samples within 2,000 serialized UTF-8 bytes; legacy Edit and patch validators are unchanged. `suggest_document_intent=true` returns task-only heuristic advice without rerouting or applying changes. This release does not claim a trained multilingual classifier.

For style-aware Edit, pass a bounded, read-only `STYLE_REFERENCE` to the existing block writer through optional runtime/service arguments. Select representative untargeted prose outside protected/excluded areas. Keep target boundaries, frozen content, numerical checks and exact patch replay unchanged. Legacy callers without a reference retain current behavior.

Test that the reference cannot become an extra target, add unsupported facts or override the actual instruction. Only after this step should the product claim that editing a selected paragraph can match the rest of a long article.

For automatic routing, build a labeled set of actual tasks first. A lightweight classifier may suggest Ask, Draft or Edit based on the user task and selection state; it should not ingest the full untrusted document merely to choose an action. Explicit actions win. Ambiguous requests default to a chat response or a focused clarification. Automatic routing can create an edit proposal, but never accepts it or grants new write capabilities.

Acceptance: “What would you rewrite?” remains advisory; “Rewrite the conclusion” proposes edits; “Write the next section” drafts new content; “Summarize this” does not silently replace the source. Test multilingual and mixed-intent requests.

## 11 File and dependency checklist

| File or integration | Required change |
| --- | --- |
| `api.py` | Both request models/validators; preparation before billing and design planning; runner arguments; response schema and JSON; new metadata and character accounting |
| Proposed `src/document_context.py` | Strict context/range types; focus conversion; pure packer; serialization; metadata; deterministic context-limit outcomes |
| `src/text_processing_agent.py` | Both wrapper signatures; early document-context route; static prompt rules at effective builder; pass prepared data to the existing runner |
| `src/request_plan.py` | Reuse existing plan fields; add only behavior required for the document execution mode |
| `src/models.py` | Verified candidate capacities/budget support as needed; retain model IDs and existing output helper |
| `config.py` | One deployment kill switch for new context behavior; avoid an initial collection of independent flags |
| `src/observability.py` | Content exclusion and new context metadata; preserve existing actual-usage accumulation |
| `src/document_edit/runtime.py`, `service.py` | No initial verifier changes; optional style-reference propagation in Change 7 |
| `src/document_edit/schema.py`, `router.py` | Existing patch/application contract retained; insertion operations remain out of scope |
| Tests and fixtures | API validation/parity, packing, focus conversion, prompt/tool inspection, accounting, fallback, privacy, concurrency and legacy edits |
| Demo/evaluation scripts | Optional new arguments default safely; add synthetic source-grounding and coverage examples |
| README and existing API/frontend handoffs | Correct the no-selection advice and document the new additive contract |
| Upstream backend outside this repo | Request forwarding, process credit calculation, result metadata forwarding |
| Editor outside this repo | Snapshot serialization/mapping, action controls, rendering, insert/review behavior, undo/save, stale-source handling |
| Dependencies/exports | No new runtime dependency for the first release. Follow the existing direct module imports; no package-wide public export is needed unless an actual caller uses it. |

## 12 Acceptance tests and evaluation

Use deterministic tests for boundaries and model evaluations for semantic quality. A mocked model proves routing and preservation, not accurate summaries.

| Test case | Required result |
| --- | --- |
| Ask about a unique invented fact present only in the source | The fact appears in the actual model input; answer is checked against it |
| Ask when answer is absent | Clear absence statement; no invented source claim |
| Draft at the cursor with text after it | New content fits between both sides and does not repeat the suffix |
| Full summary with a selected paragraph | Uses complete source if it fits; selection does not secretly narrow scope |
| Whole-source Ask exceeds allocation | `needs_context`; no model call; no generation charge |
| Large Draft | Bounded focused context; partial coverage and warning |
| New explicit Edit | Existing `document_edit_ready`, source hash and patch verifier |
| Legacy `document_text` request | Existing validations, output and event sequence unchanged |
| Document containing image requests, blog requests, code and URLs | No competing route, no web/image tools, no design price |
| Fake system roles, closing tags and embedded instruction attacks | Data remains serialized; no tools/actions become available; evaluate answer contamination separately |
| Quoted security guidance and multilingual joiners | Source fidelity retained; no destructive generic sanitization |
| Selection appears twice in source | Exact offsets choose the captured occurrence |
| Emoji before/inside focus; CRLF and BOM | Correct source coordinates and hash |
| Two simultaneous documents and users | No context or metadata crosses requests |
| Provider fallback before content | Same permitted evidence and correct provider metadata |
| Failure after visible content | No restart that duplicates or changes the answer |
| Body logging enabled | Raw context/focus excluded from logs; no prompt in error details |
| Insufficient credits, abort, failed call, early limit response | Correct reservation/settlement once; no unauthorised provider candidates |
| User types or switches document before Insert/Accept | Stale response cannot overwrite or insert into the wrong document |
| Existing `complete`, autocomplete, charts and typed operations | Current contracts retained |

Create a small labeled evaluation set: approximately 20 document questions, 15 continuations, 15 edit/advisory distinctions, 10 coverage/long-source cases and 10 injection/source-fidelity cases. These are starting sample counts, not an accuracy guarantee. Include supported languages and documents with evidence in different positions.

Score source correctness, unsupported claims, relevance, duplication, task/action agreement and source-reference validity. Use human review for contradictory claims and stylistic fit. Record the actual configured provider/model, context strategy, input/output usage and failure rate. Run live evaluations only on synthetic or explicitly permitted documents.

For performance, measure time to first visible content and total latency separately from response headers or progress events. Record p50/p95, source size and estimated/actual token sizes. Set a product latency target after collecting the deployed baseline; the blueprint's 1.2-second claim is not a release assertion.

The current security/output-budget/task-routing tests are also relevant. After implementation, run targeted context tests first, then the existing document-edit, request-planning, type-routing, provider-fallback, security and output-limit tests, followed by the full repository suite before release. Browser acceptance remains a separate gate.

## 13 Rollout and completion criteria

Release the additive API behind a deployment switch. Requests with the new fields must receive an explicit feature-disabled response while it is off; ignoring them would recreate document-blind behavior. Legacy requests remain available.

Order deployment as backend contract and accounting support, compatible client forwarding, then editor enablement. This prevents a new frontend from receiving an old backend's successful-but-context-free answer. Test the new-field behavior against the real deployed backend rather than assuming unknown fields are rejected; current top-level models do not generally forbid extras.

Start with internal/small cohorts using the actual supported providers. Observe action agreement, context-limit rate, grounding failures, time to first content, actual token usage, credit reconciliation and privacy checks. Expand only after the browser flow and billing reconcile. Rollback disables the new path and its UI while preserving legacy patch edits.

The initial release is complete when a user can ask a grounded question about an attached document, draft a new passage with explicit insertion, and request an existing supported edit with review; no incompatible legacy behavior changes; both endpoints agree on action and coverage; accounting includes selected context; and stale/other-document responses cannot mutate the editor.

A rough planning allowance is **5–8 backend engineering days, 2–4 editor/backend-integration days and 1–2 evaluation/release days**, with some work overlapping. This is an estimate, not a commitment; the absent frontend code and provider-limit validation are the largest unknowns. Large-document retrieval, multi-stage analysis and advisory intent suggestions are now implemented in this backend; editor integration remains separate.

## 14 Work deliberately deferred

Defer embeddings/vector infrastructure, persistent document memory, file-upload parsing, arbitrary HTML round trips, general chat history, changes to autocomplete, editor replacement, structural patch operations and model/provider migrations. Each adds a separate correctness or lifecycle problem and is unnecessary for the first useful release.

The principal remaining risks are model hallucination despite supplied evidence, incomplete analysis when context is partial, prompt-injection influence on generated prose, serialization/position mistakes in the editor and inconsistent billing between endpoints. The proposed evidence checks, scope metadata, tool restrictions, stale-source handling and integration tests address these directly; none establishes perfect semantic accuracy.
