# Content Refinement & Visually Rich Blog Generation — Integration Plan

*Status: **Mode 1 (structure-preserving refinement) and Mode 2 (visually rich blog generation) are implemented**, behind `LONGFORM_REFINE_ENABLED=false` and `BLOG_AI_ENABLED=false` (see §13). Date: 2026-10-22. Scope: this agent service + the Eddyter editor contract.*

Two new modes, both built on what the repo already does well (deterministic planning in code, validate → one
repair pass, the `placeholder::<uuid>` + `image_ready` image contract):

| Mode | Input | Output | Problem today |
|---|---|---|---|
| **1. Structure-preserving refinement** | Up to ~5,000 words of pasted Markdown with tables, blockquotes and images, through `enhance_with_ai`, `fix_grammar`, `change_tone`, `simplify` | The same document, prose polished, every table / quote / image byte-identical and in place, length within ±15% | One LLM call over the whole text. Structure is protected by one prompt line ("Preserve any existing formatting"). No output-token limit is set, nothing is verified, nothing is restored |
| **2. Visually rich blog generation** | A topic or short brief | An editorial-grade post with at least one table, several pull quotes and real inline images | `general` writes a wall of prose; tables appear only when asked, images only when the user says "with 2 images", quotes never |

---

## 1. Diagnosis (what the code does today)

Mode 1 path: `api.py` builds `"Enhance: <secondary>, Text: <query>"` (`build_full_query`), `process_user_query_stream()`
runs one agent from `AGENT_CONFIGS` and streams whatever comes back. Four concrete causes of the failures in the spec:

1. **No output budget.** `get_model()` never sets `max_tokens`. agno's `Claude` defaults to 8,192 output tokens; a
   5,000-word Markdown document with tables is 7,000–9,000 tokens, so Claude truncates it, and the other providers
   are left on their own defaults. Nothing detects a truncated answer.
2. **Length is a prompt wish.** No instruction says "same length"; no code measures it. Models shorten long inputs.
3. **Structure is a prompt wish.** Tables, quotes, image links and URLs pass through the model and are re-typed by it.
   One dropped pipe or an altered URL is invisible to the server.
4. **Short-input assumptions.** `grammar`, `simplify`, `tone` run on the basic tier with no streaming-by-section, so
   a long document also has the worst first-token latency (the whole thing is one response).

Mode 2 path: `general` → `smart_global`, prompt from `build_smart_global_instructions(plan)`. The plan
(`src/request_plan.py`) grants images only from the user's words (`extract_image_count`), the prompt mentions tables
and lists only as "when the user asks", and there is no notion of quotes, visual density or an outline.

What is already in place and reused below:

- `src/email_template/` — the pattern for Mode 2: pydantic schema with `extra="forbid"`, byte-stable system prompt,
  request data in the user message, `validate()` → one `repair_message()` retry, injectable `LLM` callable for tests.
- `src/image_gen.generate_image_unified` + `pending_images` + `resolve_image_placeholders` + the `image_ready` /
  `image_failed` SSE events — the whole media-fulfilment layer already exists (Gemini default, ~3.5 s and ~$0.03 per
  image, S3/CDN upload, credits via `image_billing`).
- `src/request_plan.py` — `needs_web`, `extract_image_count`, `_WORDCOUNT`, `ToolBudget` (thread-safe, shared
  across parallel calls).
- `src/task_router.py` — `writing` task defaults to OpenAI with a fallback tree; `generate_blog` joins it.
- `src/code_validate.py` + the `content_replace` SSE event — validate-after-stream precedent (frontend support pending).
- `markdown-it-py` 4.2.0 is already installed (transitively via `rich`). With the CommonMark preset plus the `table`
  rule it gives a line range (`token.map`) for every top-level block: heading, paragraph, table, blockquote, list,
  fence, html block, image-only paragraph (verified locally). That is all the parser Mode 1 needs: we slice the
  user's original text by line, never re-render it, so round-trip fidelity is exact. Pin it in `requirements.txt`.
- `_TOOL_EXECUTOR` (16 threads) and `ThreadPoolExecutor` use in the email service — parallel calls per request.
- `src/observability.py` traces (`trace_set`, `record_text_usage`) and `scripts/run_local_server.py` for local runs.

---

## 2. Locked decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Markdown in, Markdown out.** The editor already converts Markdown to Lexical nodes; no AST format is introduced. HTML input is converted to GFM at the API boundary if the editor sends HTML (open item O1). | Zero editor change for the output; one place to normalise. |
| D2 | **Code owns structure, the model owns prose.** Tables, blockquotes, images, code fences, raw HTML and link URLs are frozen into tokens before the model sees the text and restored by code afterwards. | Preservation becomes a property of the pipeline, not of the prompt. |
| D3 | Tables are frozen whole in v1 (no cell polishing). Blockquotes are frozen whole in v1 (`LONGFORM_FREEZE_QUOTES=true`). | The spec's acceptance test is "exact tables, exact quotes". Cell polishing is a later, separately validated step (§5.10). |
| D4 | Chunks of ~1,000 words (600–1,400), cut only between top-level blocks, preferring a cut before an H2/H3. A block larger than the maximum is its own chunk. | Never splits a list, quote or table; keeps sections together for tone consistency. |
| D5 | Chunks run **in parallel** (default 3 at a time) and are **emitted in document order**. A chunk is streamed to the client only after it passed verification. | Total time ≈ two chunk-rounds instead of one 90 s response; no bad text reaches the editor. Live streaming of unverified chunks waits for `content_replace` support (O3). |
| D6 | Every chunk is **verified by code** (token set and order, heading count/levels, list shape, paragraph count, word ratio). Fail → one repair call → still failing → the original chunk is restored unchanged and a warning is returned. **The document is never summarised and never loses a block.** | Deterministic guarantee; the repair pattern is the one `email_template/service.py` uses. |
| D7 | The chunked path runs only for documents that qualify: ≥ `LONGFORM_REFINE_MIN_WORDS` (600) **or** containing any table/quote/image. Short plain text keeps today's single call. | No latency change for the quick edits that make up most traffic. |
| D8 | Mode 1 v1 covers `enhance_with_ai`, `fix_grammar`, `change_tone`, `simplify`. `make_longer` follows with its own ratio; `make_shorter`, `translate`, `complete`, `emojify` later. | Length parity is the acceptance rule; the shrink/grow operations need their own ratios and tests. |
| D9 | Mode 2 gets an explicit request type **`generate_blog`** (toolbar action), task `writing`. Promotion from `general` by start-anchored regex ships off and in shadow mode first, like code intent. | The `type` field decides the agent (house rule); nothing is guessed from text until logs prove the regex. |
| D10 | Blog = three stages: **outline (JSON, validated) → sections (parallel, ordered) → lint**. Visual quotas (tables, quotes, images) are decided by code from length, before the model writes a word. | The model cannot "forget" a table: the outline assigns it, the lint checks it. |
| D11 | **The model never writes image markup.** Code inserts `![alt](placeholder::<uuid>)` at outline-assigned positions and starts generation right after the outline validates, in parallel with drafting. No `<ImagePrompt>` tags, no post-scan. | Hallucinated URLs are impossible by construction; images are usually ready before the text finishes. |
| D12 | Image source v1 = the existing adapter (Gemini default). Stock photos (Pexels/Unsplash) and Mermaid diagrams sit behind the same `media.py` interface, later (O4, O5). | No new vendor, keys, attribution rules or editor renderer needed to ship. |
| D13 | **Quotes are never invented.** A pull quote is an unattributed key sentence in the author's voice. An attributed quote appears only when a page was actually read through the web tools, with its link. No quote is attributed to a named real person otherwise. | Same rule as the existing "never cite what you did not retrieve". |
| D14 | Visual density is a lint: no more than 220 words of prose between visual breaks (heading, table, quote, image, list, code), ≤ 1 bulleted list per section, paragraphs ≤ 4 sentences. One repair per failing section; then accept with a warning. | A blog that is 95% right ships; a blocked request helps nobody. |
| D15 | Output budget is explicit on every call: `max_tokens` sized from the expected output (chunk ≈ 1.6 × input tokens, section ≈ 3 × target words, outline 2,000). | Removes silent truncation; cheap insurance even for the single-call path (Phase 0). |
| D16 | Billing unchanged in kind: text per streamed character, images through `image_billing`; blog images reserved up front after the outline (non-BYOK). `/agent/process` keeps backend-owned billing. | Nothing new for the backend except an additive `warnings` field. |
| D17 | Every new behaviour has a kill switch in `config.py` and ships **off**. | House rule. |
| D18 | System prompts are byte-stable per (operation, mode); all per-request data goes in the user message. | Prompt-cache hits (Claude `cache_system_prompt=True` already on). |

---

## 3. Mode 1 — Structure-preserving refinement

### 3.1 Lifecycle (`/agent/stream`, `type: enhance_with_ai`, 5,000-word Markdown)

```
request (type, query = document, secondary_prompt = instruction)
  │
  ├─ 1. qualifies?            words ≥ 600 or any table/quote/image → chunked path, else today's single call
  ├─ 2. normalise             HTML → GFM if needed (O1); CRLF, BOM, tabs; keep original text for slicing
  ├─ 3. parse                 markdown-it (commonmark + table) → top-level blocks with line ranges
  ├─ 4. freeze                tables, blockquotes, images, fences, html, link URLs → [[TABLE_1]] ... tokens
  ├─ 5. chunk                 ~1,000 words, cut between blocks, prefer before H2/H3
  ├─ 6. refine (parallel 5)   per chunk: system prompt (operation) + user message (instruction, context, chunk)
  ├─ 7. verify + repair       tokens, headings, lists, paragraphs, word ratio → repair once → restore original
  ├─ 8. emit in order         SSE `content` per verified chunk, `refine_progress` between chunks
  ├─ 9. thaw                  tokens → original tables/quotes/images/URLs (exact bytes)
  └─ 10. done                 `done` carries `warnings[]`; `/agent/process` returns the joined text + `warnings[]`
```

### 3.2 Normalise (`src/longform/markdown_ast.py`)

- `parse(text) -> list[Block]`: `Block(kind, start_line, end_line, text)` from markdown-it's `token.map`, slicing the
  original lines. Kinds: `heading(level)`, `paragraph`, `image` (paragraph whose only inline child is an image),
  `table`, `blockquote`, `list(ordered, shape)`, `fence`, `html`, `hr`, `blank`.
- `join(blocks) == text` is a tested invariant (round trip is identity on every fixture).
- Counts helpers used by verification: `shape(blocks)` = the sequence of block kinds; `heading_levels`, `list_shape`
  (sequence of item depths), `word_count` (prose only: tokens are not words).
- Input that looks like HTML (`<p`, `<table`, `<img`) is converted with `markdownify` (new, small dependency) only if
  O1 says the editor sends HTML; otherwise this step is a no-op.

### 3.3 Freeze (`src/longform/freeze.py`)

| Element | Token | Placed | Model may change |
|---|---|---|---|
| GFM table | `[[TABLE_n]]` | own line, replaces the block | nothing |
| Blockquote (any depth, multi-paragraph) | `[[QUOTE_n]]` | own line | nothing (flag `LONGFORM_FREEZE_QUOTES`) |
| Image-only paragraph `![alt](src)` | `[[IMAGE_n]]` | own line | nothing (alt text stays frozen too) |
| Inline image inside prose | `[[IMAGE_n]]` | inline | nothing |
| Fenced / indented code, raw HTML block | `[[CODE_n]]`, `[[HTML_n]]` | own line | nothing |
| Link `[text](url "title")` | `[text](link://n)` | inline | the anchor text only |
| Bare URL / autolink | left in place | – | nothing (verified by URL-set equality) |
| Headings | left in place | – | wording; never the level or the count |
| Lists | left in place | – | item wording; never the item count or nesting |
| Emphasis, inline code | left in place | – | preserved by instruction (not structural) |

`freeze(text) -> (frozen_text, Vault)`; `thaw(text, vault) -> text`. The vault maps token → original bytes.
Tokens are numbered per document so a chunk's expected token list is a strict subsequence of the vault.
The token regex is the only thing `verify.py` and `thaw()` need: `\[\[(TABLE|QUOTE|IMAGE|CODE|HTML)_(\d+)\]\]` and
`link://(\d+)`.

### 3.4 Chunk (`src/longform/chunk.py`)

- Greedy over blocks: add blocks until ≥ `LONGFORM_CHUNK_WORDS` (1,000); close the chunk at the next heading
  boundary if one arrives before 1,400 words, else at the next block boundary.
- Start a new chunk at an H2/H3 whenever the current chunk already has ≥ 600 words.
- A single block above 1,400 words (one giant paragraph) is a chunk by itself.
- Each chunk records: its blocks, expected tokens (in order), heading levels, list shape, paragraph count,
  prose word count, the previous chunk's last ~60 words and the next chunk's first heading (context only).

### 3.5 Refine (`src/longform/refine.py`, `src/longform/prompts.py`)

- One byte-stable **system prompt per operation** (`enhance`, `grammar`, `tone`, `simplify`): the operation's
  existing rules from `AGENT_CONFIGS`, plus the locked-structure contract:
  - the `[[...]]` tokens and `link://n` references are locked: output each exactly once, in the same order, on
    their own line where they were on their own line;
  - keep every heading at its level, every list with the same number of items, every paragraph as a paragraph;
  - output the same passage at the same length (operation-specific ratio); never summarise, merge or drop;
  - output only the refined passage, no preamble, no fences.
- **User message** (data, not instructions): the user's instruction (`secondary_prompt`, e.g. the tone), the
  context lines marked "for continuity only, do not output", then the chunk between `<<<PASSAGE` / `PASSAGE>>>`.
- Model: the operation's tier as today (`grammar`/`simplify`/`tone` basic, `enhance` advanced), the operation's
  temperature, no tools, `max_tokens` from D15. Provider candidates come from the same `resolve_route()` list as
  today; a retryable error before the first token of a chunk moves that chunk to the next candidate.
- **Parallel + ordered** (`src/longform/ordered_stream.py`): submit chunks to a pool (`LONGFORM_PARALLEL`, 5 — measured, §13);
  the generator yields chunk *i* only after chunks < *i* were yielded. Each chunk is buffered until verified (D5).
  Yields a `{"__progress__": {"chunk": i, "of": n}}` sentinel between chunks for the `refine_progress` event.

### 3.6 Verify, repair, restore (`src/longform/verify.py`)

| Check | Rule | On failure |
|---|---|---|
| Tokens | same multiset and order as expected; none invented | repair → restore |
| Headings | same count and levels | repair → restore |
| Lists | same list shape (item count and nesting) | repair → restore |
| Paragraphs | `grammar`: equal count; others: within ±25% | repair → restore |
| New structure | no table / blockquote / image / fence appears that was not a token | repair → restore |
| Word ratio | `enhance`, `grammar`, `tone`: 0.85–1.25; `simplify`: 0.6–1.1 (configurable) | repair → restore |
| URLs | bare URLs identical | restore the URL in place |
| Wrapping | strips a leading/trailing code fence or "Here is…" line (existing `/agent/process` logic, moved here) | fix in place |

Repair = the same conversation plus a numbered error list ("1. [[TABLE_2]] is missing. 2. 412 words; the passage
has 980. Return the full passage."), temperature 0. Restore = the chunk's original text, and
`warnings.append({"chunk": i, "kind": ..., "detail": ...})`. Trace: `longform={chunks, parallel, repaired, restored,
word_ratio, seconds}`.

### 3.7 Output contract

- `/agent/stream`: unchanged events (`start`, `content`, `done`, `stream_end`, `error`) plus two **additive** ones:
  `{"type": "refine_progress", "chunk": 2, "of": 5}` and `warnings` inside `done`
  (`{"type": "done", ..., "warnings": [...]}`). Clients that ignore both keep working.
- `/agent/process`: `markdown_response` is the thawed full text; adds `warnings: []`.
- Credits: output characters are counted when a verified chunk is emitted (same per-1,000-char reservation loop).

### 3.8 Hook point

`process_user_query_stream()` after the candidate list is resolved and before the agent loop:

```python
if agent_type in LONGFORM_AGENTS and settings.longform_refine_enabled and longform.qualifies(user_text):
    yield from longform.refine_stream(user_text, instruction, agent_type, candidates, key_resolver, ...)
    return
```

`api.py` passes the raw `request_data.query` and `secondary_prompt` through as new keyword arguments
(`user_text`, `instruction`); the existing `full_query` string is still built for the single-call path, so nothing
else moves.

### 3.9 Latency (estimate, to be measured in Phase 5)

A 1,000-word chunk is ~1,400 output tokens: ~10 s on the basic tier, ~15–18 s on the advanced tier. Five chunks at
parallel 3 → two rounds → ~25–40 s wall-clock for 5,000 words, first text at ~10–18 s, versus one 7,000-token
response today (60–90 s when it is not truncated). Phase 5 measures p50/p95 per provider. **Measured** (§13): the basic
OpenAI tier took ~34 s per 1,000-word chunk and the advanced tier ~12 s, so the estimate held only for the advanced tier;
that moved the defaults to `LONGFORM_PARALLEL=5` and a 300-word first chunk.

### 3.10 Later (not v1)

Table cell polishing: send each table as JSON rows, validate same row/column count and that numbers and dates
are unchanged, restore on failure. Live streaming of chunk 1 with `content_replace` on verification failure (O3).
`translate` on the chunked path with tables translated cell-wise. `make_shorter` with ratio 0.4–0.8.

---

## 4. Mode 2 — Visually rich blog generation

### 4.1 Lifecycle (`type: generate_blog`)

```
request (query = topic/brief, secondary_prompt = style notes, blog_options?)
  │
  ├─ 1. plan (code)           target words, section count, visual quotas, image count, web yes/no
  ├─ 2. outline (LLM, JSON)   title, sections with angle + target words + assigned visuals → validate → repair → fill
  │       └─ `outline` SSE event (additive); image generation starts now, in background, credits reserved
  ├─ 3. sections (LLM ×n)     parallel 3, ordered emission; code writes headings and inserts image placeholders
  ├─ 4. lint + repair         per section: assigned table/quote present, density, length, no URLs/placeholders
  ├─ 5. emit                  `content` in order; `image_ready` as images finish; `done` with warnings
  └─ 6. /agent/process        waits for images, resolves placeholders (existing), returns text + warnings
```

### 4.2 Entry points

- `VALID_AGENT_TYPES` + `AGENT_FOR_REQUEST_TYPE`: `generate_blog` → `blog_writer`; `TYPE_PREFIX_MAP` entry `''`.
- `WRITING_REQUEST_TYPES += ('generate_blog',)` so the task is `writing` (OpenAI default, Claude fallback).
- Optional `blog_options` on both request models: `{"length": "short|standard|long", "images": int|"auto",
  "tone": str, "audience": str}`; every field optional; validated with `extra="forbid"`.
- `general` promotion (`src/blog/intent.py`, flag `BLOG_INTENT_REGEX_ENABLED`, `BLOG_INTENT_SHADOW_MODE`): start-
  anchored after the polite-filler prefix from `src/intent.py`: `write|draft|create (me )?(a|an) (blog( post)?|
  article|guide|long[- ]form piece) (about|on|covering)`. Image intent and code intent are checked first, as today.

### 4.3 Plan (`src/blog/plan.py`, `BlogPlan`, frozen dataclass)

| Field | Rule |
|---|---|
| `target_words` | explicit number in the brief (`_WORDCOUNT`) → `blog_options.length` (short 800, standard 1,450, long 2,200) → 1,450 (the plan said 1,400; see §13, 16) |
| `sections` | 4 (≤ 1,000 words), 5 (≤ 1,600), 6 (≤ 2,200), 7 above; intro and conclusion are extra |
| `tables` | ≥ 1; ≥ 2 from 2,000 words |
| `quotes` | ≥ 2; ≥ 3 from 2,000 words |
| `images` | explicit count in the brief (`extract_image_count`) → `blog_options.images` → `1 + target_words // 700`; capped by `IMAGE_MAX_PER_REQUEST` (4); **0 when no image provider is usable** (`_image_tool_available`) |
| `use_web` | `needs_web(brief)` as today; budgets shared across sections through `ToolBudget` |
| `max_prose_run` | 220 words (`BLOG_MAX_PROSE_RUN`) |

### 4.4 Stage 1 — outline (`src/blog/schema.py`, `src/blog/outline.py`)

```json
{
  "title": "…", "subtitle": "…",
  "intro": {"target_words": 120, "hook": "…"},
  "sections": [
    {"id": "s1", "heading": "…", "angle": "what this section argues, 1–2 sentences", "target_words": 260,
     "visuals": [
       {"kind": "image", "prompt": "…", "alt": "…", "caption": "…", "position": "after_first_paragraph"},
       {"kind": "table", "title": "…", "columns": ["Option", "Cost", "Best for"], "rows_hint": 4},
       {"kind": "quote", "mode": "pull"}
     ]}
  ],
  "conclusion": {"heading": "…", "target_words": 120, "takeaways": 3}
}
```

Schema (pydantic, `extra="forbid"`): section count = plan, `sum(target_words)` within ±15% of the plan, visual
counts ≥ quotas, ≤ 1 image per section, no two consecutive sections without a visual, 2–5 table columns, image
prompts pass the email image-prompt rules (no text or logos in the picture, no real people, subject centred).
Validation failure → one repair call (numbered errors) → still failing → **rule-based fill**: a missing table goes
to the section whose angle mentions compare / versus / options / cost / steps, else section 2; missing quotes to the
two longest sections; missing images to section 1 and the middle section. Fields the code owns (ids, positions)
are overwritten, never trusted. Model: advanced tier, temperature 0.4, no tools, `max_tokens` 2,000, JSON parsed by
the same tolerant `parse_json` as email. The outline is never streamed as content; an additive
`{"type": "outline", "title": ..., "sections": [{"id", "heading"}]}` event lets the editor show a skeleton at once.

### 4.5 Media start (`src/blog/media.py`)

Right after the outline validates: for each image directive, `placeholder_id = uuid4()`, a `pending_images` entry
in the exact shape the direct image route builds (status, prompt, provider, model, credits), credits reserved for
non-BYOK users (`credit_manager.reserve`, as the direct route does), then `generate_image_unified(...)` on a
background thread with `aspect_ratio: "16:9"`, quality `medium`, provider from `plan_image_target`. Failures mark
the entry `failed` → `image_failed` event; `/agent/process` swaps the placeholder for the existing
"image could not be generated" note. `BLOG_IMAGE_SOURCE = generate` in v1; `stock` and `diagram` are later
backends behind the same `fulfil(directive) -> placeholder_id` interface (O4, O5).

### 4.6 Stage 2 — sections (`src/blog/draft.py`, `src/blog/prompts.py`)

- One call per unit: intro, each section, conclusion. Parallel 3, ordered emission through the shared
  `ordered_stream.py` from Mode 1.
- **Code writes the `## Heading` line** and inserts the image block (`![alt](placeholder::<uuid>)` followed by an
  italic caption line) at the directive's position. The model writes only the body of its section.
- System prompt (byte-stable): house style — editorial voice, paragraphs ≤ 4 sentences, no more than 220 words
  without a visual break, ≤ 1 bulleted list (≤ 7 items) per section, prefer a numbered list for procedures, GFM
  table rules (header row, 2–5 columns, every row complete, real values from the brief / research / general
  knowledge, never fabricated statistics presented as measured), quote policy (D13), no URLs unless read through
  the web tools, no image markup, no "Section 1:" labels, no preamble.
- User message (data): the full outline (for coherence), this unit's directives, the previous section's `angle`
  ("already covered, do not repeat"), target words, the user's style notes.
- Model: advanced tier, temperature 0.7, web tools when `plan.use_web`, `max_tokens = 3 × target_words`.

### 4.7 Stage 3 — lint and repair (`src/blog/validate.py`)

| Check (per unit) | On failure |
|---|---|
| Assigned table present: `table_open` from markdown-it, header with ≥ 2 columns, ≥ 2 data rows, consistent width | repair once |
| Assigned quote present: a `blockquote` block; attributed only when a web page was read in this request | repair once; an attributed quote without a read page is turned into an unattributed pull quote by code |
| Longest prose run ≤ `max_prose_run` | repair once |
| Word count within ±35% of target | repair once |
| No raw URL (unless web-sourced), no `placeholder::`, no `<ImagePrompt`, no heading lines (code owns them) | strip in place |
| Whole article: ≥ quota tables and quotes, images placed = planned, headings = outline | report |

A unit still failing after its repair is accepted with a warning (D14). Trace:
`blog={sections, tables, quotes, images, repairs, warnings, outline_seconds, draft_seconds}`.

### 4.8 Output contract

- `/agent/stream`: `start` → `outline` (additive) → `content` per unit in order → `done` (with `warnings`) → the
  existing image wait → `image_ready` / `image_failed` → `stream_end`. Content is plain GFM: `##` headings, `|`
  tables, `>` quotes, `![alt](url)` images with an italic caption line beneath.
- `/agent/process`: full Markdown with image URLs resolved (existing 120 s wait), plus `warnings`.
- Credits: text per streamed character; images reserved after the outline and settled by `image_billing` as today.

---

## 5. Module layout

```
src/longform/                 # shared by both modes
  __init__.py
  markdown_ast.py             # parse → blocks with line ranges; join; shape/count helpers
  freeze.py                   # freeze()/thaw(), Vault, token regex
  chunk.py                    # structure-aware chunking (§3.4)
  ordered_stream.py           # parallel workers, in-order emission, progress sentinels
  verify.py                   # checks table (§3.6), repair errors, restore
  prompts.py                  # per-operation system prompts + user message builder
  refine.py                   # refine_stream(): qualifies(), the Mode 1 pipeline
src/blog/
  __init__.py
  plan.py                     # BlogPlan (§4.3)
  schema.py                   # Outline pydantic models + limits
  prompts.py                  # outline / section / repair prompts
  outline.py                  # outline call, validate, repair, rule-based fill
  media.py                    # image directives → pending_images entries + background generation
  draft.py                    # unit calls, heading + image insertion, ordered emission
  validate.py                 # lint (§4.7)
  intent.py                   # optional regex promotion from `general`
  service.py                  # generate_blog_stream(): the Mode 2 pipeline
```

Existing files touched:

| File | Change |
|---|---|
| `src/text_processing_agent.py` | `get_model(max_tokens=...)` (cache key includes it); hook in `process_user_query_stream` for both modes; `AGENT_CONFIGS["blog_writer"]`; length/structure lines in the four refinement prompts (Phase 0) |
| `api.py` | `generate_blog` in `VALID_AGENT_TYPES`/`TYPE_PREFIX_MAP`; `blog_options`; pass `user_text`/`instruction`; `refine_progress`/`outline` events; `warnings` in `done` and in the process response |
| `src/task_router.py` | `WRITING_REQUEST_TYPES += ('generate_blog',)` |
| `config.py` | flags (§6) |
| `src/observability.py` | `longform` and `blog` trace fields; `latency_stats()` grouped by them |
| `requirements.txt` | pin `markdown-it-py`; `markdownify` only if O1 needs it |
| `docs/API_DOCUMENTATION.md`, `README.md` | contract and agent table |

---

## 6. Configuration (all new, all off or inert by default)

| Variable | Default | Meaning |
|---|---|---|
| `LONGFORM_REFINE_ENABLED` | `false` | Chunked, verified refinement for the Mode 1 types |
| `LONGFORM_REFINE_MIN_WORDS` | `600` | Below this and with no structure, keep the single call |
| `LONGFORM_CHUNK_WORDS` | `1000` | Target chunk size (600–1,400 window derived from it) |
| `LONGFORM_PARALLEL` | `5` | Chunks in flight per request |
| `LONGFORM_CALL_TIMEOUT_SECONDS` | `150` | One model call; a call with no answer by then is a retryable failure (`0` = no limit) |
| `LONGFORM_FIRST_CHUNK_WORDS` | `300` | Size of the first chunk only, so text reaches the reader sooner (`0` = off) |
| `LONGFORM_FREEZE_QUOTES` | `true` | Blockquotes frozen (false = polished, still verified) |
| `LONGFORM_REPAIR_ENABLED` | `true` | One repair call per failing chunk |
| `LONGFORM_WORD_RATIO` | per operation | JSON override of the ratio windows |
| `LONGFORM_STREAM_LIVE` | `false` | Stream chunk 1 before verification, fix with `content_replace` (O3) |
| `BLOG_AI_ENABLED` | `false` | `generate_blog` returns 503 when off |
| `BLOG_IMAGE_SOURCE` | `generate` | `generate` / `stock` / `none` |
| `BLOG_SECTION_PARALLEL` | `3` | Units in flight per request |
| `BLOG_MAX_PROSE_RUN` | `220` | Density rule |
| `BLOG_REPAIR_ENABLED` | `true` | One repair per failing unit / outline |
| `BLOG_INTENT_REGEX_ENABLED` | `false` | Promote `general` requests to the blog pipeline |
| `BLOG_INTENT_SHADOW_MODE` | `false` | Log the decision only |

Reused as-is: `IMAGE_MAX_PER_REQUEST`, `IMAGE_WAIT_SECONDS`, `WEB_*_CALL_LIMIT`, `MODEL_TIER_ROUTING_ENABLED`.

---

## 7. Phases

Each phase is independently shippable, behind its flag, with current behaviour unchanged when the flag is off.

### Phase 0 — Quick wins on the single-call path *(S, ship first, no flag)*

Files: `src/text_processing_agent.py`, `src/models.py`, `requirements.txt`, tests.

- `get_model(..., max_tokens=None)`: per-provider output limit (`Claude.max_tokens`, `OpenAIChat.max_completion_tokens`
  for gpt-5, `max_tokens` otherwise, `Gemini.max_output_tokens`). The four refinement agents pass
  `clamp(1.6 × estimated input tokens, 4096, 16384)`; everything else passes nothing (today's behaviour).
- Add to the `enhance`, `grammar`, `tone`, `simplify` prompts: same number of paragraphs, same headings, same
  tables/quotes/images/links, same length; never summarise or drop a section.
- Pin `markdown-it-py` in `requirements.txt`.
- Why ship this alone: it removes the Claude 8,192-token truncation for mid-length documents today and costs nothing;
  it is not the fix for 5,000 words, Phase 2 is.

### Phase 1 — Longform core, no LLM *(M)*

Files: `src/longform/markdown_ast.py`, `freeze.py`, `chunk.py`, `verify.py`, `ordered_stream.py`;
`tests/longform/`; `tests/fixtures/longform/*.md`.

- Fixtures: the acceptance document (5,000 words, 4 tables, 6 blockquotes, 5 images, 10 links, 2 code fences),
  plus edge cases: pipes inside table cells, `>` inside fences, nested quotes, an inline image mid-sentence,
  reference-style links, a 2,000-word single paragraph, CRLF input, a document that is only a table.
- Property: `thaw(join(chunks(freeze(doc))))` is byte-identical to `doc` for every fixture.
- `verify.py` tested with scripted bad outputs: dropped token, duplicated token, reordered tokens, 40% length,
  merged paragraphs, added table, changed heading level, changed list count, altered URL.

### Phase 2 — Refinement pipeline *(M)*

Files: `src/longform/refine.py`, `prompts.py`; `src/text_processing_agent.py`; `api.py`; `config.py`; docs.

- `refine_stream()` with an injectable `LLM` callable (same signature as `email_template.service.LLM`, plus a
  streaming variant) and the candidate/fallback behaviour of §3.5.
- Hook (§3.8), API passthrough, `refine_progress` event, `warnings` on `done` and on `/agent/process`.
- Tests: FakeLLM identity → output equals input; FakeLLM that summarises → chunk restored and warned; event order on
  `/agent/stream` through `TestClient`; the four operations' ratio windows; `qualifies()` boundaries.

### Phase 3 — Blog plan and outline *(M)*

Files: `src/blog/plan.py`, `schema.py`, `prompts.py`, `outline.py`; tests.

- Plan rules (§4.3) with tests for every row, including "no image provider → 0 images, lint does not require them".
- Outline schema, validation, repair, rule-based fill; a few-shot example file per length (`examples/`), checked by
  a test that the example contains no fact absent from its input (the email rule).

### Phase 4 — Drafting, media, lint, API *(L)*

Files: `src/blog/media.py`, `draft.py`, `validate.py`, `service.py`; `api.py`; `src/task_router.py`;
`src/text_processing_agent.py` (`blog_writer`); `config.py`; docs.

- `generate_blog` end to end behind `BLOG_AI_ENABLED`; `outline` event; images started after the outline; ordered
  section emission; lint with per-unit repair; warnings.
- Tests with FakeLLM: every generated post has ≥ 1 table, ≥ 2 quotes, images placed where the outline put them;
  `pending_images` entries have the direct-route shape; the stream event sequence; a FakeLLM that writes an
  attributed quote without web access gets it de-attributed; a unit that never produces its table ends in a warning,
  not a failure; credits reserved for images on non-BYOK and not on BYOK.

### Phase 5 — Observability, evaluation, benchmarks *(M)*

Files: `src/observability.py`, `scripts/evaluate_longform.py`, `scripts/evaluate_blog.py`, `benchmarks/`.

- Trace fields from §3.6 and §4.7; `latency_stats()` grouped by mode.
- `evaluate_longform.py --live`: 10 real documents (800–5,000 words, mixed structure) × 4 operations × providers.
  Records structural loss (must be 0), word ratio, chunks repaired / restored, p50/p95 latency, tokens, cost.
- `evaluate_blog.py --live`: 30 topics × providers. Records tables / quotes / images per post, longest prose run,
  repairs, latency, cost; writes side-by-side Markdown files for manual review, as `evaluate_code.py` does.
- Targets: Mode 1 — 0 structural loss on every run; word ratio 0.9–1.1 for `enhance`/`grammar`; restored chunks
  ≤ 5%; p50 ≤ 45 s for 5,000 words. Mode 2 — 100% of posts meet the quotas when an image provider is available;
  longest prose run ≤ 250 words in ≥ 95%; p50 ≤ 40 s for 1,400 words with 3 images; first `content` ≤ 12 s.

### Phase 6 — Rollout *(S)*

- `LONGFORM_REFINE_ENABLED=true` in staging → a week of logs → production.
- `BLOG_AI_ENABLED=true` once the frontend ships the `generate_blog` action.
- `BLOG_INTENT_SHADOW_MODE=true` for a week → review → `BLOG_INTENT_REGEX_ENABLED=true`.

### Phase 7 — Later *(not scheduled)*

Table cell polishing; `translate` / `make_shorter` / `make_longer` on the chunked path; live first-chunk streaming
with `content_replace`; stock photo backend; Mermaid diagrams; HTML input.

---

## 8. Rollout order

1. Phase 0 — now.
2. Phases 1 + 2 with `LONGFORM_REFINE_ENABLED=false` — code live, behaviour unchanged.
3. Phase 5 longform half → turn Mode 1 on in staging.
4. Phases 3 + 4 with `BLOG_AI_ENABLED=false`.
5. Phase 5 blog half → frontend action → turn Mode 2 on.
6. Phase 6 shadow → regex promotion.

---

## 9. Frontend handoff (Eddyter)

- **Refinement**: nothing required. Optional: show `refine_progress` ("Refining section 2 of 5") and surface
  `done.warnings` ("One passage was kept as written because the AI changed its structure").
- **Blog**: new toolbar action → `type: "generate_blog"`, `query` = topic/brief, `secondary_prompt` = style notes,
  optional `blog_options`. Optional `outline` event to render a skeleton before the first section arrives.
- Rendering: content is GFM — confirm the editor's Markdown import handles tables, blockquotes and an image
  followed by an italic caption line (O2). Images arrive through the existing `image_ready` swap.
- **Please confirm O1**: the exact serialisation the editor sends in `query` for a selection containing a table or an
  image (Markdown from Lexical's export, or HTML). Both are handled, but the converter is only added if needed.
- Later: `content_replace` support unlocks live streaming of the first chunk (O3).

## 10. Backend handoff (`/agent/process`)

- Accepts `generate_blog` and `blog_options`; returns `warnings: []` in addition to today's fields.
- A 5,000-word refinement or a blog with images can take over a minute: keep the client timeout above 3 minutes
  (the doc already asks for 2 minutes because of image waits).
- Billing unchanged: text per character, images through the existing image details.

---

## 11. Open items

| # | Item | Owner | Blocks |
|---|---|---|---|
| O1 | What `query` contains for a selection with tables/images (Markdown vs HTML) | Frontend | Whether `markdownify` is added in Phase 2 |
| O2 | Lexical import of GFM tables / blockquotes / image + caption line | Frontend | Mode 2 rendering, Mode 1 only if tables were not already round-tripping |
| O3 | `content_replace` support in the editor | Frontend | `LONGFORM_STREAM_LIVE` |
| O4 | Stock photo provider (Pexels/Unsplash): keys, attribution requirements, caching | Product | `BLOG_IMAGE_SOURCE=stock` |
| O5 | Mermaid rendering in the editor | Frontend | Diagram directives |
| O6 | Credits price for a blog (outline + n sections + images in one request) | Product | Nothing — current prices apply; Phase 5 gives the data |
| O7 | Per-provider output-token ceilings for the current advanced IDs (verify against live model cards) | Dev | Phase 0 clamp values |

## 12. Risks and how the plan handles them

- **Model ignores tokens or shortens anyway** → verify + repair + restore; the failure is a warning, never a loss.
- **Parallel chunks drift in tone** → same system prompt, same instruction, context lines; `tone` requests get the
  tone in every chunk; Phase 5 side-by-side review checks seams.
- **Rate limits with 3 parallel calls per request** → `LONGFORM_PARALLEL`/`BLOG_SECTION_PARALLEL` are runtime knobs;
  per-chunk fallback to the next provider uses the existing `can_fall_back` rules.
- **Image cost per blog** → count capped by `IMAGE_MAX_PER_REQUEST`, reserved up front, 0 when no provider.
- **Collision with in-flight work** → the uncommitted provider-fallback change in `process_user_query_stream` and
  `src/provider_errors.py` lands first; the hooks here sit before that loop and reuse its helpers.

---

## 13. Implementation status

### Mode 2 — built

| Plan item | Where |
|---|---|
| Phase 0: `get_model(max_tokens=...)`, per-provider limit field, cache key, reasoning headroom | `src/text_processing_agent.py` (`get_model`), `src/models.py` (`output_token_budget`), `tests/core/test_output_limits.py` |
| Shared `ordered_stream.py` (parallel, in-order, fail-fast, cancel on close, progress sentinel) | `src/longform/ordered_stream.py` |
| Shared Markdown block parser (`markdown_ast.py`), `markdown-it-py` pinned | `src/longform/markdown_ast.py`, `requirements.txt` |
| Phase 3: plan (§4.3), outline schema / validation / one repair / rule-based fill (§4.4), per-length few-shot examples | `src/blog/plan.py`, `schema.py`, `outline.py`, `prompts.py`, `examples/` |
| Phase 4: media (§4.5), drafting (§4.6), lint (§4.7), service, `blog_writer` agent, `generate_blog` type, `blog_options`, `outline` event, `warnings` in `done` and in the process response, 503 gate, `WRITING_REQUEST_TYPES` | `src/blog/media.py`, `draft.py`, `validate.py`, `service.py`, `runtime.py`, `llm.py`; `api.py`; `src/task_router.py`; `src/text_processing_agent.py` |
| `general` promotion (D9): start-anchored regex, flag + shadow mode | `src/blog/intent.py`, hook in `process_user_query_stream` |
| Phase 5, blog half: trace fields, `latency_stats()` grouped by mode, `evaluate_blog.py` (offline structural check, `--live`) | `src/observability.py`, `scripts/evaluate_blog.py` |
| Review-only preview: saved posts as one self-contained HTML page, two generations side by side, page rhythm and counts, unrendered tables flagged (not the editor's renderer, see O2) | `scripts/render_blog_preview.py`, `tests/blog/test_render_blog_preview.py` |
| Docs / handoff (§9, §10) | `docs/API_DOCUMENTATION.md`, `README.md` |

### Where the implementation goes beyond or differs from the plan

1. **Extra modules** in `src/blog/`: `llm.py` (the injectable `LLM` type and the agno-backed `AgnoLLM` with candidates and
   fallback), `sources.py` (which pages were read), `text.py` (making model strings safe to write into Markdown) and
   `runtime.py` (the composition root binding the pipeline to this app). `service.py` imports nothing from the app.
2. **Word-count clamp**: a brief can name any length, so the plan clamps to 300-4,000 words and says so in `plan.reasons`.
3. **"No images" is understood**: "without images", "no pictures", "text-only" in the brief mean zero, which §4.3 left implicit.
   When the user explicitly asked for more pictures than the post gets (cap, or no usable provider), `done.warnings` says so.
4. **Outline token budget** is `max(2000, 400 + 330 x sections)` rather than a flat 2,000: a 7-section outline with picture
   prompts can exceed 2,000 tokens, and a cut-off outline is invalid JSON.
5. **Output budget**: `max_tokens` is the visible output wanted. It is rounded up to a multiple of 1,024 (so per-request
   budgets share a bounded set of cached models) and models whose limit also counts hidden reasoning (gpt-5 / o-series /
   Grok 4 reasoning) get 4,096 extra. `OUTPUT_TOKEN_CEILING` (16,384) and that headroom are **not yet verified against live
   model cards (O7)**.
6. **Lint scope**: only the §4.7 checks trigger a repair. Paragraph length and list count are recorded in the trace as `notes`,
   not sent to the client. Every heading line the model writes is stripped (code owns all headings).
7. **D13 is enforced by a record, not a regex on the answer**: the page reader reports each page it really read
   (`create_web_read_tool(on_read=...)`); an attribution or a URL survives only if its page is in that record.
8. **Pictures start when the stream moves past the `outline` event**, not before it is yielded, so a client that disconnects
   right after the outline never starts (or pays for) a picture.
9. **`create_agent` serves the blog writer** with per-call `instructions`, `temperature` and `max_tokens`; `markdown` is off
   for it (the outline is JSON) and the date is added so research can resolve relative dates.
10. **`process_user_query_with_images(meta=...)`** returns the warnings through an optional out-parameter, so the existing
    `(text, pending_images)` return value and its callers are untouched.
11. **When a part fails after pictures started**, the stream hook still hands the pending-images dict to the caller before the
    error, so the endpoint can release the credits reserved for them.

### Found by running it live, and what changed

12. **Outline latency (`BLOG_OUTLINE_MODEL`).** The first live post reached first content at ~14 s against the 12 s target; the
    outline call is the cause. A probe showed it is pure generation speed (~950 output tokens at ~75 tok/s for gpt-5.2): lowering
    reasoning effort does nothing (`none` = same, `low` slower, `minimal` unsupported), a compact JSON example does nothing, and
    OpenAI's own lighter tier (gpt-4o-mini) is barely faster. A faster *provider* is what helps, so the outline's first attempt can
    run on a configured model (`BLOG_OUTLINE_MODEL=provider[:tier]`, unset by default) while the repair and every part stay on the
    drafting model. It only uses providers the request may use (BYOK: own keys), falls back to the drafting providers, and billing
    and `model_used` follow the model that writes the post. Measured on live posts: first content **13.0 s -> 4.2 s** (median of 3),
    total 29.8 s -> 22.3 s; details, the probe and the tail analysis are in `docs/LATENCY_BENCHMARK.md` (Phase 6).
    Lossless outline faults (`visual_gap`, quote count, section lengths, duplicates) are now fixed by `finalize()` without the repair
    call §4.4 prescribes: 3 of 60 fast-model outlines paid a 6.5 s repair for something code does for free.
13. **A miscounted table delimiter row** (4-column header, `|---|---|---|`) leaves a paragraph of pipes in the editor. `sanitize`
    now rebuilds the delimiter row. The lint only ever looked at *assigned* tables, which is how it slipped through.
14. **Unassigned visuals.** Even with a prompt rule, gpt-5.2 added tables and quotes it was not assigned (3-7 extra quotes and 1-3
    extra tables per post). Naming what is *not* assigned in the part's request (`not_assigned`) fixed it at the source. As a safety
    net, extra block quotes are dropped by code, but only when that adds no lint error (a quote is also a visual break and part of
    the word count: trimming blindly caused up to 4 part repairs in one post). Extra tables are never removed (they carry content).

15. **First text on GPT (`BLOG_STREAM_TITLE`, default on).** The whole 13-15 s to first content was the outline call (gpt-5.2: ~960 output
    tokens at ~74 tok/s, no reasoning tokens in 18 runs, ~0.01 s of anything else). The outline is now streamed, and the title and
    subtitle open the post the moment both are written: **1.5-2.3 s** on live gpt-5.2 posts (median 1.6 s) against 13-15 s, with the
    outline, the model and the post itself unchanged. `TitleWatcher` reads only the top level of the JSON, applies the same cleaning and
    limits as `normalise`, reports once, and `build_outline` makes the finished outline carry exactly what was shown (a repair or a
    provider fallback cannot change it). The stream order is now `start`, `content` (title), `outline`, `content`...; pictures still
    start only after the `outline` event has been consumed (8) and the pictures record is handed over at once through a private
    marker, because the next text chunk is now a whole part away (11). The first *paragraph* still waits for the whole outline plus
    the intro call (~18 s on gpt-5.2): that needs a faster outline call, not a streamed one (`docs/LATENCY_BENCHMARK.md`, Phase 7).
16. **Standard length is 1,450 words** (plan: 1,400), and writers are asked for 4% less than a part's target
    (`draft.WRITER_LENGTH_FACTOR`): 7 live posts planned at 1,450 came to 1,512 words on average (sections +3.6%, intro +12%).
    With the factor 8 live posts came to 1,470 on average (1,339-1,542) and every one stayed within 10% of the plan.
17. **A post was lost to a 301-character angle.** gpt-5.2 wrote section angles up to 335 characters against a 300 cap, and its repair
    did not shorten them (a model cannot count characters). An angle (and the intro hook) is guidance the reader never sees, so the
    cap is now 500 and anything longer is cut by code at a sentence (`outline.trim_guidance`); text the reader sees is still the model's
    to shorten.

### Mode 1 — built

| Plan item | Where |
|---|---|
| Phase 0: the four refinement prompts say what must not move (structure, length) and each call states its output budget (1.6 x the input, 4,096-16,384) | `src/text_processing_agent.py` (`REFINE_STRUCTURE_RULE`, `REFINE_LENGTH_RULES`, `refinement_max_tokens`), `tests/core/test_output_limits.py` |
| Phase 1: normalise (§3.2), freeze / thaw (§3.3), chunk (§3.4) | `src/longform/markdown_ast.py`, `freeze.py`, `chunk.py` |
| Phase 2: prompts (§3.5), verify / repair / restore (§3.6), the pipeline, the runtime binding, the `LONGFORM_*` flags, `refine_progress`, `warnings`, the hook (§3.8) | `src/longform/prompts.py`, `verify.py`, `refine.py`, `runtime.py`, `llm.py`; `config.py`; `api.py`; `src/text_processing_agent.py` |
| Phase 5, longform half: trace (`mode=longform`, `longform={...}`, marks `refine_planned` / `first_chunk`), `latency_stats()` grouped by mode, `evaluate_longform.py` (offline: 10 documents x 4 operations with a well-behaved and a damaging scripted model; `--live`: real documents, spelling slips, an independent structure check) | `src/observability.py`, `scripts/evaluate_longform.py` |
| Local end-to-end scenarios | `scripts/local_test_requests.py` (enhance over `/agent/stream`, grammar over `/agent/process`, on this repository's README), `scripts/run_local_server.py` (flag on) |
| Docs / handoff (§9, §10) | `docs/API_DOCUMENTATION.md`, `README.md` |

Tests: `tests/longform/` (`test_freeze`, `test_chunk`, `test_verify`, `test_prompts`, `test_refine`, `test_runtime`, `test_api`,
`test_settings`, `test_evaluate_longform`). The model is a scripted stand-in; mutation checks on each rule (a missing token, a
merged paragraph, a table typed by the model, a summary, a wrapper) confirm every test fails when its rule is removed.

### Where Mode 1 goes beyond or differs from the plan

18. **Bare URLs are tokens (`[[URL_n]]`), not "left in place and verified by set equality".** A model that rewrites a sentence
    around a URL can drop a trailing `)` or lowercase the host; with a token it cannot touch the URL at all, and the check is
    the same token check as for everything else. A horizontal rule and any other block the parser has no name for are frozen
    as `[[HTML_n]]` for the same reason.
19. **`LLM`, `CallSpec` and `AgnoLLM` moved to `src/longform/llm.py`** (`src/blog/llm.py` re-exports them): both pipelines make
    the same kind of call (an agno agent per call, candidates and fallback, usage recorded), and `AgnoLLM(agent_type=...)` says
    which `AGENT_CONFIGS` entry a pipeline's calls are made with (`longform_refiner`: no tools, the operation's own prompt).
20. **The instruction follows the single call.** `secondary_prompt` is the instruction of `enhance_with_ai` and the tone of
    `change_tone` (every chunk gets it on its own `INSTRUCTION:` / `TONE:` line; `professional` when no tone was given, so
    chunks cannot choose their own); `fix_grammar` and `simplify` never used it and still do not.
21. **Phase 0 shipped without a flag**, as §7 says, and changes the single call too: the four agents now carry the structure and
    length rules, and a 100,000-character document is no longer cut at a provider default.
22. **`chunk_restored` warnings** are `{"kind": "chunk_restored", "chunk": n, "detail": ...}` (§3.6 sketched `{"chunk", "kind", "detail"}`).
23. **`/agent/process` does not strip a leading or trailing code fence from a document refined this way.** The endpoint removes
    ```` ```markdown ```` wrappers from single-call answers; applied to a document it would eat the document's own closing fence.
    Wrappers are already removed per chunk (`verify.unwrap`).
24. **A smaller first chunk (`LONGFORM_FIRST_CHUNK_WORDS`, default 300).** A chunk is shown only when verified (D5), so the first
    text arrives after the first call; making that call short moves the first text from the time of a 1,000-word call to the
    time of a 300-word one at the price of one more call per document. Off with `0`.
25. **HTML input, text that already looks like the tokens, and documents with reference images keep the single call**
    (`qualifies`; open item O1 is the HTML decision).

### Mode 1: found by running it live, and what changed

Measured on this repository's own documents with `scripts/evaluate_longform.py --live` (OpenAI, Gemini, Grok); numbers and tables in
`docs/LATENCY_BENCHMARK.md` (Phase 8). Against the Phase 5 targets: structural loss 0 (after 26-28), restored passages 0-1.6% (<= 5%), word ratio
0.9-1.1 for OpenAI, p50 for 4,000+ word documents 7.7-26.6 s (<= 45 s).

26. **The basic OpenAI tier is the slow one** (`gpt-4o-mini`: ~54 tokens/s, 24-29 s per 1,000-word passage; `gpt-5.2`: ~96 tokens/s, 14.7 s). §3.9's 10 s estimate held only
    for the advanced tier. So the defaults moved: **`LONGFORM_PARALLEL` 3 -> 5** (a 4,850-word `fix_grammar`: 43.7 s -> 29.0 s) and a **300-word first passage**
    (`LONGFORM_FIRST_CHUNK_WORDS`; first text 12-35 s -> 5-18 s). Gemini's basic tier does the same job in 2-9 s, Grok's in 4-12 s; Grok's advanced reasoning model needs 2-3 minutes.
27. **Whitespace around a locked block is not the model's.** The independent structure check caught a `change_tone` run in which two spaces after a table token (a Markdown hard
    break) came back as two spaces on the table's last row, a byte difference verification had not seen. `verify.snap_blocks` now restores every locked line's indent and the blank lines around
    it before checking (a dropped blank line would glue the next paragraph to a quote).
28. **A friendly tone adds paragraphs** ("Hey there! ..." under a heading). §3.6's paragraph tolerance let it through, so `verify` also compares the *sequence* of headings, lists,
    locked blocks and runs of paragraphs (`order`); the repair message names a lead-in paragraph. Gemini and Grok needed the repair on a list-heavy passage 8 times in 8 and it fixed all 8.
29. **Prompt wording matters more than the plan assumed.** Saying what a token stands for and that it moves with its sentence cut Grok's dropped tokens from 9 of 24 first answers to 0; telling
    the model a window tighter than the enforced one (`Operation.advertised`) cut Gemini's `enhance_with_ai` drift from +5.7% to +2.6% (90-110%) or +0.4% (95-105%) without more rule breaks.
30. **Inline HTML tags and comments, and email addresses, are tokens** (`[[HTML_n]]`, `[[URL_n]]`): a model that rewrote around `<kbd>Ctrl</kbd>` or lowercased an address changed bytes the plan's table had
    left to instructions.
31. **The safety net does its job on real models.** `gpt-4o-mini` answered some passages with a short summary (6 of 18 first answers on one passage): every one failed verification and the repair fixed it.
    Repaired passages: OpenAI 9%, Grok 11%, Gemini 18% (mostly `change_tone` growing a passage past +25%); restored: 0, 0 and 1.6%.
32. **A single call over the same 4,963 words took 135 s and changed a protected block**; the chunked path takes 25-45 s on that model (documents that contain the literal text of the
    tokens, such as this plan, keep the single call by design).
33. **A provider call can hang for ever**: a Gemini call stayed open and silent for minutes during the evaluation. The single call streams, so a stall shows; a chunk that is
    never answered holds the whole request, and every passage after it. Each call is now bounded (`LONGFORM_CALL_TIMEOUT_SECONDS`, 150): the timeout is an error the
    fallback rules already retry on the next provider (and the stuck provider is skipped by the passages that follow).
34. **A cold Gemini model failed parallel first calls** ("Cannot send a request, as the client has been closed"): agno builds a model's client on first use with a check-then-set, `get_model`
    stored its model after checking the cache outside the lock, and google-genai closes a Client's connection when it is garbage collected. 2 of 10 fresh processes with six parallel calls
    failed; after the fix (one model per key, its client built once under the lock) 0 of 26. The race was in shared code: the blog parts and the chat paths had the same window on a cold start.
35. **The Anthropic key in this environment returns 401**; provider fallback hid it behind OpenAI answers until the evaluation compared the model that served each run with the one asked for.
    It now leaves such runs out of the numbers, so Claude is unmeasured.

### Not built (and why)

- Mode 1: the later items of §3.10 (table-cell polishing, live streaming of the first chunk with `content_replace` (O3), `translate` and `make_shorter` on the chunked path), and HTML input: `markdownify` is not added and an HTML document keeps the single call (O1). The chunked path is built and measured for English Markdown only.
- Stock photo and Mermaid backends (O4, O5): `BLOG_IMAGE_SOURCE=stock` is accepted and means "no pictures" until built.
- Live measurements for the Phase 5 targets: Mode 1 is measured (`docs/LATENCY_BENCHMARK.md`, Phase 8; not for Claude, whose key is invalid here); Mode 2: run `python scripts/evaluate_blog.py --live`, the targets are printed with the results.
- Frontend items O1-O3, O6, and verification of O7.
