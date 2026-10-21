# AI Email Templating — Implementation Plan

Status: Phase 1 backend implemented (src/email_template, /agent/email/generate, /agent/email/image fit), plus launch/feature-update types, optional sign-off, social links and user themes (§3.8) · Owner: TBD · Scope: Eddyter editor + this agent service

## 1. Verdict on the approach

The base idea is right: **keep the format fixed, let AI produce the content**. Official emails need to look
consistent, render reliably in Outlook/Gmail, and never contain made-up facts. A fixed format makes all three
things checkable.

Four refinements to the base:

| # | Refinement | Why |
|---|---|---|
| 1 | AI returns **JSON blocks**, never HTML. Our renderer produces the HTML. | Email HTML is fragile; the model is bad at it and we can't validate it cheaply. |
| 2 | Render JSON → **MJML** → HTML instead of hand-writing table markup. | MJML already handles Outlook/Gmail quirks (buttons, columns, mobile). We test the renderer once, not every AI output. |
| 3 | Unknown facts become **fill-in slots** (`[[fill: deadline date]]`), not guesses. | Official emails must not contain invented dates, names, prices or reference numbers. |
| 4 | Edits are **patches to blocks**, not full regenerations. | Users keep what they like; edits are faster, cheaper and predictable. |
| 5 | **Category is auto-detected** when the user doesn't pick one: UI selection → app-side regex → `general` fallback (§3.5). | Users can just describe the email; routing stays deterministic and testable. |
| 6 | Every generation returns **3 subject-line options + 1 preheader**. | The subject decides whether an official email gets opened at all. |
| 7 | **One call to action** per email. | Competing buttons dilute the action an official email asks for. |
| 8 | Categories live in a **registry** (config + one few-shot example each), which grows from usage logs (§3.6). | New email types need no engine or prompt rewrites. |

Keep the current `generate_email_template` agent (raw HTML from a reference image, `src/text_processing_agent.py:1599`)
as a **legacy/"copy a design" path** until Phase 3 replaces it with image → JSON.

Flexibility model: **hybrid**. Each email *category* (notice, invitation, reminder…) has a recommended skeleton;
the AI may add, drop or reorder blocks within that category's allowed set. Users always have final say in the editor.

---

## 2. Architecture

```
Eddyter (frontend)                         Agent service (this repo)
──────────────────                         ─────────────────────────
Brief form / block toolbar  ──request──▶  /agent/email/*  routes (api.py)
                                            │
                                            ├─ prompt builder   (category + tone + brand + inputs)
                                            ├─ LLM call         (structured JSON output)
                                            ├─ validator        (schema → semantics → tone lint)
                                            ├─ repair pass      (≤1 retry, "fix only these issues")
                                            └─ returns template JSON + warnings
                                                     │
Block editor  ◀─────────── template JSON ────────────┘
   │
   └─ renderer: JSON → MJML → HTML (+ plain-text)  → preview iframe / export / send
```

Ownership:
- **Agent service**: generation, validation, repair, tone rules. It never renders HTML.
- **Eddyter**: block editor, renderer, preview, export. The template JSON is the saved source of truth; HTML is a
  derived artifact (cache it, never edit it directly).

Assumption to confirm: Eddyter's editor (Lexical, currently using `HtmlCanvasNode` for raw HTML) can host a
block-based email node. If not, the renderer output can still go into `HtmlCanvasNode` for preview, with block
editing in a side panel.

---

## 3. Template format (schema v1)

### 3.1 Top level

```json
{
  "schema_version": 1,
  "meta": {
    "category": "policy_notice",
    "language": "en-GB",
    "tone": "formal",
    "routing": "selected",
    "subject_options": [
      "Update to our remote working policy",
      "Remote working policy: what is changing",
      "Changes to remote working arrangements"
    ],
    "subject": "Update to our remote working policy",
    "preheader": "Changes take effect from [[fill: effective date]]"
  },
  "theme": { "brand_kit_id": "bk_123" },
  "blocks": [
    { "id": "b1", "type": "header", "logo_asset": "asset:logo_primary", "org_name": "{{company_name}}" },
    { "id": "b2", "type": "greeting", "text": "Dear {{first_name}}," },
    { "id": "b3", "type": "paragraph", "text": "We are writing to inform you of an update to our remote working policy, effective from [[fill: effective date]]." },
    { "id": "b4", "type": "key_value", "rows": [
        { "label": "Policy", "value": "Remote Working Policy v3" },
        { "label": "Effective date", "value": "[[fill: effective date]]" }
    ]},
    { "id": "b5", "type": "list", "style": "bulleted", "items": [
        "Employees may work remotely up to three days per week.",
        "Requests must be approved by a line manager in advance."
    ]},
    { "id": "b6", "type": "button", "label": "Read the full policy", "link": "link:policy_document" },
    { "id": "b7", "type": "paragraph", "text": "If you have any questions, please contact the HR team." },
    { "id": "b8", "type": "signature", "closing": "Kind regards,", "name": "{{sender_name}}", "title": "{{sender_title}}", "org": "{{company_name}}" },
    { "id": "b9", "type": "footer", "address": "{{company_address}}", "legal": "This message is intended for employees of {{company_name}}." }
  ]
}
```

### 3.2 Three kinds of placeholder (they must never be confused)

| Syntax | Meaning | Who fills it | Validation |
|---|---|---|---|
| `{{first_name}}` | Merge tag — per-recipient, filled at send time | Sending platform | Must be in the request's `allowed_merge_tags`; unknown tags rejected |
| `link:policy_document` | Link reference — resolves to a URL the user supplied | Renderer | Must be a key in the request's `links`; raw URLs from the model rejected |
| `asset:logo_primary` | Asset reference — logo/image from the asset library | Renderer | Must exist in the brand kit / asset list |
| `[[fill: effective date]]` | Missing fact the AI could not know | Author, in editor | Highlighted in editor; **export/send blocked** until none remain |

### 3.3 Block types (v1)

| Type | Fields | Limits / rules |
|---|---|---|
| `header` | `logo_asset?`, `org_name?` | Exactly one, first block |
| `greeting` | `text` | ≤ 1; formal salutations only |
| `heading` | `text`, `level` (1–2) | ≤ 80 chars, no trailing `!` |
| `paragraph` | `text` (inline: `**bold**`, `*italic*`, `[text](link:key)`) | ≤ 600 chars; nothing else is parsed — all other markup is escaped |
| `list` | `style` (bulleted/numbered), `items[]` | 2–8 items, ≤ 200 chars each |
| `key_value` | `rows[{label, value}]` | 1–10 rows; for reference numbers, dates, venues, amounts |
| `notice` | `variant` (info/important), `text` | ≤ 1 per email; replaces "URGENT!!" style copy |
| `button` | `label`, `link` | ≤ 25 chars; verb-first label; **at most one per email** (the single CTA); categories with an action require it. Secondary links go inline in a paragraph |
| `image` | `slot` (`hero`/`inline`), `source` (`asset`/`refined`/`generated`/`null`), `asset?`, `generation_prompt?`, `alt` | Size, ratio and position are fixed by `slot` (§3.7), never chosen by the AI. Only added when the brief asks for or supplies an image. An empty slot (`source: null`) must be filled or removed before export. The AI never invents image files or URLs |
| `divider` / `spacer` | `size?` | Purely visual |
| `signature` | `closing`, `name`, `title?`, `org?`, `contact?` | Required for letter-style categories |
| `footer` | `address?`, `legal?`, `unsubscribe?` (bool) | Exactly one, last block; `unsubscribe` required for bulk categories |

Global limits: 3–20 blocks; exactly 3 `subject_options`, each ≤ 50 chars (hard 60); preheader 40–90 chars.
`meta.subject` defaults to the first option; the user picks another in the editor.
Later (v2): `two_column`, `table`, `event_details` (date/time/venue + calendar link), `quote`.

### 3.4 Categories (official use)

Each category = recommended skeleton + required/allowed blocks + tone default.

| Category | Skeleton | Required | Default tone |
|---|---|---|---|
| `announcement` | header, heading, paragraph×2, button?, signature, footer | header, footer | professional |
| `policy_notice` | header, greeting, paragraph, key_value, list, button?, paragraph, signature, footer | key_value, signature | formal |
| `invitation` | header, heading, greeting, paragraph, key_value (date/time/venue), button (RSVP), signature, footer | key_value, button | professional_warm |
| `meeting_request` | header, greeting, paragraph, key_value, paragraph, signature, footer | key_value, signature | professional |
| `reminder_deadline` | header, greeting, notice, paragraph, key_value, button, signature, footer | key_value | formal |
| `onboarding_welcome` | header, heading, greeting, paragraph, list (next steps), button, signature, footer | list | professional_warm |
| `newsletter` | header, heading, (heading+paragraph+button?)×n, footer | footer w/ unsubscribe | professional |
| `confirmation` (order/registration) | header, heading, paragraph, key_value, button?, footer | key_value | professional |
| `incident_apology` | header, greeting, paragraph (what happened), list (what we're doing), paragraph, signature, footer | signature | formal |
| `follow_up` / `thank_you` | header, greeting, paragraph×2, signature, footer | signature | professional_warm |
| `general` (fallback) | AI chooses from all v1 blocks | header, footer | professional |

Start Phase 1 with 4 + fallback: `announcement`, `policy_notice`, `invitation`, `reminder_deadline`, `general`.

### 3.5 Category routing (app-side, deterministic)

The model never guesses the category. It receives a resolved category and builds the email for it.

Precedence:
1. **Selected** — the user picked a category in the UI → use it, skip regex. `routing: "selected"`
2. **Matched** — regex over the user's brief picks a category. `routing: "matched"`
3. **General** — no match → `general` category: same schema and renderer, the AI picks blocks from the full
   allowed set. `routing: "general"`

The editor shows the resolved category so the user can change it before (or after) generating.

Patterns (case-insensitive, `src/email_template/routing.py`). Phrases, not single generic words —
words like `update`, `order`, `confirm`, `event` misroute ordinary sentences ("in order to…", "update staff on…"):

| Category | Pattern |
|---|---|
| `incident_apology` | `\b(apolog(y\|ies\|ise\|ize)\|sorry\|outage\|downtime\|disruption\|incident\|service (issue\|interruption))\b` |
| `confirmation` | `\border (#\|no\.\|number\|confirmation)\|\byour order\b\|\breceipt\b\|\binvoice\b\|\bbooking confirmation\b\|\bregistration confirmation\b\|\bpayment (received\|confirmation)\b\|\bpassword reset\b` |
| `reminder_deadline` | `\b(reminder\|remind\|deadline\|due (date\|by\|on)\|last date\|expir(es\|y\|ing)\|overdue)\b` |
| `policy_notice` | `\b(polic(y\|ies)\|guidelines?\|compliance\|regulations?\|terms (of service\|and conditions)\|code of conduct\|procedures?)\b` |
| `meeting_request` | `\b(meeting\|schedule a (call\|meeting)\|book a (call\|slot)\|your availability\|calendar invite)\b` |
| `follow_up` | `\b(follow[- ]?up\|following up\|thank you for\|thanks for\|feedback)\b` |
| `invitation` | `\b(invit(e\|ed\|ation\|ing)\|rsvp\|save the date\|join us (for\|at)\|webinar\|ceremony\|gala\|seminar\|conference)\b` |
| `onboarding_welcome` | `\b(welcome\|onboard(ing)?\|first day\|new (employee\|joiner\|member\|customer)s?)\b` |
| `announcement` | `\b(announc(e\|es\|ement\|ing)\|pleased to (share\|inform)\|relocat(e\|es\|ion\|ing)\|office (move\|closure)\|will be closed\|new (office\|branch\|appointment\|campus))\b` |
| `newsletter` | `\b(newsletter\|digest\|roundup\|bulletin\|(monthly\|quarterly\|weekly) update)\b` |

Algorithm:
- Score each category = number of **distinct** matched phrases (repeats don't count).
- Highest score wins. Ties break by the table order above (urgent and sensitive first, so
  "Sorry, your order is delayed" becomes `incident_apology`, never `confirmation`).
- Zero hits → `general`.
- Only categories registered and enabled for the tenant take part.
- Send the model only the resolved category, never the pattern table.

The patterns were checked against 14 sample briefs (all routed correctly); those become the seed of
`test_email_routing.py`. Every pattern change must keep that file green, and new misroutes found in production are
added to it as cases.

### 3.6 Category registry and growth loop

- A category is **data, not code**: an entry in `categories.py` (skeleton, required/allowed blocks, default tone,
  footer rules, routing pattern) plus **one few-shot example** file. Adding a category needs no engine change
  and no rewrite of the shared prompt.
- Few-shot examples are written by a person, and every fact in an example's output must appear in its input or be
  a `[[fill: …]]` slot. The model copies its examples more closely than its rules, so an example that adds a
  claim teaches the model to invent claims. A unit test checks each example against this rule.
- Every `routing: "general"` generation is logged with the brief (subject to the tenant's data settings).
  Review weekly; an intent that recurs (≥ 5 times) is promoted to a registered category with its own example.
  Patterns that misfire more than they match are pruned.

### 3.7 Images — fixed slots, fixed sizes

Email has no official image standard, but guides agree on a **600px body width** and on supplying images at
**2× display size** so they look sharp on high-resolution screens, with explicit `width`/`height` attributes
because Outlook ignores CSS sizing. A hero banner is commonly 2:1 (600×300) and a slim header 3:1 (600×200).
Sizes for files vary widely between sources; we take a conservative target.

**Slot table (the only image sizes an email can contain):**

| Slot | Position (fixed) | Ratio | Displayed | File delivered | Max per email |
|---|---|---|---|---|---|
| `logo` | Inside `header` block | any (fit, never cropped) | ≤ 180 × 60 | 2× of displayed, PNG | 1 (brand kit only, never AI-generated) |
| `hero` | Directly after `header`, before the first text block | **2:1** | 600 × 300 | **1200 × 600** | 1 |
| `inline` | After the main body text, before the button | **16:9** | 560 × 315 | **1120 × 630** | 1 |

Why these two ratios: 2:1 is the common hero ratio, and both are cheap to produce from what our generators output
(`1:1`, `16:9`, `9:16`, `4:3`, `3:4` — `_VALID_ASPECTS` in `src/image_gen.py`). 16:9 needs only a resize; 2:1 is a
16:9 image with ~11% of the height cropped. A 3:1 banner was rejected: cropping it from 16:9 throws away ~41% of
the picture.

Each category says which slots it allows, e.g. `invitation`, `announcement`, `newsletter`, `onboarding_welcome`
→ hero + inline; `policy_notice`, `reminder_deadline` → inline only; `confirmation`, `incident_apology` → none.
The AI cannot place an image anywhere else; the renderer puts each slot at its fixed position whatever the block
order says.

**File rules (applied by code to every image, whatever its source):**
- Resized and cropped to the exact slot size above; centre crop by default, or the user's focal point.
- JPEG (quality ~80) for photos, PNG only when transparency is needed; metadata stripped.
- Target ≤ 100 KB, hard limit 200 KB (lower quality step by step until it fits).
- Stored through the existing S3/CDN upload path (`src/s3_service.py`); the template stores an `asset:` id, the
  renderer resolves the URL.
- Rendered as `<img width="600" height="300" alt="…" style="display:block;width:100%;max-width:600px;height:auto;">`
  (inline slot: 560/315), so Outlook and mobile both size it correctly.
- `alt` is required; essential information (dates, prices, venue) must also appear as text, because many clients
  block images by default.

**Three ways an image gets into a slot:**

1. **Select / upload (`source: "asset"`)** — the user picks from the asset library or uploads a file. The editor
   shows a crop frame locked to the slot ratio; the server crops and compresses to the slot size.
2. **Upload and refine (`source: "refined"`)** — the user adds an image and asks for a change ("brighten it",
   "remove the background clutter", "make it look more corporate"). We call the existing edit path
   (`generate_image_unified(..., reference_image_bytes=…)`, which supports Gemini, OpenAI and Grok edits), then
   crop/compress to the slot. The original is kept so the user can undo.
3. **Generate from the brief (`source: "generated"`)** — the brief says e.g. "…and generate an image of our team
   at the annual meet". Flow:
   - Detect the request in code (reuse `extract_image_count` from `src/request_plan.py` on the user's own words),
     capped at the slots the category allows. No image request → the model may not add a generated image.
   - The template JSON comes back with the image block filled except the file: `slot`, `alt` and
     `generation_prompt` (written by the model).
   - After the JSON validates, the backend generates the image at the nearest native ratio (16:9 for both slots),
     crops to the slot size, compresses, uploads, and sets `asset`.
   - The email text is shown immediately; the image arrives a few seconds later (an SSE `image_ready` event, as
     `/agent/stream` already does for chat images). If generation fails, the slot stays empty with
     its `alt`/prompt kept, and the user can retry or pick an asset.
   - Billed with the existing per-image credit logic (`image_billing` in `api.py`).

**Generated-image prompt rules (added by code around the model's `generation_prompt`):**
- Professional, clean, business-appropriate style that fits the category's tone; brand colours from the brand kit.
- **No text, letters, numbers or logos in the image** — clients block images, crops cut text, and generators
  misspell. Text belongs in text blocks.
- Crop-safe composition: main subject centred, nothing important near the top/bottom edges.
- No recognisable real people or public figures; no third-party brand marks.
- The model's `generation_prompt` goes through the same guardrails as other image prompts.

**Implementation notes:**
- `generate_image_unified` currently uploads the provider's output as-is. Add an optional post-process step
  (crop + compress to a target size) that runs before upload, so we don't store and then replace a wrong-sized
  file. Pillow is already a dependency.
- New `src/email_template/images.py`: slot table, crop/compress, focal-point crop, validation that every
  delivered file matches its slot exactly.

### 3.8 Launch emails, sign-off, social links and themes (implemented)

**New email types:** `product_launch` (a new product) and `feature_update` (a new feature or a product
update). Both are routed automatically ("launching", "introducing our", "now available" → launch; "rolling
out", "new feature", "product update", "now supports", "what's new" → feature update). Both are bulk emails, so
unsubscribe is required.

**New blocks (launch types and newsletter only; letter-style types never get them):**

| Block | Fields | Rules |
|---|---|---|
| `hero_banner` | `eyebrow?` (≤ 20, e.g. "New"), `headline` (≤ 60), `summary?` (≤ 140), `button?` {label, link} | Directly after the header. Its button is the email's single call to action. The hero image slot sits right under it |
| `features` | `items`: 2–3 × {`title` ≤ 20, `text` ≤ 60} | One per email; each item must be a benefit the brief states. Columns stack on mobile |

**Sign-off is optional:** request `sign_off: "none"` ends the email at the footer (no "Kind regards"), even for
types that normally require a signature; `"signature"` requires one; omitted = the type's default.

**Signature OR social footer, never both:** an email ends either with a signature ("Kind regards, …") or with a
footer carrying social icons. Any `social_links` in the request mean no signature (even for policy notices and
reminders); a request with both `sign_off: "signature"` and `social_links` is rejected with a 400.

**Footer social icons:** request `social_links` {platform: URL}, at most 4, from linkedin, x, facebook,
instagram, youtube, email (`mailto:`). Code writes them into `footer.social` as `link:social_<platform>`
references in the given order; the model never writes or sees them. The renderer must use hosted PNG icons with
alt text (Gmail and Outlook do not show SVG) in a centred row.

**Theme is the user's choice, never the model's:** request `theme` = `accent_color` (hex) plus `hero` and
`footer` sections, each `{style: light | dark | brand | custom, background?, text?}` (`custom` needs a
background colour). It is copied into `template.theme` for the Eddyter renderer.

Rendering stays with the Eddyter team. `scripts/render_email_preview.py` is a review-only preview of the JSON.

**Frontend integration:** see `docs/EMAIL_API_HANDOFF.md` (endpoints incl. `GET /agent/email/categories`, template format, errors, 39 verified edge cases).

---

## 4. Official tone — rules the AI must follow

Tones offered: `formal`, `professional` (default), `professional_warm`. No casual/playful tone for this feature.

**Prompt rules (all tones)**
1. Plain, precise language; short sentences (target ≤ 25 words); active voice.
2. Formal salutation (`Dear {{first_name}},` / `Dear colleagues,`) and closing (`Kind regards,`, `Yours sincerely,`).
3. No emojis, slang, idioms, rhetorical questions or exclamation marks (max one in `professional_warm`).
4. No hype or marketing superlatives ("amazing", "revolutionary", "don't miss out") and no false urgency.
5. No ALL-CAPS words except acronyms.
6. **Never invent facts**: dates, times, amounts, names, reference numbers, policy details, URLs. Use
   `[[fill: …]]` for anything not in the brief.
7. No legal, medical or financial commitments beyond what the brief states.
8. Inclusive, neutral language; respect the requested locale's spelling (en-GB vs en-US) and date format.
9. Put the purpose in the first paragraph; one clear call to action.

**Deterministic tone lint (code, runs after every generation)** — `src/email_template/tone.py`
- emoji regex, `!` count, ALL-CAPS words not in an acronym allowlist
- banned-phrase list (hype, spam triggers: "act now", "100% free", "guaranteed"…), configurable per org
- sentence length > 35 words → warning
- salutation/closing present when the category requires a signature
- digits/dates/currency in output that don't appear in the brief → flag as possible invented fact

Lint errors trigger the repair pass; warnings are returned to the editor and shown to the user.

---

## 5. Backend implementation (this repo)

### 5.1 New module layout

```
src/email_template/
  __init__.py
  schema.py        # Pydantic models: Template, Meta, Block (discriminated union by "type"), limits
  categories.py    # category registry: skeleton, required/allowed blocks, default tone, footer rules, pattern
  routing.py       # selected → regex-matched → general (§3.5)
  examples/        # one few-shot example (input + output JSON) per category
  images.py        # image slot table, crop/compress to slot size, generate/refine orchestration (§3.7)
  tone.py          # tone profiles (prompt text) + deterministic lint
  prompts.py       # builds system/user prompts from category, tone, brand, inputs, schema
  validate.py      # parse → schema → semantic checks → lint; returns (template, errors, warnings)
  patch.py         # patch ops for refine: replace_block, insert_after, delete_block, update_meta, move_block
  service.py       # generate(), refine(), block_action(), subject_lines(), review()
```

Tests follow the repo's root-level convention: `test_email_schema.py`, `test_email_validate.py`,
`test_email_tone.py`, `test_email_patch.py`, `test_email_routing.py`, `test_email_examples.py` (examples add no facts), `test_email_images.py` (every slot
output is exactly its size and under the byte limit, for each source ratio), `test_email_service.py` (LLM mocked), plus an opt-in eval
`test_email_generate_eval.py` (real providers, see §8).

### 5.2 Structured output across providers

Supported providers today: openai, claude, grok, deepseek, gemini (`api.py:92`).

- **Implemented (Phase 1):** JSON requested in the prompt, then parsed and validated by our own code, with one repair pass. This behaves the same on all five providers. agno 2.8.5 does expose `output_schema` and `use_json_mode`; switch per provider later if the repair rate is high.
- Originally preferred: agno's structured-output support (`output_schema=<Pydantic model>` in agno 2.x — **verify the exact
  parameter on the pinned agno==2.8.5**), which maps to native JSON-schema / tool-calling per provider.
- Fallback for providers without strict schema support (likely DeepSeek, possibly Grok): JSON mode + schema
  embedded in the prompt + our own Pydantic validation.
- Either way **our validator is the gate**, not the provider.
- Create the agent with `markdown=False` (the shared factory sets `markdown=True`), `use_tools=False`,
  `use_advanced=True`, temperature 0.3 for generate, 0.2 for refine.
- Reuse `get_model()` and the existing BYOK/key resolution; don't build a parallel model path.

### 5.3 Validation pipeline (`validate.py`)

1. Strip code fences (all variants, including ```` ```json ````), parse JSON.
2. Pydantic schema: block types, enums, length limits, block count.
3. Semantic checks: merge tags ⊆ `allowed_merge_tags`; links ⊆ `links`; assets exist; category's required blocks
   present; header first, footer last; ≤ 1 button; exactly 3 subject options within limits;
   image blocks only when the brief mentions or supplies an image; unsubscribe present for bulk categories; unique block ids.
4. Sanitise text: strip any HTML tags, normalise whitespace, reject control characters.
5. Tone lint (§4).
6. Errors → **one** repair call: send the JSON + numbered error list, "fix only these, change nothing else".
   Still failing → return `422` with errors (generate) or drop the offending op (refine) — never return
   unvalidated output.

### 5.4 API routes (`api.py`)

All `POST`, same licence/credit/BYOK handling as `/agent/process`. Prefix `/agent/email/`.

| Route | Input (besides auth/model/api_key) | Output |
|---|---|---|
| `/agent/email/generate` | `category?` (omitted → routing §3.5), `brief`, `tone?` (defaults to the category's), `audience?`, `language`, `length` (short/standard/detailed), `brand_kit`, `links{}`, `allowed_merge_tags[]`, `assets[]`, `sender{}`, `variants` (1–3) | `templates[]` (each with 3 subject options + preheader), `resolved_category`, `routing`, `warnings[]`, `fill_slots[]` |
| `/agent/email/refine` | `template`, `instruction`, `target_block_ids?` | `ops[]` (patch), `template` (patched), `warnings[]` |
| `/agent/email/block-action` | `template`, `block_id`, `action` (rewrite/shorten/expand/more_formal/simplify/translate), `params?` | replacement block |
| `/agent/email/image` | `slot`, `action` (`fit` / `refine` / `generate`), `asset?` or upload, `instruction?`, `focal_point?` | `asset` id + URL at the exact slot size, `alt` suggestion |
| `/agent/email/subject-lines` | `template`, `count` (≤ 5) | `options[{subject, preheader}]` — for *more* options after the 3 returned by `/generate` |
| `/agent/email/review` | `template` | `issues[{block_id, severity, kind, message, suggestion}]` — tone, clarity, missing info, accessibility, spam risk |
| `/agent/email/translate` (Phase 2) | `template`, `target_language` | translated template (placeholders preserved) |

Keep `generate_email_template` on `/agent/stream` unchanged for the legacy HTML path.

Streaming: Phase 1 returns complete JSON (generation is short). Phase 3 can stream SSE `block` events as each
block validates, reusing the `/agent/stream` SSE pattern (`api.py:890`).

### 5.5 Credits & observability
- Bill via existing `fire_credit_deduction`; repair calls on platform keys are absorbed (not billed to the user).
- Add to `start_trace`/`trace_set`: category, tone, provider, block count, repair_needed, lint error kinds,
  fill-slot count. These become the quality dashboard (§8).
- Config kill switches in `config.py`: `EMAIL_AI_ENABLED`, `EMAIL_AI_REPAIR_ENABLED`, banned-phrase list path.

---

## 6. Frontend implementation (Eddyter)

1. **Shared schema**: generate TypeScript types from the Pydantic models (export JSON Schema → `json-schema-to-typescript`)
   so both sides agree. Commit the JSON Schema as the contract, versioned.
2. **Renderer** (`email-renderer` package): block → MJML component map, theme tokens from the brand kit
   (colours, font stack, logo, button style, footer defaults), → `mjml` → HTML. Also emits a plain-text version.
   Resolves `link:` / `asset:` refs; leaves `{{merge}}` tags intact; refuses to export while `[[fill:]]` slots remain.
3. **Brief form**: category picker → fields that category needs (e.g. invitation asks for date/venue so the AI
   doesn't produce fill slots for them), tone, language, links, sender.
4. **Block editor**: list of blocks with drag-reorder, inline text editing, per-block AI menu
   (Rewrite / Shorten / More formal / Translate), whole-email "Ask AI to change…" box → `/refine`.
5. **Preview**: rendered HTML in the existing sandboxed iframe (`HtmlCanvasNode` / `srcdoc`), desktop/mobile toggle.
6. **Review panel**: shows lint warnings, `/review` issues and remaining fill slots, each clickable to its block.
7. **Variants**: side-by-side compare of 2–3 generated variants; pick one to continue.
8. **Storage**: save template JSON (+ `schema_version`) and cached HTML; write a migration function per schema bump.

Renderer is tested once per block type in Litmus / Email on Acid (Outlook 2016/365, Gmail web/app, Apple Mail,
iOS) — AI output can't break layout because it never touches markup.

---

## 7. Prompt design (sketch)

System prompt (static per category+tone → good prefix caching, matching how other agents keep prompts byte-stable):
- Role: "You write official, professional emails for organisations."
- Tone profile text (§4) + category skeleton and allowed blocks.
- The JSON Schema and the category's few-shot example from `examples/` (§3.6).
- Placeholder rules (§3.2) with the exact lists of allowed merge tags, link keys and asset keys.

User message: the brief and structured inputs only. User free text is treated as content, never as instructions
that override the system rules (reuse `security_guardrails.py` style fencing for any pasted reference text).

Refine prompt: current template JSON + instruction + target ids → "return patch ops only".

---

## 8. Quality & evaluation

- **Golden set**: ~10 briefs per category (40 in Phase 1), including tricky ones: missing facts, very long briefs,
  non-English, briefs that ask for emojis/hype (must refuse politely by staying formal).
- **Automated metrics per provider**: schema-valid on first try, repair rate, final failure rate, lint pass rate,
  merge-tag/link preservation (100% required), invented-fact flags, latency, tokens.
- **Routing accuracy**: a labelled set of briefs (seeded from §3.5) scored on every pattern change; track the
  share of `general` routings in production as the signal for new categories.
- **Human rubric (1–5)**: officialness, clarity, correctness to brief, structure fit. Target ≥ 4 average before GA.
- Run the eval matrix on every prompt or schema change (like the existing `benchmark_*.json` before/after files).

---

## 9. Phased delivery

| Phase | Scope | Rough size |
|---|---|---|
| **0 — Decisions** | Confirm block set, 4 launch categories, renderer in Eddyter, brand-kit source, schema owner | 2–3 days |
| **1 — MVP** | `schema.py`, `categories.py` (4 + `general`), `routing.py`, few-shot examples, `tone.py`, `validate.py` + repair, `/generate` with 3 subject options; image slots with select/upload + crop (`fit`); renderer for v1 blocks; brief form; preview; manual block editing; golden-set eval | ~2–3 weeks |
| **2 — AI editing** | image generate-from-brief and upload-and-refine; `/refine` (patches), `/block-action`, `/subject-lines`, `/review`, plain-text output, remaining categories, translate | ~2 weeks |
| **3 — Advanced** | registry growth loop (log `general` routings, promote recurring intents), reference image → JSON (vision maps layout to blocks + theme), variants/A-B, org "house style" examples, streamed blocks, v2 blocks (columns, event details), retire legacy HTML agent | ~3+ weeks |

Estimates assume one backend and one frontend engineer and are indicative only.

---

## 10. Risks & open questions

| Risk / question | Mitigation |
|---|---|
| Providers differ in structured-output reliability | Validator + one repair pass; track per-provider failure rate; default model per tenant |
| Users want designs outside the block set | Keep legacy HTML path; grow block set from real requests |
| Fill slots annoy users | Brief form asks for the facts each category needs up front |
| Over-formal, robotic copy | `professional_warm` tone; few-shot examples written by a human editor |
| RTL languages (Arabic, Hebrew) | Renderer sets `dir`; include in golden set before enabling |
| Where is the brand kit stored? | Open — Eddyter account settings vs. this service; decide in Phase 0 |
| Who owns the schema? | Recommend: JSON Schema in this repo, versioned; Eddyter consumes generated types |
