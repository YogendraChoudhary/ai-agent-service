# AI Email Templates — Frontend Integration Handoff

For: the Eddyter frontend engineer integrating AI email generation.
Backend: this agent service (`src/email_template`, routes in `api.py`).
Verified: 2026-10-21. 322 automated tests pass, including `test_email_edge_cases.py`, which asserts every
edge case in §8. A live end-to-end request through the route (OpenAI) returned 200.

---

## 1. What you get, in one paragraph

You send a short **brief** ("We are launching X…") plus the links, merge tags and look the user chose. The
service picks the email type, has an AI write the email, checks it against strict rules, and returns
**template JSON**: an ordered list of typed blocks (header, hero banner, paragraphs, button, footer…).
**It never returns HTML.** Eddyter renders the JSON (recommended: JSON → MJML → HTML). Colours, fonts and
the dark/light look come from the `theme` the user picked, which the AI cannot change.

---

## 2. Architecture and security (read first)

```
Browser (Eddyter UI) ──► Eddyter backend ──► Agent service  /agent/email/*
                          adds provider key,      returns template JSON
                          bills credits
```

- **Do not call these routes from the browser.** `api_key` is the raw **AI provider key** (OpenAI,
  Anthropic…), and the routes have **no auth of their own**. They use the same server-to-server contract as
  `/agent/process`: Eddyter's backend resolves the key (user's own or platform key) and bills the call.
  CORS is currently open (`*`), so a browser call would technically work, and it would leak the key.
- Every response carries `X-Request-ID` (and `X-Response-Time-Ms`). Log it and show it in error reports.

---

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/agent/email/categories` | Email types to offer in the type picker |
| POST | `/agent/email/generate` | Generate 1–3 template variants from a brief |
| POST | `/agent/email/image` | Crop and compress an uploaded image to an email image slot; returns a CDN URL |

All error bodies are `{"error": "<readable message>"}`; a 422 adds `issues` (§7).

### 3.1 `GET /agent/email/categories`

```json
{"categories": [
  {"id": "reminder_deadline", "label": "Reminder / deadline", "default_tone": "formal",
   "image_slots": ["inline"], "needs_button": true, "needs_signature": true, "bulk": false},
  {"id": "product_launch", "label": "New product launch", "default_tone": "professional",
   "image_slots": ["hero", "inline"], "needs_button": true, "needs_signature": false, "bulk": true}
]}
```

Currently returns: `reminder_deadline`, `policy_notice`, `invitation`, `product_launch`, `feature_update`,
`announcement`, `general`. **Load it, don't hard-code it**: types are switched on server-side, and sending a
type that isn't listed is a 400. Add an **"Auto-detect"** option that omits `category`.

### 3.2 `POST /agent/email/generate`

#### Request

| Field | Type | Required | Rules |
|---|---|---|---|
| `model` | string | yes | `openai` \| `claude` \| `grok` \| `deepseek` \| `gemini` (case-insensitive) |
| `api_key` | string | yes | Provider key (added by Eddyter backend) |
| `brief` | string | yes | 3–5000 chars after trimming. What the email is about, in the user's words |
| `category` | string | no | An id from `/categories`. Omit or `""` = auto-detect |
| `tone` | string | no | `formal` \| `professional` \| `professional_warm`. Omit = the type's `default_tone` |
| `audience` | string | no | ≤ 300 chars, e.g. "All employees" |
| `language` | string | no | Default `en-GB`. Format `xx` or `xx-XX` (e.g. `en-US`, `hi-IN`) |
| `length` | string | no | `short` \| `standard` (default) \| `detailed` |
| `allowed_merge_tags` | string[] | no | ≤ 50. **Bare names**: `first_name`, not `{{first_name}}`. Lowercase, digits, `_` |
| `links` | object | no | `{key: url}`, ≤ 20. Keys lowercase/digits/`_`, must not start with `social_`. URLs must be `https://` |
| `assets` | object | no | `{key: description}`, ≤ 20, description ≤ 200. E.g. `{"logo_primary": "Company logo"}` |
| `sender` | object | no | Only `name`, `title`, `org`, `contact`, each ≤ 120 |
| `org_name` | string | no | ≤ 80 |
| `brand_kit_id` | string | no | ≤ 64, echoed into `template.theme.brand_kit_id` |
| `variants` | int | no | 1–3 (default 1). Variants are generated in parallel |
| `sign_off` | string | no | `signature` \| `none`. Omit = type default. See §5.3 |
| `social_links` | object | no | `{platform: url}`, ≤ 4, **in display order**. Platforms: `linkedin`, `x`, `facebook`, `instagram`, `youtube` (https), `email` (`mailto:`) |
| `theme` | object | no | §5.4. Chosen by the user, passed through unchanged |

**Unknown fields are rejected** (400), so typos like `socialLinks` fail loudly instead of being ignored.

Only link **keys** and asset **descriptions** reach the AI. URLs never do.

#### Example request

```json
{
  "model": "openai",
  "api_key": "<provider key, added server-side>",
  "brief": "We are launching Eddyter AI Assist, a writing assistant built into the editor. It drafts emails from a short brief, rewrites any paragraph in a chosen tone, and checks copy against your brand voice. Available to all workspace plans. Link to the launch page. Add a banner image showing the editor on a laptop.",
  "audience": "Existing customers",
  "allowed_merge_tags": ["first_name", "company_name", "company_address"],
  "links": {"launch_page": "https://eddyter.example.com/ai-assist"},
  "assets": {"logo_primary": "Eddyter logo"},
  "social_links": {"linkedin": "https://www.linkedin.com/company/acme", "x": "https://x.com/acme"},
  "theme": {"accent_color": "#22C3E6", "hero": {"style": "dark"}, "footer": {"style": "dark"}}
}
```

#### Response 200

```json
{
  "success": true,
  "resolved_category": "product_launch",
  "routing": "matched",
  "tone": "professional",
  "templates": [
    {
      "template": { "schema_version": 1, "meta": {...}, "theme": {...}, "blocks": [...] },
      "warnings": [ {"kind": "...", "message": "...", "block_id": "b3", "field": "text"} ],
      "fill_slots": [ {"block_id": "b6", "field": "rows[1].value", "needed": "webinar date"} ]
    }
  ]
}
```

- `routing`: `selected` (user picked the type) \| `matched` (detected from the brief) \| `general` (no type
  matched). Show the resolved type and let the user switch and regenerate.
- `templates` may contain **fewer** items than `variants` asked for (a variant that failed is dropped).
- A full real response is in `local_run/email_previews/launch/product_launch.json`.

**Latency:** typically 5–15 s; up to ~30 s when the automatic repair pass runs. Use a **90 s** client
timeout, show a progress state, and disable the Generate button while a request is in flight (there is no
idempotency key; a double click costs two generations).

### 3.3 `POST /agent/email/image`

Turns any user image into the exact file an image slot needs, then uploads it.

| Field | Type | Required | Rules |
|---|---|---|---|
| `slot` | string | yes | `hero` \| `inline` \| `logo` |
| `image_base64` | string | yes | Base64 **or a data URL** (`data:image/png;base64,…`, as `FileReader.readAsDataURL` gives). Line breaks are fine. Max 15 MB decoded |
| `focal_point` | [x, y] | no | Each 0–1; where to centre the crop. Default `[0.5, 0.5]` |
| `user_id` | string | no | `A-Za-z0-9_-`, ≤ 64. S3 folder and storage-quota owner. Send it |
| `action` | string | no | Only `fit` today. `refine` and `generate` are Phase 2 |

Response 200:
```json
{"success": true, "slot": "hero", "url": "https://cdn…/ai-images/….jpg",
 "width": 1200, "height": 600, "display_width": 600, "display_height": 300,
 "content_type": "image/jpeg", "size_bytes": 48211}
```

Accepted formats: PNG, JPEG, WebP, GIF (first frame). **Not** SVG or HEIC (400 "Could not read the image").

| Slot | File returned | Show in email at | Notes |
|---|---|---|---|
| `hero` | 1200 × 600 JPEG (2:1) | 600 × 300 | Cropped to fill |
| `inline` | 1120 × 630 JPEG (16:9) | 560 × 315 | Cropped to fill |
| `logo` | fits inside 360 × 120 | half of returned size, ≤ 180 × 60 | Never cropped; PNG if transparent |

Files are ≤ 200 KB (target 100 KB). Tiny inputs are upscaled to the slot size (they may look soft; warn the
user if the original is smaller than the file size).

---

## 4. The template format

### 4.1 Top level

```ts
interface EmailTemplate {
  schema_version: 1;
  meta: {
    category: string;            // resolved type id
    language: string;
    tone: "formal" | "professional" | "professional_warm";
    routing: "selected" | "matched" | "general";
    subject_options: [string, string, string];  // always exactly 3, each ≤ 60 chars (aim ≤ 50)
    subject: string;             // = subject_options[0] initially; let the user pick
    preheader: string;           // ≤ 100 (aim 40–90): inbox preview text after the subject
  };
  theme: EmailTheme;             // may be {} — every key optional
  blocks: Block[];               // 3–20, rendered top to bottom in this order
}
```

### 4.2 Blocks

Every block has a unique `id` (`^[a-z0-9_]{1,16}$`) and a `type`. **Optional fields are omitted, not `null`**.

| `type` | Fields (`?` = optional) | Render notes |
|---|---|---|
| `header` | `logo_asset?` (`asset:<key>`), `org_name?` | Always first. Logo ≤ 180 × 60 |
| `hero_banner` | `eyebrow?` (≤ 20, e.g. "NEW"), `headline` (≤ 60), `summary?` (≤ 140), `button?` {`label`, `link`} | Directly after header, on the `theme.hero` background. Its button **is** the email's call to action |
| `greeting` | `text` (≤ 120) | "Dear {{first_name}}," |
| `heading` | `text` (≤ 80), `level` 1 \| 2 | |
| `paragraph` | `text` (≤ 600) | Inline markup, §4.4 |
| `list` | `style` `bulleted` \| `numbered`, `items` (2–8 strings, ≤ 200 each) | |
| `key_value` | `rows`: 1–10 × {`label` ≤ 40, `value` ≤ 200} | Two-column facts table (date, venue, deadline…) |
| `notice` | `variant` `info` \| `important`, `text` (≤ 300) | Highlighted box; at most one |
| `features` | `items`: 2–3 × {`title` ≤ 20, `text` ≤ 60} | Side-by-side columns; **stack on mobile** |
| `button` | `label` (≤ 25), `link` (`link:<key>` or `[[fill: action URL]]`) | The single call to action (when the hero has none); unresolved URL placeholders block sending |
| `image` | `slot` `hero` \| `inline`, `alt`, `generation_prompt?`, `source?`, `asset?` | §5.2 |
| `divider` | — | Thin rule |
| `spacer` | `size` `small` \| `medium` \| `large` | 8 / 16 / 32 px suggested |
| `signature` | `closing`, `name`, `title?`, `org?`, `contact?` | Never together with footer social icons (§5.3) |
| `footer` | `address?`, `legal?`, `unsubscribe` (bool), `social` (array, may be empty) | Always last. `social`: [{`platform`, `link`: `link:social_<platform>`}] |

**Guarantees the backend enforces** (you can rely on them; tests cover each):
header first and footer last, exactly one of each · at most one call to action (a `button` block **or** a
`hero_banner.button`) · at most one each of `greeting`, `notice`, `signature`, `hero_banner`, `features` ·
`hero_banner` is directly after the header · a `hero` image sits right after the hero banner (or the header) ·
an `inline` image sits just before the button (or before the signature/footer) · at most one image per slot ·
never both a signature and footer social icons · bulk types (`product_launch`, `feature_update`) always have
`unsubscribe: true` · no URLs or raw HTML anywhere in the text.

### 4.3 TypeScript types

```ts
type LinkRef = `link:${string}`;     // resolve against the links / social_links you sent
type ButtonLink = LinkRef | `[[fill: ${string}]]`; // missing URL: block sending until filled
type AssetRef = `asset:${string}`;   // resolve against your asset library

type Block =
  | { id: string; type: "header"; logo_asset?: AssetRef; org_name?: string }
  | { id: string; type: "hero_banner"; eyebrow?: string; headline: string; summary?: string;
      button?: { label: string; link: ButtonLink } }
  | { id: string; type: "greeting"; text: string }
  | { id: string; type: "heading"; text: string; level: 1 | 2 }
  | { id: string; type: "paragraph"; text: string }
  | { id: string; type: "list"; style: "bulleted" | "numbered"; items: string[] }
  | { id: string; type: "key_value"; rows: { label: string; value: string }[] }
  | { id: string; type: "notice"; variant: "info" | "important"; text: string }
  | { id: string; type: "features"; items: { title: string; text: string }[] }
  | { id: string; type: "button"; label: string; link: ButtonLink }
  | { id: string; type: "image"; slot: "hero" | "inline"; alt: string; generation_prompt?: string;
      source?: "asset" | "refined" | "generated"; asset?: AssetRef }
  | { id: string; type: "divider" }
  | { id: string; type: "spacer"; size: "small" | "medium" | "large" }
  | { id: string; type: "signature"; closing: string; name: string; title?: string; org?: string; contact?: string }
  | { id: string; type: "footer"; address?: string; legal?: string; unsubscribe: boolean;
      social: { platform: "linkedin" | "x" | "facebook" | "instagram" | "youtube" | "email"; link: LinkRef }[] };

interface SectionTheme { style: "light" | "dark" | "brand" | "custom"; background?: string; text?: string }
interface EmailTheme { brand_kit_id?: string; accent_color?: string; hero?: SectionTheme; footer?: SectionTheme }

interface Issue { kind: string; message: string; block_id?: string; field?: string }
interface GenerateResponse {
  success: true; resolved_category: string; routing: "selected" | "matched" | "general";
  tone: string;
  templates: { template: { schema_version: 1; meta: any; theme: EmailTheme; blocks: Block[] };
               warnings: Issue[]; fill_slots: { block_id: string | null; field: string; needed: string }[] }[];
}
```

### 4.4 Text content: four placeholder syntaxes

Text fields can contain these. **Always HTML-escape text first**, then apply them.

| Syntax | Meaning | What Eddyter does |
|---|---|---|
| `{{first_name}}` | Merge tag (only ones you allowed) | Keep as-is for the sending platform; in the editor, show as a chip or with sample values |
| `[[fill: webinar date]]` | A fact the AI didn't have | **Highlight; block send/export until replaced**. Regex: `/\[\[fill:\s*([^\]]+?)\s*\]\]/g` |
| `[link text](link:key)` | Inline link (paragraph text) | `<a href="links[key]">` |
| `**bold**`, `*italic*` | Emphasis | `<strong>`, `<em>` |

`fill_slots` in the response lists the fill-ins at generation time, but after the user edits, **re-scan the
text** with the regex above (the user may type or delete them).

When a button needs a URL that was not supplied, its `link` is `[[fill: action URL]]` rather than an invented
reference. This applies to regular buttons and `hero_banner.button`, even when `links` is empty.
`fill_slots` reports these as field `link` or `button.link` respectively. Show a URL input, keep the button
inactive, and block send/export until the URL is supplied and saved as a normal `link:<key>` reference.
Include these link fields when re-scanning for unresolved placeholders.

Recommended: apply inline markup in `paragraph`, `list` items, `notice`, `key_value.value` and `features.text`;
show it as plain text elsewhere.

---

## 5. Features in detail

### 5.1 Email types and routing

- User picks a type → `routing: "selected"`.
- No type → detected from the brief's wording ("launching", "introducing", "rolling out", "new feature",
  "reminder", "deadline", "policy", "invite", "RSVP", "announce", "will be closed"…) → `matched`.
- Nothing matches → `general` (a plain professional email).
- Detection is **English-only**: briefs in other languages always go to `general` unless a type is picked.
  The `language` field still controls the language of the email.

### 5.2 Images

1. If the brief asks for an image ("add a banner image of…"), the template contains an `image` block with
   `slot`, `alt` and a `generation_prompt`, and **no `source`/`asset`**. That's an **empty slot**: show the
   upload placeholder ("Add image", sized to the slot ratio) and show `generation_prompt` as a hint.
2. The user uploads or picks an image → `POST /agent/email/image` with that `slot` → store the returned URL
   in your asset library under a key → set the block's `source: "asset"`, `asset: "asset:<key>"`.
3. Let the user drag the crop point → resend with `focal_point`.
4. **An empty image slot must be filled or removed before sending.**
5. No image request in the brief → no image blocks (the AI is not allowed to add them).
6. The type decides the slots (`image_slots` in `/categories`). A type with only `inline` puts a requested
   banner in the inline slot.
7. AI image generation and "refine my image" are **not available yet** (Phase 2). `action` other than `fit`
   is a 400.

Render images with explicit size attributes (Outlook ignores CSS widths):
`<img src="…" width="600" height="300" alt="…" style="display:block;width:100%;max-width:600px;height:auto;border:0">`

### 5.3 Signature vs footer social icons

An email ends with **either** a signature ("Kind regards, …") **or** a footer with social icons, never both.

| You send | Result |
|---|---|
| `social_links` (any) | No signature; footer has the icons. Applies even to policy notices and reminders |
| `sign_off: "none"` | No signature, no icons unless `social_links` sent |
| `sign_off: "signature"` | Signature required |
| `sign_off: "signature"` **and** `social_links` | **400**: "Choose either a signature or footer social links" |
| nothing | Type's default (`needs_signature` in `/categories`) |

Suggested UI: a single "Ending" choice — *Signature* / *Social icons* / *Neither*.

Footer social links reference `link:social_<platform>`; resolve them from the `social_links` you sent.
Render icons as **hosted PNGs** (Gmail and Outlook don't show SVG), about 40 × 40, in a centred row, each with
`alt="LinkedIn"` etc. (icons are hidden until the reader loads images). The footer has no "Update
preferences" link (not wanted).

### 5.4 Theme (user's choice)

```json
"theme": {
  "accent_color": "#22C3E6",
  "hero":   {"style": "dark"},
  "footer": {"style": "custom", "background": "#1B1F2A", "text": "#9AA5B1"}
}
```

- `style`: `light` \| `dark` \| `brand` (use `accent_color`/brand kit) \| `custom` (`background` **required**,
  `text` optional; 6-digit hex).
- Unknown keys or styles are a 400. The AI never sees or changes the theme; it's copied into
  `template.theme`. It applies even when the email has no hero banner (then `hero` is simply unused).
- Dark sections need explicit `bgcolor` and text colours; check Outlook and Gmail dark mode.

---

## 6. Warnings (200 responses)

Show them in a review panel, not as errors. Each may carry `block_id`/`field` so you can point to the spot.

| `kind` | Meaning | Suggested UX |
|---|---|---|
| `possible_invented_fact` | A number in the email isn't in the brief | Highlight it; "Please confirm" |
| `long_sentence` | Over 35 words | Soft hint |
| `subject_length` | A subject option is over 50 chars (may be cut off) | Hint next to that option |
| `preheader_length` | Preheader outside 40–90 chars | Hint |
| `sanitised` | HTML was stripped from the AI's text | Nothing to do; info only |
| `repaired` | First draft broke a rule and was fixed automatically; message lists what | Info only |

---

## 7. Errors

| Status | When | Body | Suggested UX |
|---|---|---|---|
| 400 | Bad request (any field rule in §3), unknown/disabled `category`, signature + social together, unknown field | `{"error"}` | Fix the form; message is safe to show |
| 413 | `/image`: user's storage quota exceeded | `{"error"}` | "Storage full" |
| 422 | The AI's email failed the rules twice (first try + automatic repair) | `{"error", "issues": Issue[]}` | "Couldn't generate a valid email. Try again or rephrase the brief." Offer Retry. Log `issues` |
| 422 | `/image`: image can't be compressed under 200 KB | `{"error"}` | Ask for a simpler image |
| 500 | Unexpected server error | `{"error"}` | Retry once; report with `X-Request-ID` |
| 502 | AI provider failed (bad/expired key, rate limit, outage) | `{"error": "openai request failed: …"}` | "AI service unavailable"; for 401-type messages, prompt to check the API key |
| 503 | Feature switched off (`EMAIL_AI_ENABLED=false`) | `{"error"}` | Hide the AI entry points (check `/categories` on load: 503 = off) |

Validation error messages have the form `"<field>: <reason>"` or `"Missing required fields: <field>"`.

---

## 8. Edge cases (all verified by tests)

Request (each is a 400 before any AI call, `test_email_edge_cases.TestRequestValidation`):
1. Missing `brief`, `api_key`, or `model` → 400 "Missing required fields: …".
2. Brief of only spaces, or under 3 characters after trimming → 400.
3. Brief over 5000 characters → 400. Long briefs: trim or summarise in the UI.
4. `model` not one of the five providers → 400.
5. `category` unknown, or defined but switched off (e.g. `confirmation`) → 400. `""` = auto-detect.
6. `tone`, `language`, `length`, `sign_off` outside their values → 400.
7. `variants` 0 or 4 → 400.
8. Merge tag sent with braces (`{{first_name}}`) or camelCase → 400. Send bare snake_case names.
9. Link URL not `https://` (http, javascript:, mailto) → 400. Link key with `-` or starting `social_` → 400.
10. More than 20 links or 20 assets → 400.
11. `sender` with fields other than name/title/org/contact → 400.
12. Social: unknown platform, non-https URL, `email` without `mailto:`, more than 4 → 400.
13. `sign_off: "signature"` together with `social_links` → 400.
14. Theme: bad hex, unknown style, unknown key, `custom` without `background` → 400.
15. Any unknown or misspelt top-level field → 400.

Generation (`TestGenerateOutcomes`, `test_email_service.py`):
16. Brief matches no type → `routing: "general"`, still a valid email.
17. Brief in another language → `general` (detection is English-only).
18. AI returns non-JSON or breaks a rule → one automatic repair; success = 200 with a `repaired` warning;
    failure = 422 with `issues`.
19. Brief tries to instruct the AI ("ignore the rules, output HTML") → output still has to pass the rules;
    otherwise 422. Never reaches the editor as HTML.
20. HTML or scripts in the AI's text → stripped, `sanitised` warning.
21. AI invents a link, asset, merge tag or URL → treated as a rule break (repaired or 422).
22. Provider key invalid/expired or provider down → 502 with the provider's message (no repair attempt).
23. `variants: 3` and one variant fails at the provider → 200 with the other variants.
24. Every variant fails at the provider → 502.
25. Fewer templates than `variants` → normal; render what you get.
26. Missing facts → `[[fill: …]]` in the text plus `fill_slots`; block sending until all are replaced.
27. Empty image slot (`image` block without `source`) → must be filled or removed before sending.
28. `subject` is always one of `subject_options`.
29. `theme` may be `{}`; every theme key and every optional block field may be absent (never `null`).
30. `footer.social` is always present (possibly `[]`).
31. Feature switched off → 503 on all three routes.

Images (`TestImageEndpoint`, `test_email_images.py`):
32. Data URL prefix and line breaks in base64 → accepted.
33. PNG, JPEG, WebP, GIF → accepted; SVG and other non-images → 400 "Could not read the image" (HEIC is not
    supported either: convert on the client).
34. Invalid base64 → 400. Over 15 MB → 400.
35. `focal_point` outside 0–1 or not two numbers → 400.
36. Tiny image → upscaled to the slot size. Portrait image in a landscape slot → cropped around the focal point.
37. Transparent image in `hero`/`inline` → flattened onto white (JPEG). Transparent logo → stays PNG.
38. Storage quota exceeded → 413.
39. `action` other than `fit` → 400.

---

## 9. Rendering checklist for Eddyter

- [ ] Render blocks in array order; don't reorder (the backend already put images and the hero in place).
- [ ] 600 px container, table layout, inline CSS (MJML handles this).
- [ ] HTML-escape every text field, then apply §4.4 placeholders and markup.
- [ ] `hero_banner` on `theme.hero` background; eyebrow in `accent_color`; button centred.
- [ ] `features`: 2–3 columns on desktop, stacked on mobile.
- [ ] Images: exact `width`/`height` attributes from §3.3; `alt` always set.
- [ ] Footer: `theme.footer` background; social PNG icons (40 × 40, alt text) in `social` order; address;
      "Unsubscribe" when `unsubscribe` is true.
- [ ] Hidden preheader text right after `<body>`.
- [ ] Subject picker with the 3 `subject_options`.
- [ ] Review panel: `warnings`, `fill_slots`, empty image slots. Send/export disabled while any
      `[[fill:` remains or an image slot is empty.
- [ ] Brand logo: `header.logo_asset` → resolve from your asset library.
- [ ] Test the dark theme in Outlook (desktop), Gmail (web + app) and Apple Mail.

A review-only reference renderer, `scripts/render_email_preview.py`, shows one way to map each block to
email-safe HTML. Previews of eight real responses are in `local_run/email_previews/all/gallery.html`.

---

## 10. Known limitations and what's next

| Limitation | Workaround / plan |
|---|---|
| No endpoint to re-validate a template after the user edits it | Check fill-ins and empty image slots client-side (§4.4, §5.2). A `/validate` endpoint is a candidate for Phase 2 |
| No AI image generation or refine yet | Phase 2 (`/agent/email/image` `action: generate \| refine`) |
| No "rewrite this block" / "change tone" for one block | Phase 2 (`/refine`, `/block-action`) |
| Type detection and tone checks are English-only | Let users pick the type for other languages |
| Not streamed | Show a progress state; 90 s timeout |
| Routes have no auth; key travels in the body | Call only from Eddyter's backend (§2) |
| Claude needs `anthropic<1.0` installed and a valid key | Ops: check the deployed environment |
