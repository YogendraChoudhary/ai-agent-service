# Document Upload API — Python Backend Handoff

> **Endpoint:** `POST /agent/documents/upload`  
> **Audience:** Node Backend & Frontend Engineers  
> **Status:** Implemented & Verified (79 unit/integration tests passing, plus a live HTTP run; see section 7)  
> **Author:** Antigravity  

---

## 1. Overview & Purpose

The `POST /agent/documents/upload` endpoint provides an ingestion pipeline for Markdown (`.md`, `.markdown`) and plain-text (`.txt`) files on the Python backend.

When a user uploads a document file in the frontend, the Node backend forwards the multipart payload to this endpoint. The Python service:
1. Caps the request body (about 432 KB) **before** the multipart body is received or written to disk, so an oversized upload is refused with `413` without being stored, even from a caller who has no licence.
2. Authenticates the caller with the same resolver as `/agent/email/image` (backend secret, or the `/agent/stream` licence checks).
3. Validates the file type from the extension, the content type and the file's leading bytes (PDF, DOCX, images and archives are refused even when labelled `application/octet-stream` or renamed to `.md`).
4. Decodes **strictly** as UTF-8 / UTF-8-sig (stripping a BOM). Legacy encodings (Windows-1252, ISO-8859-x) and UTF-16 are refused with a clear message instead of being guessed, because a wrong guess silently corrupts the text.
5. Validates document content (rejection of HTML, NUL bytes, empty files).
6. Extracts structured document metadata (character count, word count, line count, table of contents / headings). This CPU work never runs on the event loop that serves other requests. It runs on small dedicated thread pools (see "Tunables"): files up to 16 KB on a thread of their own, larger ones on at most 2 threads. A burst of crafted documents therefore takes a bounded share of the CPU, and a small upload never waits behind large ones.
7. Returns a deterministic SHA-256 fingerprint alongside the clean text, making it directly ready for `document_context.text` on `/agent/process` and `/agent/stream`.

**No AI models are invoked and zero credits are billed.**

> Word (.docx), PDF and HTML files, and web addresses, are imported by the companion endpoint `POST /agent/documents/import` ([DOCUMENT_IMPORT_API_HANDOFF.md](DOCUMENT_IMPORT_API_HANDOFF.md)). It returns this endpoint's response plus a `source` object, and converts in a sandboxed child process. A PDF, DOCX or `.html` sent to *this* endpoint is still refused with `415`.

---

## 2. Request Contract

- **Method:** `POST`
- **Path:** `/agent/documents/upload`
- **Content-Type:** `multipart/form-data`

### Form Fields & Attachments

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` (binary) | **Yes** | The `.md`, `.markdown` or `.txt` file to upload, UTF-8 encoded. Max size: 400 KB (≈ 100,000 characters). A file with no extension is accepted only with a `text/plain`, `text/markdown` or `text/x-markdown` content type. |
| `api_key` | `string` (Form text) | **Yes** (non-empty) | The user's Eddyter licence key (or any non-empty string when called with `X-Agent-Backend-Secret`). |

### Request Headers

| Header | Type | Required | Description |
|---|---|---|---|
| `X-Agent-Backend-Secret` | `string` | Optional | Shared backend secret (`AGENT_BACKEND_SECRET`). If valid, bypasses the Redis licence check. A wrong secret is not an error: the caller is then checked as a licence caller. |
| `Content-Type` | `string` | **Yes** | `multipart/form-data` |

---

## 3. Success Response (`200 OK`)

```json
{
  "success": true,
  "text": "# Project Roadmap\n\n## Phase 1\nDeliver the core ingestion pipeline.\n",
  "char_count": 68,
  "word_count": 10,
  "line_count": 5,
  "sha256": "4b68bbad51a7...",
  "filename": "roadmap.md",
  "headings": [
    {
      "title": "Project Roadmap",
      "level": 1
    },
    {
      "title": "Phase 1",
      "level": 2
    }
  ],
  "features": ["front_matter", "math", "task_list"],
  "warnings": []
}
```

### Response Field Descriptions

| Field | Type | Description |
|---|---|---|
| `success` | `boolean` | Always `true` on successful 200 response. |
| `text` | `string` | The extracted, validated Markdown text with BOM stripped. Pass this directly as `document_context.text` in subsequent agent calls. |
| `char_count` | `integer` | Total character count of `text` (guaranteed $\le 100,000$). |
| `word_count` | `integer` | Total word count according to standard prose word boundary tokenizer. |
| `line_count` | `integer` | Total line count (`len(text.splitlines())`). |
| `sha256` | `string` | 64-char lowercase hex SHA-256 fingerprint of the UTF-8 encoded text. Matches `source_sha256` in `/agent/stream` context metadata. |
| `filename` | `string` | The original uploaded filename (e.g., `"notes.md"`). |
| `headings` | `list[object]` | Structured Table of Contents (TOC) array with `title` (`str`) and `level` (`int`, 1–6). Empty if the document has no headings. YAML/TOML front matter is not a heading, and a trailing `{#id .class}` attribute is stripped from the title. |
| `features` | `list[str]` | The Markdown dialect features the file uses, in this fixed order: `front_matter`, `math`, `footnotes`, `task_list`, `wikilinks`, `strikethrough`, `attributes`, `emoji_codes`. Empty for plain CommonMark. Code blocks are never searched. |
| `warnings` | `list[object]` | Empty when nothing was found. Otherwise one object per code, see "Warnings" below. The `text` is never changed because of a warning. |

---

## 4. Error Responses

| Status Code | Error Message / Condition | Cause / Resolution |
|---|---|---|
Every error body is `{"error": "<message>"}`; licence errors may add `"code"`. They are exactly the errors `/agent/stream` and `/agent/email/*` return for a licence key.

| Status Code | Error Message / Condition | Cause / Resolution |
|---|---|---|
| `400 Bad Request` | `Missing required fields: ...` / `api_key: String should have at least 1 character` | Missing `file`, or `api_key` missing or empty. |
| `402 Payment Required` | `Your payment is past due...` | Managed plan with a past-due subscription. Nothing is billed by this route, but the licence is checked like on every other route. BYOK plans are not blocked. |
| `403 Forbidden` | `Invalid license: <reason>` | The `api_key` is expired, inactive, or not found (when calling without the backend secret). |
| `403 Forbidden` | `Your access to the Eddyter editor has been suspended...` with `"code": "EDITOR_ACCESS_BLOCKED"` | The licence's owner is blocked by an administrator. Show the message as-is. |
| `413 Content Too Large` | `File too large...` | Request body over about 432 KB (refused before it is read), or the file itself over 400 KB. |
| `413 Content Too Large` | `Document too long...` | Extracted text exceeds 100,000 characters. |
| `415 Unsupported Media Type` | `Unsupported file type... Only .md and .txt...` | Extension is not `.md`/`.markdown`/`.txt`, or the content type contradicts it (e.g. `application/pdf`, `image/*`), or no extension without a text content type. |
| `415 Unsupported Media Type` | `'<name>' is a PDF / a ZIP-based file / a PNG image... not plain text or Markdown` | The file's leading bytes show a binary format, whatever its name or content type. |
| `422 Unprocessable Content` | `File is not valid UTF-8...` | Legacy encoding (e.g. Windows ANSI `.txt`). Ask the user to re-save as UTF-8. |
| `422 Unprocessable Content` | `File is saved as UTF-16...` | Detected from the BOM. Ask the user to re-save as UTF-8. |
| `422 Unprocessable Content` | `File is empty...` | File has 0 bytes or only whitespace. |
| `422 Unprocessable Content` | `File contains NUL characters...` | Binary file with `\x00` bytes that no signature check caught. |
| `422 Unprocessable Content` | `File appears to be HTML...` | Raw HTML document uploaded instead of Markdown. |
| `503 Service Unavailable` | `License validation service unavailable` / `Database error occurred` | Redis is down or unreachable and no backend secret was provided. |
| `503 Service Unavailable` | `Document processing is busy right now...` with `Retry-After: 2` | The upload waited longer than `DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS` for a free extraction thread (only under a burst of very large uploads). Retry after a moment. |

The `413` for an oversized body can arrive while the client is still sending: some HTTP clients report a connection reset (`EPIPE` / `ECONNRESET`) instead of reading the response. Treat that as "file too large" when the file you sent was over 400 KB.

---

### Markdown dialects

CommonMark and GFM tables are the base. These are not CommonMark, and are found in code before the model is ever involved, so an AI action cannot "fix" them:

| Construct | Example | Treated as |
|---|---|---|
| Front matter (YAML `---` or TOML `+++`, first thing in the file) | `---\ntitle: x\n---` | One opaque block. Not a heading. "Add at the top" lands below it. |
| Display math | `$$ ... $$` (also one line) and `\[ ... \]` | One opaque block |
| Footnote definitions | `[^1]: text` (indented continuation lines included) | One opaque block |
| Inline math | `$x_1$`, `$$x$$` | Opaque token inside prose. `$5 and $10`, `US$5` are not math. |
| Footnote references | `[^1]` | Opaque token |
| Wikilinks and embeds | `[[Page]]`, `![[img.png]]` | Opaque token |
| Strikethrough | `~~old~~` | Opaque token |
| Emoji codes | `:tada:` | Opaque token |
| Task-list boxes | `- [ ]`, `- [x]` | The box is an opaque token; the words of the item are still prose |
| Attribute lists | `{#id .class key=value}` | Opaque token |

Not covered: definition lists, `\begin{equation}` environments, and a math block or footnote that sits *between items of one list*. These stay visible to the model as ordinary text. A construct inside a fenced code block is code and is never treated as a dialect construct.

Short documents that contain any of these (except emoji codes alone) take the structure-preserving refine path, like a document with a table. Plain Markdown parses and freezes byte for byte as before: 6,295 of 6,361 real and fuzzed documents compared identical with the previous release, and the 66 that differ all contain one of the constructs above.

### Warnings

`warnings` is a list of `{"code", "message", "matches"}`. Each match is `{"kind", "label", "line" (1-based), "excerpt"}`; `excerpt` is empty for invisible characters. The scan is read-only: `text` and `sha256` are exactly what the file said.

| `code` | `kind` values | Meaning |
|---|---|---|
| `possible_prompt_injection` | `instruction_override`, `persona_hijack`, `system_prompt_override`, `forced_adversarial_output`, `data_exfiltration`, `safety_bypass`, `addressed_to_ai`, `reveal_prompt`, `control_tokens` | The file contains wording that reads like instructions to an AI. Show it to the user. |
| `hidden_characters` | `hidden_characters` | Invisible Unicode tag characters (a run of 8 or more, which can spell a hidden instruction; flag emoji use at most 7) or a text-direction override. |

This is a heuristic warning, not a guarantee and not the defence. The model is told on every document path (Ask, Draft, Edit, analysis, key points, charts) that the document is untrusted data and must never be obeyed. Measured live on 2 providers (OpenAI, Anthropic) with 5 payloads hidden in a memo (2 trials each, 20 calls), none was obeyed, and the upload scanner flagged all 5. On 357 real documents it flagged 0 of 174 third-party package documents; the 43 it flagged were this project's own docs and their generated copies, which do contain injection test strings.

### Tunables (environment variables)

All optional; leave a variable unset for the default. A blank or out-of-range value stops start-up. `DOCUMENT_UPLOAD_MAX_CONCURRENT` is read once at start-up.

| Variable | Default | Meaning |
|---|---|---|
| `DOCUMENT_UPLOAD_MAX_CONCURRENT` | `2` | Threads for extracting files larger than the small-file limit. The work is pure-Python CPU, so extra threads add no speed; they only take the GIL from the event loop. `1` gives the lowest lag under abuse; `2` also survives one slow outlier. Range 1-16. |
| `DOCUMENT_UPLOAD_SMALL_FILE_BYTES` | `16000` | Files up to this many bytes are extracted on one extra thread of their own and never queue behind large files. Range 1,000-400,000. |
| `DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS` | `10.0` | A request that has waited this long for a free thread gets `503` + `Retry-After` instead of queueing behind a burst. It is answered the moment a thread next frees up after the deadline, so the `503` can arrive up to one file's processing time later than the deadline (measured live with a 2 s deadline: 503s arrived at 2.8 s, while the running files finished). Range above 0 up to 60. |

Measured on a local server with 8 clients uploading crafted 100,000-character list documents nonstop for 8 s: with one shared pool of 8 threads `/health` peaked at 1,279 ms; with one shared pool of 2 threads it peaked at 406 ms but a legitimate 2 KB upload waited 2.5 s at the median; with this design (2 threads for large files plus 1 for small) `/health` peaked at 216-290 ms over two runs and the 2 KB upload took about 0.25 s at the median. Idle uploads are unchanged at about 4 ms. The trade-off: while such an attack is running, a legitimate *large* upload (60 KB) waits about 2.8 s behind the attacker's large uploads (it still succeeds, and gets `503` only after `DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS`).

---

## 5. Usage in the Complete Flow

```
[Browser / Frontend]
        │
        │ 1. User picks file (notes.md)
        ▼
[Node Backend]
        │
        │ 2. Forward multipart file: POST /agent/documents/upload
        ▼
[Python Agent Backend]
        │
        │ 3. Validate, extract text, compute sha256 + TOC
        │    Returns: { "text": "...", "sha256": "...", "headings": [...] }
        ▼
[Node / Frontend State]
        │
        │ 4. Store text in editor / state
        │    Call POST /agent/stream:
        │    {
        │      "type": "general",
        │      "document_action": "auto",
        │      "document_context": { "text": "<text from step 3>" },
        │      "query": "Summarize key findings"
        │    }
        ▼
[Python Agent Backend]
        │
        └─► Runs Ask/Draft/Edit/Analysis against the exact text!
```

---

## 6. Integration Examples

### cURL (Direct)
```bash
curl -X POST "http://localhost:5000/agent/documents/upload" \
  -H "X-Agent-Backend-Secret: <your_backend_secret>" \
  -F "file=@/path/to/my_notes.md;type=text/markdown" \
  -F "api_key=internal"
```

### Node.js (FormData forwarding)
```typescript
import FormData from 'form-data';
import fs from 'fs';
import axios from 'axios';

async function uploadDocument(filePath: string, filename: string, userLicenseKey: string) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath), { filename });
  form.append('api_key', userLicenseKey);

  const response = await axios.post('http://python-agent:5000/agent/documents/upload', form, {
    headers: {
      ...form.getHeaders(),
      'X-Agent-Backend-Secret': process.env.AGENT_BACKEND_SECRET,
    },
  });

  return response.data; // { success, text, char_count, word_count, sha256, headings, ... }
}
```

---

## 7. Verification

The upload route is verified by `tests/test_document_upload.py` (99 automated test cases), `tests/longform/test_markdown_dialects.py` (55) and `tests/test_document_threats.py` (50), and by live runs over real HTTP:
- Happy paths (`.md`, `.txt`, `.markdown`, octet-stream MIME fallback), exact text round trip (CRLF, emoji, BOM), and that the returned text and `sha256` are accepted by `DocumentContext` / equal `source_sha256`
- Body cap before parsing: oversized `Content-Length`, chunked bodies and a wrong `Content-Length`, including for callers with no licence
- Byte-level limit (400 KB) and character limit (100,000 chars), including exactly 100,000 four-byte characters
- Format rejection by extension, content type and magic bytes (PDF/DOCX/PNG/JPEG/RTF, also as `application/octet-stream` or with no content type)
- Strict UTF-8: Windows-1252 and ISO-8859-2 files refused instead of corrupted, UTF-16 detected
- Security validation (NUL bytes, raw HTML, empty files)
- Auth matrix on the real `RedisManager` licence rules: valid, unknown, expired, inactive, editor-blocked (with code), past-due managed vs BYOK, wrong secret, backend secret with Redis down, Redis outage, heartbeat
- Extraction runs in the threadpool
- Idempotency & SHA-256 validation
- Dialects, on a real model (OpenAI, `fix_grammar`, same document sent to the previous release and this one): the previous release let the model edit front matter in 5 of 5 runs and corrupt display math (`$$` became `$`) in 3 of 3 short-document runs; this release left every construct byte for byte in every run while still fixing the prose.
- Crafted inputs: every new pattern repetition is bounded. While testing, the pre-existing link-label pattern in `freeze` was found to take about 23 s on 100,000 characters of `\[` lines; it is now bounded (0.9 s).
