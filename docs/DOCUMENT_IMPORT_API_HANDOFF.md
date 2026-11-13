# Document Import API — Python Backend Handoff

> **Endpoint:** `POST /agent/documents/import`  
> **Audience:** Node Backend & Frontend Engineers  
> **Status:** Implemented and verified (unit/integration tests, plus live runs against real pages, PDFs and a Word file; see section 8)  
> **Companion of:** `POST /agent/documents/upload` ([DOCUMENT_UPLOAD_API_HANDOFF.md](DOCUMENT_UPLOAD_API_HANDOFF.md)); the response is the same shape, so everything that takes an upload's `text` takes an import's.

---

## 1. What it does

Turns a **Word file (.docx), a PDF, an HTML file, or a web address** into Markdown for the editor. Send either a `file` or a `url`.

| Source | What you get | What you do not get |
|---|---|---|
| **.docx** | Headings (a *Title* paragraph becomes `#` and the headings move down one level), lists (nested), tables (header row, pipes escaped), links, bold/italic/strike, inline code, code and quote styles, footnotes and endnotes (`[^1]` / `[^e1]`), accepted tracked changes. | Pictures, text boxes (partly), equations, comments, headers/footers, macros (never run). Each is named in `warnings`. |
| **.pdf** | Reading order (two columns are read column by column), headings (from font size and bold), paragraphs (lines joined, hyphenated words rejoined, a paragraph that runs over a page break is one), bullets and numbered lists, running headers/footers and page numbers removed, rotated margin stamps (the arXiv stamp) skipped. | **Tables** (their text comes out cell by cell and the pages are named in a warning), pictures, formulas, a scanned PDF (no text layer: refused, no OCR), a password-protected or copy-protected PDF (refused). |
| **web page / .html** | The article (navigation, footers, cookie banners, share bars, comments dropped), headings, lists, GFM tables (layout tables are read as ordinary blocks), code with its language, quotes, links made absolute, math (`$..$`), `[ ]`/`[x]` task boxes. | Pages that build their content with JavaScript (not run; a warning says so), pages behind a login, pages that refuse automated readers. |
| **a `.md` / `.txt` address** | The text as it is, under the upload endpoint's rules. | |
| **a PDF / .docx address** | Fetched, then converted as above. | |

Everything the converter had to leave out or guess is reported in `warnings` (codes in section 5). Nothing is silently dropped.

**No AI model is called and nothing is billed.** The import returns Markdown; running a model on it is a separate request (`/agent/process`, `/agent/stream`) with the returned `text` as `document_context.text`.

---

## 2. Request

- **Method / path:** `POST /agent/documents/import`, `multipart/form-data`
- **Auth:** the upload endpoint's: `X-Agent-Backend-Secret`, or a valid Eddyter licence key as `api_key` (same `403`/`402`/`503` licence errors as `/agent/stream`).

| Field | Required | Description |
|---|---|---|
| `api_key` | yes (non-empty) | Licence key (any non-empty string with the backend secret). |
| `file` | one of `file` / `url` | `.docx`, `.pdf`, `.html`/`.htm`. Name, content type and first bytes must agree. At most `DOCUMENT_IMPORT_MAX_FILE_BYTES` (15 MB). |
| `url` | one of `file` / `url` | A public `http(s)` address, at most 2,048 characters, no user name/password, port 80/443/8080/8443. Sending both, or neither, is `422 file_or_url`. |

`.md`/`.txt` files are refused with `415 use_upload` (use the upload endpoint). Old `.doc`, `.rtf`, `.odt`, `.pages`, `.docm`, `.dotx` are refused with a message to save as `.docx`.

## 3. Response (`200`)

The upload response plus `source`:

```json
{
  "success": true,
  "text": "# Annual report\n\n## Intro\n...",
  "char_count": 4821, "word_count": 702, "line_count": 96, "sha256": "…",
  "filename": "report.docx",
  "headings": [{"title": "Annual report", "level": 1}],
  "features": ["footnotes"],
  "warnings": [{"code": "images_omitted", "message": "4 image(s) were left out: pictures are not imported."}],
  "source": {
    "kind": "docx",                 // docx | pdf | html | text
    "title": "Annual report",       // from the document (Title style / PDF metadata / <h1> or <title>)
    "original_bytes": 58213,
    "stats": {"headings": 17, "tables": 5, "images_omitted": 4},   // what the converter counted (keys vary by kind)
    "url": "https://…", "final_url": "https://…", "content_type": "text/html", "redirects": 1   // address imports only
  }
}
```

`text` obeys the upload rules: at most **100,000 characters** (`413 too_long` otherwise, with the count; a huge document is cut at 130,000 and reported as "over 130,000"), no NUL, not empty. `sha256` is of `text`, so `document_context` agrees with it. `warnings` also carries the upload endpoint's prompt-injection warnings (`possible_prompt_injection`, `hidden_characters`) computed on the converted text.

## 4. Errors

Body: `{"error": "<message for the user>"}` (or `{"detail": …}` from FastAPI's own validation); the stable code is in the **`X-Error-Code`** response header. Licence errors are the upload endpoint's.

| Status | `X-Error-Code` | When |
|---|---|---|
| 413 | *(body cap)* | Request body over 15 MB + 64 KB; refused before the form is parsed, even without a licence. |
| 413 | `too_large` / `fetch_too_large` | File, or the fetched address, over its size limit (web page 5 MB, file 15 MB, counted after decompression). |
| 413 | `too_long` | Converted text over 100,000 characters. |
| 413 | `docx_too_complex` / `too_complex` | A Word file that expands past its limits (a zip bomb), or a conversion that ran out of memory. |
| 415 | `use_upload`, `unsupported_type`, `type_mismatch`, `fetch_unsupported_type` | Wrong or unsupported type (a `.png`, a PDF named `.docx`, an address that returns JSON or an image). |
| 422 | `file_or_url` | Both or neither of `file`/`url`. |
| 422 | `invalid_url`, `url_not_fetchable` | Malformed address; or not a public web address. One message for every case, so it never says whether a private host exists. |
| 422 | `fetch_status`, `too_many_redirects`, `fetch_empty`, `tls_failed` | The site answered 4xx (401/403/429 add "may need a login or may not allow automated readers"), redirected more than 5 times, returned nothing, or has a bad certificate. |
| 422 | `legacy_word_format`, `not_a_docx`, `docx_encrypted`, `docx_unsafe`, `docx_unreadable` | Old `.doc` or a password-protected Word file; not a Word file; encrypted; contains XML declarations (`DOCTYPE`/`ENTITY`) Word never writes; damaged. |
| 422 | `not_a_pdf`, `pdf_encrypted`, `pdf_copy_protected`, `scanned_pdf`, `pdf_unreadable` | PDF cases (a scan says OCR is not supported). |
| 422 | `no_text`, `import_timeout`, `unreadable` | Nothing readable (also: nested too deeply, JavaScript-only page); conversion killed at the time limit; conversion crashed. |
| 502 / 504 | `fetch_status` / `fetch_timeout`, `fetch_failed` | The site answered 5xx; did not answer in time; or sent an invalid answer. |
| 503 | `busy` (+ `Retry-After: 5`) | No conversion slot within `DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS`. |
| 503 | `import_disabled`, `urls_disabled`, `import_unavailable` | Switched off (see tunables), or the child process could not be started. |

## 5. Warning codes (`warnings[].code`)

`images_omitted`, `hidden_text_removed`, `hidden_characters_removed`, `white_text`, `tracked_changes_accepted`, `text_boxes`, `equations_skipped`, `comments_skipped`, `macros_ignored`, `table_merged_cells`, `tables_not_rebuilt` (names the pages), `pages_truncated`, `output_truncated`, `little_text` (a JavaScript page), `nesting_too_deep`, plus the upload endpoint's `possible_prompt_injection` and `hidden_characters`. Each has a `message` written for the user. Suggested UI: show them as a notice above the imported text.

---

## 6. Safety (read this before changing anything)

**Opening a stranger's file is the risky part, and an address someone gives a server is a way into its network.** Defences, in order:

1. **Body cap** before parsing; **auth** before the file or address is looked at.
2. **Every conversion runs in a child process** (`src/document_import/sandbox.py`, `worker.py`): `python -I`, an **empty environment** (no API keys, Redis URL or cloud credentials), a private temp directory as its working directory, no stdin, its own session, limits set before parsing (CPU seconds, address space on Linux, output file size, open files, no core dumps), killed with its whole process group at `DOCUMENT_IMPORT_TIMEOUT_SECONDS`. A parser bug costs one child. At most `DOCUMENT_IMPORT_MAX_CONCURRENT` (2) run at once; a burst gets `503 busy`.
3. **Word files:** zip checked first (entry count, total size, expansion ratio, encryption, must be a Word package); parts containing `<!DOCTYPE`/`<!ENTITY` are refused (XXE, entity bombs); text a reader cannot see (Word "hidden" formatting, 1 pt font) is removed and counted; white text on an unshaded background is reported, not removed (white on a dark heading band is legitimate); macros are never run.
4. **Web pages:** hidden elements (`hidden`, `aria-hidden`, `display:none`, `visibility:hidden`, `opacity:0`, `font-size:0`, off-screen positioning, comments, scripts) and zero-width / Unicode "tag" / bidi characters are removed. That is where hidden prompt-injection text lives. Visible instruction-like wording is **flagged, never altered** (`possible_prompt_injection`).
5. **Fetching** (`src/document_import/fetch.py`): public hosts only. The name is resolved once per connection and the connection goes to exactly the addresses that were checked (no DNS-rebinding window); **every** address of a name must be public or none is used (`src/net_safety.py` also closes NAT64/6to4/Teredo/multicast forms that `ipaddress.is_global` accepts, e.g. `64:ff9b::a9fe:a9fe` is 169.254.169.254). Redirects are followed by hand (max 5) and every hop is checked again; no cookies, no proxies from the environment, none of our credentials are ever sent; body counted after decompression; whole-fetch deadline; only web pages, PDFs, Word files and text are accepted. httpx's own request log has query strings cut (they can hold tokens), and the import log line carries the host only.
6. **Output:** the same length/NUL/empty rules as an upload; `escape` everywhere so text that looks like Markdown (`1.`, `#`, `*`) stays text.

The fetcher assigns `transport._pool._network_backend` (httpx has no public hook). That is why `httpx==0.28.*` / `httpcore==1.0.*` is asserted in `tests/document_import/test_fetch.py`: an upgrade that moves it fails the suite instead of silently dropping the address check. The existing `read_web_page` tool uses the same technique.

**The user agent** is `Mozilla/5.0 (compatible; DocumentImport/1.0)`, honest about being an automated reader. Some sites refuse it: in the live run **Wikipedia answered 403**. That is reported as such; it is not worked around.

## 7. Tunables (environment variables)

All optional; leave a variable unset for the default. A blank or out-of-range value stops start-up. Pool sizes and the size cap are read once at start-up.

| Variable | Default | Meaning |
|---|---|---|
| `DOCUMENT_IMPORT_ENABLED` | `true` | `false` makes the endpoint answer `503 import_disabled` (switch off without a deploy). |
| `DOCUMENT_IMPORT_URLS_ENABLED` | `true` | `false` refuses `url` (the part that makes outgoing requests) and leaves files working. |
| `DOCUMENT_IMPORT_MAX_CONCURRENT` | `2` | Conversions at once. Each is a child process, so this is also the memory cap. Measured peak per child: a 12-15 page PDF about 55 MB, a Word file about 40 MB, a pathological 14 MB HTML file (a million elements) about 500 MB. Range 1-8. |
| `DOCUMENT_IMPORT_FETCH_CONCURRENT` | `4` | Addresses fetched at once (they wait on the network and hold no conversion slot). Range 1-16. |
| `DOCUMENT_IMPORT_MAX_FILE_BYTES` | `15000000` | Largest Word file or PDF, and the body cap (+64 KB). Range 100,000-50,000,000. |
| `DOCUMENT_IMPORT_TIMEOUT_SECONDS` | `30.0` | A conversion running longer is killed (`422 import_timeout`). A PDF stops reading earlier by itself at 60% of this and says so. Range 3-120. |
| `DOCUMENT_IMPORT_MAX_PDF_PAGES` | `200` | Pages read from a PDF; the rest is reported (`pages_truncated`). Range 1-1000. |
| `DOCUMENT_IMPORT_MEMORY_MB` | `1536` | Address-space limit of the child (**Linux only**: macOS does not enforce it). Range 256-8192. |

It also uses `DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS` (how long a request waits for a free slot before `503`).

**Deployment:** `pip install -r requirements.txt` adds `mammoth==1.13.0` (BSD-2), `pdfminer.six==20260107` (MIT) and `lxml` (BSD; already present through `ddgs`). PyMuPDF is faster but AGPL, so it is not used. The server needs outbound HTTPS for `url`, and a writable temp directory for the child's private working directory.

---

## 8. Verification

- **Tests:** `tests/document_import/` (converters, sandbox containment, fetcher against a real local server, endpoint, settings) and the unchanged upload suites. Run: `.venv/bin/python -m pytest tests/document_import tests/test_document_upload.py -q`.
- **Live, real sites and files** (harness server on `127.0.0.1:8765`):
  - Pages: the Python docs `venv` page (29 KB of Markdown, 5 headings, 6 code blocks, a table; navigation and footers gone, 0.2 s), the GNU "What is Free Software" essay (22 headings, 1.6 s), `example.com`, a raw GitHub `README.md`.
  - PDFs: *Attention Is All You Need* (15 pp, 38,000 characters, 11-heading outline, 1.0 s) and *Deep Residual Learning* (12 pp, two columns, 58,000 characters, 10-heading outline, reference list in order, 0.8 s).
  - Word: calibre's public demo `.docx` (17 headings, 5 tables, footnotes, strikethrough, 2.5 s).
  - Imported Word text through the real **translate** operation (Spanish, 24.7 s): all 17 headings, all 5 tables and their shapes, footnote labels and links kept. A small imported DOCX table and a small imported PDF answered correctly by a real model through **Ask**.
  - SSRF: 21 addresses aimed at the machine or its network (127.0.0.1, `localhost`, `::1`, `2130706433`, `0x7f.1`, `017700000001`, `::ffff:127.0.0.1`, `64:ff9b::7f00:1`, 6to4, 169.254.169.254, `metadata.google.internal`, 10.x, 192.168.x, `localtest.me`, `127.0.0.1.nip.io`, credentials in the address, `ftp:`, `file:`, ports 22 and 6379) all `422`; a canary listening on `127.0.0.1:8080` received **0 requests**. A real public redirect (`github.com` http→https; a redirector to `example.com`) is followed. (A public redirector pointed at the canary answered 403 itself, so the redirect-to-internal case is proven by the local-server unit tests, where the hop is shown never to be made.)
  - Hostile files: a 199 KB zip holding 200 MB of zeros → `413`; XXE and billion-laughs Word files → `422 docx_unsafe`; a truncated PDF; a PDF named `.docx`; HTML nested 100,000 deep, 1,000,000 spans (3.7 s), a 10 MB single paragraph, regex-hostile HTML. Other requests answered in a median of 1 ms (worst 5 ms) while these ran; no child process was left behind.
  - Burst: 60 simultaneous imports of the 12-page PDF → 32 served, 28 `503 busy` (Retry-After 5); the server never had more than **2** conversion processes at once (counted as children of the server process), and 0 afterwards.

## 9. Known limits

- **PDF tables are not rebuilt**; formulas and figures are not read; a PDF with no text layer is refused (no OCR). Heading detection is heuristic (font size, bold): a PDF that sets everything in one size gets no headings. Author lines under a paper's title are made plain text by a rule that can occasionally demote a real heading on page 1. A hyphen at a line end is rejoined unless the next word contains a hyphen, so `high-` `level` can become `highlevel`.
- **Word:** pictures are not imported; text boxes are read only in some forms; equations, comments, headers and footers are left out; drop caps come out as a one-letter paragraph; a heading style under another language's name is not recognised.
- **Web:** no JavaScript is run; no cookies or login; some sites refuse automated readers (Wikipedia returned 403). Main-content detection is a heuristic; `main_content` is always on for addresses.
- **Size.** Import rejects converted text over 100,000 characters. Ask can read the whole document when its serialized context fits within 6,000 UTF-8 bytes, including the context metadata. Send `document_scope: "auto"` to answer questions such as “Summarize section 4” from matching passages when the document is larger. Retrieved answers include references and carry a `partial_document_context` warning when coverage is partial. The default scope returns `needs_context` for an oversized document unless the server sets `DOCUMENT_LARGE_ASK=auto`. Use `document_analysis: "summary"` or `"contradictions"` with document scope for full-document analysis. See [Long-document Ask](DOCUMENT_CONTEXT_API_HANDOFF.md#long-document-ask).
- On a Mac, if a conversion child ever dies from a fatal signal (a real parser crash), macOS shows its "Python quit unexpectedly" dialog. That is a development-machine effect only; the request still gets a clean `422`/`413`. The test suite is written so it never triggers it (signal tests are skipped on macOS).
- macOS has no enforced address-space limit; the memory limit of the child is real on Linux only. CPU time, file size, open files and the kill timer apply everywhere.
