"""
Document import: DOCX, PDF and HTML files, and web addresses, to Markdown.

POST /agent/documents/import takes a multipart form with `api_key` and EITHER `file` (.docx, .pdf, .html) OR `url`, and returns the same
response as POST /agent/documents/upload (text, counts, sha256, headings, features, warnings) plus `source` (what was read and what the
converter noticed). The Markdown goes to document_context.text like an upload's. No model calls, nothing billed.

Order of defences, cheapest first:
  1. body cap      - the declared and streamed body size are checked before the form is parsed;
  2. auth          - the upload endpoint's resolver (backend secret, or licence), before any file or address is looked at;
  3. what is it    - a file's extension, content type and magic bytes must agree; an address is checked and fetched by fetch.py (public hosts only,
                     size and time limits, redirects re-checked) on its own small pool, which holds no conversion slot while it waits on the network;
  4. conversion    - in a child process (sandbox.py) with an empty environment and resource limits, on a small pool of at most
                     DOCUMENT_IMPORT_MAX_CONCURRENT, so a bad file or a burst of them costs bounded memory and CPU. A request that cannot get a
                     slot within DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS gets 503 + Retry-After;
  5. the result    - the same checks as an upload (length, NUL, empty), then headings, features and the document-injection scan.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from src.document_upload.router import _UploadCaller, _warnings, body_capped_route, build_payload

from .errors import ImportRefused
from .fetch import fetch_url
from .html_to_md import decode_html
from .sandbox import run_converter

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES: int = settings.document_import_max_file_bytes
_MAX_BODY_BYTES: int = _MAX_FILE_BYTES + 64 * 1024
_MAX_CHARS: int = 100_000                          # what DocumentContext.text allows; the same cap as an upload
_CONVERT_CHARS: int = _MAX_CHARS + 30_000          # convert a little past the cap so the refusal can say how far over it is, but never the whole of a huge file

_CONVERT_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=settings.document_import_max_concurrent, thread_name_prefix="doc_import")
_FETCH_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=settings.document_import_fetch_concurrent, thread_name_prefix="doc_import_fetch")

_TOO_LARGE = f"File too large. Maximum import size is {_MAX_FILE_BYTES // 1_000_000} MB."

router = APIRouter(prefix="/agent/documents", tags=["Document import"],
                   route_class=body_capped_route(lambda: _MAX_BODY_BYTES, lambda: _TOO_LARGE))

# ---------------------------------------------------------------------------
# What kind of file is it
# ---------------------------------------------------------------------------
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_HTML_START = re.compile(rb"\s*(?:<\?xml[^>]*>\s*)?(?:<!doctype\s+html|<html|<head|<body|<meta|<title|<h[1-6][\s>]|<p[\s>]|<div[\s>]|<article|<section|<table)", re.I)
_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".html": "html", ".htm": "html", ".xhtml": "html"}
_CONTENT_TYPES = {
    "pdf": {"", "application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream"},
    "docx": {"", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/octet-stream", "application/zip",
             "application/x-zip-compressed", "binary/octet-stream"},
    "html": {"", "text/html", "application/xhtml+xml", "text/plain", "application/octet-stream"},
}
_REDIRECT_TO_UPLOAD = {".md": "Markdown", ".markdown": "Markdown", ".txt": "text"}
_SAVE_AS_DOCX = {".doc": "an old Word file", ".rtf": "an RTF file", ".odt": "an OpenDocument file", ".pages": "a Pages file", ".dotx": "a Word template",
                 ".docm": "a macro-enabled Word file"}


def file_kind(filename: str, content_type: Optional[str], raw: bytes) -> str:
    """"pdf", "docx" or "html" when the name, content type and first bytes agree; ImportRefused (415) when they do not or the type is unsupported."""
    name = os.path.basename((filename or "").replace("\\", "/")).lower()
    ext = os.path.splitext(name)[1]
    ct = (content_type or "").split(";")[0].strip().lower()
    shown = filename or "this file"
    if ext in _REDIRECT_TO_UPLOAD:
        raise ImportRefused(415, "use_upload", f"'{shown}' is {_REDIRECT_TO_UPLOAD[ext]} already. Upload it with the Markdown/text upload instead.")
    if ext in _SAVE_AS_DOCX:
        raise ImportRefused(415, "unsupported_type", f"'{shown}' is {_SAVE_AS_DOCX[ext]}. Open it in Word and save it as .docx, then import that.")
    head = raw[:1024]
    sniffed = ("pdf" if b"%PDF-" in head else "docx" if raw.startswith(b"PK\x03\x04") else "ole" if raw.startswith(_OLE)
               else "html" if _HTML_START.match(raw[:2048]) else None)
    kind = _EXTENSIONS.get(ext) or (sniffed if ext == "" and sniffed in ("pdf", "docx", "html") else None)
    if kind is None:
        raise ImportRefused(415, "unsupported_type", f"'{shown}' cannot be imported. Supported: .docx, .pdf and .html files, or a web address.")
    if sniffed == "ole" and kind == "docx":
        raise ImportRefused(422, "legacy_word_format", "This is an old Word file (.doc) or a password-protected Word file, which cannot be read. "
                                                       "Open it in Word and save it as .docx (without a password), then import that.")
    if kind != sniffed and not (kind == "html" and sniffed is None and b"\x00" not in head):
        label = {"pdf": "a PDF", "docx": "a Word (.docx) document", "html": "an HTML page"}[kind]
        raise ImportRefused(415, "type_mismatch", f"'{shown}' is named like {label} but its content is not.")
    if ct not in _CONTENT_TYPES[kind] and not (kind == "html" and ct.startswith("text/")):
        raise ImportRefused(415, "type_mismatch", f"'{shown}' was sent as {ct}, which does not match its name.")
    return kind


# ---------------------------------------------------------------------------
# Work that runs on the pools
# ---------------------------------------------------------------------------

def _queued_before(deadline: float, fn: Callable, *args, **kwargs):
    """Worker-thread entry: a job that waited past its deadline for a free thread is dropped without doing any work."""
    if time.monotonic() > deadline:
        raise ImportRefused(503, "busy", "Document import is busy right now. Try again in a moment.", headers={"Retry-After": "5"})
    return fn(*args, **kwargs)


def _convert(kind: str, raw: bytes, filename: str, *, base_url: str = "", charset: Optional[str] = None, source: Optional[dict] = None) -> dict:
    """Convert in the sandbox, apply the upload rules to the Markdown, and build the response payload. Raises ImportRefused / HTTPException."""
    extra = {}
    if kind == "html":
        extra = {"DOC_IMPORT_BASE_URL": base_url, "DOC_IMPORT_CHARSET": charset or ""}
    converted = run_converter(kind, raw, timeout=settings.document_import_timeout_seconds, memory_bytes=settings.document_import_memory_mb * 1024 * 1024,
                              max_chars=_CONVERT_CHARS, max_pages=settings.document_import_max_pdf_pages, extra_env=extra)
    return _payload(converted.markdown, converted.warnings, converted.stats, converted.title, filename, kind, source or {}, len(raw), check_html=False)


def _text_payload(raw: bytes, filename: str, charset: Optional[str], source: dict) -> dict:
    """A .md/.txt address: the text is the document, with the same rules as an upload."""
    text = decode_html(raw, charset).lstrip("﻿")
    return _payload(text, [], {}, "", filename, "text", source, len(raw), check_html=True)


def _payload(markdown: str, converter_warnings: list, stats: dict, title: str, filename: str, kind: str, source: dict, original_bytes: int, *,
             check_html: bool) -> dict:
    if not markdown.strip():
        nested = any(w.get("code") == "nesting_too_deep" for w in converter_warnings)
        raise ImportRefused(422, "no_text", "No readable text was found. " + (
            "Its text is nested too deeply to be read." if nested else
            "The page may build its content with JavaScript (which is not run), or its text is in a form that cannot be read."))
    if len(markdown) > _MAX_CHARS:
        truncated = any(w.get("code") == "output_truncated" for w in converter_warnings)
        size = f"over {_CONVERT_CHARS:,}" if truncated else f"{len(markdown):,}"
        raise ImportRefused(413, "too_long", f"This document is {size} characters as text; the limit is {_MAX_CHARS:,}. "
                                              "Split it into parts (for a PDF, a Word file or a page) and import them one at a time.")
    payload = build_payload(markdown, filename, check_html=check_html)
    payload["warnings"] = [*converter_warnings, *payload["warnings"]]
    payload["source"] = {"kind": kind, "title": title, "original_bytes": original_bytes, "stats": stats, **source}
    return payload


def _http(exc: ImportRefused) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.message, headers={**exc.headers, "X-Error-Code": exc.code})


def _safe_name(url: str) -> str:
    parts = urlsplit(url)
    last = os.path.basename(parts.path.rstrip("/")) or ""
    return f"{parts.hostname or 'page'}/{last}"[:120] if last else (parts.hostname or "page")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/import")
async def import_document(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Form(..., min_length=1, description="Eddyter licence key (any non-empty string when X-Agent-Backend-Secret is sent)."),
    file: Optional[UploadFile] = File(None, description="A .docx, .pdf or .html file. Send this or `url`."),
    url: Optional[str] = Form(None, description="A public http(s) address of a web page, PDF, Word file or text file. Send this or `file`."),
) -> JSONResponse:
    """
    Convert a Word file, PDF, HTML file or web address to Markdown, ready for `document_context.text`.

    The response is the upload endpoint's (`text`, `char_count`, `word_count`, `line_count`, `sha256`, `filename`, `headings`, `features`,
    `warnings`) plus `source`: `{kind, title, original_bytes, stats, ...}` and, for an address, `url`, `final_url` and `content_type`.
    `warnings` carry what the conversion could not do (tables in a PDF, text boxes in a Word file, hidden text it left out, tracked changes it
    accepted) and the upload endpoint's prompt-injection warnings. Errors carry a stable code in the `X-Error-Code` header.
    """
    import api as _api  # noqa: PLC0415 - deferred: api.py imports this module at startup

    request_id = str(uuid.uuid4())[:8]
    headers = {"X-Request-ID": request_id}
    if not settings.document_import_enabled:
        raise HTTPException(status_code=503, detail="Document import is switched off.", headers={"X-Error-Code": "import_disabled"})

    caller = await _api.resolve_email_caller(request, _UploadCaller(api_key), request_id, background_tasks)

    address = (url or "").strip()
    if bool(address) == (file is not None and bool(file.filename or "")):
        raise HTTPException(status_code=422, detail="Send either a file or a url, not both and not neither.", headers={"X-Error-Code": "file_or_url"})

    loop = asyncio.get_running_loop()
    started = time.monotonic()
    try:
        deadline = started + settings.document_upload_queue_wait_seconds
        if address:
            if not settings.document_import_urls_enabled:
                raise ImportRefused(503, "urls_disabled", "Importing from a web address is switched off.")
            fetched = await loop.run_in_executor(
                _FETCH_POOL, lambda: _queued_before(deadline, fetch_url, address, max_html_bytes=5 * 1024 * 1024, max_file_bytes=_MAX_FILE_BYTES))
            source = {"url": address, "final_url": fetched.final_url, "content_type": fetched.content_type, "redirects": fetched.redirects}
            name = _safe_name(fetched.final_url)
            deadline = time.monotonic() + settings.document_upload_queue_wait_seconds
            if fetched.kind == "text":
                payload = await loop.run_in_executor(_CONVERT_POOL, lambda: _queued_before(deadline, _text_payload, fetched.data, name, fetched.charset, source))
            else:
                payload = await loop.run_in_executor(
                    _CONVERT_POOL, lambda: _queued_before(deadline, _convert, fetched.kind, fetched.data, name, base_url=fetched.final_url,
                                                          charset=fetched.charset, source=source))
            where = f"url host={urlsplit(fetched.final_url).hostname}"
        else:
            raw = await file.read(_MAX_FILE_BYTES + 1)
            if len(raw) > _MAX_FILE_BYTES:
                raise ImportRefused(413, "too_large", _TOO_LARGE)
            name = file.filename or ""
            kind = file_kind(name, file.content_type, raw)
            payload = await loop.run_in_executor(_CONVERT_POOL, lambda: _queued_before(deadline, _convert, kind, raw, name))
            where = f"file={name!r}"
    except ImportRefused as exc:
        logger.info(f"[{request_id}] import refused: {exc.code} status={exc.status}")
        raise _http(exc) from None

    logger.info(f"[{request_id}] import ok: {where} kind={payload['source']['kind']} chars={payload['char_count']} words={payload['word_count']} "
                f"warnings={[w['code'] for w in payload['warnings']]} ms={int((time.monotonic() - started) * 1000)} "
                f"caller={'backend' if caller['backend'] else caller['user_id']}")
    return JSONResponse(status_code=200, content=payload, headers=headers)
