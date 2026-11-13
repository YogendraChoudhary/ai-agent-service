"""
Markdown / plain-text file upload endpoint.

Receives a .md or .txt file from the Node backend, validates and extracts its
text, and returns it ready to be passed as document_context.text in a
subsequent /agent/process or /agent/stream call.

No model calls, no billing. Auth is the one /agent/email/image uses
(api.resolve_email_caller): a caller with X-Agent-Backend-Secret is trusted and
needs no licence; anyone else must send a valid Eddyter licence key as api_key and
gets exactly the /agent/stream licence errors (403 invalid or editor-blocked with
its code, 402 past-due managed plan, 503 licence service down).

Order of defences, cheapest first:
  1. body cap      - Content-Length and the streamed byte count are checked before
                     the multipart body is parsed, so an oversized upload is cut off
                     (413) without being received or written to disk;
  2. auth          - before any file content is looked at;
  3. file type     - extension, content type and magic bytes (no PDF/DOCX/images);
  4. extraction    - strict UTF-8, NUL/HTML/empty checks, headings, hash. This is
                     pure-Python CPU work, so it never runs on the event loop that
                     serves every other request (and SSE stream). It runs on small
                     dedicated pools: at most DOCUMENT_UPLOAD_MAX_CONCURRENT threads
                     for large files plus one for small files, so even a burst of
                     crafted documents takes a bounded share of the GIL, and a small
                     upload does not queue behind large ones. A request that cannot
                     get a thread within DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS gets 503.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from config import settings

logger = logging.getLogger(__name__)

# Extraction is pure-Python CPU work. It runs on its own small pool, not the shared one, for two reasons: the pool size
# is the concurrency cap (more threads would only take the GIL from the event loop that serves every other request), and a burst
# of slow uploads cannot occupy threads other routes use. Read once at start-up, like the classifier pools in src/intent_classifier.py.
_EXTRACT_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=settings.document_upload_max_concurrent, thread_name_prefix="doc_upload")
# Small files get one thread of their own: a queue is first-in-first-out, so without it a user's 2 KB upload would wait behind every
# large upload ahead of it. They are cheap by construction (see document_upload_small_file_bytes), so one thread keeps up.
_SMALL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="doc_upload_small")

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
# Worst case: 100 000 Unicode chars x 4 UTF-8 bytes each = 400 000 bytes.
_MAX_FILE_BYTES: int = 400_000
# The whole multipart body may carry a little more than the file: boundaries, the api_key field, the filename.
_MAX_BODY_BYTES: int = _MAX_FILE_BYTES + 32 * 1024
# Characters DocumentContext.text allows.
_MAX_CHARS: int = 100_000

_TOO_LARGE = (f"File too large. Maximum upload size is {_MAX_FILE_BYTES // 1000} KB "
              f"(≈ {_MAX_CHARS:,} characters).")

# ---------------------------------------------------------------------------
# File type policy
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".txt")

# A file with no extension has only its content type to go on, so that must say "text".
_TEXT_CONTENT_TYPES: frozenset[str] = frozenset({"text/plain", "text/markdown", "text/x-markdown"})

# With an accepted extension the content type is a second opinion: clients send text/*, nothing, or the
# catch-all octet-stream for .md; anything that names another kind of file (application/pdf, image/*,
# Office types...) contradicts the name and is refused.
_NEUTRAL_CONTENT_TYPES: frozenset[str] = frozenset({
    "", "application/octet-stream", "application/markdown", "application/x-markdown",
})

# Leading bytes of common binary formats. A renamed or mislabelled file still starts with these.
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "a PDF"),
    (b"PK\x03\x04", "a ZIP-based file (DOCX, XLSX, PPTX, ODT...)"),
    (b"PK\x05\x06", "a ZIP-based file (DOCX, XLSX, PPTX, ODT...)"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "a legacy Office file (DOC, XLS, PPT)"),
    (b"\x89PNG\r\n\x1a\n", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF87a", "a GIF image"),
    (b"GIF89a", "a GIF image"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"7z\xbc\xaf\x27\x1c", "a 7z archive"),
    (b"Rar!\x1a\x07", "a RAR archive"),
    (b"{\\rtf", "an RTF document"),
)

_UTF16_BOMS: tuple[bytes, ...] = (b"\xff\xfe", b"\xfe\xff")        # also the start of UTF-32 LE


def _unsupported_type(filename: str) -> HTTPException:
    return HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{filename}'. Only .md and .txt (plain text / Markdown) files are accepted.",
    )


def _validate_file_type(filename: str | None, content_type: str | None) -> None:
    """415 unless the name and content type both describe a plain text / Markdown file."""
    shown = filename or ""
    name = os.path.basename(shown.replace("\\", "/")).lower()       # some clients send a full Windows path
    ext = os.path.splitext(name)[1]
    ct = (content_type or "").split(";")[0].strip().lower()

    if ext:
        name_ok = ext in _ALLOWED_EXTENSIONS
        type_ok = ct.startswith("text/") or ct in _NEUTRAL_CONTENT_TYPES
    else:
        name_ok = ct in _TEXT_CONTENT_TYPES
        type_ok = True
    if not (name_ok and type_ok):
        raise _unsupported_type(shown)


# ---------------------------------------------------------------------------
# Extraction (CPU-bound: always run through the threadpool)
# ---------------------------------------------------------------------------

def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_ATTRIBUTES_TAIL = re.compile(r"\s*\{[#.][^}\n]{0,100}\}\s*$")


def _headings(found: list[tuple[int, str]]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for level, line in found:
        title = _ATTRIBUTES_TAIL.sub("", re.sub(r"^#{1,6}\s*|\s*#+\s*$", "", line).strip()).strip()   # "Install {#install}" -> "Install"
        if title:
            headings.append({"title": title, "level": level})
    return headings


def _warnings(text: str) -> list[dict[str, Any]]:
    """Signals the user should see before running an AI action on this file. The text itself is never changed."""
    from src.security_guardrails import detect_document_threats
    threats = detect_document_threats(text)
    wording = [t for t in threats if t["kind"] != "hidden_characters"]
    hidden = [t for t in threats if t["kind"] == "hidden_characters"]
    out: list[dict[str, Any]] = []
    if wording:
        out.append({"code": "possible_prompt_injection", "matches": wording,
                    "message": "This file contains wording that reads like instructions to an AI. AI actions treat document text as data and "
                               "ignore it, but review these lines before you rely on the result."})
    if hidden:
        out.append({"code": "hidden_characters", "matches": hidden,
                    "message": "This file contains invisible characters that can hide text from you. Review the file's source before using it."})
    return out


def _decode(raw: bytes, filename: str) -> str:
    """Bytes -> str, strictly. A wrong guess at a legacy encoding corrupts the text silently, so refuse instead."""
    for signature, kind in _BINARY_SIGNATURES:
        if raw.startswith(signature):
            raise HTTPException(
                status_code=415,
                detail=(f"'{filename}' is {kind}, not plain text or Markdown. "
                        "Export it as .txt or .md and upload that instead."),
            )
    if raw.startswith(_UTF16_BOMS):
        raise HTTPException(
            status_code=422,
            detail="File is saved as UTF-16. Re-save it as UTF-8 (in Notepad: Save As > Encoding: UTF-8) and upload it again.",
        )
    try:
        return raw.decode("utf-8-sig")                              # utf-8-sig drops a leading BOM
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail=("File is not valid UTF-8, so its text cannot be read reliably (often a Windows ANSI or other legacy "
                    "encoding). Re-save it as UTF-8 and upload it again."),
        ) from None


def _extract_before(deadline: float, raw: bytes, filename: str, declared_size: int | None) -> dict[str, Any]:
    """Worker-thread entry: a job that waited past its deadline for a free thread is dropped without doing any work."""
    if time.monotonic() > deadline:
        raise HTTPException(
            status_code=503,
            detail="Document processing is busy right now. Try the upload again in a moment.",
            headers={"Retry-After": "2"},
        )
    return _extract_document(raw, filename, declared_size)


def _extract_document(raw: bytes, filename: str, declared_size: int | None = None) -> dict[str, Any]:
    """
    Validate the file bytes and build the response payload.

    Raises HTTPException (413 / 415 / 422) on any failure. Pure CPU work, no I/O.
    """
    # 1. Bytes: the read is capped one byte over the limit, so anything over it is oversized.
    if len(raw) > _MAX_FILE_BYTES:
        size = declared_size if declared_size is not None and declared_size > _MAX_FILE_BYTES else None
        raise HTTPException(status_code=413, detail=_TOO_LARGE + (f" Got {size:,} bytes." if size else ""))

    # 2. Decode (rejects binary formats, UTF-16 and non-UTF-8)
    text = _decode(raw, filename).lstrip("﻿")                  # a doubled BOM survives utf-8-sig once

    return build_payload(text, filename)


def build_payload(text: str, filename: str, *, check_html: bool = True) -> dict[str, Any]:
    """
    The response for text that is already decoded: length, NUL, HTML and empty checks, then headings, features, hash and warnings.

    Shared with POST /agent/documents/import (src/document_import/router.py), whose converted Markdown must obey the same rules because it
    goes to the same place (document_context.text). `check_html` is False there: converted text has its angle brackets escaped, and a page that
    was HTML is exactly what is being converted.
    """
    # 3. Characters: the limit DocumentContext enforces
    if len(text) > _MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Document too long. Maximum is {_MAX_CHARS:,} characters; this file has {len(text):,}.",
        )

    # 4. NUL (rejected by DocumentContext.valid_source)
    if "\x00" in text:
        raise HTTPException(
            status_code=422,
            detail="File contains NUL characters, so it is a binary file and not plain text or Markdown.",
        )

    # 5. HTML (lazy import keeps module load cheap and avoids an import cycle at startup)
    from src.longform.markdown_ast import looks_like_html, scan, word_count
    if check_html and looks_like_html(text):
        raise HTTPException(
            status_code=422,
            detail=("File appears to be HTML. Supply plain text or Markdown. "
                    "If you exported from a word processor, save as Plain Text (.txt) first."),
        )

    # 6. Empty / whitespace-only
    if not text.strip():
        raise HTTPException(status_code=422, detail="File is empty. Upload a non-empty plain text or Markdown document.")

    scanned = scan(text)                                            # headings and dialect features in one block-structure pass
    return {
        "success": True,
        "text": text,
        "char_count": len(text),
        "word_count": word_count(text),
        "line_count": len(text.splitlines()),
        "sha256": _fingerprint(text),
        "filename": filename,
        "headings": _headings(scanned.headings),
        "features": scanned.features,
        "warnings": _warnings(text),
    }


# ---------------------------------------------------------------------------
# Body cap: reject before the multipart body is received, parsed or spooled to disk
# ---------------------------------------------------------------------------

def body_capped_route(limit: Callable[[], int], message: Callable[[], str]) -> type[APIRoute]:
    """
    A route class that answers 413 once the request body passes `limit()` bytes.

    FastAPI parses a multipart body (and spools every file part to a temp file) before the endpoint runs, so a
    check inside the endpoint cannot stop a large body, least of all one from a caller who is not authorised.
    The class checks the declared Content-Length first and counts the streamed bytes after, and answers
    413 the moment either passes the cap. The limit and message are read per request, so a router built here
    and one built for another endpoint (src/document_import/router.py) each keep their own.
    """

    class _Capped(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
            handler = super().get_route_handler()

            def too_large() -> HTTPException:
                return HTTPException(status_code=413, detail=message(), headers={"Connection": "close"})

            async def capped(request: Request) -> Response:
                cap = limit()
                declared = request.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > cap:
                    raise too_large()                               # not a single body byte has been read

                received = 0
                receive = request.receive

                async def counting_receive():
                    nonlocal received
                    message_ = await receive()
                    if message_["type"] == "http.request":
                        received += len(message_.get("body", b""))
                        if received > cap:                          # no Content-Length (chunked) or a wrong one
                            raise too_large()
                    return message_

                return await handler(Request(request.scope, counting_receive))

            return capped

    return _Capped


_BodyCappedRoute = body_capped_route(lambda: _MAX_BODY_BYTES, lambda: _TOO_LARGE)

router = APIRouter(prefix="/agent/documents", tags=["Document upload"], route_class=_BodyCappedRoute)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _UploadCaller:
    """What api.resolve_email_caller reads from a request body: the licence key (no model, so no provider key is resolved)."""
    api_key: str


@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A .md or .txt file, UTF-8 encoded, at most 100 000 characters (400 KB)."),
    api_key: str = Form(..., min_length=1, description="Eddyter licence key (any non-empty string when X-Agent-Backend-Secret is sent)."),
) -> JSONResponse:
    """
    Ingest a Markdown or plain-text file and return its extracted text.

    The returned ``text`` field is ready to be supplied as
    ``document_context.text`` in a ``/agent/process`` or ``/agent/stream``
    request.  No model calls are made and nothing is billed.

    **Response** (200):
    ```json
    {
      "success": true,
      "text": "<extracted document text>",
      "char_count": 4821,
      "word_count": 702,
      "line_count": 96,
      "sha256": "<SHA-256 hex of UTF-8 encoded text>",
      "filename": "notes.md",
      "headings": [{"title": "Intro", "level": 1}]
    }
    ```
    """
    import api as _api  # noqa: PLC0415 - deferred: api.py imports this module at startup

    request_id = str(uuid.uuid4())[:8]
    headers = {"X-Request-ID": request_id}

    # Auth: the same resolver as /agent/email/image (backend secret, or the /agent/stream licence checks).
    caller = await _api.resolve_email_caller(request, _UploadCaller(api_key), request_id, background_tasks)

    filename = file.filename or ""
    _validate_file_type(filename, file.content_type)

    # The body cap already bounded the upload; this bounds the in-memory copy too.
    raw = await file.read(_MAX_FILE_BYTES + 1)
    deadline = time.monotonic() + settings.document_upload_queue_wait_seconds
    pool = _SMALL_POOL if len(raw) <= settings.document_upload_small_file_bytes else _EXTRACT_POOL
    payload = await asyncio.get_running_loop().run_in_executor(pool, _extract_before, deadline, raw, filename, file.size)

    logger.info(
        f"[{request_id}] upload ok: file={filename!r} chars={payload['char_count']} words={payload['word_count']} "
        f"sha256={payload['sha256'][:12]}… caller={'backend' if caller['backend'] else caller['user_id']}"
    )
    return JSONResponse(status_code=200, content=payload, headers=headers)
