"""
Tests for POST /agent/documents/upload

Covers:
- happy path: .md and .txt files, BOM stripping, exact round trip, hash agrees with the context pipeline
- size guards: body cap before parsing (Content-Length and streamed), byte-level and char-level limits
- content validation: NUL, HTML, empty, strict UTF-8 (no silent legacy-encoding guesses), UTF-16
- file type: extension, content type and magic bytes (PDF/DOCX/images are refused even as octet-stream)
- auth: the /agent/email/image resolver (backend secret, licence errors with codes, past-due, heartbeat)
- extraction runs in the threadpool, not on the event loop
- response shape and X-Request-ID header
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
import threading
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
import src.document_upload.router as upload_router
from config import Settings, settings
from src.document_context import DocumentContext, prepare_context
from src.document_edit.blocks import document_blocks, utf16_length
from src.redis_manager import RedisManager

VALID_MD = "# Hello\n\nThis is a **test** document.\n"
VALID_TXT = "Hello world. This is plain text.\n"

DOCX_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 30


def _docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w:document>hi</w:document>")
    return buf.getvalue()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend_client(monkeypatch):
    """Client that presents AGENT_BACKEND_SECRET — skips licence check."""
    monkeypatch.setattr(settings, "agent_backend_secret", "test-secret")
    return TestClient(api.app, raise_server_exceptions=False), {"X-Agent-Backend-Secret": "test-secret"}


class StubRedisClient:
    """The few Redis hash reads/writes the licence check makes, in memory."""

    def __init__(self):
        self.store: dict[str, dict[str, str]] = {}
        self.heartbeats: list[str] = []

    def hgetall(self, key):
        return dict(self.store.get(key, {}))

    def hset(self, key, *args, **kwargs):
        self.heartbeats.append(key)
        return 1


def _licence(**fields) -> dict[str, str]:
    return {"isValid": "true", "status": "active", "expires_at": "2099-01-01T00:00:00Z", **fields}


@pytest.fixture
def licences(monkeypatch):
    """
    The real RedisManager.validate_license / get_user_api_keys over an in-memory client, so the real
    licence rules (blocked, expired, past-due, outage) are what the upload route is checked against.
    """
    monkeypatch.setattr(settings, "agent_backend_secret", "")
    client = StubRedisClient()
    client.store = {
        "license_key:valid-licence": _licence(user_id="u1"),
        "license_key:blocked-licence": _licence(user_id="u2", editor_blocked="true"),
        "license_key:expired-licence": _licence(user_id="u3", expires_at="2020-01-01T00:00:00Z"),
        "license_key:inactive-licence": _licence(user_id="u4", status="suspended"),
        "license_key:pastdue-licence": _licence(user_id="u5", subscription_status="past_due", requires_user_api_keys="false"),
        "license_key:pastdue-byok": _licence(user_id="u6", subscription_status="past_due", requires_user_api_keys="true"),
    }
    manager = object.__new__(RedisManager)
    manager.redis_url = "stub://"
    manager._license_user_cache = {}
    manager._cache_lock = threading.Lock()
    manager.redis_client = client
    monkeypatch.setattr(api, "redis_manager", manager)
    return TestClient(api.app, raise_server_exceptions=False), client


def _upload(client, content, filename="doc.md", content_type="text/markdown",
            api_key="backend", headers=None):
    """POST multipart upload helper."""
    return client.post(
        "/agent/documents/upload",
        data={"api_key": api_key},
        files={"file": (filename, io.BytesIO(content if isinstance(content, bytes)
                                              else content.encode("utf-8")), content_type)},
        headers=headers or {},
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_markdown_file_returns_text_and_sha256(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, headers=hdrs)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["text"] == VALID_MD
        assert body["char_count"] == len(VALID_MD)
        assert body["sha256"] == _sha256(VALID_MD)
        assert body["filename"] == "doc.md"
        assert body["word_count"] > 0
        assert body["line_count"] == len(VALID_MD.splitlines())
        assert body["headings"] == [{"title": "Hello", "level": 1}]

    def test_txt_file_accepted(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, VALID_TXT, filename="notes.txt",
                       content_type="text/plain", headers=hdrs)
        assert resp.status_code == 200
        assert resp.json()["text"] == VALID_TXT

    def test_markdown_extension_accepted(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, filename="readme.markdown",
                       content_type="text/plain", headers=hdrs)
        assert resp.status_code == 200

    def test_bom_is_stripped(self, backend_client):
        client, hdrs = backend_client
        raw = b"\xef\xbb\xbf" + "# Title\n\nBody.\n".encode("utf-8")      # real BOM bytes EF BB BF
        resp = _upload(client, raw, headers=hdrs)
        assert resp.status_code == 200
        assert resp.json()["text"] == "# Title\n\nBody.\n"

    def test_double_bom_is_stripped(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, b"\xef\xbb\xbf\xef\xbb\xbf# Title\n", headers=hdrs)
        assert resp.status_code == 200
        assert resp.json()["text"] == "# Title\n"

    def test_response_has_request_id_header(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, headers=hdrs)
        assert "x-request-id" in resp.headers

    def test_octet_stream_content_type_accepted_for_md_extension(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, content_type="application/octet-stream", headers=hdrs)
        assert resp.status_code == 200

    @pytest.mark.parametrize("text", [
        "# Café ☕ — naïve “quotes” 日本語 🚀\n\nemoji 👩‍👩‍👧‍👦 and\ttabs\n",
        "# Title\r\n\r\nCRLF line one\r\nline two\r\n",
        "no trailing newline",
        "\n\n\n# leading blank lines\nbody",
    ])
    def test_text_round_trips_exactly(self, backend_client, text):
        client, hdrs = backend_client
        body = _upload(client, text, headers=hdrs).json()
        assert body["text"] == text and body["char_count"] == len(text) and body["sha256"] == _sha256(text)

    def test_upload_and_apply_keep_yaml_and_table_alignment(self, backend_client):
        client, headers = backend_client
        source = ("---\r\n{title: Quarterly report, tags: [docs, finance]}\r\n---\r\n\r\n"
                  "# Sales\r\n\r\n| Name | Units | Total |\r\n| :--- | :---: | ---: |\r\n| A | 12 | 19 |\r\n")
        upload = _upload(client, source, headers=headers).json()
        assert upload["features"] == ["front_matter"]
        assert upload["headings"] == [{"title": "Sales", "level": 1}]
        assert upload["text"] == source

        metadata, heading, _ = [block for block in document_blocks(source) if block.kind != "blank"]
        def apply(block, after):
            patch = {"block_id": block.id, "start_utf16": utf16_length(source[:block.start]),
                     "end_utf16": utf16_length(source[:block.end]), "before": block.text, "after": after, "reason": "Test edit"}
            return client.post("/agent/documents/apply", headers=headers, json={
                "document_text": upload["text"], "base_sha256": upload["sha256"], "patches": [patch]})

        assert apply(metadata, metadata.text.replace("Quarterly report", "Other report")).status_code == 400
        response = apply(heading, "# Quarterly sales\r\n")
        assert response.status_code == 200, response.text
        edited = source.replace("# Sales", "# Quarterly sales")
        assert response.json() == {"document_text": edited, "result_sha256": _sha256(edited)}

    def test_output_is_accepted_by_document_context_with_the_same_hash(self, backend_client):
        """The handoff promises the returned text is ready for document_context.text and the hash matches source_sha256."""
        client, hdrs = backend_client
        body = _upload(client, "\ufeff# Plan\r\n\r\nShip it ☕\r\n".encode("utf-8"), headers=hdrs).json()
        prepared = prepare_context(DocumentContext(text=body["text"]), "What is the plan?", "ask")
        assert prepared.metadata.source_sha256 == body["sha256"]

    def test_headings_skip_code_fences_and_hashtags(self, backend_client):
        client, hdrs = backend_client
        md = "# Real\n\n```bash\n# comment in code\n```\n\nSetext\n------\n\n#hashtag\n"
        body = _upload(client, md, headers=hdrs).json()
        assert body["headings"] == [{"title": "Real", "level": 1}, {"title": "Setext", "level": 2}]


# ---------------------------------------------------------------------------
# Auth: /agent/email/image's resolver, so the licence errors are the /agent/stream ones
# ---------------------------------------------------------------------------

class TestAuth:
    def test_valid_licence_key_accepted(self, licences):
        client, _ = licences
        resp = _upload(client, VALID_MD, api_key="valid-licence")
        assert resp.status_code == 200

    def test_valid_licence_records_a_heartbeat_like_other_licence_routes(self, licences):
        client, redis_client = licences
        _upload(client, VALID_MD, api_key="valid-licence")
        assert redis_client.heartbeats == ["license_key:valid-licence"]

    def test_unknown_licence_returns_403(self, licences):
        client, _ = licences
        resp = _upload(client, VALID_MD, api_key="no-such-key")
        assert resp.status_code == 403
        assert resp.json() == {"error": "Invalid license: License key not found"}

    @pytest.mark.parametrize("key, message", [
        ("expired-licence", "Invalid license: License has expired"),
        ("inactive-licence", "Invalid license: License is suspended"),
    ])
    def test_expired_or_inactive_licence_returns_403(self, licences, key, message):
        client, _ = licences
        resp = _upload(client, VALID_MD, api_key=key)
        assert resp.status_code == 403 and resp.json() == {"error": message}

    def test_editor_blocked_licence_gets_the_unwrapped_message_and_code(self, licences):
        """Same body as /agent/stream and /agent/email/*: the sentence as-is plus EDITOR_ACCESS_BLOCKED."""
        client, _ = licences
        resp = _upload(client, VALID_MD, api_key="blocked-licence")
        body = resp.json()
        assert resp.status_code == 403
        assert body["code"] == "EDITOR_ACCESS_BLOCKED"
        assert body["error"].startswith("Your access to the Eddyter editor has been suspended")
        assert "Invalid" not in body["error"]

    def test_past_due_managed_plan_returns_402(self, licences):
        client, _ = licences
        resp = _upload(client, VALID_MD, api_key="pastdue-licence")
        assert resp.status_code == 402
        assert "past due" in resp.json()["error"]

    def test_past_due_byok_plan_is_still_allowed(self, licences):
        client, _ = licences
        assert _upload(client, VALID_MD, api_key="pastdue-byok").status_code == 200

    def test_wrong_backend_secret_falls_back_to_the_licence_check(self, licences, monkeypatch):
        client, _ = licences
        monkeypatch.setattr(settings, "agent_backend_secret", "right-secret")
        assert _upload(client, VALID_MD, api_key="valid-licence", headers={"X-Agent-Backend-Secret": "wrong"}).status_code == 200
        assert _upload(client, VALID_MD, api_key="bogus", headers={"X-Agent-Backend-Secret": "wrong"}).status_code == 403

    def test_no_redis_manager_returns_503(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_backend_secret", "")
        monkeypatch.setattr(api, "redis_manager", None)
        client = TestClient(api.app, raise_server_exceptions=False)
        resp = _upload(client, VALID_MD, api_key="any")
        assert resp.status_code == 503

    def test_backend_secret_bypasses_redis(self, monkeypatch):
        """Backend secret must work even if Redis is None."""
        monkeypatch.setattr(settings, "agent_backend_secret", "secret-xyz")
        monkeypatch.setattr(api, "redis_manager", None)
        client = TestClient(api.app, raise_server_exceptions=False)
        resp = _upload(client, VALID_MD, headers={"X-Agent-Backend-Secret": "secret-xyz"})
        assert resp.status_code == 200

    @pytest.mark.parametrize("fields", [{}, {"api_key": ""}])
    def test_missing_or_empty_api_key_is_400(self, backend_client, fields):
        client, hdrs = backend_client
        resp = client.post("/agent/documents/upload", data=fields,
                           files={"file": ("a.md", io.BytesIO(b"# ok\n"), "text/markdown")}, headers=hdrs)
        assert resp.status_code == 400 and "api_key" in resp.json()["error"]

    def test_auth_runs_before_the_file_is_inspected(self, licences):
        """A caller without a valid licence learns nothing about the file: a PDF still gets the 403, not a 415."""
        client, _ = licences
        resp = _upload(client, PDF_BYTES, filename="a.pdf", content_type="application/pdf", api_key="bogus")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# File type validation
# ---------------------------------------------------------------------------

class TestFileType:
    @pytest.mark.parametrize("filename, content, content_type", [
        ("doc.pdf", PDF_BYTES, "application/pdf"),
        ("doc.pdf", PDF_BYTES, "application/octet-stream"),        # the content type the router itself allow-lists
        ("doc.docx", _docx_bytes(), DOCX_CT),
        ("doc.docx", _docx_bytes(), "application/octet-stream"),
        ("pic.png", PNG_BYTES, "image/png"),
        ("pic.png", PNG_BYTES, "application/octet-stream"),
        ("data.json", b'{"a": 1}', "application/json"),
        ("data.json", b'{"a": 1}', "application/octet-stream"),
        ("data.csv", b"a,b\n1,2\n", "text/csv"),
        ("page.html", b"<html><body>hi</body></html>", "text/html"),
        ("page.html", b"<html><body>hi</body></html>", "application/octet-stream"),
        ("notes", VALID_MD.encode(), "application/octet-stream"),    # no extension: only a text content type will do
        ("notes.txt", VALID_TXT.encode(), "application/pdf"),        # the content type contradicts the name
        ("notes.md", VALID_MD.encode(), "image/png"),
    ])
    def test_unsupported_type_rejected_415_and_names_the_file(self, backend_client, filename, content, content_type):
        client, hdrs = backend_client
        resp = _upload(client, content, filename=filename, content_type=content_type, headers=hdrs)
        assert resp.status_code == 415, resp.text
        assert ".md and .txt" in resp.json()["error"] or "not plain text" in resp.json()["error"]

    def test_pdf_with_no_content_type_header_is_rejected(self, backend_client):
        client, hdrs = backend_client
        boundary = "XXBOUNDARYXX"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="api_key"\r\n\r\nx\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n').encode() \
               + PDF_BYTES + f"\r\n--{boundary}--\r\n".encode()
        resp = client.post("/agent/documents/upload", content=body,
                           headers={**hdrs, "Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert resp.status_code == 415

    @pytest.mark.parametrize("content, kind", [
        (PDF_BYTES, "PDF"),
        (_docx_bytes(), "ZIP-based"),
        (PNG_BYTES, "PNG"),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(range(1, 200)), "JPEG"),
        (b"{\\rtf1\\ansi Hello}", "RTF"),
    ])
    def test_binary_file_renamed_to_md_is_caught_by_its_magic_bytes(self, backend_client, content, kind):
        client, hdrs = backend_client
        resp = _upload(client, content, filename="notes.md", content_type="text/markdown", headers=hdrs)
        assert resp.status_code == 415
        assert kind in resp.json()["error"] and "not plain text" in resp.json()["error"]

    def test_no_extension_with_text_plain_accepted(self, backend_client):
        """No extension but text/plain content-type → accept (allow markdown without extension)."""
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, filename="Makefile",
                       content_type="text/plain", headers=hdrs)
        assert resp.status_code == 200

    @pytest.mark.parametrize("filename, content_type", [
        ("NOTES.MD", "text/markdown"),
        ("C:\\Users\\me\\Documents\\notes.md", "application/octet-stream"),
        ("notes.md", "text/markdown; charset=utf-8"),
        ("notes.md", "text/x-markdown"),
        ("notes.md", "application/markdown"),
        ("notes.txt", "text/plain; charset=utf-8"),
        ("отчёт.md", "text/markdown"),
    ])
    def test_ordinary_text_files_are_accepted(self, backend_client, filename, content_type):
        client, hdrs = backend_client
        resp = _upload(client, VALID_MD, filename=filename, content_type=content_type, headers=hdrs)
        assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Encoding: strict UTF-8, never a silent guess
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_cp1252_text_is_refused_not_decoded_into_control_characters(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "He said “hello” — it’s café".encode("cp1252"), filename="win.txt",
                       content_type="text/plain", headers=hdrs)
        assert resp.status_code == 422
        assert "UTF-8" in resp.json()["error"]

    def test_other_legacy_encodings_are_refused_not_turned_into_mojibake(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "Zażółć gęślą jaźń".encode("iso-8859-2"), headers=hdrs)
        assert resp.status_code == 422 and "UTF-8" in resp.json()["error"]

    @pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be"])
    def test_utf16_gets_a_specific_message(self, backend_client, codec):
        client, hdrs = backend_client
        raw = ("\ufeff# Title\nHello\n").encode(codec) if codec != "utf-16" else "# Title\nHello\n".encode("utf-16")
        resp = _upload(client, raw, filename="u16.txt", content_type="text/plain", headers=hdrs)
        assert resp.status_code == 422 and "UTF-16" in resp.json()["error"]

    def test_arbitrary_bytes_without_nul_are_refused(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, bytes(range(1, 256)) * 4, filename="blob.txt", content_type="text/plain", headers=hdrs)
        assert resp.status_code == 422

    def test_valid_utf8_in_many_scripts_is_untouched(self, backend_client):
        client, hdrs = backend_client
        text = "# 日本語\n\nΕλληνικά, Русский, العربية, עברית, हिन्दी 🚀\n"
        assert _upload(client, text, headers=hdrs).json()["text"] == text


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

class TestContentValidation:
    def test_empty_file_returns_422(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "", headers=hdrs)
        assert resp.status_code == 422
        assert "empty" in resp.json()["error"].lower()

    def test_whitespace_only_returns_422(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "   \n\n\t  ", headers=hdrs)
        assert resp.status_code == 422

    def test_nul_character_returns_422(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "Hello\x00World", headers=hdrs)
        assert resp.status_code == 422
        assert "NUL" in resp.json()["error"]

    def test_html_content_returns_422(self, backend_client):
        client, hdrs = backend_client
        html = "<!DOCTYPE html><html><body><p>Hello</p></body></html>"
        resp = _upload(client, html, headers=hdrs)
        assert resp.status_code == 422
        assert "HTML" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Size guards
# ---------------------------------------------------------------------------

class TestSizeGuards:
    def test_file_over_400kb_bytes_returns_413(self, backend_client):
        client, hdrs = backend_client
        big = b"a" * 401_000
        resp = _upload(client, big, headers=hdrs)
        assert resp.status_code == 413
        assert "large" in resp.json()["error"].lower()

    def test_size_in_the_message_is_the_real_size(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, b"a" * 410_000, headers=hdrs)
        assert resp.status_code == 413 and "410,000" in resp.json()["error"]

    def test_file_over_100k_chars_returns_413(self, backend_client):
        """A file that fits in bytes but has >100 000 chars (e.g. ASCII)."""
        client, hdrs = backend_client
        big = "x" * 100_001
        resp = _upload(client, big, headers=hdrs)
        assert resp.status_code == 413
        assert "long" in resp.json()["error"].lower()

    def test_exactly_100k_chars_accepted(self, backend_client):
        client, hdrs = backend_client
        text = "a" * 100_000
        resp = _upload(client, text, filename="big.txt", content_type="text/plain", headers=hdrs)
        assert resp.status_code == 200
        assert resp.json()["char_count"] == 100_000

    def test_100k_four_byte_chars_is_exactly_the_byte_limit_and_accepted(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "🚀" * 100_000, headers=hdrs)       # 400,000 bytes, 100,000 chars
        assert resp.status_code == 200 and resp.json()["char_count"] == 100_000

    def test_one_char_over_with_four_byte_chars_is_rejected(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, "🚀" * 100_001, headers=hdrs)       # 400,004 bytes
        assert resp.status_code == 413


async def _call_app(headers: dict[str, str], chunks: list[bytes]):
    """Drive the ASGI app directly and report how much of the request body it pulled before answering."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/agent/documents/upload", "raw_path": b"/agent/documents/upload", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    pending = iter(chunks)
    state = {"receives": 0, "bytes": 0}

    async def receive():
        state["receives"] += 1
        try:
            body = next(pending)
        except StopIteration:
            return {"type": "http.disconnect"}
        state["bytes"] += len(body)
        return {"type": "http.request", "body": body, "more_body": True}

    sent = []

    async def send(message):
        sent.append(message)

    await api.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), body, state


class TestBodyCapBeforeParsing:
    """The size guard has to act before FastAPI receives and spools the multipart body, and before auth."""

    async def test_oversized_content_length_is_refused_without_reading_any_body(self):
        status, headers, body, state = await _call_app(
            {"content-type": "multipart/form-data; boundary=x", "content-length": "150000000"},
            [b"x" * 65536] * 4,
        )
        assert status == 413
        assert state["receives"] == 0 and state["bytes"] == 0
        assert b"File too large" in body
        assert headers.get(b"connection") == b"close"

    async def test_chunked_upload_without_content_length_is_cut_off_at_the_cap(self):
        first = b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\nContent-Type: text/markdown\r\n\r\n'
        chunks = [first + b"a" * 65536] + [b"a" * 65536] * 200       # ~13 MB if it were all read
        status, _, body, state = await _call_app({"content-type": "multipart/form-data; boundary=x"}, chunks)
        assert status == 413
        assert state["bytes"] < 600_000, f"server pulled {state['bytes']:,} bytes before refusing"
        assert b"File too large" in body

    async def test_a_lying_content_length_is_still_capped_by_the_byte_count(self):
        first = b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\nContent-Type: text/markdown\r\n\r\n'
        chunks = [first + b"a" * 65536] + [b"a" * 65536] * 200
        status, _, _, state = await _call_app(
            {"content-type": "multipart/form-data; boundary=x", "content-length": "1000"}, chunks)
        assert status == 413 and state["bytes"] < 600_000

    def test_the_cap_applies_to_unauthenticated_callers_too(self, licences):
        client, _ = licences
        resp = _upload(client, b"a" * 5_000_000, api_key="not-a-licence")
        assert resp.status_code == 413              # refused on size, before any licence lookup

    def test_a_file_just_over_the_limit_still_gets_the_precise_message(self, backend_client):
        client, hdrs = backend_client
        resp = _upload(client, b"a" * 400_001, headers=hdrs)
        assert resp.status_code == 413 and "400,001" in resp.json()["error"]


# ---------------------------------------------------------------------------
# The event loop must stay free
# ---------------------------------------------------------------------------

class TestEventLoop:
    def _record_threads(self, monkeypatch):
        seen = []
        real = upload_router._extract_document

        def spy(raw, filename, declared_size=None):
            seen.append(threading.current_thread().name)
            return real(raw, filename, declared_size)

        monkeypatch.setattr(upload_router, "_extract_document", spy)
        return seen

    def test_large_files_are_extracted_on_the_dedicated_pool_not_the_event_loop(self, backend_client, monkeypatch):
        client, hdrs = backend_client
        seen = self._record_threads(monkeypatch)
        resp = _upload(client, "x" * (settings.document_upload_small_file_bytes + 1), headers=hdrs)
        assert resp.status_code == 200
        assert len(seen) == 1 and re.fullmatch(r"doc_upload_\d+", seen[0]), seen

    def test_small_files_use_the_small_file_thread(self, backend_client, monkeypatch):
        client, hdrs = backend_client
        seen = self._record_threads(monkeypatch)
        assert _upload(client, VALID_MD, headers=hdrs).status_code == 200
        assert len(seen) == 1 and re.fullmatch(r"doc_upload_small_\d+", seen[0]), seen

    def test_the_boundary_is_in_bytes(self, backend_client, monkeypatch):
        client, hdrs = backend_client
        monkeypatch.setattr(settings, "document_upload_small_file_bytes", 1_000)
        seen = self._record_threads(monkeypatch)
        _upload(client, "a" * 1_000, headers=hdrs)
        _upload(client, "a" * 1_001, headers=hdrs)
        _upload(client, "🚀" * 300, headers=hdrs)                      # 300 characters but 1,200 bytes
        assert [n.startswith("doc_upload_small") for n in seen] == [True, False, False]


# ---------------------------------------------------------------------------
# Concurrency cap, small-file lane, queue deadline
# ---------------------------------------------------------------------------

BACKEND_HEADERS = {"X-Agent-Backend-Secret": "test-secret"}


async def _aupload(client: httpx.AsyncClient, content: bytes, name: str = "d.md"):
    return await client.post("/agent/documents/upload", data={"api_key": "x"},
                             files={"file": (name, content, "text/markdown")}, headers=BACKEND_HEADERS)


def _async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver", timeout=30)


class TestConcurrencyCap:
    LARGE = b"# Large\n\n" + b"a" * 20_000                           # over the 16,000-byte small-file limit

    @pytest.fixture(autouse=True)
    def _secret(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_backend_secret", "test-secret")

    @pytest.fixture
    def gate(self):
        """Blocks worker threads until set. Always released at teardown: the pools are module-level, so a failing test must not leave them blocked."""
        event = threading.Event()
        yield event
        event.set()

    async def test_large_extractions_never_run_more_than_the_cap_at_once(self, monkeypatch):
        cap = settings.document_upload_max_concurrent
        active, peak, lock = 0, 0, threading.Lock()
        real = upload_router._extract_document

        def slow(raw, filename, declared_size=None):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            try:
                return real(raw, filename, declared_size)
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(upload_router, "_extract_document", slow)
        async with _async_client() as client:
            responses = await asyncio.gather(*[_aupload(client, self.LARGE) for _ in range(3 * cap + 2)])
        assert [r.status_code for r in responses] == [200] * (3 * cap + 2)
        assert 1 <= peak <= cap, f"{peak} extractions ran at once, cap is {cap}"

    async def test_a_small_upload_does_not_wait_behind_large_ones(self, monkeypatch, gate):
        cap = settings.document_upload_max_concurrent
        real = upload_router._extract_document

        def blocked_when_large(raw, filename, declared_size=None):
            if len(raw) > settings.document_upload_small_file_bytes:
                gate.wait(10)
            return real(raw, filename, declared_size)

        monkeypatch.setattr(upload_router, "_extract_document", blocked_when_large)
        async with _async_client() as client:
            large = [asyncio.create_task(_aupload(client, self.LARGE)) for _ in range(cap + 2)]   # fills every large slot and queues two more
            await asyncio.sleep(0.3)
            assert not any(t.done() for t in large)
            small = await asyncio.wait_for(_aupload(client, VALID_MD.encode()), timeout=5)       # must not wait for the gate
            assert small.status_code == 200 and small.json()["text"] == VALID_MD
            assert not any(t.done() for t in large)
            gate.set()
            done = await asyncio.gather(*large)
        assert [r.status_code for r in done] == [200] * (cap + 2)

    async def test_a_request_that_waits_past_the_deadline_gets_503_and_the_slots_recover(self, monkeypatch, gate):
        cap = settings.document_upload_max_concurrent
        monkeypatch.setattr(settings, "document_upload_queue_wait_seconds", 0.2)
        real = upload_router._extract_document

        def blocked(raw, filename, declared_size=None):
            gate.wait(10)
            return real(raw, filename, declared_size)

        monkeypatch.setattr(upload_router, "_extract_document", blocked)
        async with _async_client() as client:
            holding = [asyncio.create_task(_aupload(client, self.LARGE)) for _ in range(cap)]    # every large slot busy
            await asyncio.sleep(0.1)
            late = asyncio.create_task(_aupload(client, self.LARGE))                            # queued, deadline 0.2 s away
            await asyncio.sleep(0.5)
            gate.set()
            first = await asyncio.gather(*holding)
            late_response = await late
            assert late_response.status_code == 503
            assert late_response.headers["retry-after"] == "2"
            assert "busy" in late_response.json()["error"].lower()
            assert [r.status_code for r in first] == [200] * cap                                # the ones already running finish normally
            monkeypatch.setattr(upload_router, "_extract_document", real)                       # and nothing is left blocked
            again = await asyncio.gather(*[_aupload(client, self.LARGE) for _ in range(cap + 1)])
        assert [r.status_code for r in again] == [200] * (cap + 1)

    async def test_validation_errors_from_the_worker_thread_keep_their_status_and_free_the_slot(self, monkeypatch):
        cap = settings.document_upload_max_concurrent
        async with _async_client() as client:
            bad = await asyncio.gather(*[_aupload(client, b"\xff\xfe\x00" + b"a" * 20_000, "u.txt") for _ in range(3 * cap)])
            good = await _aupload(client, self.LARGE)
        assert {r.status_code for r in bad} == {422}                                            # UTF-16 BOM, raised inside the worker
        assert good.status_code == 200


# ---------------------------------------------------------------------------
# Settings: defaults, environment overrides, bounds, and that they are documented
# ---------------------------------------------------------------------------

UPLOAD_SETTINGS = {
    # variable: (field, override text, override value, documented default)
    "DOCUMENT_UPLOAD_MAX_CONCURRENT": ("document_upload_max_concurrent", "4", 4, 2),
    "DOCUMENT_UPLOAD_SMALL_FILE_BYTES": ("document_upload_small_file_bytes", "8000", 8000, 16000),
    "DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS": ("document_upload_queue_wait_seconds", "5", 5.0, 10.0),
}
ROOT = Path(__file__).resolve().parents[1]


def _fresh_settings(monkeypatch, **env):
    for name in UPLOAD_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


class TestUploadSettings:
    def test_defaults_are_the_documented_ones(self, monkeypatch):
        built = _fresh_settings(monkeypatch)
        assert {f: getattr(built, f) for f, *_ in UPLOAD_SETTINGS.values()} == {f: d for f, _, _, d in UPLOAD_SETTINGS.values()}

    @pytest.mark.parametrize("variable, spec", UPLOAD_SETTINGS.items())
    def test_each_setting_can_be_overridden_from_the_environment(self, monkeypatch, variable, spec):
        field, raw, value, _ = spec
        assert getattr(_fresh_settings(monkeypatch, **{variable: raw}), field) == value

    @pytest.mark.parametrize("variable, bad", [
        ("DOCUMENT_UPLOAD_MAX_CONCURRENT", "0"), ("DOCUMENT_UPLOAD_MAX_CONCURRENT", "99"), ("DOCUMENT_UPLOAD_MAX_CONCURRENT", ""),
        ("DOCUMENT_UPLOAD_SMALL_FILE_BYTES", "10"), ("DOCUMENT_UPLOAD_SMALL_FILE_BYTES", "999999999"),
        ("DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS", "0"), ("DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS", "600"), ("DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS", ""),
    ])
    def test_a_bad_or_blank_value_stops_start_up(self, monkeypatch, variable, bad):
        with pytest.raises(ValidationError):
            _fresh_settings(monkeypatch, **{variable: bad})

    def test_env_example_lists_every_setting_commented_out(self):
        text = (ROOT / ".env.example").read_text()
        for variable, (_, _, _, default) in UPLOAD_SETTINGS.items():
            assert re.search(rf"^# {variable}=\S+$", text, re.M), variable
            assert not re.search(rf"^{variable}=\s*$", text, re.M), variable

    def test_the_handoff_documents_every_setting_with_its_default(self):
        text = (ROOT / "docs" / "DOCUMENT_UPLOAD_API_HANDOFF.md").read_text()
        for variable, (_, _, _, default) in UPLOAD_SETTINGS.items():
            row = next((ln for ln in text.splitlines() if ln.startswith(f"| `{variable}`")), None)
            assert row is not None, variable
            assert f"`{default}`" in row or f"`{default:g}`" in row or (isinstance(default, float) and f"`{default}`" in row), (variable, row)


# ---------------------------------------------------------------------------
# Idempotency: same file → same sha256
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_file_produces_same_sha256(self, backend_client):
        client, hdrs = backend_client
        r1 = _upload(client, VALID_MD, headers=hdrs)
        r2 = _upload(client, VALID_MD, headers=hdrs)
        assert r1.json()["sha256"] == r2.json()["sha256"]
