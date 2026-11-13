"""
Fetch a public web address for import, safely.

A server that fetches addresses people give it can be turned against its own network (SSRF): to reach 127.0.0.1, a private subnet, or a cloud
metadata service. What this does about that, and about a hostile site:

  * only http/https, no credentials in the address, only ports 80, 443, 8080 and 8443, an address of at most 2048 characters;
  * the name is resolved ONCE per connection and the connection goes to exactly the addresses that were checked (so a DNS answer that changes
    between the check and the connect cannot redirect it); every address of the name must be public (src/net_safety.py), or none is used;
  * redirects are followed by hand, at most 5, and every hop is checked again like the first address;
  * no cookies, no proxies from the environment, no credentials of ours are ever sent;
  * the body is streamed and counted AFTER decompression, so a small gzip bomb stops at the cap; the whole fetch also has a time limit, so a
    site that drips one byte a second is cut off;
  * the answer must be a web page, a PDF, a Word file or plain text; anything else (images, archives, JSON, video) is refused before it is read.

Everything raises ImportRefused with a message for the user. The messages do not say whether a refused address exists on a private network.
"""
from __future__ import annotations

import http.cookiejar
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from .errors import ImportRefused
from ..net_safety import resolve_public               # a module attribute on purpose: tests replace it, the backend reads it at call time

MAX_URL_CHARS = 2048
ALLOWED_PORTS = {80, 443, 8080, 8443}
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 6.0
READ_TIMEOUT = 10.0
TOTAL_TIMEOUT = 25.0                  # a read that has already started can finish, so the real worst case is TOTAL_TIMEOUT + READ_TIMEOUT
USER_AGENT = "Mozilla/5.0 (compatible; DocumentImport/1.0)"
ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,application/vnd.openxmlformats-officedocument.wordprocessingml.document;q=0.9,text/plain;q=0.8,*/*;q=0.1"

_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
_NOT_FETCHABLE = "This address could not be fetched. It must be a public web address (http or https)."


class _RedactQuery(logging.Filter):
    """httpx logs "HTTP Request: GET <full address> ..." at INFO. A query string can hold a token or a signed-link secret, so it is cut from that line."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 2:
            text = str(args[1])
            if "?" in text:
                record.args = (args[0], text.split("?", 1)[0] + "?...", *args[2:])
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(f, _RedactQuery) for f in _httpx_logger.filters):
    _httpx_logger.addFilter(_RedactQuery())


@dataclass
class Fetched:
    data: bytes
    kind: str                         # "html" | "pdf" | "docx" | "text"
    final_url: str
    content_type: str
    charset: Optional[str]
    redirects: int


def _refuse(status: int, code: str, message: str) -> ImportRefused:
    return ImportRefused(status, code, message)


def check_url(url: str):
    """The parsed address, or ImportRefused. Does no network I/O (the host is checked when the connection is made)."""
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_CHARS or any(c in url for c in "\r\n\t\x00 "):
        raise _refuse(422, "invalid_url", "Enter a complete web address that starts with http:// or https://.")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise _refuse(422, "invalid_url", "Enter a complete web address that starts with http:// or https://.") from None
    if parts.scheme.lower() not in ("http", "https") or not host:
        raise _refuse(422, "invalid_url", "Enter a complete web address that starts with http:// or https://.")
    if parts.username is not None or parts.password is not None:
        raise _refuse(422, "invalid_url", "Addresses with a user name or password cannot be imported.")
    if (port or (443 if parts.scheme.lower() == "https" else 80)) not in ALLOWED_PORTS:
        raise _refuse(422, "url_not_fetchable", _NOT_FETCHABLE)
    try:
        host.encode("idna")
    except UnicodeError:
        raise _refuse(422, "invalid_url", "That address is not valid.") from None
    return parts


class _PublicOnlyBackend(httpcore.SyncBackend):
    """Connects only to addresses that were just checked to be public: the lookup and the connection use the same answer."""

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        from . import fetch as _self                           # read through the module so a replaced resolve_public is honoured
        addresses = _self.resolve_public(host)
        if not addresses:
            raise httpcore.ConnectError(f"blocked: {host}")
        last: Optional[Exception] = None
        for address in addresses:
            try:
                return super().connect_tcp(address, port, timeout, local_address, socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last = exc
        raise last or httpcore.ConnectError(f"could not connect: {host}")


class _NoCookieJar(http.cookiejar.CookieJar):
    """A site may set a cookie during a redirect chain; none is stored, so none is sent back (the same choice as the page reader in text_processing_agent)."""

    def set_cookie(self, cookie, *args, **kwargs):
        return None


def make_client() -> httpx.Client:
    transport = httpx.HTTPTransport(limits=httpx.Limits(max_connections=1, max_keepalive_connections=0), trust_env=False, retries=0)
    # httpx has no public hook for the network backend; this assignment is tested (tests/document_import/test_fetch.py) so an httpx upgrade that
    # moves it fails loudly instead of silently dropping the address check.
    pool = transport._pool
    if not isinstance(pool, httpcore.ConnectionPool):
        raise RuntimeError("httpx transport layout changed: cannot pin DNS")
    pool._network_backend = _PublicOnlyBackend()
    return httpx.Client(transport=transport, follow_redirects=False, trust_env=False, cookies=_NoCookieJar(),
                        timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                        headers={"User-Agent": USER_AGENT, "Accept": ACCEPT, "Accept-Language": "en-US,en;q=0.8"})


def _kind(content_type: str, head: bytes, final_url: str) -> Optional[str]:
    ct = content_type.split(";")[0].strip().lower()
    sniffed = ("pdf" if head.lstrip()[:5] == b"%PDF-" else "docx" if head.startswith(b"PK\x03\x04") else None)
    if ct == "application/pdf":
        return "pdf"
    if ct == _DOCX:
        return "docx"
    if ct in _HTML_TYPES:
        return "html"
    if ct in _TEXT_TYPES:
        return "text"
    if ct in ("", "application/octet-stream", "binary/octet-stream", "application/download", "application/x-download"):
        if sniffed:
            return sniffed
        path = urlsplit(final_url).path.lower()
        if path.endswith((".md", ".markdown", ".txt")):
            return "text"
        if path.endswith((".html", ".htm")) or re.match(rb"\s*(<!doctype html|<html|<head|<body)", head[:200], re.I):
            return "html"
        return "text" if ct == "" and head and b"\x00" not in head[:1024] else None
    return None


def fetch_url(url: str, *, max_html_bytes: int = 5 * 1024 * 1024, max_file_bytes: int = 15 * 1024 * 1024,
              total_timeout: float = TOTAL_TIMEOUT) -> Fetched:
    deadline = time.monotonic() + total_timeout
    current = url.strip()
    check_url(current)
    redirects = 0
    with make_client() as client:
        while True:
            try:
                with client.stream("GET", current) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location", "")
                        if not location:
                            raise _refuse(422, "fetch_status", "The site sent a redirect without saying where.")
                        redirects += 1
                        if redirects > MAX_REDIRECTS:
                            raise _refuse(422, "too_many_redirects", "The address redirects too many times.")
                        current = urljoin(current, location)
                        check_url(current)                                       # scheme, credentials and port of the next hop
                        continue
                    return _read(response, current, redirects, deadline, max_html_bytes, max_file_bytes)
            except ImportRefused:
                raise
            except httpx.TimeoutException:
                raise _refuse(504, "fetch_timeout", "The site took too long to answer.") from None
            except httpx.ConnectError as exc:
                if "CERTIFICATE" in str(exc).upper() or "SSL" in str(exc).upper():
                    raise _refuse(422, "tls_failed", "The site's security certificate could not be verified.") from None
                raise _refuse(422, "url_not_fetchable", _NOT_FETCHABLE) from None
            except (httpx.HTTPError, httpcore.ConnectError, OSError, UnicodeError):
                raise _refuse(502, "fetch_failed", "The site could not be reached or sent an invalid answer.") from None


def _read(response: httpx.Response, url: str, redirects: int, deadline: float, max_html: int, max_file: int) -> Fetched:
    status = response.status_code
    if status >= 400:
        if status >= 500:
            raise _refuse(502, "fetch_status", f"The site had a problem and answered with status {status}.")
        hint = " It may need a login, or it may not allow automated readers; neither can be worked around here." if status in (401, 403, 429) else ""
        raise _refuse(422, "fetch_status", f"The site answered with status {status}.{hint}")
    if not 200 <= status < 300:
        raise _refuse(422, "fetch_status", f"The site answered with status {status}.")
    content_type = response.headers.get("content-type", "")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_file and "gzip" not in response.headers.get("content-encoding", "").lower():
        raise _refuse(413, "fetch_too_large", f"The file is larger than the {max_file // (1024 * 1024)} MB limit.")
    buf = bytearray()
    kind: Optional[str] = None
    limit = max_file
    # Decoded bytes, as they arrive: a compressed bomb is counted as what it expands to, and the deadline is looked at after every network read.
    # (iter_bytes(65536) would wait for 64 KB before yielding anything, so a site dripping one byte a second would never be checked.)
    for chunk in response.iter_bytes():
        if time.monotonic() > deadline:
            raise _refuse(504, "fetch_timeout", "The site took too long to send the page.")
        buf += chunk
        if kind is None and len(buf) >= 1024:
            kind, limit = _decide(content_type, bytes(buf[:4096]), url, max_html, max_file)
        if len(buf) > limit:
            raise _refuse(413, "fetch_too_large", f"The page is larger than the {limit // (1024 * 1024)} MB limit.")
    if not bytes(buf).strip():
        raise _refuse(422, "fetch_empty", "The address returned nothing.")
    if kind is None:                                                     # a short answer: decided once it is complete
        kind, limit = _decide(content_type, bytes(buf[:4096]), url, max_html, max_file)
        if len(buf) > limit:
            raise _refuse(413, "fetch_too_large", f"The page is larger than the {limit // (1024 * 1024)} MB limit.")
    m = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    return Fetched(bytes(buf), kind, url, content_type.split(";")[0].strip().lower(), m.group(1) if m else None, redirects)


def _decide(content_type: str, head: bytes, url: str, max_html: int, max_file: int) -> Tuple[str, int]:
    kind = _kind(content_type, head, url)
    if kind is None:
        shown = content_type.split(";")[0].strip() or "unknown type"
        raise _refuse(415, "fetch_unsupported_type", f"This address is not a web page, PDF, Word file or text file (it is {shown}).")
    return kind, (max_html if kind in ("html", "text") else max_file)
