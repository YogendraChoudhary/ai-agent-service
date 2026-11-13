"""
fetch.py: the safe fetcher behind URL import.

A real HTTP server runs on 127.0.0.1 inside the test. By default that address is refused, which is the SSRF defence working against a real
socket. To exercise the rest of the fetcher (redirects, size caps, timeouts, content types) the tests make ONE name, allowed.test, resolve to
127.0.0.1 by replacing fetch.resolve_public; every other name still goes through the real policy.
"""
from __future__ import annotations

import gzip
import json
import socket
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore
import httpx
import pytest

import src.document_import.fetch as fetch
from src.document_import.errors import ImportRefused
from src.net_safety import resolve_public as real_resolve_public
from tests.document_import.builders import column_page, make_docx, make_pdf, p

HTML = b"<html><head><title>T</title></head><body><h1>Hello</h1><p>" + b"word " * 50 + b"</p></body></html>"
PDF = make_pdf([column_page([("B", "Heading", 20), ("R", "Body text here. " * 15, 11)])])
DOCX = make_docx([p("A title", "Title"), p("Some words")])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *a):
        pass

    def reply(self, status=200, body=b"", ctype="text/html; charset=utf-8", headers=None, chunked=False):
        self.send_response(status)
        if ctype is not None:
            self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        Handler.seen.append((self.path, dict(self.headers)))
        path = self.path.split("?")[0]
        port = self.server.server_address[1]
        if path == "/html":
            self.reply(body=HTML)
        elif path == "/pdf":
            self.reply(body=PDF, ctype="application/pdf")
        elif path == "/docx":
            self.reply(body=DOCX, ctype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        elif path == "/octet-pdf":
            self.reply(body=PDF, ctype="application/octet-stream")
        elif path == "/readme.md":
            self.reply(body=b"# Readme\n\nText.\n", ctype="application/octet-stream")
        elif path == "/text":
            self.reply(body="café\n".encode("latin-1"), ctype="text/plain; charset=latin-1")
        elif path == "/png":
            self.reply(body=b"\x89PNG\r\n\x1a\n" + b"\0" * 2000, ctype="image/png")
        elif path == "/json":
            self.reply(body=b'{"a": 1}', ctype="application/json")
        elif path == "/empty":
            self.reply(body=b"", ctype="text/html")
        elif path == "/404":
            self.reply(404, b"nope")
        elif path == "/403":
            self.reply(403, b"login")
        elif path == "/500":
            self.reply(500, b"boom")
        elif path == "/redirect":
            self.reply(302, b"", headers={"Location": "/html"})
        elif path == "/redirect-private":
            self.reply(302, b"", headers={"Location": f"http://127.0.0.1:{port}/html"})
        elif path == "/redirect-metadata":
            self.reply(302, b"", headers={"Location": "http://169.254.169.254/latest/meta-data/"})
        elif path == "/redirect-file":
            self.reply(302, b"", headers={"Location": "file:///etc/passwd"})
        elif path == "/redirect-creds":
            self.reply(302, b"", headers={"Location": f"http://user:pw@allowed.test:{port}/html"})
        elif path == "/redirect-port":
            self.reply(302, b"", headers={"Location": "http://allowed.test:22/"})
        elif path == "/loop":
            self.reply(302, b"", headers={"Location": "/loop"})
        elif path == "/redirect-nolocation":
            self.reply(302, b"")
        elif path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(50 * 1024 * 1024))
            self.end_headers()
            try:
                for _ in range(50 * 16):
                    self.wfile.write(b"a" * 65536)
            except OSError:
                pass
        elif path == "/big-chunked":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(400):
                    self.wfile.write(b"10000\r\n" + b"a" * 65536 + b"\r\n")
            except OSError:
                pass
        elif path == "/gzip-bomb":
            bomb = gzip.compress(b"a" * (200 * 1024 * 1024), 9)
            self.reply(body=bomb, headers={"Content-Encoding": "gzip"})
        elif path == "/gzip-ok":
            self.reply(body=gzip.compress(HTML), headers={"Content-Encoding": "gzip"})
        elif path == "/drip":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"1\r\na\r\n")
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass
        elif path == "/silent":
            time.sleep(8)
        elif path == "/headers":
            self.reply(body=json.dumps(dict(self.headers)).encode(), ctype="text/html")
        elif path == "/set-cookie":
            self.reply(body=HTML, headers={"Set-Cookie": "sid=1; Path=/"})
        elif path == "/cookie-then-redirect":
            self.reply(302, b"", headers={"Set-Cookie": "sid=1; Path=/", "Location": "/headers"})
        else:
            self.reply(404, b"")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):          # a client that hangs up mid-answer is the point of several tests
        pass


@pytest.fixture(scope="module")
def server():
    srv = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


@pytest.fixture
def url(server, monkeypatch):
    port = server.server_address[1]
    monkeypatch.setattr(fetch, "ALLOWED_PORTS", {port, 80, 443, 8080, 8443})
    monkeypatch.setattr(fetch, "resolve_public", lambda host: ["127.0.0.1"] if host == "allowed.test" else real_resolve_public(host))
    Handler.seen.clear()
    return lambda path: f"http://allowed.test:{port}{path}"


def refused(address, **kw) -> ImportRefused:
    with pytest.raises(ImportRefused) as e:
        fetch.fetch_url(address, **kw)
    return e.value


# ---------------------------------------------------------------------------------------------------------------------------------------

class TestAddressPolicy:
    @pytest.mark.parametrize("address", [
        "http://127.0.0.1/", "http://localhost/", "http://[::1]/", "http://0.0.0.0/", "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/", "http://192.168.1.1/", "http://172.16.0.1/", "http://2130706433/", "http://0x7f.0.0.1/", "http://017700000001/",
        "http://[::ffff:127.0.0.1]/", "http://[64:ff9b::a9fe:a9fe]/", "http://[2002:7f00:1::]/", "http://[fe80::1]/", "http://[fd00::1]/",
        "http://metadata.google.internal/", "http://100.64.0.1/", "http://224.0.0.1/",
    ])
    def test_private_and_special_addresses_are_refused(self, address):
        e = refused(address, total_timeout=5)
        assert e.code == "url_not_fetchable" and e.status == 422

    @pytest.mark.parametrize("address", [
        "", "   ", "example.com", "ftp://example.com/", "file:///etc/passwd", "javascript:alert(1)", "data:text/html,hi", "http://", "https://",
        "http://user:pw@example.com/", "http://user@example.com/", "http://exa mple.com/", "http://example.com/a\nb", "http://" + "a" * 3000 + ".com",
        "gopher://example.com/", "//example.com/", "http://[::1", "http://example.com:99999/",
    ])
    def test_malformed_addresses_are_refused_before_any_network_use(self, address):
        e = refused(address, total_timeout=5)
        assert e.code in ("invalid_url", "url_not_fetchable") and e.status == 422

    @pytest.mark.parametrize("port", [22, 25, 3306, 6379, 9200, 11211, 2375])
    def test_only_web_ports(self, port):
        assert refused(f"http://example.com:{port}/").code == "url_not_fetchable"

    def test_the_refusal_does_not_reveal_why(self):
        messages = {refused(a).message for a in ("http://127.0.0.1/", "http://10.0.0.5/", "http://localhost:8080/", "http://nonexistent.invalid/")}
        assert len(messages) == 1

    def test_local_server_is_refused_by_default(self, server):
        port = server.server_address[1]
        Handler.seen.clear()
        e = refused(f"http://127.0.0.1:{port}/html")
        assert e.code == "url_not_fetchable"
        assert not Handler.seen                                           # not a single request reached the server

    def test_a_name_with_one_private_answer_is_refused(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0)), (2, 1, 6, "", ("10.0.0.1", 0))])
        assert real_resolve_public("mixed.test") == []

    def test_dns_rebinding_cannot_change_the_address_between_check_and_connect(self, monkeypatch):
        """The connection uses the answer that was checked, not a second lookup: a resolver that flips to a private address is not asked twice."""
        calls = []

        def flipping(host, *a, **k):
            calls.append(host)
            return [(2, 1, 6, "", ("93.184.216.34" if len(calls) == 1 else "127.0.0.1", 0))]
        monkeypatch.setattr(socket, "getaddrinfo", flipping)
        backend = fetch._PublicOnlyBackend()
        seen = {}
        monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", lambda self, host, port, *a, **k: seen.setdefault("host", host) and (_ for _ in ()).throw(httpcore.ConnectError("stop")))
        with pytest.raises(httpcore.ConnectError):
            backend.connect_tcp("rebind.test", 80)
        assert seen["host"] == "93.184.216.34" and len(calls) == 1

    def test_the_network_backend_is_installed_on_the_pool(self):
        with fetch.make_client() as client:
            assert isinstance(client._transport._pool._network_backend, fetch._PublicOnlyBackend)

    def test_httpx_version_is_the_one_the_pinning_was_written_against(self):
        assert httpx.__version__.startswith("0.28.") and httpcore.__version__.startswith("1.0.")


class TestFetching:
    def test_html(self, url):
        r = fetch.fetch_url(url("/html"))
        assert (r.kind, r.content_type, r.charset, r.redirects) == ("html", "text/html", "utf-8", 0) and b"<h1>Hello</h1>" in r.data

    def test_pdf_docx_text_kinds(self, url):
        assert fetch.fetch_url(url("/pdf")).kind == "pdf"
        assert fetch.fetch_url(url("/docx")).kind == "docx"
        assert fetch.fetch_url(url("/text")).kind == "text"

    def test_octet_stream_is_sniffed_or_judged_by_extension(self, url):
        assert fetch.fetch_url(url("/octet-pdf")).kind == "pdf"
        assert fetch.fetch_url(url("/readme.md")).kind == "text"

    @pytest.mark.parametrize("path,kind", [("/png", "image/png"), ("/json", "application/json")])
    def test_other_types_are_refused_before_they_are_stored(self, url, path, kind):
        e = refused(url(path))
        assert e.status == 415 and e.code == "fetch_unsupported_type" and kind in e.message

    def test_empty_answer(self, url):
        assert refused(url("/empty")).code == "fetch_empty"

    def test_status_errors(self, url):
        assert (refused(url("/404")).status, refused(url("/404")).code) == (422, "fetch_status")
        e = refused(url("/403"))
        assert e.status == 422 and "login" in e.message
        assert refused(url("/500")).status == 502

    def test_gzip_is_decoded(self, url):
        assert b"<h1>Hello</h1>" in fetch.fetch_url(url("/gzip-ok")).data

    def test_request_carries_no_cookies_or_credentials(self, url):
        fetch.fetch_url(url("/set-cookie"))
        fetch.fetch_url(url("/headers"))
        sent = {k.lower() for k in Handler.seen[-1][1]}
        assert not ({"cookie", "authorization", "referer", "x-agent-backend-secret"} & sent)
        assert "user-agent" in sent

    def test_a_cookie_set_during_a_redirect_chain_is_not_sent_back(self, url):
        r = fetch.fetch_url(url("/cookie-then-redirect"))
        sent = {k.lower() for k in json.loads(r.data)}
        assert "cookie" not in sent and r.redirects == 1

    def test_environment_proxy_is_not_used(self, url, monkeypatch):
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
        assert fetch.fetch_url(url("/html")).kind == "html"


class TestRedirects:
    def test_followed(self, url):
        r = fetch.fetch_url(url("/redirect"))
        assert r.redirects == 1 and r.final_url.endswith("/html")

    def test_to_loopback_is_refused_and_never_requested(self, url):
        e = refused(url("/redirect-private"))
        assert e.code == "url_not_fetchable"
        assert [s for s, _ in Handler.seen] == ["/redirect-private"]       # the hop was never made

    def test_to_the_metadata_service_is_refused(self, url):
        assert refused(url("/redirect-metadata")).code == "url_not_fetchable"

    @pytest.mark.parametrize("path", ["/redirect-file", "/redirect-creds", "/redirect-port"])
    def test_to_a_bad_scheme_credentials_or_port_is_refused(self, url, path):
        assert refused(url(path)).code in ("invalid_url", "url_not_fetchable")

    def test_loop_is_cut_off(self, url):
        e = refused(url("/loop"))
        assert e.code == "too_many_redirects"
        assert len(Handler.seen) == fetch.MAX_REDIRECTS + 1

    def test_redirect_without_a_location(self, url):
        assert refused(url("/redirect-nolocation")).code == "fetch_status"


class TestLimits:
    def test_declared_length_over_the_cap_is_refused_unread(self, url):
        t = time.perf_counter()
        e = refused(url("/big"), max_html_bytes=1024 * 1024, max_file_bytes=2 * 1024 * 1024)
        assert e.status == 413 and e.code == "fetch_too_large"
        assert time.perf_counter() - t < 3

    def test_chunked_body_over_the_cap_is_cut(self, url):
        e = refused(url("/big-chunked"), max_html_bytes=512 * 1024, max_file_bytes=2 * 1024 * 1024)
        assert e.status == 413

    def test_gzip_bomb_stops_at_the_cap(self, url):
        t = time.perf_counter()
        e = refused(url("/gzip-bomb"), max_html_bytes=1024 * 1024, max_file_bytes=2 * 1024 * 1024)
        assert e.status == 413
        assert time.perf_counter() - t < 5

    def test_slow_drip_hits_the_total_deadline(self, url):
        t = time.perf_counter()
        e = refused(url("/drip"), total_timeout=2.0)
        assert e.status == 504 and e.code == "fetch_timeout"
        assert time.perf_counter() - t < 6

    def test_silent_server_hits_the_read_timeout(self, url, monkeypatch):
        monkeypatch.setattr(fetch, "READ_TIMEOUT", 1.0)
        t = time.perf_counter()
        e = refused(url("/silent"), total_timeout=5.0)
        assert e.status == 504
        assert time.perf_counter() - t < 7

    def test_connection_refused(self, monkeypatch):
        monkeypatch.setattr(fetch, "ALLOWED_PORTS", {1, 80, 443})
        monkeypatch.setattr(fetch, "resolve_public", lambda host: ["127.0.0.1"])
        assert refused("http://allowed.test:1/").code == "url_not_fetchable"
