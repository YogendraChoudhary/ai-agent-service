"""
Which web pages the research tools actually READ during this request.

Rule D13: an attributed quote, and a URL left in the text, are allowed only for a page that was read. The reader
tool reports each successful read here; the lint asks. Search results do not count: a snippet is not a page.
"""

from __future__ import annotations

import re
import threading
from typing import Iterable, Set
from urllib.parse import urlsplit

_TRAILING_RE = re.compile(r"[.,;:!?)\]}>'\"]+$")


def normalise_url(url: str) -> str:
    """Scheme-insensitive, fragment-free, no trailing slash or punctuation: what two spellings of a page share."""
    url = _TRAILING_RE.sub("", (url or "").strip())
    if "//" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return f"{host}{path}" + (f"?{parts.query}" if parts.query else "")


class WebSources:
    """Thread-safe: sections run in parallel and each may read pages."""

    def __init__(self, urls: Iterable[str] = ()):
        self._urls: Set[str] = {normalise_url(u) for u in urls}
        self._lock = threading.Lock()

    def add(self, url: str) -> None:
        with self._lock:
            self._urls.add(normalise_url(url))

    def allows(self, url: str) -> bool:
        with self._lock:
            return normalise_url(url) in self._urls

    def __len__(self) -> int:
        with self._lock:
            return len(self._urls)

    def __bool__(self) -> bool:
        return len(self) > 0
