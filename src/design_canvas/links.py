"""
Button links stay exactly what the user gave. The schema already rejects unsafe links; this check catches what it
cannot: a model that turns a link it may not use into a different one (live run 2026-11-11: "/checkout" came back as
"#checkout"), or quietly drops a link the user asked for. Such links are removed and the editor gets a warning.
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

from .schema import ButtonNode, ContainerNode, DesignDocumentAST, safe_href

_LINK = re.compile(
    r"(?:https?://[^\s'\"<>)\]]+|mailto:[^\s'\"<>)\]]+|tel:\+?[\d][\d\s().-]*\d|(?:javascript|data|vbscript|file):[^\s'\"<>]*|"
    r"(?<![\w/:.])/[A-Za-z0-9][\w\-./?=&%]*|(?<![\w&])#[A-Za-z][\w\-:.]*|"
    r"\bwww\.[^\s'\"<>)\]]+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|ai|dev|co|app|me|info|biz|us|uk|in)(?:/[^\s'\"<>)\]]*)?)",
    re.I)


def _buttons(node) -> Iterator[ButtonNode]:
    if isinstance(node, ButtonNode):
        yield node
    if isinstance(node, ContainerNode):
        for child in node.children:
            yield from _buttons(child)


def _hrefs_in(raw) -> set[str]:
    found = set()
    if isinstance(raw, dict):
        if raw.get("node_type") == "button" and isinstance(raw.get("href"), str) and raw["href"]:
            found.add(raw["href"])
        for value in raw.values():
            found |= _hrefs_in(value)
    elif isinstance(raw, list):
        for value in raw:
            found |= _hrefs_in(value)
    return found


def _site(link: str) -> str:
    """host + path, without scheme, www. or a trailing slash, for matching "example.com" to "https://example.com/"."""
    link = re.sub(r"^(?:https?://)?(?:www\.)?", "", link.strip().casefold())
    return link.rstrip("/.,;:!?")


def requested_links(user_query: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?") for m in _LINK.finditer(user_query or "")]


def check_links(ast: DesignDocumentAST, user_query: str, existing_ast_json: Optional[str] = None) -> list[dict]:
    """Remove button links the user did not give; return warnings for those and for requested links that are unusable."""
    asked = requested_links(user_query)
    asked_sites = {_site(a) for a in asked
                   if not a.startswith(("#", "/")) and not re.match(r"(?:javascript|data|vbscript|file|mailto|tel):", a, re.I)}
    try:
        existing = _hrefs_in(json.loads(existing_ast_json)) if existing_ast_json else set()
    except (TypeError, ValueError):
        existing = set()
    warnings, kept = [], set()
    for button in _buttons(ast.root):
        href = button.href
        if not href:
            continue
        given = (href in existing or href in asked or (href.lower().startswith(("http://", "https://")) and _site(href) in asked_sites)
                 or (href.lower().startswith(("mailto:", "tel:")) and href.casefold() in (user_query or "").casefold()))
        if given:
            kept.add(href)
            continue
        button.href = None
        warnings.append({"kind": "link_removed", "node_id": button.id,
                         "detail": f"The link '{href}' on '{button.label}' was not in your request, so it was removed."})
    for link in asked:
        try:
            usable = safe_href(link) is not None
        except ValueError:
            usable = False
        if usable or link in kept or any(_site(link) == _site(h) for h in kept if h.lower().startswith("http")):
            continue
        warnings.append({"kind": "link_not_set", "link": link,
                         "detail": f"'{link}' can't be used as a button link. Use a full https:// address, a mailto: or "
                                   "tel: link, or a #section anchor."})
    return warnings
