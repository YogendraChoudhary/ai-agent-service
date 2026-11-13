"""
HTML to Markdown, for web pages (URL import), DOCX (mammoth writes HTML) and HTML files.

Written for this service, not a general converter, so it makes the choices an import needs:

  * the page's main content is found and the rest (navigation, footers, cookie banners, share bars, comments) is dropped;
  * text a person cannot see is dropped: elements hidden with the `hidden` attribute, aria-hidden, display:none, visibility:hidden, opacity:0,
    font-size:0 or pushed off screen, plus zero-width and Unicode "tag" characters. That is where hidden prompt-injection text lives;
  * Markdown's own punctuation in the text is escaped, so a sentence that starts with "1." or "#" does not become a list or a heading;
  * a data table becomes a GFM table; a layout table (cells holding paragraphs, lists or other tables) is read as ordinary blocks;
  * math (MathML, KaTeX, MathJax, Wikipedia) becomes $...$ and $$...$$, code keeps its language, task-list boxes become [ ] and [x];
  * every loop is bounded (depth, rows, output size), because the input is a stranger's HTML.

html_to_markdown() returns the Markdown plus counts and warnings; it never raises on bad HTML.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit

from lxml import etree
from lxml import html as lxml_html

from .mdutil import escape_inline, escape_line_starts, strip_invisible

MAX_DEPTH = 120
MAX_TABLE_ROWS = 500
MAX_TABLE_COLS = 30
MAX_OUTPUT_CHARS = 400_000

DROP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed", "audio", "video", "source", "track",
             "map", "area", "head", "link", "meta", "base", "title", "select", "textarea", "button", "option", "optgroup", "datalist",
             "dialog", "frame", "frameset", "applet", "rt", "rp"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "header", "footer", "aside", "nav", "address", "figure", "figcaption", "form",
              "fieldset", "center", "body", "html", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "blockquote",
              "pre", "table", "hr", "details", "summary", "menu", "dir", "legend", "caption", "hgroup", "search"}
CONTAINER_TAGS = {"div", "section", "article", "main", "header", "footer", "aside", "nav", "address", "form", "fieldset", "center", "body",
                  "html", "hgroup", "search", "figure", "legend"}
BOLD, ITALIC, STRIKE, CODE = {"b", "strong"}, {"i", "em", "cite", "dfn", "var"}, {"s", "del", "strike"}, {"code", "kbd", "samp", "tt"}

_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?\s*(?:;|!|$)|font-size\s*:\s*0(?:px|pt|em|rem|%)?\s*(?:;|!|$)|"
    r"(?:left|top|text-indent)\s*:\s*-\d{3,}|clip\s*:\s*rect\(\s*0|(?:max-)?height\s*:\s*0(?:px)?\s*;[^\"]*overflow\s*:\s*hidden", re.I)
_BOILERPLATE = re.compile(
    r"(?:^|[\s_-])(?:nav|navbar|navigation|menu|sidebar|side-bar|footer|masthead|banner|breadcrumbs?|cookies?|consent|gdpr|newsletter|subscribe|"
    r"signup|sign-up|login|promo|adverts?|ads?|advertisement|sponsor(?:ed)?|social|share|sharing|related|recommend(?:ed|ations)?|comments?|"
    r"respond|popup|modal|overlay|toolbar|skip(?:link)?|pagination|pager|editsection|mw-editsection|noprint|navbox|printfooter|catlinks|"
    r"hatnote|widget|tooltip|toc-toggle)(?:$|[\s_-])", re.I)
_CONTENT_HINT = re.compile(r"(?:^|[\s_-])(?:content|article|post|entry|story|main|body|text|prose|markdown)(?:$|[\s_-])", re.I)
_LANG = [re.compile(p, re.I) for p in (r"(?:^|\s)language-([\w+#.-]+)", r"(?:^|\s)lang-([\w+#.-]+)", r"highlight-(?:source|text)-([\w+#.-]+)",
                                        r"(?:^|\s)brush:\s*([\w+#.-]+)", r"(?:^|\s)sourceCode\s+([\w+#.-]+)", r"(?:^|\s)prettyprint\s+lang-([\w+#.-]+)")]
_WS = re.compile(r"[ \t\r\n\f\v  -   　]+")
_BR = ""
_TRACKING_SIZE = re.compile(r"^\s*[0-2]\s*(?:px)?\s*$")


@dataclass
class Converted:
    markdown: str
    title: str = ""
    stats: Dict[str, int] = field(default_factory=dict)
    warnings: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Decoding and parsing
# ---------------------------------------------------------------------------------------------------------------------------------------

def decode_html(data: bytes, declared: Optional[str] = None) -> str:
    """Bytes to text: a BOM, then the declared charset (HTTP header), then <meta charset>, then a guess. Never raises."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    candidates = []
    if declared:
        candidates.append(declared)
    head = data[:4096].decode("ascii", "ignore")
    meta = re.search(r"<meta[^>]+charset\s*=\s*[\"']?\s*([\w-]+)", head, re.I)
    if meta:
        candidates.append(meta.group(1))
    for name in candidates:
        try:
            return data.decode(name.strip(), "replace")
        except (LookupError, ValueError):
            continue
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data[:200_000]).best()
        if best is not None:
            return data.decode(best.encoding, "replace")
    except Exception:
        pass
    return data.decode("latin-1", "replace")


PARSER_DEPTH_LIMIT = 255          # libxml2 stops building the tree at 256 levels and silently drops what is deeper


def _depth(root, cap: int = PARSER_DEPTH_LIMIT + 2) -> int:
    """How deep the parsed tree goes (stops counting at `cap`); iterative, so a hostile page cannot exhaust the stack."""
    deepest, stack = 0, [(root, 1)]
    while stack:
        el, d = stack.pop()
        if d > deepest:
            deepest = d
            if deepest >= cap:
                return deepest
        stack.extend((c, d + 1) for c in el if isinstance(c.tag, str))
    return deepest


def _parse(text: str):
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)                      # an XML declaration makes lxml refuse a str
    text = text.replace("\x00", "")
    parser = lxml_html.HTMLParser(recover=True, remove_comments=True, remove_pis=True, no_network=True, huge_tree=False)
    try:
        return lxml_html.document_fromstring(text, parser=parser)
    except (etree.ParserError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------------------------------------------------------------------

class _Converter:
    def __init__(self, base_url: str, images: str, footnotes: bool, max_chars: int):
        self.base_url = base_url
        self.images = images                    # "keep" (web) or "omit" (DOCX: the picture is not in the output)
        self.footnotes = footnotes              # read mammoth's footnote markup
        self.max_chars = max_chars
        self.stats: Dict[str, int] = {}
        self.chars = 0
        self.truncated = False
        self.footnote_defs: List[Tuple[str, List[str]]] = []

    def count(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    # --- text ----------------------------------------------------------------------------------------------------------------------

    def clean(self, s: str) -> str:
        if not s:
            return ""
        s, hidden = strip_invisible(s)
        if hidden:
            self.count("hidden_characters", hidden)
        return s

    def text(self, s: Optional[str], code: bool = False) -> str:
        s = self.clean(s or "")
        if not s:
            return ""
        s = _WS.sub(" ", s)
        return s if code else escape_inline(s)

    @staticmethod
    def wrap(marker: str, inner: str) -> str:
        core = inner.strip(" ")
        if not core or core.strip(_BR + " ") == "":
            return inner
        lead, trail = inner[:len(inner) - len(inner.lstrip(" "))], inner[len(inner.rstrip(" ")):]
        return f"{lead}{marker}{core}{marker}{trail}"

    @staticmethod
    def code_span(s: str) -> str:
        s = _WS.sub(" ", s).strip()
        if not s:
            return ""
        longest = max((len(m) for m in re.findall(r"`+", s)), default=0)
        fence = "`" * (longest + 1)
        pad = " " if s.startswith("`") or s.endswith("`") else ""
        return f"{fence}{pad}{s}{pad}{fence}"

    # --- addresses -----------------------------------------------------------------------------------------------------------------

    def url(self, href: str, allow_mailto: bool = True) -> str:
        href = (href or "").strip()
        if not href or href.startswith("#"):
            return ""
        try:
            absolute = urljoin(self.base_url, href) if self.base_url else href
            scheme = urlsplit(absolute).scheme.lower()
        except ValueError:
            return ""
        if scheme in ("http", "https") or (allow_mailto and scheme in ("mailto", "tel")):
            return absolute
        return ""

    @staticmethod
    def dest(url: str) -> str:
        return quote(url, safe=":/?#[]@!$&'*+,;=%~-._").replace("(", "%28").replace(")", "%29")

    # --- tree questions ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def is_hidden(el) -> bool:
        if el.get("hidden") is not None or (el.get("aria-hidden") or "").lower() == "true":
            return True
        if el.tag == "input" and (el.get("type") or "").lower() == "hidden":
            return True
        style = el.get("style")
        return bool(style and _HIDDEN_STYLE.search(style))

    @staticmethod
    def has_block(el) -> bool:
        return any(isinstance(d.tag, str) and d.tag in BLOCK_TAGS for d in el.iterdescendants())

    # --- inline --------------------------------------------------------------------------------------------------------------------

    def inline_children(self, el, depth: int, flags: frozenset) -> str:
        parts = [self.text(el.text, "code" in flags)]
        for child in el:
            if isinstance(child.tag, str):
                parts.append(self.inline(child, depth + 1, flags))
            parts.append(self.text(child.tail, "code" in flags))
        return "".join(parts)

    def inline(self, el, depth: int, flags: frozenset = frozenset()) -> str:
        tag = el.tag
        if depth > MAX_DEPTH:
            return self.text(el.text_content())
        if tag in DROP_TAGS:
            return ""
        if tag == "br":
            return _BR
        if tag == "wbr":
            return ""
        if tag == "math":
            return self.math(el)
        if tag == "input":
            return ""
        if tag == "img":
            return self.image(el)
        if tag == "a":
            return self.link(el, depth, flags)
        if tag in BOLD and "bold" not in flags:
            return self.wrap("**", self.inline_children(el, depth, flags | {"bold"}))
        if tag in ITALIC and "italic" not in flags:
            return self.wrap("*", self.inline_children(el, depth, flags | {"italic"}))
        if tag in STRIKE and "strike" not in flags:
            return self.wrap("~~", self.inline_children(el, depth, flags | {"strike"}))
        if tag in CODE and "code" not in flags:
            return self.code_span(self.inline_children(el, depth, flags | {"code"}))
        if "mwe-math-element" in (el.get("class") or ""):
            math = next(iter(el.iter("math")), None)
            return self.math(math) if math is not None else ""
        if tag == "sup" or tag == "sub":
            return self.footnote_ref(el, depth, flags) if self.footnotes and tag == "sup" else self.inline_children(el, depth, flags)
        return self.inline_children(el, depth, flags)

    def footnote_ref(self, el, depth: int, flags: frozenset) -> str:
        anchor = next(iter(el.iter("a")), None)
        href = (anchor.get("href") or "") if anchor is not None else ""
        m = re.fullmatch(r"#(footnote|endnote)-(\d+)", href)
        if not m:
            return self.inline_children(el, depth, flags)
        return f"[^{m.group(2) if m.group(1) == 'footnote' else 'e' + m.group(2)}]"

    def link(self, el, depth: int, flags: frozenset) -> str:
        inner = self.inline_children(el, depth, flags)
        text = inner.strip()
        if not text or text.strip("\\#¶🔗") == "":
            return ""
        dest = self.url(el.get("href") or "")
        if not dest or "link" in flags:
            return inner
        if text.replace("\\", "") in (dest, (el.get("href") or "").strip()):
            return f"<{dest}>"
        self.count("links")
        return f"[{text}]({self.dest(dest)})"

    def image(self, el) -> str:
        src = (el.get("src") or el.get("data-src") or "").strip()
        alt = self.text(el.get("alt") or "").strip()
        if src == "docx-image:omitted":
            self.count("images_omitted")
            return ""
        if self.images == "omit" or not src or src.startswith("data:"):
            return ""
        if _TRACKING_SIZE.match(el.get("width") or "x") or _TRACKING_SIZE.match(el.get("height") or "x"):
            return ""
        dest = self.url(src, allow_mailto=False)
        if not dest or len(alt) < 2:
            self.count("images_skipped")
            return ""
        self.count("images")
        return f"![{alt}]({self.dest(dest)})"

    def math(self, el) -> str:
        tex = (el.get("alttext") or "").strip()
        if not tex:
            ann = el.xpath(".//*[local-name()='annotation'][@encoding='application/x-tex' or @encoding='TeX']")
            tex = (ann[0].text_content() if ann else "").strip()
        tex = re.sub(r"^\{\\displaystyle\s*(.*)\}$", r"\1", tex, flags=re.S).strip()
        if not tex:
            return self.text(el.text_content())
        self.count("math")
        return f"$${tex}$$" if (el.get("display") or "") == "block" else f"${tex}$"

    # --- paragraphs ----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def finish(inline: str, single_line: bool = False) -> str:
        s = _WS.sub(" ", inline).strip()
        s = s.replace(f"{_BR} ", _BR).replace(f" {_BR}", _BR).strip(_BR + " ")
        if single_line:
            return s.replace(_BR, " ")
        s = re.sub(f"{_BR}+", "\\\\\n", s)
        return escape_line_starts(s)

    def add(self, out: List[str], block: str) -> None:
        if not block.strip() or self.truncated:
            return
        room = self.max_chars - self.chars
        if len(block) > room:
            self.truncated = True
            if room < 200:
                return
            cut = block[:room]
            block = cut[:cut.rfind(" ")] if " " in cut[room // 2:] else cut             # a single huge block is cut at a word, not dropped
        self.chars += len(block) + 2
        out.append(block)

    def blocks(self, el, out: List[str], depth: int = 0) -> None:
        """The content of `el` as Markdown blocks: loose text and inline elements gather into paragraphs, block elements stand alone."""
        if depth > MAX_DEPTH:
            self.add(out, self.finish(self.text(el.text_content())))
            return
        run: List[str] = [self.text(el.text)]

        def flush():
            self.add(out, self.finish("".join(run)))
            run.clear()

        for child in el:
            tag = child.tag
            if isinstance(tag, str):
                if tag in DROP_TAGS:
                    pass
                elif tag in BLOCK_TAGS or (tag not in ("a", "span", "b", "i", "em", "strong", "u", "font", "small", "big", "mark", "label", "abbr", "cite",
                                                       "code", "kbd", "samp", "tt", "var", "s", "del", "ins", "strike", "q", "sup", "sub", "time", "bdi",
                                                       "bdo", "data", "dfn", "img", "br", "wbr", "math", "input", "picture") and self.has_block(child)) \
                        or (tag in ("a", "span", "b", "i", "em", "strong", "font", "label", "u", "small", "mark") and self.has_block(child)):
                    flush()
                    self.block(child, out, depth + 1)
                elif tag == "picture":
                    run.append(self.inline(next(iter(child.iter("img")), child), depth + 1))
                else:
                    run.append(self.inline(child, depth + 1))
            run.append(self.text(child.tail))
        flush()

    # --- blocks --------------------------------------------------------------------------------------------------------------------

    def block(self, el, out: List[str], depth: int) -> None:
        tag = el.tag
        if tag in DROP_TAGS or self.truncated:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = self.finish(self.inline_children(el, depth, frozenset({"link"})), single_line=True).strip(" #¶")
            if text:
                self.count("headings")
                self.add(out, f"{'#' * int(tag[1])} {text}")
        elif tag == "p" or tag in ("dt", "legend", "caption", "summary"):
            text = self.finish(self.inline_children(el, depth, frozenset()))
            if tag in ("dt", "summary") and text:
                text = f"**{text}**" if "**" not in text else text
            self.add(out, text)
        elif tag in ("ul", "ol", "menu", "dir"):
            ordered = tag == "ol"
            previous = re.match(r"([-*+]|\d{1,9}[.)])\s", out[-1]) if out else None
            marker = previous.group(1) if previous else ""
            # a list straight after a list of the same kind would merge into it, so it takes the other marker (- or *, 1. or 1))
            same_kind = bool(marker) and (marker[-1] in ".)") == ordered
            alt = same_kind and (marker == "-" or marker.endswith("."))
            self.add(out, self.list_(el, depth, ordered=ordered, level=0, alt=alt))
        elif tag == "li":
            self.add(out, self.list_item(el, depth, "-", 0))
        elif tag == "blockquote":
            inner: List[str] = []
            self.blocks(el, inner, depth + 1)
            quoted = "\n\n".join(inner)
            self.add(out, "\n".join(("> " + line).rstrip() for line in quoted.split("\n")))
        elif tag == "pre":
            self.count("code_blocks")
            self.add(out, self.pre(el))
        elif tag == "table":
            self.table(el, out, depth)
        elif tag == "hr":
            self.add(out, "---")
        elif tag == "figure":
            self.blocks(el, out, depth + 1)
        elif tag == "figcaption":
            text = self.finish(self.inline_children(el, depth, frozenset()))
            self.add(out, f"*{text}*" if text and "*" not in text else text)
        elif tag in ("dl", "dd", "details", "hgroup"):
            self.blocks(el, out, depth + 1)
        elif tag == "math":
            self.add(out, self.math(el))
        else:
            self.blocks(el, out, depth + 1)

    def list_(self, el, depth: int, ordered: bool, level: int, alt: bool = False) -> str:
        items: List[str] = []
        number = 1
        try:
            number = int(el.get("start") or 1)
        except ValueError:
            pass
        previous = None
        for child in el:
            if not isinstance(child.tag, str) or child.tag in DROP_TAGS:
                continue
            if child.tag in ("ul", "ol") and items:                       # a list straight inside a list: nested under the item before it
                nested = self.list_(child, depth + 1, child.tag == "ol", level + 1)
                if nested:
                    items[-1] = items[-1] + "\n" + _indent(nested, previous)
                continue
            if child.tag != "li":
                continue
            marker = (f"{number})" if alt else f"{number}.") if ordered else ("*" if alt else "-")
            number += 1
            previous = len(marker) + 1
            items.append(self.list_item(child, depth + 1, marker, level))
        loose = any("\n\n" in item for item in items)
        return ("\n\n" if loose else "\n").join(i for i in items if i.strip())

    def list_item(self, el, depth: int, marker: str, level: int) -> str:
        checkbox = el.xpath("./input[@type='checkbox'] | ./p/input[@type='checkbox'] | ./label/input[@type='checkbox']")
        prefix = ""
        if checkbox:                                           # a task-list item: the item's own box, not one of a nested list's
            prefix = "[x] " if checkbox[0].get("checked") is not None else "[ ] "
        parts: List[str] = []
        lead: List[str] = [self.text(el.text)]
        nested_start = False
        for child in el:
            if isinstance(child.tag, str) and child.tag in ("ul", "ol"):
                nested_start = True
                if "".join(lead).strip():
                    parts.append(self.finish("".join(lead)))
                lead = []
                nested = self.list_(child, depth + 1, child.tag == "ol", level + 1)
                if nested:
                    parts.append(nested)
            elif isinstance(child.tag, str) and child.tag in BLOCK_TAGS and child.tag != "li":
                if "".join(lead).strip():
                    parts.append(self.finish("".join(lead)))
                lead = []
                inner: List[str] = []
                self.block(child, inner, depth + 1)
                parts.extend(inner)
            elif isinstance(child.tag, str) and child.tag not in DROP_TAGS:
                lead.append(self.inline(child, depth + 1))
            lead.append(self.text(child.tail))
        if "".join(lead).strip():
            parts.append(self.finish("".join(lead)))
        if not parts:
            return ""
        width = len(marker) + 1
        first = parts[0]
        if first.startswith(("- ", "1. ")) and nested_start and len(parts) == 1:
            first, body = "", parts[0]
            return f"{marker} {prefix}".rstrip() + "\n" + _indent(body, width)
        text = f"{marker} {prefix}{_indent_after_first(first, width)}"
        for extra in parts[1:]:
            glue = "\n\n" if not extra.startswith(("- ", "* ")) and not re.match(r"\d+\. ", extra) else "\n"
            text += glue + _indent(extra, width)
        return text

    def pre(self, el) -> str:
        pieces: List[str] = []

        def walk(node):
            if node.text:
                pieces.append(node.text)
            for child in node:
                if isinstance(child.tag, str):
                    if child.tag == "br":
                        pieces.append("\n")
                    elif child.tag not in DROP_TAGS:
                        walk(child)
                if child.tail:
                    pieces.append(child.tail)
        walk(el)
        code = self.clean("".join(pieces)).replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").strip("\n")
        language = ""
        holders = [el, *[c for c in el if isinstance(c.tag, str) and c.tag == "code"][:1]]
        if el.getparent() is not None:
            holders.append(el.getparent())
        for holder in holders:
            attr = (holder.get("class") or "") + " " + (holder.get("data-lang") or "")
            for pattern in _LANG:
                m = pattern.search(attr)
                if m:
                    language = re.sub(r"[^\w+#.-]", "", m.group(1))[:20]
                    break
            if language:
                break
        longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{fence}{language}\n{code}\n{fence}" if code.strip() else ""

    # --- tables --------------------------------------------------------------------------------------------------------------------

    def table(self, el, out: List[str], depth: int) -> None:
        rows = [tr for tr in el.iter("tr") if self.row_owner(tr) is el]
        cells = [[c for c in tr if isinstance(c.tag, str) and c.tag in ("td", "th")] for tr in rows]
        cells = [r for r in cells if r]
        caption = next((c for c in el if isinstance(c.tag, str) and c.tag == "caption"), None)
        if caption is not None:
            self.block(caption, out, depth + 1)
        if not cells:
            return
        width = max(len(r) for r in cells)
        data = (len(cells) >= 2 and width >= 2 and len(cells) <= MAX_TABLE_ROWS and width <= MAX_TABLE_COLS
                and all(self.simple_cell(c) for r in cells for c in r))
        if not data:                                                   # a layout table: its cells are the page's blocks
            self.count("layout_tables")
            for r in cells:
                for c in r:
                    self.blocks(c, out, depth + 1)
            return
        self.count("tables")
        grid: List[List[str]] = []
        merged = False
        for r in cells:
            line: List[str] = []
            for c in r:
                text = self.finish(self.cell_inline(c, depth + 1), single_line=True).replace("|", "\\|")
                line.append(text)
                try:
                    span = int(c.get("colspan") or 1)
                except ValueError:
                    span = 1
                if span > 1 or (c.get("rowspan") or "1") not in ("1", ""):
                    merged = True
                line.extend([""] * (min(span, MAX_TABLE_COLS) - 1))
            grid.append(line)
        if merged:
            self.count("table_merged_cells")
        grid = [r + [""] * (width - len(r)) for r in grid]
        if all(re.fullmatch(r"\*\*[^*]+\*\*", c) or not c for c in grid[0]):          # a bold header row (Word) is already a header
            grid[0] = [c[2:-2] for c in grid[0]]
        align = []
        for c in cells[0] + [None] * 0:
            style = ((c.get("style") or "") + " " + (c.get("align") or "")).lower()
            align.append(":-:" if "center" in style else "--:" if "right" in style else ":--" if "left" in style else "---")
        align = (align + ["---"] * width)[:width]
        lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join(align) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
        self.add(out, "\n".join(lines))

    @staticmethod
    def simple_cell(cell) -> bool:
        """A data cell holds text and inline markup, or up to three plain paragraphs; a list, a table or a heading makes it a layout cell."""
        paragraphs = 0
        for d in cell.iterdescendants():
            if isinstance(d.tag, str) and d.tag in BLOCK_TAGS:
                if d.tag != "p":
                    return False
                paragraphs += 1
        return paragraphs <= 3

    def cell_inline(self, cell, depth: int) -> str:
        parts = [self.text(cell.text)]
        for child in cell:
            if isinstance(child.tag, str):
                piece = self.inline(child, depth + 1)
                parts.append(f" {piece} " if child.tag == "p" else piece)
            parts.append(self.text(child.tail))
        return "".join(parts)

    @staticmethod
    def row_owner(tr):
        node = tr.getparent()
        while node is not None and node.tag != "table":
            node = node.getparent()
        return node


def _indent(text: str, width: int) -> str:
    pad = " " * width
    return "\n".join((pad + line) if line.strip() else line for line in text.split("\n"))


def _indent_after_first(text: str, width: int) -> str:
    first, _, rest = text.partition("\n")
    return first + ("\n" + _indent(rest, width) if rest else "")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Pruning and choosing the main content
# ---------------------------------------------------------------------------------------------------------------------------------------

def _attr_tokens(el) -> str:
    return f"{el.get('id') or ''} {el.get('class') or ''}"


def _prune(root, conv: _Converter) -> Tuple[int, int]:
    """Remove what is never content, and what a person cannot see. Returns (hidden elements, characters in them)."""
    hidden_n = hidden_chars = 0
    for el in list(root.iter()):
        tag = el.tag
        if not isinstance(tag, str):
            continue
        if tag == "script" and (el.get("type") or "").lower().startswith("math/tex") and (el.text or "").strip():
            span = etree.Element("span")
            display = "mode=display" in (el.get("type") or "")
            span.text = ("$$" if display else "$") + el.text.strip() + ("$$" if display else "$")
            span.tail = el.tail
            conv.count("math")
            el.getparent().replace(el, span) if el.getparent() is not None else None
            continue
        if tag in DROP_TAGS or tag == "input" and (el.get("type") or "").lower() != "checkbox":
            _remove(el)
            continue
        if _Converter.is_hidden(el):
            text = (el.text_content() or "").strip()
            if text:
                hidden_n += 1
                hidden_chars += len(text)
            _remove(el)
    return hidden_n, hidden_chars


def _remove(el) -> None:
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail
    prev = el.getprevious()
    if tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


def _text_len(el) -> int:
    return len(_WS.sub(" ", el.text_content() or "").strip())


def _link_len(el) -> int:
    return sum(_text_len(a) for a in el.iter("a"))


def _drop_boilerplate(container, keep_total: int) -> int:
    """Within the chosen content: navigation, footers, share bars, comments and the like (by tag, then by id/class)."""
    removed = 0
    for el in list(container.iter()):
        if el is container or not isinstance(el.tag, str):
            continue
        if el.tag in ("nav", "aside", "footer"):
            _remove(el)
            removed += 1
            continue
        if _BOILERPLATE.search(_attr_tokens(el)) and not any(True for _ in el.iter("h1", "h2")) and _text_len(el) < max(300, keep_total // 3):
            _remove(el)
            removed += 1
    return removed


def _choose_container(body) -> Tuple[object, bool]:
    total = max(1, _text_len(body))
    named = body.xpath(".//article|.//main|.//*[@role='main']")
    best = max(named, key=_text_len, default=None)
    if best is not None and _text_len(best) >= max(0.35 * total, 600):
        return best, True
    scored = []
    for el in body.iter("div", "section", "td"):
        length = _text_len(el)
        if length < 200:
            continue
        score = length - 2 * _link_len(el)
        if _BOILERPLATE.search(_attr_tokens(el)):
            score *= 0.2
        if _CONTENT_HINT.search(_attr_tokens(el)):
            score *= 1.25
        scored.append((score, el))
    if scored:
        score, el = max(scored, key=lambda s: s[0])
        if _text_len(el) >= 0.3 * total:
            return el, True
    return body, False


def _title(doc, container) -> str:
    h1 = container.find(".//h1") if container is not None else None
    heading = (h1.text_content().strip() if h1 is not None else "")
    page = doc.findtext(".//title") or ""
    page = _WS.sub(" ", page).strip()
    if page:
        parts = [p.strip() for p in re.split(r"\s+[|–—·•»:-]\s+", page) if p.strip()]
        if parts:
            page = max(parts, key=len) if len(parts) > 1 and not heading else parts[0]
    return (_WS.sub(" ", heading).strip() or page).strip(" #¶🔗")                 # a heading's own anchor link ("¶") is not part of its title


# ---------------------------------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------------------------------

def html_to_markdown(html: "str | bytes", *, base_url: str = "", main_content: bool = True, images: str = "keep", footnotes: bool = False,
                     charset: Optional[str] = None, max_chars: int = MAX_OUTPUT_CHARS) -> Converted:
    text = decode_html(html, charset) if isinstance(html, bytes) else html
    raw_len = len(text)
    doc = _parse(text)
    if doc is None:
        return Converted("", warnings=[{"code": "unreadable_html", "message": "The page could not be read as HTML."}])
    base = doc.find(".//base")
    if base is not None and (base.get("href") or "").strip().lower().startswith(("http://", "https://")):
        base_url = base.get("href").strip()
    too_deep = _depth(doc) >= PARSER_DEPTH_LIMIT
    conv = _Converter(base_url, images, footnotes, max_chars)
    body = doc.find("body")
    if body is None:
        body = doc
    page_title = _title(doc, body)
    hidden_n, hidden_chars = _prune(doc, conv)
    container, chosen = body, False
    if main_content:
        container, chosen = _choose_container(body)
        conv.count("boilerplate_removed", _drop_boilerplate(container, _text_len(container)))
        if container is body:
            for el in list(body.iter("header", "nav", "aside", "footer")):
                _remove(el)
    out: List[str] = []
    if footnotes:
        _take_footnotes(container, conv)
    conv.blocks(container, out)
    if footnotes and conv.footnote_defs:
        out.append("\n\n".join(f"[^{label}]: {' '.join(lines).strip()}" for label, lines in conv.footnote_defs if " ".join(lines).strip()))
    markdown = "\n\n".join(b for b in out if b.strip())
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]+$", "", markdown, flags=re.M).strip() + "\n"
    title = page_title
    if main_content and title and not re.search(r"(?m)^# ", markdown) and markdown.strip():
        markdown = f"# {conv.text(title)}\n\n{markdown}"
    warnings: List[dict] = []
    if hidden_n:
        warnings.append({"code": "hidden_text_removed", "message": f"{hidden_n} hidden element(s) holding {hidden_chars} characters of text that a "
                                                                    "reader cannot see were left out."})
    if conv.stats.get("hidden_characters"):
        warnings.append({"code": "hidden_characters_removed", "message": f"{conv.stats['hidden_characters']} invisible or text-direction control "
                                                                         "character(s) were removed."})
    if conv.stats.get("images_omitted"):
        warnings.append({"code": "images_omitted", "message": f"{conv.stats['images_omitted']} image(s) were left out: pictures are not imported."})
    if conv.stats.get("table_merged_cells"):
        warnings.append({"code": "table_merged_cells", "message": f"{conv.stats['table_merged_cells']} table(s) had merged cells; the merged cells "
                                                                  "became empty cells."})
    if too_deep:
        warnings.append({"code": "nesting_too_deep", "message": "Part of the page is nested too deeply to be read (over 250 levels) and was left out."})
    if conv.truncated:
        warnings.append({"code": "output_truncated", "message": "The converted text was cut at the size limit."})
    if main_content and len(markdown.strip()) < 200 and raw_len > 20_000:
        warnings.append({"code": "little_text", "message": "Very little text was found on this page. It may load its content with JavaScript, "
                                                           "which is not run."})
    conv.stats.update({"hidden_elements": hidden_n, "main_content": int(chosen)})
    return Converted(markdown, title=title, stats=conv.stats, warnings=warnings)


def _take_footnotes(container, conv: _Converter) -> None:
    """mammoth writes footnotes and endnotes as <ol><li id="footnote-1">…</li></ol> at the end: pulled out to be written as [^1]: definitions."""
    for li in list(container.xpath(".//li[starts-with(@id,'footnote-') or starts-with(@id,'endnote-')]")):
        m = re.fullmatch(r"(footnote|endnote)-(\d+)", li.get("id") or "")
        if not m:
            continue
        for back in li.xpath(".//a[starts-with(@href,'#footnote-ref-') or starts-with(@href,'#endnote-ref-')]"):
            _remove(back)
        inner: List[str] = []
        conv.blocks(li, inner)
        conv.footnote_defs.append((m.group(2) if m.group(1) == "footnote" else f"e{m.group(2)}", inner))
        parent = li.getparent()
        _remove(li)
        if parent is not None and parent.tag in ("ol", "ul") and not len(parent):
            _remove(parent)
