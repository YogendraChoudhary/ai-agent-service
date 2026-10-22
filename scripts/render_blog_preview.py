"""
Review-only HTML preview of generated blog posts.

Renders Markdown posts (the files scripts/evaluate_blog.py writes, or any GFM file) the way a CommonMark + GFM-tables
renderer shows them, so tables, pull quotes, pictures and captions can be judged by eye, and lines two generations of
the same brief up side by side. Pictures are resized and inlined, so the result is ONE self-contained file.

NOT the editor's renderer (Eddyter / Lexical): how the editor shows "a picture followed by an italic caption line" is
open item O2 of docs/rich_content_engine_plan.md. A table whose delimiter row has the wrong number of cells is not a
table in any strict renderer; here it shows as plain pipe characters, flagged, exactly as a reader would see it.

Beside each post: its page rhythm (every block, top to bottom, sized by how much room it takes), and an at-a-glance
table of what was measured (from the post's header comment when scripts/evaluate_blog.py wrote it) and what is counted
in the Markdown itself.

Usage:
  python scripts/render_blog_preview.py --out local_run/blog_preview.html \\
      --compare "Sleep routine" local_run/blog_eval/old.md local_run/blog_eval/new.md \\
      --compare "One-on-ones"   local_run/blog_eval/old2.md local_run/blog_eval/new2.md
  python scripts/render_blog_preview.py --out preview.html --post "My post" post.md
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import html
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_it import MarkdownIt          # noqa: E402
from PIL import Image, ImageOps             # noqa: E402

from src.blog.validate import longest_prose_run              # noqa: E402
from src.longform.markdown_ast import Block, parse, prose_words   # noqa: E402

IMAGE_MAX_WIDTH = 960
IMAGE_QUALITY = 80
_HEADER_RE = re.compile(r"\A\s*<!--(.*?)-->\s*", re.S)
_IMG_RE = re.compile(r"<img\b[^>]*?src=\"([^\"]*)\"[^>]*?alt=\"([^\"]*)\"[^>]*>")
_PIPE_LINE_RE = re.compile(r"^\s*\|")


# ---------------------------------------------------------------------------
# Reading a post
# ---------------------------------------------------------------------------

@dataclass
class Post:
    role: str                       # "Previous" | "Current" | "Post"
    path: Path
    markdown: str
    header: Dict[str, Optional[str]] = field(default_factory=dict)      # what the evaluation script recorded, if anything
    facts: Dict[str, int] = field(default_factory=dict)                 # counted from the Markdown itself
    rhythm: List[Tuple[str, int, str]] = field(default_factory=list)    # (kind, weight, tooltip) per block
    body: str = ""


@dataclass
class PostSet:
    label: str
    posts: List[Post]

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.label.lower()).strip("-") or "posts"


def parse_header(text: str) -> Tuple[Dict[str, Optional[str]], str]:
    """(fields, markdown without the header comment). The comment is what scripts/evaluate_blog.py writes on top of a post."""
    match = _HEADER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)

    def pick(pattern: str) -> Optional[str]:
        found = re.search(pattern, raw, re.S)
        return found.group(1).strip() if found else None
    fields = {
        "model": pick(r"^\s*([^|]+?)\s*\|"),
        "outline_model": pick(r"outline model:\s*([^|]+?)\s*\|"),
        "brief": pick(r"brief:\s*(.*?)\s*\|\s*options:"),
        "total_seconds": pick(r"([\d.]+)s total"),
        "first_content_seconds": pick(r"first content\s+([\d.]+)s"),
        "words": pick(r"\bwords\s+(\d+)"),
    }
    return fields, text[match.end():]


def _looks_like_pipe_table(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(_PIPE_LINE_RE.match(ln) for ln in lines)


def classify(blocks: List[Block]) -> List[Block]:
    """Blocks as a reader experiences them: a paragraph of pipes that markdown did not turn into a table is 'broken'."""
    return [dataclasses.replace(b, kind="broken") if b.kind == "paragraph" and _looks_like_pipe_table(b.text) else b
            for b in blocks]


def count_facts(blocks: List[Block]) -> Dict[str, int]:
    return {
        "words": prose_words(blocks),
        "sections": sum(b.kind == "heading" and b.level == 2 for b in blocks),
        "tables": sum(b.kind == "table" for b in blocks),
        "broken_tables": sum(b.kind == "broken" for b in blocks),
        "quotes": sum(b.kind == "blockquote" for b in blocks),
        "pictures": sum(b.kind == "image" for b in blocks),
        "lists": sum(b.kind == "list" for b in blocks),
        "longest_prose_run": longest_prose_run(blocks),
    }


def _rows(block: Block) -> int:
    return max(1, len([ln for ln in block.text.splitlines() if ln.strip()]) - 2)


def rhythm_of(blocks: List[Block]) -> List[Tuple[str, int, str]]:
    """Every block, top to bottom, as (kind, weight, tooltip). The weight is roughly how much room it takes on the page."""
    out: List[Tuple[str, int, str]] = []
    for b in blocks:
        if b.kind == "blank":
            continue
        if b.kind == "heading":
            out.append(("heading", 14 if b.level == 1 else 9, f"Heading level {b.level}"))
        elif b.kind == "image":
            out.append(("image", 70, "Picture with caption"))
        elif b.kind == "table":
            out.append(("table", 22 + 12 * _rows(b), f"Table, {_rows(b)} rows"))
        elif b.kind == "broken":
            out.append(("broken", 22 + 12 * _rows(b), "Table that does not render (wrong delimiter row)"))
        elif b.kind == "blockquote":
            out.append(("quote", 26 + b.words, f"Pull quote, {b.words} words"))
        elif b.kind == "list":
            out.append(("list", 10 + b.words, f"List, {b.items} items"))
        else:
            out.append(("text", max(b.words, 8), f"Paragraph, {b.words} words"))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def inline_image(src: str, base_dir: Path, cache: Dict[Path, Optional[str]]) -> Optional[str]:
    """A data: URI for a local picture, resized so a page of several posts stays small. None when it cannot be read."""
    if src.startswith("data:"):
        return src
    path = (base_dir / src).resolve()
    if path not in cache:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((IMAGE_MAX_WIDTH, IMAGE_MAX_WIDTH * 2))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=IMAGE_QUALITY, optimize=True, progressive=True)
            cache[path] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except (OSError, ValueError):
            cache[path] = None
    return cache[path]


def render_markdown(markdown: str, base_dir: Path, cache: Dict[Path, Optional[str]]) -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")      # raw HTML in a post is shown as text, never run
    tokens = md.parse(markdown)
    for i, token in enumerate(tokens):
        # a paragraph made of pipes is a table that did not render: show it as the reader sees it, and say so
        if token.type == "paragraph_open" and _looks_like_pipe_table(tokens[i + 1].content):
            token.attrSet("class", "not-a-table")
    out = md.renderer.render(tokens, md.options, {})

    def picture(match: re.Match) -> str:
        src, alt = match.group(1), html.unescape(match.group(2))
        data = inline_image(src, base_dir, cache) if not src.startswith("placeholder::") else None
        if data is None:
            why = "the picture was still generating" if src.startswith("placeholder::") else f"picture file not found: {src}"
            return f'<span class="missing-picture">{html.escape(why)}<br><em>{html.escape(alt)}</em></span>'
        return f'<img src="{data}" alt="{html.escape(alt)}" width="{IMAGE_MAX_WIDTH}" height="{round(IMAGE_MAX_WIDTH * 9 / 16)}">'
    out = _IMG_RE.sub(picture, out)
    return out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def load_post(role: str, path: Path, cache: Dict[Path, Optional[str]]) -> Post:
    header, markdown = parse_header(path.read_text(encoding="utf-8"))
    blocks = classify(parse(markdown))
    return Post(role=role, path=path, markdown=markdown, header=header, facts=count_facts(blocks), rhythm=rhythm_of(blocks),
                body=render_markdown(markdown, path.parent, cache))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #F3F5F8; --surface: #FFFFFF; --surface-2: #F7F8FA; --text: #18212C; --muted: #566272; --border: #DCE1E8;
  --accent: #0B6B6F; --accent-ink: #FFFFFF; --accent-soft: #E2F1F1; --prev: #7A4F0E; --prev-soft: #F8EEDC;
  --bad: #B3261E; --bad-soft: #FCE9E7; --focus: #1D6FE0;
  --k-text: #C5CCD6; --k-heading: #566272; --k-image: #2F6FDB; --k-table: #1B9A68; --k-quote: #D9801A; --k-list: #8553C4;
  --k-broken: #D12B22;
  --ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --read: "Source Serif 4", "Iowan Old Style", Georgia, "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0E131A; --surface: #161D26; --surface-2: #1B2430; --text: #E5EAF0; --muted: #9AA6B5; --border: #2A3543;
    --accent: #55C4C7; --accent-ink: #062A2C; --accent-soft: #123A3C; --prev: #E2B56F; --prev-soft: #3A2D14;
    --bad: #FF8A80; --bad-soft: #3D1B19; --focus: #7FB0FF;
    --k-text: #3A4555; --k-heading: #9AA6B5; --k-image: #6FA3FF; --k-table: #4CD6A0; --k-quote: #FFB04D; --k-list: #B896F0;
    --k-broken: #FF6B60; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0E131A; --surface: #161D26; --surface-2: #1B2430; --text: #E5EAF0; --muted: #9AA6B5; --border: #2A3543;
  --accent: #55C4C7; --accent-ink: #062A2C; --accent-soft: #123A3C; --prev: #E2B56F; --prev-soft: #3A2D14;
  --bad: #FF8A80; --bad-soft: #3D1B19; --focus: #7FB0FF;
  --k-text: #3A4555; --k-heading: #9AA6B5; --k-image: #6FA3FF; --k-table: #4CD6A0; --k-quote: #FFB04D; --k-list: #B896F0;
  --k-broken: #FF6B60; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font: 15px/1.5 var(--ui); padding-block: 28px 64px;
       padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.page { max-width: 1240px; margin-inline: auto; display: grid; gap: 22px; min-width: 0; }
h1, h2, h3 { text-wrap: balance; }
button { font: inherit; color: inherit; }
button:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }

.top { display: grid; gap: 8px; max-width: 74ch; }
.eyebrow { margin: 0; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.top h1 { margin: 0; font-size: clamp(24px, 3.4vw, 34px); line-height: 1.15; font-weight: 600; }
.lede { margin: 0; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 6px 0 0; padding: 0; list-style: none; font-size: 12.5px; color: var(--muted); }
.legend li { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; background: var(--k-text); display: inline-block; }

.tabs { display: flex; flex-wrap: wrap; gap: 8px; }
.tab { border: 1px solid var(--border); background: var(--surface); border-radius: 999px; padding: 7px 16px; cursor: pointer; font-weight: 500; }
.tab[aria-current="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }

.set { display: grid; gap: 16px; min-width: 0; }
.set-head { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 10px 20px; }
.set-head h2 { margin: 0; font-size: 20px; font-weight: 600; }
.brief { margin: 2px 0 0; color: var(--muted); font-size: 14px; max-width: 80ch; }
.views { display: inline-flex; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface); }
.views button { border: 0; background: transparent; padding: 7px 14px; cursor: pointer; font-weight: 500; }
.views button + button { border-left: 1px solid var(--border); }
.views button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent); }

.glance-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
table.glance { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
.glance th, .glance td { padding: 8px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
.glance thead th { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; background: var(--surface-2); }
.glance tbody th { font-weight: 500; color: var(--muted); }
.glance tr:last-child th, .glance tr:last-child td { border-bottom: 0; }
.glance td.changed { font-weight: 600; background: var(--accent-soft); }
.glance td.bad { font-weight: 600; color: var(--bad); background: var(--bad-soft); }
@media (max-width: 640px) { table.glance { font-size: 13px; } .glance th, .glance td { white-space: normal; padding: 7px 9px; } }

.cols { display: grid; gap: 22px; grid-template-columns: minmax(0, 1fr); align-items: start; }
@media (min-width: 1040px) { .cols { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.set[data-view="previous"] .col.current, .set[data-view="current"] .col.previous { display: none; }
.set[data-view="previous"] .cols, .set[data-view="current"] .cols, .set[data-single] .cols { grid-template-columns: minmax(0, 760px); justify-content: center; }
.set[data-single] .glance-wrap { width: 100%; max-width: 760px; margin-inline: auto; }
.col { display: grid; gap: 10px; min-width: 0; }
.col-head { display: grid; gap: 8px; }
.role { justify-self: start; font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        background: var(--accent-soft); color: var(--accent); }
.col.previous .role { background: var(--prev-soft); color: var(--prev); }
.summary { margin: 0; font-size: 13.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.rhythm { display: flex; height: 26px; gap: 1px; border-radius: 6px; overflow: hidden; background: var(--border); }
.rhythm span { display: block; min-width: 2px; background: var(--k-text); }
.rhythm .heading { background: var(--k-heading); } .rhythm .image { background: var(--k-image); }
.rhythm .table { background: var(--k-table); } .rhythm .quote { background: var(--k-quote); }
.rhythm .list { background: var(--k-list); } .rhythm .broken { background: var(--k-broken); }
.swatch.heading { background: var(--k-heading); } .swatch.image { background: var(--k-image); } .swatch.table { background: var(--k-table); }
.swatch.quote { background: var(--k-quote); } .swatch.list { background: var(--k-list); } .swatch.broken { background: var(--k-broken); }

.post { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; min-width: 0;
        padding: clamp(20px, 4vw, 44px); font: 17px/1.72 var(--read); overflow-wrap: break-word; }
.post > :first-child { margin-top: 0; }
.post h1 { font: 600 clamp(26px, 3.2vw, 34px)/1.2 var(--ui); margin: 0 0 6px; text-wrap: balance; }
.post h1 + p { color: var(--muted); font-size: 1.08em; margin-top: 0; }
.post h2 { font: 600 22px/1.3 var(--ui); margin: 2em 0 .5em; text-wrap: balance; }
.post p { margin: 0 0 1.05em; }
.post a { color: var(--accent); }
.post img { display: block; width: 100%; height: auto; border-radius: 10px; margin: 1.4em 0 .4em; background: var(--surface-2); }
.post p > img + em { display: block; text-align: center; font-size: .86em; color: var(--muted); margin-bottom: 1.4em; }
.post ul, .post ol { margin: 0 0 1.1em; padding-left: 1.3em; } .post li { margin: .3em 0; }
.post blockquote { margin: 1.5em 0; padding: .1em 0 .1em 1.1em; border-left: 4px solid var(--k-quote); font-size: 1.14em; font-style: italic; line-height: 1.5; }
.post blockquote p { margin: .3em 0; }
.table-wrap { overflow-x: auto; margin: 1.4em 0; border: 1px solid var(--border); border-radius: 10px; }
.post table { border-collapse: collapse; width: 100%; min-width: 460px; font: 14.5px/1.45 var(--ui); }
.post th, .post td { padding: 9px 13px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
.post thead th { background: var(--surface-2); font-weight: 600; }
.post tbody tr:last-child td { border-bottom: 0; }
.post tbody tr:nth-child(even) td { background: color-mix(in srgb, var(--surface-2) 70%, transparent); }
.not-a-table { margin: 1.4em 0; padding: 12px 14px; border: 2px dashed var(--bad); border-radius: 10px; background: var(--bad-soft);
               font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.not-a-table::before { content: "Not a valid table, so it shows as plain text"; display: block; margin-bottom: 6px; white-space: normal;
                       font: 600 12px/1.4 var(--ui); color: var(--bad); text-transform: uppercase; letter-spacing: .05em; }
.missing-picture { display: block; margin: 1.2em 0; padding: 40px 16px; text-align: center; border: 1px dashed var(--border); border-radius: 10px;
                   color: var(--muted); font: 14px var(--ui); }

.notes { color: var(--muted); font-size: 13px; max-width: 90ch; display: grid; gap: 6px; }
.notes p { margin: 0; }
.js .set[hidden] { display: none; }
@media (prefers-reduced-motion: no-preference) { .tab, .views button { transition: background-color .15s; } }
"""

JS = """
(function () {
  document.documentElement.classList.add('js');
  var sets = [].slice.call(document.querySelectorAll('.set'));
  var tabs = [].slice.call(document.querySelectorAll('.tab'));
  function show(id) {
    sets.forEach(function (s) { s.hidden = s.id !== id; });
    tabs.forEach(function (t) { t.setAttribute('aria-current', String(t.getAttribute('data-tab') === id)); });
  }
  tabs.forEach(function (t) { t.addEventListener('click', function () { show(t.getAttribute('data-tab')); }); });
  var wanted = location.hash.slice(1);
  show(sets.some(function (s) { return s.id === wanted; }) ? wanted : sets[0].id);
  [].slice.call(document.querySelectorAll('.views button')).forEach(function (b) {
    b.addEventListener('click', function () {
      var set = b.closest('.set');
      set.setAttribute('data-view', b.getAttribute('data-view'));
      [].slice.call(set.querySelectorAll('.views button')).forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
    });
  });
})();
"""

LEGEND = (("text", "Text"), ("heading", "Heading"), ("image", "Picture"), ("table", "Table"), ("quote", "Pull quote"),
          ("list", "List"), ("broken", "Table that does not render"))


def _seconds(value: Optional[str]) -> str:
    return f"{float(value):.1f} s" if value else "n/a"


def _model(value: Optional[str]) -> str:
    return value.split("/")[-1].strip() if value else "n/a"


def _outline_model(header: Dict[str, Optional[str]]) -> str:
    """The model that wrote the outline: the evaluation header says 'same' (or nothing) when it was the drafting model."""
    outline = header.get("outline_model")
    return _model(header.get("model") if outline in (None, "same") else outline)


def _glance_rows(posts: List[Post]) -> List[Tuple[str, List[str], bool]]:
    """(label, one value per post, warn if not zero). Counts come from the Markdown, timings from the header."""
    def facts(key: str) -> List[str]:
        return [f"{p.facts[key]:,}" for p in posts]
    rows: List[Tuple[str, List[str], bool]] = []
    if any(p.header for p in posts):
        rows += [("Outline written by", [_outline_model(p.header) for p in posts], False),
                 ("First text on screen", [_seconds(p.header.get("first_content_seconds")) for p in posts], False),
                 ("Whole post", [_seconds(p.header.get("total_seconds")) for p in posts], False)]
    rows += [("Words", facts("words"), False), ("Tables", facts("tables"), False), ("Pull quotes", facts("quotes"), False),
             ("Pictures", facts("pictures"), False), ("Lists", facts("lists"), False),
             ("Longest run of text without a visual break", [f"{p.facts['longest_prose_run']} words" for p in posts], False)]
    if any(p.facts["broken_tables"] for p in posts):
        rows.append(("Tables that do not render", facts("broken_tables"), True))
    return rows


def _glance(posts: List[Post]) -> str:
    head = "".join(f"<th scope='col'>{html.escape(p.role)}</th>" for p in posts)
    body = []
    for label, values, warn in _glance_rows(posts):
        cells = []
        for i, value in enumerate(values):
            cls = ""
            if len(posts) > 1 and value != values[0] and i > 0:
                cls = " class='changed'"
            if warn and value != "0":
                cls = " class='bad'"
            cells.append(f"<td{cls}>{html.escape(value)}</td>")
        body.append(f"<tr><th scope='row'>{html.escape(label)}</th>{''.join(cells)}</tr>")
    return (f"<div class='glance-wrap'><table class='glance'><thead><tr><th scope='col'>At a glance</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _rhythm(post: Post) -> str:
    f = post.facts
    label = (f"Page rhythm: {f['pictures']} pictures, {f['tables']} tables, {f['quotes']} pull quotes, {f['lists']} lists"
             + (f", {f['broken_tables']} table that does not render" if f["broken_tables"] else ""))
    segments = "".join(f"<span class='{kind}' style='flex-grow:{weight}' title='{html.escape(tip)}'></span>"
                       for kind, weight, tip in post.rhythm)
    return f"<div class='rhythm' role='img' aria-label='{html.escape(label)}'>{segments}</div>"


def _column(post: Post) -> str:
    f = post.facts
    summary = (f"{f['tables']} table{'s' if f['tables'] != 1 else ''} · {f['quotes']} pull quote{'s' if f['quotes'] != 1 else ''} · "
               f"{f['pictures']} picture{'s' if f['pictures'] != 1 else ''} · {f['words']:,} words")
    return (f"<div class='col {post.role.lower()}'><div class='col-head'><span class='role'>{html.escape(post.role)}</span>"
            f"<p class='summary'>{html.escape(summary)}</p>{_rhythm(post)}</div><article class='post'>{post.body}</article></div>")


def _set(index: int, ps: PostSet) -> str:
    brief = next((p.header.get("brief") for p in ps.posts if p.header.get("brief")), None)
    toggle = ""
    if len(ps.posts) > 1:
        toggle = ("<div class='views' role='group' aria-label='What to show'>"
                  "<button type='button' data-view='both' aria-pressed='true'>Side by side</button>"
                  + "".join(f"<button type='button' data-view='{html.escape(p.role.lower())}' aria-pressed='false'>{html.escape(p.role)} only</button>"
                            for p in ps.posts) + "</div>")
    single = " data-single" if len(ps.posts) == 1 else ""
    return (f"<section class='set' id='{ps.slug}' data-view='both'{single} aria-label='{html.escape(ps.label)}'>"
            f"<div class='set-head'><div><h2>{html.escape(ps.label)}</h2>"
            + (f"<p class='brief'>Brief: {html.escape(brief)}</p>" if brief else "") + f"</div>{toggle}</div>"
            f"{_glance(ps.posts)}<div class='cols'>{''.join(_column(p) for p in ps.posts)}</div></section>")


def render_page(sets: List[PostSet], title: str, intro: str) -> str:
    tabs = ""
    if len(sets) > 1:
        tabs = ("<nav class='tabs' aria-label='Briefs'>"
                + "".join(f"<button type='button' class='tab' data-tab='{s.slug}' aria-current='{'true' if i == 0 else 'false'}'>{html.escape(s.label)}</button>"
                          for i, s in enumerate(sets)) + "</nav>")
    legend = "".join(f"<li><span class='swatch {k}'></span>{html.escape(name)}</li>" for k, name in LEGEND)
    return f"""<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
<div class="page">
  <header class="top">
    <p class="eyebrow">Generated blog posts</p>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(intro)}</p>
    <ul class="legend" aria-label="Colours in the page rhythm bar"><li>Page rhythm, top to bottom:</li>{legend}</ul>
  </header>
  {tabs}
  <main>{''.join(_set(i, s) for i, s in enumerate(sets))}</main>
  <footer class="notes">
    <p>Rendered with a standard CommonMark and GitHub-table renderer. The editor styles posts its own way; how it shows a picture
    followed by an italic caption line is still an open item. Pictures are resized for this page; the type here is for reading.
    Each side is one live run, so some difference in wording is ordinary run-to-run variation.</p>
    <p>The rhythm bar shows every block of a post in order, sized by how much room it takes: hover a segment to see what it is.</p>
  </footer>
</div>
<script>{JS}</script>
"""


def build(compares: List[List[str]], posts: List[List[str]], title: str, intro: str) -> str:
    cache: Dict[Path, Optional[str]] = {}
    sets: List[PostSet] = []
    for label, previous, current in compares:
        sets.append(PostSet(label, [load_post("Previous", Path(previous), cache), load_post("Current", Path(current), cache)]))
    for label, path in posts:
        sets.append(PostSet(label, [load_post("Post", Path(path), cache)]))
    if not sets:
        raise ValueError("Nothing to render: give at least one --compare or --post")
    if len({s.slug for s in sets}) != len(sets):
        raise ValueError("Each brief needs its own label: two of them would share a tab")
    return render_page(sets, title, intro)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="The HTML file to write")
    parser.add_argument("--title", default="Blog output review")
    parser.add_argument("--intro", default="Generated posts as a reader would see them, with what was measured beside each.")
    parser.add_argument("--compare", nargs=3, action="append", default=[], metavar=("LABEL", "PREVIOUS.md", "CURRENT.md"),
                        help="Two generations of one brief, side by side (repeat for more briefs)")
    parser.add_argument("--post", nargs=2, action="append", default=[], metavar=("LABEL", "POST.md"), help="A single post")
    args = parser.parse_args()
    if not args.compare and not args.post:
        parser.error("give at least one --compare or --post")
    for _, *paths in args.compare + args.post:
        for p in paths:
            if not Path(p).is_file():
                parser.error(f"no such file: {p}")
    try:
        page = build(args.compare, args.post, args.title, args.intro)
    except ValueError as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"Wrote {args.out} ({len(page) / 1024 / 1024:.1f} MB, {len(args.compare) + len(args.post)} brief(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
