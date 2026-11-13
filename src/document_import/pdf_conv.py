"""
PDF to Markdown, from the text layer, with layout heuristics (pdfminer.six, MIT).

A PDF stores glyphs at positions, not paragraphs, so structure is inferred and approximate. What is done:

  * reading order comes from pdfminer's layout analysis (columns are kept apart);
  * the most common font size is the body text; a short block set larger than that, or all in bold, is a heading, and the distinct heading
    sizes map to # ## ### ####;
  * running headers, footers and page numbers (the same text in the top or bottom margin on most pages) are removed;
  * lines are joined into paragraphs, words split by a line-end hyphen are rejoined, and a paragraph that runs on across a page break is one;
  * bullets and numbers at the start of a line become list items;
  * Markdown punctuation in the text is escaped; ligatures are normalised; invisible characters are removed.

What is not done, and is said in the result: tables are not rebuilt (their text comes out cell by cell and the pages are named), pictures are not
imported, and a PDF made of scanned images has no text to read, so it is refused (no OCR). A password-protected PDF, or one whose permissions forbid
copying text, is refused.
"""
from __future__ import annotations

import io
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .errors import ImportRefused
from .html_to_md import Converted
from .mdutil import escape_inline, escape_line_starts, strip_invisible

MAX_PAGES = 200
TIME_BUDGET_SECONDS = 20.0
MAX_OUTPUT_CHARS = 400_000

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", " ": " ", "\xa0": " "})
_BULLET = re.compile(r"^\s*(?:[•●◦▪▫■□◆◇‣⁃∙·–—*]|-(?=\s))\s+")
_NUMBERED = re.compile(r"^\s*(\d{1,3})[.)]\s+(?=\S)")
_LETTERED = re.compile(r"^\s*\(?([a-zA-Z])[.)]\s+(?=\S)")
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$|^-\s*\d{1,4}\s*-$|^[ivxlc]{1,6}$", re.I)
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+\S")
_SECTION_LABEL = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[A-Z]\.?|[IVX]{1,5}\.?)$")
_SENTENCE_END = re.compile(r"[.!?:;…”\"')\]]\s*$")
_JUNK_TITLE = re.compile(r"^(?:microsoft word - |untitled|document\d*|\w+\.(?:docx?|pdf|tex|indd)$)", re.I)


def _refuse(code: str, message: str, status: int = 422) -> ImportRefused:
    return ImportRefused(status, code, message)


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    x0: float
    x1: float
    y0: float
    y1: float
    page: int
    box: int


@dataclass
class _Block:
    kind: str                      # "heading" | "para" | "item"
    text: str
    page: int
    level: int = 0                 # heading level, or list nesting
    marker: str = "-"              # "-" or "1."
    size: float = 0.0
    bold_only: bool = False
    column: int = 0


@dataclass
class _Page:
    number: int
    width: float
    height: float
    lines: List[_Line] = field(default_factory=list)
    images: int = 0
    rotated: int = 0


# ---------------------------------------------------------------------------------------------------------------------------------------
# Reading the layout
# ---------------------------------------------------------------------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text.translate(_LIGATURES))
    text, _ = strip_invisible(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _read_pages(data: bytes, max_pages: int, budget: float) -> Tuple[List[_Page], int, str, bool]:
    """(pages, total page count if known, metadata title, stopped early because the time budget ran out)."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTChar, LTFigure, LTImage, LTTextContainer, LTTextLine
    from pdfminer.pdfdocument import PDFDocument, PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfpage import PDFTextExtractionNotAllowed
    from pdfminer.pdftypes import resolve1

    total, title = 0, ""
    try:
        document = PDFDocument(PDFParser(io.BytesIO(data)))
        try:
            total = int(resolve1(resolve1(document.catalog["Pages"]).get("Count", 0)) or 0)
        except Exception:
            total = 0
        try:
            info = resolve1(document.info[0].get("Title")) if document.info else None
            title = info.decode("utf-16", "ignore") if isinstance(info, bytes) and info.startswith((b"\xfe\xff", b"\xff\xfe")) \
                else info.decode("latin-1", "ignore") if isinstance(info, bytes) else ""
        except Exception:
            title = ""
    except PDFPasswordIncorrect:
        raise _refuse("pdf_encrypted", "This PDF is password-protected. Remove the password and import it again.") from None
    except Exception as exc:
        if "ncrypt" in type(exc).__name__ or "ncrypt" in str(exc):
            raise _refuse("pdf_encrypted", "This PDF is encrypted in a way that cannot be read. Save an unprotected copy and import that.") from None
        raise _refuse("pdf_unreadable", "This PDF is damaged or could not be read.") from None

    pages: List[_Page] = []
    started = time.monotonic()
    stopped = False
    laparams = LAParams(line_margin=0.5, char_margin=2.0, word_margin=0.1, boxes_flow=0.5, detect_vertical=False)
    try:
        for number, layout in enumerate(extract_pages(io.BytesIO(data), laparams=laparams, maxpages=max_pages, caching=True), start=1):
            if time.monotonic() - started > budget:
                stopped = True
                break
            page = _Page(number, layout.width, layout.height)
            box_id = 0
            for obj in layout:
                if isinstance(obj, (LTImage, LTFigure)):
                    page.images += 1
                elif isinstance(obj, LTTextContainer):
                    box_id += 1
                    for line in obj:
                        if not isinstance(line, LTTextLine):
                            continue
                        chars = [c for c in line if isinstance(c, LTChar) and c.get_text().strip()]
                        text = _clean(line.get_text())
                        if not text or not chars:
                            continue
                        if sum(1 for c in chars if not c.upright) > 0.5 * len(chars):          # text turned on its side (the arXiv stamp, rotated labels)
                            page.rotated += 1
                            continue
                        bold = sum("bold" in c.fontname.lower() or "black" in c.fontname.lower() or "heavy" in c.fontname.lower() for c in chars) > 0.6 * len(chars)
                        size = statistics.median(c.size for c in chars)
                        page.lines.append(_Line(text, round(size * 2) / 2, bold, line.x0, line.x1, line.y0, line.y1, number, box_id))
            pages.append(page)
    except PDFTextExtractionNotAllowed:
        raise _refuse("pdf_copy_protected", "This PDF's permissions do not allow its text to be copied, so it cannot be imported.") from None
    except PDFPasswordIncorrect:
        raise _refuse("pdf_encrypted", "This PDF is password-protected. Remove the password and import it again.") from None
    except RecursionError:
        raise _refuse("pdf_unreadable", "This PDF is too deeply structured to be read.") from None
    except Exception as exc:
        if pages:                                           # a damaged later page should not lose the pages already read
            return pages, total, title, True
        if "ncrypt" in type(exc).__name__:
            raise _refuse("pdf_encrypted", "This PDF is encrypted in a way that cannot be read. Save an unprotected copy and import that.") from None
        raise _refuse("pdf_unreadable", "This PDF is damaged or could not be read.") from None
    return pages, total, title, stopped


# ---------------------------------------------------------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------------------------------------------------------

def _body_size(pages: List[_Page]) -> float:
    weight: Counter = Counter()
    for page in pages:
        for line in page.lines:
            weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 10.0


def _margin_noise(pages: List[_Page]) -> set:
    """
    Lines that are page furniture, as (page, box, text) keys: a bare page number in the top or bottom 8% of the page, and text that repeats (digits
    aside) in the top or bottom 5.5% on at least 30% of the pages (a running header or footer; needs 3 pages or more). The repeated-text zone is
    narrower so that a body paragraph starting near the top edge of a page is never taken for a header.
    """
    keys: Dict[str, set] = {}
    candidates: List[Tuple[_Line, str]] = []
    noise = set()
    for page in pages:
        for line in page.lines:
            if (line.y1 >= page.height * 0.92 or line.y0 <= page.height * 0.08) and _PAGE_NUMBER.match(line.text.strip()):
                noise.add((line.page, line.box, line.text))
            if line.y1 >= page.height * 0.945 or line.y0 <= page.height * 0.055:
                key = re.sub(r"\d+", "#", line.text.lower())
                candidates.append((line, key))
                keys.setdefault(key, set()).add(page.number)
    threshold = max(2, math.ceil(0.3 * len(pages))) if len(pages) >= 3 else 99
    for line, key in candidates:
        if len(keys[key]) >= threshold:
            noise.add((line.page, line.box, line.text))
    return noise


def _split_word(before: str, after: str) -> bool:
    """A hyphen ending a line: is the word being wrapped (inter-|national) or is the hyphen part of the word (English-|to-German, x-|ray)?"""
    first = after.split(" ", 1)[0]
    fragment = re.split(r"[\s-]", before[:-1])[-1]
    return "-" not in first and len(fragment) >= 2


def _join_lines(lines: List[str]) -> str:
    out = ""
    for text in lines:
        if not out:
            out = text
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and text[:1].isalnum():
            # a hyphen at a line end: a wrapped word loses it (inter-|national), a compound keeps it (English-|to-German); neither gets a space
            out = out[:-1] + text if text[:1].islower() and _split_word(out, text) else out + text
        else:
            out += " " + text
    return out


def _classify(lines: List[_Line], body: float) -> bool:
    """Is this run of lines heading text?"""
    text = _join_lines([l.text for l in lines])
    size = statistics.median(l.size for l in lines)
    bold = all(l.bold for l in lines)
    if len(text) > 170 or len(lines) > 4 or _PAGE_NUMBER.match(text):
        return False
    if size >= body * 1.18:
        return True
    if bold and size >= body * 0.95 and len(text) <= 100 and len(lines) <= 2 and (not text.rstrip().endswith(".") or _SECTION_NUMBER.match(text)):
        return True
    return False


def _split_box(lines: List[_Line], body: float, noise: set) -> List[_Block]:
    lines = [l for l in sorted(lines, key=lambda l: -l.y1) if (l.page, l.box, l.text) not in noise]
    blocks: List[_Block] = []
    run: List[_Line] = []

    def flush():
        if not run:
            return
        heading = _classify(run, body)
        size = statistics.median(l.size for l in run)
        if heading:
            blocks.append(_Block("heading", _join_lines([l.text for l in run]), run[0].page, size=size, bold_only=size < body * 1.18,
                                 column=int(run[0].x0 // 200)))
        else:
            para: List[_Line] = []
            left = min(l.x0 for l in run)

            def end_para():
                if para:
                    blocks.append(_Block("para", _join_lines([l.text for l in para]), run[0].page, size=size, column=int(left // 200)))
                    para.clear()

            for line in run:
                bullet, numbered, lettered = _BULLET.match(line.text), _NUMBERED.match(line.text), _LETTERED.match(line.text)
                if bullet or numbered or (lettered and len(run) > 1):
                    end_para()
                    match = bullet or numbered or lettered
                    indent = 1 if line.x0 - left > 1.5 * max(body, 1) else 0
                    blocks.append(_Block("item", line.text[match.end():].strip(), run[0].page, level=indent,
                                         marker=f"{numbered.group(1)}." if numbered else "-", size=size, column=int(left // 200)))
                    continue
                if para and blocks and blocks[-1].kind == "item" and False:
                    pass
                if (not para and blocks and blocks[-1].kind == "item" and blocks[-1].page == run[0].page and line.x0 > left - 1
                        and not _SENTENCE_END.search(blocks[-1].text) and not para):
                    blocks[-1].text = _join_lines([blocks[-1].text, line.text])            # a wrapped list-item line
                    continue
                if para and line.x0 - para[0].x0 > 1.2 * max(body, 1) and _SENTENCE_END.search(para[-1].text):
                    end_para()                                                              # first-line indent after a sentence end
                para.append(line)
            end_para()
        run.clear()

    previous: Optional[bool] = None
    for line in lines:
        kind = line.size >= body * 1.18 or (line.bold and line.size >= body * 0.95)
        if run and (kind != previous or (kind and abs(line.size - run[-1].size) > 0.6)):
            flush()
        run.append(line)
        previous = kind
    flush()
    return blocks


def _levels(blocks: List[_Block]) -> None:
    headings = [b for b in blocks if b.kind == "heading"]
    sized = sorted({b.size for b in headings if not b.bold_only}, reverse=True)
    for b in headings:
        b.level = (min(sized.index(b.size) + 1, 4) if b.size in sized else min(max(len(sized) + 1, 2), 6)) if not b.bold_only else min(max(len(sized) + 1, 2), 6)


def _merge_runs(blocks: List[_Block]) -> List[_Block]:
    """A paragraph that carries on after a box or page boundary is one paragraph."""
    merged: List[_Block] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if (prev and prev.kind == "para" and b.kind == "para" and not _SENTENCE_END.search(prev.text) and b.text[:1].islower()
                and abs(prev.size - b.size) < 0.6):
            prev.text = _join_lines([prev.text, b.text])
        elif prev and prev.page == b.page and b.kind == "heading" and prev.kind in ("para", "heading") and _SECTION_LABEL.match(prev.text):
            b.text = f"{prev.text} {b.text}"                                                # "1" and "Introduction" were laid out as two boxes
            merged[-1] = b
        else:
            merged.append(b)
    return merged


_FRONT_MATTER_KEYWORD = re.compile(r"abstract|summary|introduction|contents|overview|preface|foreword", re.I)


def _demote_bylines(blocks: List[_Block]) -> int:
    """
    Author names and affiliations are set larger than body text on a paper's first page and look like headings. After the title, a short heading
    that is not an Abstract/Introduction-style heading and is followed by more short text (the next author, an affiliation) rather than by a long
    paragraph is a byline, so it becomes plain text. A heading that leads straight into a paragraph stays a heading.
    """
    first_heading = next((i for i, b in enumerate(blocks) if b.kind == "heading" and b.page == 1), None)
    if first_heading is None:
        return 0
    demoted = 0
    for i in range(first_heading + 1, len(blocks)):
        b = blocks[i]
        if b.page != 1 or (b.kind == "para" and len(b.text) > 200):
            break
        following = blocks[i + 1] if i + 1 < len(blocks) else None
        leads_into_text = following is not None and following.kind == "para" and len(following.text) > 200
        if (b.kind == "heading" and len(b.text.split()) <= 6 and not _FRONT_MATTER_KEYWORD.search(b.text) and not _SECTION_NUMBER.match(b.text)
                and not leads_into_text):
            b.kind = "para"
            demoted += 1
    return demoted


def _reading_order(page: _Page) -> List[int]:
    """
    Box ids in reading order. pdfminer's own order is kept for ordinary pages. A two-column page (at least three boxes wholly in each half) is read
    column by column: full-width boxes (title, abstract, a wide figure's caption) split the page into bands, and inside a band the left column comes
    before the right one. Without this, a heading at the top of a column can come after the other column's text.
    """
    boxes: Dict[int, List[float]] = {}                                  # id -> [x0, x1, y0, y1]
    for line in page.lines:
        b = boxes.setdefault(line.box, [line.x0, line.x1, line.y0, line.y1])
        b[0], b[1], b[2], b[3] = min(b[0], line.x0), max(b[1], line.x1), min(b[2], line.y0), max(b[3], line.y1)
    original = list(dict.fromkeys(line.box for line in page.lines))
    mid, tol = page.width / 2, page.width * 0.03
    left = {i for i, b in boxes.items() if b[1] <= mid + tol}
    right = {i for i, b in boxes.items() if b[0] >= mid - tol}
    span = (set(boxes) - left - right) | (left & right)                  # wider than a column, or sitting on the centre line (a page number)
    left, right = left - span, right - span
    if len(left) < 3 or len(right) < 3 or len(span) > 0.4 * len(boxes):
        return original
    ordered: List[int] = []
    band_left: List[int] = []
    band_right: List[int] = []

    def flush():
        ordered.extend(sorted(band_left, key=lambda i: -boxes[i][3]))
        ordered.extend(sorted(band_right, key=lambda i: -boxes[i][3]))
        band_left.clear()
        band_right.clear()

    for i in sorted(boxes, key=lambda i: -(boxes[i][2] + boxes[i][3]) / 2):
        if i in span:
            flush()
            ordered.append(i)
        elif i in left:
            band_left.append(i)
        else:
            band_right.append(i)
    flush()
    return ordered


def _table_pages(pages: List[_Page]) -> List[int]:
    """Pages where several rows each hold three or more text pieces side by side: a table or a column chart. Not rebuilt; named in a warning."""
    found = []
    for page in pages:
        rows: Dict[int, set] = {}
        for line in page.lines:
            rows.setdefault(round((line.y0 + line.y1) / 2 / 3), set()).add(round(line.x0 / 12))
        if sum(1 for xs in rows.values() if len(xs) >= 3) >= 4:
            found.append(page.number)
    return found


def _render(blocks: List[_Block], max_chars: int) -> Tuple[str, bool]:
    out: List[str] = []
    chars = 0
    truncated = False
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "heading":
            text = escape_inline(b.text).strip()
            piece = f"{'#' * max(1, b.level)} {text}" if text else ""
            i += 1
        elif b.kind == "item":
            items = []
            while i < len(blocks) and blocks[i].kind == "item":
                it = blocks[i]
                if items and (it.marker == "-") != (blocks[i - 1].marker == "-"):          # a bulleted list then a numbered one: two lists
                    items.append("")
                items.append(f"{'  ' * it.level}{it.marker} {escape_inline(it.text)}")
                i += 1
            piece = "\n".join(items)
        else:
            piece = escape_line_starts(escape_inline(b.text))
            i += 1
        if not piece.strip():
            continue
        if chars + len(piece) > max_chars:
            truncated = True
            break
        chars += len(piece) + 2
        out.append(piece)
    return "\n\n".join(out) + "\n", truncated


def _good_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", strip_invisible(raw or "")[0]).strip()
    return "" if len(title) < 3 or len(title) > 200 or _JUNK_TITLE.match(title) else title


def pdf_to_markdown(data: bytes, *, max_pages: int = MAX_PAGES, time_budget: float = TIME_BUDGET_SECONDS, max_chars: int = MAX_OUTPUT_CHARS) -> Converted:
    if b"%PDF-" not in data[:1024]:
        raise _refuse("not_a_pdf", "This file is not a PDF.")
    pages, total, meta_title, stopped = _read_pages(data, max_pages, time_budget)
    if not pages:
        raise _refuse("no_text", "No pages could be read from this PDF.")
    text_chars = sum(len(l.text) for p in pages for l in p.lines)
    images = sum(p.images for p in pages)
    if text_chars < 25 * len(pages):
        if images:
            raise _refuse("scanned_pdf", "This PDF is made of scanned images and has no text to read. OCR is not supported: run it through an OCR "
                                         "tool first, or import a text version.")
        raise _refuse("no_text", "No text was found in this PDF.")

    body = _body_size(pages)
    noise = _margin_noise(pages)
    blocks: List[_Block] = []
    for page in pages:
        boxes: Dict[int, List[_Line]] = {}
        for line in page.lines:
            boxes.setdefault(line.box, []).append(line)
        for box in _reading_order(page):
            blocks.extend(_split_box(boxes[box], body, noise))
    bold = [b for b in blocks if b.kind == "heading" and b.bold_only]
    if bold and len(bold) > 0.35 * len(blocks):                                   # a document where bold lines are a third of it: bold is not a heading marker
        for b in bold:
            b.kind = "para"
    blocks = _merge_runs(blocks)
    _demote_bylines(blocks)
    _levels(blocks)
    markdown, truncated = _render(blocks, max_chars)
    if not markdown.strip():
        raise _refuse("no_text", "No text was found in this PDF.")

    warnings: List[dict] = []
    tables = _table_pages(pages)
    if tables:
        shown = ", ".join(str(n) for n in tables[:8]) + (" ..." if len(tables) > 8 else "")
        warnings.append({"code": "tables_not_rebuilt", "message": f"Page(s) {shown} look like they contain a table or columns of figures. Tables are not "
                                                                  "rebuilt from a PDF: their text appears cell by cell. Check those parts."})
    if images:
        warnings.append({"code": "images_omitted", "message": f"{images} image(s) were left out: pictures are not imported."})
    if stopped or (total and total > len(pages)):
        warnings.append({"code": "pages_truncated", "message": f"Only the first {len(pages)} of {total or 'more'} pages were imported "
                                                               f"({'the time limit was reached' if stopped else 'page limit'})."})
    if truncated:
        warnings.append({"code": "output_truncated", "message": "The converted text was cut at the size limit."})
    first = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), "")
    title = _good_title(meta_title) or _good_title(first)
    stats = {"pages": len(pages), "headings": sum(1 for b in blocks if b.kind == "heading"), "list_items": sum(1 for b in blocks if b.kind == "item"),
             "images": images, "table_pages": len(tables), "rotated_lines_skipped": sum(p.rotated for p in pages)}
    return Converted(markdown, title=title, stats=stats, warnings=warnings)
