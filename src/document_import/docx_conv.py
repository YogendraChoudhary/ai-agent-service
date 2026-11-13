"""
Word (.docx) to Markdown.

mammoth (BSD-2) reads the file's structure (styles, lists, tables, links, footnotes, tracked changes) and writes HTML; html_to_md turns that
into Markdown. This module is the safe front door:

  * the zip is checked before anything is read (entry count, total size, ratio, encryption, that it really is a Word document);
  * text a reader cannot see is removed before conversion: runs formatted as hidden (w:vanish) or at 1 pt or smaller. White-coloured text is
    reported, not removed, because white text on a dark heading band is legitimate;
  * tracked changes are accepted (inserted text kept, deleted text left out) and the user is told; so are text boxes (which mammoth reads only in
    some forms), equations and comments, which it leaves out, and pictures, which are not imported;
  * a Title paragraph becomes the "#" heading, and when the document also has Heading 1s they move down one level, so there is one title.

Macros are never run: nothing here executes document content.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Tuple

from lxml import etree

from .errors import ImportRefused
from .html_to_md import Converted, html_to_markdown

MAX_ENTRIES = 3_000
MAX_UNCOMPRESSED = 150 * 1024 * 1024
MAX_PART = 20 * 1024 * 1024                    # word/document.xml, footnotes.xml: mammoth builds a tree of these in memory
MAX_RATIO = 250                                # uncompressed / compressed, for parts over 2 MB

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_NOTES = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")

STYLE_MAP = """
p[style-name='Title'] => h1.doc-title:fresh
p[style-name='Quote'] => blockquote:fresh
p[style-name='Intense Quote'] => blockquote:fresh
p[style-name='Block Text'] => blockquote:fresh
p[style-name='Code'] => pre.docx-code:fresh
p[style-name='Source Code'] => pre.docx-code:fresh
p[style-name='Preformatted Text'] => pre.docx-code:fresh
p[style-name='HTML Preformatted'] => pre.docx-code:fresh
r[style-name='Code Char'] => code
r[style-name='Source Code Char'] => code
r[style-name='Verbatim Char'] => code
r[style-name='HTML Code'] => code
"""


def _refuse(code: str, message: str, status: int = 422) -> ImportRefused:
    return ImportRefused(status, code, message)


# ---------------------------------------------------------------------------------------------------------------------------------------
# The zip
# ---------------------------------------------------------------------------------------------------------------------------------------

def _open_package(data: bytes) -> zipfile.ZipFile:
    if data.startswith(_OLE_MAGIC):
        raise _refuse("legacy_word_format", "This is an old Word file (.doc) or a password-protected Word file, which cannot be read. "
                                            "Open it in Word and save it as .docx (without a password), then import that.")
    if not data.startswith(b"PK"):
        raise _refuse("not_a_docx", "This file is not a Word (.docx) document.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError, OSError):
        raise _refuse("not_a_docx", "This file is damaged or is not a Word (.docx) document.") from None
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise _refuse("docx_too_complex", f"This document has {len(infos):,} internal parts; the limit is {MAX_ENTRIES:,}.", 413)
    total = 0
    for info in infos:
        if info.flag_bits & 0x1:
            raise _refuse("docx_encrypted", "This Word file is password-protected. Remove the password in Word and import it again.")
        total += info.file_size
        if info.file_size > 2 * 1024 * 1024 and info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
            raise _refuse("docx_too_complex", "This document is not safe to read: a part of it expands to far more than its stored size.", 413)
    if total > MAX_UNCOMPRESSED:
        raise _refuse("docx_too_complex", f"This document expands to {total // (1024 * 1024)} MB, over the {MAX_UNCOMPRESSED // (1024 * 1024)} MB limit.", 413)
    names = {i.filename for i in infos}
    if "word/document.xml" not in names:
        kind = ("a PowerPoint file" if any(n.startswith("ppt/") for n in names) else "an Excel file" if any(n.startswith("xl/") for n in names)
                else "an OpenDocument file" if "mimetype" in names or "content.xml" in names else "")
        raise _refuse("not_a_docx", f"This is {kind}, not a Word document." if kind else "This file is not a Word (.docx) document.")
    for name in _NOTES:
        if name in names and zf.getinfo(name).file_size > MAX_PART:
            raise _refuse("docx_too_complex", f"{name.split('/')[-1]} is over {MAX_PART // (1024 * 1024)} MB uncompressed, which this import does not accept.", 413)
    return zf


# ---------------------------------------------------------------------------------------------------------------------------------------
# Text a reader cannot see
# ---------------------------------------------------------------------------------------------------------------------------------------

def _q(tag: str) -> str:
    return f"{{{_W}}}{tag}"


def _is_hidden_run(run) -> bool:
    props = run.find(_q("rPr"))
    if props is None:
        return False
    vanish = props.find(_q("vanish"))
    if vanish is not None and vanish.get(_q("val"), "true").lower() not in ("0", "false", "off"):
        return True
    size = props.find(_q("sz"))
    if size is not None:
        try:
            return int(size.get(_q("val"), "99")) <= 2           # half-points: 2 is 1 pt
        except ValueError:
            return False
    return False


def _shaded(el) -> bool:
    """Is this run, its paragraph or its table cell filled with a colour (so white text on it can be read)?"""
    node = el
    while node is not None and node.tag not in (_q("body"),):
        for holder in (node.find(_q("rPr")), node.find(_q("pPr")), node.find(_q("tcPr"))):
            if holder is not None:
                shd = holder.find(_q("shd"))
                if shd is not None and (shd.get(_q("fill")) or "auto").lower() not in ("auto", "ffffff", "none"):
                    return True
        node = node.getparent()
    return False


def _clean_part(xml: bytes) -> Tuple[bytes, int, int, int]:
    """(xml without hidden runs, hidden runs removed, characters in them, white-text runs that were kept)."""
    if b"vanish" not in xml and b'w:sz w:val="' not in xml and b"FFFFFF" not in xml.upper():
        return xml, 0, 0, 0
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, remove_comments=True, remove_pis=True)
    try:
        root = etree.fromstring(xml, parser)
    except etree.XMLSyntaxError:
        return xml, 0, 0, 0
    removed = chars = white = 0
    for run in list(root.iter(_q("r"))):
        if _is_hidden_run(run):
            text = "".join(t.text or "" for t in run.iter(_q("t"))).strip()
            if text:
                removed += 1
                chars += len(text)
                parent = run.getparent()
                if parent is not None:
                    parent.remove(run)
            continue
        colour = run.find(f"{_q('rPr')}/{_q('color')}")
        if colour is not None and (colour.get(_q("val")) or "").upper() == "FFFFFF" and not _shaded(run):
            if len("".join(t.text or "" for t in run.iter(_q("t"))).strip()) >= 20:
                white += 1
    if not removed:
        return xml, 0, 0, white
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True), removed, chars, white


def _repackage(zf: zipfile.ZipFile, replaced: Dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for info in zf.infolist():
            out.writestr(info.filename, replaced[info.filename] if info.filename in replaced else zf.read(info.filename))
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------------------------------------------------

def _count(xml: bytes, *tokens: bytes) -> int:
    return sum(xml.count(t) for t in tokens)


def _shape_headings(html: str) -> Tuple[str, int]:
    """Title paragraphs are h1.doc-title. When real headings exist too, they move down a level so the title is the only "#"."""
    from lxml import html as lxml_html
    if "doc-title" not in html and "docx-code" not in html:
        return html, 0
    root = lxml_html.fragment_fromstring(html, create_parent="div")
    shifted = 0
    titled = bool(root.xpath(".//h1[contains(@class,'doc-title')]"))
    if titled and root.xpath(".//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5][not(contains(@class,'doc-title'))]"):
        for el in root.iter("h5", "h4", "h3", "h2", "h1"):                     # deepest first, so a shifted heading is not shifted twice
            if "doc-title" not in (el.get("class") or ""):
                el.tag = f"h{int(el.tag[1]) + 1}"
                shifted += 1
    for pre in list(root.iter("pre")):                                          # consecutive code paragraphs are one code block
        prev = pre.getprevious()
        if "docx-code" in (pre.get("class") or "") and prev is not None and prev.tag == "pre" and "docx-code" in (prev.get("class") or ""):
            prev.text = (prev.text_content() or "") + "\n" + (pre.text_content() or "")
            for child in list(prev):
                prev.remove(child)
            pre.getparent().remove(pre)
    return lxml_html.tostring(root, encoding="unicode"), shifted


def docx_to_markdown(data: bytes, *, max_chars: int = 400_000) -> Converted:
    zf = _open_package(data)
    signals: Dict[str, int] = {}
    replaced: Dict[str, bytes] = {}
    hidden_runs = hidden_chars = white_runs = 0
    document_xml = b""
    names = {i.filename for i in zf.infolist()}
    for name in _NOTES:
        if name not in names:
            continue
        xml = zf.read(name)
        lowered = xml.lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered:                # Word never writes these; they are how XML entity attacks are made
            raise _refuse("docx_unsafe", "This document contains XML declarations that Word files do not use, so it is not safe to read.")
        if name == "word/document.xml":
            document_xml = xml
        cleaned, runs, chars, white = _clean_part(xml)
        hidden_runs += runs
        hidden_chars += chars
        white_runs += white
        if cleaned is not xml:
            replaced[name] = cleaned
    signals["tracked_insertions"] = _count(document_xml, b"<w:ins ")
    signals["tracked_deletions"] = _count(document_xml, b"<w:del ")
    signals["text_boxes"] = _count(document_xml, b"<w:txbxContent")
    signals["equations"] = _count(document_xml, b"<m:oMath>", b"<m:oMath ")
    signals["comments"] = int("word/comments.xml" in names)
    signals["macros"] = int("word/vbaProject.bin" in names)
    source = _repackage(zf, replaced) if replaced else io.BytesIO(data)

    import mammoth
    try:
        result = mammoth.convert_to_html(source, style_map=STYLE_MAP,
                                         convert_image=mammoth.images.img_element(lambda image: {"src": "docx-image:omitted", "alt": ""}))
    except Exception:                                                           # mammoth raises plain exceptions on bad XML / relationships
        raise _refuse("docx_unreadable", "This Word document could not be read. It may be damaged.") from None
    html, shifted = _shape_headings(result.value)
    converted = html_to_markdown(html, main_content=False, images="omit", footnotes=True, max_chars=max_chars)
    if not converted.markdown.strip():
        raise _refuse("no_text", "No text was found in this Word document.")

    warnings: List[dict] = list(converted.warnings)
    if hidden_runs:
        warnings.append({"code": "hidden_text_removed", "message": f"{hidden_runs} piece(s) of hidden text ({hidden_chars} characters; Word's "
                                                                    "'hidden' formatting or a 1 pt font) were left out."})
    if white_runs:
        warnings.append({"code": "white_text", "message": f"{white_runs} piece(s) of white text on an unshaded background were found. They are "
                                                           "included; check that the text is meant to be read."})
    if signals["tracked_insertions"] or signals["tracked_deletions"]:
        warnings.append({"code": "tracked_changes_accepted", "message": "This document has tracked changes. Inserted text is included and deleted "
                                                                         "text is left out, as if every change were accepted."})
    if signals["text_boxes"]:
        warnings.append({"code": "text_boxes", "message": f"This document has {signals['text_boxes']} text box(es). Text inside text boxes may be missing or in "
                                                          "the wrong place; check those parts."})
    if signals["equations"]:
        warnings.append({"code": "equations_skipped", "message": f"{signals['equations']} Word equation(s) were left out."})
    if signals["comments"]:
        warnings.append({"code": "comments_skipped", "message": "The document's comments were left out."})
    if signals["macros"]:
        warnings.append({"code": "macros_ignored", "message": "This file contains macros. They were not run and are not imported."})
    if converted.stats.get("tables") and converted.stats.get("table_merged_cells"):
        pass                                                                    # html_to_md already warned
    stats = dict(converted.stats)
    stats.update({k: v for k, v in signals.items() if v}, headings_shifted=shifted, hidden_runs=hidden_runs)
    return Converted(converted.markdown, title=converted.title, stats=stats, warnings=warnings)
