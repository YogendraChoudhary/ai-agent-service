"""
Builders for the DOCX and PDF files the import tests (and the live checks) feed to the converters.

No Office or PDF library is installed in this project, and files from the internet must not be test inputs, so these write the formats by
hand: a DOCX is a zip of XML parts, a PDF is objects with a cross-reference table and a content stream per page.
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = (f'xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
      'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"')

PNG_1x1 = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
           b"\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82")


# ---------------------------------------------------------------------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------------------------------------------------------------------

def run(text: str, *, b: bool = False, i: bool = False, u: bool = False, strike: bool = False, code: bool = False, sup: bool = False) -> str:
    props = ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + ("<w:u w:val='single'/>" if u else "") + ("<w:strike/>" if strike else "")
    props += "<w:rStyle w:val='CodeChar'/>" if code else ""
    props += "<w:vertAlign w:val='superscript'/>" if sup else ""
    return f'<w:r>{"<w:rPr>" + props + "</w:rPr>" if props else ""}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def p(content: "str | Sequence[str]", style: Optional[str] = None, *, num: Optional[Tuple[int, int]] = None, align: Optional[str] = None) -> str:
    """A paragraph. `content` is plain text or a list of run()/raw XML pieces; `num` is (numId, level) for list items."""
    body = run(content) if isinstance(content, str) else "".join(content)
    props = ""
    if style:
        props += f'<w:pStyle w:val="{style}"/>'
    if num:
        props += f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>'
    if align:
        props += f'<w:jc w:val="{align}"/>'
    return f'<w:p>{"<w:pPr>" + props + "</w:pPr>" if props else ""}{body}</w:p>'


def h(level: int, text: str) -> str:
    return p(text, f"Heading{level}")


def bullet(text: str, level: int = 0) -> str:
    return p(text, "ListParagraph", num=(1, level))


def numbered(text: str, level: int = 0) -> str:
    return p(text, "ListParagraph", num=(2, level))


def link(text: str, rid: str = "rIdL1") -> str:
    return f'<w:hyperlink r:id="{rid}"><w:r><w:t>{escape(text)}</w:t></w:r></w:hyperlink>'


def footnote_ref(n: int) -> str:
    return f'<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:footnoteReference w:id="{n}"/></w:r>'


def table(rows: Sequence[Sequence[str]], *, header: bool = True, merge_first: bool = False) -> str:
    out = ["<w:tbl><w:tblPr><w:tblW w:w='0' w:type='auto'/></w:tblPr>"]
    for r, cells in enumerate(rows):
        out.append("<w:tr>")
        for c, cell in enumerate(cells):
            span = "<w:tcPr><w:gridSpan w:val='2'/></w:tcPr>" if merge_first and r == 0 and c == 0 else ""
            text = cell if cell.startswith("<w:p") else p([run(cell, b=header and r == 0)])
            out.append(f"<w:tc>{span}{text}</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def image_run(alt: str = "A chart", rid: str = "rIdImg1") -> str:
    return ('<w:r><w:drawing><wp:inline><wp:extent cx="990000" cy="792000"/>'
            f'<wp:docPr id="1" name="Picture 1" descr="{escape(alt)}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="i.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/></pic:blipFill>'
            '<pic:spPr/></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


def _styles() -> str:
    def style(sid: str, name: str, based: str = "", extra: str = "") -> str:
        return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
                f'{f"<w:basedOn w:val=\"{based}\"/>" if based else ""}{extra}</w:style>')
    parts = [style("Normal", "Normal"), style("Title", "Title", "Normal"), style("Subtitle", "Subtitle", "Normal"),
             style("Quote", "Quote", "Normal"), style("ListParagraph", "List Paragraph", "Normal"),
             style("Code", "Code", "Normal")]
    parts += [style(f"Heading{n}", f"heading {n}", "Normal", f"<w:pPr><w:outlineLvl w:val='{n - 1}'/></w:pPr>") for n in range(1, 7)]
    parts.append('<w:style w:type="character" w:styleId="CodeChar"><w:name w:val="Code Char"/></w:style>')
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NS}>{"".join(parts)}</w:styles>'


def _numbering() -> str:
    def level(i: int, fmt: str, text: str) -> str:
        return (f'<w:lvl w:ilvl="{i}"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/><w:lvlText w:val="{text}"/>'
                f'<w:pPr><w:ind w:left="{720 * (i + 1)}"/></w:pPr></w:lvl>')
    bullets = "".join(level(i, "bullet", "•") for i in range(9))
    decimals = "".join(level(i, "decimal", f"%{i + 1}.") for i in range(9))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering {NS}>'
            f'<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>{bullets}</w:abstractNum>'
            f'<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>{decimals}</w:abstractNum>'
            '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num></w:numbering>')


def make_docx(body: "str | Iterable[str]", *, footnotes: Optional[Dict[int, str]] = None, endnotes: Optional[Dict[int, str]] = None,
              comments: bool = False, image: bool = False, header_text: str = "", macros: bool = False, extra: Optional[Dict[str, bytes]] = None,
              compress: int = zipfile.ZIP_DEFLATED, links: Optional[Dict[str, str]] = None) -> bytes:
    body_xml = body if isinstance(body, str) else "".join(body)
    rels = [("rIdStyles", "styles", "styles.xml"), ("rIdNum", "numbering", "numbering.xml")]
    types = ['<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
             '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
             '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>']
    parts: Dict[str, "str | bytes"] = {"word/styles.xml": _styles(), "word/numbering.xml": _numbering()}
    ext_rels = "".join(f'<Relationship Id="{rid}" Type="{R}/hyperlink" Target="{escape(url)}" TargetMode="External"/>'
                       for rid, url in (links or {"rIdL1": "https://example.test/page"}).items())
    if footnotes:
        notes = ('<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
                 '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>')
        notes += "".join(f'<w:footnote w:id="{n}"><w:p><w:r><w:footnoteRef/></w:r>{run(" " + t)}</w:p></w:footnote>' for n, t in footnotes.items())
        parts["word/footnotes.xml"] = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:footnotes {NS}>{notes}</w:footnotes>'
        rels.append(("rIdFn", "footnotes", "footnotes.xml"))
        types.append('<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>')
    if endnotes:
        notes = "".join(f'<w:endnote w:id="{n}"><w:p><w:r><w:endnoteRef/></w:r>{run(" " + t)}</w:p></w:endnote>' for n, t in endnotes.items())
        parts["word/endnotes.xml"] = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:endnotes {NS}>{notes}</w:endnotes>'
        rels.append(("rIdEn", "endnotes", "endnotes.xml"))
        types.append('<Override PartName="/word/endnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml"/>')
    if comments:
        parts["word/comments.xml"] = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:comments {NS}><w:comment w:id="0" w:author="Rev">'
                                      f'{p("Reviewer note")}</w:comment></w:comments>')
        rels.append(("rIdCm", "comments", "comments.xml"))
    if header_text:
        parts["word/header1.xml"] = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr {NS}>{p(header_text)}</w:hdr>'
        rels.append(("rIdHd", "header", "header1.xml"))
    if image:
        parts["word/media/image1.png"] = PNG_1x1
        rels.append(("rIdImg1", "image", "media/image1.png"))
        types.append('<Default Extension="png" ContentType="image/png"/>')
    if macros:
        parts["word/vbaProject.bin"] = b"\xd0\xcf\x11\xe0 macro bytes"
    rel_xml = "".join(f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>' for rid, kind, target in rels) + ext_rels
    parts["word/_rels/document.xml.rels"] = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                                             f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rel_xml}</Relationships>')
    parts["_rels/.rels"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                            f'<Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>')
    parts["[Content_Types].xml"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                                    '<Default Extension="xml" ContentType="application/xml"/>' + "".join(types) + "</Types>")
    parts["word/document.xml"] = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>{body_xml}'
                                  '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr></w:body></w:document>')
    for name, data in (extra or {}).items():
        parts[name] = data
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as z:
        z.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, data in parts.items():
            z.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------------------------------------------------------

PAGE_W, PAGE_H = 612, 792
Line = Tuple[float, float, float, str, str]            # x, y (from the bottom), font size, "R" regular / "B" bold, text


def _pdf_text(s: str) -> str:
    return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def make_pdf(pages: Sequence[Sequence[Line]], *, title: str = "", encrypt: bool = False, image_only: bool = False, extra_page_xobject: bool = False) -> bytes:
    """A PDF whose pages draw the given lines in Helvetica / Helvetica-Bold (WinAnsi, so plain ASCII and Latin-1 text works)."""
    objects: List[bytes] = []                          # objects[i] is object number i + 1

    def add(body: "str | bytes") -> int:
        objects.append(body if isinstance(body, bytes) else body.encode("latin-1"))
        return len(objects)

    catalog = add("")                                  # 1, filled in once the page tree exists
    pages_obj = add("")                                # 2
    font_r = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_b = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    image = add(b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n\x80\nendstream") \
        if image_only or extra_page_xobject else 0
    kids = []
    for lines in pages:
        ops = []
        if image_only:
            ops.append("q 400 0 0 400 100 200 cm /Im1 Do Q")
        for x, y, size, face, text in lines:
            matrix = f"0 1 -1 0 {x} {y}" if face == "RR" else f"1 0 0 1 {x} {y}"             # "RR" is regular text turned 90 degrees
            ops.append(f"BT /F{'2' if face == 'B' else '1'} {size} Tf {matrix} Tm {_pdf_text(text)} Tj ET")
        stream = "\n".join(ops).encode("cp1252", "replace")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        xobject = f" /XObject << /Im1 {image} 0 R >>" if image else ""
        page = add(f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Contents {content} 0 R "
                   f"/Resources << /Font << /F1 {font_r} 0 R /F2 {font_b} 0 R >>{xobject} >> >>")
        kids.append(page)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    info = add(f"<< /Title {_pdf_text(title)} >>") if title else 0
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R" + (f" /Info {info} 0 R" if info else "")
    if encrypt:                                         # a standard-handler dictionary with an owner/user entry nobody can guess: opening needs a password
        trailer += " /Encrypt << /Filter /Standard /V 1 /R 2 /O <" + "ab" * 32 + "> /U <" + "cd" * 32 + "> /P -4 >> /ID [<00112233445566778899aabbccddeeff> <00112233445566778899aabbccddeeff>]"
    out += (trailer + f" >>\nstartxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def column_page(blocks: Sequence[Tuple[str, str, float]], *, left: float = 72, top: float = 720, leading: float = 1.25, gap: float = 10,
                width_chars: int = 85) -> List[Line]:
    """Lay out (kind, text, size) blocks down a page: kind "B" bold or "R" regular. Long text wraps at width_chars; returns the lines."""
    import textwrap
    lines: List[Line] = []
    y = top
    for face, text, size in blocks:
        for piece in textwrap.wrap(text, width=max(10, int(width_chars * 11 / size))) or [""]:
            lines.append((left, round(y, 1), size, face, piece))
            y -= size * leading
        y -= gap
    return lines
