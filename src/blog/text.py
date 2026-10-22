"""
Text hygiene for strings that code interpolates into Markdown (title, headings, alt text, captions).

The model supplies the words; code writes the `# Title`, `## Heading`, `![alt](...)` and `*caption*` lines around
them, so a stray newline, `#`, `]` or `*` in a model string would break the document, not just look odd.
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]{0,500}>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_LEAD_MARKS_RE = re.compile(r"^(?:[#>\-*•+\s]+|\d{1,2}[.)]\s+)+")
_SECTION_LABEL_RE = re.compile(r"^(?:section|part|chapter)\s+\d+\s*[:.\-–—]\s*", re.I)   # not "Step 1:": that is content
_CLOSING_HASHES_RE = re.compile(r"\s+#+$")                                                    # "## Title ##", never "C#"


def clean_line(text: str) -> str:
    """One line of plain text: tags and control characters removed, whitespace collapsed."""
    text = _TAG_RE.sub("", text or "")
    text = _CONTROL_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def clean_heading(text: str) -> str:
    """A heading: no leading #, bullet or number, no 'Section 2:' label, no emphasis marks."""
    text = clean_line(text)
    text = _LEAD_MARKS_RE.sub("", text)
    text = _SECTION_LABEL_RE.sub("", text)
    return _CLOSING_HASHES_RE.sub("", text.replace("*", "").replace("`", "")).strip()


def clean_alt(text: str) -> str:
    """Image alt text: the existing placeholder swap (resolve_image_placeholders) cannot cope with brackets."""
    return clean_line(text).replace("[", "").replace("]", "")


def clean_caption(text: str) -> str:
    """A caption line is written as *caption*, so it may not contain emphasis marks of its own."""
    return clean_line(text).replace("*", "").strip()
