"""Markdown escaping shared by the converters: text from a web page, a Word file or a PDF must not turn into structure by accident."""
from __future__ import annotations

import re

_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿­\U000e0000-\U000e007f\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_OR_BIDI = re.compile("[\U000e0000-\U000e007f‪-‮⁦-⁩]")


def strip_invisible(s: str) -> tuple[str, int]:
    """(`s` without zero-width, soft-hyphen, control, text-direction and Unicode tag characters, how many tag/direction ones there were)."""
    return _INVISIBLE.sub("", s), len(_TAG_OR_BIDI.findall(s))


def escape_inline(s: str) -> str:
    """Punctuation that would start emphasis, a link, code, HTML or strikethrough in the middle of a line."""
    s = s.replace("\\", "\\\\")
    s = re.sub(r"([`*\[\]<>])", r"\\\1", s)
    s = re.sub(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])", r"\\_", s)            # snake_case words stay as they are
    return s.replace("~~", "\\~\\~")


def escape_line_starts(text: str) -> str:
    """What would turn a line into a heading, a list item, a rule or a fence when it comes first on its line."""
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^(\s{0,3})(#{1,6})(\s|$)", r"\1\\\2\3", line)
        line = re.sub(r"^(\s{0,3})([-+])(\s|$)", r"\1\\\2\3", line)
        line = re.sub(r"^(\s{0,3})(\d{1,9})([.)])(\s|$)", r"\1\2\\\3\4", line)
        line = re.sub(r"^(\s{0,3})(=+|-{2,}|~{3,})\s*$", r"\1\\\2", line)
        lines.append(line)
    return "\n".join(lines)
