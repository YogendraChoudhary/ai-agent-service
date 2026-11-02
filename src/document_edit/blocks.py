"""Exact source blocks. Python owns boundaries and preserves every non-target byte."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

from src.longform.markdown_ast import looks_like_html, parse


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


@dataclass(frozen=True)
class DocumentBlock:
    id: str
    kind: str
    section: tuple[str, ...]
    start: int
    end: int
    text: str

    @property
    def editable(self) -> bool:
        return self.kind in ("paragraph", "list") and len(self.text) <= 12_000


def document_blocks(text: str) -> list[DocumentBlock]:
    if not text.strip() or len(text) > 100_000 or looks_like_html(text) or "\x00" in text:
        raise ValueError("Supply a non-empty plain-text or Markdown document of at most 100,000 characters")
    result, headings, start = [], [], 0
    inline_parser = MarkdownIt("commonmark")
    if text.startswith("\ufeff"):
        result.append(DocumentBlock(f"b0_{fingerprint(text[0])[:12]}", "other", (), 0, 1, text[0]))
        start = 1
    for i, block in enumerate(parse(text[start:]), len(result)):
        if block.kind == "heading":
            heading = re.sub(r"^#{1,6}\s*|\s*#+\s*$", "", block.text.splitlines()[0]).strip()
            children = inline_parser.parseInline(heading)[0].children or []
            heading = "".join(t.content for t in children if t.type in ("text", "code_inline", "image"))
            headings = [(level, value) for level, value in headings if level < block.level]
            headings.append((block.level, heading))
        end = start + len(block.text)
        result.append(DocumentBlock(f"b{i}_{fingerprint(block.text)[:12]}", block.kind,
                                    tuple(value for _, value in headings), start, end, block.text))
        start = end
    if len(result) > 400:
        raise ValueError("This document has too many blocks; use a smaller document (at most 400 blocks)")
    assert "".join(b.text for b in result) == text
    return result


def allowed_targets(instruction: str, blocks: list[DocumentBlock]) -> tuple[set[str], str | None]:
    """Enforce named/protected headings independently of the model's proposed targets."""
    headings = {heading for b in blocks for heading in b.section if len(heading) >= 3}

    def mentioned(heading, in_text):
        return re.search(r"(?<!\w)" + re.escape(heading) + r"(?!\w)", in_text, re.I) is not None

    protected_clauses = [m.group(0) for m in re.finditer(
        r"\b(?:leave|leaving|keep|keeping)\b[^.;\n]*?\b(?:unchanged|intact|as is)\b|"
        r"\b(?:do not|don't|never)\s+(?:change|edit|touch|modify)\b[^.;\n]*", instruction, re.I)]
    protected = {h for h in headings if any(mentioned(h, c) for c in protected_clauses)}
    positive = instruction
    for clause in protected_clauses:
        positive = positive.replace(clause, "")
    named = {h for h in headings if mentioned(h, positive)}
    selected_paths = []
    for heading in named:
        paths = [b.section for b in blocks if b.kind == "heading" and b.section[-1:] == (heading,)]
        if len(paths) > 1 and not re.search(r"\b(?:all|both|every)\b", positive, re.I):
            # A parent heading in the same request disambiguates nested sections.
            matching = [p for p in paths if any(mentioned(parent, positive) for parent in p[:-1])]
            if len(matching) != 1:
                return set(), f"There is more than one '{heading}' section. Name its parent section or ask for all occurrences."
            paths = matching
        selected_paths.extend(paths)
    qualifiers = {h for h in named if re.search(r"\b(?:under|within|inside|in)\s+(?:the\s+)?" + re.escape(h) + r"(?!\w)", positive, re.I)}
    selected_paths = [path for path in selected_paths if not (path[-1] in qualifiers and any(
        len(other) > len(path) and other[:len(path)] == path for other in selected_paths))]
    broad = re.search(r"\b(?:throughout|entire|whole|other|all|every|wherever)\b", positive, re.I)
    allowed = {b.id for b in blocks if b.editable and not (set(b.section) & protected)
               and (not named or broad or any(b.section[:len(path)] == path for path in selected_paths))}
    return allowed, None
