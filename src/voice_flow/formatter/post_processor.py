"""
Deterministic clean-up of the model's output and stitching at the caret. No model is trusted with whitespace.

Runs in well under a millisecond on a dictated utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.voice_flow.formatter.patterns import GREETING_SPLIT_RE, SIGN_OFF_RE
from src.voice_flow.schemas import AppContext, CursorContext

_TAG_ECHO = re.compile(r"</?(?:formatted_text|raw_transcript|app_context|preceding_text|following_text|dictionary|list_items|output)>",
                       re.IGNORECASE)
_FENCE = re.compile(r"\A```[a-zA-Z]*\n(.*?)\n?```\Z", re.DOTALL)
# " ," / " ." / " ?" ... but not ".env", ".gitignore" or "3 .5": the mark must be followed by a space, a closer or the end.
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?])(?=[ \t\n)\]\"']|$)")
_INNER_SPACE_RUN = re.compile(r"(?<=\S)[ \t]{2,}")      # only runs after a character, so indentation survives
_LIST_ITEM = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+\S")
_LIST_PREFIX = re.compile(r"^(\s*(?:\d{1,3}[.)]|[-*•])\s+)(\S)")

_SENTENCE_END = (".", "!", "?")
_OPENERS = "([{`/@#$"
_NO_SPACE_BEFORE = ",.;:!?)]}"


@dataclass(frozen=True)
class Stitched:
    text: str                  # sanitised; no edge spaces added
    leading_space: bool
    trailing_space: bool

    @property
    def insert_text(self) -> str:
        return (" " if self.leading_space else "") + self.text + (" " if self.trailing_space else "")


def _is_item(line: str) -> bool:
    return bool(_LIST_ITEM.match(line))


def _tidy_lists(lines: List[str], capitalize_items: bool) -> List[str]:
    # 1. No blank lines inside a list, nor between a lead-in and its first item.
    out: List[str] = []
    for idx, line in enumerate(lines):
        if line.strip():
            out.append(line)
            continue
        nxt = next((l for l in lines[idx + 1:] if l.strip()), None)
        prev = out[-1] if out and out[-1].strip() else None
        if prev is not None and nxt is not None and _is_item(nxt) and (
                _is_item(prev) or not prev.rstrip().endswith(",")):
            continue
        out.append(line)

    # 2. A line that introduces a list ends with a colon ("We have two tasks." -> "We have two tasks:").
    for i in range(1, len(out)):
        if _is_item(out[i]) and not _is_item(out[i - 1]) and out[i - 1].strip():
            lead = out[i - 1].rstrip()
            if lead.endswith("."):
                out[i - 1] = lead[:-1] + ":"
            elif lead[-1].isalnum():
                out[i - 1] = lead + ":"

    # 3. A list of fragments has no closing period on its last item when the other items have none (a model or Whisper adds it).
    for i in range(len(out)):
        if _is_item(out[i]) and out[i].rstrip().endswith(".") and (i + 1 == len(out) or not _is_item(out[i + 1])):
            start = i
            while start > 0 and _is_item(out[start - 1]):
                start -= 1
            if start < i and not any(l.rstrip().endswith(".") for l in out[start:i]):
                out[i] = out[i].rstrip()[:-1]

    # 4. Every item starts with a capital.
    if capitalize_items:
        for i, line in enumerate(out):
            m = _LIST_PREFIX.match(line)
            if m and m.group(2).islower():
                out[i] = m.group(1) + m.group(2).upper() + line[m.end():]
    return out


def _capital_expected(preceding: str) -> bool:
    tail = preceding.rstrip(" \t")
    return not tail or tail.endswith(_SENTENCE_END) or tail.endswith("\n")


def _opening_quote_at_end(preceding: str) -> bool:
    if not preceding or preceding[-1] not in "\"'":
        return False
    return len(preceding) == 1 or preceding[-2].isspace() or preceding[-2] in "([{"


class PostProcessor:
    @staticmethod
    def sanitize(text: str, app_context: AppContext = AppContext.GENERAL) -> str:
        if not text:
            return ""
        text = _TAG_ECHO.sub("", text).replace("\u200b", "").replace("\r\n", "\n").replace("\r", "\n").strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()

        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _INNER_SPACE_RUN.sub(" ", text)
        if app_context is AppContext.EMAIL:
            text = GREETING_SPLIT_RE.sub(lambda m: m.group(1) + "\n\n" + m.group(2).upper(), text)      # "Hey Sarah, can we..." -> own line
            text = SIGN_OFF_RE.sub(PostProcessor._sign_off_lines, text)
        lines = [line.rstrip() for line in text.split("\n")]
        lines = _tidy_lists(lines, capitalize_items=app_context is not AppContext.CODE_COMMENT)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    @staticmethod
    def stitch(text: str, cursor: Optional[CursorContext] = None,
               app_context: AppContext = AppContext.GENERAL) -> Stitched:
        """Sanitise `text` and decide how it joins the text around the caret."""
        text = PostProcessor.sanitize(text, app_context)
        if not text:
            return Stitched("", False, False)
        preceding = cursor.preceding_text if cursor else ""
        following = cursor.following_text if cursor else ""

        # The first sentence starts with a capital wherever a sentence starts.
        if (app_context is not AppContext.CODE_COMMENT and _capital_expected(preceding)
                and text[0].isalpha() and text[0].islower() and text.split(None, 1)[0].islower()):
            text = text[0].upper() + text[1:]

        text = PostProcessor._close_sentence(text, app_context)

        # A list cannot continue a line: start it on a new line when the caret is mid-line.
        starts_with_item = _is_item(text.split("\n", 1)[0])
        if starts_with_item and preceding and not preceding.endswith("\n"):
            return Stitched(PostProcessor._edge_tidy("\n" + text, following), False,
                            PostProcessor._needs_trailing(text, following))

        text = PostProcessor._edge_tidy(text, following)
        leading = bool(
            preceding
            and not preceding[-1].isspace()
            and preceding[-1] not in _OPENERS
            and not _opening_quote_at_end(preceding)
            and text[0] not in _NO_SPACE_BEFORE
            and text[0] != "\n"
        )
        return Stitched(text, leading, PostProcessor._needs_trailing(text, following))

    @staticmethod
    def _sign_off_lines(m: "re.Match[str]") -> str:
        """"thanks, alex." -> "Thanks,\nAlex". Whisper writes a name it does not know in lower case; a name that already has capitals is kept."""
        word, name = m.group(1), m.group(2)
        return "\n\n" + word[:1].upper() + word[1:].lower() + ",\n" + (name.title() if name.islower() else name)

    @staticmethod
    def _close_sentence(text: str, app_context: AppContext) -> str:
        """A single-line statement ends with a period. Models drop it often, especially after "casual" or caret context.
        Not for code contexts (a commit subject or a command has none), list items, headings, code, or multi-line text (sign-offs)."""
        if app_context is AppContext.CODE_COMMENT or "\n" in text or text.startswith(("#", "`")) or _is_item(text):
            return text
        if len(text.split()) >= 3 and (text[-1].isalnum() or text[-1] in "%)"):
            return text + "."
        return text

    @staticmethod
    def _edge_tidy(text: str, following: str) -> str:
        """Drop a closing period the text after the caret makes wrong ("... the logs." before "and the metrics")."""
        head = following.lstrip()[:1]
        if head and "\n" not in text and text.endswith(".") and (head in ",.;:!?" or head.islower()):
            return text[:-1]
        return text

    @staticmethod
    def _needs_trailing(text: str, following: str) -> bool:
        return bool(following and text and following[0].isalnum() and not text[-1].isspace())
