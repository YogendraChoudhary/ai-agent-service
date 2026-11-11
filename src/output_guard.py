"""
Keeps assistant talk out of text that goes straight into the user's document.

Prompt rules alone were measured not to hold on the basic tier (live sweep, 2026-11-11): Write mode answered "Add 3 more
items to the list" with "Please provide the list…", ended a product description with "Feel free to adjust…", and Enhance
copied its instruction into the document. This wrapper streams the model's text through unchanged except for:

* Write mode: an answer that is only a request for input becomes a `needs_context` event (no content); a one-line
  preface ("Sure! Here's a checklist:") and a closing note about the draft itself ("Feel free to adjust…") are dropped.
* Enhance: an echo of the instruction (or a "Text:" label) at the very start is dropped.

Only the first paragraph is held until it is complete (and, in Write mode, the last one until the end), so streaming
is otherwise unchanged. Ordinary content such as "Let me know if you have questions." in a drafted email is kept: the
closing check matches notes about the draft or the assistant, not ordinary sign-offs.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, Optional

_FIRST_PARAGRAPH_LIMIT = 400        # characters held while waiting for the first paragraph to end
_REQUEST_ANSWER_LIMIT = 600         # an answer longer than this is content, not a request for input

_MD = re.compile(r"[*_`>#]+")
_REQUEST = re.compile(
    r"\b(?:please\s+(?:provide|share|paste|send|upload|give|include|specify)|"
    r"(?:could|can|would)\s+you\s+(?:please\s+)?(?:provide|share|paste|send|upload|give|specify|clarify)|"
    r"(?:paste|share|provide|send)\s+(?:the|your|me\s+the)\s+\w+|"
    r"i\s+(?:need|would\s+need|'ll\s+need|will\s+need)\s+(?:the|your|more|some)\b|"
    r"(?:if|once|when)\s+you\s+(?:paste|share|provide|send|upload|give)|"
    r"i\s+(?:don't|do\s+not|can't|cannot)\s+(?:see|have\s+access\s+to|access)\b|"
    r"there\s+(?:is|was)\s+no\s+(?:list|text|document|notes?|content|attachment)\b)", re.I)
_PREFACE = re.compile(
    r"^(?:(?:sure|certainly|of\s+course|absolutely|okay|ok|great|got\s+it)[!.,]*\s*)?"
    r"(?:(?:here(?:'s|’s|\s+is|\s+are)|below\s+is|below\s+are|i've\s+(?:written|drafted|created|prepared)|"
    r"i\s+have\s+(?:written|drafted|created|prepared))\b[^\n]{0,160}:)?\s*$", re.I)
_CLOSING = re.compile(
    r"^(?:feel\s+free\s+to\s+(?:adjust|customi[sz]e|modify|tweak|edit|change|add|replace|fill|expand|personali[sz]e)|"
    r"(?:you\s+can|you\s+may)\s+(?:adjust|customi[sz]e|modify|tweak|edit|replace|fill\s+in|personali[sz]e)|"
    r"let\s+me\s+know\s+if\s+you(?:'d|\s+would)?\s+(?:like|want|need)\s+(?:me\s+to|any|more|changes|adjustments|"
    r"a\s+different|further|another|additional)|"
    r"if\s+you(?:'d|\s+would)?\s+(?:like|want)(?:,)?\s+(?:i\s+can|me\s+to)|would\s+you\s+like\s+me\s+to|"
    r"i\s+can\s+(?:also\s+|easily\s+)?(?:adjust|expand|shorten|add|tailor|help|create|make|revise|rewrite)|"
    r"i\s+hope\s+this\s+helps|hope\s+this\s+helps|"
    r"(?:note|tip)\s*:\s*(?:replace|fill|customi[sz]e|adjust|update|swap)\b)", re.I)
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
# The request is the assistant's own ("…you'd like me to add…", "…so I can…", "provide the list/notes…"), not a line
# of drafted content such as "Could you please share your availability?" in an email the user asked for.
_ASSISTANT_ASK = re.compile(
    r"\b(?:you(?:'d|\s+would)\s+like\s+(?:me\s+to|summari[sz]ed|added|rewritten|edited|expanded|improved)|"
    r"you\s+want\s+me\s+to|for\s+me\s+to|i(?:'ll|\s+will|\s+can|\s+could)\s+(?:then\s+)?(?:add|summari[sz]e|rewrite|"
    r"edit|help|create|write|include|insert|update|assist|expand|continue|proceed|work)|so\s+(?:that\s+)?i\s+can|"
    r"i(?:'d|\s+would)\s+be\s+happy\s+to|i\s+(?:don't|do\s+not|can't|cannot)\s+(?:see|have\s+access|access)|"
    r"(?:provide|share|paste|send|upload)\s+(?:me\s+)?(?:with\s+)?(?:the|your)\s+(?:existing\s+|current\s+|original\s+|"
    r"full\s+|actual\s+)?(?:list|text|document|doc|notes|meeting\s+notes|content|draft|paragraph|section|table|file|"
    r"transcript|article|essay|outline|data|items))\b", re.I)
_GREETING = re.compile(r"^(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening)|greetings|subject\s*:|to\s*:)\b", re.I)
# Prompts whose answer is itself a message to someone: a request inside it is content.
_MESSAGE_TASK = re.compile(r"\b(?:e-?mail|message|reminder|letter|memo|announcement|invit(?:e|ation)|reply|request|"
                           r"note|tweet|post|dm|slack|text\s+message|survey|questionnaire|form)\b", re.I)


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", _MD.sub("", text)).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def is_request_for_input(answer: str, query: str = "") -> bool:
    """A short answer in which the assistant asks the user for material instead of writing content."""
    plain = _plain(answer)
    return (bool(plain) and len(plain) <= _REQUEST_ANSWER_LIMIT and not _GREETING.match(plain)
            and not _MESSAGE_TASK.search(query) and bool(_REQUEST.search(plain)) and bool(_ASSISTANT_ASK.search(plain)))


def _strip_echo(paragraph: str, instruction: str) -> str:
    """The first paragraph without an echoed instruction or "Text:" label (empty when it was only that)."""
    text = paragraph.lstrip()
    wanted = _norm(instruction)
    if wanted:
        words = text.split()
        # Find the shortest prefix of the paragraph whose normalised form equals the instruction.
        for i in range(1, min(len(words), 80) + 1):
            prefix = " ".join(words[:i])
            if _norm(prefix) == wanted:
                text = text[text.index(words[i - 1]) + len(words[i - 1]):]
                text = re.sub(r"^[\s.,:;!-]*", "", text)
                break
            if len(_norm(prefix)) > len(wanted):
                break
    text = re.sub(r"^\s*(?:<text>|text\s*:)\s*", "", text, flags=re.I)
    text = re.sub(r"\s*</text>\s*$", "", text, flags=re.I)
    return text


class _Held:
    """Streams text while holding the first paragraph (until it ends) and, optionally, the last paragraph."""

    def __init__(self, hold_last: bool):
        self.buffer = ""
        self.first_done = False
        self.hold_last = hold_last
        self.released = False

    def releasable(self) -> str:
        if not self.hold_last:
            out, self.buffer = self.buffer, ""
            self.released = self.released or bool(out.strip())
            return out
        parts = self.buffer.split("\n\n")
        keep = 1
        if not parts[-1].strip():
            keep = 2                      # the buffer ends at a break: the paragraph before it may be the last one
        # A horizontal rule right before the held paragraph is held too, so "---" before a closing note goes with it.
        while keep < len(parts) and _RULE.match(parts[-keep - 1]):
            keep += 1
        if keep >= len(parts):
            return ""
        out = "\n\n".join(parts[:-keep]) + "\n\n"
        self.buffer = "\n\n".join(parts[-keep:])
        self.released = self.released or bool(out.strip())
        return out


def guard_write(chunks: Iterable, query: str) -> Iterator:
    """Write mode. Yields text chunks, or one `needs_context` event when the whole answer only asks for input."""
    held = _Held(hold_last=True)
    waiting_for_end = False
    for chunk in chunks:
        if not isinstance(chunk, str):
            yield chunk
            continue
        held.buffer += chunk
        if waiting_for_end:
            if len(_plain(held.buffer)) > _REQUEST_ANSWER_LIMIT:
                waiting_for_end, held.first_done = False, True   # long: it is content after all
            else:
                continue
        if not held.first_done:
            stripped = held.buffer.lstrip()
            end = stripped.find("\n\n")
            if end < 0 and len(stripped) < _FIRST_PARAGRAPH_LIMIT:
                continue
            first = stripped if end < 0 else stripped[:end]
            if _REQUEST.search(_plain(first)) and not _GREETING.match(_plain(first)) and not _MESSAGE_TASK.search(query):
                waiting_for_end = True
                continue
            held.first_done = True
            if end >= 0 and _PREFACE.match(_plain(first)) and _plain(first):
                held.buffer = stripped[end + 2:].lstrip("\n")
        out = held.releasable()
        if out:
            yield out
    rest = held.buffer
    if not held.first_done:
        stripped = rest.lstrip()
        if is_request_for_input(stripped, query):
            from .document_intent import _MODE_HINT, suggest_document_intent
            suggestion = suggest_document_intent(query)
            action = suggestion.suggested_action
            yield {"__event__": {"event": "needs_context", "data": {
                "status": "needs_context", "suggested_action": action, "ambiguous": suggestion.ambiguous,
                "message": "This needs your document, which wasn't included. " + _MODE_HINT[action],
                "source": "model"}}}
            return
        end = stripped.find("\n\n")
        if end >= 0 and _plain(stripped[:end]) and _PREFACE.match(_plain(stripped[:end])):
            stripped = stripped[end + 2:].lstrip("\n")
        rest = stripped
    paragraphs = rest.split("\n\n")
    trailing = ""
    while paragraphs and not paragraphs[-1].strip():
        paragraphs.pop()
        trailing = "\n"
    # Drop one closing note about the draft itself, but never the whole answer.
    if paragraphs and _CLOSING.match(_plain(paragraphs[-1])) and (held.released or len(paragraphs) > 1):
        paragraphs.pop()
        while paragraphs and _RULE.match(paragraphs[-1]):
            paragraphs.pop()
        trailing = "\n" if paragraphs or held.released else ""
    text = "\n\n".join(paragraphs)
    if text and trailing and not text.endswith("\n"):
        text += trailing
    if text:
        yield text


def guard_enhance(chunks: Iterable, instruction: Optional[str]) -> Iterator:
    """Enhance. Drops an echoed instruction or "Text:" label at the start; everything else streams as it comes."""
    buffer, done = "", False
    for chunk in chunks:
        if not isinstance(chunk, str) or done:
            yield chunk
            continue
        buffer += chunk
        stripped = buffer.lstrip()
        end = stripped.find("\n\n")
        if end < 0 and len(stripped) < _FIRST_PARAGRAPH_LIMIT:
            continue
        done = True
        cleaned = _strip_echo(stripped, instruction or "")
        if cleaned:
            yield cleaned.lstrip("\n") if cleaned != stripped else buffer
    if not done and buffer:
        cleaned = _strip_echo(buffer.lstrip(), instruction or "")
        if cleaned:
            yield cleaned.lstrip("\n")
