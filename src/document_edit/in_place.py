"""Recognize exact in-place instructions and check generated replacements."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from src.longform.markdown_ast import word_count

_LEAD = r"(?:(?:please|just|(?:can|could|would) you(?: please)?)\s+)*+"
_QUOTED = r'''(?:"[^"\r\n]+"|“[^”\r\n]+”|'(?:[^'\r\n]|(?<=\w)'(?=\w))+'|‘(?:[^’\r\n]|(?<=\w)’(?=\w))+’|`[^`\r\n]+`)'''
_BARE_WORD = r"(?:[^\s\"“”'‘’`]|(?<=\w)['’](?=\w))+"
_BARE = rf"{_BARE_WORD}(?:[ \t]+{_BARE_WORD}){{0,5}}?"
_REPLACE = re.compile(rf"{_LEAD}(?:replace|change|correct)\s+(?P<old>{_QUOTED}|{_BARE})\s+"
                      rf"(?:with|to)\s+(?P<new>{_QUOTED}|{_BARE})(?:[.?])?", re.I)
_WRITE = re.compile(rf"{_LEAD}(?:write|put|use)\s+(?P<new>{_QUOTED}|{_BARE})\s+(?:in[ -]place|in (?:the|this) selection)[.?]?", re.I)
_ADD = re.compile(rf"{_LEAD}(?:add|insert|put)\s+(?P<new>{_QUOTED}|\([^()\r\n]+\)|\[[^\[\]\r\n]+\]|"
                  rf"\d[\w.,-]*|(?:Inc|Ltd|LLC|Corp|Co|Jr|Sr)\.?)\s+"
                  rf"(?P<side>after|before)\s+(?P<old>{_QUOTED}|[^\s\"'`.,!?]+)[.?]?", re.I)
_PHRASE = re.compile(rf"{_LEAD}(?:fix|correct|proofread|edit|rewrite|rephrase|shorten|polish)\s+"
                     rf"(?:(?:the\s+)?(?:word|phrase|text)\s+)?(?P<old>{_QUOTED})[.?]?", re.I)
_EXACT_LEAD = re.compile(rf"^{_LEAD}(?:replace\b|(?:change|correct)\s+.+?\s+(?:to|with)\s+|"
                         rf"(?:fix|correct|proofread|edit|rewrite|rephrase|shorten|polish)\s+(?:{_QUOTED}|"
                         rf"(?:the\s+)?(?:word|phrase|text)\s+{_QUOTED})|(?:write|put|use)\b.*\b(?:in[ -]place|in (?:the|this) selection)\b)", re.I)
_LOCAL_EDIT = re.compile(rf"^{_LEAD}(?:fix|correct|proofread|edit|rewrite|rephrase|polish|improve)\b(?:\s+(?:it|"
                         r"(?:this|that)(?:\s+(?:word|phrase|text|sentence|selection))?|(?:the\s+)?(?:selection|selected text)|"
                         r"(?:the\s+)?(?:typos?|spelling|grammar|punctuation|mistakes?)"
                         r"(?:\s+in\s+(?:this|that)\s+(?:word|phrase|text|sentence|selection))?|"
                         r"the\s+(?:company|brand|product)\s+name))?\s*(?:in[ -]place\s*)?[.!?]?$", re.I)
_LOCAL_REWRITE = re.compile(rf"{_LEAD}make\s+(?:this|it|(?:the\s+)?(?:selection|selected text))\s+\w+(?:\s+\w+)?[.!?]?", re.I)
_BARE_CORRECTION = re.compile(rf"{_LEAD}(?:fix|correct)\s+([\w-]+)[.?]?", re.I)
_SPELLING = re.compile(rf"^{_LEAD}(?:fix|correct)\s+(?:the\s+)?(?:spelling|typos?)\b", re.I)
_CLAUSE = re.compile(r"\b(?:and|or|then|but|also|using|under|within|in|before|after|with|to)\b|[;?!]|\.\s+", re.I)
_SELECTED = re.compile(r"(?:it|this|that|(?:(?:the|this|that)\s+)?(?:selected text|selection))", re.I)
_UNQUOTED_INSTRUCTION = re.compile(r"^(?:a|an|the|be|make|keep)\s+|\b(?:heading|title|section|paragraph|table|quote)\b", re.I)
_REWRITE = re.compile(rf"^{_LEAD}(?:rewrite|rephrase|shorten|polish|simplify|condense|improve|make)\b", re.I)
_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_SENTENCE_BREAK = re.compile(r"[.!?]+(?=\s|$)|[。！？]+")
_GRAMMAR_WORDS = frozenset("a an the is am are was were be been being have has had do does did to of in on for at by".split())


@dataclass(frozen=True)
class InPlace:
    phrase: str | None = None
    replacement: str | None = None
    side: str | None = None
    ambiguous: bool = False


def in_place_request(instruction: str) -> InPlace | None:
    """Empty target means a selection is required. A side means insert beside the matched phrase."""
    instruction = instruction.strip()
    for pattern in (_REPLACE, _WRITE, _ADD, _PHRASE):
        match = pattern.fullmatch(instruction)
        if match is None:
            continue
        fields = match.groupdict()
        old, new = fields.get("old"), fields.get("new")
        if old and not re.fullmatch(_QUOTED, old) and _SELECTED.fullmatch(old):
            old = None
        if any(value and not re.fullmatch(_QUOTED, value) and _CLAUSE.search(value) for value in (old, new)):
            return InPlace(ambiguous=True)
        if pattern in (_REPLACE, _WRITE) and any(value and not re.fullmatch(_QUOTED, value) and
                (_UNQUOTED_INSTRUCTION.search(value) or _SELECTED.fullmatch(value)) for value in (old, new)):
            return InPlace(ambiguous=True)
        old, new = (value[1:-1] if value and re.fullmatch(_QUOTED, value) else value for value in (old, new))
        return InPlace(old, new, fields.get("side", "").lower() or None)
    if _EXACT_LEAD.search(instruction):
        return InPlace(ambiguous=True)
    if _LOCAL_EDIT.fullmatch(instruction) or _LOCAL_REWRITE.fullmatch(instruction):
        return InPlace()
    correction = _BARE_CORRECTION.fullmatch(instruction)
    return InPlace(phrase=correction[1]) if correction else None


def _sentence_count(text: str) -> int:
    return sum(bool(_WORDS.search(sentence)) for sentence in _SENTENCE_BREAK.split(text))


def safe_replacement(before: str, after: str, instruction: str) -> bool:
    if not after.strip() or len(after) > 24_000 or '\x00' in after or _sentence_count(after) > _sentence_count(before):
        return False
    if (re.match(r'\s*', before)[0] != re.match(r'\s*', after)[0] or
            re.search(r'\s*\Z', before)[0] != re.search(r'\s*\Z', after)[0]):
        return False
    if _SPELLING.search(instruction) and _WORDS.split(before) != _WORDS.split(after):
        return False
    if _REWRITE.search(instruction):
        return word_count(after) <= word_count(before)

    original = _WORDS.findall(before.casefold().replace('’', "'"))
    revised = _WORDS.findall(after.casefold().replace('’', "'"))
    if len(revised) > len(original) + 3:
        return False
    # shortcut: conservative proofreading permits nearby spelling corrections and simple grammar, with exact wording for larger changes.
    for kind, i, j, k, end in SequenceMatcher(None, original, revised).get_opcodes():
        if kind == 'equal':
            continue
        old, new = original[i:j], revised[k:end]
        if all(word in _GRAMMAR_WORDS for word in old + new):
            continue
        if kind == 'delete' and all(word in original[max(0, i - 1):i] + original[j:j + 1] for word in old):
            continue
        if len(old) != len(new) or any(SequenceMatcher(None, a, b).ratio() < 0.6 for a, b in zip(old, new)):
            return False
    return True
