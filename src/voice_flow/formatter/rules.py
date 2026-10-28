"""
Rule-based formatter: the answer when the model is unavailable, too slow, or rejected by the guards.

It is deliberately conservative. It repairs what Whisper is known to get wrong ("mark it point 1" -> "market.1"), removes
clear fillers and stutters, resolves numeric self-corrections, builds `point N` / bullet lists and fixes casing. Anything
it cannot decide with confidence (a non-numeric "send it to Bob, actually make that Alice") is left in the text rather than
guessed at, so the speaker's words are never silently lost. The model handles those.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from src.voice_flow.formatter.patterns import MAY_AS_MONTH_RE, TERM_CASING, starts_question
from src.voice_flow.schemas import CursorContext

STRICT_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
# Mis-hearings of a list number. "point 2" is heard as "point to"; "point 4" as "point for". They count ONLY after a list
# marker and ONLY as the next number in the sequence, so "point for improvement" and "for" opening a list are never numbers.
HOMOPHONE_NUMBERS = {"to": 2, "too": 2, "for": 4}
_SPELLINGS = {n: {w for w, v in {**STRICT_NUMBERS, **HOMOPHONE_NUMBERS}.items() if v == n} | {str(n)} for n in range(1, 11)}

_POINT_CUE = re.compile(
    r"\b(?:(?:point|step|item|bullet)\s+number|point|number|bullet|step|item)\s*[,.:]?\s*(\d{1,2}|one|two|to|too|three|four|for|five|six|seven|eight|nine|ten)\b[,.:]?",
    re.IGNORECASE)
_MARKET = re.compile(r"\bmarket\s*(?:\.|point\b)\s*(?=\d)", re.IGNORECASE)         # "mark it point 1" heard as "market.1"
_BARE_DOT_NUMBER = re.compile(r"(?<=\s)\.(\d{1,2})\b(?!\.\d)")                        # "... and .2 deploy"
_LEAD_IN_TAIL = re.compile(
    r"[,;]?\s*\b(?:(?:mark|make)\s+(?:it|this|that|them)(?:\s+as)?(?:\s+a)?(?:\s+(?:numbered\s+list|list|points?|bullets?))?"
    r"|numbered\s+list)\s*[,.:]?\s*$", re.IGNORECASE)
_BULLET_CUE = re.compile(
    r"\b(?:make|mark)\s+(?:it|this|that|these)?\s*(?:as\s+)?(?:a\s+)?(?:bullets?|bullet\s+(?:list|points?))\b[,.:]?\s*",
    re.IGNORECASE)
_BULLET_WORD = re.compile(r"\b(?:bullet\s+point|bullet|dash)\b[,.:]?\s+", re.IGNORECASE)
_NEW_PARAGRAPH = re.compile(r"\s*\bnew\s+paragraph\b[,.:]?\s*", re.IGNORECASE)
_ORDINAL = re.compile(r"\b(?:first|second|third|fourth|fifth|firstly|secondly|thirdly|next|finally|lastly)\b[,.:]?\s*",
                      re.IGNORECASE)

# Words that can start a list. Without one of these in the speech, a list in the output was not asked for.
LIST_CUES = re.compile(r"\b(?:point|points|bullet|bullets|dash|number|numbered|step|item|first|firstly|second|secondly|third|thirdly|"
                       r"next|finally|lastly|list|make\s+it|mark\s+it)\b", re.IGNORECASE)
_FILLER = re.compile(r"\b(?:um+|uh+|uhh+|er+m?|erm|hmm+|mm+)\b[,.]?\s*", re.IGNORECASE)
# Only plain stutters: a function word said twice ("the the", "I I"), or a 2-3 word phrase said twice with no comma between
# ("we need to we need to"). Never across a comma ("Yes, yes") and never an intensifier or a content word ("very very", "bye bye").
_SAFE_STUTTER_WORDS = r"(?:the|a|an|i|it|is|to|in|on|of|for|and|but|or|so|that|this|we|they|he|she)"
_STUTTER_WORD = re.compile(rf"\b({_SAFE_STUTTER_WORDS})(?:[ \t]+\1\b)+", re.IGNORECASE)
_STUTTER_PHRASE = re.compile(r"\b(\w+[ \t]+\w+(?:[ \t]+\w+)?)[ \t]+\1\b", re.IGNORECASE)
_CORRECTION = (
    r"(?:no[,.]?\s+wait|wait[,.]?(?:\s+(?:no|actually))?|actually[,.]?\s+make\s+that|make\s+that|scratch\s+that|i\s+mean)")
_NUMERIC_FIX = re.compile(
    r"(?P<n1>\d+(?::\d\d)?)\s*(?P<u1>[A-Za-z%]+)?[\s,.;?!]*\b" + _CORRECTION
    + r"\b[\s,.;]*(?P<n2>\d+(?::\d\d)?)(?P<s2>\s*)(?P<u2>[A-Za-z%]+)?", re.IGNORECASE)
_MERIDIEM = re.compile(r"\b(\d{1,2}(?::\d\d)?)\s*([ap])\.?m\b\.?", re.IGNORECASE)

# Words that can only continue a sentence; if the caret is mid-sentence and the text starts with one, it is lower-cased.
_CONTINUATION_WORDS = {"the", "a", "an", "and", "but", "or", "so", "to", "of", "in", "on", "for", "with", "that", "this",
                       "it", "we", "they", "you", "he", "she", "is", "are", "was", "were", "my", "our", "your", "their",
                       "at", "by", "from", "as", "if", "then", "because", "when", "which", "not", "no"}


def apply_dictionary(text: str, terms: Iterable[str]) -> str:
    """Spell the user's own words the way they wrote them ("kubernetes" -> "Kubernetes"). Whole words only."""
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", lambda _m, t=term: t, text, flags=re.IGNORECASE)
    return text


def _repair_asr(text: str) -> str:
    text = _MARKET.sub("point ", text)
    if re.search(r"\bpoint\s*\d", text, re.IGNORECASE):
        text = _BARE_DOT_NUMBER.sub(r"point \1", text)
    return text


def _clean_speech(text: str) -> str:
    text = _FILLER.sub("", text)
    text = _STUTTER_WORD.sub(r"\1", text)
    text = _STUTTER_PHRASE.sub(r"\1", text)

    def keep_last(m: "re.Match[str]") -> str:
        if m.group("u2"):
            return m.group("n2") + m.group("s2") + m.group("u2")
        return m.group("n2") + (" " + m.group("u1") if m.group("u1") else "")
    return _NUMERIC_FIX.sub(keep_last, text)


MAX_SKIPPED_NUMBER = 2          # a speaker who says "point 1, 2, 3, 5" has skipped one; "point 1 ... point 9" is not a list


def _cue_number(word: str) -> Optional[int]:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return STRICT_NUMBERS.get(word)


def _numbered_cues(text: str) -> List[re.Match]:
    """Cues that count up from 1, in order. A written or spoken number may skip ahead a little (the speaker's own slip:
    1, 2, 3, 5); a mis-hearing ("point to", "point for") counts only as exactly the next number."""
    accepted: List[re.Match] = []
    last = 0
    for m in _POINT_CUE.finditer(text):
        word = m.group(1).lower()
        number = _cue_number(word)
        if number is not None:
            ok = number == 1 if last == 0 else last < number <= last + 1 + MAX_SKIPPED_NUMBER
        else:
            ok = last > 0 and HOMOPHONE_NUMBERS[word] == last + 1
            number = last + 1
        if ok:
            accepted.append(m)
            last = number
    return accepted if len(accepted) >= 2 else []


def _item(text: str) -> str:
    text = re.sub(r"^(?:and|then)\b[,\s]*", "", text.strip(" ,.;:"), flags=re.IGNORECASE)
    text = re.sub(r"[,\s]+(?:and|then)$", "", text, flags=re.IGNORECASE).strip(" ,.;:")
    text = re.sub(r"^(?:is|are|was|would\s+be|will\s+be|be)\b[,\s]*", "", text, flags=re.IGNORECASE)    # "point 2 is X" -> "X"
    return text[:1].upper() + text[1:]


def _lead_in(text: str) -> str:
    text = _LEAD_IN_TAIL.sub("", text).strip(" ,;")
    return re.sub(r"[,\s]+(?:the|a|an|our|my|your|their)$", "", text, flags=re.IGNORECASE)     # "... now the" + cue


def _split_closing(items: List[str]) -> Tuple[List[str], str]:
    """Whisper punctuates, so the last item ends at its first sentence break; what follows ("Thanks, Alex.") is not an item."""
    m = re.search(r"[.!?]\s+(?=\S)", items[-1])
    if not m:
        return items, ""
    return items[:-1] + [items[-1][: m.start()]], items[-1][m.end():].strip()


def _build_list(text: str) -> Optional[Tuple[str, List[str], str, str]]:
    """(lead-in, items, marker style, closing text) when the speech holds a list, else None."""
    cues = _numbered_cues(text)
    if cues:
        items = [_item(text[m.end(): cues[i + 1].start() if i + 1 < len(cues) else len(text)]) for i, m in enumerate(cues)]
        if all(items):
            items, closing = _split_closing(items)
            return _lead_in(text[: cues[0].start()]), items, "number", closing
    marks = list(_BULLET_WORD.finditer(text))
    if len(marks) >= 2:
        items = [_item(text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]) for i, m in enumerate(marks)]
        if all(items):
            items, closing = _split_closing(items)
            return text[: marks[0].start()].strip(" ,;"), items, "bullet", closing
    bullet = _BULLET_CUE.search(text)
    if bullet:
        tail = text[bullet.end():]
        marks = list(_ORDINAL.finditer(tail))
        if len(marks) >= 2:
            items = [_item(tail[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(tail)]) for i, m in enumerate(marks)]
            if all(items):
                items, closing = _split_closing(items)
                return text[: bullet.start()].strip(" ,;"), items, "bullet", closing
    return None


def list_style(raw: str) -> Optional[str]:
    """"number" when the speaker counted ("point 1, point 2"), "bullet" when they said bullets/dash, else None."""
    text = _clean_speech(_repair_asr(" ".join((raw or "").split())))
    cues = _numbered_cues(text)
    if cues:
        return "bullet" if cues[0].group(0).lower().startswith("bullet") else "number"
    built = _build_list(text)
    return built[2] if built else None


def list_lead_in(raw: str) -> Optional[str]:
    """What the speaker said before the first list item (cue words removed), or None when the speech holds no list."""
    built = _build_list(_clean_speech(_repair_asr(" ".join((raw or "").split()))))
    return built[0] if built else None


def expected_list_items(raw: str) -> int:
    """How many list items the speaker counted (0 when the speech holds no list): the number a good answer must contain."""
    built = _build_list(_clean_speech(_repair_asr(" ".join((raw or "").split()))))
    return len(built[1]) if built else 0


def detect_list(raw: str) -> bool:
    """True when the speech counts through items ("point 1 ... point 2 ...", "make it bullets first ... second ...")."""
    return _build_list(_clean_speech(_repair_asr(" ".join((raw or "").split())))) is not None


def _apply_term_casing(text: str) -> str:
    """Acronyms, product names, days, months and places, from the shared registry (patterns.py)."""
    text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*\b",
                  lambda m: TERM_CASING.get(m.group(0), m.group(0)) if m.group(0) == m.group(0).lower() else m.group(0), text)
    return MAY_AS_MONTH_RE.sub(lambda m: re.sub(r"may$", "May", m.group(0), flags=re.IGNORECASE), text)


def _sentence_case(text: str) -> str:
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.strip())
    text = re.sub(r"\bi\b(?!\.)", "I", text)
    text = _MERIDIEM.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}M", text)
    return _apply_term_casing(text)


def _finish(text: str, closing: str = ".", infer_question: bool = True) -> str:
    text = _sentence_case(text)
    if not text or text[-1] in ".!?:":
        return text
    if infer_question and starts_question(text):
        return text + "?"
    return text + closing


def continue_sentence(text: str, cursor: Optional[CursorContext]) -> str:
    """Mid-sentence caret: "that the Redis pool ran out", not "That the Redis pool ran out"."""
    tail = (cursor.preceding_text if cursor else "").rstrip(" \t")
    if tail and not tail.endswith((".", "!", "?", "\n")) and text:
        first = text.split(None, 1)[0]
        if first.lower() in _CONTINUATION_WORDS and first[0].isupper():
            return first[0].lower() + text[1:]
    return text


def rule_format(raw: str, cursor: Optional[CursorContext] = None, dictionary: Iterable[str] = ()) -> str:
    paragraphs = _NEW_PARAGRAPH.split(raw or "")
    if len(paragraphs) > 1:
        done = [rule_format(part, cursor if i == 0 else None, dictionary) for i, part in enumerate(paragraphs)]
        return "\n\n".join(p for p in done if p)
    text = _clean_speech(_repair_asr(" ".join((raw or "").split())))
    if not text:
        return ""
    built = _build_list(text)
    if built:
        lead, items, style, closing = built
        items = [_finish(i, closing="", infer_question=False) for i in items]
        body = "\n".join((f"{n}. {i}" if style == "number" else f"- {i}") for n, i in enumerate(items, 1))
        head = ""
        if lead:
            head = _finish(lead, closing=":")
            head = (head[:-1] + ":" if head.endswith(".") else head) + "\n"
        out = head + body + ("\n\n" + _finish(closing) if closing else "")
    else:
        out = _finish(text)
    return apply_dictionary(continue_sentence(out, cursor), dictionary)
