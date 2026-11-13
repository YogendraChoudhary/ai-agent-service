"""
Does a question about a document need the WHOLE document, or only the part it is about?

A question about a part ("What error rate did the ensemble reach?", "How do I activate a venv in PowerShell?") is answered from the few passages that
match it. A question about the whole ("Summarize this paper", "What are the main arguments?", "List every deadline", "Is the document consistent?")
cannot be: passages picked by matching words are a sample, and an answer built from a sample looks like a review of the whole. So when a document is too
large to send whole, a question of the first kind is answered from retrieved passages, labelled partial, and a question of the second kind is refused
(without a model call) with a pointer to the document analysis, which reads every chunk.

Rules only: no model, no network, microseconds. English wording (other languages are treated as questions about a part, so they get a partial answer
with the partial-coverage warning rather than a refusal). The cost of a wrong call is small either way: a part-question taken for a whole one is
refused with advice that says how to rephrase, a whole-question taken for a part one is answered from passages and labelled partial.
"""
from __future__ import annotations

import re

_DOC = r"(?:document|doc|paper|article|text|essay|report|page|file|manual|chapter|book|draft|post|piece|thesis|study|source|material|contract|policy|spec|specification)"
_DOC_REFERENCE = rf"(?:the|this|my|our)\s+(?:(?:entire|whole|full|complete)\s+)?{_DOC}\b"
_THING = r"(?:points?|ideas?|themes?|arguments?|findings?|takeaways?|take-aways?|contributions?|conclusions?|topics?|claims?|messages?|lessons?|results?|highlights?|sections?|steps?|risks?|recommendations?)"
_SECTION = r"(?:sections?|subsections?|chapters?|paragraphs?|parts?|appendix|appendices)"
_HEADING_LABELS = {"section", "subsection", "chapter", "part", "appendix"}
_SECTION_NUMBER = r"\d+(?:\.\d+)*[a-z]?|[a-z]|[ivxlcdm]{2,}"
_QUOTED_TITLE = r'''(?:"[^"\n]{1,100}"|'[^'\n]{1,100}'|“[^”\n]{1,100}”|‘[^’\n]{1,100}’)'''
_TITLE_WORD = rf"(?!(?:the|this|my|our|of|in|within|from|on|about|by|and|or|including|especially|all|every|each|{_DOC})\b)[\w-]+"
_SECTION_TITLE = rf"(?:{_QUOTED_TITLE}|{_TITLE_WORD}(?:\s+(?:(?:and|or|of)\s+)?{_TITLE_WORD}){{0,4}})"
_SECTION_TARGET = re.compile(
    rf"\b(?:(?P<label>{_SECTION})\s+(?P<number>{_SECTION_NUMBER})(?!\w|[.\-]\d)|(?P<quoted_label>{_SECTION})\s+(?P<quoted>{_QUOTED_TITLE})"
    rf"|(?:first|second|third|last|final|\d+(?:st|nd|rd|th))\s+{_SECTION}\b"
    rf"|the\s+(?:{_SECTION_TITLE}\s+{_SECTION}\b|{_SECTION}\s+(?:on|about|titled|called)\s+{_SECTION_TITLE}"
    r"|(?:introduction|preface|foreword|background|methods|results|discussion|conclusion|appendix)\b))"
    rf"(?:\s+(?:of|in|from)\s+{_DOC_REFERENCE})?", re.I,
)
_NUMBERED_LIST_TAIL = re.compile(r"\s*(?:,\s*|(?:and|or|to|through)\s+|[-–—]\s*)\d", re.I)
_ENTIRE_DOC = rf"\b(?:entire|whole|full|complete)\s+{_DOC}\b"
_ACROSS_DOC = rf"\b(?:throughout|across|over)\s+(?:the\s+|this\s+|my\s+|our\s+)?(?:entire\s+|whole\s+)?{_DOC}\b"
_EXPLICIT_WHOLE = re.compile(
    rf"{_ENTIRE_DOC}|{_ACROSS_DOC}"
    rf"|\b(?:summari[sz]e|recap|review|proofread|analy[sz]e|in|of|from)\s+{_DOC_REFERENCE}"
    r"(?!['’]s\b|\s+(?:in|within|under|from)\s+\[section\])"
    rf"|\b{_DOC_REFERENCE}\s+(?:as a whole|overall)\b", re.I,
)

_WHOLE = [re.compile(p, re.I) for p in (
    r"\bsummar(?:y|ies|i[sz]e[sd]?|i[sz]ing)\b", r"\btl\s*;?\s*dr\b", r"\brecap\b", r"\bin a nutshell\b", r"\bthe gist\b", r"\bbrief(?:ly)? (?:me|us) on\b",
    rf"\b(?:what|which)\s+(?:is|are|does|do|was|were)\s+(?:this|the|my|our)\s+{_DOC}s?\s+(?:about|cover|covers|say|says|argue|argues|discuss|discusses|mention|mentions|contain|contains)\b",
    rf"\b(?:what|which)\s+(?:topics?|subjects?|themes?)\s+(?:does|do|is|are)\s+(?:this|the|my|our)\s+{_DOC}\b",
    rf"\b(?:overview|outline|synopsis|abstract|table of contents|structure|gist)\s+of\s+(?:this|the|my|our)\s+{_DOC}\b",
    rf"\b(?:give|show|provide|write|make|create|produce|draft)\s+(?:me\s+)?(?:a|an|the)\s+(?:quick\s+|short\s+|brief\s+|high-level\s+)?(?:overview|outline|synopsis|abstract|digest)\b",
    r"\b(?:list|enumerate|extract|identify|find|name|give me|show me|collect|gather)\s+(?:all|every|each)\b",
    r"\ball\s+(?:the\s+)?(?:\w+\s+){0,2}(?:mentioned|listed|named|referenced|cited)\b",
    r"\bhow many\s+(?:\w+\s+){0,3}(?:are there|in total|overall|in all|in the (?:whole|entire))\b", r"\bhow many times\b", r"\b(?:count|tally)\s+(?:the|all|every|how many)\b",
    _ENTIRE_DOC, _ACROSS_DOC,
    r"\b(?:contradict\w*|inconsisten\w*|conflicting|discrepanc\w*)\b",
    rf"\b(?:proofread|review|critique|evaluate|assess|grade|rate|audit|check|analy[sz]e)\s+(?:the|this|my|our)\s+(?:whole\s+|entire\s+|overall\s+)?{_DOC}\b",
    rf"\b(?:feedback|thoughts|comments|critique|opinion)\s+(?:on|about)\s+(?:the|this|my|our)\s+{_DOC}\b",
    rf"\bis\s+(?:the|this|my|our)\s+{_DOC}\s+(?:too long|too short|well[- ]written|clear|consistent|complete|accurate|persuasive|good|ready|any good)\b",
    rf"\bhow\s+(?:long|many (?:words|pages|sections|chapters|paragraphs))\s+(?:is|are)\s+(?:the|this|my|our)\s+{_DOC}\b",
    r"\bcompare\s+(?:the\s+)?(?:sections?|chapters?|parts?|beginning|start|opening)\b",
)]


# "the main arguments of the add_argument method" asks about a part of an API reference; "the main arguments of the paper" asks about the whole.
_MAIN_THING = re.compile(rf"\b(?:main|key|major|central|principal|core|top|most important|important)\s+{_THING}\b", re.I)
_CODE_TARGET = re.compile(r"\b(?:of|for|to|in)\s+(?:the\s+|a\s+|an\s+)?[\w.()]+\s+(?:method|function|command|class|decorator|parameter|option|call|api|script|module|"
                          r"library|keyword|constructor|operator|flag|macro|endpoint|hook)\b", re.I)


def asks_whole_document(query: str) -> bool:
    """True when the question is about the document as a whole (a summary, an overview, an enumeration of everything, a review), not about a part."""
    text = " ".join((query or "").split())[:2000]
    # Mask section references so an explicit request for the whole document can still take precedence.
    remainder, sections = _SECTION_TARGET.subn("[section]", text)
    if sections:
        return bool(_EXPLICIT_WHOLE.search(remainder))
    if _MAIN_THING.search(text) and not _CODE_TARGET.search(text):
        return True
    return any(pattern.search(text) for pattern in _WHOLE)


def requested_section(query: str) -> tuple[str, str | None] | None:
    """A single quoted or numbered heading, with an optional bare-number form."""
    text = " ".join(query.split())
    matches = list(_SECTION_TARGET.finditer(text))
    if len(matches) != 1 or _EXPLICIT_WHOLE.search(_SECTION_TARGET.sub("[section]", text)):
        return None
    match = matches[0]
    label = (match["label"] or match["quoted_label"] or "").lower()
    if label not in _HEADING_LABELS:
        return None
    if match["number"]:
        if _NUMBERED_LIST_TAIL.match(text, match.end()):
            return None
        return f'{label} {match["number"]}', match["number"]
    return match["quoted"][1:-1].strip(), None
