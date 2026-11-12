"""
How an autocomplete suggestion joins the text the user has typed.

The old prompt asked the model to begin with a leading space after a complete word and to skip it mid-word. Models do not reliably emit a
leading space (an invisible token that chat APIs and models habitually drop): measured 2026-11-12 on 40 labelled contexts, a suggestion after a
complete word started without the space it needed on 57% of calls for gpt-4o-mini and 88% for Claude Haiku 5.5, so "...next step is to" +
"schedule" reached the user as "toschedule". The same contexts after "," or "." (no space typed yet) failed 83-100% of the time.

Now the model states the join in a visible, structured way: its whole answer goes in <w>...</w> when it starts a NEW word (a space belongs before it)
or in <c>...</c> when it continues straight on (finishing the word being typed, attaching punctuation, or the text already ends with a space).
Models judge "is the last word finished?" well; what they got wrong was emitting the whitespace. Text outside the tag is dropped, so an
explanation can never reach the document. A few deterministic guards cover the joins the model may still miss.
"""
from __future__ import annotations

import re

NEW_WORD_TAGS = ("w", "new")
CONTINUE_TAGS = ("c", "cont")
_NAMES = "|".join(NEW_WORD_TAGS + CONTINUE_TAGS)
_TAGGED = re.compile(rf"<(?P<tag>{_NAMES})>(?P<body>.*?)(?:</(?:{_NAMES})>|\Z)", re.IGNORECASE | re.DOTALL)
_DANGLING_CLOSE = re.compile(r"</[a-z]{0,4}>?$", re.IGNORECASE)                # a reply cut off inside its own closing tag
_COMMENTARY = re.compile(
    r"^\s*(?:the\s+)?(?:user'?s?\s+)?(?:text|line|sentence)\s+(?:ends|already\s+ends|is)\b|\bmid-?word\b|\bwhole\s+word\b"
    r"|\bunfinished\s+(?:word|fragment)\b|\bthe\s+missing\s+letters\b", re.IGNORECASE)


def is_commentary(suggestion: str) -> bool:
    """
    The model explaining itself instead of continuing the text ("The text ends mid-word ("dep"), so I finish the word ..."). Seen live on
    Claude Haiku 5.5 (3-7% of calls) with a long untagged join description in the prompt; such text must never reach the user's document.
    """
    return bool(_COMMENTARY.search((suggestion or "").lstrip()))


def apply_join_tag(text_before: str, reply: str) -> tuple[str, bool]:
    """
    (suggestion, tagged). A tagged reply gives the text inside the tag, with a space in front when the tag is <w> and the text before the cursor
    does not already end in whitespace; anything outside the tag is discarded. An untagged reply is returned as it is (the old behaviour).
    """
    match = _TAGGED.search(reply or "")
    if not match:
        return reply, False
    body = _DANGLING_CLOSE.sub("", match.group("body"))
    if match.group("tag").lower() in NEW_WORD_TAGS and body.strip():
        body = body.lstrip()
        if text_before and not text_before[-1].isspace() and (text_before[-1].isalnum() or text_before[-1] in ".,;:!?)]\"'"):
            body = " " + body
    return body, True


def fix_join(text_before: str, suggestion: str) -> str:
    """Deterministic repairs of the join, whether or not the model tagged it."""
    if not text_before or not suggestion:
        return suggestion
    last, first = text_before[-1], suggestion[0]
    if last.isspace():
        return suggestion.lstrip(" \t")                          # never a double space; a newline in the suggestion is kept
    if not first.isalnum():
        return suggestion                                        # leading space, punctuation that attaches, a quote ...
    if last in ",;:!?)]":
        if last in ",:" and text_before[-2:-1].isdigit() and first.isdigit():
            return suggestion                                    # 1,000 and 10:30
        return " " + suggestion
    if last == "." and first.isupper() and text_before[-2:-1].isalpha():
        return " " + suggestion                                  # a new sentence; not a decimal or a domain name
    return suggestion
