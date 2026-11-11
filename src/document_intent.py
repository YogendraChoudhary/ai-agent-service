"""Advisory action suggestions from the task alone; suggestions never execute editor changes."""
import re
from typing import Literal

from pydantic import BaseModel

from .document_context import DocumentAction


class IntentSuggestion(BaseModel):
    suggested_action: DocumentAction
    requested_action: DocumentAction | None
    effective_action: DocumentAction
    advisory: Literal[True] = True
    ambiguous: bool
    reason: str


# Asking for new content at a place in the document ("add X below the heading", "insert a table after the intro",
# "put this above the conclusion"). The gap between the verb and the position word may not contain to/into/in, so
# "add more detail to the paragraph below the heading" stays a rewrite of an existing paragraph.
_INSERT_VERB = r"(?:add|insert|put|place|append|prepend|include)"
_POSITION = (r"(?:below|above|after|before|beneath|underneath|under|between|following|"
             r"(?:at|to)\s+the\s+(?:very\s+)?(?:end|start|beginning|top|bottom))")
_PLACED = re.compile(r"\b" + _INSERT_VERB + r"\b(?:(?!\b(?:to|into|in)\b)[^.;!?\n]){0,80}?\b" + _POSITION + r"\b", re.I)
# Asking for a new block by name, with or without a position ("add a new section", "insert a table").
_NEW_BLOCK = re.compile(
    r"\b(?:add|insert|append|prepend)\s+(?:(?:a|an|the|one|two|three|another|new|extra|additional|short|brief|"
    r"separate|small|\d+)\s+){0,3}(?:sections?|paragraphs?|headings?|subheadings?|sub-headings?|tables?|charts?|"
    r"graphs?|images?|pictures?|bullet\s+points?|bullets?|lists?|quotes?|code\s+blocks?|summary|conclusion|"
    r"introduction|intro|faqs?|footnotes?|chapters?)\b|\bnew\s+(?:section|paragraph|heading|chapter)\b", re.I)


# Small in-line changes ("add a comma after 'However'", "insert a hyphen", "add bold to the key phrase") rewrite an existing
# block: they are removed before the insert check, so the rest of the task is still checked.
_INLINE = re.compile(
    r"\b(?:add|insert|put|place|include|append)\s+(?:(?:a|an|the|some|one|two|three|missing|\d+)\s+)*"
    r"(?:commas?|periods?|full\s+stops?|semicolons?|colons?|hyphens?|dash(?:es)?|apostrophes?|quotation\s+marks?|"
    r"quotes?\s+around|parenthes[ie]s|brackets?|spaces?|line\s+breaks?|exclamation\s+marks?|question\s+marks?|"
    r"words?|letters?|articles?|emphasis|bold|italics?|capitals?|capitali[sz]ation|punctuation|accents?)\b", re.I)
# Removing or rearranging whole blocks ("delete the Conclusion section", "move the table below the intro").
_STRUCTURE_NOUN = r"(?:sections?|headings?|paragraphs?|tables?|lists?|chapters?|images?|pictures?|quotes?|code\s+blocks?|blocks?|parts?)"
_RESTRUCTURE = re.compile(
    r"\b(?:delete|remove|drop|cut|get\s+rid\s+of|erase)\s+(?:(?:the|this|that|these|those|a|an|whole|entire|first|second|"
    r"third|last|final|\d+(?:st|nd|rd|th)?)\s+)*(?:[\w'’-]+\s+){0,2}?" + _STRUCTURE_NOUN + r"\b|"
    r"\b(?:move|reorder|rearrange|swap|merge|combine|split|re-?order)\b[^.;!?\n]*\b" + _STRUCTURE_NOUN + r"\b|"
    r"\b(?:put|place|move)\s+(?:the\s+)?[\w\s]{1,40}?\s+(?:first|last|at\s+the\s+(?:top|end|start|beginning|bottom))\b",
    re.I)


def asks_to_insert(text: str) -> bool:
    """True when the task asks for new content to be placed in the document rather than for existing text to change."""
    text = _INLINE.sub(" ", text)
    return bool(_PLACED.search(text) or _NEW_BLOCK.search(text))


def asks_to_restructure(text: str) -> bool:
    """True when the task deletes, moves, merges or splits whole blocks rather than rewriting their text."""
    return bool(_RESTRUCTURE.search(text))


# Adding substance that is not in the text yet ("add more detail to the Benefits paragraph", "include two examples",
# "add statistics"). Edit must not invent facts, so a live run turned such a request into an unrelated rewrite.
_AMOUNT = r"(?:(?:some|more|a\s+few|a\s+bit\s+(?:more\s+)?(?:of\s+)?|extra|additional|further|concrete|specific|real|relevant|"
_AMOUNT += r"supporting|a|an|one|two|three|four|five|\d+)\s+)*"
_NEW_FACTS = re.compile(
    r"\b(?:add|include|insert|put\s+in|give|provide|bring\s+in|work\s+in|back\s+(?:it|this|that)\s+up\s+with)\s+" + _AMOUNT +
    r"(?:details?|examples?|facts?|statistics|stats|data|figures|numbers|evidence|citations?|sources?|references?|"
    r"specifics|context|information|info|case\s+stud(?:y|ies)|anecdotes?|use\s+cases?|quotes?\s+from|research)\b|"
    r"\b(?:elaborate|go\s+into\s+(?:more\s+)?detail)\s+(?:on|about)\b|\bflesh\s+(?:it\s+|this\s+|them\s+)?out\b|"
    r"\bmore\s+detailed\b|\bin\s+(?:more|greater)\s+detail\b", re.I)
# Style that rewrites what is there ("add some warmth", "add more emphasis"): an Edit.
_ADD_STYLE = re.compile(r"\badd\s+" + _AMOUNT + r"(?:clarity|nuance|colou?r|warmth|personality|energy|emphasis|flair|"
                        r"polish|humou?r|punch|urgency|formality|variety|rhythm|depth|substance)\b", re.I)


def asks_for_new_facts(text: str) -> bool:
    """True when the task wants details, examples or facts the text does not contain yet."""
    return bool(_NEW_FACTS.search(text))


def suggest_document_intent(query, requested_action=None):
    text = re.sub(r'^(?:(?:please|can you|could you|would you)\s+)+', '', query.strip(), flags=re.I)
    ask = re.match(r"(?:what|why|how|where|when|should|is|are|suggest|critique|review|summari[sz]e|explain|analy[sz]e|résume|resume|qué|cómo|इसका सारांश)\b", text, re.I)
    draft = (re.search(r'\b(?:continue(?: writing)?|draft|write (?:the |a )?(?:next|new)|add (?:a |the )?new section)\b|continúa|अगला अनुच्छेद', text, re.I)
             or asks_to_insert(text)
             # "add three items to the list", "append a line": new content, once in-line fixes such as "add a comma" are set aside
             or asks_for_new_facts(text)
             or (re.search(r'\b(?:add|append|insert)\b', _INLINE.sub(' ', text), re.I) and not _ADD_STYLE.search(text)))
    edit = _ADD_STYLE.search(text) or _INLINE.search(text) or re.search(r'\b(?:expand|lengthen|rewrite|rephrase|shorten|improve|simplify|proofread|polish|tighten|condense|translate|'
                     r'make (?:this|it|the .+?) (?:shorter|clearer|longer|more \w+|less \w+)|remove repeated|'
                     r'fix|correct|reformat|tidy|clean up|reescribe)\b|दोबारा लिख', text, re.I)
    ambiguous = not ask and bool(draft) == bool(edit)
    suggested = 'ask' if ask or ambiguous else 'draft' if draft else 'edit'
    return IntentSuggestion(suggested_action=suggested, requested_action=requested_action,
        effective_action=requested_action or 'ask', ambiguous=bool(ambiguous),
        reason='question_or_advice' if ask else 'mixed_or_unclear_task' if ambiguous else 'new_content' if draft else 'existing_prose')


# Write mode (type general with no document_context): a prompt that works on a document the request did not include.
_DOC_NOUN = (r"(?:document|doc|draft|article|essay|report|post|page|text|content|writing|paragraphs?|sections?|lists?|"
             r"tables?|email|letter|intro|introduction|conclusion|chapter|notes|story|outline)")
_REFERS_TO_DOCUMENT = re.compile(
    r"\b(?:this|my|our|the\s+(?:current|existing|above|attached|whole|entire|same))\s+" + _DOC_NOUN + r"\b|"
    r"\bthe\s+(?:document|doc)\b|\b(?:text|paragraph|section|list|table|content)\s+above\b|"
    r"\babove\s+(?:text|paragraph|section|list|table|content)\b|\b(?:selected|highlighted)\s+text\b|\bthe\s+selection\b|"
    r"^(?:rewrite|rephrase|shorten|improve|simplify|summari[sz]e|proofread|fix|edit|translate|expand|continue|"
    r"polish|tighten|condense|explain|review|critique|extend|finish)\s+(?:this|it|that|them|these)\b|"
    r"^continue(?:\s+writing)?\s*[.!]?$|\bwrite\s+the\s+(?:next|following)\s+(?:paragraph|section|part|chapter)\b|"
    # A verb that works on existing material plus "the <document part>": "add 3 items to the list", "summarize the meeting
    # notes", "fix the table". Writing something new ("write the introduction for an article") is not matched.
    r"^(?:add|append|insert|summari[sz]e|rewrite|rephrase|shorten|fix|edit|update|translate|expand|continue|proofread|"
    r"improve|polish|condense|reformat|format|sort|reorder|turn|convert|clean\s+up|tidy|finish|complete|extend)\b"
    r"[^.;!?\n]{0,60}?\bthe\s+(?:\w+\s+)?(?:list|notes|meeting\s+notes|minutes|transcript|text|paragraph|section|table|"
    r"draft|document|doc|email|letter|essay|article|story|poem|outline|bullets|bullet\s+points|items|slides?)\b",
    re.I)
_MODE_HINT = {"edit": "Use Edit to change existing text.", "draft": "Use Draft to add new content to it.",
              "ask": "Use Ask to ask about it."}


def write_needs_document(query: str) -> dict | None:
    """
    The needs_context answer for a Write prompt that refers to a document that was not sent ("rewrite this", "continue
    my draft", "summarise the document"), or None to write as usual. A prompt that carries text (e.g. the selection
    appended on a new line) is never stopped. Zero model calls; the mode is the advisory intent suggestion.
    """
    text = re.sub(r'^(?:(?:please|can you|could you|would you|kindly)\s+)+', '', query.strip(), flags=re.I)
    if not text or "\n" in text or len(text) > 300 or not _REFERS_TO_DOCUMENT.search(text):
        return None
    suggestion = suggest_document_intent(text)
    action = suggestion.suggested_action
    return {"status": "needs_context", "suggested_action": action, "ambiguous": suggestion.ambiguous,
            "message": "This refers to your document, which wasn't included. " + _MODE_HINT[action]}
