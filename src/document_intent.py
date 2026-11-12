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
_POSITION = (r"(?:below|above|after|before|beneath|underneath|under|between|following|next\s+to|beside|alongside|adjacent\s+to|"
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


# A small addition INSIDE an existing line: "add (2024) next to the title", "add a phone number after the address", "put Inc. after
# Acme". What is added is a short item (a bracketed or quoted literal, a figure, or a named kind of detail), and it goes beside a word or
# phrase, not into a new block: no "below/under/above" (those start a new line), no section/paragraph/table/list as the place. The
# added text then stays in the same line, which Edit can do without inventing a block; new blocks still go to Draft.
_ITEM_KIND = (r"(?:(?:phone|telephone|mobile|cell|fax|contact|reference|invoice|order|version|issue|serial|zip|postal|post|"
              r"employee|account|ticket|case)\s+(?:number|no\.?|code)|numbers?|e-?mail(?:\s+address(?:es)?)?|addresses|address|dates?|"
              r"years?|times?|prices?|currency(?:\s+(?:symbol|sign))?|percent(?:age)?\s+sign|urls?|web\s*sites?|"
              r"(?:first|last|middle|full|company|brand|product|author)\s+names?|names?|initials|emojis?|trademark(?:\s+symbol)?|"
              r"copyright(?:\s+symbol)?|units?|abbreviations?|acronyms?|suffix|prefix|labels?|symbols?|asterisks?|arrows?|"
              r"check\s*marks?|ticks?|version)")
_ITEM_LITERAL = (r"(?:\([^()\n]{1,40}\)|\[[^\[\]\n]{1,40}\]|[\"“‘'][^\"”’'\n]{1,60}[\"”’']|"
                 r"[^\s()\[\]\"“”\d]*+\d[^\s()\[\]\"“”]*+|(?:Inc|Ltd|LLC|Corp|Co|Jr|Sr|PhD|MD|Esq)\.?)")
_DETERMINER = r"(?:(?:a|an|the|my|our|your|their|some)\s+)?"
_ADD_VERB = r"(?:add|insert|put|place|append|include|attach|tack\s+on)"
_INLINE_PLACE = (r"(?:next\s+to|beside|alongside|adjacent\s+to|right\s+(?:after|before)|just\s+(?:after|before)|immediately\s+(?:after|before)|"
                 r"after|before|following|preceding|(?:at|to)\s+the\s+(?:very\s+)?(?:end|start|beginning)\s+of|to)")
_SMALL_ADDITION = re.compile(
    r"\b" + _ADD_VERB + r"\s+" + _DETERMINER + r"(?:" + _ITEM_KIND + r"|" + _ITEM_LITERAL + r")(?:\s+" + _ITEM_LITERAL + r")?"
    r"\s+(?P<place>" + _INLINE_PLACE + r")\s+(?P<anchor>(?:(?:the|this|that|my|our|each|every)\s+|\w+['’]s\s+)?[^.;!?,\n]{1,60}?)"
    r"(?=\s+(?:and|then|but|also|plus)\b|[.;!?,\n]|$)",
    re.I)
# Places that are blocks, not a phrase: an addition "after the Pricing section" starts a new block.
_BLOCK_ANCHOR = re.compile(r"\b(?:sections?|chapters?|paragraphs?|tables?|lists?|images?|figures?|charts?|graphs?|quotes?|"
                           r"code\s+blocks?|documents?|docs?|pages?|articles?|intro(?:duction)?|conclusion|summary|bullets?)\b", re.I)
_EDGE_BLOCK_ANCHOR = re.compile(r"\b(?:sections?|chapters?|tables?|lists?|images?|figures?|charts?|graphs?|code\s+blocks?|documents?|docs?|"
                                r"pages?|articles?|posts?|files?)\b", re.I)
# "to the end of the paragraph" and "after the heading" put the item inside that block; "to the table" does not.
_INLINE_TO_ANCHOR = re.compile(r"^(?:the\s+|this\s+|that\s+|my\s+|our\s+)?(?:\w+\s+){0,2}?(?:title|headline|heading|subtitle|name|address|"
                               r"date|price|line|sentence|phrase|word|item|label|caption)\b", re.I)


def _small_additions(text: str):
    """The small in-line additions in `text` (see _SMALL_ADDITION), as regex matches."""
    found = []
    for match in _SMALL_ADDITION.finditer(text):
        place, anchor = match.group("place").lower(), match.group("anchor").strip()
        if place == "to":
            ok = _INLINE_TO_ANCHOR.match(anchor) is not None
        elif place.startswith(("at", "to")):
            # "to the end of the paragraph" / "of the title": inside that block; "of the document/section/table" starts something new
            ok = not _EDGE_BLOCK_ANCHOR.search(anchor)
        else:
            # only the thing the item goes next to counts, not what that thing belongs to ("the title of my document")
            ok = not _BLOCK_ANCHOR.search(re.split(r"\s+of\s+", anchor, maxsplit=1)[0])
        if ok:
            found.append(match)
    return found


def asks_for_small_addition(text: str) -> bool:
    """True when the task adds a short item inside an existing line ("add (2024) next to the title"), which Edit can do."""
    return bool(_small_additions(text))


def without_small_additions(text: str) -> str:
    for match in reversed(_small_additions(text)):
        text = text[:match.start()] + " " + text[match.end():]
    return text


def asks_to_insert(text: str) -> bool:
    """True when the task asks for new content to be placed in the document rather than for existing text to change."""
    text = without_small_additions(_INLINE.sub(" ", text))
    return bool(_PLACED.search(text) or _NEW_BLOCK.search(text))


def asks_to_restructure(text: str) -> bool:
    """True when the task deletes, moves, merges or splits whole blocks rather than rewriting their text."""
    return bool(_RESTRUCTURE.search(text))


# Adding substance that is not in the text yet ("add more detail to the Benefits paragraph", "include two examples",
# "add statistics"). Edit must not invent facts, so a live run turned such a request into an unrelated rewrite.
_AMOUNT = r"(?:(?:some|more|a\s+few|a\s+bit\s+(?:more\s+)?(?:of\s+)?|extra|additional|further|concrete|specific|real|relevant|"
_AMOUNT += r"supporting|a|an|one|two|three|four|five|\d+)\s+)*"
_FACT_NOUNS = (r"(?:details?|examples?|facts?|statistics|stats|data|figures|numbers|evidence|citations?|sources?|"
               r"references?(?!\s+(?:number|no\b|code|id)\b)|specifics|context|information|info|case\s+stud(?:y|ies)|anecdotes?|"
               r"use\s+cases?|quotes?\s+from|research)")
_NEW_FACTS = re.compile(
    r"\b(?:add|include|insert|put\s+in|give|provide|bring\s+in|work\s+in|back\s+(?:it|this|that)\s+up\s+with)\s+" + _AMOUNT +
    _FACT_NOUNS + r"\b|"
    r"\b(?:elaborate|go\s+into\s+(?:more\s+)?detail)\s+(?:on|about)\b|\bflesh\s+(?:it\s+|this\s+|them\s+)?out\b|"
    r"\bmore\s+detailed\b|\bin\s+(?:more|greater)\s+detail\b", re.I)
# "improve it by adding more detail": the -ing forms ask for the same. Unlike "add more detail", they are also how a constraint is
# phrased ("shorten it without adding details", "avoid giving more detail"), so a negation earlier in the clause cancels them.
# ("including" is left out: "proofread everything, including references" is a scope, not a request for references.)
_NEW_FACTS_GERUND = re.compile(
    r"\b(?:adding|inserting|putting\s+in|giving|providing|bringing\s+in|working\s+in|backing\s+(?:it|this|that)\s+up\s+with)\s+" +
    _AMOUNT + _FACT_NOUNS + r"\b", re.I)
_NEGATION = re.compile(r"\b(?:without|avoid(?:ing)?|refrain(?:ing)?|forgo(?:ing)?|hold(?:ing)?\s+off|do(?:es)?\s*n['’]?t|do\s+not|never|no|not|stop|skip(?:ping)?|rather\s+than|"
                       r"instead\s+of|neither|nor|exclud(?:e|ing)|leave\s+out|leaving\s+out|omit(?:ting)?|cannot|can['’]?t|won['’]?t|"
                       r"shouldn['’]?t|mustn['’]?t)\b", re.I)
# Style that rewrites what is there ("add some warmth", "add more emphasis"): an Edit.
_ADD_STYLE = re.compile(r"\badd\s+" + _AMOUNT + r"(?:clarity|nuance|colou?r|warmth|personality|energy|emphasis|flair|"
                        r"polish|humou?r|punch|urgency|formality|variety|rhythm|depth|substance)\b", re.I)


def asks_for_new_facts(text: str) -> bool:
    """True when the task wants details, examples or facts the text does not contain yet."""
    if _NEW_FACTS.search(text):
        return True
    return any(not _NEGATION.search(re.split(r"[.;:!?\n]", text[:match.start()])[-1]) for match in _NEW_FACTS_GERUND.finditer(text))


def _lead_stripped(query: str) -> str:
    return re.sub(r'^(?:(?:please|can you|could you|would you)\s+)+', '', query.strip(), flags=re.I)


# Senses of "add" that are not new content: arithmetic ("add up the prices", "add the two prices together") and a link put on a word
# ("insert a link to our website on the word Contact", a change to existing text).
_ARITHMETIC = re.compile(r"\badd(?:ing)?\s+up\b|\badd\b[^.;!?\n]{0,40}\btogether\b|\badd\s+(?:\d[\d.,]*|two|three|four|five)\s+(?:and|plus|to)\s+\d", re.I)
_LINK_ON_WORD = re.compile(r"\b(?:add|insert|put|attach|apply)\s+(?:an?\s+|the\s+)?(?:hyper)?links?\b[^.;!?\n]{0,60}?\b(?:on|to|onto|over)\s+"
                           r"(?:the\s+)?(?:word|words|phrase|text|name)\b", re.I)
# "my draft", "this draft" name the document; they are not a request to draft.
_DRAFT_NOUN = re.compile(r"\b(?:my|our|this|your|current|latest|existing|old|first|attached)\s+drafts?\b", re.I)
_EDIT_VERBS = re.compile(r'\b(?:expand|lengthen|rewrite|rephrase|shorten|improve|simplify|proofread|polish|tighten|condense|translate|'
                         r'make (?:this|it|the .+?) (?:shorter|clearer|longer|more \w+|less \w+)|remove repeated|'
                         r'fix|correct|reformat|tidy|clean up|rename|re-?title|reescribe)\b|दोबारा लिख', re.I)
# Where an instruction may begin: the start, after a sentence or clause break, after "and/then/please…", after "can you…".
_CLAUSE_LEAD = re.compile(r"(?:[.;!?:,]|\b(?:and|then|but|please|just|also|now|kindly)|\b(?:can|could|would|will)\s+you(?:\s+please)?|"
                          r"\bi(?:\s+would|['’]d)?\s+(?:want|need|like)\s+you\s+to|\blet['’]s)\s*$", re.I)


def intent_signals(query: str) -> dict:
    """
    What the wording alone says: ask (a question or a request for advice), draft (new content), edit (existing text changes).
    Any of the three can be true at once; the caller decides what to do with a mix.
    """
    text = _lead_stripped(query)
    small = asks_for_small_addition(text)
    link = _LINK_ON_WORD.search(text)
    rest = _ARITHMETIC.sub(" ", _LINK_ON_WORD.sub(" ", without_small_additions(text)))
    ask = re.match(r"(?:what|why|how|where|when|should|is|are|suggest|critique|review|summari[sz]e|explain|analy[sz]e|résume|resume|qué|cómo|इसका सारांश)\b", text, re.I)
    draft = (re.search(r'\b(?:continue(?: writing)?|draft|write (?:the |a )?(?:next|new)|add (?:a |the )?new section)\b|continúa|अगला अनुच्छेद',
                       _DRAFT_NOUN.sub(" ", text), re.I)
             or asks_to_insert(text)
             # "add three items to the list", "append a line": new content, once in-line fixes such as "add a comma" are set aside
             or asks_for_new_facts(text)
             or (re.search(r'\b(?:add|append|insert)\b', _INLINE.sub(' ', rest), re.I) and not _ADD_STYLE.search(text)))
    edit = small or link or _ADD_STYLE.search(text) or _INLINE.search(text) or _EDIT_VERBS.search(text)
    return {"ask": bool(ask), "draft": bool(draft), "edit": bool(edit)}


def has_work_beyond_small_addition(text: str) -> bool:
    """True when the task asks for more than its in-line additions ("make the text shorter and add (2024) next to the title")."""
    rest = without_small_additions(text)
    return bool(_EDIT_VERBS.search(rest) or _ADD_STYLE.search(rest) or _INLINE.search(rest))


def imperative_edit(query: str) -> bool:
    """
    True when an edit verb or an in-line addition BEGINS an instruction ("make the intro shorter", "and fix the typo", "can you
    rename the heading"), not when the verb merely occurs inside other words ("a story about a robot who wants to rename itself",
    "Polish bakery traditions", "correct me if I am wrong").
    """
    text = _lead_stripped(query)
    candidates = [*_small_additions(text), *_ADD_STYLE.finditer(text), *_INLINE.finditer(text), *_EDIT_VERBS.finditer(text),
                  *_LINK_ON_WORD.finditer(text)]
    return any(not text[:m.start()].strip() or _CLAUSE_LEAD.search(text[max(0, m.start() - 40):m.start()]) for m in candidates)


def suggest_document_intent(query, requested_action=None):
    signals = intent_signals(query)
    ask, draft, edit = signals["ask"], signals["draft"], signals["edit"]
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
def refers_to_document(query: str) -> bool:
    """True when the wording points at the open document ("this paragraph", "my draft", "the section above")."""
    return bool(_REFERS_TO_DOCUMENT.search(re.sub(r'^(?:(?:please|can you|could you|would you|kindly)\s+)+', '', query.strip(), flags=re.I)))


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
