"""The exact text a Draft task supplies ("add Acme Inc. below the Contact heading"): pure wording rules, no model, no imports from the document modules."""
import re

# Exact text for Draft: "add Acme Inc. below the Contact heading" supplies the words to insert, not a topic to write about. A draft
# that elaborates ("Acme Inc. is a company that…") is wrong there, so a request that is clearly a literal is answered with that
# literal and no model call. "Clearly" is deliberately narrow: quoted text, or a short item with a digit, "@", brackets, a currency
# symbol, or Title Case words ("Acme Cloud", "Springfield, IL", "Draft v2"), followed by nothing but the place it goes. Anything
# else still goes to the model, whose Draft prompt carries the same rule for the cases wording alone cannot settle
# ("put Hiring now above Careers") and the opposite one (a note ABOUT Acme is written, not replaced by "Acme").
MAX_LITERAL_REQUEST_CHARS = 400      # a literal request is one short sentence; longer text is never matched (bounds the regex work too)
_LITERAL_LEAD = (r"(?:(?:please|kindly|just|simply)\s+|(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
                 r"i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+)*")
_LITERAL_VERB = r"(?:add|insert|put|place|append|prepend|write|type|enter|include)"
_LITERAL_PLACE = (r"(?:below|under|beneath|underneath|after|following|above|before|ahead\s+of|preceding|next\s+to|beside|alongside|"
                  r"(?:at|to)\s+the\s+(?:very\s+)?(?:end|start|beginning|top|bottom))")
# Possessive separators and an object that starts and ends on a non-space keep the match linear in the length of the request.
_LITERAL_REQUEST = re.compile(r"^\s*(?i:" + _LITERAL_LEAD + r")(?P<verb>(?i:" + _LITERAL_VERB + r"))\s++(?P<obj>\S(?:.{0,200}?\S)??)\s++(?P<place>" +
                              _LITERAL_PLACE + r")\b(?P<rest>.*)$", re.S)
_LITERAL_QUOTED = re.compile(r"^(?i:(?:the\s+(?:text|words?|phrase|line|label|following)\s*:?\s*)?)(?P<open>[\"“‘'])(?P<text>.+)(?P<close>[\"”’'])$", re.S)
_QUOTE_PAIRS = {'"': '"“”', "“": "”\"", "‘": "’'", "'": "'’"}
# A task that goes on after the place ("…and make it bold", "…with a link") asks for more than the literal.
_MORE_TO_DO = re.compile(r"\b(?:and|then|with|but|also|using|make|as|so|while|that|which|because|where|to)\b", re.I)
# Words that ask for something to be done to the text (formatting, repetition): never part of a bare literal or of its place.
_DO_WORDS = (r"bold|italics?|underlined?|capitals?|uppercase|lowercase|twice|thrice|again|times|repeated|repeat|centred|centered|red|blue|green|"
             r"larger|bigger|smaller|highlighted|bulleted|numbered|linked|strikethrough")
_ANCHOR_NOUNS = r"heading|title|section|chapter|paragraph|table|list|image|figure|quote|document|doc|page|cursor|selection"
_OBJECT_INSTRUCTION = re.compile(r"\b(?:and|then|with|but|also|using|make|as|so|while|that|which|because|where|to|" + _DO_WORDS + r")\b", re.I)
# What may follow the place word: only the place itself, one anchor noun at the end of one line ("the Contact heading", "the second
# paragraph", "of the document"). Anything else ("… in bold", "… twice", "… centred under the title page") is more to do, which a
# literal answer would silently drop; the free words before the noun may not be anchors or formatting words themselves.
_ANCHOR_TAIL = re.compile(r"^[ \t]*(?:(?:the|this|that|my|our)[ \t]+)?(?:(?!(?:" + _ANCHOR_NOUNS + "|" + _DO_WORDS + r")\b)[\w'’\-]+[ \t]+){0,5}?"
                          r"(?:" + _ANCHOR_NOUNS + r")[ \t]*[.!?]?[ \t]*$", re.I)
# "3 hashtags", "5+ ideas", "up to 5 slogans", "top 10 tips", "3-5 options", "~5 keywords", "3x variants", "3 Taglines": a request for N things.
_QUANTITY = re.compile(r"^(?:(?:up\s+to|at\s+least|at\s+most|about|around|approximately|roughly|over|under|top|best|first|last)\s+)?"
                       r"[~≈]?\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+)?\s*(?:\+|x|×)?\s*\s[A-Za-z]|^[~≈]?\d+\s*(?:\+|x|×)(?=\s|$)", re.I)
_NOT_LITERAL_LEAD = {"a", "an", "the", "some", "more", "another", "any", "my", "our", "your", "their", "this", "that", "these",
                     "those", "new", "short", "brief", "quick", "little", "few", "one", "two", "three", "several", "each", "every",
                     "extra", "additional", "good", "nice", "full", "whole", "entire", "final", "last", "first", "next", "four", "five", "six",
                     "seven", "eight", "nine", "ten", "dozen", "couple", "many", "various", "multiple", "top", "best"}
# Words that name something to WRITE (a part of a document, a kind of text), so the task is not a literal.
_NOT_LITERAL_WORDS = {
    "about", "regarding", "describing", "explaining", "summarizing", "summarising", "covering", "mentioning", "on",
    "summary", "summaries", "conclusion", "introduction", "intro", "overview", "abstract", "outline", "disclaimer", "disclaimers",
    "faq", "faqs", "references", "reference", "bibliography", "appendix", "glossary", "contents", "caption", "captions",
    "heading", "headings", "title", "titles", "subtitle", "subheading", "subheadings", "description", "descriptions", "bio", "blurb",
    "text", "content", "copy", "paragraph", "paragraphs", "sentence", "sentences", "section", "sections", "chapter", "chapters", "line",
    "lines", "row", "rows", "column", "columns", "table", "tables", "list", "lists", "image", "images", "picture", "pictures",
    "chart", "charts", "graph", "graphs", "quote", "quotes", "tip", "tips", "word", "words", "idea", "ideas", "reason", "reasons",
    "step", "steps", "fact", "facts", "stat", "stats", "detail", "details", "note", "notes", "comment", "comments", "bullet",
    "bullets", "point", "points", "item", "items", "example", "examples", "thing", "things", "something", "anything", "characters",
    "chars", "pages", "page", "slides", "slide", "link", "links", "button", "statement", "message", "pitch",
    "testimonial", "testimonials", "story", "stories", "plan", "plans", "answer", "answers", "question", "questions",
    # kinds of text and lists of things a user asks to have written
    "takeaways", "takeaway", "highlights", "highlight", "recap", "agenda", "schedule", "timeline", "roadmap", "hashtags", "hashtag",
    "keywords", "keyword", "emojis", "emoji", "options", "option", "headlines", "headline", "citations", "citation", "articles",
    "article", "sources", "source", "reviews", "review", "benefits", "benefit", "features", "feature", "pros", "cons", "tldr",
    "posts", "post", "taglines", "tagline", "slogans", "slogan", "mottos", "motto", "haiku", "poem", "poems", "limerick", "joke", "jokes",
    "riddle", "toast", "speech", "email", "emails", "letter", "tweet", "tweets", "essay", "paragraphs", "stars", "star", "tags", "tag",
}


def _literal_from_object(obj: str, verb: str = "add"):
    obj = obj.strip().strip(",:;")
    quoted = _LITERAL_QUOTED.match(obj)
    if quoted and quoted.group("close") in _QUOTE_PAIRS[quoted.group("open")]:
        text = quoted.group("text").strip()
        # Two quoted items ('add "A" and "B" below …') are not one literal.
        if not text or any(mark in text for mark in _QUOTE_PAIRS[quoted.group("open")] + quoted.group("open")):
            return None
        return text
    words = obj.split()
    if not words or len(words) > 6 or len(obj) > 60 or "\n" in obj:
        return None
    if words[0].strip(",.").lower() in _NOT_LITERAL_LEAD:
        return None
    if _QUANTITY.match(obj) or _OBJECT_INSTRUCTION.search(obj):
        # A count of things ("3 hashtags", "up to 5 slogans", "top 10 tips") or an instruction in front of the place ("5 stars in bold"):
        # a request to write or format, not a literal.
        return None
    if any(w.strip(".,;:!?()[]\"“”'’").lower() in _NOT_LITERAL_WORDS for w in words):
        return None
    evidence = re.search(r"\d|@|://|www\.|[()\[\]]|[$€£¥%#]", obj)
    capitalised = all(w[0].isupper() for w in words if w[0].isalpha()) and any(w[0].isalpha() for w in words)
    if verb.lower() == "write":
        # "write Haiku below …" names a kind of text to write; only a figure, an address, a bracket or a currency sign says "these words".
        return obj if evidence else None
    return obj if evidence or capitalised else None


def literal_draft_text(query: str, heading_names=()):
    """
    The exact text the task asks to insert ("add Acme Inc. below the Contact heading" -> "Acme Inc."), or None when the task asks
    to write something, names more to do after the place, or is not clearly a literal. `heading_names` are the document's headings
    (any case): a place that is just a heading's name ("below Pricing") is as good as "below the Pricing heading".
    """
    if len(query) > MAX_LITERAL_REQUEST_CHARS:
        return None
    match = _LITERAL_REQUEST.match(query)
    if match is None:
        return None
    rest = match.group("rest")
    if _MORE_TO_DO.search(rest):
        return None
    at_an_edge = re.match(r"(?:at|to)\s+the", match.group("place"), re.I) is not None
    if not (at_an_edge and not rest.strip(" .!?")) and not _ANCHOR_TAIL.match(rest):
        bare = re.sub(r"^(?:the|this|that|my|our)\s+", "", rest.strip(" .!?\"“”'’"), flags=re.I).casefold()
        if not bare or bare not in {re.sub(r"\s+", " ", name).strip().casefold() for name in heading_names}:
            return None
    return _literal_from_object(match.group("obj"), match.group("verb"))
