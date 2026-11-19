# Length adherence: "write X in 100 words" must give about 100 words

Status: **plan only, nothing changed in `src/` yet.** Written 2026-11-19 against commit `a09ee93`.

The replacement code below is not a sketch. The new module (`src/length_spec.py`) and the new blog plan were run in a scratch copy
against this repository's own `markdown_ast` and `config`. Their real outputs are pasted in sections 3 and 5.

---

## 1. The reported problem

In the Live demo editor, Auto mode, the user typed:

> Write an article on Weather Forcast in 100 Words.

The editor received an article with a title, a subtitle, an intro, sections with headings and a picture. The editor's counter showed
**Words: 714** (the screenshot shows 714; it was reported as 754 in the message). The title was *"A 100-word weather forecast that's
useful"*: the length the user asked for became the topic of the post.

## 2. What the code does today, step by step

### 2.1 Auto chooses Write, and Write is sent as a plain `general` request

`api.py:505` (`request_for_action`) turns an Auto "write" decision into a plain prompt with no document:

```python
# api.py:505
    if decision.action == 'write':
        return request.model_copy(update={'document_action': None, 'document_context': None, 'document_scope': None,
                                          'document_analysis': None, 'suggest_document_intent': False})
```

### 2.2 The "general" request is promoted to the blog pipeline

`src/blog/intent.py:20-25` matches "write/draft/create … article/blog/guide … about/on/covering". "Write an article on …" matches:

```python
# src/blog/intent.py:20
_ADJECTIVES = r"(?:(?:short|long|detailed|comprehensive|in-depth|engaging|practical|complete|beginner'?s?)\s+)?"
_FORM = r"(?:blog(?:\s+post)?|article|guide|long[- ]form\s+(?:piece|article|post))"
_BLOG = re.compile(
    POLITE_PREFIX + rf"(?:write|draft|create)\s+(?:me\s+)?(?:an?\s+){_ADJECTIVES}{_FORM}\s+(?:about|on|covering)\b",
    re.I,
)
```

`src/text_processing_agent.py:2601-2613` then switches the request to the staged blog writer. **Nothing here looks at the length the
user asked for:**

```python
# src/text_processing_agent.py:2601
    elif (request_type == 'general' and not image_urls and settings.blog_ai_enabled and not design_mode and not document_mode
          and not (image_intent and image_intent['kind'] in ('generate', 'edit'))
          and (settings.blog_intent_regex_enabled or settings.blog_intent_shadow_mode)):
        # "write a blog post about ..." in free-form chat: image and code intent were checked first, as for code.
        from .blog.intent import detect_blog_intent
        blog_decision = detect_blog_intent(query)
        trace_set(request_id, blog_intent=blog_decision['rule'])
        if blog_decision['kind'] == 'blog':
            if settings.blog_intent_shadow_mode:
                logger.info(f"{log_prefix}[BLOG SHADOW] would route to the blog pipeline (rule: {blog_decision['rule']})")
            else:
                force_agent_type, request_type = 'blog_writer', 'generate_blog'
                task, intent_attrs = task_for_request(request_type, query)
                intent_attrs.update(intent_rule=blog_decision['rule'])
```

The blog pipeline was built and tuned for posts of 800 to 2,200 words (outline → parallel parts → lint).

### 2.3 The blog planner reads "100", then overrides it with fixed minimums

**(a) The number is read, then clamped up to 300.** `src/blog/plan.py:18-22` and `:94-104`:

```python
# src/blog/plan.py:18
LENGTH_WORDS = {"short": 800, "standard": 1450, "long": 2200}
DEFAULT_LENGTH = "standard"
# A brief can name any number ("a 50,000-word post"). Outside these bounds the post is not a blog post and the
# bill is not the user's to run up by accident, so the plan clamps the length and says so in `reasons`.
MIN_WORDS, MAX_WORDS = 300, 4000
```

```python
# src/blog/plan.py:94
    # length: the brief's own number, else the option, else standard
    asked = extract_word_count(request.brief)
    if asked is not None:
        target, reasons["length"] = asked, f"brief asked for {asked} words"
    else:
        bucket = request.length if request.length in LENGTH_WORDS else DEFAULT_LENGTH
        target, reasons["length"] = LENGTH_WORDS[bucket], f"option {bucket}" if request.length in LENGTH_WORDS else "default"
    clamped = max(MIN_WORDS, min(MAX_WORDS, target))      # <-- 100 becomes 300
    if clamped != target:
        reasons["length"] += f", clamped to {clamped}"    # <-- only in logs; the user is never told
    target = clamped
```

**(b) The structure never shrinks below 4 sections, 80-word edges, 1 table, 2 quotes and 1 picture.** `src/blog/plan.py:81-83`
and `:106-128`:

```python
# src/blog/plan.py:81
def section_count(target_words: int) -> int:
    return 4 if target_words <= 1000 else 5 if target_words <= 1600 else 6 if target_words <= 2200 else 7
```

```python
# src/blog/plan.py:106
    sections = section_count(target)
    edge = min(180, max(80, round(target * 0.085 / 10) * 10))   # intro and conclusion: never under 80 words each
    tables = 2 if target >= 2000 else 1                         # always at least one table
    quotes = 3 if target >= 2000 else 2                         # always at least two pull quotes
    ...
    else:
        images, reasons["images"] = 1 + target // 700, "auto"  # always at least one picture
```

**(c) A section can never be planned under 60 words, so the outline rounds the plan back up.** `src/blog/schema.py:27-29` and
`src/blog/outline.py:341-346`:

```python
# src/blog/schema.py:27
SECTIONS = (3, 8)
SECTION_WORDS = (60, 800)
EDGE_WORDS = (40, 300)          # intro and conclusion
```

```python
# src/blog/outline.py:341
    total, want = sum(sec["target_words"] for sec in secs), plan.section_words
    if total and abs(total - want) > 0.15 * want:
        for sec in secs:
            sec["target_words"] = max(s.SECTION_WORDS[0], min(s.SECTION_WORDS[1], round(sec["target_words"] * want / total)))
```

**What the planner actually produced for this prompt** (run on the real `plan_blog`, no model call):

```
brief: "Write an article on Weather Forcast in 100 Words"
target_words=300  sections=4  intro=80  conclusion=80  section_words=140 (35 each -> raised to 60 each = 240)
tables=1  quotes=2  images=1  reasons: "brief asked for 100 words, clamped to 300"
```

So the prose plan is 80 + 240 + 80 = **400 words** before the model writes anything.

### 2.4 Each part may run 35% over, and nothing checks the whole post

`src/blog/validate.py:24` and `:235-239` check each part only against its own (already inflated) target, with ±35%:

```python
# src/blog/validate.py:24
LENGTH_TOLERANCE = 0.35
```

```python
# src/blog/validate.py:235
    if target_words:
        low, high = round(target_words * (1 - LENGTH_TOLERANCE)), round(target_words * (1 + LENGTH_TOLERANCE))
        if not low <= out.words <= high:
            out.errors.append(Issue("length", ...))
```

400 planned words may therefore come back as up to ~540 and still pass. `src/blog/service.py:162-175` never compares the finished
post with the number the user named.

### 2.5 The blog counts prose only; the editor counts everything

`src/longform/markdown_ast.py:31` and `:449-451`:

```python
PROSE_KINDS = ("paragraph", "list", "blockquote")

def prose_words(blocks: Sequence[Block], kinds: Sequence[str] = PROSE_KINDS) -> int:
    """Words a reader reads as text. Tables, code, HTML, headings and image captions are not counted."""
```

The editor's "Words" counter also counts the title, subtitle, five headings, every table cell and the picture caption. That's how
~400–540 words of prose plus all the structure reaches **714** in the editor.

### 2.6 The length phrase is sent to the model as part of the topic

`src/blog/outline.py:64-67` and `src/blog/draft.py:108` pass the brief unchanged, "in 100 Words" included:

```python
# src/blog/outline.py:64
def outline_request(request: BlogRequest, plan: BlogPlan) -> dict:
    return {
        "brief": request.brief,          # "Write an article on Weather Forcast in 100 Words"
```

The outline model reads "100 words" as subject matter → title *"A 100-word weather forecast that's useful"*.

### 2.7 The number parser misses most ways people ask

`src/request_plan.py:144-150`:

```python
_WORDCOUNT = re.compile(r"\b(\d[\d,]{2,5})[- ]?(?:words?|word\s+count)\b", re.I)

def extract_word_count(text: Optional[str]) -> Optional[int]:
    m = _WORDCOUNT.search(text or "")
    return int(m.group(1).replace(",", "")) if m else None
```

`\d[\d,]{2,5}` needs **at least three characters**. Results from the real function:

| Request | Today |
|---|---|
| "Write an article on weather in **50 words**" | `None` → planned at the **1,450** default |
| "Write about cats in **one hundred words**" | `None` → 1,450 |
| "under 150 words", "no more than 80 words" | read as "about", never as a ceiling |
| "in **3 sentences**", "**two paragraphs**", "**5 bullet points**", "a **280-character** tweet" | not recognised |
| "Write a **short** article on weather" | ignored → 1,450 |

### 2.8 Writes that don't go to the blog have no length handling at all

"Write a paragraph about dogs in 100 words" (free-form `smart_global` writer) and Draft at the cursor
(`document_context`, action `draft`) send the user text as-is:

```python
# src/text_processing_agent.py:2969
                stream = _agent_text(agent, prepared_context.prompt if document_mode else query, images, provider, use_tools,
                                     request_id, output_limit=output_limit)
```

No length line goes to the model and nothing is measured afterwards. Draft is also capped at 1,000 output tokens
(`src/document_context.py:27`, `OUTPUT_TOKENS = {"ask": 2_000, "draft": 1_000}`), so "draft 900 words here" gets cut off.
**I haven't measured how far off these paths are; the live test set in section 7 will measure it.**

## 3. Why prompt wording alone can't fix it

The overshoot is decided in code before the model writes a word: the 300 clamp, the 4-section minimum, the 80-word edges, the
60-word section floor, the required table, quotes and picture, and ±35% per part. Even a model that followed every instruction
perfectly would deliver 400+ words of prose plus headings. So code has to own the length, in three places:

1. **Which writer runs.** A short piece goes to the single-call writer, not the 4-section blog pipeline.
2. **How big the plan is.** When the blog does run, its structure scales with the number, and title/heading/table/caption words
   come out of the budget.
3. **Whether the result is accepted.** The output is measured the way the editor counts, and repaired once if it misses.

## 4. New module: `src/length_spec.py` (complete file)

One parser and one counter for every writer, so "in 100 words" means the same thing everywhere.

```python
"""
The length the user asked for, read from their own words, and counts that match what the editor shows.

One parser for every text-writing path (Write, Draft, blog), so "in 100 words" means the same thing everywhere.
Only phrases that NAME a length count: "in 100 words", "a 100-word article", "under 150 words", "word limit of 80",
"3 sentences", "two paragraphs". A number of words that is the subject of the request ("list 10 words that rhyme",
"the 100 words every child should know") is content, never a length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.longform.markdown_ast import parse, word_count

# ---------------------------------------------------------------------------
# Numbers: digits ("100", "1,500", "1.5k") and words ("one hundred", "a hundred and fifty", "fifty")
# ---------------------------------------------------------------------------
_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
_NUMBER_WORD = r"(?:" + "|".join(sorted([*_ONES, *_TENS, "hundred", "thousand"], key=len, reverse=True)) + r")"
_SPELLED = rf"(?:(?:a\s+)?{_NUMBER_WORD}(?:[\s-]+(?:and\s+)?{_NUMBER_WORD})*)"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?k?)"
NUM = rf"(?:{_DIGITS}|{_SPELLED})"

_UNITS = {
    "word": "words", "words": "words",
    "character": "characters", "characters": "characters", "char": "characters", "chars": "characters",
    "sentence": "sentences", "sentences": "sentences",
    "paragraph": "paragraphs", "paragraphs": "paragraphs", "para": "paragraphs", "paras": "paragraphs",
    "line": "lines", "lines": "lines",
    "bullet": "bullets", "bullets": "bullets", "bullet point": "bullets", "bullet points": "bullets",
    "point": "bullets", "points": "bullets",
}
UNIT = r"(?:bullet\s+points?|words?|characters?|chars?|sentences?|paragraphs?|paras?|lines?|bullets?|points?)"

_AT_MOST = r"under|below|within|less\s+than|fewer\s+than|no\s+more\s+than|not\s+more\s+than|at\s+most|max(?:imum)?|up\s+to"
_AT_LEAST = r"at\s+least|minimum(?:\s+of)?|min|over|more\s+than"
_EXACT = r"exactly|precisely"
_ABOUT = r"about|around|approximately|approx\.?|roughly|~|some|just|only"
_PREP = r"in|of|with|using"

# "in 100 words", "under 150 words", "in about 100 words", "between 80 and 120 words", "in 100-150 words"
_PREP_FORM = re.compile(
    rf"\b(?P<q>{_AT_MOST}|{_AT_LEAST}|{_EXACT}|{_ABOUT}|{_PREP})\s+(?:(?P<q2>{_AT_MOST}|{_AT_LEAST}|{_EXACT}|{_ABOUT})\s+)?"
    rf"(?P<n>{NUM})(?:\s*(?:-|–|to)\s*(?P<n2>{NUM}))?\s+(?P<u>{UNIT})\b", re.I)
_BETWEEN = re.compile(rf"\bbetween\s+(?P<n>{NUM})\s+and\s+(?P<n2>{NUM})\s+(?P<u>{UNIT})\b", re.I)
# "a 100-word article", "a 1,500 word post", "a three-paragraph note", "a 280-character tweet"
_ADJ_FORM = re.compile(rf"\b(?P<n>{NUM})[\s-](?P<u>word|character|sentence|paragraph|line|bullet)\b(?!s)", re.I)
# "word limit of 100", "word count: 100", "character limit 280"
_LIMIT_FORM = re.compile(rf"\b(?P<u>word|character|char)\s+(?P<q>limit|count|cap|max(?:imum)?)\s*(?:of|is|:|=)?\s*(?P<n>{NUM})\b", re.I)
# "weather, 100 words.", "100 words max", "100 words or less", "100 words long"
_TRAILING_FORM = re.compile(
    rf"\b(?P<n>{NUM})\s+(?P<u>{UNIT})(?P<tail>\s+(?:long|max(?:imum)?|tops|only|or\s+(?:less|fewer|under|so|more)|"
    rf"limit|total))?\s*(?=[.!?;)\]]|$)", re.I)

# After the phrase: the words are the subject ("10 words that rhyme", "100 words for snow")
_CONTENT_AFTER = re.compile(r"\s*(?:that|which|who|to\b|for\b|from|starting|beginning|ending|related|like|meaning|"
                            r"in\s+(?:english|french|spanish|german|another|other)|you|i\b)", re.I)
# "write two paragraphs about Rome", "give me 5 bullet points": a count of sentences, paragraphs, lines or bullets
_COUNT_FORM = re.compile(
    rf"\b(?P<n>{NUM})\s+(?:short\s+|brief\s+|concise\s+)?(?P<u>bullet\s+points?|sentences?|paragraphs?|paras?|lines?|bullets?)\b",
    re.I)
# Before a counted unit: a determiner means the sentences already exist ("translate these 3 sentences")
_COUNT_CONTENT_BEFORE = re.compile(r"\b(?:these|those|the|my|your|our|following|above|below|first|last|next)\s+$", re.I)
# Before the phrase: a verb whose object is the words themselves ("list 10 words", "give me 5 words")
_CONTENT_BEFORE = re.compile(r"\b(?:list|give(?:\s+me)?|name|suggest|find|brainstorm|translate|spell|define|"
                             r"rhyme|top|best|common|key|new|vocabulary)\s+(?:me\s+)?(?:the\s+)?(?:top\s+)?$", re.I)

# Reading of each qualifier
_KIND = [(re.compile(rf"^(?:{_AT_MOST}|limit|cap|max(?:imum)?|or\s+(?:less|fewer|under)|tops)$", re.I), "at_most"),
         (re.compile(rf"^(?:{_AT_LEAST}|or\s+more)$", re.I), "at_least"),
         (re.compile(rf"^(?:{_EXACT})$", re.I), "exact")]

ABOUT_TOLERANCE = 0.10       # "in 100 words" -> 90-110
ABOUT_MIN_SLACK = 5          # ...but never tighter than +-5 words, so "in 20 words" allows 15-25
EXACT_TOLERANCE = 0.03       # "exactly 100 words" -> 97-103
AT_MOST_FLOOR = 0.6          # "under 100 words" must still be a real answer: at least 60
COUNTED_UNITS = ("sentences", "paragraphs", "lines", "bullets")   # these are counted exactly


def to_int(text: str) -> Optional[int]:
    """'1,500' -> 1500, '1.5k' -> 1500, 'one hundred and fifty' -> 150, 'a hundred' -> 100."""
    t = text.strip().lower().replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?k", t):
        return int(float(t[:-1]) * 1000)
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return int(float(t))
    total = current = 0
    seen = False
    for w in re.split(r"[\s-]+", t):
        if w in ("and", ""):
            continue
        if w == "a":
            current = current or 1
        elif w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w == "thousand":
            total, current = total + (current or 1) * 1000, 0
        else:
            return None
        seen = True
    return total + current if seen else None


@dataclass(frozen=True)
class LengthSpec:
    unit: str              # words | characters | sentences | paragraphs | lines | bullets
    target: int
    low: int
    high: int
    kind: str              # about | exact | at_most | at_least | range
    start: int             # where the phrase is in the request, so it can be taken out of a topic
    end: int

    def contains(self, n: int) -> bool:
        return self.low <= n <= self.high

    def instruction(self) -> str:
        """One line for the model's user message. Code adds it; the system prompt stays byte-identical for caching."""
        if self.unit in COUNTED_UNITS:
            what = f"exactly {self.target} {self.unit}" if self.low == self.high else f"{self.low}-{self.high} {self.unit}"
        elif self.kind == "at_most":
            what = f"at most {self.high} {self.unit} (aim for about {round(self.high * 0.9)})"
        elif self.kind == "at_least":
            what = f"at least {self.low} {self.unit}"
        else:
            what = f"{self.target} {self.unit} (between {self.low} and {self.high})"
        return (f"LENGTH: the whole answer must be {what}, counting every word you output, including any title, "
                f"heading, list item or table cell. Do not add a title unless the task asks for one.")

    def describe(self) -> str:
        return {"at_most": f"at most {self.high}", "at_least": f"at least {self.low}",
                "range": f"{self.low}-{self.high}"}.get(self.kind, str(self.target)) + f" {self.unit}"


def _window(unit: str, target: int, kind: str, n2: Optional[int] = None) -> tuple:
    if n2 is not None:
        lo, hi = sorted((target, n2))
        return lo, hi, round((lo + hi) / 2), "range"
    if unit in COUNTED_UNITS:
        if kind == "at_most":
            return 1, target, target, kind
        if kind == "at_least":
            return target, target * 3, target, kind
        return target, target, target, "exact"
    if kind == "at_most":
        return max(1, round(target * AT_MOST_FLOOR)), target, target, kind
    if kind == "at_least":
        return target, target * 3, target, kind
    slack = EXACT_TOLERANCE if kind == "exact" else ABOUT_TOLERANCE
    d = max(round(target * slack), 1 if kind == "exact" else ABOUT_MIN_SLACK)
    return max(1, target - d), target + d, target, kind


def _qualifier_kind(*words: Optional[str]) -> str:
    for w in words:
        w = " ".join((w or "").split())
        for rx, kind in _KIND:
            if w and rx.match(w):
                return kind
    return "about"


def _is_content(text: str, m: re.Match, counted: bool = False) -> bool:
    before = _COUNT_CONTENT_BEFORE if counted else _CONTENT_BEFORE
    return bool(_CONTENT_AFTER.match(text, m.end()) or before.search(text[:m.start()]))


def parse_length(text: Optional[str]) -> Optional[LengthSpec]:
    """The length the request names, or None. The first length phrase wins, as the user reads it."""
    text = text or ""
    found = []
    for rx in (_BETWEEN, _PREP_FORM, _LIMIT_FORM, _ADJ_FORM, _TRAILING_FORM, _COUNT_FORM):
        for m in rx.finditer(text):
            if _is_content(text, m, counted=rx is _COUNT_FORM):
                continue
            g = m.groupdict()
            n, n2 = to_int(g["n"]), to_int(g["n2"]) if g.get("n2") else None
            unit = _UNITS.get(re.sub(r"\s+", " ", g["u"].lower()))
            if not n or not unit or (g.get("n2") and not n2):
                continue
            if unit == "words" and n < 5:
                continue          # "explain this in two words" is a style of answer, not a length to plan for
            kind = _qualifier_kind(g.get("q"), g.get("q2"), (g.get("tail") or "").strip())
            if unit == "characters" and kind == "about":
                kind = "at_most"  # "a 280-character tweet" is a ceiling
            if rx is _LIMIT_FORM and (g["q"] or "").lower() == "count":
                kind = "about"
            lo, hi, target, kind = _window(unit, n, kind, n2)
            found.append((m.start(), LengthSpec(unit, target, lo, hi, kind, m.start(), m.end())))
    return min(found, key=lambda f: f[0])[1] if found else None


def strip_length(text: str, spec: Optional[LengthSpec]) -> str:
    """The request without its length phrase: 'an article on weather in 100 words' -> 'an article on weather'."""
    if spec is None or spec.unit in COUNTED_UNITS:
        return text           # "write two paragraphs about Rome": the count reads as part of the task, leave it
    head, tail = text[:spec.start].rstrip(" ,;-–"), text[spec.end:].lstrip(" ,;.")
    return (head + (" " + tail if tail else "")).strip()


_SIZE_WORD = re.compile(r"\b(short|brief|quick|concise|long|lengthy)\b", re.I)


def size_word(text: Optional[str]) -> Optional[str]:
    """'short' | 'long' | None from an adjective in the request ('write a short article on ...')."""
    m = _SIZE_WORD.search(text or "")
    if not m:
        return None
    return "short" if m.group(1).lower() in ("short", "brief", "quick", "concise") else "long"


# ---------------------------------------------------------------------------
# Counting the way the editor counts
# ---------------------------------------------------------------------------
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$", re.M)
_MARKS = re.compile(r"^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)", re.M)


def visible_text(markdown: str) -> str:
    """What the reader sees once the editor renders the Markdown: no image markup, URLs, pipes or #/>/- markers."""
    t = _IMAGE.sub(" ", markdown or "")
    t = _LINK.sub(r"\1", t)
    t = _TABLE_RULE.sub(" ", t)
    t = _MARKS.sub("", t)
    return re.sub(r"[|*_`~]+", " ", t)


def count_words(markdown: str) -> int:
    """Every word the editor's counter sees: title, headings, list items, table cells and captions included."""
    return word_count(visible_text(markdown))


def measure(spec: LengthSpec, markdown: str) -> int:
    """The output measured in the unit the user asked for."""
    if spec.unit == "words":
        return count_words(markdown)
    if spec.unit == "characters":
        return len(" ".join(visible_text(markdown).split()))
    blocks = [b for b in parse(markdown or "") if b.kind != "blank"]
    if spec.unit == "paragraphs":
        return sum(b.kind == "paragraph" for b in blocks)
    if spec.unit == "bullets":
        return len(re.findall(r"^\s*(?:[-*+]|\d+[.)])\s+\S", markdown or "", re.M))
    if spec.unit == "lines":
        return sum(1 for line in (markdown or "").splitlines() if line.strip())
    prose = " ".join(b.text for b in blocks if b.kind in ("paragraph", "list", "blockquote"))
    return len(re.findall(r"[^.!?]+[.!?]+(?=\s|$)", visible_text(prose)))


def trim_to(spec: LengthSpec, markdown: str) -> str:
    """Last resort for an 'at most' that is still over: drop whole sentences from the end, never cut one in half."""
    if spec.unit != "words" or count_words(markdown) <= spec.high:
        return markdown
    parts = re.split(r"(?<=[.!?])(\s+)", markdown.rstrip())
    while len(parts) > 1 and count_words("".join(parts)) > spec.high:
        parts = parts[:-2] if len(parts) >= 3 else parts[:-1]
    return "".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"placeholder::[0-9A-Za-z-]+")


def placeholders(markdown: str) -> list:
    """Image placeholders in a reply. A length repair must keep them exactly, or a started (billed) picture is orphaned."""
    return _PLACEHOLDER.findall(markdown or "")


def repair_prompt(spec: LengthSpec, reply: str, measured: int) -> str:
    """The one follow-up a reply gets when it misses the length: its real count, the window, and what must not change."""
    direction = "Shorten" if measured > spec.high else "Lengthen"
    return (f"LENGTH CHECK: your answer below has {measured} {spec.unit}; the task asks for {spec.describe()} "
            f"(acceptable: {spec.low}-{spec.high} {spec.unit}, counting every word including titles, headings and list items). "
            f"{direction} it to fit. Keep its facts, its order and its Markdown structure; keep every image line exactly as it is; "
            "do not add a title. Output only the rewritten text.\n\nYOUR ANSWER:\n" + reply)
```

**Real output of `parse_length` on a test set** (run in the scratch copy):

| Request | unit, target, window, kind | Topic after `strip_length` |
|---|---|---|
| Write an article on Weather Forcast in 100 Words | words, 100, 90–110, about | Write an article on Weather Forcast |
| Write an article on weather in 50 words | words, 50, 45–55, about | Write an article on weather |
| write a paragraph about dogs in only 100 words | words, 100, 90–110, about | write a paragraph about dogs |
| Write a 1,500-word blog post about remote work | words, 1500, 1350–1650, about | Write a blog post about remote work |
| write a blog post about SEO under 150 words | words, 150, 90–150, at_most | write a blog post about SEO |
| Write about the sea in no more than 80 words | words, 80, 48–80, at_most | Write about the sea |
| Write an essay of at least 300 words on climate | words, 300, 300–900, at_least | Write an essay on climate |
| Write a poem about rain, 40 words. | words, 40, 35–45, about | Write a poem about rain |
| Summarise this in 3 sentences | sentences, 3, 3–3, exact | (kept) |
| Write two paragraphs about Rome | paragraphs, 2, 2–2, exact | (kept) |
| Give me 5 bullet points on safety | bullets, 5, 5–5, exact | (kept) |
| Write a 280-character tweet about our launch | characters, 280, 168–280, at_most | Write a tweet about our launch |
| Write about cats in one hundred words | words, 100, 90–110, about | Write about cats |
| Write about cats in a hundred and fifty words | words, 150, 135–165, about | Write about cats |
| Write between 80 and 120 words about tea | words, 100, 80–120, range | Write about tea |
| write about tea in 100-150 words | words, 125, 100–150, range | write about tea |
| Write a product description, word limit of 60 | words, 60, 36–60, at_most | Write a product description |
| Write exactly 50 words about the moon | words, 50, 48–52, exact | Write about the moon |
| Write an article on weather in about 1.5k words | words, 1500, 1350–1650, about | Write an article on weather |
| List 10 words that rhyme with cat | **None** (content) | |
| give me 5 words for happy | **None** (content) | |
| Write about the 100 words every child should know | **None** (content) | |
| a list of 20 words related to winter | **None** (content) | |
| Translate these 3 sentences into French | **None** (content) | |
| Explain this with two words | **None** (under 5 words is a style, not a length) | |
| Write a short article on weather | None (`size_word` → "short") | |

`count_words` on a sample with a title, heading, table, quote, picture with caption, and a list with a link returned **27**, which
matches a hand count of what a reader sees. `trim_to` for "under 12 words" on a 14-word, 4-sentence text kept the first two
sentences (8 words). It drops whole sentences only, never cuts one in half.

## 5. Changes to existing files, with current and replacement code

### 5.1 `src/request_plan.py`: use the new parser

**Current (`src/request_plan.py:144-170`):**

```python
_WORDCOUNT = re.compile(r"\b(\d[\d,]{2,5})[- ]?(?:words?|word\s+count)\b", re.I)


def extract_word_count(text: Optional[str]) -> Optional[int]:
    """The length the user asked for in their own words ("a 1,500-word post" -> 1500), else None."""
    m = _WORDCOUNT.search(text or "")
    return int(m.group(1).replace(",", "")) if m else None


def needs_advanced_model(text: Optional[str], use_web: bool, image_count: int) -> bool:
    ...
    if _LONGFORM.search(t):
        return True
    m = _WORDCOUNT.search(t)
    if m and int(m.group(1).replace(",", "")) >= 500:
        return True
    return False
```

**Replacement:**

```python
def extract_word_count(text: Optional[str]) -> Optional[int]:
    """The number of words the user asked for in their own words ("a 1,500-word post", "in 50 words") -> int, else None."""
    from src.length_spec import parse_length
    spec = parse_length(text)
    return spec.target if spec is not None and spec.unit == "words" else None


def needs_advanced_model(text: Optional[str], use_web: bool, image_count: int) -> bool:
    ...
    if _LONGFORM.search(t):
        return True
    asked = extract_word_count(t)
    if asked is not None and asked >= 500:
        return True
    return False
```

`_WORDCOUNT` is deleted. The import is local because `src/length_spec.py` imports `src.longform.markdown_ast`, and `request_plan`
is imported very early.

### 5.2 `config.py`: three settings

**Current (`config.py:353-354`):**

```python
    blog_intent_regex_enabled: bool = Field(True, alias="BLOG_INTENT_REGEX_ENABLED")  # promote "write a blog post about..." from general
    blog_intent_shadow_mode: bool = Field(False, alias="BLOG_INTENT_SHADOW_MODE")  # log the promotion decision only
```

**Add after it:**

```python
    # Follow the length the user names ("in 100 words") on Write, Draft and blog (src/length_spec.py). Off = the old behaviour.
    length_contract_enabled: bool = Field(True, alias="LENGTH_CONTRACT_ENABLED")
    # One follow-up call when a Write/Draft reply misses the length; off = measure and warn only.
    length_repair_enabled: bool = Field(True, alias="LENGTH_REPAIR_ENABLED")
    # A named length below this goes to the single-call writer, not the staged blog pipeline (which plans sections, a table,
    # quotes and pictures).
    blog_min_words: int = Field(600, ge=250, le=4000, alias="BLOG_MIN_WORDS")
```

### 5.3 `src/text_processing_agent.py`: send short pieces to the single-call writer

**Current (`src/text_processing_agent.py:2599-2613`):** shown in section 2.2. Any "write an article on …" becomes `blog_writer`.

**Replacement:**

```python
    # The length the user named ("in 100 words"), read once for every writer below (src/length_spec.py).
    from .length_spec import parse_length
    length_spec = parse_length(query) if settings.length_contract_enabled else None
    # A short piece is one call to the free-form writer, not an outline with sections, a table, quotes and a picture.
    short_piece = length_spec is not None and (length_spec.unit != 'words' or length_spec.high < settings.blog_min_words)

    if task == 'code' and request_type == 'general':
        force_agent_type = 'code_writer'
    elif (request_type == 'general' and not image_urls and settings.blog_ai_enabled and not design_mode and not document_mode
          and not (image_intent and image_intent['kind'] in ('generate', 'edit'))
          and (settings.blog_intent_regex_enabled or settings.blog_intent_shadow_mode)):
        from .blog.intent import detect_blog_intent
        blog_decision = detect_blog_intent(query)
        if blog_decision['kind'] == 'blog' and short_piece:
            blog_decision = {'kind': 'other', 'rule': 'length_below_blog'}
        trace_set(request_id, blog_intent=blog_decision['rule'])
        if blog_decision['kind'] == 'blog':
            ...  # unchanged
```

**Why this fixes the reported case:** "Write an article on Weather Forcast in 100 Words" → spec (100, 90–110) → `short_piece` is true
→ the request stays with the free-form writer. That writer gets a length line and a measured check (5.4), instead of a 4-section
plan with a floor of 400 words.

Requests with no number, or with 600+ words, take exactly the path they take today.

### 5.4 `src/text_processing_agent.py`: tell the writer the length, measure, repair once

**Current (`src/text_processing_agent.py:2932-2934`):**

```python
    chunks = []
    validate_output = agent_type == 'code_writer' and settings.code_validation_enabled
    max_tokens = OUTPUT_TOKENS[prepared_context.metadata.action] if document_mode else refinement_max_tokens(agent_type, query)
```

**Replacement:**

```python
    chunks = []
    validate_output = agent_type == 'code_writer' and settings.code_validation_enabled
    # Write and Draft follow a named length: told up front, measured after, repaired once (src/length_spec.py).
    length_check = length_spec if (length_spec is not None and (
        (agent_type == FREE_FORM_AGENT and not document_mode and operation_type == 'general')
        or (document_mode and prepared_context.metadata.action == 'draft'))) else None
    max_tokens = OUTPUT_TOKENS[prepared_context.metadata.action] if document_mode else refinement_max_tokens(agent_type, query)
    if length_check is not None and document_mode and length_check.unit == 'words':
        # Draft's 1,000-token budget cuts "draft 900 words here" in half. Free-form Write keeps the provider default on purpose:
        # on reasoning models max_tokens includes thinking, so a tight cap there would cut the answer.
        max_tokens = max(max_tokens, min(8_000, 3 * length_check.high))
```

**Current (`src/text_processing_agent.py:2969-2970`):**

```python
                stream = _agent_text(agent, prepared_context.prompt if document_mode else query, images, provider, use_tools,
                                     request_id, output_limit=output_limit)
```

**Replacement:**

```python
                model_input = prepared_context.prompt if document_mode else query
                if length_check is not None:
                    # In the user message, after the task: the system prompt stays byte-identical, so prompt caching still hits.
                    model_input = model_input + "\n\n" + length_check.instruction()
                stream = _agent_text(agent, model_input, images, provider, use_tools, request_id, output_limit=output_limit)
```

**Current (`src/text_processing_agent.py:2990-2991`):**

```python
                    if validate_output:
                        chunks.append(chunk)
```

**Replacement:**

```python
                    if validate_output or length_check is not None:
                        chunks.append(chunk)
```

**Current (`src/text_processing_agent.py:3056-3058`), end of the code-repair block:**

```python
                raise CodeValidationError(errors)

    yield from graph_suggestions()
```

**Replacement (new block between the two):**

```python
                raise CodeValidationError(errors)

    if length_check is not None and not (cancel_event is not None and cancel_event.is_set()):
        reply = ''.join(c for c in chunks if isinstance(c, str))
        yield from _enforce_length(length_check, reply, agent, model_input, images, provider, model_id, request_id)

    yield from graph_suggestions()
```

**New helper (next to `_agent_text`, around `src/text_processing_agent.py:2242`).** It follows the same pattern as the code-repair
block at `:3036-3050`, which already sends `__content_replace__`:

```python
def _enforce_length(spec, reply, agent, model_input, images, provider, model_id, request_id):
    """
    Measure a finished Write/Draft reply the way the editor counts it. If it misses the asked length: one rewrite call, kept only if it
    lands closer and keeps every image placeholder; an "at most" still over is trimmed at a sentence boundary. The new text replaces the
    streamed one (`content_replace`); a reply that still misses is reported as a `length` warning, never silently.
    """
    from agno.run.base import RunStatus
    from .length_spec import measure, placeholders, repair_prompt, trim_to
    from .observability import record_text_usage
    first = measure(spec, reply)
    result, final = {'unit': spec.unit, 'asked': spec.describe(), 'first': first, 'status': 'ok'}, reply
    if not spec.contains(first) and settings.length_repair_enabled and reply.strip():
        repaired = agent.run(model_input + '\n\n' + repair_prompt(spec, reply, first), stream=False, images=images)
        if not (getattr(repaired, 'status', None) == RunStatus.error or getattr(repaired, 'event', None) == RunEvent.run_error):
            record_text_usage(request_id, model_id, getattr(repaired, 'metrics', None), feature='length_repair')
            candidate = (getattr(repaired, 'content', '') or '').strip()
            if (candidate and placeholders(candidate) == placeholders(reply)
                    and abs(measure(spec, candidate) - spec.target) < abs(first - spec.target)):
                final, result['status'] = candidate + '\n', 'repaired'
    if spec.kind == 'at_most' and measure(spec, final) > spec.high:
        final, result['status'] = trim_to(spec, final), 'trimmed'
    result['delivered'] = measure(spec, final)
    trace_set(request_id, length=result)
    if final != reply:
        yield {'__content_replace__': final}
    if not spec.contains(result['delivered']):
        yield {'__warnings__': [{'kind': 'length', 'detail':
                                 f"Asked for {spec.describe()}; the text has {result['delivered']} {spec.unit}."}]}
```

**Why this design:**
- **Ask first.** The length line is free and makes the first answer land in range most of the time.
- **Check every time.** It costs nothing: `measure` is local code.
- **Repair only on a miss.** That's one extra call, a few seconds, and only when needed.
- **Never trade a reply for a worse one.** The repair is kept only if it lands closer to the target.
- **Never orphan a picture.** A repair that drops an image placeholder is thrown away, because the picture is billed work.
- **`content_replace` is already on and handled.** `CODE_REPAIR_STREAM_EVENT` is on by default (`docs/FRONTEND_HANDOFF.md`), and
  non-streaming mode already handles `__content_replace__` (`src/text_processing_agent.py:3190`).

### 5.5 `api.py` and `src/text_processing_agent.py`: keep every warning, not only the last batch

Today a second `__warnings__` chunk **overwrites** the first. A `length` warning after a Draft would wipe out the document-context
warnings.

**Current (`api.py:1572-1574`):**

```python
                    if isinstance(chunk, dict) and '__warnings__' in chunk:
                        stream_warnings = chunk['__warnings__']
                        continue
```

**Replacement:**

```python
                    if isinstance(chunk, dict) and '__warnings__' in chunk:
                        stream_warnings = stream_warnings + [w for w in chunk['__warnings__'] if w not in stream_warnings]
                        continue
```

**Current (`src/text_processing_agent.py:3187-3189`):**

```python
        elif isinstance(chunk, dict) and "__warnings__" in chunk:
            if meta is not None:
                meta["warnings"] = chunk["__warnings__"]
```

**Replacement:**

```python
        elif isinstance(chunk, dict) and "__warnings__" in chunk:
            if meta is not None:
                seen = meta.get("warnings") or []
                meta["warnings"] = seen + [w for w in chunk["__warnings__"] if w not in seen]
```

### 5.6 `src/blog/schema.py`: let short posts have 2 sections and short edges

**Current (`src/blog/schema.py:27-29`):**

```python
SECTIONS = (3, 8)
SECTION_WORDS = (60, 800)
EDGE_WORDS = (40, 300)          # intro and conclusion
```

**Replacement:**

```python
SECTIONS = (2, 8)               # a 250-450-word post has 2 body sections
SECTION_WORDS = (60, 800)
EDGE_WORDS = (25, 300)          # intro and conclusion; a 300-word post has ~30-word edges
```

### 5.7 `src/blog/plan.py`: scale the structure to the number and budget the whole post

**Current:** shown in section 2.3 (`:18-22`, `:81-83`, `:94-128`).

**Replacement (whole length/structure part of the file):**

```python
from src.length_spec import parse_length, size_word, strip_length
from src.request_plan import image_count_requested, needs_web

LENGTH_WORDS = {"short": 800, "standard": 1450, "long": 2200}
DEFAULT_LENGTH = "standard"
# A brief can name any number ("a 50,000-word post"). Outside these bounds the post is not a blog post: the plan clamps the length,
# says so in `reasons`, and the service reports it to the user as a `length` warning. Below BLOG_MIN_WORDS a Write request never
# reaches this pipeline (text_processing_agent routes it to the single-call writer); only a declared type=generate_blog can.
MIN_WORDS, MAX_WORDS = 250, 4000
BRIEF_MAX_CHARS = 5000
# Words the editor counts that are not prose. Used only when the user named a number, so the WHOLE post lands on it.
TITLE_WORDS, HEADING_WORDS, CAPTION_WORDS, TABLE_WORDS = 22, 5, 12, 45
EXPLICIT_PART_TOLERANCE = 0.15   # per part, when the user named a number (default posts keep validate.LENGTH_TOLERANCE)
```

```python
@dataclass(frozen=True)
class BlogRequest:
    ...  # fields unchanged

    @property
    def topic(self) -> str:
        """The brief without its length phrase, so "in 100 words" never becomes the subject of the post."""
        return strip_length(self.brief, parse_length(self.brief))


@dataclass(frozen=True)
class BlogPlan:
    ...  # existing fields unchanged, plus:
    words_asked: Optional[int] = None        # the number the user named, before the clamp; None = default length
    visible_words: int = 0                   # what the editor should show at the end (prose + title, headings, captions, tables)
    part_tolerance: Optional[float] = None   # per-part length tolerance; None = validate.LENGTH_TOLERANCE


def section_count(target_words: int) -> int:
    if target_words <= 450:
        return 2
    if target_words <= 700:
        return 3
    return 4 if target_words <= 1000 else 5 if target_words <= 1600 else 6 if target_words <= 2200 else 7
```

```python
    # inside plan_blog(), replacing everything from "# length:" down to the images block
    # length: the brief's own number, else the option, else "short"/"long" in the brief, else standard
    spec = parse_length(request.brief)
    asked = spec.target if spec is not None and spec.unit == "words" else None
    if asked is not None:
        target, reasons["length"] = asked, f"brief asked for {spec.describe()}"
    elif request.length in LENGTH_WORDS:
        target, reasons["length"] = LENGTH_WORDS[request.length], f"option {request.length}"
    elif size_word(request.brief) in LENGTH_WORDS:
        bucket = size_word(request.brief)
        target, reasons["length"] = LENGTH_WORDS[bucket], f"'{bucket}' in the brief"
    else:
        target, reasons["length"] = LENGTH_WORDS[DEFAULT_LENGTH], "default"
    clamped = max(MIN_WORDS, min(MAX_WORDS, target))
    if clamped != target:
        reasons["length"] += f", clamped to {clamped}"
    target = clamped

    sections = section_count(target)
    edge = min(180, max(80 if target >= 800 else 30, round(target * 0.085 / 10) * 10))
    tables = 0 if target < 600 else 2 if target >= 2000 else 1
    quotes = 0 if target < 450 else 1 if target < 800 else 3 if target >= 2000 else 2

    # images: "no images" in the brief, else the brief's count, else the option, else 1 + one per 700 words (none under 600)
    asked_images: Optional[int] = None
    if _NO_IMAGES.search(request.brief):
        images, reasons["images"] = 0, "brief asked for none"
    elif image_count_requested(request.brief) > 0:
        asked_images = images = image_count_requested(request.brief)
        reasons["images"] = "count from the brief"
    elif request.images is not None:
        asked_images = images = request.images
        reasons["images"] = "option"
    elif target < 600:
        images, reasons["images"] = 0, "auto: too short for a picture"
    else:
        images, reasons["images"] = 1 + target // 700, "auto"
    images = max(0, min(images, cap, sections))
    source = (settings.blog_image_source or "").strip().lower()
    if images and source != "generate":
        images, reasons["images"] = 0, f"BLOG_IMAGE_SOURCE={source or 'unset'}: no image backend"
    elif images and not image_available:
        images, reasons["images"] = 0, "no usable image provider"

    # A named number is what the editor will show, so the words that are not prose come out of the prose budget.
    # The default length (no number) keeps its prose-only budget: those posts are unchanged.
    visible, prose = target, target
    if asked is not None:
        overhead = TITLE_WORDS + HEADING_WORDS * (sections + 1) + CAPTION_WORDS * images + TABLE_WORDS * tables
        prose = max(target // 2, target - overhead)
        edge = min(edge, max(30, round(prose * 0.12)))
        reasons["length"] += f"; {prose} words of prose + ~{target - prose} in title, headings, captions and tables"

    use_web = needs_web(request.brief)
    reasons["web"] = "cue in the brief" if use_web else "no freshness/research cue"
    return BlogPlan(
        target_words=prose, length=length_bucket(target), sections=sections, intro_words=edge,
        conclusion_words=edge, tables=tables, quotes=quotes, images=images, use_web=use_web,
        max_prose_run=int(settings.blog_max_prose_run), images_asked=asked_images, words_asked=asked,
        visible_words=visible, part_tolerance=EXPLICIT_PART_TOLERANCE if asked is not None else None, reasons=reasons,
    )
```

**Old vs new plan, real output** (both functions run on the same briefs, no model call):

| Brief | Today | Proposed |
|---|---|---|
| …weather in 100 words (only if sent as `type=generate_blog`) | prose 300, 4 sections, edges 80, table/quotes/pictures 1/2/1 | prose 213, editor 250 (clamped, **warned**), 2 sections, edges 30, 0/0/0 |
| …weather in 300 words | prose 300, 4 sections, edges 80, 1/2/1 | prose 263, editor 300, 2 sections, edges 30, 0/0/0 |
| …weather in 600 words | prose 600, 4 sections, edges 80, 1/2/1 | prose 501, editor 600, 3 sections, edges 50, 1/1/1 |
| a 1,500-word blog post… | prose 1500, 5 sections, edges 130, 1/2/3 | prose 1367, editor 1500, 5 sections, edges 130, 1/2/3 |
| an article on weather (no number) | prose 1450, 5 sections, edges 120, 1/2/3 | **identical**: prose 1450, 5, 120, 1/2/3 |
| a **short** article on weather | prose 1450 (word ignored) | prose 800, 4 sections, edges 80, 1/2/2 |
| a **long** article on weather | prose 1450 (word ignored) | prose 2200, 6 sections, edges 180, 2/3/4 |

The default post (no number, no "short"/"long") is byte-for-byte the same plan. That keeps the ~1,450-word GPT output you chose to
keep. "comprehensive" and "in-depth" are deliberately **not** size words, because they're common in ordinary briefs and would move
those posts to 2,200 words.

### 5.8 `src/blog/validate.py` and `src/blog/draft.py`: tighter tolerance when a number was named

**Current (`src/blog/validate.py:211-212` and `:235-236`):**

```python
def lint(text: str, *, unit: Optional[str] = None, expect_table: bool = False, expect_quotes: int = 0,
         target_words: int = 0, max_prose_run: int = 220) -> Lint:
    ...
    if target_words:
        low, high = round(target_words * (1 - LENGTH_TOLERANCE)), round(target_words * (1 + LENGTH_TOLERANCE))
```

**Replacement:**

```python
def lint(text: str, *, unit: Optional[str] = None, expect_table: bool = False, expect_quotes: int = 0,
         target_words: int = 0, max_prose_run: int = 220, tolerance: Optional[float] = None) -> Lint:
    ...
    if target_words:
        slack = LENGTH_TOLERANCE if tolerance is None else tolerance
        low, high = round(target_words * (1 - slack)), round(target_words * (1 + slack))
```

**Current (`src/blog/draft.py:134-136`):**

```python
def _lint(text: str, unit: Unit, plan: BlogPlan) -> Lint:
    return lint(text, unit=unit.id, expect_table=unit.table is not None, expect_quotes=1 if unit.quote else 0,
                target_words=unit.target_words, max_prose_run=plan.max_prose_run)
```

**Replacement:**

```python
def _lint(text: str, unit: Unit, plan: BlogPlan) -> Lint:
    return lint(text, unit=unit.id, expect_table=unit.table is not None, expect_quotes=1 if unit.quote else 0,
                target_words=unit.target_words, max_prose_run=plan.max_prose_run, tolerance=plan.part_tolerance)
```

A part that is more than 15% off now gets the existing one repair (`write_unit`, `src/blog/draft.py:157-170`) instead of passing.

### 5.9 `src/blog/outline.py` and `src/blog/draft.py`: stop sending the length phrase as topic

**Current (`src/blog/outline.py:66-67`)** `"brief": request.brief,` and **(`src/blog/draft.py:108`)** `"brief": request.brief,`

**Replacement in both:**

```python
        "brief": request.topic,      # without "in 100 words": the plan carries the length, the topic is only the subject
```

### 5.10 `src/blog/service.py`: tell the user when the post misses or was clamped

The blog streams each part as it's written, so it can't be swapped afterwards like Write. It reports instead.

**Current (`src/blog/service.py:175-176`):**

```python
    if warnings:
        yield {"__warnings__": warnings}
```

**Replacement:**

```python
    if plan.words_asked is not None:
        from src.length_spec import count_words
        shown = count_words(document)
        if plan.words_asked != plan.visible_words:
            warnings.append({"kind": "length", "detail": f"{plan.words_asked} words were asked for; a blog post here is "
                             f"{plan.visible_words} words at {'least' if plan.words_asked < plan.visible_words else 'most'}."})
        elif abs(shown - plan.visible_words) > 0.15 * plan.visible_words:
            warnings.append({"kind": "length", "detail": f"{plan.words_asked} words were asked for; the post has {shown}."})
        trace_set(request_id, length={"asked": plan.words_asked, "planned": plan.visible_words, "delivered": shown})
    if warnings:
        yield {"__warnings__": warnings}
```

### 5.11 `docs/FRONTEND_HANDOFF.md`: document the new warning

Add `length` to the `done.warnings` kinds: `{kind: "length", detail}`. It means the text is outside the length the user asked for,
or a blog length was clamped.

Also ask the editor team to confirm that `content_replace` replaces the text inserted at the cursor (Draft), not only chat output.

## 6. Why this resolves the issue

| Cause today (section 2) | What changes | Result for "article … in 100 words" |
|---|---|---|
| 2.2 any "write an article on" goes to the blog | 5.3: a named length under 600 stays with the single-call writer | One call, one short piece, no sections/table/quotes/picture |
| 2.3a 100 clamped to 300, silently | 5.3 means Write never reaches the clamp; 5.7/5.10 clamp at 250 and **warn** for direct `generate_blog` | No hidden 3× inflation |
| 2.3b/c fixed 4 sections, 80/60-word floors, 1 table, 2 quotes, 1 picture | 5.6/5.7: structure scales with the number | A 600-word post gets 3 sections, 1 table, 1 quote, 1 picture |
| 2.4 ±35% per part, no whole-post check | 5.8: ±15% when a number was named; 5.4 measures the whole Write reply | Out-of-range text gets repaired, or the user is told |
| 2.5 prose-only count vs the editor's full count | `count_words` counts what the editor shows; 5.7 budgets title, heading, caption and table words | The number checked is the number the user sees |
| 2.6 "100 words" used as the topic | 5.9: `request.topic` drops the phrase | No more "A 100-word weather forecast…" title |
| 2.7 "50 words", "one hundred words", "under", "3 sentences" missed | 5.1 / section 4 parser | All of those are read (table in section 4) |
| 2.8 Write/Draft never told or checked | 5.4: length line, measure, one repair, sentence-boundary trim, warning | Same rule on every writer |

## 7. Tests and measurements before switching it on

1. **`tests/test_length_spec.py` (new).** Every row of the section 4 table as a parametrized case, plus:
   - `to_int`: "one hundred and fifty" → 150, "1.5k" → 1500, "2,000" → 2000.
   - `count_words` on Markdown with a title, table, quote, captioned picture and a linked list item → hand-counted number.
   - `trim_to` never cuts inside a sentence.
   - `strip_length` keeps counted units ("two paragraphs").
2. **`tests/test_blog_plan.py` (extend).**
   - No-number brief → plan unchanged (prose 1450, 5 sections, edges 120, 1/2/3).
   - "in 300 words" → 2 sections, no table/quote/picture.
   - "in 600 words" → 3 sections, prose + overhead = 600.
   - "short" → 800.
3. **Routing test.** `general` "Write an article on X in 100 words" is not routed to `blog_writer`; "in 1,500 words" and
   no-number are.
4. **`_enforce_length` test with a fake agent.** In range → nothing yielded. Over → one `run` call, then `content_replace`.
   Repair drops a placeholder → original kept and a `length` warning. "Under N" still over → trimmed.
5. **Old-vs-new parser comparison on a large prompt set** (the lesson from the document-actions work: wording rules need it). Run
   `extract_word_count` old vs new over the existing eval prompts (`scripts/document_actions/eval_auto.py` sets, the blog briefs,
   the chat test prompts). Every request whose result changes must be read by hand before the switch is turned on.
6. **Live test set (new script `scripts/length_eval.py`).**
   - Targets: 50, 100, 150, 300, 600, 1,450.
   - Shapes: "article … in N words", "a paragraph about … in N words", Draft at cursor, "under N words", "N sentences".
   - Providers: OpenAI, Claude, Gemini.
   - Record the first answer, whether a repair ran, the final count (with `count_words`) and the time.
   - **Pass bar:** ≥95% in the window, 100% within ±20%, "under N" never over, repair rate reported.
7. **No-regression check on the default blog.** `python scripts/evaluate_blog.py --live` with no number: still ~1,450 words and the
   same look (see memory: the blog output is pinned to gpt-5.2 on purpose).
8. **Full suite.** `.venv/bin/python -m pytest tests -q`.

## 8. Rollout

- Everything sits behind `LENGTH_CONTRACT_ENABLED`. Off means today's behaviour exactly.
- `LENGTH_REPAIR_ENABLED=false` keeps the measuring and warnings but skips the extra call, in case cost or latency needs it.
- `BLOG_MIN_WORDS` (default 600) can move after the live numbers come in.

## 9. Open points

1. **How does the editor count words?** Do headings, table cells, captions and hyphenated words count? `count_words` must match it
   exactly. Otherwise the server passes a check the user sees fail. One sample document counted by both sides settles it.
2. **Draft and `content_replace`.** Confirm the editor applies `content_replace` to text inserted at the cursor (Draft), not only to
   chat output.
3. **Declared `type=generate_blog` under 250 words.** It is clamped and warned (5.10). The alternative is to answer it with the
   single-call writer as well. That changes what the declared type returns, so it's a product decision.
4. **Edit, Shorten and Lengthen with a number** ("rewrite the intro in 50 words", "shorten to 200 words") should use the same
   parser and check. They're left out of this change on purpose and are the natural next step.
