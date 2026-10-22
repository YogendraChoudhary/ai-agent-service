"""
Stage 1: the outline. model -> parse -> normalise -> schema -> plan rules -> (one repair) -> rule-based fill.

The model proposes sections, angles and where the visuals go. Code decides what is binding: ids and image positions
are overwritten, intro and conclusion lengths come from the plan, and any visual the plan requires but the outline
lacks is added by rule (so a post never ships without its table), after a single repair attempt.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from config import settings
from src.email_template.validate import parse_json
from src.observability import trace_set

from . import schema as s
from .llm import LLM, CallSpec
from .plan import BlogPlan, BlogRequest
from .prompts import outline_system_prompt, outline_user_message, repair_outline_message
from .text import clean_alt, clean_caption, clean_heading, clean_line

logger = logging.getLogger(__name__)

OUTLINE_TEMPERATURE = 0.4
# Findings code can satisfy with no loss of quality, so they never cost a model call. A pull-quote directive has no content
# of its own (the section writer lifts the sentence later), a length is a proportion, a duplicate is dropped. A table or a
# picture is different: the model's own columns and picture prompt beat generic ones, so those still go back to the model.
# Measured: 3 of 60 fast-model outlines failed only on `visual_gap` and paid a ~6.5 s repair for something finalize() does.
CODE_FIXABLE = frozenset({"visual_gap", "quotes", "words_total", "duplicate_visual"})
MAX_EXTRA_VISUALS = 1       # tables and quotes beyond the plan's minimum before the post stops reading as editorial
_COMPARE_RE = re.compile(r"\b(compar\w*|versus|vs\.?|options?|costs?|pric\w*|steps?|checklist|timeline|rank\w*|"
                         r"pros|cons|trade-?offs?|tools?|alternatives?|differences?)\b", re.I)
_NEGATORS = {"no", "not", "without", "avoid", "avoiding", "never", "zero", "free", "nor"}
_TEXT_NOUNS = {"text", "words", "word", "letters", "lettering", "typography", "numbers", "digits", "logo", "logos",
               "watermark", "watermarks", "caption", "captions", "slogan", "headline", "headlines", "label",
               "labels", "signage", "chart", "graph", "infographic", "screenshot", "poster", "billboard", "banner"}
_REAL_PEOPLE_RE = re.compile(r"\b(?:celebrit\w+|politician|president|prime minister|actor|actress|footballer|influencer)\b",
                             re.I)
_QUOTED_RE = re.compile(r"[\"“”]")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass
class OutlineCheck:
    outline: Optional[s.Outline] = None          # None = no usable structure (bad JSON or schema)
    errors: List[s.Issue] = field(default_factory=list)


@dataclass
class OutlineResult:
    outline: s.Outline
    repaired: bool = False
    filled: List[str] = field(default_factory=list)       # what the rule-based fill had to add or change
    warnings: List[dict] = field(default_factory=list)


def outline_request(request: BlogRequest, plan: BlogPlan) -> dict:
    """The outline call's user data. Plan numbers are what the model must follow."""
    return {
        "brief": request.brief,
        "style_notes": request.style_notes,
        "audience": request.audience,
        "tone": request.tone,
        "plan": {
            "target_words": plan.target_words,
            "sections": plan.sections,
            "section_words": plan.section_words,
            "intro_words": plan.intro_words,
            "conclusion_words": plan.conclusion_words,
            "images": plan.images,
            "tables_at_least": plan.tables,
            "quotes_at_least": plan.quotes,
        },
        "web_research": plan.use_web,
    }


# ---------------------------------------------------------------------------
# Normalise: make the model's strings safe to write into Markdown, overwrite what code owns
# ---------------------------------------------------------------------------

def _clean_visual(v: Any) -> Any:
    if not isinstance(v, dict):
        return v
    kind = v.get("kind")
    if kind == "image":
        v = {**v, "position": s.IMAGE_POSITION}
        for key, fn in (("prompt", clean_line), ("alt", clean_alt), ("caption", clean_caption)):
            if isinstance(v.get(key), str):
                v[key] = fn(v[key])
    elif kind == "table":
        if isinstance(v.get("title"), str):
            v = {**v, "title": clean_line(v["title"])}
        if isinstance(v.get("columns"), list):
            v = {**v, "columns": [clean_line(c) if isinstance(c, str) else c for c in v["columns"]]}
    return v


def trim_guidance(text: str, limit: int) -> str:
    """
    Guidance for the writer (a section's angle, the intro's hook) that runs past its limit, cut at the last sentence that fits,
    else at the last word. Readers never see it, and a model cannot count characters: asking it to shorten one failed as often
    as it worked (a post was lost to a 301-character angle after its repair).
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    if ends and ends[-1] >= limit // 2:
        return head[:ends[-1]].rstrip()
    return (head.rsplit(" ", 1)[0] if " " in head else head).rstrip(" ,;:-–—")


def normalise(data: dict) -> dict:
    """Never raises on a wrong shape: the schema reports that. Strings Markdown will carry are cleaned."""
    data = dict(data)
    if isinstance(data.get("title"), str):
        data["title"] = clean_heading(data["title"])
    if isinstance(data.get("subtitle"), str):
        data["subtitle"] = clean_line(data["subtitle"]) or None
    if isinstance(data.get("intro"), dict) and isinstance(data["intro"].get("hook"), str):
        data["intro"] = {**data["intro"], "hook": trim_guidance(clean_line(data["intro"]["hook"]), s.HOOK_MAX)}
    if isinstance(data.get("conclusion"), dict) and isinstance(data["conclusion"].get("heading"), str):
        data["conclusion"] = {**data["conclusion"], "heading": clean_heading(data["conclusion"]["heading"])}
    if isinstance(data.get("sections"), list):
        sections = []
        for i, sec in enumerate(data["sections"], 1):
            if isinstance(sec, dict):
                sec = {**sec, "id": f"s{i}"}                       # code-owned
                if isinstance(sec.get("heading"), str):
                    sec["heading"] = clean_heading(sec["heading"])
                if isinstance(sec.get("angle"), str):
                    sec["angle"] = trim_guidance(clean_line(sec["angle"]), s.ANGLE[1])
                if isinstance(sec.get("visuals"), list):
                    sec["visuals"] = [_clean_visual(v) for v in sec["visuals"]]
            sections.append(sec)
        data["sections"] = sections
    return data


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def image_prompt_problems(prompt: str) -> List[str]:
    """Why a picture prompt would produce text, logos or a real person in the image (negations like 'no text' are fine)."""
    problems = []
    if _QUOTED_RE.search(prompt):
        problems.append("it contains quoted text that would be drawn into the picture")
    tokens = [t.lower() for t in _TOKEN_RE.findall(prompt)]
    for i, tok in enumerate(tokens):
        if tok in _TEXT_NOUNS and not any(t in _NEGATORS for t in tokens[max(0, i - 4):i]):
            problems.append(f"it asks for '{tok}' in the picture")
            break
    if _REAL_PEOPLE_RE.search(prompt):
        problems.append("it names a kind of real, recognisable person")
    return problems


def plan_issues(o: s.Outline, plan: BlogPlan, *, prompts: bool = True) -> List[s.Issue]:
    """Rule findings on a schema-valid outline. `prompts=False` skips the picture-prompt lint (used after the fill)."""
    errs: List[s.Issue] = []
    if len(o.sections) != plan.sections:
        errs.append(s.Issue("section_count", f"Exactly {plan.sections} sections are required, not {len(o.sections)}."))
    total, want = sum(sec.target_words for sec in o.sections), plan.section_words
    if abs(total - want) > 0.15 * want:
        errs.append(s.Issue("words_total", f"The sections' target_words add up to {total}; they must add up to "
                                           f"about {want} (within 15%)."))
    for kind, need, extra in (("table", plan.tables, MAX_EXTRA_VISUALS), ("quote", plan.quotes, MAX_EXTRA_VISUALS),
                              ("image", plan.images, 0)):
        found = o.count(kind)
        if found < need or found > need + extra:
            wanted = f"Exactly {need}" if not extra else f"Between {need} and {need + extra}"
            errs.append(s.Issue(f"{kind}s", f"{wanted} {kind} directive(s) are required in the whole post, found {found}."))
    for sec in o.sections:
        for kind in ("image", "table", "quote"):
            if len(sec.of_kind(kind)) > 1:
                errs.append(s.Issue("duplicate_visual", f"Section {sec.id} has more than one {kind}; at most one "
                                                        "of each kind per section.", sec.id))
        for img in sec.of_kind("image") if prompts else ():
            for problem in image_prompt_problems(img.prompt):
                errs.append(s.Issue("image_prompt", f"Image prompt in {sec.id}: {problem}. Describe only the "
                                                    "scene: no text, letters, numbers, logos or real people.", sec.id))
    for a, b in zip(o.sections, o.sections[1:]):
        if not a.visuals and not b.visuals:
            errs.append(s.Issue("visual_gap", f"Sections {a.id} and {b.id} follow each other with no visual. "
                                              "Give one of them a visual.", b.id))
    return errs


def _schema_issues(err: ValidationError) -> List[s.Issue]:
    return [s.Issue("schema", f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}", field=".".join(str(p) for p in e["loc"]))
            for e in err.errors()]


def validate_outline(raw: Any, plan: BlogPlan) -> OutlineCheck:
    """Parse, normalise, schema-check and rule-check a model reply (a JSON string or an already-parsed dict)."""
    try:
        data = parse_json(raw) if isinstance(raw, str) else raw
    except (ValueError, json.JSONDecodeError) as e:
        return OutlineCheck(errors=[s.Issue("json", f"Reply is not a valid JSON object: {e}")])
    if not isinstance(data, dict):
        return OutlineCheck(errors=[s.Issue("json", "Reply must be a JSON object.")])
    try:
        outline = s.Outline.model_validate(normalise(data))
    except ValidationError as e:
        return OutlineCheck(errors=_schema_issues(e))
    return OutlineCheck(outline=outline, errors=plan_issues(outline, plan))


# ---------------------------------------------------------------------------
# Fill: when the rules are still broken after the repair, code completes the outline
# ---------------------------------------------------------------------------

def _spread(n: int, k: int) -> List[int]:
    """k indices spread evenly over range(n)."""
    if k <= 0:
        return []
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _fallback_image(sec: dict) -> dict:
    heading = sec["heading"].rstrip(".").replace('"', "")
    return {"kind": "image", "position": s.IMAGE_POSITION,
            "prompt": f"A clean editorial illustration representing {heading[:1].lower() + heading[1:]}",
            "alt": f"Illustration for {heading}", "caption": heading}


def _fallback_table(sec: dict) -> dict:
    return {"kind": "table", "title": f"{sec['heading']} at a glance"[:s.TABLE_TITLE[1]],
            "columns": ["Aspect", "Key point"], "rows_hint": 4}


def _has(sec: dict, kind: str) -> bool:
    return any(v["kind"] == kind for v in sec["visuals"])


def _add_images(secs: List[dict], missing: int) -> int:
    order = list(dict.fromkeys([0, len(secs) // 2, *range(len(secs))]))
    added = 0
    for i in order:
        if added == missing:
            break
        if not _has(secs[i], "image"):
            secs[i]["visuals"].append(_fallback_image(secs[i]))
            added += 1
    return added


def _add_tables(secs: List[dict], missing: int) -> int:
    added = 0
    for _ in range(missing):
        free = [i for i, sec in enumerate(secs) if not _has(sec, "table")]
        if not free:
            break
        pick = next((i for i in free if _COMPARE_RE.search(secs[i]["heading"] + " " + secs[i]["angle"])),
                    1 if 1 in free else free[0])
        secs[pick]["visuals"].append(_fallback_table(secs[pick]))
        added += 1
    return added


def _add_quotes(secs: List[dict], missing: int) -> int:
    added = 0
    for _ in range(missing):
        free = [i for i, sec in enumerate(secs) if not _has(sec, "quote")]
        if not free:
            break
        pick = max(free, key=lambda i: (secs[i]["target_words"], -i))
        secs[pick]["visuals"].append({"kind": "quote", "mode": "pull"})
        added += 1
    return added


def _count(secs: List[dict], kind: str) -> int:
    return sum(v["kind"] == kind for sec in secs for v in sec["visuals"])


def finalize(outline: s.Outline, plan: BlogPlan) -> Tuple[s.Outline, List[str]]:
    """
    Apply what code owns and complete what the rules require. Returns the outline and a note per thing that had
    to be added or changed beyond routine overwrites (ids, image positions, intro and conclusion lengths).
    """
    data = outline.model_dump(exclude_none=True)
    secs: List[dict] = data["sections"]
    notes: List[str] = []

    if len(secs) > plan.sections:
        notes.append(f"dropped {len(secs) - plan.sections} extra section(s)")
        del secs[plan.sections:]
    for sec in secs:                                   # at most one image, table and quote per section
        seen, kept = set(), []
        for v in sec["visuals"]:
            if v["kind"] not in seen:
                seen.add(v["kind"])
                kept.append(v)
        sec["visuals"] = kept

    def trim(kind: str, limit: int) -> None:
        found = [v for sec in secs for v in sec["visuals"] if v["kind"] == kind]
        if len(found) > limit:
            keep = {id(found[j]) for j in _spread(len(found), limit)}
            for sec in secs:
                sec["visuals"] = [v for v in sec["visuals"] if v["kind"] != kind or id(v) in keep]
            notes.append(f"removed {len(found) - limit} surplus {kind}(s)")

    trim("image", plan.images)
    added = _add_images(secs, plan.images - _count(secs, "image"))
    if added:
        notes.append(f"added {added} image(s)")
    for kind, need, adder in (("table", plan.tables, _add_tables), ("quote", plan.quotes, _add_quotes)):
        trim(kind, need + MAX_EXTRA_VISUALS)
        added = adder(secs, need - _count(secs, kind))
        if added:
            notes.append(f"added {added} {kind}(s)")
    gaps = 0
    for a, b in zip(secs, secs[1:]):
        if not a["visuals"] and not b["visuals"]:
            pick = a if a["target_words"] >= b["target_words"] else b
            # a table or image would cost more than the cap allows; a quote is the cheapest visual, so it is the
            # only one added here, and only while the cap has room (otherwise the model's own extra is moved)
            if _count(secs, "quote") >= plan.quotes + MAX_EXTRA_VISUALS:
                donor = next((x for x in secs if _has(x, "quote") and len(x["visuals"]) > 1), None)
                if donor is None:
                    continue
                donor["visuals"] = [v for v in donor["visuals"] if v["kind"] != "quote"]
            pick["visuals"].append({"kind": "quote", "mode": "pull"})
            gaps += 1
    if gaps:
        notes.append(f"added {gaps} quote(s) to close gaps between visuals")

    total, want = sum(sec["target_words"] for sec in secs), plan.section_words
    if total and abs(total - want) > 0.15 * want:
        for sec in secs:
            sec["target_words"] = max(s.SECTION_WORDS[0], min(s.SECTION_WORDS[1], round(sec["target_words"] * want / total)))
        notes.append("rescaled section lengths to the plan")
    for i, sec in enumerate(secs, 1):                  # code-owned
        sec["id"] = f"s{i}"
        for v in sec["visuals"]:
            if v["kind"] == "image":
                v["position"] = s.IMAGE_POSITION
    data["intro"]["target_words"] = plan.intro_words
    data["conclusion"]["target_words"] = plan.conclusion_words
    return s.Outline.model_validate(data), notes


def outline_max_tokens(plan: BlogPlan) -> int:
    """JSON for 7 sections with image prompts runs to ~2,500 tokens; a cut-off outline is invalid JSON."""
    return max(2000, 400 + 330 * plan.sections)


# ---------------------------------------------------------------------------
# Streaming: the title opens the post before the rest of the outline exists
# ---------------------------------------------------------------------------

def top_level_strings(text: str) -> Dict[str, Optional[str]]:
    """
    The string (or null) values completed so far at the top of a JSON object that is still being written, in order, up to its
    first object, array or number. Nothing nested is read, so a table's own "title" can never pass for the post's.
    """
    found: Dict[str, Optional[str]] = {}
    start = text.find("{")
    if start < 0:
        return found
    pos, end = start + 1, len(text)

    def skip(chars: str) -> None:
        nonlocal pos
        while pos < end and text[pos] in chars:
            pos += 1

    while True:
        skip(" \t\r\n,")
        if pos >= end or text[pos] != '"':
            return found
        try:
            key, pos = json.decoder.scanstring(text, pos + 1, False)
        except ValueError:                              # the key is still being written
            return found
        skip(" \t\r\n")
        if pos >= end or text[pos] != ":":
            return found
        pos += 1
        skip(" \t\r\n")
        if pos < end and text[pos] == '"':
            try:
                found[key], pos = json.decoder.scanstring(text, pos + 1, False)
            except ValueError:                          # the value is still being written
                return found
        elif text.startswith("null", pos):
            found[key], pos = None, pos + 4
        else:
            return found


class TitleWatcher:
    """
    Listens to the outline reply as it streams and reports the title and subtitle the moment both are written. Code decides what
    counts: both must sit at the top of the object, ahead of any section; they are cleaned exactly as `normalise` cleans them and
    must fit the schema's limits (otherwise the outline fails its first check and nothing is shown early). It reports once, and
    build_outline makes the finished outline carry what it reported, so the post never opens with one title and ends with another.
    """

    def __init__(self, on_title: Callable[[str, Optional[str]], None]):
        self.on_title = on_title
        self.shown: Optional[Tuple[str, Optional[str]]] = None

    def feed(self, text: str) -> None:
        """`text` is everything written so far (a retried provider starts again from nothing; the first report stands)."""
        if self.shown is not None:
            return
        found = top_level_strings(text)
        if not isinstance(found.get("title"), str) or "subtitle" not in found:
            return
        title = clean_heading(found["title"]).strip()
        subtitle = clean_line(found["subtitle"] or "").strip() or None
        if not 3 <= len(title) <= s.TITLE_MAX or (subtitle is not None and len(subtitle) > s.SUBTITLE_MAX):
            return
        self.shown = (title, subtitle)
        self.on_title(title, subtitle)


def build_outline(llm: LLM, request: BlogRequest, plan: BlogPlan, request_id: Optional[str] = None,
                  on_title: Optional[Callable[[str, Optional[str]], None]] = None) -> OutlineResult:
    """
    Outline for the request. Raises BlogError when the model's structure is unusable even after one repair.
    `on_title(title, subtitle)` is called once, from the thread making the model call, as soon as the model has written both
    (the call is streamed); the outline returned then carries exactly those two strings.
    """
    system = outline_system_prompt(plan.length)
    messages = [{"role": "user", "content": outline_user_message(outline_request(request, plan))}]
    watcher = TitleWatcher(on_title) if on_title is not None else None
    listen = watcher.feed if watcher else None
    spec = CallSpec("outline", OUTLINE_TEMPERATURE, outline_max_tokens(plan), on_text=listen)
    reply = llm(system, messages, spec)
    check = validate_outline(reply, plan)

    repaired = False
    code_only = check.outline is not None and bool(check.errors) and all(e.kind in CODE_FIXABLE for e in check.errors)
    if check.errors:
        # What the first reply got wrong, for tuning: a rule code could satisfy without asking the model again shows up here.
        trace_set(request_id, outline_first_pass=[{"kind": e.kind, "message": e.message[:160]} for e in check.errors])
    if check.errors and settings.blog_repair_enabled and not code_only:
        logger.info(f"[{request_id}] outline needs repair: {[e.kind for e in check.errors]}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": repair_outline_message(s.format_errors(check.errors))}]
        check = validate_outline(llm(system, messages, CallSpec("outline_repair", OUTLINE_TEMPERATURE, spec.max_tokens,
                                                                on_text=listen)), plan)
        repaired = True
    if check.outline is None:
        raise s.BlogError("The model did not return a usable outline: " + "; ".join(e.message for e in check.errors[:3]),
                          check.errors)

    outline, notes = finalize(check.outline, plan)
    if watcher is not None and watcher.shown is not None:       # the reader has already seen these two
        outline = outline.model_copy(update={"title": watcher.shown[0], "subtitle": watcher.shown[1]})
    warnings = []
    if notes and not code_only:         # a lossless fix is not news to the client; it is in the trace (`outline_filled`)
        warnings.append({"kind": "outline_filled", "detail": "The outline was completed automatically: " + "; ".join(notes) + "."})
    left = plan_issues(outline, plan, prompts=False)
    if left:                                           # finalize should make this impossible; never ship it silently
        logger.warning(f"[{request_id}] outline still breaks rules after fill: {[e.kind for e in left]}")
        warnings.append({"kind": "outline_rules", "detail": "; ".join(e.message for e in left)})
    return OutlineResult(outline=outline, repaired=repaired, filled=notes, warnings=warnings)
