"""
Rewrite an existing email template from an instruction ("Make this a Black Friday sale email", "Make it shorter").

The model rewrites the copy; code keeps the design. After the reply is parsed, every block is matched to the
original by id and rebuilt from the ORIGINAL block: only reader-visible text is taken from the model. Ids, types,
order, links, assets, image blocks, heading levels, list styles, footer settings and the theme come from the
original whatever the model wrote. What code cannot restore (a dropped block, a lost merge tag or inline link, a
missing text field) is an error that goes back to the model for the usual single repair pass.

Blocks are added or removed only when the author asks: explicitly (`allow_block_changes`), in the instruction
("add a button", "remove the second paragraph"), or through sign_off / social_links, which add or remove the
signature exactly as they do on /generate.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from config import settings

from .categories import GENERAL, enabled_categories, get_category
from .prompts import rewrite_message, rewrite_system_prompt
from .schema import BLOCK_TYPES, SCHEMA_VERSION, SOCIAL_MAX, TEXT_FIELDS, Template
from .service import (
    LLM, EmailGenerationError, GenerateRequest, _generate_one, effective_sign_off, run_variants,
)
from .tone import Issue, iter_text, lint
from .validate import (
    MERGE_TAG_RE, _ANY_INLINE_LINK_RE, _RAW_URL_RE, Context, Result, _duplicate_id_errors,
    _sanitise, _schema_errors, content_errors, cta_count, fill_slots, parse_json,
)

logger = logging.getLogger(__name__)

# Text the model must not change even though it is copy: identity and contact facts.
LOCKED_TEXT: dict[str, frozenset[str]] = {
    "header": frozenset({"org_name"}),
    "signature": frozenset({"name", "title", "org", "contact"}),
    "footer": frozenset({"address"}),
}
# Containers whose text the model rewrites inside a fixed shape (hero button label, key_value rows, feature columns).
_NESTED_TEXT = frozenset({"button", "rows", "items"})
_SINGLE = ("greeting", "notice", "signature", "hero_banner", "features")
_NEW_ID_RE = re.compile(r"^[a-z0-9_]{1,16}$")

_NOUN = (r"(?:sections?|blocks?|paragraphs?|buttons?|ctas?|call[- ]to[- ]actions?|images?|pictures?|photos?"
         r"|banners?|hero(?: banner)?|headings?|sub-?headings?|headlines?|lists?|bullet(?:ed)?(?: points?| list)?"
         r"|bullets|notices?|callouts?|signatures?|sign[- ]?off|dividers?|spacers?|features?|columns?|tables?"
         r"|greetings?|footers?|headers?|p\.s\.?)")
_VERB = (r"(?:add(?:s|ed|ing)?|insert(?:s|ed|ing)?|includ(?:e|es|ed|ing)|append(?:s|ed|ing)?|put|"
         r"introduc(?:e|es|ed|ing)|remov(?:e|es|ed|ing)|delet(?:e|es|ed|ing)|drop(?:s|ped|ping)?|cut(?:s|ting)?|"
         r"omit(?:s|ted|ting)?|get rid of|take out|lose)")
_BLOCK_CHANGE_RE = re.compile(rf"\b{_VERB}\b[^.;:!?\n]{{0,40}}?\b{_NOUN}(?![a-z])", re.IGNORECASE)

# Instruction cues. Tone cues apply only when the request sends no tone; length cues are checked on the result.
_WARM_RE = re.compile(r"\b(friendl(?:y|ier)|warm(?:er|th)?|more personal|less formal|more casual|less stiff)\b", re.I)
_FORMAL_RE = re.compile(r"\b(more formal|formal tone|make it formal|less casual)\b", re.I)
_PROFESSIONAL_RE = re.compile(r"\b(more professional|professional tone)\b", re.I)
_SHORTER_RE = re.compile(r"\b(shorter|shorten\w*|more concise|concise|briefer|trim\w*|cut (?:it |this )?down|condense\w*"
                         r"|tighten\w*|fewer words)\b", re.I)
_LONGER_RE = re.compile(r"\b(longer|lengthen\w*|expand\w*|more detail(?:ed)?|elaborate|flesh (?:it |this )?out)\b", re.I)
SHORTER_MAX_RATIO, LONGER_MIN_RATIO = 0.85, 1.15


class EmailTemplateInputError(ValueError):
    """The template sent for rewriting is not a valid schema-v1 template (the API returns 400 with issues)."""

    def __init__(self, errors: list[Issue]):
        super().__init__("The template to rewrite is not a valid email template")
        self.errors = errors


@dataclass
class RewriteRequest:
    instruction: str
    template: dict
    category: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    language: Optional[str] = None                             # None = the template's language
    length: Optional[str] = None                               # None = about as long as now
    allowed_merge_tags: tuple[str, ...] = ()                   # added to the tags the template already uses
    links: dict[str, str] = field(default_factory=dict)
    assets: dict[str, str] = field(default_factory=dict)
    sender: dict[str, str] = field(default_factory=dict)
    org_name: Optional[str] = None
    brand_kit_id: Optional[str] = None
    variants: int = 1
    sign_off: Optional[str] = None
    social_links: dict[str, str] = field(default_factory=dict)
    theme: Optional[dict] = None
    allow_block_changes: Optional[bool] = None                 # None = only if the instruction asks


@dataclass(frozen=True)
class BlockChanges:
    """Which block types may be added / removed, and what the signature must end up as."""
    may_add: frozenset[str] = frozenset()
    may_remove: frozenset[str] = frozenset()
    signature: Optional[bool] = None                           # True must have, False must not, None as is

    @property
    def locked(self) -> bool:
        return not self.may_add and not self.may_remove

    @property
    def mode(self) -> str:
        if self.locked:
            return "none"
        return "signature" if self.may_add | self.may_remove == {"signature"} else "allowed"


@dataclass
class RewriteResult(Result):
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def asks_for_block_changes(instruction: str) -> bool:
    """True when the instruction asks to add or remove a block ("add a button", "drop the list")."""
    return bool(_BLOCK_CHANGE_RE.search(instruction or ""))


def tone_from_instruction(instruction: str) -> Optional[str]:
    """'Make it friendlier' -> professional_warm, 'more formal' -> formal, 'more professional' -> professional."""
    if _WARM_RE.search(instruction or ""):
        return "professional_warm"
    if _FORMAL_RE.search(instruction or ""):
        return "formal"
    if _PROFESSIONAL_RE.search(instruction or ""):
        return "professional"
    return None


def length_direction(instruction: str) -> Optional[str]:
    """"shorter" / "longer" when the instruction asks for it (and not both), else None."""
    shorter, longer = bool(_SHORTER_RE.search(instruction or "")), bool(_LONGER_RE.search(instruction or ""))
    return "shorter" if shorter and not longer else "longer" if longer and not shorter else None


def body_chars(template: dict) -> int:
    """Readable text in the blocks (not the subject or preheader)."""
    return sum(len(text) for bid, _f, text in iter_text(template) if bid is not None)


def _length_issue(direction: Optional[str], original: dict, template: dict) -> Optional[Issue]:
    if not direction:
        return None
    before, after = body_chars(original), body_chars(template)
    if direction == "shorter" and after > before * SHORTER_MAX_RATIO:
        return Issue("not_shorter", f"The instruction asks for a shorter email, but the text went from {before} to "
                                    f"{after} characters. Cut it to at most {int(before * SHORTER_MAX_RATIO)} "
                                    "characters by removing words and sentences, keeping the blocks.")
    if direction == "longer" and after < before * LONGER_MIN_RATIO:
        return Issue("not_longer", f"The instruction asks for a longer email, but the text went from {before} to "
                                   f"{after} characters. Expand it to at least {int(before * LONGER_MIN_RATIO) + 1} "
                                   "characters without inventing facts.")
    return None


def block_changes(req: RewriteRequest, original_types: list[str], sign_off: Optional[str]) -> BlockChanges:
    open_ = req.allow_block_changes if req.allow_block_changes is not None else asks_for_block_changes(req.instruction)
    every = frozenset(BLOCK_TYPES)
    add = remove = every if open_ else frozenset()
    has_signature = "signature" in original_types
    wanted = {"none": False, "signature": True}.get(sign_off or "")
    if wanted is False and has_signature:
        remove = remove | {"signature"}
    if wanted is True and not has_signature:
        add = add | {"signature"}
    if req.allow_block_changes is False and (add or remove):
        raise ValueError("sign_off / social_links would add or remove the signature block, but allow_block_changes "
                         "is false. Drop those options or allow block changes.")
    return BlockChanges(add, remove, wanted)


# ---------------------------------------------------------------------------
# Reading the original
# ---------------------------------------------------------------------------
def load_original(template: Any) -> dict:
    """The template to rewrite, schema-validated and normalised; EmailTemplateInputError when it is not valid."""
    if not isinstance(template, dict):
        raise EmailTemplateInputError([Issue("schema", "template must be a JSON object")])
    try:
        original = Template.model_validate(template).model_dump(mode="json", exclude_none=True)
    except ValidationError as e:
        raise EmailTemplateInputError(_schema_errors(e)) from None
    dup = _duplicate_id_errors(original["blocks"])
    if dup:
        raise EmailTemplateInputError(dup)
    return original


def _texts(template: dict) -> list[str]:
    return [text for _bid, _fld, text in iter_text(template)]


def _refs(texts: list[str]) -> tuple[set[str], Counter]:
    """(merge tags, inline link targets) in a list of texts."""
    tags: set[str] = set()
    links: Counter = Counter()
    for t in texts:
        tags.update(MERGE_TAG_RE.findall(t))
        links.update(_ANY_INLINE_LINK_RE.findall(t))
    return tags, links


def _template_refs(t: dict) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Merge tags, link keys, asset keys, non-key inline link targets and raw URLs the template already uses."""
    texts = _texts(t)
    tags, inline = _refs(texts)
    link_keys = {x.removeprefix("link:") for x in inline if x.startswith("link:")}
    other_targets = {x for x in inline if not x.startswith("link:")}
    urls = {m.group(0) for text in texts for m in _RAW_URL_RE.finditer(text)}
    assets: set[str] = set()
    for b in t["blocks"]:
        btn = b if b["type"] == "button" else (b.get("button") if b["type"] == "hero_banner" else None)
        if btn and btn["link"].startswith("link:"):
            link_keys.add(btn["link"].removeprefix("link:"))
        for k in ("logo_asset", "asset"):
            if isinstance(b.get(k), str) and b[k].startswith("asset:"):
                assets.add(b[k].removeprefix("asset:"))
    return tags, link_keys, assets, other_targets, urls


def _for_model(original: dict) -> dict:
    """What the model sees: subject options, preheader and blocks. Code-owned fields stay out."""
    blocks = []
    for b in original["blocks"]:
        b = {k: v for k, v in b.items() if not (b["type"] == "footer" and k == "social")}
        blocks.append(b)
    meta = original["meta"]
    return {"meta": {"subject_options": meta["subject_options"], "preheader": meta["preheader"]}, "blocks": blocks}


# ---------------------------------------------------------------------------
# Merging the model's reply onto the original
# ---------------------------------------------------------------------------
def _editable(key: str) -> bool:
    return key in TEXT_FIELDS or key in _NESTED_TEXT


def _merge_value(orig: Any, new: Any, key: str, bid: str, errors: list[Issue]) -> Any:
    """The original value with only its text replaced by the model's. Shape follows the original."""
    if isinstance(orig, dict):
        if not isinstance(new, dict):
            errors.append(Issue("missing_text", f"Keep the '{key}' field and rewrite its text.", bid, key))
            return copy.deepcopy(orig)
        return {k: (_merge_value(v, new.get(k), k, bid, errors) if _editable(k) else copy.deepcopy(v))
                for k, v in orig.items()}
    if isinstance(orig, list):
        if all(isinstance(x, str) for x in orig):           # list items: the count may change
            if isinstance(new, list) and new:
                return new
            errors.append(Issue("missing_text", f"Keep the '{key}' field and rewrite its text.", bid, key))
            return list(orig)
        if not isinstance(new, list) or len(new) != len(orig):
            errors.append(Issue("shape_changed", f"Keep exactly {len(orig)} entries in '{key}'.", bid, key))
            return copy.deepcopy(orig)
        return [_merge_value(o, n, key, bid, errors) for o, n in zip(orig, new)]
    if isinstance(orig, str):
        if isinstance(new, str) and new.strip():
            return new
        errors.append(Issue("missing_text", f"Keep the '{key}' field and rewrite its text.", bid, key))
        return orig
    return orig


def merge_block(orig: dict, new: dict, errors: list[Issue]) -> dict:
    """The original block with the model's copy. Non-text and locked fields come from the original."""
    if orig["type"] == "image":
        return copy.deepcopy(orig)
    locked = LOCKED_TEXT.get(orig["type"], frozenset())
    out = {}
    for k, v in orig.items():
        if k in ("id", "type") or k in locked or not _editable(k):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = _merge_value(v, new.get(k), k, orig["id"], errors)
    return out


def _ref_lock_errors(orig: dict, merged: dict, bid: Optional[str]) -> list[Issue]:
    """
    Merge tags and inline links the original had must still be there. orig / merged are template-shaped
    ({"blocks": [one block]} or {"meta": ..., "blocks": []}).
    """
    errs = []
    o_tags, o_links = _refs(_texts(orig))
    n_tags, n_links = _refs(_texts(merged))
    for tag in sorted(o_tags - n_tags):
        errs.append(Issue("merge_tag_removed", f"Keep the merge tag {{{{{tag}}}}} in this "
                                               f"{'block' if bid else 'subject/preheader'}.", bid))
    if o_links != n_links:
        want = ", ".join(sorted(o_links.elements())) or "none"
        errs.append(Issue("link_changed", f"Keep exactly the inline links this block had ({want}); you may reword "
                                          "their text but not add, drop or change a link.", bid))
    return errs


def _merge_blocks(model_blocks: Any, original: list[dict], changes: BlockChanges
                  ) -> tuple[list[dict], list[Issue], list[str], list[str], list[dict]]:
    """(blocks, errors, added ids, removed ids, added image blocks)."""
    if not isinstance(model_blocks, list):
        return [copy.deepcopy(b) for b in original], [Issue("schema", "blocks: return the list of blocks")], [], [], []
    by_id = {b["id"]: b for b in original}
    order = {b["id"]: i for i, b in enumerate(original)}
    out: list[dict] = []
    errors: list[Issue] = []
    seen: set[str] = set()
    added: list[str] = []
    new_images: list[dict] = []
    used_ids = set(by_id)

    for i, mb in enumerate(model_blocks):
        if not isinstance(mb, dict):
            errors.append(Issue("schema", f"blocks[{i}]: each block must be a JSON object"))
            continue
        bid, btype = mb.get("id"), mb.get("type")
        if isinstance(bid, str) and bid in by_id and bid not in seen:
            orig = by_id[bid]
            if btype != orig["type"]:
                errors.append(Issue("block_type_changed", f"Block {bid} must stay a '{orig['type']}' block.", bid))
                out.append(copy.deepcopy(orig))
            else:
                block_errors: list[Issue] = []
                merged = merge_block(orig, mb, block_errors)
                errors += block_errors or _ref_lock_errors({"blocks": [orig]}, {"blocks": [merged]}, bid)
                out.append(merged)
            seen.add(bid)
            continue
        if btype not in changes.may_add:
            what = f"'{btype}' block" if isinstance(btype, str) else "block"
            errors.append(Issue("block_added", f"Do not add a {what}: keep exactly the blocks of the current email.",
                                bid if isinstance(bid, str) else None))
            continue
        block = dict(mb)
        if not (isinstance(bid, str) and _NEW_ID_RE.match(bid)) or bid in used_ids:
            n = 1
            while f"n{n}" in used_ids:
                n += 1
            block["id"] = f"n{n}"
        used_ids.add(block["id"])
        added.append(block["id"])
        if btype == "image":
            new_images.append(block)
        out.append(block)

    removed = [b["id"] for b in original if b["id"] not in seen]
    for rid in removed:
        if by_id[rid]["type"] not in changes.may_remove:
            errors.append(Issue("block_removed", f"Keep block {rid} ({by_id[rid]['type']}) and rewrite its text.", rid))

    kept = [b["id"] for b in out if b["id"] in order]
    if kept != sorted(kept, key=order.__getitem__):
        if not added:   # same blocks, shuffled: put them back in the original order
            out.sort(key=lambda b: order[b["id"]])
        else:
            errors.append(Issue("block_order", "Keep the original blocks in their original order."))
    return out, errors, added, removed, new_images


def _changed_structure_errors(blocks: list[dict], original: list[dict], changes: BlockChanges) -> list[Issue]:
    """Only when blocks were added or removed: positions, singletons, one call to action, the signature rule."""
    types = [b["type"] for b in blocks]
    o_types = [b["type"] for b in original]
    errs: list[Issue] = []
    if "header" in types and (types.count("header") != 1 or types[0] != "header"):
        errs.append(Issue("header_position", "Exactly one header block, as the first block."))
    if "footer" in types and (types.count("footer") != 1 or types[-1] != "footer"):
        errs.append(Issue("footer_position", "Exactly one footer block, as the last block."))
    for single in _SINGLE:
        if types.count(single) > max(1, o_types.count(single)):
            errs.append(Issue("too_many", f"At most one '{single}' block."))
    if "hero_banner" in types and "hero_banner" not in o_types and \
            types.index("hero_banner") != (1 if types[0] == "header" else 0):
        errs.append(Issue("hero_banner_position", "The hero_banner block comes directly after the header."))
    if cta_count(blocks) > max(1, cta_count(original)):
        errs.append(Issue("too_many_buttons", "Keep one call to action (a button block or the hero_banner button). "
                                              "Put any other links inline in a paragraph."))
    slots = [b.get("slot") for b in blocks if b["type"] == "image"]
    for slot in sorted({s for s in slots if slots.count(s) > 1}, key=str):
        errs.append(Issue("image_slot", f"Only one '{slot}' image per email."))
    if changes.signature is True and "signature" not in types:
        errs.append(Issue("missing_block", "Add a signature block before the footer."))
    if changes.signature is False and "signature" in types:
        errs.append(Issue("signature_not_wanted", "Remove the signature block; the email ends with the footer."))
    return errs


def _locked_fields(template: dict) -> set[tuple[str, str]]:
    out = set()
    for b in template["blocks"]:
        if b["type"] == "image":
            out.add((b["id"], "alt"))
        for f in LOCKED_TEXT.get(b["type"], ()):
            out.add((b["id"], f))
    return out


def validate_rewrite(raw: Any, original: dict, ctx: Context, changes: BlockChanges,
                     social: Optional[tuple[str, ...]] = None, length: Optional[str] = None,
                     length_is_error: bool = True) -> RewriteResult:
    """
    Validate a rewrite reply against the original. social: platforms to show in the footer (None = keep the
    original's icons). length: "shorter" / "longer" asked by the instruction; a miss is an error (sent back for
    repair) when length_is_error, otherwise a warning.
    """
    res = RewriteResult()
    try:
        data = parse_json(raw) if isinstance(raw, str) else raw
    except (ValueError, json.JSONDecodeError) as e:
        res.errors.append(Issue("json", f"Reply is not a valid JSON object: {e}"))
        return res
    if not isinstance(data, dict):
        res.errors.append(Issue("json", "Reply must be a JSON object."))
        return res
    changed: list[str] = []
    data = _sanitise(data, None, changed)
    if changed:
        res.warnings.append(Issue("sanitised", "Markup was removed from: " + ", ".join(sorted(set(changed)))))

    blocks, errors, added, removed, new_images = _merge_blocks(data.get("blocks"), original["blocks"], changes)
    model_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    o_meta = original["meta"]
    meta = {
        "subject_options": model_meta.get("subject_options"),
        "preheader": model_meta.get("preheader"),
        "category": ctx.category.id, "routing": ctx.routing, "tone": ctx.tone, "language": ctx.language,
    }
    if social is not None:
        for b in blocks:
            if b.get("type") == "footer":
                b["social"] = [{"platform": p, "link": f"link:social_{p}"} for p in social]
    candidate = {"schema_version": SCHEMA_VERSION, "meta": meta,
                 "theme": {**(ctx.theme or {}), "brand_kit_id": ctx.brand_kit_id}, "blocks": blocks}
    try:
        template = Template.model_validate(candidate).model_dump(mode="json", exclude_none=True)
    except ValidationError as e:
        res.errors.extend(errors + _schema_errors(e))
        return res

    opts = template["meta"]["subject_options"]
    o_opts = o_meta["subject_options"]
    index = o_opts.index(o_meta["subject"]) if o_meta.get("subject") in o_opts else 0
    template["meta"]["subject"] = opts[index]
    errors += _ref_lock_errors({"meta": o_meta, "blocks": []}, {"meta": template["meta"], "blocks": []}, None)

    errors += _duplicate_id_errors(template["blocks"])
    if added or removed:
        errors += _changed_structure_errors(template["blocks"], original["blocks"], changes)
    added_images = [b for b in template["blocks"] if b["id"] in {i["id"] for i in new_images}]
    errors += content_errors(template, ctx, image_blocks=added_images)

    tone = lint(template, ctx.tone, ctx.source_texts)
    locked = _locked_fields(template)
    errors += [e for e in tone.errors if (e.block_id, e.field) not in locked]
    length_issue = _length_issue(length, original, template)
    if length_issue and length_is_error:
        errors.append(length_issue)
    res.errors.extend(errors)
    res.warnings.extend(tone.warnings)
    if length_issue and not length_is_error:
        res.warnings.append(length_issue)
    res.fill_slots = fill_slots(template)
    res.template = template
    res.added, res.removed = added, removed
    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _resolve_category(req: RewriteRequest, original: dict):
    if req.category:
        return get_category(req.category), "selected"
    current = original["meta"]["category"]
    if current in enabled_categories():
        return get_category(current), original["meta"].get("routing", "selected")
    return get_category(GENERAL), "general"


def rewrite(req: RewriteRequest, llm: LLM, request_id: Optional[str] = None) -> dict:
    """Rewritten templates. Raises ValueError / EmailTemplateInputError (bad input) or EmailGenerationError."""
    instruction = (req.instruction or "").strip()
    if len(instruction) < 3:
        raise ValueError("instruction must contain at least 3 characters of text")
    original = load_original(req.template)
    o_meta = original["meta"]
    sign_off = effective_sign_off(GenerateRequest(brief=instruction, sign_off=req.sign_off,
                                                  social_links=req.social_links))
    changes = block_changes(req, [b["type"] for b in original["blocks"]], sign_off)
    category, routing = _resolve_category(req, original)
    tone = req.tone or tone_from_instruction(instruction) or o_meta.get("tone") or category.default_tone
    length = length_direction(instruction)
    language = req.language or o_meta.get("language") or "en-GB"

    tags, link_keys, asset_keys, other_targets, urls = _template_refs(original)
    o_theme = {k: v for k, v in (original.get("theme") or {}).items() if k != "brand_kit_id"}
    texts = _texts(original)
    ctx = Context(
        category=category, routing=routing, tone=tone, language=language,
        allowed_merge_tags=frozenset(req.allowed_merge_tags) | tags,
        link_keys=frozenset(req.links) | link_keys,
        asset_keys=frozenset(req.assets) | asset_keys,
        brand_kit_id=req.brand_kit_id or (original.get("theme") or {}).get("brand_kit_id"),
        sign_off=sign_off, theme=req.theme if req.theme is not None else o_theme,
        source_texts=(instruction, req.audience or "", req.org_name or "", *req.sender.values(),
                      *req.assets.values(), *texts),
        known_link_targets=frozenset(other_targets), known_urls=frozenset(urls),
    )
    social = tuple(req.social_links)[:SOCIAL_MAX] if req.social_links else None

    model_request = {
        "email_type": category.label,
        "audience": req.audience,
        "language": language,
        "language_note": (f"The current email is in {o_meta.get('language')}; write the rewrite in {language}."
                          if language != o_meta.get("language") else None),
        "length": req.length,
        "length_target": ({"shorter": f"clearly shorter: at most {int(body_chars(original) * SHORTER_MAX_RATIO)} "
                                      f"characters of block text (now {body_chars(original)})",
                           "longer": f"clearly longer: at least {int(body_chars(original) * LONGER_MIN_RATIO) + 1} "
                                     f"characters of block text (now {body_chars(original)})"}.get(length or "")),
        "allowed_merge_tags": sorted(ctx.allowed_merge_tags),
        "links": sorted(ctx.link_keys),
        "assets": {**{k: "" for k in asset_keys}, **req.assets},
        "sender": req.sender or None,
        "org_name": req.org_name,
        "block_changes": {"none": "none: keep every block",
                          "allowed": "only those the instruction explicitly asks for",
                          "signature": None}[changes.mode],
        "signature": ({True: "add a signature block before the footer",
                       False: "remove the signature block"}.get(changes.signature)
                      if "signature" in (changes.may_add | changes.may_remove) else None),
    }
    model_request = {k: v for k, v in model_request.items() if v not in (None, {}, "", [])}
    system = rewrite_system_prompt(tone, not changes.locked)
    first_user = rewrite_message(instruction, model_request, _for_model(original))
    n = max(1, min(req.variants, settings.email_ai_max_variants))

    def one(_i):
        attempts = []

        def check(reply: str) -> RewriteResult:
            # A length miss goes back for the one repair; after that it is a warning, not a failed request.
            attempts.append(1)
            return validate_rewrite(reply, original, ctx, changes, social, length, length_is_error=len(attempts) == 1)

        return _generate_one(llm, system, first_user, ctx, request_id, check=check)

    results = run_variants(n, one)
    good = [r for r in results if r.ok]
    logger.info(f"[{request_id}] email rewrite: category={category.id} tone={tone} blocks={changes.mode} "
                f"ok={len(good)}/{n}")
    if not good:
        raise EmailGenerationError(results[0].errors)
    return {
        "resolved_category": category.id,
        "routing": routing,
        "tone": tone,
        "block_changes": changes.mode,
        "templates": [
            {"template": r.template, "warnings": [w.as_dict() for w in r.warnings], "fill_slots": r.fill_slots,
             "added_blocks": r.added, "removed_blocks": r.removed}
            for r in good
        ],
    }
