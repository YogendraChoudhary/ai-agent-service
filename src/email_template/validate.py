"""
The gate between the model and the user: parse -> sanitise -> schema -> rules -> tone lint.

Nothing the model writes reaches the editor without passing here. Fields that code owns (category, routing,
tone, language, schema version) are overwritten from the request, never trusted from the model. Image blocks
are moved to their fixed slot positions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from .categories import Category
from .schema import SCHEMA_VERSION, TEXT_FIELDS, Template
from .tone import Issue, iter_text, lint

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?\s*```\s*$")
_TAG_RE = re.compile(r"<[^>]{0,500}>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t]+")
MERGE_TAG_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\((link:[a-z0-9_]+)\)")
_ANY_INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]*)\)")
FILL_SLOT_RE = re.compile(r"\[\[fill:\s*([^\]]+?)\s*\]\]")
_RAW_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


@dataclass
class Context:
    """What the request allows. Everything the model may reference must be listed here."""
    category: Category
    routing: str
    tone: str
    language: str
    allowed_merge_tags: frozenset[str] = frozenset()
    link_keys: frozenset[str] = frozenset()
    asset_keys: frozenset[str] = frozenset()
    image_slots: tuple[str, ...] = ()          # slots this request may fill (brief asked for images)
    brand_kit_id: Optional[str] = None
    sign_off: Optional[str] = None             # "signature" | "none" | None (category default)
    social: tuple[str, ...] = ()               # platforms with a link in the request, in display order
    theme: Optional[dict] = None               # user-chosen look (accent, hero/footer style)

    @property
    def signature_rule(self) -> Optional[str]:
        """"required", "forbidden" or None (optional). A footer with social icons replaces the signature."""
        if self.sign_off == "none" or self.social:
            return "forbidden"
        if self.sign_off == "signature" or self.category.needs_signature:
            return "required"
        return None
    source_texts: tuple[str, ...] = ()         # user-supplied text, for the invented-number check


@dataclass
class Result:
    template: Optional[dict] = None
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    fill_slots: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.template is not None and not self.errors


def parse_json(raw: str) -> Any:
    """The JSON object in a model reply, tolerating code fences and stray prose around it."""
    text = _FENCE_RE.sub("", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the reply.")
    return json.loads(text[start:end + 1])


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _sanitise(node: Any, key: Optional[str], changed: list[str]) -> Any:
    """Strip markup and control characters from copy fields, recursively. Keys and references are untouched."""
    if isinstance(node, dict):
        return {k: _sanitise(v, k, changed) for k, v in node.items()}
    if isinstance(node, list):
        return [_sanitise(v, key, changed) for v in node]
    if isinstance(node, str) and key in (TEXT_FIELDS | {"subject_options", "subject", "preheader"}):
        cleaned = _clean(node)
        if cleaned != node.strip():
            changed.append(key)
        return cleaned
    return node


def _schema_errors(err: ValidationError) -> list[Issue]:
    issues = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        issues.append(Issue("schema", f"{loc}: {e['msg']}", field=loc))
    return issues


def _normalise_image_positions(blocks: list[dict]) -> list[dict]:
    """
    hero goes right after the hero banner if there is one, else right after the header; inline goes just
    before the button (or the signature/footer).
    """
    hero = [b for b in blocks if b.get("type") == "image" and b.get("slot") == "hero"]
    inline = [b for b in blocks if b.get("type") == "image" and b.get("slot") == "inline"]
    rest = [b for b in blocks if b.get("type") != "image"]
    if hero:
        banner = next((i for i, b in enumerate(rest) if b.get("type") == "hero_banner"), None)
        at = banner + 1 if banner is not None else (1 if rest and rest[0].get("type") == "header" else 0)
        rest[at:at] = hero[:1]
    if inline:
        anchor = next((i for i, b in enumerate(rest) if b.get("type") == "button"), None)
        if anchor is None:
            anchor = next((i for i, b in enumerate(rest) if b.get("type") in ("signature", "footer")), len(rest))
        rest[anchor:anchor] = inline[:1]
    # Extra images in the same slot are kept at the end so the rule check reports them rather than losing them.
    return rest + hero[1:] + inline[1:]


def _rule_errors(t: dict, ctx: Context) -> list[Issue]:
    cat = ctx.category
    blocks = t["blocks"]
    types = [b["type"] for b in blocks]
    errs: list[Issue] = []

    ids = [b["id"] for b in blocks]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errs.append(Issue("duplicate_id", f"Block id '{dup}' is used more than once.", dup))

    for b in blocks:
        if b["type"] not in cat.allowed:
            errs.append(Issue("block_not_allowed", f"'{b['type']}' blocks are not used in {cat.id} emails.", b["id"]))
    required = cat.required - ({"signature"} if ctx.signature_rule == "forbidden" else set())
    for req in sorted(required - set(types)):
        errs.append(Issue("missing_block", f"A '{req}' block is required for {cat.id} emails."))

    if types.count("header") != 1 or types[0] != "header":
        errs.append(Issue("header_position", "Exactly one header block, as the first block."))
    if types.count("footer") != 1 or types[-1] != "footer":
        errs.append(Issue("footer_position", "Exactly one footer block, as the last block."))
    for single in ("greeting", "notice", "signature", "hero_banner", "features"):
        if types.count(single) > 1:
            errs.append(Issue("too_many", f"At most one '{single}' block."))
    if "hero_banner" in types and types.index("hero_banner") != 1:
        errs.append(Issue("hero_banner_position", "The hero_banner block comes directly after the header."))
    ctas = types.count("button") + sum(1 for b in blocks if b["type"] == "hero_banner" and b.get("button"))
    if ctas > 1:
        errs.append(Issue("too_many_buttons", "Exactly one call to action in the whole email (a button block or "
                                              "the hero_banner button). Put any other links inline in a paragraph."))
    if cat.needs_button and ctas == 0:
        errs.append(Issue("missing_block", f"{cat.id} emails need one button for the main action."))
    if ctx.signature_rule == "required" and "signature" not in types:
        errs.append(Issue("missing_block", "End the email with a signature block before the footer."))
    if ctx.signature_rule == "forbidden" and "signature" in types:
        errs.append(Issue("signature_not_wanted", "Do not add a signature block; the email ends with the footer"
                                                  + (" and its social links." if ctx.social else ".")))

    footer = next((b for b in blocks if b["type"] == "footer"), None)
    if cat.bulk and footer is not None and not footer.get("unsubscribe"):
        errs.append(Issue("unsubscribe", "Bulk emails must set unsubscribe: true in the footer.", footer["id"]))

    seen_slots: list[str] = []
    for b in blocks:
        if b["type"] != "image":
            continue
        slot = b["slot"]
        if slot not in cat.image_slots:
            errs.append(Issue("image_slot", f"{cat.id} emails have no '{slot}' image slot.", b["id"]))
        elif slot not in ctx.image_slots:
            errs.append(Issue("image_not_requested",
                              "Add an image only when the brief asks for one.", b["id"]))
        if slot in seen_slots:
            errs.append(Issue("image_slot", f"Only one '{slot}' image per email.", b["id"]))
        seen_slots.append(slot)
        if b.get("source") not in (None, "asset"):
            errs.append(Issue("image_source", "Leave source null; the file is attached after generation.", b["id"]))
        if b.get("source") == "asset" and b.get("asset", "").removeprefix("asset:") not in ctx.asset_keys:
            errs.append(Issue("unknown_asset", "Use only the asset keys listed in the request.", b["id"]))
        if b.get("source") is None and not b.get("generation_prompt"):
            errs.append(Issue("image_prompt", "An empty image slot needs a generation_prompt.", b["id"]))

    header = blocks[0] if blocks and blocks[0]["type"] == "header" else None
    if header and header.get("logo_asset") and header["logo_asset"].removeprefix("asset:") not in ctx.asset_keys:
        errs.append(Issue("unknown_asset", "Use only the asset keys listed in the request.", header["id"], "logo_asset"))

    for b in blocks:
        btn = b if b["type"] == "button" else (b.get("button") if b["type"] == "hero_banner" else None)
        if btn and not FILL_SLOT_RE.fullmatch(btn["link"]) and btn["link"].removeprefix("link:") not in ctx.link_keys:
            errs.append(Issue("unknown_link", f"'{btn['link']}' is not one of the links supplied.", b["id"], "link"))

    for bid, fld, text in iter_text(t):
        for tag in MERGE_TAG_RE.findall(text):
            if tag not in ctx.allowed_merge_tags:
                errs.append(Issue("unknown_merge_tag", f"'{{{{{tag}}}}}' is not an allowed merge tag.", bid, fld))
        for target in _ANY_INLINE_LINK_RE.findall(text):
            if not target.startswith("link:") or target.removeprefix("link:") not in ctx.link_keys:
                errs.append(Issue("unknown_link", f"Inline link '{target}' is not one of the links supplied.", bid, fld))
        if _RAW_URL_RE.search(INLINE_LINK_RE.sub("", text)):
            errs.append(Issue("raw_url", "Do not write URLs; use the link keys supplied.", bid, fld))
    return errs


def fill_slots(t: dict) -> list[dict]:
    out = []
    for bid, fld, text in iter_text(t):
        for need in FILL_SLOT_RE.findall(text):
            out.append({"block_id": bid, "field": fld, "needed": need})
    for b in t["blocks"]:
        btn = b if b["type"] == "button" else (b.get("button") if b["type"] == "hero_banner" else None)
        if btn:
            for need in FILL_SLOT_RE.findall(btn["link"]):
                out.append({"block_id": b["id"], "field": "button.link" if b["type"] == "hero_banner" else "link",
                            "needed": need})
    return out


def validate(raw: Any, ctx: Context) -> Result:
    """Validate a model reply (a JSON string or an already-parsed dict) against the request context."""
    res = Result()
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

    # Code-owned fields.
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta.update(category=ctx.category.id, routing=ctx.routing, tone=ctx.tone, language=ctx.language)
    data["meta"] = meta
    data["schema_version"] = SCHEMA_VERSION
    data["theme"] = {**(ctx.theme or {}), "brand_kit_id": ctx.brand_kit_id}
    # Footer social links come from the request, never from the model.
    blocks = data.get("blocks")
    for b in blocks if isinstance(blocks, list) else []:
        if isinstance(b, dict) and b.get("type") == "footer":
            b["social"] = [{"platform": p, "link": f"link:social_{p}"} for p in ctx.social]

    try:
        template = Template.model_validate(data).model_dump(mode="json", exclude_none=True)
    except ValidationError as e:
        res.errors.extend(_schema_errors(e))
        return res

    opts = template["meta"]["subject_options"]
    if template["meta"].get("subject") not in opts:
        template["meta"]["subject"] = opts[0]
    template["blocks"] = _normalise_image_positions(template["blocks"])

    res.errors.extend(_rule_errors(template, ctx))
    tone = lint(template, ctx.tone, ctx.source_texts)
    res.errors.extend(tone.errors)
    res.warnings.extend(tone.warnings)
    res.fill_slots = fill_slots(template)
    res.template = template
    return res


def format_errors(errors: Iterable[Issue]) -> str:
    """Numbered list for the repair prompt."""
    lines = []
    for i, e in enumerate(errors, 1):
        where = " ".join(x for x in (f"block {e.block_id}" if e.block_id else "", e.field or "") if x)
        lines.append(f"{i}. [{e.kind}]{' (' + where + ')' if where else ''} {e.message}")
    return "\n".join(lines)
