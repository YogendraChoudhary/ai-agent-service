"""
Review-only HTML preview of email template JSON.

NOT the production renderer (that lives in Eddyter: JSON -> MJML -> HTML). This exists so people can look at
what /agent/email/generate produced in a browser. It follows the same layout rules: 600px table layout,
inline CSS only, fixed image slots, one button.

In the preview:
  - {{merge_tags}} are shown with sample values, underlined in blue
  - [[fill: ...]] slots are highlighted in yellow (export is blocked until they are filled)
  - empty image slots are grey boxes at the exact slot ratio, labelled with the file size they need

Usage:
  python scripts/render_email_preview.py <generate-response-or-template.json> [more.json ...] --out DIR
  A file may also hold {"<name>": <generate response>, ...}. An index.html links every preview.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.email_template.images import SLOTS  # noqa: E402

THEME = {
    "brand": "#1F4E79", "text": "#1F2933", "muted": "#616E7C", "bg": "#F2F4F7", "card": "#FFFFFF",
    "border": "#E4E7EB", "notice_info": "#EAF2FB", "notice_important": "#FDF1E7", "font": "Arial, Helvetica, sans-serif",
}
SAMPLE_MERGE = {
    "first_name": "Alex", "company_name": "Acme Corporation", "sender_name": "Jordan Lee",
    "sender_title": "Operations Manager", "company_address": "12 Example Street, Bengaluru 560001",
}
SECTION_STYLES = {  # (background, text, muted) per theme style; "brand" uses the accent colour
    "light": ("#FFFFFF", "#1F2933", "#616E7C"),
    "dark": ("#1B1F2A", "#FFFFFF", "#9AA5B1"),
}
SOCIAL_LABEL = {"linkedin": "in", "x": "X", "facebook": "f", "instagram": "IG", "youtube": "YT", "email": "@"}
_MERGE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_FILL_RE = re.compile(r"\[\[fill:\s*([^\]]+?)\s*\]\]")


def _inline(text: str, links: dict) -> str:
    """Escape, then apply the only markup the schema allows: **bold**, *italic*, [text](link:key)."""
    out = html.escape(text or "")
    out = _FILL_RE.sub(lambda m: (
        f'<span style="background:#FFF3B0;color:#7A5A00;padding:0 3px;border-radius:2px;">'
        f'[Fill: {m.group(1)}]</span>'), out)
    out = _MERGE_RE.sub(lambda m: (
        f'<span title="merge tag {{{{{m.group(1)}}}}}" style="border-bottom:1px dotted #2B6CB0;">'
        f'{html.escape(SAMPLE_MERGE.get(m.group(1), m.group(1)))}</span>'), out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(link:([a-z0-9_]+)\)", lambda m: (
        f'<a href="{html.escape(links.get(m.group(2), "#"))}" style="color:{THEME["brand"]};">{m.group(1)}</a>'), out)
    return out.replace("\n", "<br>")


def _row(inner: str, pad: str = "0 20px 16px 20px") -> str:
    return f'<tr><td style="padding:{pad};">{inner}</td></tr>'


def _placeholder_svg(slot_id: str, alt: str) -> str:
    s = SLOTS[slot_id]
    label = f"{slot_id} slot · {s.width}×{s.height} file · shown at {s.display_width}×{s.display_height}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.width}" height="{s.height}" viewBox="0 0 {s.width} {s.height}">'
           f'<rect width="100%" height="100%" fill="#D9DEE5"/>'
           f'<text x="50%" y="46%" font-family="Arial" font-size="34" fill="#52606D" text-anchor="middle">{html.escape(label)}</text>'
           f'<text x="50%" y="56%" font-family="Arial" font-size="26" fill="#7B8794" text-anchor="middle">{html.escape(alt[:90])}</text>'
           '</svg>')
    from urllib.parse import quote
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _section(theme: dict, name: str) -> tuple[str, str, str]:
    sec = (theme or {}).get(name) or {}
    style = sec.get("style", "light")
    if style == "custom":
        return sec.get("background", "#FFFFFF"), sec.get("text", "#1F2933"), sec.get("text", "#616E7C")
    if style == "brand":
        return (theme or {}).get("accent_color") or THEME["brand"], "#FFFFFF", "#E4E7EB"
    return SECTION_STYLES.get(style, SECTION_STYLES["light"])


def _button(label: str, href: str, accent: str, text_colour: str = "#FFFFFF") -> str:
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center"><tr>'
            f'<td style="background:{accent};border-radius:22px;">'
            f'<a href="{html.escape(href)}" style="display:inline-block;padding:12px 26px;font-size:15px;font-weight:bold;'
            f'color:{text_colour};text-decoration:none;">{html.escape(label)}</a></td></tr></table>')


def _block(b: dict, links: dict, assets: dict, theme: dict | None = None) -> str:
    t, c = b["type"], THEME
    accent = (theme or {}).get("accent_color") or c["brand"]
    if t == "header":
        logo = ""
        if b.get("logo_asset"):
            key = b["logo_asset"].removeprefix("asset:")
            logo = (f'<div style="display:inline-block;width:120px;height:40px;line-height:40px;text-align:center;'
                    f'background:#D9DEE5;color:#52606D;font-size:11px;margin-right:10px;vertical-align:middle;">'
                    f'logo: {html.escape(assets.get(key, key))}</div>')
        name = f'<span style="font-size:18px;font-weight:bold;color:{c["brand"]};vertical-align:middle;">' \
               f'{_inline(b.get("org_name", ""), links)}</span>'
        return f'<tr><td style="padding:20px;border-bottom:3px solid {c["brand"]};">{logo}{name}</td></tr>'
    if t == "hero_banner":
        bg, fg, muted = _section(theme, "hero")
        eyebrow = (f'<div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:bold;'
                   f'color:{accent};margin-bottom:10px;">{_inline(b["eyebrow"], links)}</div>') if b.get("eyebrow") else ""
        summary = (f'<p style="margin:8px 0 0;font-size:16px;line-height:1.5;color:{muted};">{_inline(b["summary"], links)}</p>'
                   if b.get("summary") else "")
        btn = ""
        if b.get("button"):
            href = links.get(b["button"]["link"].removeprefix("link:"), "#")
            dark_text = "#0B1220" if bg != "#FFFFFF" else "#FFFFFF"
            btn = f'<div style="margin-top:24px;">{_button(b["button"]["label"], href, accent, dark_text)}</div>'
        return (f'<tr><td align="center" style="background:{bg};padding:40px 28px;text-align:center;">{eyebrow}'
                f'<h1 style="margin:0;font-size:32px;line-height:1.2;font-weight:normal;color:{fg};">{_inline(b["headline"], links)}</h1>'
                f'{summary}{btn}</td></tr>')
    if t == "features":
        n = len(b["items"])
        cells = "".join(
            f'<td width="{100 // n}%" valign="top" style="border:1px solid {c["border"]};padding:12px 8px;text-align:center;">'
            f'<div style="font-size:16px;color:{c["text"]};">{_inline(i["title"], links)}</div>'
            f'<div style="font-size:13px;line-height:1.4;color:{c["muted"]};margin-top:4px;">{_inline(i["text"], links)}</div></td>'
            for i in b["items"])
        return _row(f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                    f'style="border-collapse:collapse;"><tr>{cells}</tr></table>', "8px 20px 20px 20px")
    if t == "image":
        s = SLOTS[b["slot"]]
        pad = "0" if b["slot"] == "hero" else "0 20px 16px 20px"
        src = _placeholder_svg(b["slot"], b.get("alt", "")) if not b.get("asset") else "#"
        tip = html.escape(b.get("generation_prompt", ""))
        return (f'<tr><td style="padding:{pad};" title="generation_prompt: {tip}">'
                f'<img src="{src}" width="{s.display_width}" height="{s.display_height}" alt="{html.escape(b.get("alt", ""))}" '
                f'style="display:block;width:100%;max-width:{s.display_width}px;height:auto;border:0;"></td></tr>')
    if t == "heading":
        size = 24 if b.get("level", 1) == 1 else 19
        return _row(f'<h{b.get("level", 1)} style="margin:0;font-size:{size}px;line-height:1.3;color:{c["text"]};">'
                    f'{_inline(b["text"], links)}</h{b.get("level", 1)}>', "20px 20px 12px 20px")
    if t in ("greeting", "paragraph"):
        return _row(f'<p style="margin:0;font-size:15px;line-height:1.6;color:{c["text"]};">{_inline(b["text"], links)}</p>')
    if t == "list":
        tag = "ol" if b.get("style") == "numbered" else "ul"
        items = "".join(f'<li style="margin:0 0 6px 0;">{_inline(i, links)}</li>' for i in b["items"])
        return _row(f'<{tag} style="margin:0;padding-left:22px;font-size:15px;line-height:1.6;color:{c["text"]};">'
                    f'{items}</{tag}>')
    if t == "key_value":
        rows = "".join(
            f'<tr><td style="padding:8px 12px;font-size:14px;color:{c["muted"]};width:38%;border-bottom:1px solid {c["border"]};">'
            f'{_inline(r["label"], links)}</td><td style="padding:8px 12px;font-size:14px;color:{c["text"]};font-weight:bold;'
            f'border-bottom:1px solid {c["border"]};">{_inline(r["value"], links)}</td></tr>' for r in b["rows"])
        return _row(f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                    f'style="border:1px solid {c["border"]};border-bottom:0;">{rows}</table>')
    if t == "notice":
        bg = c["notice_important"] if b.get("variant") == "important" else c["notice_info"]
        bar = "#C05621" if b.get("variant") == "important" else c["brand"]
        return _row(f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
                    f'<td style="background:{bg};border-left:4px solid {bar};padding:12px 14px;font-size:15px;'
                    f'line-height:1.5;color:{c["text"]};">{_inline(b["text"], links)}</td></tr></table>')
    if t == "button":
        href = links.get(b["link"].removeprefix("link:"), "#")
        return _row(_button(b["label"], href, accent), "8px 20px 20px 20px")
    if t == "divider":
        return _row(f'<div style="border-top:1px solid {c["border"]};font-size:0;line-height:0;">&nbsp;</div>')
    if t == "spacer":
        h = {"small": 8, "medium": 16, "large": 32}[b.get("size", "medium")]
        return f'<tr><td style="height:{h}px;font-size:0;line-height:0;">&nbsp;</td></tr>'
    if t == "signature":
        lines = [f'<span>{_inline(b["closing"], links)}</span>', f'<strong>{_inline(b["name"], links)}</strong>']
        lines += [_inline(b[k], links) for k in ("title", "org", "contact") if b.get(k)]
        return _row(f'<p style="margin:0;font-size:15px;line-height:1.6;color:{c["text"]};">{"<br>".join(lines)}</p>',
                    "8px 20px 24px 20px")
    if t == "footer":
        bg, fg, muted = _section(theme, "footer")
        if bg == "#FFFFFF":
            bg, muted = c["bg"], c["muted"]
        icons = ""
        if b.get("social"):
            # Production: hosted PNG icons with alt text (SVG is not shown by Gmail/Outlook).
            icons = '<table role="presentation" align="center" cellpadding="0" cellspacing="0" border="0"><tr>' + "".join(
                f'<td style="padding:0 6px;"><a href="{html.escape(links.get(sl["link"].removeprefix("link:"), "#"))}" '
                f'title="{html.escape(sl["platform"])}" style="display:inline-block;width:40px;height:40px;line-height:40px;'
                f'border-radius:20px;background:#FFFFFF;color:#111111;font-weight:bold;font-size:14px;text-decoration:none;'
                f'text-align:center;">{SOCIAL_LABEL.get(sl["platform"], "?")}</a></td>' for sl in b["social"]) + '</tr></table>'
        parts = [_inline(b[k], links) for k in ("address", "legal") if b.get(k)]
        if b.get("unsubscribe"):
            parts.append(f'<a href="#" style="color:{muted};">Unsubscribe</a>')
        text = f'<div style="margin-top:{16 if icons else 0}px;">{"<br>".join(parts) or "&nbsp;"}</div>'
        return (f'<tr><td align="center" style="padding:28px 20px;background:{bg};font-size:13px;line-height:1.6;'
                f'color:{muted};text-align:center;">{icons}{text}</td></tr>')
    return _row(f'<p style="color:red;">Unknown block type {html.escape(t)}</p>')


def render_email(template: dict, links: dict | None = None, assets: dict | None = None) -> str:
    """The email itself: a standalone, email-safe HTML document."""
    links, assets, c, meta = links or {}, assets or {}, THEME, template["meta"]
    theme = template.get("theme") or {}
    blocks = template["blocks"]
    parts = []
    for i, b in enumerate(blocks):
        parts.append(_block(b, links, assets, theme))
        nxt = blocks[i + 1]["type"] if i + 1 < len(blocks) else None
        # Breathing room under the header, except when the hero banner sits flush against it.
        if b["type"] in ("header", "hero_banner", "image") and nxt not in (None, "image", "heading", "hero_banner", "footer"):
            parts.append('<tr><td style="height:20px;font-size:0;line-height:0;">&nbsp;</td></tr>')
    body = "".join(parts)
    pre = html.escape(meta.get("preheader", ""))
    return f"""<!DOCTYPE html>
<html lang="{html.escape(meta.get('language', 'en'))}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(meta.get('subject', ''))}</title></head>
<body style="margin:0;padding:0;background:{c['bg']};">
<div style="display:none;max-height:0;overflow:hidden;">{pre}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{c['bg']};">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
 style="width:100%;max-width:600px;background:{c['card']};font-family:{c['font']};">
{body}
</table></td></tr></table></body></html>"""


def render_review_page(name: str, response: dict, links: dict | None = None, assets: dict | None = None) -> str:
    """Inbox-style review page: subject options, preheader, routing, warnings and fill slots, then the email."""
    tpl_entry = response["templates"][0] if "templates" in response else {"template": response}
    template, meta = tpl_entry["template"], tpl_entry["template"]["meta"]
    theme_desc = html.escape(json.dumps(template.get("theme") or {"style": "default (light)"}))
    email = render_email(template, links, assets)
    subjects = "".join(f"<li>{html.escape(s)}{' <b>(selected)</b>' if s == meta.get('subject') else ''}</li>"
                       for s in meta["subject_options"])
    fills = "".join(f"<li>{html.escape(f['needed'])} <small>({html.escape(str(f.get('block_id')))})</small></li>"
                    for f in tpl_entry.get("fill_slots", [])) or "<li>none</li>"
    warns = "".join(f"<li>{html.escape(w['kind'])}: {html.escape(w['message'])}</li>"
                    for w in tpl_entry.get("warnings", [])) or "<li>none</li>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preview: {html.escape(name)}</title>
<style>body{{margin:0;font-family:system-ui,Arial,sans-serif;background:#E9ECF1;color:#1F2933}}
.bar{{max-width:900px;margin:0 auto;padding:16px}} .card{{background:#fff;border-radius:8px;padding:14px 18px;margin-bottom:12px}}
h1{{font-size:18px;margin:0 0 8px}} .meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
ul{{margin:4px 0 0;padding-left:18px;font-size:14px}} .k{{font-size:12px;text-transform:uppercase;color:#616E7C;letter-spacing:.04em}}
.w{{display:flex;gap:8px;margin-bottom:8px}} button{{padding:6px 12px;border:1px solid #CBD2D9;background:#fff;border-radius:4px;cursor:pointer}}
iframe{{width:100%;border:0;background:#F2F4F7;border-radius:8px;display:block;margin:0 auto}}
a{{color:#1F4E79}}</style></head><body><div class="bar">
<p><a href="index.html">← All previews</a></p>
<div class="card"><h1>{html.escape(name)}</h1>
<div class="meta">
<div><div class="k">Category / routing / tone</div>{html.escape(meta['category'])} · {html.escape(meta['routing'])} · {html.escape(meta['tone'])}</div>
<div><div class="k">Theme (user's choice)</div><code style="font-size:12px">{theme_desc}</code></div>
<div><div class="k">Preheader</div>{html.escape(meta['preheader'])}</div>
<div><div class="k">Subject options</div><ul>{subjects}</ul></div>
<div><div class="k">Fill before sending</div><ul>{fills}</ul></div>
<div><div class="k">Warnings</div><ul>{warns}</ul></div>
</div></div>
<div class="w"><button onclick="f.style.width='100%'">Desktop</button><button onclick="f.style.width='375px'">Mobile 375px</button>
<a href="{html.escape(name)}.email.html" target="_blank" style="align-self:center;font-size:14px">Open the raw email HTML</a></div>
<iframe id="f" srcdoc="{html.escape(email, quote=True)}" onload="this.style.height=this.contentDocument.body.scrollHeight+'px'"></iframe>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--links", help="JSON file of {name: {link_key: url}} to resolve buttons", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    links_by_name = json.loads(Path(args.links).read_text()) if args.links else {}

    items: dict[str, dict] = {}
    for f in args.files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        if "templates" in data or "blocks" in data:
            items[Path(f).stem] = data
        else:
            items.update(data)

    index = []
    for name, resp in items.items():
        links = links_by_name.get(name, {})
        tpl = resp["templates"][0]["template"] if "templates" in resp else resp
        (out / f"{name}.email.html").write_text(render_email(tpl, links), encoding="utf-8")
        (out / f"{name}.html").write_text(render_review_page(name, resp, links), encoding="utf-8")
        index.append(f'<li><a href="{name}.html">{html.escape(name)}</a> — {html.escape(tpl["meta"]["subject"])}</li>')
    (out / "index.html").write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Email previews</title></head>'
        '<body style="font-family:system-ui,Arial,sans-serif;max-width:700px;margin:40px auto;padding:0 16px">'
        f'<h1>Email template previews</h1><ul style="line-height:1.9">{"".join(index)}</ul></body></html>',
        encoding="utf-8")
    print(f"Wrote {len(items)} previews to {out}/index.html")


if __name__ == "__main__":
    main()
