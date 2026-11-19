"""Standalone HTML for fidelity eval. Mirrors the layout rules in FRONTEND_HANDOFF §4.5 and §4.8."""

from html import escape
from urllib.parse import urlencode

from src.design_canvas.schema import safe_href
from src.design_canvas.schema_v11 import ALLOWED_FONT_FAMILIES, cdn_image_url
from src.design_canvas.typography_match import FONT_FILES, _axes

WEIGHT = {"light": 300, "normal": 400, "medium": 500, "semibold": 600, "bold": 700, "extrabold": 800}


def font_query(family):
    """Google Fonts CSS2 family spec with the bundled file's axes: every weight, and optical size where the font has
    it, so the browser applies optical sizing like the typography matcher does."""
    directory, files = FONT_FILES[family]
    if isinstance(files, dict):
        return f"{family}:wght@" + ";".join(str(w) for w in sorted(files))
    axes = {name: (int(low), int(high)) for name, low, high, _ in _axes(directory, files)}
    weight = "{}..{}".format(*axes["Weight"])
    if "Optical size" in axes:
        return f"{family}:opsz,wght@" + "{}..{}".format(*axes["Optical size"]) + "," + weight
    return f"{family}:wght@{weight}"
ICONS = dict(zip(
    ("arrow_left", "arrow_right", "chevron_left", "chevron_right", "plus", "close", "check",
     "sparkles", "search", "send", "copy", "arrow_up_right", "chevron_down"),
    ("←", "→", "‹", "›", "+", "×", "✓", "✦", "⌕", "➤", "⧉", "↗", "⌄"),
))


def _gradient(value):
    return f'linear-gradient({value["angle_deg"]}deg,{",".join(value["stops"])})'


def _style(value):
    css = []
    for field, property_name, unit in (
        ("background_color", "background-color", ""), ("border_radius", "border-radius", "px"),
        ("max_width", "max-width", "px"), ("width_percent", "width", "%"),
    ):
        if value.get(field) is not None:
            css.append(f"{property_name}:{value[field]}{unit}")
    if value.get("border_width") and value.get("border_color"):
        css.append(f'border:{value["border_width"]}px solid {value["border_color"]}')
    if value.get("padding"):
        css.append("padding:" + " ".join(f'{value["padding"][side]}px' for side in ("top", "right", "bottom", "left")))
    if value.get("background_gradient"):
        css.append("background-image:" + _gradient(value["background_gradient"]))
    if value.get("shadow") == "soft":
        css.append("box-shadow:0 8px 24px rgb(0 0 0 / .35)")
    if value.get("shadow") == "glow":
        css.append(f'box-shadow:0 0 24px color-mix(in srgb, {value.get("shadow_color") or "#FFFFFF"} 40%, transparent)')
    return css


def _font(token):
    return f'"{token["family"]}",{token["fallback"]}'


def _span(span):
    css = [f'color:{span["color"]}', f'font-weight:{span.get("font_weight_value") or WEIGHT[span["font_weight"]]}']
    if span["italic"]:
        css.append("font-style:italic")
    if span["underline"]:
        css.append("text-decoration:underline")
    if span.get("gradient"):
        css.append("--ink:" + _gradient(span["gradient"]))
    if span.get("baseline") in ("super", "sub"):
        css.append(f'vertical-align:{span["baseline"]}')
    if span.get("size_scale"):
        css.append(f'font-size:{span["size_scale"]}em')
    class_attr = ' class="ink"' if span.get("gradient") else ""
    return f'<span{class_attr} style="{escape(";".join(css))}">{escape(span["text"])}</span>'


def _node(node, fonts, parent_layout=None):
    variant = node["node_type"]
    style = node.get("style") or {}
    css = _style(style)
    attrs = f'data-node-id="{escape(node["id"])}"'
    token = fonts.get(node.get("font_role", "body"))
    if token:
        css.append("font-family:" + _font(token))
    if parent_layout == "flex_row" and style.get("flex_grow"):
        css.extend([f'flex:{style["flex_grow"]} 1 0', "min-width:0"])
    if variant == "container":
        tag = "div"
        layout = node["layout"]
        if layout == "grid":
            ratios = node.get("column_ratios") or [1] * (node["columns"] or min(len(node["children"]), 3))
            css.extend(["display:grid", "grid-template-columns:" + " ".join(f"minmax(0,{ratio}fr)" for ratio in ratios)])
        else:
            css.extend(["display:flex", "flex-direction:" + ("row" if layout == "flex_row" else "column"),
                        "flex-wrap:" + ("wrap" if node.get("wrap") and layout == "flex_row" else "nowrap")])
        css.extend([f'gap:{node["gap_px"]}px', f'align-items:{node["align_items"]}',
                    f'justify-content:{node["justify_content"].replace("_", "-")}'])
        content = "".join(_node(child, fonts, layout) for child in node["children"])
    elif variant == "text":
        tag = "p" if node["tag"] == "caption" else node["tag"]
        css.extend(["margin:0", f'text-align:{node["align"]}', f'font-size:{node["font_size_px"]}px',
                    f'line-height:{node["line_height"]}', f'letter-spacing:{node.get("letter_spacing_em", 0)}em',
                    f'text-transform:{node.get("text_transform", "none")}', "white-space:pre-wrap"])
        content = "".join(map(_span, node["spans"]))
    elif variant == "divider":
        tag, content = "div", ""
        edge = f'{node["thickness_px"]}px solid {node["color"]}'
        if node.get("orientation") == "vertical":
            css.extend([f"border-left:{edge}", "align-self:stretch", "flex-shrink:0"])
            attrs += ' role="separator" aria-orientation="vertical"'
        else:
            css.extend([f"border-top:{edge}", f'width:{node["width_percent"]}%',
                        f'margin:{node["margin_y_px"]}px 0'])
            attrs += ' role="separator"'
    elif variant == "button":
        href = safe_href(node["href"])
        tag = "a" if href else "span"
        if href:
            attrs += f' href="{escape(href)}"'
        css.extend(["display:inline-flex", "align-items:center", "gap:.5em", "text-decoration:none",
                    f'color:{node["text_color"]}', f'font-size:{node["font_size_px"]}px',
                    f'font-weight:{node.get("font_weight_value") or WEIGHT[node["font_weight"]]}',
                    f'letter-spacing:{node.get("letter_spacing_em", 0)}em',
                    f'text-transform:{node.get("text_transform", "none")}'])
        content = escape(node["label"])
        if node.get("trailing_icon"):
            content += f'<span aria-hidden="true">{ICONS[node["trailing_icon"]]}</span>'
    elif variant == "image":
        css.extend([f'width:{node["width_percent"]}%', "display:block", "overflow:hidden",
                    "aspect-ratio:" + node["aspect_ratio"].replace(":", " / ")])
        try:
            src = cdn_image_url(node.get("src"))
        except ValueError:
            src = None                           # never load an image from outside the CDN
        if src:
            css.append(f'object-fit:{node["object_fit"]}')
            return (f'<img {attrs} src="{escape(src)}" alt="{escape(node["alt"])}" '
                    f'style="{escape(";".join(css))}">')
        # Empty slot: a neutral placeholder labelled with the alt text; shimmer while a photo is generating.
        css.extend(["background-color:" + ((node.get("style") or {}).get("background_color") or "#E7E7EC"),
                    "display:flex", "align-items:center", "justify-content:center", "color:#7A7A85",
                    "font-size:13px", "text-align:center", "padding:8px"])
        if node.get("placeholder_id"):
            attrs += ' data-generating="true"'
        tag, content = "div", escape(node["alt"])
    elif variant == "icon_button":
        tag, content = "span", ICONS[node["icon_name"]]
        attrs += f' role="img" aria-label="{escape(node["aria_label"])}"'
        css.extend([f'font-size:{node["size_px"]}px', f'color:{node["color"]}'])
    else:
        tag = "svg"
        attrs += f' viewBox="{node["view_box"]}" width="{node["width_px"]}" height="{node["height_px"]}"'
        content = ""
        for layer in node["layers"]:
            geometry = " ".join(f'{key}="{escape(str(value))}"' for key, value in layer["attributes"].items()
                                if key in {"d", "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1",
                                           "x2", "y2", "width", "height", "transform", "points"})
            content += (f'<{layer["svg_element"]} data-layer-id="{layer["layer_id"]}" {geometry}'
                        f' fill="{layer["fill_color"]}" stroke="{layer["stroke_color"] or "none"}"'
                        f' stroke-width="{layer["stroke_width"]}" opacity="{layer["opacity"]}"/>')
    return f'<{tag} {attrs} style="{escape(";".join(css))}">{content}</{tag}>'


def render_html(ast, width):
    fonts = {role: token for role, token in (ast.get("fonts") or {}).items()
             if token and token["family"] in ALLOWED_FONT_FAMILIES}
    families = sorted({token["family"] for token in fonts.values()})
    font_links = []
    for family in families:
        query = urlencode({"family": font_query(family), "display": "swap"})
        font_links.append(f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{escape(query)}">')
    body_font = _font(fonts["body"]) if "body" in fonts else "system-ui,sans-serif"
    natural_width = ast["root"]["style"].get("max_width") or width
    return (f'<!doctype html><meta charset="UTF-8">{"".join(font_links)}<style>'
            '*{box-sizing:border-box}body{margin:0}#canvas>div{width:100%}'
            '@supports (background-clip:text){.ink{background-image:var(--ink);background-clip:text;color:transparent!important}}'
            f'</style><main id="canvas" style="width:{natural_width}px;font-family:{escape(body_font)}">'
            f'{_node(ast["root"], fonts)}</main>'
            '<script>const canvas=document.getElementById("canvas");'
            'function fit(){canvas.style.zoom=Math.min(1,document.documentElement.clientWidth/canvas.offsetWidth)}'
            'addEventListener("resize",fit);fit();</script>')
