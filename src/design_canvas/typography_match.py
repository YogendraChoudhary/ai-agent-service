"""
Measured typography for screenshot recreation: which allow-listed family, weight and letter spacing reproduce each
large reference line, chosen by rendering candidates with the bundled OFL fonts and comparing ink masks.

The describe call supplies a pixel box per display/heading line; everything here is deterministic CPU work.
Browsers apply a font's optical-size axis from the CSS pixel size (font-optical-sizing: auto), so renders here do
the same. Letter spacing is solved from the width, not searched: CSS adds it after every character.
"""

from __future__ import annotations

import io
import statistics
import time
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageChops, ImageDraw, ImageFont

FONT_DIR = Path(__file__).parent / "fonts"

# family -> (directory, variable file) or (directory, {weight: static file}). Upright styles only.
FONT_FILES: dict[str, tuple[str, str | dict[int, str]]] = {
    "Inter": ("inter", "Inter[opsz,wght].ttf"),
    "Inter Tight": ("intertight", "InterTight[wght].ttf"),
    "DM Sans": ("dmsans", "DMSans[opsz,wght].ttf"),
    "Manrope": ("manrope", "Manrope[wght].ttf"),
    "Plus Jakarta Sans": ("plusjakartasans", "PlusJakartaSans[wght].ttf"),
    "Outfit": ("outfit", "Outfit[wght].ttf"),
    "Sora": ("sora", "Sora[wght].ttf"),
    "Space Grotesk": ("spacegrotesk", "SpaceGrotesk[wght].ttf"),
    "Instrument Sans": ("instrumentsans", "InstrumentSans[wdth,wght].ttf"),
    "Geist": ("geist", "Geist[wght].ttf"),
    "Urbanist": ("urbanist", "Urbanist[wght].ttf"),
    "Poppins": ("poppins", {300: "Poppins-Light.ttf", 400: "Poppins-Regular.ttf", 500: "Poppins-Medium.ttf",
                            600: "Poppins-SemiBold.ttf", 700: "Poppins-Bold.ttf", 800: "Poppins-ExtraBold.ttf"}),
    "Montserrat": ("montserrat", "Montserrat[wght].ttf"),
    "Playfair Display": ("playfairdisplay", "PlayfairDisplay[wght].ttf"),
    "Fraunces": ("fraunces", "Fraunces[SOFT,WONK,opsz,wght].ttf"),
    "DM Serif Display": ("dmserifdisplay", {400: "DMSerifDisplay-Regular.ttf"}),
    "Instrument Serif": ("instrumentserif", {400: "InstrumentSerif-Regular.ttf"}),
    "JetBrains Mono": ("jetbrainsmono", "JetBrainsMono[wght].ttf"),
    "IBM Plex Mono": ("ibmplexmono", {300: "IBMPlexMono-Light.ttf", 400: "IBMPlexMono-Regular.ttf",
                                      500: "IBMPlexMono-Medium.ttf", 600: "IBMPlexMono-SemiBold.ttf",
                                      700: "IBMPlexMono-Bold.ttf"}),
    "Space Mono": ("spacemono", {400: "SpaceMono-Regular.ttf", 700: "SpaceMono-Bold.ttf"}),
    "Geist Mono": ("geistmono", "GeistMono[wght].ttf"),
}
SERIF_FAMILIES = ("Playfair Display", "Fraunces", "DM Serif Display", "Instrument Serif")
MONO_FAMILIES = ("JetBrains Mono", "IBM Plex Mono", "Space Mono", "Geist Mono")
SANS_FAMILIES = tuple(f for f in FONT_FILES if f not in SERIF_FAMILIES + MONO_FAMILIES)
# Spec roles measured, and the document font token each one sets.
MEASURED_ROLES = {"display": "display", "numeral": "display", "heading": "heading", "subheading": "heading"}
NAMED_WEIGHTS = {300: "light", 400: "normal", 500: "medium", 600: "semibold", 700: "bold", 800: "extrabold"}

WORK_HEIGHT = 96          # ink height every line is compared at
MIN_LINE_PX = 18          # reference ink height below this is too small to measure
MIN_IOU = 0.45            # a best match below this means the box or the text is wrong: do not use it
FAMILY_TIE = 0.03         # one family for display and headings unless another fits clearly better
SPACING_SLACK = 0.01      # how far solved spacing may leave the range of the tracking class the description saw


@dataclass(frozen=True)
class MeasuredText:
    text_id: str
    role: str                 # document font token: display | heading
    family: str
    weight: int
    letter_spacing_em: float
    iou: float


@dataclass
class TypographyMatch:
    fonts: dict[str, str] = field(default_factory=dict)            # font token -> family
    texts: dict[str, MeasuredText] = field(default_factory=dict)   # spec text id -> measurement
    ranking: dict[str, list[tuple[str, float]]] = field(default_factory=dict)   # token -> (family, mean IoU), best first
    skipped: dict[str, str] = field(default_factory=dict)          # spec text id -> reason
    seconds: float = 0.0

    def as_report(self) -> dict:
        return {"fonts": self.fonts, "seconds": round(self.seconds, 3), "skipped": self.skipped,
                "ranking": {role: [(f, round(s, 3)) for f, s in rows[:4]] for role, rows in self.ranking.items()},
                "texts": {k: {"family": v.family, "weight": v.weight, "letter_spacing_em": v.letter_spacing_em,
                              "iou": round(v.iou, 3), "role": v.role} for k, v in self.texts.items()}}


def family_class(family: str) -> str:
    return "mono" if family in MONO_FAMILIES else "serif" if family in SERIF_FAMILIES else "sans"


def weight_range(family: str) -> tuple[int, int]:
    directory, files = FONT_FILES[family]
    if isinstance(files, dict):
        return min(files), max(files)
    axis = next(a for a in _axes(directory, files) if a[0] == "Weight")
    return int(axis[1]), int(axis[2])


def available_weight(family: str, weight: int) -> int:
    """The weight the family can actually draw: clamped for a variable font, the nearest file for a static one."""
    directory, files = FONT_FILES[family]
    if isinstance(files, dict):
        return min(files, key=lambda w: (abs(w - weight), w))
    low, high = weight_range(family)
    return max(low, min(high, weight))


def named_weight(weight: int) -> str:
    return NAMED_WEIGHTS[min(NAMED_WEIGHTS, key=lambda w: (abs(w - weight), -w))]


@lru_cache(maxsize=None)
def _axes(directory: str, filename: str) -> tuple[tuple[str, float, float, float], ...]:
    font = ImageFont.truetype(str(FONT_DIR / directory / filename), 20)
    out = []
    for axis in font.get_variation_axes():
        name = axis["name"].decode() if isinstance(axis["name"], bytes) else axis["name"]
        out.append(("Weight" if name.lower() == "weight" else "Optical size" if name.lower().startswith("optical")
                    else name, axis["minimum"], axis["maximum"], axis["default"]))
    return tuple(out)


@lru_cache(maxsize=512)
def _font(family: str, weight: int, size_px: int, opsz_px: int = 0) -> ImageFont.FreeTypeFont:
    directory, files = FONT_FILES[family]
    if isinstance(files, dict):
        return ImageFont.truetype(str(FONT_DIR / directory / files[available_weight(family, weight)]), size_px)
    font = ImageFont.truetype(str(FONT_DIR / directory / files), size_px)
    values = []
    for name, low, high, default in _axes(directory, files):
        wanted = weight if name == "Weight" else (opsz_px or size_px) if name == "Optical size" else default
        values.append(max(low, min(high, wanted)))
    font.set_variation_by_axes(values)
    return font


@lru_cache(maxsize=2048)
def render_line(family: str, weight: int, size_px: int, letter_spacing_em: float, text: str,
                opsz_px: int = 0) -> Image.Image:
    """White-on-black ink of one line, cropped to its ink (cached: callers must not modify it). Prefix advances
    keep the font's kerning."""
    font = _font(family, available_weight(family, weight), size_px, opsz_px)
    tracking = letter_spacing_em * size_px
    ascent, descent = font.getmetrics()
    pad = size_px // 2
    width = ceil(font.getlength(text) + max(tracking, 0) * len(text)) + 2 * pad
    img = Image.new("L", (max(width, 1), ascent + descent + 2 * pad), 0)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(text):
        if not ch.isspace():
            draw.text((pad + font.getlength(text[:i]) + tracking * i, pad), ch, font=font, fill=255)
    box = img.getbbox()
    return img.crop(box) if box else img


def _otsu(values: Iterable[int]) -> int:
    hist = [0] * 256
    for v in values:
        hist[v] += 1
    total = sum(hist)
    weighted = sum(i * h for i, h in enumerate(hist))
    best, threshold, below, below_sum = -1.0, 127, 0, 0
    for t in range(256):
        below += hist[t]
        below_sum += t * hist[t]
        if below in (0, total):
            continue
        mean_below, mean_above = below_sum / below, (weighted - below_sum) / (total - below)
        spread = below * (total - below) * (mean_below - mean_above) ** 2
        if spread > best:
            best, threshold = spread, t
    return threshold


def reference_line(image: Image.Image, box: list[int], pad: float = 0.06) -> Optional[Image.Image]:
    """
    Ink of the line inside a describe-call box, white on black, cropped to its ink.

    Ink is the colour distance from the box's background, normalised by the strongest ink nearby along the line, so
    a gradient headline (dim lavender into bright mint) is cut at the same relative edge everywhere, on light or
    dark backgrounds. Of several lines in the box, the one holding the most ink is kept.
    """
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    crop = image.convert("RGB").crop((max(0, int(x0 - dx)), max(0, int(y0 - dy)),
                                      min(image.width, int(x1 + dx)), min(image.height, int(y1 + dy))))
    w, h = crop.size
    if w < 8 or h < 8:
        return None
    pixels = list(crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata())
    border = pixels[:w] + pixels[-w:] + pixels[::w] + pixels[w - 1::w]
    bg = tuple(int(statistics.median(p[c] for p in border)) for c in range(3))
    dist = [max(abs(p[0] - bg[0]), abs(p[1] - bg[1]), abs(p[2] - bg[2])) for p in pixels]
    threshold = max(_otsu(dist), 24)
    rows = [sum(1 for v in dist[y * w:(y + 1) * w] if v > threshold) for y in range(h)]
    runs, start = [], None
    for y, count in enumerate(rows + [0]):
        if count and start is None:
            start = y
        elif not count and start is not None:
            runs.append([start, y - 1])
            start = None
    if not runs:
        return None
    # A run touching the crop's top or bottom edge is a neighbouring line cut by the box (a descender from above,
    # an ascender from below): never part of this line unless it is the line itself.
    main = max(runs, key=lambda r: sum(rows[r[0]:r[1] + 1]))
    runs = [r for r in runs if r is main or (r[0] > 0 and r[1] < h - 1)]
    tallest = max(b - a + 1 for a, b in runs)
    merged = [runs[0]]
    for a, b in runs[1:]:                       # i/j dots and accents sit a few rows above their stems
        if a - merged[-1][1] - 1 <= max(2, tallest * 0.18):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    top, bottom = max(merged, key=lambda r: sum(rows[r[0]:r[1] + 1]))
    band_h = bottom - top + 1
    if band_h < MIN_LINE_PX:
        return None
    top, bottom = max(0, top - 1), min(h - 1, bottom + 1)       # keep the anti-aliased edge rows
    column_peak = [max(dist[y * w + x] for y in range(top, bottom + 1)) for x in range(w)]
    reach = max(1, band_h // 2)
    local = [max(column_peak[max(0, x - reach):x + reach + 1]) for x in range(w)]
    floor = 0.25 * max(column_peak)
    coverage = Image.new("L", (w, bottom - top + 1))
    coverage.putdata([min(255, round(255 * dist[y * w + x] / local[x])) if local[x] >= floor else 0
                      for y in range(top, bottom + 1) for x in range(w)])
    # Upscale the grey coverage before thresholding so anti-aliased edges keep their sub-pixel position.
    scale = WORK_HEIGHT / band_h
    coverage = coverage.resize((max(1, round(w * scale)), max(1, round(coverage.height * scale))),
                               Image.Resampling.BICUBIC)
    line = coverage.point(lambda v: 255 if v >= 128 else 0)
    ink_box = line.getbbox()
    if not ink_box:
        return None
    line = line.crop(ink_box)
    line.info["source_height"] = (ink_box[3] - ink_box[1]) / scale      # ink height in the image's own pixels
    return line


def _scaled(mask: Image.Image, height: int) -> Image.Image:
    return mask.resize((max(1, round(mask.width * height / mask.height)), height), Image.Resampling.BICUBIC)


def _iou(reference: Image.Image, candidate: Image.Image) -> float:
    width = max(reference.width, candidate.width)
    a = Image.new("L", (width, reference.height), 0)
    b = Image.new("L", (width, reference.height), 0)
    a.paste(reference, (0, 0))
    b.paste(candidate.point(lambda v: 255 if v >= 128 else 0), (0, 0))
    inter = ImageChops.multiply(a, b).histogram()[255]
    union = ImageChops.lighter(a, b).histogram()[255]
    return inter / union if union else 0.0


def fit_line(reference: Image.Image, text: str, family: str, weight: int, opsz_px: int = 0,
             bounds: tuple[float, float] = (-0.1, 0.3)) -> tuple[float, float]:
    """
    (IoU, letter_spacing_em) for one family and weight: spacing solved so the ink widths agree, within ``bounds``.
    Bounding it keeps spacing from standing in for glyph width: a family narrower than the reference cannot win by
    spreading its letters (live 2026-11-19: "Burger", described as tight, measured as Inter +0.056 em).
    """
    size = 200
    base = render_line(family, weight, size, 0.0, text, opsz_px)
    scale = reference.height / base.height
    gaps = max(len(text) - 1, 1)
    spacing = (reference.width - base.width * scale) / (gaps * size * scale)
    spacing = round(max(bounds[0], min(bounds[1], spacing)), 3)
    candidate = _scaled(render_line(family, weight, size, spacing, text, opsz_px), reference.height)
    return _iou(reference, candidate), spacing


def tracking_range(letter_spacing: str, font_size_px: int) -> tuple[float, float]:
    """
    letter_spacing_em that reads like the reference's tracking class. Measured 2026-11-19 on the promo reference by
    matching each line's ink width against renders: tight 96 px display -0.03 in Inter Tight and -0.04 in DM Sans,
    tight 46 px heading -0.01 (Inter Tight) and -0.019 (DM Sans), wide mono labels +0.10 to +0.20. Large type takes
    more negative tracking than mid-size type; the band spans both families. Measured texts use their own value.
    """
    if letter_spacing == "wide":
        return 0.08, 0.25
    if letter_spacing == "tight":
        return (-0.045, -0.025) if font_size_px >= 72 else (-0.025, -0.01) if font_size_px >= 48 else (-0.02, -0.005)
    return -0.01, 0.02


def _ink(mask: Image.Image) -> int:
    return mask.histogram()[255]


def weight_by_area(reference: Image.Image, text: str, family: str, opsz_px: int = 0) -> int:
    """
    The family weight whose ink area, at the reference's line height, equals the reference's. Area grows steadily
    with weight and does not depend on letter positions or spacing, unlike overlap. Rounded to 25.
    """
    low, high = weight_range(family)
    target = _ink(reference)
    points = []
    for weight in sorted({available_weight(family, w) for w in (low, 400, 600, high)}):
        candidate = _scaled(render_line(family, weight, 200, 0.0, text, opsz_px), reference.height)
        points.append((weight, _ink(candidate.point(lambda v: 255 if v >= 128 else 0))))
    estimate = points[0][0] if target <= points[0][1] else points[-1][0]
    for (a, area_a), (b, area_b) in zip(points, points[1:]):
        if area_a <= target <= area_b and area_b > area_a:
            estimate = a + (b - a) * (target - area_a) / (area_b - area_a)
            break
    return available_weight(family, int(25 * round(estimate / 25)))


def measure_line(reference: Image.Image, text: str, family: str, tracking_class: Optional[str] = None,
                 ink_height_px: Optional[float] = None) -> tuple[float, int, float]:
    """
    (IoU, weight, letter_spacing_em) of one family against one reference line.

    ``ink_height_px`` is the line's ink height in the pixels the design is laid out in (the describe call's image).
    It gives the font size this family would need, from the box rather than the model's size guess: that size sets
    the optical size (drawn as a browser draws it) and, with ``tracking_class``, the spacing bounds.
    """
    size, bounds = 0, (-0.1, 0.3)
    if ink_height_px:
        probe = render_line(family, available_weight(family, 500), 200, 0.0, text)
        size = max(8, round(200 * ink_height_px / probe.height))
        if tracking_class:
            low, high = tracking_range(tracking_class, size)
            bounds = (low - SPACING_SLACK, high + SPACING_SLACK)
    weight = weight_by_area(reference, text, family, size)
    iou, spacing = fit_line(reference, text, family, weight, size, bounds)
    return iou, weight, spacing


def match_typography(image_bytes: bytes, spec, supplied_width: Optional[int] = None,
                     families: Optional[dict[str, tuple[str, ...]]] = None) -> TypographyMatch:
    """
    Measure every display/heading/numeral text that has a box. Each candidate family gets the weight that matches
    the line's ink area and the spacing that matches its width; per font token the family with the highest mean
    IoU wins. A later token takes a family the design already uses when it scores within FAMILY_TIE of its best.
    ``image_bytes`` may be the original upload: boxes are in the describe call's image pixels (``supplied_width``
    wide) and are scaled to it, so small headings are measured at full resolution.
    ``families`` maps a font class (sans, serif, mono) to candidates (default: every bundled family of that class).
    """
    started = time.perf_counter()
    result = TypographyMatch()
    with Image.open(io.BytesIO(image_bytes)) as img:
        image = img.convert("RGB")
    scale = image.width / supplied_width if supplied_width else 1.0
    groups: dict[str, list[tuple[object, str, Image.Image]]] = {}
    for entry in spec.texts:
        role = MEASURED_ROLES.get(entry.role)
        if role is None:
            continue
        line = entry.text.split("\n")[0].strip()
        if not entry.box:
            result.skipped[entry.id] = "no box"
        elif entry.raised_suffix:
            result.skipped[entry.id] = "raised suffix"
        elif len(line.replace(" ", "")) < 2:
            result.skipped[entry.id] = "too short"
        elif (reference := reference_line(image, [round(v * scale) for v in entry.box])) is None:
            result.skipped[entry.id] = "no measurable line in box"
        else:
            groups.setdefault(role, []).append((entry, line, reference))

    measured: dict[str, dict[str, list[tuple[float, int, float]]]] = {}
    for role, items in groups.items():
        cls = "mono" if all(e.font_class == "mono" for e, _, _ in items) else (
            "serif" if all(e.font_class in ("serif", "display_serif") for e, _, _ in items) else "sans")
        candidates = (families or {}).get(cls) or {"mono": MONO_FAMILIES, "serif": SERIF_FAMILIES}.get(cls, SANS_FAMILIES)
        measured[role] = {f: [measure_line(ref, line, f, e.letter_spacing, ref.info["source_height"] / scale)
                              for e, line, ref in items] for f in candidates}
        result.ranking[role] = sorted(((f, sum(r[0] for r in rows) / len(rows)) for f, rows in measured[role].items()),
                                      key=lambda row: row[1], reverse=True)

    for role in sorted(groups, key=lambda r: r != "display"):
        means = dict(result.ranking[role])
        family = result.ranking[role][0][0]
        for used in result.fonts.values():
            if used in means and means[used] >= means[family] - FAMILY_TIE:
                family = used
                break
        kept = False
        for (entry, _, _), (iou, weight, spacing) in zip(groups[role], measured[role][family]):
            if iou < MIN_IOU:
                result.skipped[entry.id] = f"best match only {iou:.2f}"
                continue
            result.texts[entry.id] = MeasuredText(entry.id, role, family, weight, spacing, iou)
            kept = True
        if kept:
            result.fonts[role] = family
    result.seconds = time.perf_counter() - started
    return result
