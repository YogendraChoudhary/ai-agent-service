"""Deterministic Pillow colour analysis for the reference-image prompt."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from math import atan2, dist, hypot, log1p, pi
from typing import Optional

from .colors import lab, linearize


@dataclass(frozen=True)
class PaletteRoles:
    backgrounds: list[str]
    surfaces: list[str]
    text: list[str]
    accents: list[str]


@dataclass(frozen=True)
class VisionMetadata:
    width: int
    height: int
    top_hex_colors: list[str]
    is_dark_theme: bool
    dominant_background: Optional[str]
    roles: Optional[PaletteRoles] = None
    quantized_hex_colors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Swatch:
    color: str
    count: int
    lab: tuple[float, float, float]
    luminance: float

    @property
    def chroma(self) -> float:
        return hypot(*self.lab[1:])


def _assign_roles(swatches: list[_Swatch], total_px: int) -> PaletteRoles:
    merged: list[_Swatch] = []
    for swatch in swatches:
        for i, kept in enumerate(merged):
            if dist(swatch.lab, kept.lab) < 6:
                merged[i] = replace(kept, count=kept.count + swatch.count)
                break
        else:
            merged.append(swatch)

    by_area = sorted(merged, key=lambda swatch: swatch.count, reverse=True)
    bg = by_area[0]
    backgrounds, surfaces, text, accents = [bg.color], [], [], []
    for swatch in by_area[1:]:
        area = swatch.count / total_px
        step = abs(swatch.lab[0] - bg.lab[0])
        contrast = (max(swatch.luminance, bg.luminance) + 0.05) / (min(swatch.luminance, bg.luminance) + 0.05)
        if area >= 0.1 and step <= 6:
            backgrounds.append(swatch.color)
        elif area >= 0.01 and step <= 18:
            surfaces.append(swatch.color)
        if contrast >= 4.5 and area <= 0.15:
            text.append(swatch.color)
        if area >= 0.0005 and (swatch.chroma >= 20 or (area <= 0.05 and swatch.chroma >= 6 and contrast >= 3)):
            accents.append(swatch)

    accents.sort(key=lambda swatch: swatch.chroma * log1p(swatch.count), reverse=True)
    distinct_hues: dict[int, str] = {}
    for swatch in accents:
        _, a, b = swatch.lab
        distinct_hues.setdefault(int((atan2(b, a) + pi) / (pi / 4)) % 8, swatch.color)
    accent_colors = dict.fromkeys([*distinct_hues.values(), *(swatch.color for swatch in accents)])
    return PaletteRoles(backgrounds[:3], surfaces[:5], text[:8], list(accent_colors)[:8])


def extract_vision_metadata(image_bytes: bytes, max_colors: int = 8) -> VisionMetadata:
    """Quantize a 400px thumbnail to 64 swatches. Keep unmerged swatches for fidelity checks."""
    from PIL import Image

    if not 1 <= max_colors <= 256:
        raise ValueError("max_colors must be between 1 and 256")
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
        img.thumbnail((400, 400), Image.Resampling.BOX)
        quantized = img.convert("RGB").quantize(colors=64, method=Image.Quantize.FASTOCTREE)
    palette = quantized.getpalette()
    counts = sorted(quantized.getcolors(maxcolors=64), key=lambda pair: -pair[0])
    swatches = []
    for count, index in counts:
        rgb = tuple(palette[index * 3:index * 3 + 3])
        r, g, b = map(linearize, rgb)
        swatches.append(_Swatch("#%02X%02X%02X" % rgb, count, lab(rgb), 0.2126 * r + 0.7152 * g + 0.0722 * b))
    colors = tuple(dict.fromkeys(swatch.color for swatch in swatches))
    return VisionMetadata(
        width=width, height=height, top_hex_colors=list(colors[:max_colors]),
        is_dark_theme=swatches[0].luminance < 0.5, dominant_background=colors[0],
        roles=_assign_roles(swatches, quantized.width * quantized.height),
        quantized_hex_colors=colors,
    )
