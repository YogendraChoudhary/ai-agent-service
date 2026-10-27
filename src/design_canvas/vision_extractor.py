"""
Deterministic Pillow-based vision preprocessor for Mode 8.

``extract_vision_metadata()`` takes raw image bytes and returns a
``VisionMetadata`` dataclass that the harness injects into the LLM prompt as
``<deterministic_vision_metadata>`` XML so the model sees exact hex colours
instead of guessing them.

Performance budget: < 15 ms on a 1080×1920 reference screenshot.
No network calls, no LLM calls — pure CPU / Pillow.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class VisionMetadata:
    """Deterministically extracted facts about the reference image."""

    width: int
    height: int
    top_hex_colors: List[str]       # up to 8, sorted by frequency descending
    is_dark_theme: bool             # True when median luminance < 0.5
    dominant_background: Optional[str]   # #RRGGBB of most frequent colour


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG 2.1 relative luminance (0.0 = black, 1.0 = white)."""
    def _linearize(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def extract_vision_metadata(image_bytes: bytes, max_colors: int = 8) -> VisionMetadata:
    """
    Deterministically extract dominant colour palette and theme polarity from
    raw image bytes using Pillow's quantization.

    Args:
        image_bytes:  Raw bytes of any Pillow-supported image format (PNG, JPEG, …).
        max_colors:   Maximum palette entries to return (default 8).

    Returns:
        VisionMetadata (frozen dataclass).
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Pillow is required for vision preprocessing. "
            "Run: pip install Pillow"
        ) from exc

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size

    # Pillow quantize → palette with pixel-count ordering
    quantized = img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    palette_data: List[int] = quantized.getpalette() or []

    # ``getcolors`` on the quantized image returns (count, palette_index) tuples
    counts_and_indices: Optional[List[Tuple[int, int]]] = quantized.getcolors(
        maxcolors=max_colors
    )

    hex_colors: List[str] = []
    if counts_and_indices and palette_data:
        sorted_pairs = sorted(counts_and_indices, key=lambda x: x[0], reverse=True)
        for _, idx in sorted_pairs[:max_colors]:
            base = idx * 3
            if base + 2 < len(palette_data):
                r, g, b = palette_data[base], palette_data[base + 1], palette_data[base + 2]
                h = _rgb_to_hex(r, g, b)
                if h not in hex_colors:
                    hex_colors.append(h)

    # Fallback: sample centre pixel
    if not hex_colors:
        cx, cy = width // 2, height // 2
        r, g, b = img.getpixel((cx, cy))  # type: ignore[misc]
        hex_colors = [_rgb_to_hex(r, g, b)]

    # Determine dark / light theme from the most-frequent colour
    dominant_bg = hex_colors[0] if hex_colors else "#000000"
    r_d = int(dominant_bg[1:3], 16)
    g_d = int(dominant_bg[3:5], 16)
    b_d = int(dominant_bg[5:7], 16)
    is_dark = _relative_luminance(r_d, g_d, b_d) < 0.5

    return VisionMetadata(
        width=width,
        height=height,
        top_hex_colors=hex_colors,
        is_dark_theme=is_dark,
        dominant_background=dominant_bg,
    )
