"""
Tests for Mode 8 Pillow-based vision preprocessor (vision_extractor.py).

Uses synthetic 1×1 and 2×2 PNG images so tests run deterministically with no
external files — only stdlib + Pillow required.
"""

from __future__ import annotations

import io
import struct
import zlib

import pytest


def _make_png(pixels: list[tuple[int, int, int]], width: int, height: int) -> bytes:
    """
    Build a minimal valid PNG in-memory for testing.

    Each pixel is an (R, G, B) tuple. The image uses 8-bit RGB truecolor (type 2).
    """
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = len(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", length) + chunk_type + data + struct.pack(">I", crc)

    # Signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT — raw scanlines, one filter byte per row
    raw_rows = b""
    idx = 0
    for _ in range(height):
        row = b"\x00"  # filter byte = None
        for _ in range(width):
            r, g, b = pixels[idx % len(pixels)]
            row += bytes([r, g, b])
            idx += 1
        raw_rows += row

    compressed = zlib.compress(raw_rows)
    idat = png_chunk(b"IDAT", compressed)

    # IEND
    iend = png_chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractVisionMetadata:
    def test_dimensions_extracted(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(255, 0, 0)], width=4, height=4)
        meta = extract_vision_metadata(png)
        assert meta.width == 4
        assert meta.height == 4

    def test_dominant_color_is_hex(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(0, 0, 0)], width=2, height=2)
        meta = extract_vision_metadata(png)
        assert meta.dominant_background is not None
        assert meta.dominant_background.startswith("#")
        assert len(meta.dominant_background) == 7

    def test_dark_theme_pure_black(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(0, 0, 0)], width=2, height=2)
        meta = extract_vision_metadata(png)
        assert meta.is_dark_theme is True

    def test_light_theme_pure_white(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(255, 255, 255)], width=2, height=2)
        meta = extract_vision_metadata(png)
        assert meta.is_dark_theme is False

    def test_palette_nonempty(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(52, 245, 166)], width=2, height=2)
        meta = extract_vision_metadata(png)
        assert len(meta.top_hex_colors) >= 1

    def test_all_palette_colors_are_hex(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        # Multi-color image
        pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        png = _make_png(pixels, width=2, height=2)
        meta = extract_vision_metadata(png)
        for hex_color in meta.top_hex_colors:
            assert hex_color.startswith("#"), f"Not a hex: {hex_color}"
            assert len(hex_color) == 7, f"Wrong hex length: {hex_color}"

    def test_max_colors_limit(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(0, 0, 0)], width=2, height=2)
        meta = extract_vision_metadata(png, max_colors=4)
        assert len(meta.top_hex_colors) <= 4

    def test_invalid_bytes_raises(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        with pytest.raises(Exception):
            extract_vision_metadata(b"not a valid image")

    def test_meta_is_frozen(self):
        from src.design_canvas.vision_extractor import VisionMetadata

        meta = VisionMetadata(
            width=100,
            height=100,
            top_hex_colors=["#000000"],
            is_dark_theme=True,
            dominant_background="#000000",
        )
        with pytest.raises(Exception):
            meta.width = 200  # type: ignore[misc]

    def test_white_image_dominant_hex(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        png = _make_png([(255, 255, 255)], width=1, height=1)
        meta = extract_vision_metadata(png)
        # Dominant background should be close to white
        assert meta.dominant_background is not None
        assert meta.dominant_background.upper().startswith("#")

    def test_mint_green_theme_is_light(self):
        from src.design_canvas.vision_extractor import extract_vision_metadata

        # #34F5A6 is bright mint — luminance ≈ 0.77 (light theme)
        png = _make_png([(52, 245, 166)], width=2, height=2)
        meta = extract_vision_metadata(png)
        assert meta.is_dark_theme is False
