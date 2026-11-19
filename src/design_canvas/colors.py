"""Shared sRGB, contrast and CIE Lab calculations for design references."""

from math import dist


def hex_rgb(color: str) -> tuple[int, int, int]:
    digits = color.removeprefix("#")
    if len(digits) in (3, 4):
        digits = "".join(c * 2 for c in digits)
    return int(digits[:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def linearize(channel: int) -> float:
    s = channel / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = map(linearize, rgb)
    xyz = (
        (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047,
        0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
        (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883,
    )
    x, y, z = (v ** (1 / 3) if v > (6 / 29) ** 3 else v / (3 * (6 / 29) ** 2) + 4 / 29 for v in xyz)
    return 116 * y - 16, 500 * (x - y), 200 * (y - z)


def delta_e(a: str, b: str) -> float:
    return dist(lab(hex_rgb(a)), lab(hex_rgb(b)))
