"""
Fixed image slots for email templates.

Every image in an email is one of these slots, at exactly this size, whatever its source (the user's file, an
AI-refined file, or an AI-generated one). Files are delivered at 2x the displayed size for high-density screens
and kept small, because many clients load images slowly or block them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps

TARGET_BYTES = 100 * 1024
MAX_BYTES = 200 * 1024
MAX_INPUT_BYTES = 15 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
_JPEG_QUALITIES = (82, 76, 70, 64, 58, 52, 46, 40)


@dataclass(frozen=True)
class Slot:
    id: str
    width: int             # delivered file size (2x)
    height: int
    display_width: int     # size in the email (the <img> width/height attributes)
    display_height: int
    crop: bool             # False = fit inside the box without cropping (logos)


SLOTS: dict[str, Slot] = {
    # 2:1 hero banner directly under the header.
    "hero": Slot("hero", 1200, 600, 600, 300, crop=True),
    # 16:9 image inside the 560px content column, before the call to action.
    "inline": Slot("inline", 1120, 630, 560, 315, crop=True),
    # Logo in the header: never cropped, fitted inside 180x60 (360x120 file).
    "logo": Slot("logo", 360, 120, 180, 60, crop=False),
}


class SlotImageError(ValueError):
    """The input cannot be turned into a slot image (unreadable, too large, or cannot be compressed enough)."""


@dataclass(frozen=True)
class FittedImage:
    data: bytes
    content_type: str
    width: int
    height: int
    display_width: int
    display_height: int


def _open(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise SlotImageError(f"Image is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width * img.height > MAX_INPUT_PIXELS:
            raise SlotImageError("Image dimensions are too large.")
        img.load()
    except SlotImageError:
        raise
    except Exception as e:  # PIL raises many types for corrupt or unsupported files
        raise SlotImageError(f"Could not read the image: {e}") from e
    return ImageOps.exif_transpose(img)


def _has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.convert("RGBA").getchannel("A")
        return alpha.getextrema()[0] < 255
    return False


def _cover(img: Image.Image, w: int, h: int, focal: tuple[float, float]) -> Image.Image:
    """Crop to the w:h ratio around the focal point (0..1, 0..1), then resize to exactly w x h."""
    fx, fy = (min(max(v, 0.0), 1.0) for v in focal)
    target = w / h
    if img.width / img.height > target:
        cw, ch = max(1, round(img.height * target)), img.height
    else:
        cw, ch = img.width, max(1, round(img.width / target))
    left = min(max(round(fx * img.width - cw / 2), 0), img.width - cw)
    top = min(max(round(fy * img.height - ch / 2), 0), img.height - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)


def _encode_jpeg(img: Image.Image) -> bytes:
    rgb = img.convert("RGB")
    data = b""
    for q in _JPEG_QUALITIES:
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=q, optimize=True, progressive=True)  # no exif passed: metadata stripped
        data = buf.getvalue()
        if len(data) <= TARGET_BYTES:
            break
    return data


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > TARGET_BYTES:
        buf = io.BytesIO()
        img.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(buf, "PNG", optimize=True)
        data = min(data, buf.getvalue(), key=len)
    return data


def fit_to_slot(image_bytes: bytes, slot_id: str, focal_point: Optional[tuple[float, float]] = None) -> FittedImage:
    """Turn any image into the exact file for a slot. focal_point defaults to the centre."""
    slot = SLOTS.get(slot_id)
    if slot is None:
        raise SlotImageError(f"Unknown image slot '{slot_id}'. Slots: {', '.join(SLOTS)}")
    img = _open(image_bytes)

    if slot.crop:
        out = _cover(img, slot.width, slot.height, focal_point or (0.5, 0.5))
        if _has_alpha(out):  # email photos are JPEG: flatten transparency onto white
            bg = Image.new("RGB", out.size, (255, 255, 255))
            bg.paste(out.convert("RGBA"), mask=out.convert("RGBA").getchannel("A"))
            out = bg
        data, ctype = _encode_jpeg(out), "image/jpeg"
        width, height = slot.width, slot.height
        disp_w, disp_h = slot.display_width, slot.display_height
    else:
        out = img.convert("RGBA") if _has_alpha(img) else img.convert("RGB")
        out.thumbnail((slot.width, slot.height), Image.Resampling.LANCZOS)
        data, ctype = (_encode_png(out), "image/png") if out.mode == "RGBA" else (_encode_jpeg(out), "image/jpeg")
        width, height = out.size
        scale = slot.display_width / slot.width
        disp_w, disp_h = max(1, round(width * scale)), max(1, round(height * scale))

    if len(data) > MAX_BYTES:
        raise SlotImageError(f"Could not compress the image below {MAX_BYTES // 1024} KB.")
    return FittedImage(data, ctype, width, height, disp_w, disp_h)
