"""
Every email image is delivered at exactly its slot size and under the byte limit, whatever went in.
"""

import io
import random
import unittest

from PIL import Image

from src.email_template.images import MAX_BYTES, SLOTS, SlotImageError, fit_to_slot


def _image_bytes(w, h, mode="RGB", fmt="PNG", noisy=False):
    img = Image.new(mode, (w, h), (30, 90, 160, 255) if mode == "RGBA" else (30, 90, 160))
    if noisy:  # photographic-like content that does not compress well
        rnd = random.Random(0)
        img.putdata([tuple(rnd.randrange(256) for _ in range(len(mode))) for _ in range(w * h)])
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def _open(data):
    return Image.open(io.BytesIO(data))


class TestFitToSlot(unittest.TestCase):
    # Native generator outputs: 16:9 (Gemini), 3:2 (OpenAI), 1:1, portrait.
    SOURCES = [(1344, 768), (1536, 1024), (1024, 1024), (768, 1344), (300, 150)]

    def test_cropped_slots_are_exact_size_and_jpeg(self):
        for slot in ("hero", "inline"):
            for w, h in self.SOURCES:
                with self.subTest(slot=slot, source=(w, h)):
                    out = fit_to_slot(_image_bytes(w, h), slot)
                    s = SLOTS[slot]
                    self.assertEqual((out.width, out.height), (s.width, s.height))
                    self.assertEqual(_open(out.data).size, (s.width, s.height))
                    self.assertEqual(out.content_type, "image/jpeg")
                    self.assertEqual((out.display_width, out.display_height), (s.display_width, s.display_height))

    def test_noisy_photo_is_compressed_under_limit(self):
        out = fit_to_slot(_image_bytes(1344, 768, noisy=True), "hero")
        self.assertLessEqual(len(out.data), MAX_BYTES)

    def test_one_pixel_images_keep_their_colour(self):
        for slot in ("hero", "inline"):
            for size in ((1, 100), (100, 1), (1, 1)):
                with self.subTest(slot=slot, size=size):
                    out = fit_to_slot(_image_bytes(*size), slot)
                    with _open(out.data) as image:
                        self.assertEqual(image.size, (SLOTS[slot].width, SLOTS[slot].height))
                        pixel = image.getpixel((out.width // 2, out.height // 2))
                        self.assertTrue(all(abs(actual - expected) <= 2
                                            for actual, expected in zip(pixel, (30, 90, 160))), pixel)

    def test_transparency_is_flattened_for_photos(self):
        out = fit_to_slot(_image_bytes(1344, 768, mode="RGBA"), "hero")
        self.assertEqual(_open(out.data).mode, "RGB")

    def test_logo_is_fitted_not_cropped(self):
        out = fit_to_slot(_image_bytes(1000, 1000, mode="RGBA"), "logo")  # opaque RGBA -> JPEG
        self.assertEqual((out.width, out.height), (120, 120))
        self.assertEqual((out.display_width, out.display_height), (60, 60))

    def test_transparent_logo_stays_png(self):
        img = Image.new("RGBA", (800, 200), (0, 0, 0, 0))
        img.paste((200, 20, 20, 255), (100, 50, 700, 150))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out = fit_to_slot(buf.getvalue(), "logo")
        self.assertEqual(out.content_type, "image/png")
        self.assertEqual((out.width, out.height), (360, 90))

    def test_focal_point_moves_the_crop(self):
        img = Image.new("RGB", (1000, 1000), (255, 255, 255))
        img.paste((0, 0, 0), (0, 0, 1000, 200))  # black band at the top
        buf = io.BytesIO()
        img.save(buf, "PNG")
        top = _open(fit_to_slot(buf.getvalue(), "hero", focal_point=(0.5, 0.0)).data).convert("L")
        centre = _open(fit_to_slot(buf.getvalue(), "hero").data).convert("L")
        self.assertLess(top.getpixel((600, 10)), 50)
        self.assertGreater(centre.getpixel((600, 10)), 200)

    def test_bad_input(self):
        with self.assertRaises(SlotImageError):
            fit_to_slot(b"not an image", "hero")
        with self.assertRaises(SlotImageError):
            fit_to_slot(_image_bytes(10, 10), "banner")


if __name__ == "__main__":
    unittest.main()
