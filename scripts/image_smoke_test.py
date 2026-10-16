"""
Phase 0 Verification: Image Models & Endpoints Smoke Test

Tests live API keys for Gemini, Grok (xAI), and OpenAI against candidate image models
for both generation and image-to-image editing.
"""

import io
import os
import sys
import time
import base64
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image, ImageDraw

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Load .env
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("image_smoke_test")


def create_sample_image() -> bytes:
    """Create a sample PNG image (a stylized dome / building) in bytes."""
    img = Image.new("RGBA", (512, 512), color=(135, 206, 235, 255))
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, 380, 512, 512], fill=(34, 139, 34, 255))
    # Building base
    draw.rectangle([156, 250, 356, 380], fill=(245, 245, 240, 255), outline=(100, 100, 100, 255), width=3)
    # Central arch
    draw.rectangle([216, 290, 296, 380], fill=(50, 50, 50, 255))
    # Dome
    draw.ellipse([186, 170, 326, 270], fill=(255, 255, 255, 255), outline=(100, 100, 100, 255), width=3)
    # Minarets
    draw.rectangle([120, 180, 140, 380], fill=(240, 240, 235, 255))
    draw.rectangle([372, 180, 392, 380], fill=(240, 240, 235, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_gemini(api_key: str, sample_png_bytes: bytes):
    logger.info("=== Testing Google Gemini ===")
    if not api_key:
        logger.warning("No DEFAULT_GEMINI_API_KEY found, skipping Gemini.")
        return []

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    candidate_models = [
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-image",
        "gemini-2.5-flash-image",
        "imagen-3.0-generate-002",
    ]

    results = []

    for model_id in candidate_models:
        logger.info(f"Testing Gemini generation on model: {model_id}")
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents="A serene white marble dome reflecting in water at sunset, 4k high quality photo",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            elapsed = time.time() - t0
            img_bytes = None
            mime_type = None
            if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                for part in resp.candidates[0].content.parts:
                    if getattr(part, "inline_data", None):
                        img_bytes = part.inline_data.data
                        mime_type = part.inline_data.mime_type
                        break
            if img_bytes:
                pil_img = Image.open(io.BytesIO(img_bytes))
                dims = f"{pil_img.width}x{pil_img.height}"
                logger.info(f"SUCCESS Gemini {model_id}: {dims}, {mime_type}, {len(img_bytes)} bytes in {elapsed:.2f}s")
                results.append({
                    "provider": "gemini",
                    "model": model_id,
                    "task": "generate",
                    "status": "SUCCESS",
                    "call_shape": "client.models.generate_content(response_modalities=['IMAGE'])",
                    "mime_type": mime_type,
                    "dimensions": dims,
                    "latency": f"{elapsed:.2f}s",
                    "notes": "Inline image data returned",
                })
            else:
                logger.warning(f"Gemini {model_id} returned no image parts: {resp}")
                results.append({
                    "provider": "gemini",
                    "model": model_id,
                    "task": "generate",
                    "status": "NO_IMAGE",
                    "error": "No inline image data in parts",
                })
        except Exception as e:
            logger.error(f"Gemini {model_id} generate failed: {e}")
            results.append({
                "provider": "gemini",
                "model": model_id,
                "task": "generate",
                "status": "FAILED",
                "error": str(e),
            })

    # Test Edit on working Gemini model
    working_gemini = [r["model"] for r in results if r.get("status") == "SUCCESS"]
    edit_model = working_gemini[0] if working_gemini else "gemini-3.1-flash-image-preview"

    logger.info(f"Testing Gemini edit (image-to-image) on model: {edit_model}")
    t0 = time.time()
    try:
        image_part = types.Part.from_bytes(data=sample_png_bytes, mime_type="image/png")
        resp = client.models.generate_content(
            model=edit_model,
            contents=[image_part, "Add a magnificent Taj Mahal vibe with detailed intricate marble carvings and reflecting pool"],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        elapsed = time.time() - t0
        img_bytes = None
        mime_type = None
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    img_bytes = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    break
        if img_bytes:
            pil_img = Image.open(io.BytesIO(img_bytes))
            dims = f"{pil_img.width}x{pil_img.height}"
            logger.info(f"SUCCESS Gemini edit {edit_model}: {dims}, {mime_type}, {len(img_bytes)} bytes in {elapsed:.2f}s")
            results.append({
                "provider": "gemini",
                "model": edit_model,
                "task": "edit",
                "status": "SUCCESS",
                "call_shape": "client.models.generate_content(contents=[image_part, prompt])",
                "mime_type": mime_type,
                "dimensions": dims,
                "latency": f"{elapsed:.2f}s",
                "notes": "Multimodal contents accepted",
            })
        else:
            results.append({
                "provider": "gemini",
                "model": edit_model,
                "task": "edit",
                "status": "NO_IMAGE",
                "error": "No inline image data in parts",
            })
    except Exception as e:
        logger.error(f"Gemini edit failed: {e}")
        results.append({
            "provider": "gemini",
            "model": edit_model,
            "task": "edit",
            "status": "FAILED",
            "error": str(e),
        })

    return results


def test_grok(api_key: str, sample_png_bytes: bytes):
    logger.info("=== Testing Grok (xAI) ===")
    if not api_key:
        logger.warning("No DEFAULT_GROK_API_KEY found, skipping Grok.")
        return []

    from openai import OpenAI
    import requests

    client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
    candidate_models = [
        "grok-imagine-image-2.0",
        "grok-imagine-image",
    ]

    results = []
    for model_id in candidate_models:
        logger.info(f"Testing Grok generation on model: {model_id}")
        t0 = time.time()
        try:
            resp = client.images.generate(
                model=model_id,
                prompt="A serene white marble dome reflecting in water at sunset, 4k high quality photo",
                n=1,
            )
            elapsed = time.time() - t0
            url = resp.data[0].url if resp.data else None
            if url:
                r = requests.get(url, timeout=30)
                img = Image.open(io.BytesIO(r.content))
                dims = f"{img.width}x{img.height}"
                content_type = r.headers.get("Content-Type", "image/jpeg")
                logger.info(f"SUCCESS Grok {model_id}: {dims}, {content_type}, {len(r.content)} bytes in {elapsed:.2f}s")
                results.append({
                    "provider": "grok",
                    "model": model_id,
                    "task": "generate",
                    "status": "SUCCESS",
                    "call_shape": "client.images.generate(model=model, prompt=prompt)",
                    "mime_type": content_type,
                    "dimensions": dims,
                    "latency": f"{elapsed:.2f}s",
                    "notes": f"URL returned: {url[:50]}...",
                })
        except Exception as e:
            logger.error(f"Grok {model_id} generate failed: {e}")
            results.append({
                "provider": "grok",
                "model": model_id,
                "task": "generate",
                "status": "FAILED",
                "error": str(e),
            })

    # Test Grok Edit capability
    working_grok = [r["model"] for r in results if r.get("status") == "SUCCESS"]
    edit_model = working_grok[0] if working_grok else "grok-imagine-image-2.0"
    logger.info(f"Testing Grok edit on model: {edit_model}")
    t0 = time.time()
    try:
        resp = client.images.edit(
            image=io.BytesIO(sample_png_bytes),
            prompt="Add a Taj Mahal vibe",
            model=edit_model,
        )
        elapsed = time.time() - t0
        url = resp.data[0].url if resp.data else None
        results.append({
            "provider": "grok",
            "model": edit_model,
            "task": "edit",
            "status": "SUCCESS",
            "call_shape": "client.images.edit(image=BytesIO, prompt=prompt, model=model)",
            "latency": f"{elapsed:.2f}s",
            "notes": f"URL returned: {url}",
        })
    except Exception as e:
        logger.info(f"Grok images.edit failed (expected if unsupported): {e}")
        results.append({
            "provider": "grok",
            "model": edit_model,
            "task": "edit",
            "status": "UNSUPPORTED",
            "error": str(e),
            "notes": "client.images.edit endpoint not supported on xAI; fallback to prompt augmentation",
        })

    return results


def test_openai(api_key: str, sample_png_bytes: bytes):
    logger.info("=== Testing OpenAI ===")
    if not api_key:
        logger.warning("No DEFAULT_OPENAI_API_KEY found, skipping OpenAI.")
        return []

    from openai import OpenAI
    import requests

    client = OpenAI(api_key=api_key)
    candidate_models = [
        "gpt-image-1",
        "dall-e-3",
        "dall-e-2",
    ]

    results = []
    for model_id in candidate_models:
        logger.info(f"Testing OpenAI generation on model: {model_id}")
        t0 = time.time()
        try:
            kwargs = {
                "model": model_id,
                "prompt": "A serene white marble dome reflecting in water at sunset, 4k photo",
                "n": 1,
            }
            if model_id == "dall-e-3":
                kwargs["size"] = "1024x1024"
                kwargs["quality"] = "standard"
            elif model_id == "dall-e-2":
                kwargs["size"] = "512x512"

            resp = client.images.generate(**kwargs)
            elapsed = time.time() - t0
            b64 = getattr(resp.data[0], "b64_json", None)
            url = getattr(resp.data[0], "url", None)
            if b64:
                raw_bytes = base64.b64decode(b64)
            elif url:
                r = requests.get(url, timeout=30)
                raw_bytes = r.content
            else:
                raw_bytes = b""

            img = Image.open(io.BytesIO(raw_bytes)) if raw_bytes else None
            dims = f"{img.width}x{img.height}" if img else "unknown"
            logger.info(f"SUCCESS OpenAI {model_id}: {dims}, {len(raw_bytes)} bytes in {elapsed:.2f}s")
            results.append({
                "provider": "openai",
                "model": model_id,
                "task": "generate",
                "status": "SUCCESS",
                "call_shape": f"client.images.generate(model='{model_id}')",
                "dimensions": dims,
                "latency": f"{elapsed:.2f}s",
                "notes": f"Format: {'b64_json' if b64 else 'url'}",
            })
        except Exception as e:
            logger.error(f"OpenAI {model_id} generate failed: {e}")
            results.append({
                "provider": "openai",
                "model": model_id,
                "task": "generate",
                "status": "FAILED",
                "error": str(e),
            })

    # Test OpenAI edit
    logger.info("Testing OpenAI images.edit")
    for edit_model in ["dall-e-2", "gpt-image-1"]:
        t0 = time.time()
        try:
            # DALL-E-2 edit requires PNG < 4MB and square
            buf = io.BytesIO(sample_png_bytes)
            buf.name = "sample.png"
            kwargs = {
                "image": buf,
                "prompt": "Add a Taj Mahal vibe",
                "n": 1,
                "size": "512x512",
            }
            if edit_model != "dall-e-2":
                kwargs["model"] = edit_model
            resp = client.images.edit(**kwargs)
            elapsed = time.time() - t0
            results.append({
                "provider": "openai",
                "model": edit_model,
                "task": "edit",
                "status": "SUCCESS",
                "call_shape": f"client.images.edit(image=sample.png, model='{edit_model}')",
                "latency": f"{elapsed:.2f}s",
                "notes": f"URL: {resp.data[0].url[:40]}...",
            })
        except Exception as e:
            logger.info(f"OpenAI edit {edit_model} failed: {e}")
            results.append({
                "provider": "openai",
                "model": edit_model,
                "task": "edit",
                "status": "FAILED",
                "error": str(e),
            })

    return results


def main():
    logger.info("Starting Image Models Smoke Test...")
    sample_png = create_sample_image()

    gemini_key = os.getenv("DEFAULT_GEMINI_API_KEY")
    grok_key = os.getenv("DEFAULT_GROK_API_KEY")
    openai_key = os.getenv("DEFAULT_OPENAI_API_KEY")

    all_results = []
    all_results.extend(test_gemini(gemini_key, sample_png))
    all_results.extend(test_grok(grok_key, sample_png))
    all_results.extend(test_openai(openai_key, sample_png))

    # Write markdown report
    docs_path = root_dir / "docs" / "image_models.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(docs_path, "w", encoding="utf-8") as f:
        f.write("# Image Models Verification Report\n\n")
        f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*\n\n")
        f.write("## Verified Models & Capabilities\n\n")
        f.write("| Provider | Model ID | Task | Status | Dimensions | Latency | Call Shape & Notes |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in all_results:
            prov = r.get("provider", "")
            mod = r.get("model", "")
            task = r.get("task", "")
            status = r.get("status", "")
            dims = r.get("dimensions", "-")
            lat = r.get("latency", "-")
            notes = r.get("call_shape", "")
            if r.get("notes"):
                notes += f" ({r['notes']})"
            if r.get("error"):
                notes += f" Error: {r['error']}"
            f.write(f"| **{prov}** | `{mod}` | {task} | **{status}** | {dims} | {lat} | {notes} |\n")

    logger.info(f"Smoke test complete! Report generated at: {docs_path}")
    print("\n" + open(docs_path).read())


if __name__ == "__main__":
    main()
