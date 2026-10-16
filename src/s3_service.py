"""
S3 Service for uploading AI-generated images.
Downloads images from temporary URLs (DALL-E, Grok) or uploads raw bytes (Gemini)
to S3, returning a permanent CDN URL.
"""

import io
import os
import uuid
import logging
from typing import Optional

import requests
import boto3
from PIL import Image

logger = logging.getLogger(__name__)

# Environment variables
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CDN_URL = os.getenv("CDN_URL")
BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

_webhook_base = BACKEND_WEBHOOK_URL.rsplit("/", 1)[0] if BACKEND_WEBHOOK_URL else ""
STORAGE_QUOTA_CHECK_URL = f"{_webhook_base}/check-storage-quota" if _webhook_base else ""
TRACK_FILE_URL = f"{_webhook_base}/track-file" if _webhook_base else ""
REFUND_STORAGE_URL = f"{_webhook_base}/refund-storage" if _webhook_base else ""

# Singleton S3 client (created lazily)
_s3_client = None


class StorageQuotaExceededError(Exception):
    """Raised when a user's storage quota would be exceeded by an AI image upload."""
    # Defined here — ABOVE _check_storage_quota which raises it — so that
    # static analysers and readers see the class before it is referenced.
    # (Bug #5 fix: class was previously defined after its first use on line 82)
    pass


def _get_s3_client():
    """Get or create the singleton S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _s3_client


def _check_storage_quota(user_id: str, file_size_bytes: int, request_id: str = None) -> None:
    """
    Check if the user has enough storage quota before uploading to S3.
    Raises StorageQuotaExceededError if the quota is exceeded.
    Does nothing (allows upload) if the quota check endpoint is not configured.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not STORAGE_QUOTA_CHECK_URL or not WEBHOOK_SECRET:
        logger.warning(f"{log_prefix}Storage quota check skipped — BACKEND_WEBHOOK_URL or WEBHOOK_SECRET not configured")
        return

    try:
        response = requests.post(
            STORAGE_QUOTA_CHECK_URL,
            json={"user_id": user_id, "file_size_bytes": file_size_bytes},
            headers={"x-webhook-secret": WEBHOOK_SECRET, "Content-Type": "application/json"},
            timeout=10,
        )
        data = response.json()

        if not data.get("allowed", True):
            available_mb = data.get("available_bytes", 0) / (1024 * 1024)
            raise StorageQuotaExceededError(
                f"Storage quota exceeded. Available: {available_mb:.1f} MB. "
                f"{data.get('message', '')}"
            )

        logger.info(
            f"{log_prefix}Storage quota OK — available {data.get('available_bytes', 0) / (1024*1024):.1f} MB"
        )
    except StorageQuotaExceededError:
        raise
    except Exception as e:
        # On network/parse error, log and allow the upload to proceed
        logger.warning(f"{log_prefix}Storage quota check failed (allowing upload): {e}")


def _refund_storage(user_id: str, file_size_bytes: int, request_id: str = None) -> None:
    """
    Release a pre-reserved storage reservation after an S3 upload failure.
    Fire-and-forget: logs a warning on failure but never raises.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not REFUND_STORAGE_URL or not WEBHOOK_SECRET:
        return

    try:
        requests.post(
            REFUND_STORAGE_URL,
            json={"user_id": user_id, "file_size_bytes": file_size_bytes},
            headers={"x-webhook-secret": WEBHOOK_SECRET, "Content-Type": "application/json"},
            timeout=10,
        )
        logger.info(f"{log_prefix}Storage reservation refunded: {file_size_bytes} bytes for user={user_id}")
    except Exception as e:
        logger.warning(f"{log_prefix}Storage refund failed (non-fatal): {e}")


def _track_file(s3_key: str, file_size: int, file_name: str, content_type: str, user_id: str, request_id: str = None) -> None:
    """
    Notify the backend to record the uploaded file in uploadedFiles + userStorage tables.
    Fire-and-forget: logs a warning on failure but never raises.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not TRACK_FILE_URL or not WEBHOOK_SECRET:
        logger.warning(f"{log_prefix}File tracking skipped — BACKEND_WEBHOOK_URL or WEBHOOK_SECRET not configured")
        return

    try:
        response = requests.post(
            TRACK_FILE_URL,
            json={
                "user_id": user_id,
                "file_path": s3_key,
                "file_size": file_size,
                "file_name": file_name,
                "file_type": content_type,
                "category": "ai-images",
            },
            headers={"x-webhook-secret": WEBHOOK_SECRET, "Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code == 200:
            logger.info(f"{log_prefix}File tracked in backend: {s3_key}")
        else:
            logger.warning(f"{log_prefix}File tracking returned {response.status_code}: {response.text}")
    except Exception as e:
        logger.warning(f"{log_prefix}File tracking failed (non-fatal): {e}")



def sniff_image_bytes(image_bytes: bytes) -> tuple[str, str]:
    """
    Sniff the real MIME type and file extension from raw bytes using magic numbers.
    Returns (content_type, extension).
    """
    if not image_bytes:
        return "image/png", "png"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    elif len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp", "webp"
    elif image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif", "gif"
    elif image_bytes.startswith(b"BM"):
        return "image/bmp", "bmp"

    try:
        import filetype
        kind = filetype.guess(image_bytes)
        if kind:
            ext = "jpg" if kind.extension in ("jpeg", "jpg") else kind.extension
            return kind.mime, ext
    except Exception:
        pass

    return "image/png", "png"


def convert_image_format(image_bytes: bytes, target_format: str) -> tuple[bytes, str, str]:
    """
    Convert image bytes to target format ('png', 'jpeg'/'jpg', 'webp') using Pillow if needed.
    Returns (converted_bytes, content_type, extension).
    """
    if not target_format or not image_bytes:
        ct, ext = sniff_image_bytes(image_bytes)
        return image_bytes, ct, ext

    target = target_format.lower().strip()
    if target in ("jpg", "jpeg"):
        target_ext = "jpg"
        target_mime = "image/jpeg"
        pil_format = "JPEG"
    elif target == "webp":
        target_ext = "webp"
        target_mime = "image/webp"
        pil_format = "WEBP"
    else:
        target_ext = "png"
        target_mime = "image/png"
        pil_format = "PNG"

    current_mime, current_ext = sniff_image_bytes(image_bytes)
    if current_ext == target_ext or (current_ext in ("jpg", "jpeg") and target_ext in ("jpg", "jpeg")):
        return image_bytes, current_mime, current_ext

    try:
        img = Image.open(io.BytesIO(image_bytes))
        out_buf = io.BytesIO()
        if pil_format == "JPEG":
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_buf, format="JPEG", quality=90)
        else:
            img.save(out_buf, format=pil_format)
        return out_buf.getvalue(), target_mime, target_ext
    except Exception as e:
        logger.warning(f"Image conversion to {target_format} failed ({e}), retaining format {current_ext}")
        return image_bytes, current_mime, current_ext


def get_image_dimensions(image_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
    """Extract (width, height) from raw image bytes using Pillow, or (None, None) if unreadable."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.width, img.height
    except Exception as e:
        logger.warning(f"Could not read image dimensions: {e}")
        return None, None


def upload_image_from_url_detailed(
    image_url: str,
    user_id: str,
    request_id: str = None,
    target_format: str = None,
) -> dict:
    """
    Download image from URL, sniff real format, optionally convert with Pillow,
    upload to S3, and return detailed metadata.
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    logger.info(f"{log_prefix}Downloading image from URL: {image_url[:60]}...")
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()

    raw_bytes = response.content

    # Sniff real type from bytes (magic numbers) & optionally convert
    if target_format:
        image_bytes, content_type, ext = convert_image_format(raw_bytes, target_format)
    else:
        content_type, ext = sniff_image_bytes(raw_bytes)
        image_bytes = raw_bytes

    width, height = get_image_dimensions(image_bytes)
    logger.info(f"{log_prefix}Sniffed image type: {content_type} (. {ext}), dimensions: {width}x{height}")

    # Check storage quota before uploading
    _check_storage_quota(user_id, len(image_bytes), request_id)

    file_id = str(uuid.uuid4())
    s3_key = f"{user_id}/ai-images/{file_id}.{ext}"

    logger.info(f"{log_prefix}Uploading to S3: {s3_key} ({len(image_bytes)} bytes)")
    client = _get_s3_client()
    try:
        client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType=content_type,
        )
    except Exception:
        _refund_storage(user_id, len(image_bytes), request_id)
        raise

    _track_file(s3_key, len(image_bytes), f"{file_id}.{ext}", content_type, user_id, request_id)
    cdn_url = f"{CDN_URL}/{s3_key}"
    logger.info(f"{log_prefix}Image uploaded successfully: {cdn_url}")

    return {
        "cdn_url": cdn_url,
        "width": width,
        "height": height,
        "content_type": content_type,
        "format": ext,
        "size_bytes": len(image_bytes),
    }


def upload_image_from_url(
    image_url: str,
    user_id: str,
    request_id: str = None,
    target_format: str = None,
) -> str:
    """Download image from URL, upload to S3, and return CDN URL string."""
    res = upload_image_from_url_detailed(image_url, user_id, request_id, target_format=target_format)
    return res["cdn_url"]


def upload_image_from_bytes_detailed(
    image_bytes: bytes,
    content_type: str = None,
    user_id: str = None,
    request_id: str = None,
    target_format: str = None,
) -> dict:
    """
    Upload raw image bytes directly to S3 with magic-byte sniffing and optional format conversion.
    Returns dictionary with cdn_url, width, height, format, content_type.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    # Always sniff real type from bytes (magic numbers) rather than trusting caller default
    sniffed_type, sniffed_ext = sniff_image_bytes(image_bytes)

    if target_format:
        final_bytes, final_type, final_ext = convert_image_format(image_bytes, target_format)
    else:
        # Type and extension both come from the bytes so they can never disagree.
        final_bytes = image_bytes
        final_type = sniffed_type
        final_ext = sniffed_ext

    width, height = get_image_dimensions(final_bytes)
    logger.info(f"{log_prefix}Sniffed image bytes type: {final_type} (. {final_ext}), dimensions: {width}x{height}")

    # Check storage quota before uploading
    _check_storage_quota(user_id, len(final_bytes), request_id)

    file_id = str(uuid.uuid4())
    s3_key = f"{user_id}/ai-images/{file_id}.{final_ext}"

    logger.info(f"{log_prefix}Uploading image bytes to S3: {s3_key} ({len(final_bytes)} bytes)")
    client = _get_s3_client()
    try:
        client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=final_bytes,
            ContentType=final_type,
        )
    except Exception:
        _refund_storage(user_id, len(final_bytes), request_id)
        raise

    _track_file(s3_key, len(final_bytes), f"{file_id}.{final_ext}", final_type, user_id, request_id)
    cdn_url = f"{CDN_URL}/{s3_key}"
    logger.info(f"{log_prefix}Image uploaded successfully: {cdn_url}")

    return {
        "cdn_url": cdn_url,
        "width": width,
        "height": height,
        "content_type": final_type,
        "format": final_ext,
        "size_bytes": len(final_bytes),
    }


def upload_image_from_bytes(
    image_bytes: bytes,
    content_type: str = "image/png",
    user_id: str = None,
    request_id: str = None,
    target_format: str = None,
) -> str:
    """Upload raw image bytes directly to S3 and return CDN URL string."""
    res = upload_image_from_bytes_detailed(
        image_bytes=image_bytes,
        content_type=content_type,
        user_id=user_id,
        request_id=request_id,
        target_format=target_format,
    )
    return res["cdn_url"]
