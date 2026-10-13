"""
S3 Service for uploading AI-generated images.
Downloads images from temporary URLs (DALL-E, Grok) or uploads raw bytes (Gemini)
to S3, returning a permanent CDN URL.
"""

import os
import uuid
import logging
import requests
import boto3

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


def upload_image_from_url(image_url: str, user_id: str, request_id: str = None) -> str:
    """
    Download an image from a temporary URL (e.g. DALL-E) and upload it to S3.

    Args:
        image_url: The temporary image URL to download from
        user_id: User ID for organizing files under user's directory
        request_id: Request ID for logging

    Returns:
        The permanent CDN URL for the uploaded image

    Raises:
        Exception: If download or upload fails
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    logger.info(f"{log_prefix}Downloading image from DALL-E URL...")
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()

    # Detect content type and extension
    content_type = response.headers.get("Content-Type", "image/png")
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    ext = ext_map.get(content_type, "png")

    # Check storage quota before uploading
    _check_storage_quota(user_id, len(response.content), request_id)

    # Generate unique key under user's directory
    file_id = str(uuid.uuid4())
    s3_key = f"{user_id}/ai-images/{file_id}.{ext}"

    image_bytes = response.content
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
    return cdn_url


def upload_image_from_bytes(image_bytes: bytes, content_type: str = "image/png", user_id: str = None, request_id: str = None) -> str:
    """
    Upload raw image bytes directly to S3 (used for Gemini which returns inline image data).

    Args:
        image_bytes: Raw image bytes
        content_type: MIME type of the image (e.g. "image/png")
        user_id: User ID for organizing files under user's directory
        request_id: Request ID for logging

    Returns:
        The permanent CDN URL for the uploaded image
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    ext = ext_map.get(content_type, "png")

    # Check storage quota before uploading
    _check_storage_quota(user_id, len(image_bytes), request_id)

    file_id = str(uuid.uuid4())
    s3_key = f"{user_id}/ai-images/{file_id}.{ext}"

    logger.info(f"{log_prefix}Uploading image bytes to S3: {s3_key} ({len(image_bytes)} bytes)")
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
    return cdn_url
