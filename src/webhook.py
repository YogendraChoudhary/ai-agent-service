"""
Durable credit billing delivery; prompt history remains a best-effort background notification.
"""

import os
import threading
import logging

import requests

logger = logging.getLogger(__name__)

BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


def _payload(user_id, input_chars, output_chars, operation_type, image_count, provider, model,
             duration_ms, success, error_code, images, design=None):
    body = {
        "user_id": user_id,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "operation_type": operation_type,
        "image_count": image_count,
        "provider": provider,
        "model": model,
        "duration_ms": duration_ms,
        "success": success,
        "error_code": error_code,
    }
    if images:
        body["images"] = images  # which model drew each image, and what it was priced at
    if design:
        body["design"] = design  # Mode 8 flat price: {"sub_mode", "credits_scaled"}, charged instead of the chars
    return body


_billing_delivery = None


def start_billing_delivery(redis_client):
    """Each app worker drains the shared durable queue; leases prevent concurrent delivery."""
    global _billing_delivery
    from config import settings
    from .billing_delivery import BillingDelivery
    _billing_delivery = BillingDelivery(redis_client, BACKEND_WEBHOOK_URL, WEBHOOK_SECRET,
        allow_retries=settings.billing_webhook_idempotent)
    _billing_delivery.start()


def stop_billing_delivery():
    global _billing_delivery
    if _billing_delivery is not None:
        _billing_delivery.stop()
        _billing_delivery = None


def _queue_charge(body, *, reserved_scaled=0, settle_delay=0, event_id=None):
    from .credit_manager import AUTOCOMPLETE_CREDITS_SCALED, IMAGE_CREDITS_SCALED, text_credits_scaled
    if _billing_delivery is None:
        raise RuntimeError('Durable billing delivery is unavailable')
    hold = 0
    if body['success']:
        if body.get('design'):
            hold = body['design']['credits_scaled']
        else:
            if body['operation_type'] == 'autocomplete':
                hold = AUTOCOMPLETE_CREDITS_SCALED
            elif body['operation_type'] != 'image':
                hold = text_credits_scaled(body['input_chars'], body['output_chars'])
        hold += (sum(i['credits_scaled'] for i in body['images']) if body.get('images')
                 else body['image_count'] * IMAGE_CREDITS_SCALED)
        hold = max(hold, reserved_scaled)
    return _billing_delivery.enqueue(body, event_id=event_id, reserved_scaled=reserved_scaled,
        hold_scaled=hold, settle_delay=settle_delay)


def fire_credit_deduction(
    user_id: str, input_chars: int, output_chars: int, operation_type: str = "text", image_count: int = 0,
    *, provider=None, model=None, duration_ms=None, success=True, error_code=None, images=None, design=None,
    billing_event_id=None,
):
    """Persist usage before returning; transport retries keep the same Idempotency-Key."""
    return _queue_charge(_payload(user_id, input_chars, output_chars, operation_type, image_count, provider,
        model, duration_ms, success, error_code, images, design), event_id=billing_event_id)


def fire_credit_deduction_and_release(
    user_id: str, input_chars: int, output_chars: int, operation_type: str, image_count: int,
    reserved_scaled: int, credit_manager, settle_delay_seconds: float,
    *, provider=None, model=None, duration_ms=None, success=True, error_code=None, images=None, design=None,
    billing_event_id=None,
):
    """Atomically transfer the active reservation to a durable billing hold until acknowledged."""
    return _queue_charge(_payload(user_id, input_chars, output_chars, operation_type, image_count, provider,
        model, duration_ms, success, error_code, images, design), reserved_scaled=reserved_scaled,
        settle_delay=settle_delay_seconds, event_id=billing_event_id)


def fire_prompt_history_save(user_id: str, prompt: str):
    """
    Send a non-blocking webhook to save the user's prompt to history.
    Uses the same base URL as credit deduction but hits /prompt-history.

    Args:
        user_id: The user ID
        prompt: The raw prompt text to save
    """
    if not BACKEND_WEBHOOK_URL or not WEBHOOK_SECRET:
        logger.warning("Webhook env vars missing — skipping prompt history save")
        return

    prompt_history_url = BACKEND_WEBHOOK_URL.replace("/credit-deduction", "/prompt-history")

    def _send():
        try:
            resp = requests.post(
                prompt_history_url,
                json={
                    "user_id": user_id,
                    "prompt": prompt,
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Secret": WEBHOOK_SECRET,
                },
                timeout=10,
            )
            logger.info(f"Prompt history webhook sent for user={user_id}: status={resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Backend webhook unreachable at {prompt_history_url} "
                f"(Connection refused — backend web app is not running on port 3000). "
                f"Skipping prompt history save for user={user_id}."
            )
        except Exception as e:
            logger.error(f"Prompt history webhook failed for user={user_id}: {e}")

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
