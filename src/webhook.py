"""
Webhook module for notifying the backend about credit deductions.
Uses fire-and-forget pattern (daemon thread) so streaming responses aren't delayed.
"""

import os
import threading
import time
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


def fire_credit_deduction(
    user_id: str,
    input_chars: int,
    output_chars: int,
    operation_type: str = "text",
    image_count: int = 0,
    *,
    provider: str | None = None,
    model: str | None = None,
    duration_ms: int | None = None,
    success: bool = True,
    error_code: str | None = None,
    images: list | None = None,
    design: dict | None = None,
):
    """
    Send a non-blocking credit deduction webhook to the backend.

    Args:
        user_id: The user ID to deduct credits for
        input_chars: Number of input characters
        output_chars: Number of output characters
        operation_type: Type of AI operation ("text", "image", "mixed", "autocomplete")
        image_count: Number of successfully generated images (for image/mixed operations)
        provider: AI provider used ('openai'|'claude'|'grok'|'gemini'|'deepseek') — for admin analytics
        model: Specific model name — for admin analytics (optional)
        duration_ms: Total request duration in ms — for admin analytics
        success: Whether the call succeeded; failures are logged but don't deduct credits
        error_code: Short error identifier when success=False
        images: per generated image {"provider", "model", "credits_scaled"}; sent only when there are images
        design: a delivered Mode 8 design {"sub_mode", "credits_scaled"}; its flat price replaces the char-based charge
    """
    if not BACKEND_WEBHOOK_URL or not WEBHOOK_SECRET:
        logger.warning("Webhook env vars missing (BACKEND_WEBHOOK_URL or WEBHOOK_SECRET) — skipping credit deduction")
        return

    def _send():
        try:
            resp = requests.post(
                BACKEND_WEBHOOK_URL,
                json=_payload(user_id, input_chars, output_chars, operation_type, image_count, provider,
                              model, duration_ms, success, error_code, images, design),
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Secret": WEBHOOK_SECRET,
                },
                timeout=10,
            )
            logger.info(f"Credit deduction webhook sent for user={user_id}: status={resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Backend webhook unreachable at {BACKEND_WEBHOOK_URL} "
                f"(Connection refused — backend web app is not running on port 3000). "
                f"Skipping credit deduction for user={user_id}."
            )
        except Exception as e:
            logger.error(f"Credit deduction webhook failed for user={user_id}: {e}")

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()


def fire_credit_deduction_and_release(
    user_id: str,
    input_chars: int,
    output_chars: int,
    operation_type: str,
    image_count: int,
    reserved_scaled: int,
    credit_manager,
    settle_delay_seconds: float,
    *,
    provider: str | None = None,
    model: str | None = None,
    duration_ms: int | None = None,
    success: bool = True,
    error_code: str | None = None,
    images: list | None = None,
    design: dict | None = None,
):
    """
    Streaming variant of fire_credit_deduction. POSTs the webhook, sleeps
    `settle_delay_seconds` so the backend's DB update flows through the
    Supabase edge function back into subscription:{userId}, then releases
    the in-flight reservation. Runs in a single daemon thread so the SSE
    response sees zero added latency.

    If webhook env vars aren't configured we still release the reservation
    so credits don't leak.

    Telemetry kwargs (provider/model/duration_ms/success/error_code) are
    forwarded to the backend for the admin AI Analytics dashboard.
    """
    if not BACKEND_WEBHOOK_URL or not WEBHOOK_SECRET:
        logger.warning(
            "Webhook env vars missing — releasing reservation without posting webhook"
        )
        credit_manager.schedule_release(user_id, reserved_scaled, delay_seconds=0)
        return

    def _send_then_release():
        try:
            resp = requests.post(
                BACKEND_WEBHOOK_URL,
                json=_payload(user_id, input_chars, output_chars, operation_type, image_count, provider,
                              model, duration_ms, success, error_code, images, design),
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Secret": WEBHOOK_SECRET,
                },
                timeout=10,
            )
            logger.info(
                f"Credit deduction webhook sent for user={user_id}: status={resp.status_code}"
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Backend webhook unreachable at {BACKEND_WEBHOOK_URL} "
                f"(Connection refused — backend web app is not running on port 3000). "
                f"Credit reservation will still be released for user={user_id}."
            )
        except Exception as e:
            logger.error(f"Credit deduction webhook failed for user={user_id}: {e}")
        finally:
            if settle_delay_seconds > 0:
                time.sleep(settle_delay_seconds)
            try:
                credit_manager.release(user_id, reserved_scaled)
            except Exception as e:
                logger.error(f"Reservation release failed for user={user_id}: {e}")

    threading.Thread(target=_send_then_release, daemon=True).start()


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
