"""
Run the API on this machine with every external side effect replaced, and verbose logs.

  python scripts/run_local_server.py            # http://127.0.0.1:8000

REAL (so the run exercises the production code path):
  routing, request planner, agents, LLM calls, web tools, image providers, image format conversion,
  credit reservation logic, request/response logging, latency tracing.

REPLACED (nothing outside this machine is touched):
  Redis / licence check   -> any licence key is accepted as user "local-dev-user", default provider keys from .env
  Follow-up image memory  -> in-memory (the "previous image" a follow-up refers to); the real Redis is never contacted
  CreditManager           -> in-memory balance (starts at 1000 credits), every reserve/release is logged
  Backend credit webhook  -> NOT sent; the payload it would carry is logged ("WEBHOOK would send")
  S3 upload               -> written to local_run/images and served at /local-images/<file>

Logs: terminal, plus local_run/requests.jsonl (one JSON summary per request). Request and response bodies
are logged (api keys redacted); GET /agent/metrics shows recent requests and latency percentiles;
GET /local/credits shows the in-memory credit ledger.
"""

import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

RUN_DIR = ROOT / "local_run"
IMG_DIR = RUN_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

HOST, PORT = "127.0.0.1", int(os.environ.get("LOCAL_PORT", "8000"))

# Environment BEFORE config is imported. Setting these explicitly beats .env (dotenv never overrides).
os.environ.update({
    "LOG_REQUEST_BODIES": "true",
    "LOG_BODY_MAX_CHARS": "1500",
    "REQUEST_LOG_FILE": str(RUN_DIR / "requests.jsonl"),
    "METRICS_ENDPOINT_ENABLED": "true",
    "BACKEND_WEBHOOK_URL": "",      # hard block: nothing can reach the real backend
    "WEBHOOK_SECRET": "",
    "CREDIT_PRECHECK_ENABLED": "true",
    "CREDIT_SETTLE_DELAY_SECONDS": "0",
    "BLOG_AI_ENABLED": "true",      # type=generate_blog; the flag ships off, a local run is where it is tried
    "VOICE_FLOW_ENABLED": "true",   # /v1/voice/*: local Whisper dictation; open http://127.0.0.1:8000/v1/voice/demo
    "LONGFORM_REFINE_ENABLED": "true",   # long documents through enhance / grammar / tone / simplify, verified chunk by chunk
    # Reference images must live under CDN_URL. Locally the "CDN" is this server's /local-images folder,
    # so an image_ready url can be sent straight back as image_urls for a refinement.
    "CDN_URL": f"http://127.0.0.1:{os.environ.get('LOCAL_PORT', '8000')}/local-images",
})

import warnings
warnings.filterwarnings("ignore")

import api                                   # noqa: E402  (configures logging)
import src.image_gen as image_gen            # noqa: E402
from src import webhook                      # noqa: E402
from src.credit_manager import text_credits_scaled, CREDIT_SCALE   # noqa: E402
from src.redis_manager import RedisManager   # noqa: E402
from src.s3_service import convert_image_format, sniff_image_bytes  # noqa: E402

log = logging.getLogger("local")
LOCAL_USER = "local-dev-user"
START_BALANCE_SCALED = 1000 * CREDIT_SCALE


class LocalDevRedis(RedisManager):
    """No network: accepts any licence. Inherits the real default-key resolution."""

    def __init__(self):            # deliberately no super().__init__(): it would connect to the real Redis
        self.redis_client = None

    def get_cached_user_id(self, license_key):  return LOCAL_USER
    def set_cached_user_id(self, *a, **k):      return None
    def evict_cached_user_id(self, *a, **k):    return None
    def update_heartbeat(self, *a, **k):        return True
    def close(self):                            return None

    def validate_license(self, license_key):
        return {"valid": True, "user_id": LOCAL_USER, "license": {
            "subscription_status": "active", "requires_user_api_keys": False}}

    def get_user_api_keys(self, user_id):
        return {}                  # no personal keys: platform default keys from .env are used


class FakeRedisClient:
    """The few Redis commands the image follow-up memory uses, held in memory (with expiry)."""

    def __init__(self):
        self._data = {}

    def set(self, key, value, ex=None):
        self._data[key] = (value, time.time() + ex if ex else None)
        log.info(f"🧠 image memory SET {key} (ttl={ex}s)")
        return True

    def get(self, key):
        item = self._data.get(key)
        if not item or (item[1] and item[1] < time.time()):
            log.info(f"🧠 image memory GET {key} -> miss")
            return None
        log.info(f"🧠 image memory GET {key} -> hit")
        return item[0]


class FakeCreditManager:
    """In-memory stand-in for CreditManager with a visible ledger."""

    def __init__(self, _client=None):
        self.balance = START_BALANCE_SCALED
        self.reserved = 0
        self.ledger = []
        self._lock = threading.Lock()

    def _note(self, kind, amount):
        self.ledger.append({"t": round(time.time(), 2), "kind": kind, "scaled": amount,
                            "reserved": self.reserved, "balance": self.balance})
        log.info(f"💳 {kind:<8} {amount:>5} scaled ({amount / CREDIT_SCALE:g} cr) | reserved={self.reserved / CREDIT_SCALE:g} "
                 f"balance={self.balance / CREDIT_SCALE:g} available={(self.balance - self.reserved) / CREDIT_SCALE:g}")

    def reserve(self, user_id, scaled):
        if scaled <= 0:
            return True, 0
        with self._lock:
            if self.balance - self.reserved < scaled:
                self._note("DENIED", scaled)
                return False, self.balance - self.reserved
            self.reserved += scaled
            self._note("reserve", scaled)
            return True, self.balance - self.reserved

    def release(self, user_id, scaled):
        with self._lock:
            self.reserved = max(0, self.reserved - scaled)
            self._note("release", scaled)
            return self.reserved

    def schedule_release(self, user_id, scaled, delay_seconds=0):
        threading.Timer(max(0.0, delay_seconds), self.release, args=(user_id, scaled)).start()

    def peek_available(self, user_id):
        return self.balance - self.reserved


def _simulated_backend_deduct(cm, payload):
    """
    What the real backend would charge: text by characters, images by count (flat 50 until it reads images[]).
    A delivered Mode 8 design carries its flat price in `design`; that replaces the char-based text charge here, as it
    will in the backend once it reads `design` (until then the real backend still charges the characters).
    """
    if not cm:
        return
    text = text_credits_scaled(payload["input_chars"], payload["output_chars"]) if payload["operation_type"] in ("text", "mixed") else 0
    if payload.get("design"):
        log.info(f"🎨 design price {payload['design']['credits_scaled'] / CREDIT_SCALE:g} cr replaces the text charge "
                 f"({text / CREDIT_SCALE:g} cr) — the real backend charges the text until it reads `design`")
        text = payload["design"]["credits_scaled"]
    images = sum(i.get("credits_scaled", 0) for i in payload.get("images") or [])
    with cm._lock:
        cm.balance -= (text + images)
        cm._note("DEDUCT*", text + images)


def make_webhooks(cm_holder):
    def fire_credit_deduction(user_id, input_chars, output_chars, operation_type="text", image_count=0, *,
                              provider=None, model=None, duration_ms=None, success=True, error_code=None, images=None,
                              design=None):
        body = webhook._payload(user_id, input_chars, output_chars, operation_type, image_count, provider, model,
                                duration_ms, success, error_code, images, design)
        log.info("📨 WEBHOOK would send (NOT sent): " + json.dumps(body, ensure_ascii=False))
        if success:
            _simulated_backend_deduct(cm_holder.get("cm"), body)

    def fire_credit_deduction_and_release(user_id, input_chars, output_chars, operation_type, image_count,
                                          reserved_scaled, credit_manager, settle_delay_seconds, *, provider=None,
                                          model=None, duration_ms=None, success=True, error_code=None, images=None,
                                          design=None):
        fire_credit_deduction(user_id, input_chars, output_chars, operation_type, image_count, provider=provider,
                              model=model, duration_ms=duration_ms, success=success, error_code=error_code, images=images,
                              design=design)
        credit_manager.release(user_id, reserved_scaled)

    return fire_credit_deduction, fire_credit_deduction_and_release


def local_upload(image_bytes, content_type=None, user_id=None, request_id=None, target_format=None):
    """Same sniff + format conversion as the S3 upload; the network PUT becomes a disk write."""
    from PIL import Image
    if target_format:
        final_bytes, final_type, final_ext = convert_image_format(image_bytes, target_format)
    else:
        final_type, final_ext = sniff_image_bytes(image_bytes)
        final_bytes = image_bytes
    name = f"{request_id or 'req'}_{uuid.uuid4().hex[:6]}.{final_ext}"
    (IMG_DIR / name).write_bytes(final_bytes)
    with Image.open(io.BytesIO(final_bytes)) as im:
        w, h = im.size
    return {"cdn_url": f"http://{HOST}:{PORT}/local-images/{name}", "width": w, "height": h,
            "format": final_ext, "content_type": final_type}


def install():
    holder = {}

    def make_cm(_client):
        holder["cm"] = FakeCreditManager()
        return holder["cm"]

    api.get_redis_manager = lambda: LocalDevRedis()
    # image_gen reaches Redis through the process-wide singleton, not through api: replace that too
    import src.redis_manager as redis_module
    from types import SimpleNamespace
    fake_redis = SimpleNamespace(redis_client=FakeRedisClient())
    redis_module.get_redis_manager = lambda: fake_redis
    api.CreditManager = make_cm
    fire, fire_release = make_webhooks(holder)
    api.fire_credit_deduction = fire
    api.fire_credit_deduction_and_release = fire_release
    api.fire_prompt_history_save = lambda user_id, prompt: log.info("📝 prompt-history save skipped (local)")
    image_gen.upload_image_from_bytes_detailed = local_upload

    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    api.app.mount("/local-images", StaticFiles(directory=str(IMG_DIR)), name="local-images")

    async def credits():
        cm = holder.get("cm")
        return JSONResponse({"balance_credits": cm.balance / CREDIT_SCALE, "reserved_credits": cm.reserved / CREDIT_SCALE,
                             "ledger": cm.ledger[-40:]} if cm else {"error": "not started"})
    api.app.add_api_route("/local/credits", credits, methods=["GET"])


if __name__ == "__main__":
    install()
    import uvicorn
    print(__doc__)
    print(f"\n>>> http://{HOST}:{PORT}   (images: {IMG_DIR})\n", flush=True)
    uvicorn.run(api.app, host=HOST, port=PORT, log_level="info")
