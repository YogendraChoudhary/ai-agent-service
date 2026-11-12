"""Shared pytest guards."""

import os

import pytest

# Voice is off by default in production config; the suite exercises the mounted routes.
# Set before any test module imports config, so the app is built with the router.
os.environ.setdefault("VOICE_FLOW_ENABLED", "true")
# The startup model check makes (free) network calls with the real keys in .env; tests never do.
os.environ.setdefault("MODEL_PREFLIGHT_ON_START", "false")


@pytest.fixture(autouse=True)
def _fresh_provider_health():
    """The rollout health counters are process-wide; one test's simulated failures must not alarm the next."""
    from src.provider_health import HEALTH
    HEALTH.reset()
    yield
    HEALTH.reset()


@pytest.fixture(autouse=True)
def _never_touch_real_redis(monkeypatch):
    """
    The project's .env points at a real Redis. Tests must never write to it, so the
    singleton accessor refuses to connect unless a test patches it with a fake.
    The retry cooldown is reset so one test's simulated outage cannot affect the next.
    """
    import src.image_gen as ig
    import src.redis_manager as rm

    def _blocked(*args, **kwargs):
        raise RuntimeError("Real Redis is disabled in tests")

    monkeypatch.setattr(rm, "get_redis_manager", _blocked)
    monkeypatch.setattr(ig, "_REDIS_RETRY_AFTER", 0.0)
