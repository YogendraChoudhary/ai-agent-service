"""Shared pytest guards."""

import pytest


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
