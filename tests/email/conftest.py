"""The email routes take an Eddyter licence key (like /agent/stream); these tests run on a mocked BYOK licence."""
from unittest.mock import AsyncMock

import pytest

import api
from config import settings

BYOK_LICENCE = ("sk-test", {"user_id": "u-email", "requires_user_api_keys": True, "user_keys": {}})


@pytest.fixture(autouse=True)
def _email_licence(monkeypatch, request):
    if request.node.get_closest_marker("real_licence"):
        return
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=BYOK_LICENCE))
    monkeypatch.setattr(settings, "task_routing_enabled", False)
    monkeypatch.setattr(settings, "provider_fallback_enabled", False)
