import asyncio
import base64
import socket
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from unittest.mock import call

import httpx

from src.redis_manager import RedisManager
from src.text_processing_agent import (
    _DNS_CACHE,
    _MODEL_CACHE,
    _TOOL_EXECUTOR,
    _PinnedDNSBackend,
    _client_cache,
    _client_lock,
    _get_cached_client,
    _is_publicly_routable,
    _resolve_public_addresses,
    create_image_tool,
    get_shared_web_client,
    close_shared_web_client,
    get_model,
    prewarm_models,
)
from api import (
    AbortRequest,
    _active_cancellations,
    _cancel_registry_lock,
    abort_request,
    async_validate_and_get_api_key,
    register_cancellation_event,
    cancel_request_by_id,
    unregister_cancellation_event,
    cleanup_stale_cancellations,
)


class TestLatencyOptimizations(unittest.TestCase):
    """Unit tests verifying all 8 latency bottleneck fixes."""

    # -----------------------------------------------------------------------
    # Point 1: Parallel Redis Calls & Cache
    # -----------------------------------------------------------------------
    def test_point_1_user_id_cache_and_parallel_resolution(self):
        """Verify license_user_cache and parallel Redis resolution."""
        license_key = "lic_test_123"
        rm = RedisManager.__new__(RedisManager)
        rm._license_user_cache = {}
        rm._cache_lock = threading.Lock()

        rm.evict_cached_user_id(license_key)
        self.assertIsNone(rm.get_cached_user_id(license_key))

        rm.set_cached_user_id(license_key, "usr_999")
        self.assertEqual(rm.get_cached_user_id(license_key), "usr_999")

        # Test in-memory key resolution
        user_keys = {"use_own_keys": True, "openai_api_key": "sk-user-custom-openai"}
        self.assertEqual(rm.resolve_api_key_from_user_keys(user_keys, "openai"), "sk-user-custom-openai")
        self.assertIsNone(rm.resolve_api_key_from_user_keys(user_keys, "claude"))
        deepseek_keys = {"use_own_keys": True, "deepseek_api_key": "sk-deepseek"}
        self.assertEqual(rm.resolve_api_key_from_user_keys(deepseek_keys, "deepseek"), "sk-deepseek")

        # Test async_validate_and_get_api_key executes with parallel gather
        mock_rm = MagicMock(spec=RedisManager)
        mock_rm.get_cached_user_id.return_value = "usr_999"
        mock_rm.validate_license.return_value = {
            "valid": True,
            "user_id": "usr_999",
            "license": {
                "tier": "pro",
                "requires_user_api_keys": True,
                "subscription_status": "active",
            },
        }
        mock_rm.get_user_api_keys.return_value = user_keys
        mock_rm.resolve_api_key_from_user_keys.return_value = "sk-user-custom-openai"

        with patch("api.redis_manager", mock_rm):
            loop = asyncio.new_event_loop()
            try:
                bg = MagicMock()
                key, meta = loop.run_until_complete(
                    async_validate_and_get_api_key(
                        license_key, "openai", "req_test", bg
                    )
                )
                self.assertEqual(key, "sk-user-custom-openai")
                self.assertEqual(meta["user_id"], "usr_999")
                self.assertTrue(meta["requires_user_api_keys"])
                # Ensure both methods were called
                mock_rm.validate_license.assert_called_once_with(license_key)
                mock_rm.get_user_api_keys.assert_called_once_with("usr_999")
                mock_rm.resolve_api_key_from_user_keys.assert_called_once_with(
                    user_keys=user_keys,
                    model_provider="openai",
                    default_keys=None,
                )
            finally:
                loop.close()

    def test_point_1_redis_outage_is_service_unavailable(self):
        mock_rm = MagicMock(spec=RedisManager)
        mock_rm.get_cached_user_id.return_value = None
        mock_rm.validate_license.return_value = {
            "valid": False,
            "code": "LICENSE_SERVICE_UNAVAILABLE",
            "error": "Database error occurred",
        }
        with patch("api.redis_manager", mock_rm):
            with self.assertRaises(Exception) as raised:
                asyncio.run(async_validate_and_get_api_key("lic", "openai", "req"))
        self.assertEqual(raised.exception.status_code, 503)

    def test_point_1_warm_cache_calls_run_concurrently(self):
        barrier = threading.Barrier(2)
        mock_rm = MagicMock(spec=RedisManager)
        mock_rm.get_cached_user_id.return_value = "usr_999"

        def validate(_):
            barrier.wait(timeout=1)
            return {
                "valid": True,
                "user_id": "usr_999",
                "license": {"requires_user_api_keys": True},
            }

        def keys(_):
            barrier.wait(timeout=1)
            return {"use_own_keys": True, "openai_api_key": "user-key"}

        mock_rm.validate_license.side_effect = validate
        mock_rm.get_user_api_keys.side_effect = keys
        mock_rm.resolve_api_key_from_user_keys.return_value = "user-key"
        with patch("api.redis_manager", mock_rm):
            key, _ = asyncio.run(
                async_validate_and_get_api_key("lic", "openai", "req", MagicMock())
            )
        self.assertEqual(key, "user-key")

    def test_point_1_key_store_outage_is_service_unavailable(self):
        mock_rm = MagicMock(spec=RedisManager)
        mock_rm.get_cached_user_id.return_value = None
        mock_rm.validate_license.return_value = {
            "valid": True,
            "user_id": "usr_999",
            "license": {"requires_user_api_keys": True},
        }
        mock_rm.get_user_api_keys.return_value = {
            "use_own_keys": False,
            "_error_code": "LICENSE_SERVICE_UNAVAILABLE",
        }
        with patch("api.redis_manager", mock_rm):
            with self.assertRaises(Exception) as raised:
                asyncio.run(async_validate_and_get_api_key("lic", "openai", "req"))
        self.assertEqual(raised.exception.status_code, 503)

    def test_abort_reports_redis_outage(self):
        mock_rm = MagicMock(spec=RedisManager)
        mock_rm.validate_license.return_value = {
            "valid": False,
            "code": "LICENSE_SERVICE_UNAVAILABLE",
            "error": "Database error occurred",
        }
        with patch("api.redis_manager", mock_rm):
            with self.assertRaises(Exception) as raised:
                asyncio.run(abort_request(AbortRequest(request_id="req", api_key="lic")))
        self.assertEqual(raised.exception.status_code, 503)

    # -----------------------------------------------------------------------
    # Point 2: Shared Persistent httpx.Client with Connection Pooling
    # -----------------------------------------------------------------------
    def test_point_2_shared_httpx_client(self):
        """Verify persistent httpx.Client singleton and pooling config."""
        close_shared_web_client()
        client1 = get_shared_web_client()
        client2 = get_shared_web_client()
        self.assertIs(client1, client2, "get_shared_web_client must return the same singleton instance")

        # Check limits configured for high-concurrency reuse
        self.assertEqual(client1._transport._pool._max_keepalive_connections, 20)
        self.assertEqual(client1._transport._pool._max_connections, 50)
        client1.cookies.set("session", "must-not-persist", domain="example.com")
        self.assertEqual(list(client1.cookies.jar), [])

        # Test closing
        close_shared_web_client()
        client3 = get_shared_web_client()
        self.assertIsNot(client1, client3, "New client should be created after close")
        close_shared_web_client()

    # -----------------------------------------------------------------------
    # Point 3: DNS Caching in _is_publicly_routable
    # -----------------------------------------------------------------------
    def test_point_3_dns_caching(self):
        """Verify hostname routability is cached, preventing repeated getaddrinfo calls."""
        test_host = "test-cached-domain.example.com"
        test_url = f"https://{test_host}/path/to/resource"

        # Clear test host from cache if present
        _DNS_CACHE.pop(test_host, None)

        with patch("socket.getaddrinfo") as mock_getaddr:
            mock_getaddr.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ]

            # First call -> should invoke socket.getaddrinfo
            result1 = _is_publicly_routable(test_url)
            self.assertTrue(result1)
            self.assertEqual(mock_getaddr.call_count, 1)

            # Second call -> must hit _DNS_CACHE without calling socket.getaddrinfo
            result2 = _is_publicly_routable(test_url)
            self.assertTrue(result2)
            self.assertEqual(mock_getaddr.call_count, 1, "socket.getaddrinfo must not be called again for cached host")

            # Check cache structure
            self.assertIn(test_host, _DNS_CACHE)
            cached_addresses, ts = _DNS_CACHE[test_host]
            self.assertEqual(cached_addresses, ("93.184.216.34",))
            self.assertGreater(ts, 0)

    def test_point_3_transport_pins_checked_ip_and_blocks_private_hosts(self):
        backend = _PinnedDNSBackend()
        with patch(
            "src.text_processing_agent._resolve_public_addresses",
            return_value=("93.184.216.34",),
        ), patch.object(socket, "create_connection") as connect:
            fake_socket = MagicMock()
            connect.return_value = fake_socket
            backend.connect_tcp("public.example", 443, timeout=1)
            connect.assert_called_once_with(
                ("93.184.216.34", 443), 1, source_address=None
            )

        with patch("src.text_processing_agent._resolve_public_addresses", return_value=()):
            with self.assertRaises(Exception):
                backend.connect_tcp("169.254.169.254", 80)

    # -----------------------------------------------------------------------
    # Point 4: Agno Model Caching & Pre-warming
    # -----------------------------------------------------------------------
    def test_point_4_model_caching(self):
        """Verify model instances are cached in _MODEL_CACHE and reused."""
        fake_keys = {
            "openai": "sk-fake-openai-key-for-test",
            "anthropic": "sk-fake-anthropic-key-for-test",
        }
        # Prewarm
        count = prewarm_models(fake_keys)
        self.assertGreaterEqual(count, 1)

        # Fetching model should hit cache
        start = time.perf_counter()
        model_instance = get_model("openai", fake_keys["openai"])
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(model_instance)
        # Cache hit should be fast (< 5ms)
        self.assertLess(elapsed, 0.005, f"Cached model retrieval took {elapsed*1000:.3f}ms, expected < 5ms")

        # Fetching again returns identical instance
        model_instance2 = get_model("openai", fake_keys["openai"])
        self.assertIs(model_instance, model_instance2)

        before = len(_MODEL_CACHE)
        uncached1 = get_model("openai", "byok-secret", cache_model=False)
        uncached2 = get_model("openai", "byok-secret", cache_model=False)
        self.assertIsNot(uncached1, uncached2)
        self.assertEqual(len(_MODEL_CACHE), before)

    def test_autocomplete_client_cache_is_bounded_and_closes_evictions(self):
        first = MagicMock()
        second = MagicMock()
        with _client_lock:
            _client_cache.clear()
        with patch("src.text_processing_agent._CLIENT_CACHE_MAX_SIZE", 1):
            self.assertIs(_get_cached_client(("test", "one"), lambda: first), first)
            self.assertIs(_get_cached_client(("test", "two"), lambda: second), second)
        first.close.assert_called_once_with()
        with _client_lock:
            _client_cache.clear()

    # -----------------------------------------------------------------------
    # Point 5: Shared Module-Level ThreadPoolExecutor
    # -----------------------------------------------------------------------
    def test_point_5_shared_tool_executor(self):
        """Verify _TOOL_EXECUTOR is a shared persistent pool with workers."""
        self.assertIsNotNone(_TOOL_EXECUTOR)
        self.assertEqual(_TOOL_EXECUTOR._max_workers, 16)
        future = _TOOL_EXECUTOR.submit(lambda x: x * 2, 21)
        self.assertEqual(future.result(), 42)

    # -----------------------------------------------------------------------
    # Point 6: Lock-Free Cancellation Registry
    # -----------------------------------------------------------------------
    def test_point_6_lock_free_cancellation(self):
        """Verify per-request threading.Event cancellation."""
        req_id = "test_cancel_req_001"
        event = register_cancellation_event(req_id)
        self.assertFalse(event.is_set())

        # Test pre-cancel idempotency
        cancel_request_by_id(req_id)
        self.assertTrue(event.is_set())

        # Abort called before register
        pre_req_id = "test_cancel_pre_002"
        cancel_request_by_id(pre_req_id)
        event2 = register_cancellation_event(pre_req_id)
        self.assertTrue(event2.is_set(), "Event must remain set if aborted before registration")

        # Cleanup
        unregister_cancellation_event(req_id)
        unregister_cancellation_event(pre_req_id)

    def test_point_6_active_stream_is_not_expired(self):
        req_id = "test_long_stream"
        event = register_cancellation_event(req_id)
        with _cancel_registry_lock:
            _active_cancellations[req_id] = (event, None)
        cleanup_stale_cancellations()
        cancel_request_by_id(req_id)
        self.assertTrue(event.is_set())
        unregister_cancellation_event(req_id)

    # -----------------------------------------------------------------------
    # Point 7: Immediate Image Completion Notification
    # -----------------------------------------------------------------------
    def test_point_7_image_completion_event_notification(self):
        """Verify image worker sets completion_event so api wakes up immediately."""
        completion_event = threading.Event()
        pending_images = {
            "__completion_callback__": completion_event.set,
        }
        response = MagicMock()
        response.data = [MagicMock(b64_json=base64.b64encode(b"image").decode())]
        fake_client = MagicMock()
        fake_client.images.generate.return_value = response
        with patch("openai.OpenAI", return_value=fake_client), patch(
            "src.s3_service.upload_image_from_bytes",
            return_value="https://cdn.example.com/test.png",
        ):
            create_image_tool("key", pending_images)("test prompt")
            self.assertTrue(completion_event.wait(timeout=2.0))

        images = {k: v for k, v in pending_images.items() if not k.startswith("__")}
        self.assertEqual(len(images), 1)
        self.assertEqual(next(iter(images.values()))["status"], "done")

    # -----------------------------------------------------------------------
    # Point 8: Redis Retry Recovery Window
    # -----------------------------------------------------------------------
    def test_point_8_redis_retry_bounded_backoff(self):
        """Verify retries cover a 500ms transient outage window."""
        import redis
        mock_client = MagicMock()
        mock_client.ping.return_value = "PONG"

        op = MagicMock()
        op.side_effect = [
            redis.ConnectionError("Redis connection error"),
            redis.ConnectionError("Redis connection error"),
            "OPERATION_SUCCESS",
        ]

        rm = RedisManager.__new__(RedisManager)
        rm.redis_client = mock_client

        with patch("src.redis_manager.time.sleep") as sleep:
            res = rm._execute_with_retry(op)

        self.assertEqual(res, "OPERATION_SUCCESS")
        self.assertEqual(op.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.1), call(0.4)])


if __name__ == "__main__":
    unittest.main()
