import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from agno.agent import RunEvent
from config import settings
from src.task_router import KeyResolver
from src.provider_errors import is_retryable, retry_delay
import src.text_processing_agent as tpa


def content(text):
    return SimpleNamespace(event=RunEvent.run_content, content=text)


def error(text):
    return SimpleNamespace(event=RunEvent.run_error, content=text)


class TestProviderFallback(unittest.TestCase):
    def setUp(self):
        for name in ('task_routing_enabled', 'provider_fallback_enabled'):
            p = patch.object(settings, name, True)
            p.start()
            self.addCleanup(p.stop)
        self.keys = KeyResolver({}, {'anthropic_api_key': 'a', 'openai_api_key': 'o', 'gemini_api_key': 'g'},
                                lambda **kw: kw['default_keys'].get('anthropic_api_key' if kw['model_provider'] == 'claude' else kw['model_provider'] + '_api_key'))

    def run_stream(self, create, **kw):
        values = dict(model_provider='gemini', api_key='g', force_agent_type='code_writer',
                      model_source='default', key_resolver=self.keys, report_model=True)
        values.update(kw)
        with patch.object(tpa, 'create_agent', side_effect=create), patch.object(tpa.time, 'sleep'):
            return list(tpa.process_user_query_stream('write a Python function', **values))

    def test_429_retries_once_then_switches(self):
        calls = []
        def create(agent_type, provider, key, *a, **kw):
            calls.append((provider, key, kw['plan'].advanced_model))
            events = [error('429 rate limit')] if provider == 'claude' else [content('```python\nx=1\n```')]
            return SimpleNamespace(run=lambda *a, **kw: iter(events))
        out = self.run_stream(create)
        self.assertEqual([p for p, _, _ in calls], ['claude', 'claude', 'openai'])
        self.assertTrue(all(tier for _, _, tier in calls))
        self.assertEqual(out[0]['__model_used__'][0], 'openai')

    def test_no_retry_or_fallback_after_content(self):
        create = Mock(return_value=SimpleNamespace(run=lambda *a, **kw: iter([content('partial'), error('503 overloaded')])))
        with self.assertRaises(RuntimeError):
            self.run_stream(create)
        self.assertEqual(create.call_count, 1)

    def test_auth_and_bad_request_never_fallback(self):
        for failure in ('401 invalid key', '403 permission denied', '400 bad request', 'safety block'):
            create = Mock(return_value=SimpleNamespace(run=lambda *a, **kw: iter([error(failure)])))
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.run_stream(create)
            self.assertEqual(create.call_count, 1)

    def test_nonstreaming_never_switches_provider(self):
        create = Mock(return_value=SimpleNamespace(run=lambda *a, **kw: iter([error('429 rate limit')])))
        with patch.object(tpa, 'create_agent', create), patch.object(tpa.time, 'sleep'), self.assertRaises(RuntimeError):
            tpa.process_user_query_with_images('code', 'gemini', 'g', force_agent_type='code_writer')
        self.assertEqual(create.call_count, 2)
        self.assertTrue(all(c.args[1] == 'gemini' for c in create.call_args_list))

    def test_byok_has_no_platform_candidate(self):
        self.keys = KeyResolver({'gemini_api_key': 'user-g'}, None, lambda **kw: kw['user_keys'].get(kw['model_provider'] + '_api_key'))
        create = Mock(return_value=SimpleNamespace(run=lambda *a, **kw: iter([content('ok')])))
        self.run_stream(create, is_byok=True)
        self.assertEqual(create.call_args.args[1:3], ('gemini', 'user-g'))

    def test_error_shapes_and_retry_after(self):
        import httpx
        for failure in ('HTTP 503', 'overloaded', TimeoutError(), ConnectionResetError(),
                        httpx.ConnectError('failed'), 'Connection error.', 'RESOURCE_EXHAUSTED'):
            self.assertTrue(is_retryable(failure), str(failure))
        err = RuntimeError('429')
        err.response = SimpleNamespace(headers={'retry-after': '30'})
        self.assertEqual(retry_delay(err), 2)
        self.assertEqual(retry_delay(RuntimeError('429 rate limit')), 2)
        err.response.headers['retry-after'] = 'invalid'
        self.assertEqual(retry_delay(err), 2)

    def test_real_sdk_makes_only_two_http_calls_before_fallback(self):
        import httpx
        from agno.agent import Agent
        calls = []
        def fail(request):
            calls.append(request)
            return httpx.Response(429, headers={'retry-after': '0'}, json={
                'error': {'message': 'rate limit', 'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}})
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            def create(agent_type, provider, key, *args, **kwargs):
                if provider != 'openai':
                    return SimpleNamespace(run=lambda *a, **kw: iter([content('ok')]))
                model = tpa.get_model('openai', 'dummy', cache_model=False, managed_retries=True)
                model.http_client = client
                return Agent(model=model, telemetry=False)
            out = self.run_stream(create, model_provider='openai', model_source='user')
        self.assertEqual(len(calls), 2)
        self.assertEqual(out[0]['__model_used__'][0], 'claude')

    def test_retry_policy_is_separate_in_model_cache(self):
        normal = tpa.get_model('openai', 'retry-cache-test')
        managed = tpa.get_model('openai', 'retry-cache-test', managed_retries=True)
        self.assertIsNot(normal, managed)
        self.assertIsNone(normal.client_params)
        self.assertEqual(managed.client_params['max_retries'], 0)
        self.assertIs(tpa.get_model('openai', 'retry-cache-test'), normal)

    def test_all_provider_sdks_have_retries_disabled_for_the_managed_loop(self):
        for provider in ('openai', 'claude', 'grok', 'deepseek', 'gemini'):
            with self.subTest(provider=provider):
                model = tpa.get_model(provider, 'dummy', cache_model=False, managed_retries=True)
                if provider == 'gemini':
                    self.assertEqual(model.client_params['http_options']['retry_options']['attempts'], 1)
                else:
                    self.assertEqual(model.client_params['max_retries'], 0)
