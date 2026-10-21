import json
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from agno.agent import RunEvent

import api
import src.text_processing_agent as tpa
from config import settings
from src.redis_manager import RedisManager
from src.observability import start_trace, finish_trace

BROKEN = '```python\ndef add(a, b):\nreturn a+b\n```'
FIXED = '```python\ndef add(a, b):\n    return a+b\n```'


class TestCodePipeline(unittest.TestCase):
    def setUp(self):
        for name in ('code_validation_enabled', 'code_repair_enabled'):
            p = patch.object(settings, name, True)
            p.start()
            self.addCleanup(p.stop)

    def agent(self):
        def run(query, stream=True, **kw):
            if stream:
                return iter([SimpleNamespace(event=RunEvent.run_content, content=BROKEN)])
            return SimpleNamespace(content=FIXED)
        return SimpleNamespace(run=Mock(side_effect=run))

    def test_nonstreaming_repairs_and_keeps_all_fences(self):
        agent = self.agent()
        with patch.object(tpa, 'create_agent', return_value=agent), patch.object(settings, 'task_routing_enabled', True):
            response = TestClient(api.app).post('/agent/process', json={
                'type': 'generate_code', 'query': 'Write an add function', 'model': 'gemini',
                'model_source': 'default', 'api_key': 'user-key'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['markdown_response'], FIXED)
        self.assertEqual(agent.run.call_count, 2)

    def test_nonstreaming_failed_validation_returns_422(self):
        def run(query, stream=True, **kw):
            return iter([SimpleNamespace(event=RunEvent.run_content, content=BROKEN)]) if stream else SimpleNamespace(content=BROKEN)
        for repair in (False, True):
            agent = SimpleNamespace(run=Mock(side_effect=run))
            with self.subTest(repair=repair), patch.object(settings, 'code_repair_enabled', repair), \
                    patch.object(tpa, 'create_agent', return_value=agent):
                response = TestClient(api.app).post('/agent/process', json={
                    'type': 'generate_code', 'query': 'add', 'model': 'gemini', 'api_key': 'key'})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn('issues', response.json())
            self.assertNotIn('success', response.json())
            self.assertEqual(agent.run.call_count, 2 if repair else 1)

    def test_stream_validation_logs_without_repair_until_frontend_flag(self):
        agent = self.agent()
        with patch.object(tpa, 'create_agent', return_value=agent):
            out = list(tpa.process_user_query_stream('add', 'gemini', 'g', force_agent_type='code_writer'))
        self.assertEqual(out, [BROKEN])
        self.assertEqual(agent.run.call_count, 1)

    def test_explanation_preserves_closing_code_fence(self):
        explanation = 'This function adds two values.\n' + FIXED
        events = [SimpleNamespace(event=RunEvent.run_content, content=explanation)]
        with patch.object(tpa, 'create_agent', return_value=SimpleNamespace(run=lambda *a, **kw: iter(events))):
            response = TestClient(api.app).post('/agent/process', json={
                'type': 'explain_code', 'query': 'Explain add', 'model': 'gemini', 'api_key': 'key'})
        self.assertEqual(response.json()['markdown_response'], explanation)

    def test_stream_replacement_precedes_done_and_bills_actual_provider(self):
        agent = self.agent()
        meta = {'user_id': 'u', 'requires_user_api_keys': False, 'user_keys': {}}
        resolver = RedisManager.resolve_api_key_from_user_keys.__get__(object())
        manager = SimpleNamespace(resolve_api_key_from_user_keys=resolver)
        with patch.object(settings, 'task_routing_enabled', True), \
                patch.object(settings, 'code_repair_stream_event', True), \
                patch.object(api, 'redis_manager', manager), \
                patch.dict(api.DEFAULT_API_KEYS, {'anthropic_api_key': 'claude-key'}), \
                patch.object(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('g', meta))), \
                patch.object(api, 'CREDIT_PRECHECK_ENABLED', False), \
                patch.object(api, 'fire_credit_deduction') as bill, \
                patch.object(tpa, 'create_agent', return_value=agent) as create:
            response = TestClient(api.app).post('/agent/stream', json={
                'type': 'generate_code', 'query': 'add', 'model': 'gemini', 'model_source': 'default', 'api_key': 'lic'})
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        types = [e['type'] for e in events]
        self.assertLess(types.index('content_replace'), types.index('done'))
        self.assertEqual(next(e['content'] for e in events if e['type'] == 'content_replace'), FIXED)
        self.assertEqual(create.call_args.args[1:3], ('claude', 'claude-key'))
        self.assertEqual(bill.call_args.kwargs['provider'], 'claude')
        self.assertEqual(bill.call_args.kwargs['model'], tpa.ADVANCED_MODEL_IDS['claude'])

    def test_code_agent_has_web_but_never_image_tools_or_smart_instructions(self):
        from src.request_plan import RequestPlan
        with patch.object(tpa, 'get_model', return_value=Mock()), patch.object(tpa, 'Agent') as agent:
            tpa.create_agent('code_writer', 'claude', 'key', plan=RequestPlan(use_web=True))
        values = agent.call_args.kwargs
        self.assertEqual(values['instructions'], tpa.AGENT_CONFIGS['code_writer']['instructions'])
        self.assertFalse(values['add_datetime_to_context'])
        self.assertEqual(len(values['tools']), 3)
        self.assertTrue(all('image' not in getattr(tool, '__name__', '').lower() for tool in values['tools']))

    def test_credit_rejection_bills_and_logs_only_delivered_content(self):
        original = 'x' * 100
        meta = {'user_id': 'u', 'requires_user_api_keys': False, 'user_keys': {}}
        for name, next_chunk, cap in (
            ('append', 'y' * 2500, 100 * api.CREDIT_SCALE),
            ('repair', {'__content_replace__': 'y' * 2500}, 100 * api.CREDIT_SCALE),
            ('repair_at_cap', {'__content_replace__': 'y' * 2500}, 2 * api.CREDIT_SCALE),
        ):
            manager = Mock()
            manager.reserve.side_effect = [(True, 1000), (True, 1000), (False, 0)]
            with self.subTest(name=name), \
                    patch.object(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('dummy', meta))), \
                    patch.object(api, 'credit_manager', manager), patch.object(api, 'CREDIT_PRECHECK_ENABLED', True), \
                    patch.object(api, 'MAX_RESERVATION_PER_REQUEST_SCALED', cap), \
                    patch.object(api, 'process_user_query_stream', return_value=iter([original, next_chunk])), \
                    patch.object(api, 'fire_credit_deduction_and_release') as bill, \
                    patch.object(api, 'log_response_body') as log_body:
                response = TestClient(api.app).post('/agent/stream', json={
                    'type': 'generate_code', 'query': 'code', 'model': 'gemini', 'api_key': 'lic'})
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual([event['type'] for event in events], ['start', 'content', 'credit_limit_reached', 'done'])
            self.assertEqual(events[1]['content'], original)
            self.assertEqual(events[2]['output_chars'], len(original))
            self.assertEqual(bill.call_args.kwargs['output_chars'], len(original))
            self.assertEqual(bill.call_args.kwargs['reserved_scaled'], 2 * api.CREDIT_SCALE)
            self.assertEqual(log_body.call_args.args[2]['text'], original)

    def test_exposed_search_tools_share_limits_across_provider_attempts(self):
        from src.request_plan import RequestPlan
        for agent_type in ('code_writer', 'smart_global'):
            for limit in (0, 1):
                context = {}
                with self.subTest(agent_type=agent_type, limit=limit), \
                        patch.object(settings, 'web_search_call_limit', limit), \
                        patch.object(tpa, 'get_model', return_value=Mock()), patch.object(tpa, 'Agent') as agent, \
                        patch('ddgs.DDGS') as ddgs:
                    ddgs.return_value.__enter__.return_value.text.return_value = [{'title': 'test'}]
                    ddgs.return_value.__enter__.return_value.news.return_value = [{'title': 'test'}]
                    for provider in ('claude', 'openai'):
                        tpa.create_agent(agent_type, provider, 'dummy', image_ctx=context,
                                         plan=RequestPlan(use_web=True, advanced_model=True))
                        for tool in agent.call_args.kwargs['tools']:
                            if isinstance(tool, tpa.DuckDuckGoWithRetry):
                                for function in tool.functions.values():
                                    function.entrypoint('test')
                            elif getattr(tool, '__name__', '') == 'parallel_web_search':
                                tool(['first', 'second'])
                    self.assertEqual(ddgs.call_count, limit)
                    self.assertEqual(context['web_budgets']['search'].used, limit)

    def test_approved_repair_bills_the_replacement_length(self):
        meta = {'user_id': 'u', 'requires_user_api_keys': False, 'user_keys': {}}
        for length in (50, 2500):
            manager = Mock()
            manager.reserve.return_value = (True, 1000)
            with self.subTest(length=length), \
                    patch.object(api, 'async_validate_and_get_api_key', AsyncMock(return_value=('dummy', meta))), \
                    patch.object(api, 'credit_manager', manager), patch.object(api, 'CREDIT_PRECHECK_ENABLED', True), \
                    patch.object(api, 'process_user_query_stream', return_value=iter([
                        'x' * 100, {'__content_replace__': 'y' * length}])), \
                    patch.object(api, 'fire_credit_deduction_and_release') as bill:
                response = TestClient(api.app).post('/agent/stream', json={
                    'type': 'generate_code', 'query': 'code', 'model': 'gemini', 'api_key': 'lic'})
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual([event['type'] for event in events], ['start', 'content', 'content_replace', 'done'])
            self.assertEqual(events[2]['content'], 'y' * length)
            self.assertEqual(bill.call_args.kwargs['output_chars'], length)

    def test_usage_and_price_estimate_include_repairs(self):
        from src.observability import record_text_usage
        start_trace('usage-test', '/agent/process')
        with patch.object(settings, 'text_prices_per_mtok', '{"test": [1, 2]}'):
            record_text_usage('usage-test', 'test', {'input_tokens': 100, 'output_tokens': 20})
            record_text_usage('usage-test', 'test', {'input_tokens': 100, 'output_tokens': 20})
        summary = finish_trace('usage-test')
        self.assertEqual((summary['tokens_in'], summary['tokens_out']), (200, 40))
        self.assertAlmostEqual(summary['est_cost_usd'], 0.00028)

    def test_fallback_attempts_share_web_budgets(self):
        context = {}
        with patch.object(tpa, 'get_model', return_value=Mock()), patch.object(tpa, 'Agent'):
            tpa.create_agent('code_writer', 'claude', 'a', image_ctx=context)
            first_budget = context['web_budgets']['search']
            self.assertTrue(first_budget.take())
            tpa.create_agent('code_writer', 'openai', 'o', image_ctx=context)
        self.assertIs(context['web_budgets']['search'], first_budget)
        self.assertEqual(first_budget.used, 1)

    def test_stream_requests_completed_events_for_token_usage(self):
        from agno.metrics import RunMetrics
        events = [SimpleNamespace(event=RunEvent.run_content, content=FIXED),
                  SimpleNamespace(event=RunEvent.run_completed, metrics=RunMetrics(input_tokens=10, output_tokens=20))]
        agent = SimpleNamespace(model=SimpleNamespace(id='test'), run=Mock(return_value=iter(events)))
        start_trace('stream-usage', '/agent/stream')
        self.assertEqual(list(tpa._agent_text(agent, 'q', None, 'gemini', False, 'stream-usage')), [FIXED])
        summary = finish_trace('stream-usage')
        self.assertEqual((summary['tokens_in'], summary['tokens_out']), (10, 20))
        self.assertTrue(agent.run.call_args.kwargs['stream_events'])

    def test_model_source_is_backwards_compatible_and_strict(self):
        values = dict(type='generate_code', query='add', model='gemini', api_key='k')
        self.assertEqual(api.StreamRequest(**values).model_source, 'user')
        for model in (api.ProcessRequest, api.StreamRequest):
            with self.assertRaises(ValueError):
                model(**values, model_source='auto')

    def test_own_keys_on_managed_plan_cannot_fallback_to_platform(self):
        meta = {'user_id': 'u', 'requires_user_api_keys': False,
                'user_keys': {'use_own_keys': True, 'gemini_api_key': 'user-gemini'}}
        manager = SimpleNamespace(resolve_api_key_from_user_keys=RedisManager.resolve_api_key_from_user_keys.__get__(object()))
        events = [SimpleNamespace(event=RunEvent.run_content, content=FIXED)]
        with patch.object(settings, 'task_routing_enabled', True), \
                patch.object(settings, 'provider_fallback_enabled', True), \
                patch.object(api, 'redis_manager', manager), \
                patch.dict(api.DEFAULT_API_KEYS, {'anthropic_api_key': 'platform'}), \
                patch.object(api, 'async_validate_and_get_api_key', AsyncMock(return_value=(None, meta))), \
                patch.object(api, 'CREDIT_PRECHECK_ENABLED', False), patch.object(api, 'fire_credit_deduction'), \
                patch.object(tpa, 'create_agent', return_value=SimpleNamespace(run=lambda *a, **kw: iter(events))) as create:
            response = TestClient(api.app).post('/agent/stream', json={
                'type': 'generate_code', 'query': 'add', 'model': 'claude', 'model_source': 'default', 'api_key': 'lic'})
        self.assertNotIn('"type": "error"', response.text)
        self.assertEqual(create.call_args.args[1:3], ('gemini', 'user-gemini'))

    def test_classifier_uses_only_keys_from_the_real_own_key_lookup(self):
        manager = SimpleNamespace(
            get_cached_user_id=lambda _: None,
            validate_license=lambda _: {'valid': True, 'user_id': 'u', 'license': {'requires_user_api_keys': False}},
            get_user_api_keys=lambda _: {'use_own_keys': True, 'gemini_api_key': 'user-gemini'},
            update_heartbeat=lambda _: None,
            resolve_api_key_from_user_keys=RedisManager.resolve_api_key_from_user_keys.__get__(object()),
        )
        for provider in ('openai', 'gemini'):
            events = [SimpleNamespace(event=RunEvent.run_content, content='ok')]
            with self.subTest(provider=provider), patch.object(api, 'redis_manager', manager), \
                    patch.dict(api.DEFAULT_API_KEYS, {'openai_api_key': 'platform-openai'}), \
                    patch.object(settings, 'task_routing_enabled', True), \
                    patch.object(settings, 'code_intent_regex_enabled', True), \
                    patch.object(settings, 'code_intent_classifier_enabled', True), \
                    patch.object(api, 'CREDIT_PRECHECK_ENABLED', False), patch.object(api, 'fire_credit_deduction'), \
                    patch('src.intent_classifier._call', return_value='other') as classify, \
                    patch.object(tpa, 'create_agent', return_value=SimpleNamespace(run=lambda *a, **kw: iter(events))):
                TestClient(api.app).post('/agent/stream', json={
                    'type': 'general', 'query': 'I need something in Python that sorts records',
                    'model': provider, 'api_key': 'lic'})
            if provider == 'openai':
                classify.assert_not_called()
            else:
                self.assertEqual(classify.call_args.args[1:3], ('gemini', 'user-gemini'))

    def test_required_own_keys_are_used_even_without_preference_flag(self):
        manager = SimpleNamespace(
            get_cached_user_id=lambda _: None,
            validate_license=lambda _: {'valid': True, 'user_id': 'u', 'license': {'requires_user_api_keys': True}},
            get_user_api_keys=lambda _: {'openai_api_key': 'user-openai'},
            update_heartbeat=lambda _: None,
            resolve_api_key_from_user_keys=RedisManager.resolve_api_key_from_user_keys.__get__(object()),
        )
        with patch.object(api, 'redis_manager', manager):
            key, meta = asyncio.run(api.async_validate_and_get_api_key('lic', 'openai', 'test'))
        self.assertEqual(key, 'user-openai')
        self.assertTrue(meta['user_keys']['use_own_keys'])
