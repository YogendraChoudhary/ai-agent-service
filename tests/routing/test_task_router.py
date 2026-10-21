import unittest
from unittest.mock import patch

from config import Settings, settings
from src.models import ADVANCED_MODEL_IDS, BASIC_MODEL_IDS, AUTOCOMPLETE_MODEL_IDS
from src.task_router import KeyResolver, resolve_route
from src.redis_manager import RedisManager
from src.text_processing_agent import get_model, get_autocomplete_suggestion


class TestTaskRouter(unittest.TestCase):
    def route(self, **kw):
        values = dict(task='code', model='gemini', model_source='default',
                      available_providers={'claude', 'openai', 'gemini'}, tier=True)
        values.update(kw)
        return [c.provider for c in resolve_route(**values)]

    def test_flags_off_keeps_requested_provider(self):
        self.assertEqual(self.route(), ['gemini'])

    @patch.object(settings, 'task_routing_enabled', True)
    def test_defaults_and_user_choice(self):
        self.assertEqual(self.route(), ['claude', 'openai', 'gemini'])
        self.assertEqual(self.route(model_source='user'), ['gemini', 'claude', 'openai'])
        self.assertEqual(self.route(task='content'), ['gemini', 'openai', 'claude'])
        self.assertEqual(self.route(available_providers={'gemini'}), ['gemini'])

    @patch.object(settings, 'provider_fallback_enabled', True)
    def test_same_tier_vision_filter_and_deduplication(self):
        route = resolve_route('content', 'deepseek', 'user', set(ADVANCED_MODEL_IDS), False, has_images=True)
        self.assertEqual([c.provider for c in route], ['openai', 'claude'])
        self.assertTrue(all(c.advanced is False for c in route))

    def test_byok_key_resolver_ignores_platform(self):
        resolve = RedisManager.resolve_api_key_from_user_keys.__get__(object())
        resolver = KeyResolver({'gemini_api_key': 'user'}, None, resolve)
        self.assertEqual(resolver.available_providers, {'gemini'})
        self.assertIsNone(resolver('claude'))
        self.assertEqual(resolver('gemini'), 'user')

    def test_invalid_env_json_is_ignored(self):
        for value in ('[]', '{', '{"code": ["fable"]}', '{"code": "fable"}', '{"code": {}}'):
            with self.subTest(value=value):
                config = Settings(_env_file=None, TASK_ROUTE_DEFAULTS=value)
                self.assertEqual(config.task_route_defaults, '{}')
        self.assertEqual(Settings(_env_file=None, TASK_FALLBACK_TREES='{"code": [42]}').task_fallback_trees, '{}')

    def test_config_override(self):
        with patch.object(settings, 'task_routing_enabled', True), \
                patch.object(settings, 'task_route_defaults', '{"code": "openai"}'):
            self.assertEqual(self.route()[0], 'openai')

    def test_fable_cannot_reach_any_model_factory_or_route(self):
        for table in (ADVANCED_MODEL_IDS, BASIC_MODEL_IDS):
            with patch.dict(table, {'claude': 'claude-fable-5-1'}), \
                    patch.object(settings, 'task_routing_enabled', True):
                with self.assertRaises(ValueError):
                    get_model('claude', 'key', use_advanced=table is ADVANCED_MODEL_IDS)
                route = resolve_route('code', 'claude', 'user', {'claude', 'openai'}, table is ADVANCED_MODEL_IDS)
                self.assertEqual([c.provider for c in route], ['openai'])
        with patch.dict(AUTOCOMPLETE_MODEL_IDS, {'claude': 'claude-fable-5-1'}), self.assertRaises(ValueError):
            get_autocomplete_suggestion('hi', '', 'claude', 'key')
