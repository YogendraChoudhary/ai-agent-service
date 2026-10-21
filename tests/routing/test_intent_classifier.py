import time
import unittest
from unittest.mock import Mock, patch
from config import settings
from src.intent_classifier import classify_intent


class TestClassifier(unittest.TestCase):
    def test_byok_uses_only_the_users_provider_and_key(self):
        call = Mock(return_value='code')
        self.assertEqual(classify_intent('x' * 1500, 'gemini', 'user', True, 'platform', call=call), 'code')
        self.assertEqual(call.call_args.args[:3], ('x' * 1000, 'gemini', 'user'))

    def test_platform_uses_haiku_provider(self):
        call = Mock(return_value='other')
        classify_intent('hi', 'gemini', 'user', False, 'platform', call=call)
        self.assertEqual(call.call_args.args[1:3], ('claude', 'platform'))

    def test_missing_key_garbage_and_errors_fail_safe(self):
        call = Mock(return_value='code')
        self.assertEqual(classify_intent('hi', 'gemini', None, True, 'platform', call=call), 'other')
        call.assert_not_called()
        for reply in ('code because...', '', None):
            self.assertEqual(classify_intent('hi', 'gemini', 'k', True, call=Mock(return_value=reply)), 'other')
        self.assertEqual(classify_intent('hi', 'gemini', 'k', True, call=Mock(side_effect=RuntimeError())), 'other')

    def test_timeout_is_bounded(self):
        def slow(*args):
            time.sleep(0.05)
            return 'code'
        with patch.object(settings, 'code_classifier_timeout_seconds', 0.01):
            started = time.perf_counter()
            self.assertEqual(classify_intent('hi', 'gemini', 'k', True, call=slow), 'other')
            self.assertLess(time.perf_counter() - started, 0.04)
