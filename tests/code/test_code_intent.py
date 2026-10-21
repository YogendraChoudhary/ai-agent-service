import unittest
from unittest.mock import Mock, patch
from config import settings
from tests.fixtures.code_intent_eval_cases import CASES
from src.code_intent import detect_code_intent
from src.task_router import task_for_request


class TestCodeIntent(unittest.TestCase):
    def test_labelled_corpus(self):
        self.assertGreaterEqual(len(CASES), 150)
        for group, query, expected in CASES:
            with self.subTest(group=group, query=query):
                self.assertEqual(detect_code_intent(query)['kind'], expected)

    def test_quality_corpus_covers_thirty_prompts_per_language(self):
        from collections import Counter
        from scripts.evaluate_code import quality_prompts
        self.assertEqual(Counter(lang for lang, _ in quality_prompts()),
                         {'python': 30, 'typescript': 30, 'html': 30, 'sql': 30})

    def test_no_classifier_latency_without_code_cues(self):
        classify = Mock(return_value='code')
        with patch.object(settings, 'code_intent_regex_enabled', True), \
                patch.object(settings, 'code_intent_classifier_enabled', True):
            self.assertEqual(task_for_request('general', 'Write a birthday greeting', classify)[0], 'content')
            classify.assert_not_called()
            self.assertEqual(task_for_request('general', 'I need something in Python', classify)[0], 'code')
            self.assertEqual(classify.call_count, 1)

    def test_polite_builds_and_mixed_content(self):
        self.assertEqual(detect_code_intent('Please write me a C++ function')['kind'], 'code')
        self.assertEqual(detect_code_intent('Make a React component')['kind'], 'code')
        self.assertEqual(detect_code_intent('Write me an article with a Python example')['kind'], 'other')

    def test_code_prose_stays_content_without_classification(self):
        classify = Mock(return_value='code')
        with patch.object(settings, 'code_intent_regex_enabled', True), \
                patch.object(settings, 'code_intent_classifier_enabled', True):
            for group, query, _ in CASES:
                if group == 'code_prose':
                    with self.subTest(query=query):
                        self.assertEqual(task_for_request('general', query, classify)[0], 'content')
            classify.assert_not_called()
        self.assertEqual(detect_code_intent('Write a Python function that generates a code review')['kind'], 'code')
        self.assertEqual(detect_code_intent('Write a simple code review for this function')['kind'], 'other')
        self.assertEqual(detect_code_intent('Write a new function specification for the API')['kind'], 'other')

    def test_shadow_and_typed_actions(self):
        with patch.object(settings, 'code_intent_regex_enabled', True), \
                patch.object(settings, 'code_intent_shadow_mode', True):
            task, attrs = task_for_request('general', 'write a Python function')
            self.assertEqual((task, attrs['intent_result']), ('content', 'code'))
            self.assertEqual(task_for_request('simplify', 'write a Python function')[0], 'content')
            self.assertEqual(task_for_request('generate_code', 'sort records')[0], 'code')
