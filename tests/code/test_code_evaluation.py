import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import settings
from scripts import evaluate_code


class TestCodeEvaluation(unittest.TestCase):
    def test_repaired_answer_replaces_original_in_saved_results(self):
        broken = '```python\ndef f():\nreturn 1\n```'
        corrected = '```python\ndef f():\n    return 1\n```'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            with patch.object(settings, 'default_gemini_api_key', 'dummy'), \
                    patch.object(evaluate_code, 'process_user_query_stream', return_value=iter([
                        broken, {'__content_replace__': corrected}])), \
                    patch('sys.argv', ['evaluate_code.py', '--live', '--providers', 'gemini', '--limit', '1', '--out', str(output)]):
                self.assertEqual(evaluate_code.main(), 0)
            result = json.loads(output.read_text())[0]
        self.assertEqual(result['content'], corrected)
        self.assertEqual(result['errors'], [])
        self.assertIsNotNone(result['ttft'])
