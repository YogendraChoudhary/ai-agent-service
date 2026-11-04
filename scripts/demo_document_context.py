"""Provider-free Ask/Draft API demonstration; no Redis, webhooks or paid model calls."""
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update({'DOCUMENT_CONTEXT_ENABLED': 'true', 'TASK_ROUTING_ENABLED': 'false',
    'PROVIDER_FALLBACK_ENABLED': 'false', 'CREDIT_PRECHECK_ENABLED': 'false',
    'BACKEND_WEBHOOK_URL': '', 'REDIS_URL': '', 'LOG_REQUEST_BODIES': 'false'})

from agno.agent import RunEvent
from fastapi.testclient import TestClient
import api
from src.text_processing_agent import Agent

SOURCE = '# Launch\n\nThe project codename is Velvet Mango.\n\nStart with a small cohort.\n'


def scripted_run(agent, prompt, **kwargs):
    assert not agent.tools
    data = json.JSONDecoder().raw_decode(prompt)[0]
    source = ''.join(e['text'] for e in data['excerpts'])
    assert 'Velvet Mango' in source
    answer = 'The codename is Velvet Mango.' if data['action'] == 'ask' else 'Invite the first cohort and collect their feedback before expanding.'
    yield SimpleNamespace(event=RunEvent.run_content, content=answer, metrics=None)


def main():
    logging.getLogger().setLevel(logging.ERROR)
    request = {'type': 'general', 'query': 'What is the codename?', 'model': 'openai',
               'api_key': 'scripted-key', 'document_context': {'text': SOURCE}}
    auth = AsyncMock(return_value=('scripted-key', {'user_id': 'demo', 'requires_user_api_keys': True, 'user_keys': {}}))
    with patch.object(Agent, 'run', scripted_run), patch.object(api, 'async_validate_and_get_api_key', auth):
        client = TestClient(api.app)
        ask = client.post('/agent/process', json=request)
        assert ask.status_code == 200, ask.text
        assert ask.json()['markdown_response'] == 'The codename is Velvet Mango.'
        draft = client.post('/agent/process', json={**request, 'document_action': 'draft', 'query': 'Write the next paragraph'})
        assert draft.status_code == 200, draft.text
        assert draft.json()['document_context']['response_kind'] == 'draft'
        assert SOURCE not in draft.json()['markdown_response']
        stream = client.post('/agent/stream', json=request)
        assert 'event: document_context_ready' in stream.text
        too_large = client.post('/agent/process', json={**request, 'document_context': {'text': 'word ' * 8000}})
        assert too_large.json()['document_context']['status'] == 'needs_context'
        assert too_large.json()['document_context']['billable_input_chars'] == 0
        print(json.dumps({'ask': ask.json(), 'draft': draft.json(),
                          'oversized_ask': too_large.json(), 'stream_metadata_verified': True}, indent=2))


if __name__ == '__main__':
    main()
