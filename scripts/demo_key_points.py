"""Exercise the key-point summary API on the supplied survey passage; --live calls OpenAI."""
import argparse
import html
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.update({'REDIS_URL':'','BACKEND_WEBHOOK_URL':'','WEBHOOK_SECRET':'','VOICE_FLOW_ENABLED':'false',
    'CREDIT_PRECHECK_ENABLED':'false','TASK_ROUTING_ENABLED':'false','PROVIDER_FALLBACK_ENABLED':'false',
    'LOG_REQUEST_BODIES':'false'})

from fastapi.testclient import TestClient
import api
from config import settings
from src.longform.llm import AgnoLLM

CASE=json.loads((ROOT/'tests/fixtures/key_points_summary.json').read_text())


def scripted_agent(*args,**kwargs):
    class Model:
        def run(self,*args,**kwargs): return SimpleNamespace(content=json.dumps(CASE['selection']),metrics=None)
    return Model()


def run(live):
    logging.getLogger().setLevel(logging.WARNING)
    out=ROOT/'local_run'/'key_points_summary'/('live' if live else 'scripted')
    out.mkdir(parents=True,exist_ok=True)
    key=settings.default_openai_api_key if live else 'scripted-key'
    if not key: raise RuntimeError('--live requires DEFAULT_OPENAI_API_KEY')
    request={'type':'highlight_key_points','query':CASE['source'],'model':'openai','api_key':key}
    calls=[]
    original_call=AgnoLLM.__call__
    def record_call(self,system,messages,spec):
        reply=original_call(self,system,messages,spec)
        calls.append({'stage':spec.stage,'model':self.used,'reply':reply})
        (out/'model_replies.json').write_text(json.dumps(calls,indent=2,ensure_ascii=False))
        return reply
    with patch.object(AgnoLLM,'__call__',record_call):
        with nullcontext() if live else patch('src.text_processing_agent.create_agent',scripted_agent):
            response=TestClient(api.app,raise_server_exceptions=False).post('/agent/process',json=request)
    record={'live_provider':live,'http_status':response.status_code,'request_id':response.headers.get('x-request-id'),
            'request':{**request,'api_key':'[redacted]'},'response':response.json()}
    (out/'receipt.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    assert response.status_code==200,response.text
    result=response.json()['highlighting']
    summary=response.json()['markdown_response']
    assert summary==result['summary_markdown'] and summary!=CASE['source']
    assert '73%' in result['lead']
    for percent,threshold in [('71%','100'),('68%','1,000'),('80%','1,000')]:
        point=next(p for p in result['key_points'] if percent in p)
        assert threshold.replace(',','') in point.replace(',','')
    assert 4<=len(result['key_points'])<=6
    assert all(quote in CASE['source'] for quote in result['evidence'])
    (out/'summary.md').write_text(summary)
    (out/'example.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Key-point summary example</title><style>body{font:17px/1.6 Arial,sans-serif;max-width:950px;margin:32px auto;padding:0 24px}pre{white-space:pre-wrap}section{border-top:1px solid #ccd5df;padding:18px 0}</style><h1>Highlight key points</h1><p>Provider: '+('real configured OpenAI model' if live else 'scripted provider')+'</p><section><h2>Original passage</h2><pre>'+html.escape(CASE['source'])+'</pre></section><section><h2>Key-point summary</h2><p>'+html.escape(result['lead'])+'</p><ul>'+''.join('<li>'+html.escape(p)+'</li>' for p in result['key_points'])+'</ul></section><p><a href="receipt.json">Actual request, source evidence and response</a></p></html>')
    print(json.dumps({'live_provider':live,'http_status':response.status_code,'summary':summary,'example':str(out/'example.html')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    run(parser.parse_args().live)
