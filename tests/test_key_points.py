"""Concise lead/bullet summaries with grounded statistics, thresholds and API behavior."""
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.key_points import KeyPointValidationError, Selection, highlight_key_points, resolve_summary
from tests.design.test_design_api import parse_sse_events

CASE=json.loads((Path(__file__).parent/'fixtures'/'key_points_summary.json').read_text())
SOURCE=CASE['source']
SELECTION=CASE['selection']
EXPECTED=SELECTION['lead']+'\n\n'+'\n'.join('- '+p['text'] for p in SELECTION['key_points'])


def test_users_rte_example_is_a_paraphrased_lead_and_five_bullets():
    result=resolve_summary(SOURCE,Selection.model_validate(SELECTION))
    assert result.summary_markdown==EXPECTED
    assert result.lead.startswith('73% of respondents')
    assert len(result.key_points)==5
    assert result.summary_markdown!=SOURCE
    assert len(result.summary_markdown)<len(SOURCE)
    assert '<mark>' not in result.summary_markdown and '**' not in result.summary_markdown
    assert all(quote in SOURCE for quote in result.evidence)
    assert not hasattr(result,'spans') and not hasattr(result,'highlighted_html')


@pytest.mark.parametrize('mutation', ['invented_percent','swapped_group','changed_threshold','dropped_bound','dropped_category','currency',
                                     'unsupported_evidence','duplicate','multiline','broad_evidence'])
def test_unsupported_statistics_group_threshold_changes_and_bad_structure_are_rejected(mutation):
    selection=copy.deepcopy(SELECTION)
    point=selection['key_points'][0]
    source=SOURCE
    if mutation=='invented_percent': point['text']=point['text'].replace('71%','72%')
    elif mutation=='swapped_group': point['text']=selection['key_points'][2]['text']
    elif mutation=='changed_threshold': selection['key_points'][2]['text']=selection['key_points'][2]['text'].replace('more than','up to')
    elif mutation=='dropped_bound': selection['key_points'][2]['text']=selection['key_points'][2]['text'].replace('more than ','')
    elif mutation=='dropped_category': point['text']='Importance is 71% among enterprises.'
    elif mutation=='currency':
        source='The budget is USD 20,000.'
        selection={'lead':'The budget is EUR 20,000.','lead_evidence':source,'key_points':[]}
    elif mutation=='unsupported_evidence': point['evidence']='A sentence not present in the source.'
    elif mutation=='duplicate': point['text']=selection['lead']; point['evidence']=selection['lead_evidence']
    elif mutation=='multiline': point['text']='A new line.\nAnother line.'
    elif mutation=='broad_evidence': point['evidence']=SOURCE
    with pytest.raises(ValueError): resolve_summary(source,Selection.model_validate(selection))


def test_equivalent_thousand_separators_and_bounds_are_preserved():
    selection=copy.deepcopy(SELECTION)
    selection['key_points'][1]['text']=selection['key_points'][1]['text'].replace('1,000','1000')
    assert '1000 employees' in resolve_summary(SOURCE,Selection.model_validate(selection)).key_points[1]


def test_nonbreaking_space_normalization_keeps_the_real_source_evidence():
    selection=copy.deepcopy(SELECTION)
    selection['key_points'][0]['evidence']=selection['key_points'][0]['evidence'].replace('\u00a0',' ')
    result=resolve_summary(SOURCE,Selection.model_validate(selection))
    assert '\u00a0' in result.evidence[1] and result.evidence[1] in SOURCE


@pytest.mark.parametrize('source,valid,invalid',[
    ('Revenue changed by −5%.','Revenue changed by -5%.','Revenue changed by 5%.'),
    ('The budget is USD 20,000.','The budget is USD 20000.','The budget is 20000.'),
])
def test_numeric_signs_and_units_cannot_be_lost_during_paraphrasing(source,valid,invalid):
    assert resolve_summary(source,Selection(lead=valid,lead_evidence=source)).lead==valid
    with pytest.raises(ValueError): resolve_summary(source,Selection(lead=invalid,lead_evidence=source))


def test_empty_or_single_fact_inputs_have_defined_output():
    assert resolve_summary('Hello!',Selection()).summary_markdown=='No key points found.'
    result=resolve_summary('The deadline is Friday.',Selection(lead='The deadline is Friday.',lead_evidence='The deadline is Friday.'))
    assert result.summary_markdown=='The deadline is Friday.' and not result.key_points
    with pytest.raises(ValueError): Selection(key_points=[SELECTION['key_points'][0]])
    with pytest.raises(ValueError): Selection(lead=SELECTION['lead'],key_points=SELECTION['key_points']*2)


def test_summary_rendering_cannot_inject_markup():
    source='The note says <script>alert(x)</script> and [link](https://example.org).'
    result=resolve_summary(source,Selection(lead=source,lead_evidence=source))
    assert '<script>' not in result.summary_markdown
    assert '&lt;script&gt;' in result.summary_markdown
    assert r'\[link\]' in result.summary_markdown


def test_one_repair_then_validation_error():
    calls=[]
    def model(system,messages,spec):
        calls.append(spec.stage)
        return 'bad JSON' if spec.stage=='key_points' else json.dumps(SELECTION)
    assert highlight_key_points(SOURCE,model).summary_markdown==EXPECTED
    assert calls==['key_points','key_points_repair']
    calls.clear()
    def invalid(*args): calls.append(1); return 'bad JSON'
    with pytest.raises(KeyPointValidationError): highlight_key_points(SOURCE,invalid)
    assert len(calls)==2


def test_abort_prevents_a_repair_call():
    stop=threading.Event()
    calls=[]
    def model(*args): calls.append(1); stop.set(); return 'invalid'
    assert highlight_key_points(SOURCE,model,cancel_event=stop) is None
    assert len(calls)==1


@pytest.fixture
def runtime(monkeypatch):
    state={'reply':json.dumps(SELECTION),'calls':[],'error':None}
    def factory(agent_type,provider,key,rid=None,**kwargs):
        assert agent_type=='key_point_highlighter'
        assert not kwargs['plan'].advanced_model and not kwargs['plan'].use_tools
        state['calls'].append((provider,key))
        class Model:
            def run(self,history,**kw):
                if state['error']: raise state['error']
                assert json.loads(history[0].content)['source_text']==SOURCE
                return SimpleNamespace(content=state['reply'],metrics=None)
        return Model()
    monkeypatch.setattr(tpa,'create_agent',factory)
    monkeypatch.setattr(settings,'task_routing_enabled',False)
    monkeypatch.setattr(settings,'provider_fallback_enabled',False)
    monkeypatch.setattr(api,'CREDIT_PRECHECK_ENABLED',False)
    monkeypatch.setattr(api,'credit_manager',None)
    monkeypatch.setattr(api,'async_validate_and_get_api_key',AsyncMock(return_value=('provider-key',{
        'user_id':'u1','requires_user_api_keys':True,'user_keys':{}})))
    monkeypatch.setattr(api,'fire_prompt_history_save',lambda *a,**kw:None)
    return TestClient(api.app,raise_server_exceptions=False),state


def body(**kw): return {'type':'highlight_key_points','query':SOURCE,'model':'openai','api_key':'key',**kw}


def test_process_returns_summary_as_the_main_output(runtime):
    client,state=runtime
    response=client.post('/agent/process',json=body())
    assert response.status_code==200,response.text
    result=response.json()
    assert result['markdown_response']==result['html_response']==EXPECTED
    assert result['highlighting']['summary_markdown']==EXPECTED
    assert result['highlighting']['key_points']==[p['text'] for p in SELECTION['key_points']]
    assert 'spans' not in result['highlighting']
    assert state['calls']==[('openai','key')]


def test_stream_emits_summary_content_not_the_original_paragraph(runtime):
    client,_=runtime
    events=parse_sse_events(client.post('/agent/stream',json=body()).text)
    assert [e['event'] or e['data'].get('type') for e in events]==['start','highlight_ready','content','done']
    assert events[1]['data']['lead']==SELECTION['lead']
    assert events[2]['data']['content']==EXPECTED
    assert events[2]['data']['content']!=SOURCE


@pytest.mark.parametrize('invalid',[{'query':' '},{'query':'x'*10001},{'secondary_prompt':'rewrite'},
    {'image_provider':'openai'},{'existing_design_ast':{}},{'document_text':'Some document.'}])
def test_invalid_requests_do_not_call_a_provider(runtime,invalid):
    client,state=runtime
    response=client.post('/agent/process',json=body(**invalid))
    assert response.status_code==400,response.text
    assert not state['calls']


def test_invalid_output_is_not_emitted(runtime):
    client,state=runtime
    state['reply']='invalid JSON'
    assert client.post('/agent/process',json=body()).status_code==422
    assert len(state['calls'])==2
    state['calls'].clear()
    events=parse_sse_events(client.post('/agent/stream',json=body()).text)
    assert [e['data'].get('type') for e in events]==['start','error']
    assert len(state['calls'])==2


def test_provider_failure_remains_an_error(runtime):
    client,state=runtime
    state['error']=RuntimeError('provider unavailable')
    assert client.post('/agent/process',json=body()).status_code==500
