"""Run the real highlighting API with a scripted model or --live configured OpenAI key."""
import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.update({"REDIS_URL":"","BACKEND_WEBHOOK_URL":"","WEBHOOK_SECRET":"",
    "VOICE_FLOW_ENABLED":"false","CREDIT_PRECHECK_ENABLED":"false","TASK_ROUTING_ENABLED":"false",
    "PROVIDER_FALLBACK_ENABLED":"false","LOG_REQUEST_BODIES":"false"})

from fastapi.testclient import TestClient
import api
from config import settings
from src import text_processing_agent as tpa

SAMPLES=[
    ("Launch plan","The launch is scheduled for 15 November. All teams must complete testing by 10 November. The approved budget is USD 20,000. This note is for internal planning.",
     ["launch is scheduled for 15 November","complete testing by 10 November","approved budget is USD 20,000"]),
    ("Unicode and conditions","🚀 Registration closes on Friday. Attendance is not mandatory. Remote participation is available for international team members. Please keep this information for reference.",
     ["Registration closes on Friday","Attendance is not mandatory","Remote participation is available for international team members"]),
    ("Short passage with only key facts","Revenue grew 20%, costs fell 10%.",
     ["Revenue grew 20%","costs fell 10%"]),
]


def scripted_agent(*args,**kwargs):
    class Model:
        def run(self,history,**kw):
            source=json.loads(history[0].content)["source_text"]
            points=next(points for _,text,points in SAMPLES if text==source)
            return SimpleNamespace(content=json.dumps({"key_points":points}),metrics=None)
    return Model()


def run(live):
    logging.getLogger().setLevel(logging.WARNING)
    out=ROOT/"local_run"/"key_points_demo"/("live" if live else "scripted")
    out.mkdir(parents=True,exist_ok=True)
    key=settings.default_openai_api_key if live else "scripted-key"
    if not key:
        raise RuntimeError("--live requires DEFAULT_OPENAI_API_KEY")
    client,receipts,sections=TestClient(api.app,raise_server_exceptions=False),[],[]
    with nullcontext() if live else patch.object(tpa,"create_agent",scripted_agent):
        for title,source,_ in SAMPLES:
            request={"type":"highlight_key_points","query":source,"model":"openai","api_key":key}
            response=client.post("/agent/process",json=request)
            receipts.append({"request":{**request,"api_key":"[redacted]"},"http_status":response.status_code,
                             "request_id":response.headers.get("x-request-id"),"response":response.json()})
            (out/"receipts.json").write_text(json.dumps({"live_provider":live,"requests":receipts},indent=2))
            assert response.status_code==200,response.text
            data=response.json()
            result=data["highlighting"]
            assert data["markdown_response"]==data["html_response"]==source
            assert result["source_sha256"]==hashlib.sha256(source.encode()).hexdigest()
            assert result["key_points"],"The meaningful sample should have at least one key point"
            assert html.unescape(re.sub(r"<[^>]+>","",result["highlighted_html"]))==source
            for span in result["spans"]:
                assert source.encode("utf-16-le")[span["start_utf16"]*2:span["end_utf16"]*2].decode("utf-16-le")==span["text"]
            sections.append('<section><h2>'+html.escape(title)+'</h2><h3>Original</h3><p>'+html.escape(source)+
                '</p><h3>Highlighted</h3><p>'+result["highlighted_html"]+'</p><h3>Extracted key points</h3><ul>'+''.join('<li>'+html.escape(p)+'</li>' for p in result["key_points"])+
                '</ul><p>Verified: unchanged source, exact excerpts and matching UTF-16 ranges.</p></section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Highlight key points</title><style>body{font:17px/1.65 Arial,sans-serif;color:#182434;max-width:950px;margin:36px auto;padding:0 24px}section{border-top:1px solid #ccd5df;margin-top:24px;padding:18px 0}h3{font-size:17px}mark{background:#fff1a8;color:inherit;padding:1px 0}</style><h1>Highlight key points</h1><p>Provider: '+('real configured OpenAI model' if live else 'scripted, no network')+'. Production Python API output.</p>'+''.join(sections)+'<p><a href="receipts.json">Redacted request and response evidence</a></p></html>'
    (out/"example.html").write_text(page)
    print(json.dumps({"live_provider":live,"examples":len(receipts),"unchanged_source_verified":True,"exact_ranges_verified":True,"example":str(out/"example.html")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    run(parser.parse_args().live)
