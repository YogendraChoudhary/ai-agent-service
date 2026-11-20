"""
Billing leaks found by the 2026-11-20 audit, each driven through the real /agent/stream with the real CreditManager and
BillingDelivery over fakeredis: every reserve, release and charge is recorded. Only the model and image providers are stubbed.
"""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis
import pytest
from agno.run.agent import RunEvent
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src import billing_delivery, webhook
from src.credit_manager import CreditManager
from src.design_canvas.schema_v11 import DesignDocumentASTV11
from tests.design.test_design_images import CDN, _card, _image, _listing


class Recorder(CreditManager):
    def __init__(self, store, log):
        super().__init__(store)
        self.log = log

    def reserve(self, user_id, scaled_amount):
        ok = super().reserve(user_id, scaled_amount)
        self.log.append(("reserve", scaled_amount, ok[0]))
        return ok

    def release(self, user_id, scaled_amount):
        self.log.append(("release", scaled_amount))
        return super().release(user_id, scaled_amount)


@pytest.fixture
def paid(monkeypatch):
    store = fakeredis.FakeRedis(decode_responses=True)
    store.hset("subscription:u1", "remaining_credits", 1000)
    log, charges = [], []
    credits = Recorder(store, log)
    box = billing_delivery.BillingDelivery(store, "https://billing.invalid", "test-secret")
    monkeypatch.setattr(webhook, "_billing_delivery", box)
    real_enqueue = box.enqueue

    def enqueue(payload, **kw):
        charges.append(payload)
        return real_enqueue(payload, **kw)
    monkeypatch.setattr(box, "enqueue", enqueue)
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
    monkeypatch.setattr(api, "CREDIT_SETTLE_DELAY_SECONDS", 0)
    monkeypatch.setattr(settings, "task_routing_enabled", False)
    monkeypatch.setattr(settings, "provider_fallback_enabled", False)
    monkeypatch.setattr(api, "credit_manager", credits)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
    monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
    return SimpleNamespace(store=store, log=log, charges=charges)


def left_reserved(paid):
    time.sleep(0.3)          # schedule_release runs on a thread
    return int(paid.store.get("credits_reserved:u1") or 0)


def stub_images(monkeypatch, fail=False):
    import src.image_gen as ig
    monkeypatch.setattr(ig, "plan_image_target", lambda *a, **k: ("openai", "gpt-image-test"))
    monkeypatch.setattr(ig, "image_route_available", lambda *a, **k: True)
    made = []

    def generate(**kw):
        if fail:
            raise RuntimeError("image provider down")
        made.append(kw.get("prompt"))
        return {"cdn_url": f"https://cdn.test/{len(made)}.png", "width": 10, "height": 10, "format": "png",
                "provider": "openai", "model": "gpt-image-test"}
    monkeypatch.setattr(ig, "generate_image_unified", generate)
    return made


def post(body):
    response = TestClient(api.app, raise_server_exceptions=False).post("/agent/stream", json=body)
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except ValueError:
                pass
    return response, events


def write_model(*, fail=False, abort=False):
    def run(agent, prompt, **kwargs):
        tool = next((t for t in agent.tools or [] if getattr(t, "__name__", "") == "generate_image_tool"), None)
        if tool is not None:
            tool("a cat on a sofa")
        yield SimpleNamespace(event=RunEvent.run_content, content="Cats are great. " * 3, metrics=None)
        if abort:
            for event in list(api._active_cancellations.values()):
                event.set()
        if fail:
            raise RuntimeError("provider 500 mid-stream")
        yield SimpleNamespace(event=RunEvent.run_completed, content=None, metrics={"input_tokens": 10, "output_tokens": 10})
    return run


WRITE = {"type": "general", "query": "Write a short post about cats with 1 image", "model": "openai", "api_key": "k"}


@pytest.mark.parametrize("fail, abort", [(True, False), (False, True)])
def test_write_image_credits_are_settled_when_the_answer_stops(paid, monkeypatch, fail, abort):
    made = stub_images(monkeypatch)
    monkeypatch.setattr(tpa.Agent, "run", write_model(fail=fail, abort=abort))
    post(WRITE)
    assert made and left_reserved(paid) == 0           # was 50 (5 credits) held for good


def test_direct_image_is_billed_as_its_image_only(paid, monkeypatch):
    stub_images(monkeypatch)
    monkeypatch.setattr(tpa.Agent, "run", lambda *a, **k: pytest.fail("a chat model ran"))
    post({"type": "general", "query": "Generate an image of a red fox in the snow", "model": "openai", "api_key": "k"})
    charge, = paid.charges
    assert (charge["operation_type"], charge["image_count"], charge["input_chars"], charge["output_chars"]) == ("image", 1, 0, 0)
    assert left_reserved(paid) == 0


def test_a_failed_direct_image_costs_nothing(paid, monkeypatch):
    stub_images(monkeypatch, fail=True)
    monkeypatch.setattr(tpa.Agent, "run", lambda *a, **k: pytest.fail("a chat model ran"))
    post({"type": "general", "query": "Generate an image of a red fox in the snow", "model": "openai", "api_key": "k"})
    assert paid.charges == [] and left_reserved(paid) == 0


def test_editing_an_attached_image_reserves_its_price_once(paid, monkeypatch):
    paid.store.hset("subscription:u1", "remaining_credits", 10)      # enough for 1 text + 5 image credits, not for a double hold
    made = stub_images(monkeypatch)
    monkeypatch.setattr(tpa, "_fetch_reference_bytes", lambda url, rid: b"\x89PNG fake")
    monkeypatch.setattr(tpa.Agent, "run", lambda *a, **k: pytest.fail("a chat model ran"))
    response, events = post({"type": "general", "query": "Make this image black and white", "model": "openai", "api_key": "k",
                             "image_urls": [api.CDN_URL.rstrip("/") + "/u/ref.png"]})
    assert response.status_code == 200 and made
    assert not any("insufficient credits" in (e.get("content") or "") for e in events)
    charge, = paid.charges
    assert charge["operation_type"] == "image" and charge["image_count"] == 1


@pytest.mark.parametrize("fail, slots", [(False, "filled"), (True, "empty")])
def test_photo_fill_with_no_photo_delivered_costs_nothing(paid, monkeypatch, fail, slots):
    for flag in ("design_canvas_enabled", "design_schema_v11_enabled", "design_image_fill_enabled"):
        monkeypatch.setattr(settings, flag, True)
    stub_images(monkeypatch, fail=fail)
    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness",
                        AsyncMock(side_effect=AssertionError("layout model must not run")))
    image = _image(1, src=f"{CDN}/u/ai-images/a.png") if slots == "filled" else _image(1)
    post({"type": "general", "query": "add real photos", "model": "claude", "api_key": "lic",
          "existing_design_ast": _listing([_card(1, "KFC", image)])})
    assert paid.charges == [] and left_reserved(paid) == 0


def test_an_aborted_design_costs_nothing(paid, monkeypatch):
    for flag in ("design_canvas_enabled", "design_schema_v11_enabled"):
        monkeypatch.setattr(settings, flag, True)
    for flag in ("design_text_edit_enabled", "design_section_edit_enabled"):
        monkeypatch.setattr(settings, flag, False)

    async def aborting(runner, **kwargs):
        for event in list(api._active_cancellations.values()):
            event.set()
        await asyncio.sleep(5)
    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness", aborting)
    post({"type": "generate_design", "query": "Design a landing page for a pizza shop", "model": "claude", "api_key": "lic"})
    assert paid.charges == [] and left_reserved(paid) == 0


def test_a_delivered_design_is_still_charged_its_flat_price(paid, monkeypatch):
    for flag in ("design_canvas_enabled", "design_schema_v11_enabled"):
        monkeypatch.setattr(settings, flag, True)
    for flag in ("design_text_edit_enabled", "design_section_edit_enabled"):
        monkeypatch.setattr(settings, flag, False)

    async def ok(runner, **kwargs):
        return DesignDocumentASTV11.model_validate(_listing([_card(1, "Pizza")]))
    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness", ok)
    post({"type": "edit_design", "query": "make the background dark blue", "model": "claude", "api_key": "lic",
          "existing_design_ast": _listing([_card(1, "KFC")])})
    charge, = paid.charges
    assert charge["design"]["sub_mode"] == "edit_existing_ast"


def test_webhook_lists_only_the_images_the_user_was_sent(paid, monkeypatch):
    def pipeline(*a, **kw):
        kw["credit_manager"].reserve("u1", 50)
        yield {"__pending_images__": {
            "p1": {"status": "done", "cdn_url": "https://cdn.test/1.png", "error": None, "prompt": "x", "provider": "openai",
                   "model": "gpt-image-test", "credits_scaled": 50, "reserved_scaled": 50},
            "__completion_callback__": kw["image_completion_callback"]}}
        yield "Title\n\n"
        kw["cancel_event"].set()                         # Stop pressed before image_ready was sent
        yield "more text"
    monkeypatch.setattr(api, "process_user_query_stream", pipeline)
    post({"type": "general", "query": "Write a post about foxes", "model": "openai", "api_key": "k"})
    charge, = paid.charges
    assert charge["image_count"] == 0 and not charge.get("images")


def test_process_reports_what_to_bill(monkeypatch):
    stub_images(monkeypatch)
    monkeypatch.setattr(settings, "task_routing_enabled", False)
    monkeypatch.setattr(settings, "provider_fallback_enabled", False)
    monkeypatch.setattr(tpa.Agent, "run", lambda *a, **k: pytest.fail("a chat model ran"))
    result = TestClient(api.app, raise_server_exceptions=False).post("/agent/process", json={
        "type": "general", "query": "Generate an image of a red fox in the snow", "model": "openai", "api_key": "k"}).json()
    assert result["billing"]["operation_type"] == "image" and result["billing"]["image_count"] == 1
    assert (result["billing"]["input_chars"], result["billing"]["output_chars"]) == (0, 0)


@pytest.mark.parametrize("reply, charged", [
    ({"photos": {"img1": {"prompt": "A pepperoni pizza on a wooden board", "alt": "Pizza"}}}, True),
    ({"not_photo_edit": True}, False),
])
def test_photo_prompt_rewrite_is_billed_as_text_with_the_photo_it_described(paid, monkeypatch, reply, charged):
    for flag in ("design_canvas_enabled", "design_schema_v11_enabled", "design_image_fill_enabled"):
        monkeypatch.setattr(settings, flag, True)
    made = stub_images(monkeypatch)
    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness",
                        AsyncMock(side_effect=AssertionError("layout model must not run")))

    def llm(system, messages, spec):
        return json.dumps(reply)
    llm.used = ("claude", "claude-opus-5-5")
    monkeypatch.setattr(tpa, "_design_text_llm", lambda *a, **k: llm)
    design = _listing([_card(1, "KFC", _image(1, src=f"{CDN}/u/ai-images/a.png"))])
    post({"type": "edit_design", "model": "claude", "api_key": "lic", "existing_design_ast": design,
          "query": "replace this image with a pizza\n\n(Apply this to the selected section, id img1. Keep everything else unchanged.)"})
    if charged:
        charge, = paid.charges
        assert made and charge["operation_type"] == "mixed" and charge["image_count"] == 1
        assert charge["input_chars"] > 0 and charge["output_chars"] == len(json.dumps(reply))
    else:
        assert not made and paid.charges == []                  # no photo made, nothing charged
    assert left_reserved(paid) == 0
