"""Real API/pipeline integration; only provider SDK calls and external account services are replaced."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.document_edit.blocks import document_blocks, fingerprint
from tests.design.test_design_api import parse_sse_events
from tests.document_edit.test_service import DOC, ScriptedEditor, prose

INSTRUCTION = "Improve Introduction and Benefits; leave Pricing and Conclusion unchanged."


@pytest.fixture
def runtime(monkeypatch):
    blocks = prose()
    model = ScriptedEditor(INSTRUCTION, blocks[:2], {
        blocks[0].id: blocks[0].text.replace("very very useful", "particularly useful"),
        blocks[1].id: blocks[1].text.replace("clear and clear", "clear and practical")})
    state = {"model": model, "calls": [], "error": None}

    def factory(agent_type, provider, key, rid=None, **kwargs):
        assert agent_type == "document_editor", f"Document incorrectly routed to {agent_type}"
        state["calls"].append((provider, key, kwargs))
        class Model:
            def run(self, history, **kw):
                if state["error"]:
                    raise state["error"]
                system = kwargs["instructions"]
                content = [{"role": m.role, "content": m.content} for m in history]
                stage = "document_plan" if '"allowed_block_ids"' in content[0]["content"] else "document_repair" if len(content) > 1 else "document_block"
                unit = None
                if stage != "document_plan":
                    target = json.loads(content[0]["content"])["TARGET"]
                    # The source fixture has no inline locked elements in its two edited paragraphs.
                    unit = next(b.id for b in blocks if b.text.strip() == target.strip())
                spec = SimpleNamespace(stage=stage, unit=unit)
                return SimpleNamespace(content=state["model"](system, content, spec), metrics=None)
        return Model()

    monkeypatch.setattr(tpa, "create_agent", factory)
    for flag in ["task_routing_enabled", "provider_fallback_enabled"]:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": True, "user_keys": {}})))
    monkeypatch.setattr(api, "credit_manager", None)
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", False)
    monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
    return TestClient(api.app, raise_server_exceptions=False), state


def body(**kw):
    return {"type": "general", "query": INSTRUCTION, "document_text": DOC,
            "model": "openai", "api_key": "backend-key", **kw}


@pytest.mark.parametrize("kind", ["general", "enhance_with_ai"])
def test_process_edits_document_without_selection_and_returns_exact_patch_metadata(runtime, kind):
    client, state = runtime
    response = client.post("/agent/process", json=body(type=kind))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["document_edit"]["status"] == "proposed"
    assert len(result["document_edit"]["patches"]) == 2
    assert result["markdown_response"] == DOC.replace("very very useful", "particularly useful").replace("clear and clear", "clear and practical")
    assert result["markdown_response"].endswith("The final recommendation stays the same.\n")
    assert len(state["calls"]) == 3
    assert all(key == "backend-key" for _, key, _ in state["calls"])
    assert result["graph_suggestions"] == [] and result["design"] is None


def test_stream_sends_metadata_and_one_complete_verified_snapshot_then_done(runtime):
    client, _ = runtime
    response = client.post("/agent/stream", json=body())
    assert response.status_code == 200, response.text
    events = parse_sse_events(response.text)
    assert [e["event"] or e["data"].get("type") for e in events] == ["start", "document_edit_ready", "content", "done"]
    metadata = events[1]["data"]
    assert metadata["status"] == "proposed"
    text = events[2]["data"]["content"]
    assert fingerprint(text) == metadata["result_sha256"]
    assert metadata["base_sha256"] == fingerprint(DOC)


def test_apply_replays_accepted_patch_and_rejects_a_stale_snapshot(runtime):
    client, _ = runtime
    result = client.post("/agent/process", json=body()).json()
    edit = result["document_edit"]
    request = {"document_text": DOC, "base_sha256": edit["base_sha256"], "patches": edit["patches"]}
    response = client.post("/agent/documents/apply", json=request)
    assert response.status_code == 200
    assert response.json()["document_text"] == result["markdown_response"]
    assert response.json()["result_sha256"] == edit["result_sha256"]
    stale = client.post("/agent/documents/apply", json={**request, "document_text": DOC + "An intervening edit."})
    assert stale.status_code == 409


@pytest.mark.parametrize("changed", [
    {"type": "fix_grammar"}, {"query": ""}, {"query": "x" * 2001}, {"document_text": " "},
    {"document_text": "<html><body>content</body></html>"}, {"secondary_prompt": "another instruction"},
    {"existing_design_ast": {}}, {"image_provider": "openai"}, {"image_params": {"format": "png"}},
    {"document_text": "bad\x00document"}, {"document_text": "bad\ud800document"},
])
def test_invalid_or_conflicting_document_inputs_fail_before_provider_calls(runtime, changed):
    client, state = runtime
    response = client.post("/agent/process", content=json.dumps(body(**changed)), headers={"Content-Type": "application/json"})
    assert response.status_code == 400, response.text
    assert not state["calls"]


def test_duplicate_heading_clarification_keeps_the_original_and_makes_no_calls(runtime):
    client, state = runtime
    source = "## FAQ\n\nFirst answer.\n\n## FAQ\n\nSecond answer.\n"
    result = client.post("/agent/process", json=body(query="Improve FAQ", document_text=source)).json()
    assert result["markdown_response"] == source
    assert result["document_edit"]["status"] == "needs_clarification"
    assert not state["calls"]


def test_provider_error_does_not_leak_details_or_emit_a_partial_edit(runtime):
    client, state = runtime
    state["error"] = RuntimeError("private provider detail")
    result = client.post("/agent/process", json=body()).json()
    assert result["markdown_response"] == DOC
    assert result["document_edit"]["status"] == "failed"
    assert result["document_edit"]["patches"] == []
    assert "private provider detail" not in json.dumps(result)


def test_later_provider_failure_discards_earlier_successful_replacements(runtime):
    client, state = runtime
    state["model"].replies[prose()[1].id] = RuntimeError("private later failure")
    result = client.post("/agent/process", json=body()).json()
    assert len(state["calls"]) == 3
    assert result["document_edit"]["status"] == "failed"
    assert not result["document_edit"]["patches"]
    assert result["markdown_response"] == DOC


def test_invalid_planner_json_does_not_rewrite_anything(runtime):
    client, state = runtime
    state["model"] = lambda *args: "not JSON"
    result = client.post("/agent/process", json=body()).json()
    assert result["document_edit"]["status"] == "failed"
    assert result["markdown_response"] == DOC


@pytest.mark.parametrize("byok", [True, False])
def test_document_requests_respect_provider_eligibility_and_fallback(runtime, monkeypatch, byok):
    from src.task_router import KeyResolver
    _, state = runtime
    original_factory = tpa.create_agent
    attempted = []
    def factory(agent_type, provider, *args, **kw):
        attempted.append(provider)
        if provider == "openai":
            raise RuntimeError("401 invalid api key")
        return original_factory(agent_type, provider, *args, **kw)
    monkeypatch.setattr(tpa, "create_agent", factory)
    monkeypatch.setattr(settings, "provider_fallback_enabled", True)
    keys = {"anthropic_api_key": "own-claude"} if byok else {"openai_api_key": "platform-o", "anthropic_api_key": "platform-a"}
    resolver = KeyResolver(keys if byok else {}, None if byok else keys,
                          lambda **kw: (kw["user_keys"] if byok else kw["default_keys"]).get(
                              "anthropic_api_key" if kw["model_provider"] == "claude" else kw["model_provider"] + "_api_key"))
    output = list(tpa.process_user_query_stream(INSTRUCTION, model_provider="openai", api_key="unused",
                  document_text=DOC, user_text=INSTRUCTION, key_resolver=resolver, is_byok=byok, report_model=True))
    metadata = next(c["__event__"]["data"] for c in output if isinstance(c, dict) and "__event__" in c)
    assert metadata["status"] == "proposed"
    assert output[0]["__model_used__"][0] == "claude"
    assert attempted == (["claude"] * 3 if byok else ["openai"] + ["claude"] * 3)
    if byok:
        assert all(key == "own-claude" for _, key, _ in state["calls"])


def test_body_ending_in_a_protected_fence_is_not_stripped(runtime):
    client, _ = runtime
    source = DOC + "\n```json\n{\"chart\": 20}\n```"
    result = client.post("/agent/process", json=body(document_text=source)).json()
    assert result["document_edit"]["status"] == "proposed"
    assert result["markdown_response"].endswith("```json\n{\"chart\": 20}\n```")


def test_document_mode_bypasses_code_image_and_design_routing(runtime, monkeypatch):
    client, state = runtime
    def forbidden(*a, **kw):
        pytest.fail("Document instruction was routed as a separate image/code/design request")
    monkeypatch.setattr(tpa, "_classify_image_request", forbidden)
    monkeypatch.setattr("src.request_plan.plan_smart_request", forbidden)
    monkeypatch.setattr("src.request_plan.plan_design_request", forbidden)
    state_instruction = "Improve Introduction and Benefits; do not generate code or create an image."
    state["model"].instruction = state_instruction
    response = client.post("/agent/stream", json=body(query=state_instruction))
    assert response.status_code == 200
    assert '"status": "proposed"' in response.text


def test_apply_rejects_forged_or_unbounded_payloads(runtime):
    client, _ = runtime
    request = {"document_text": DOC, "base_sha256": fingerprint(DOC), "patches": []}
    assert client.post("/agent/documents/apply", json=request).status_code == 200
    for bad in [{"document_text": ""}, {"base_sha256": "wrong"}, {"patches": [{}]}, {"unknown": True},
                {"document_text": "bad\ud800document"}]:
        response = client.post("/agent/documents/apply", content=json.dumps({**request, **bad}), headers={"Content-Type": "application/json"})
        assert response.status_code == 400


def test_stream_billing_counts_document_context_not_just_short_instruction(runtime, monkeypatch):
    client, _ = runtime
    charges = []
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
    monkeypatch.setattr(api, "fire_credit_deduction", lambda *a, **kw: charges.append((a, kw)))
    response = client.post("/agent/stream", json=body())
    assert response.status_code == 200
    (args, _), = charges
    assert args[1] == len(INSTRUCTION) + len(DOC)
    assert args[2] == len(DOC.replace("very very useful", "particularly useful").replace("clear and clear", "clear and practical"))


@pytest.mark.parametrize("limit", ["balance", "request"])
def test_document_credit_rejection_happens_before_model_work(runtime, monkeypatch, limit):
    client, state = runtime
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
    reserves = []
    def reserve(user, amount):
        reserves.append(amount)
        return False, 0
    monkeypatch.setattr(api, "credit_manager", SimpleNamespace(reserve=reserve))
    if limit == "request":
        monkeypatch.setattr(api, "MAX_RESERVATION_PER_REQUEST_SCALED", 1)
    response = client.post("/agent/stream", json=body())
    assert response.status_code == 402
    assert not state["calls"]
    if limit == "request":
        assert response.json()["code"] == "REQUEST_CREDIT_LIMIT"
        assert not reserves
    else:
        assert reserves == [api.text_credits_scaled(len(INSTRUCTION) + len(DOC), len(DOC))]
