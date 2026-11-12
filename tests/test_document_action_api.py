"""document_action "auto" through the real endpoints: which mode ran is reported, nothing existing changes."""
import json
from unittest.mock import AsyncMock

import pytest

import api
import src.document_edit.runtime as edit_runtime
from config import settings
from tests.design.test_design_api import parse_sse_events
from tests.document_edit.test_service import ScriptedEditor
from tests.test_document_context_api import SOURCE, body, runtime  # noqa: F401  (fixture)


@pytest.fixture
def classifier(monkeypatch):
    """Replaces the light-tier provider call; records what it was asked and answers with state['answer']."""
    state = {"answer": "write", "calls": []}

    def fake(text, provider, key, timeout, request_id=None, system=None, max_tokens=5, sdk_deadline_floor=0.0):
        state["calls"].append({"text": text, "provider": provider, "key": key, "timeout": timeout, "system": system})
        if isinstance(state["answer"], Exception):
            raise state["answer"]
        return state["answer"]
    monkeypatch.setattr("src.intent_classifier._call", fake)
    return state


def names(events):
    return [e["event"] or e["data"].get("type") for e in events]


def resolved_of(process):
    return process["document_action_resolved"]


# ---- decided by the wording: no classifier call, one report ------------------------------------------------------------------------

@pytest.mark.parametrize("query, action, reason", [
    ("What is the launch codename?", "ask", "question_or_advice"),
    ("add a short note about the cohort below the Conclusion heading", "draft", "new_content"),
])
def test_clear_wording_picks_the_mode_without_a_classifier_call_and_reports_it(runtime, classifier, query, action, reason):
    client, state = runtime
    result = client.post("/agent/process", json=body(document_action="auto", query=query)).json()
    assert resolved_of(result) == {"requested_action": "auto", "action": action, "source": "rules", "reason": reason, "ambiguous": False}
    assert result["document_context"]["action"] == action and result["document_edit"] is None
    assert not classifier["calls"] and len(state["calls"]) == 1


def test_stream_reports_the_mode_before_anything_else_that_mode_sends(runtime, classifier):
    client, state = runtime
    events = parse_sse_events(client.post("/agent/stream", json=body(document_action="auto")).text)
    assert names(events) == ["start", "document_action_resolved", "document_context_ready", "content", "done"]
    assert events[1]["data"] == {"requested_action": "auto", "action": "ask", "source": "rules", "reason": "question_or_advice", "ambiguous": False}
    assert events[2]["data"]["action"] == "ask"
    assert not classifier["calls"]


def test_exact_text_through_auto_is_still_free_and_instant(runtime, classifier):
    client, state = runtime
    request = body(document_action="auto", query="add Acme Inc. below the Conclusion heading")
    result = client.post("/agent/process", json=request).json()
    assert result["markdown_response"] == "Acme Inc." and resolved_of(result)["action"] == "draft"
    assert not state["calls"] and not state["charges"]


# ---- Edit through auto -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def edit_model(monkeypatch):
    """The Edit pipeline with a scripted model: every title edit becomes '# Proposal (2024)'."""
    seen = []

    class Llm:
        used = ("openai", "gpt-test")

        def __init__(self, *args, **kwargs):
            pass

        def drafting_model(self):
            return self.used

        def __call__(self, system, messages, spec):
            seen.append(spec.stage)
            from src.document_edit.blocks import document_blocks
            title = next(b for b in document_blocks(SOURCE) if b.kind == "heading")
            return ScriptedEditor("add (2024) next to the title", [title], {title.id: "# Proposal (2024)\n"})(system, messages, spec)
    monkeypatch.setattr(edit_runtime, "AgnoLLM", Llm)
    return seen


def test_auto_can_choose_edit_and_both_endpoints_report_it(runtime, classifier, edit_model):
    client, state = runtime
    request = body(document_action="auto", query="add (2024) next to the title")
    result = client.post("/agent/process", json=request).json()
    assert resolved_of(result)["action"] == "edit" and result["document_edit"]["status"] == "proposed"
    assert result["document_edit"]["patches"][0]["after"] == "# Proposal (2024)\n"
    events = parse_sse_events(client.post("/agent/stream", json=request).text)
    assert names(events) == ["start", "document_action_resolved", "document_edit_ready", "content", "done"]
    assert not classifier["calls"]


def test_auto_edit_gets_the_checks_an_explicit_edit_gets(runtime, classifier, edit_model):
    client, state = runtime
    many = "\n\n".join(f"Paragraph {i}." for i in range(401)) + "\n"
    for path in ("/agent/process", "/agent/stream"):
        response = client.post(path, json=body(document_action="auto", query="make the first paragraph shorter", document_context={"text": many}))
        assert response.status_code == 400 and "too many blocks" in response.json()["error"]
    assert not edit_model and not state["calls"]


# ---- unclear wording: one light-tier call -------------------------------------------------------------------------------------------

def test_unclear_wording_asks_the_classifier_with_the_request_provider_and_key_on_process(runtime, classifier):
    client, state = runtime
    classifier["answer"] = "draft"
    result = client.post("/agent/process", json=body(document_action="auto", query="Write a closing sentence for the Conclusion")).json()
    assert resolved_of(result) == {"requested_action": "auto", "action": "draft", "source": "model", "reason": "model_choice", "ambiguous": False}
    assert result["document_context"]["action"] == "draft"
    call, = classifier["calls"]
    assert (call["provider"], call["key"]) == ("openai", "endpoint-key")
    assert call["text"].endswith("Write a closing sentence for the Conclusion") and "Velvet Mango" not in call["text"]
    assert 0 < call["timeout"] <= settings.document_action_classifier_timeout_seconds      # what is left after any wait for a slot


def test_stream_runs_the_classifier_after_the_licence_check_with_the_resolved_key(runtime, classifier):
    client, state = runtime
    classifier["answer"] = "draft"
    events = parse_sse_events(client.post("/agent/stream", json=body(document_action="auto", query="Write a closing sentence for the Conclusion")).text)
    assert names(events) == ["start", "document_action_resolved", "document_context_ready", "content", "done"]
    assert events[1]["data"]["source"] == "model" and events[2]["data"]["action"] == "draft"
    call, = classifier["calls"]
    assert (call["provider"], call["key"]) == ("openai", "provider-key")


def test_a_platform_user_text_goes_through_the_platform_claude_key_only(runtime, classifier, monkeypatch):
    client, state = runtime
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("platform-openai", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
    monkeypatch.setitem(api.DEFAULT_API_KEYS, "anthropic_api_key", "platform-claude")
    classifier["answer"] = "draft"
    client.post("/agent/stream", json=body(document_action="auto", query="Write a closing sentence for the Conclusion"))
    call, = classifier["calls"]
    assert (call["provider"], call["key"]) == ("claude", "platform-claude")


def test_a_byok_user_with_own_keys_is_never_sent_to_the_platform_key(runtime, classifier, monkeypatch):
    client, state = runtime
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("own-openai", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {"use_own_keys": True}})))
    monkeypatch.setitem(api.DEFAULT_API_KEYS, "anthropic_api_key", "platform-claude")
    classifier["answer"] = "draft"
    client.post("/agent/stream", json=body(document_action="auto", query="Write a closing sentence for the Conclusion"))
    call, = classifier["calls"]
    assert (call["provider"], call["key"]) == ("openai", "own-openai")


@pytest.mark.parametrize("answer", [RuntimeError("provider down"), "nonsense", "", None])
@pytest.mark.parametrize("path", ["/agent/process", "/agent/stream"])
def test_when_the_classifier_cannot_answer_the_request_becomes_a_read_only_ask(runtime, classifier, path, answer):
    client, state = runtime
    classifier["answer"] = answer
    response = client.post(path, json=body(document_action="auto", query="Write a closing sentence for the Conclusion"))
    assert response.status_code == 200
    if path == "/agent/process":
        resolved, context = resolved_of(response.json()), response.json()["document_context"]
    else:
        events = parse_sse_events(response.text)
        resolved, context = events[1]["data"], events[2]["data"]
    assert resolved == {"requested_action": "auto", "action": "ask", "source": "fallback", "reason": "classifier_unavailable", "ambiguous": True}
    assert context["action"] == "ask" and context["response_kind"] == "chat_answer"


def test_the_classifier_can_be_switched_off_and_unclear_wording_then_asks(runtime, classifier, monkeypatch):
    client, state = runtime
    monkeypatch.setattr(settings, "document_action_classifier_enabled", False)
    result = client.post("/agent/process", json=body(document_action="auto", query="Write a closing sentence for the Conclusion")).json()
    assert resolved_of(result)["source"] == "fallback" and not classifier["calls"]


# ---- Write: the plain prompt, without the document -----------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/agent/process", "/agent/stream"])
def test_auto_can_choose_write_and_the_document_is_not_sent_to_it(runtime, classifier, path):
    client, state = runtime
    classifier["answer"] = "write"
    response = client.post(path, json=body(document_action="auto", query="Write a short essay about cats"))
    assert response.status_code == 200
    call, = state["calls"]
    assert call["prompt"] == "Write a short essay about cats" and "Velvet Mango" not in call["prompt"]
    if path == "/agent/process":
        result = response.json()
        assert resolved_of(result)["action"] == "write" and result["document_context"] is None and result["document_edit"] is None
        assert result["needs_context"] is None
    else:
        events = parse_sse_events(response.text)
        assert names(events)[:2] == ["start", "document_action_resolved"] and events[1]["data"]["action"] == "write"
        assert "document_context_ready" not in names(events)


# ---- requests only Ask can serve ------------------------------------------------------------------------------------------------------

def test_auto_with_a_scope_is_resolved_to_ask_instead_of_rejected(runtime, classifier):
    client, state = runtime
    selection = {"start_utf16": SOURCE.index("Velvet"), "end_utf16": SOURCE.index("Velvet") + 12}
    request = body(document_action="auto", document_scope="selection", query="rewrite this",
                   document_context={"text": SOURCE, "selection": selection})
    result = client.post("/agent/process", json=request).json()
    assert resolved_of(result) == {"requested_action": "auto", "action": "ask", "source": "request", "reason": "scope_requires_ask", "ambiguous": False}
    assert result["document_context"]["scope"] == "selection"
    assert not classifier["calls"]


def test_the_scope_rule_for_explicit_edit_and_draft_is_unchanged(runtime):
    client, _ = runtime
    for action in ("edit", "draft"):
        response = client.post("/agent/process", json=body(document_action=action, document_scope="focused"))
        assert response.status_code == 400 and "Scoped retrieval is only available for Ask" in response.json()["error"]


# ---- compatibility --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("action", [None, "ask", "draft"])
def test_explicit_actions_and_the_default_send_nothing_new(runtime, classifier, action):
    client, state = runtime
    request = body(**({"document_action": action} if action else {}))
    result = client.post("/agent/process", json=request).json()
    assert result["document_action_resolved"] is None
    events = parse_sse_events(client.post("/agent/stream", json=request).text)
    assert "document_action_resolved" not in names(events) and not classifier["calls"]


def test_auto_still_needs_a_document_and_other_values_are_still_rejected(runtime):
    client, _ = runtime
    request = body(document_action="auto")
    request.pop("document_context")
    assert client.post("/agent/process", json=request).status_code == 400
    assert client.post("/agent/process", json=body(document_action="automatic")).status_code == 400
    assert client.post("/agent/process", json=body(document_action="auto", query="x" * 8001)).status_code == 400


def test_the_decision_replaces_the_advisory_suggestion(runtime, classifier):
    client, _ = runtime
    request = body(document_action="auto", suggest_document_intent=True)
    result = client.post("/agent/process", json=request).json()
    assert result["document_intent"] is None and resolved_of(result)["action"] == "ask"
    assert "document_intent_suggested" not in names(parse_sse_events(client.post("/agent/stream", json=request).text))
    # With an explicit action the suggestion is exactly what it was.
    explicit = client.post("/agent/process", json=body(document_action="ask", suggest_document_intent=True)).json()
    assert explicit["document_intent"]["requested_action"] == "ask"


@pytest.mark.parametrize("path", ["/agent/process", "/agent/stream"])
def test_a_disabled_feature_is_a_503_before_any_classifier_call(runtime, classifier, monkeypatch, path):
    client, state = runtime
    monkeypatch.setattr(settings, "document_context_enabled", False)
    response = client.post(path, json=body(document_action="auto", query="Write a closing sentence for the Conclusion"))
    assert response.status_code == 503 and response.json()["code"] == "DOCUMENT_CONTEXT_DISABLED"
    assert not classifier["calls"] and not state["calls"]


# ---- fixes from the regression audit ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/agent/process", "/agent/stream"])
def test_the_task_text_is_never_logged_when_auto_resolves_to_write(runtime, classifier, caplog, path):
    import logging
    client, state = runtime
    classifier["answer"] = "write"
    with caplog.at_level(logging.INFO):
        response = client.post(path, json=body(document_action="auto", query="Write a short essay about ZQXQUERYMARK the lighthouse"))
    assert response.status_code == 200
    assert "ZQXQUERYMARK" not in caplog.text and "[document task]" in caplog.text


def test_the_task_text_is_never_logged_for_a_write_decided_by_the_rules(runtime, classifier, caplog):
    import logging
    client, state = runtime
    with caplog.at_level(logging.INFO):
        client.post("/agent/process", json=body(document_action="auto", query="Write a Python function named ZQXQUERYMARK that checks an email address"))
    assert "ZQXQUERYMARK" not in caplog.text


def test_a_provider_error_after_auto_resolves_to_write_is_as_generic_as_for_a_document_task(runtime, classifier):
    from src.document_context import CONTEXT_ERROR
    client, state = runtime
    classifier["answer"] = "write"
    state["error"] = RuntimeError("provider said: your prompt 'Write a short essay about cats' was rejected")
    response = client.post("/agent/process", json=body(document_action="auto", query="Write a short essay about cats"))
    assert response.status_code == 500 and response.json() == {"error": CONTEXT_ERROR}


def test_the_rules_step_before_the_licence_check_is_cheap_for_a_huge_query(runtime):
    import time
    client, _ = runtime
    from tests.test_document_context_api import body as make_body
    request = api.StreamRequest(**make_body(document_action="auto", query="write a blog post about " * 330))
    started = time.perf_counter()
    api.auto_action_by_rules(request)
    request = api.StreamRequest(**make_body(document_action="auto", query="add " + "1" * 7990))
    api.auto_action_by_rules(request)
    assert time.perf_counter() - started < 0.5
