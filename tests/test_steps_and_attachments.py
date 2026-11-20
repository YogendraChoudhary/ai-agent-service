"""
Frontend request of 2026-11-20: (1) several steps in one Auto prompt, (2) images and reference files with Ask, Draft, Edit
and Write. Real endpoints, routing and pipeline; only the model calls and account services are replaced.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agno.agent import RunEvent
from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.attachments import MAX_REFERENCE_CHARS, VISION_INPUT_CHARS_PER_IMAGE, ReferenceDocument
from src.document_context import DocumentContext, prepare_context, with_attachments
from src.document_steps import Step, plan_steps, run_steps, split_parts
from tests.design.test_design_api import parse_sse_events

DOC = ("# Guide\n\n## Introduction\n\nThis introduction is very very long and useful for readers.\n\n"
       "## Pricing\n\nThe price is USD 20.\n")
CDN = settings.cdn_url.rstrip("/")
IMAGE = f"{CDN}/u/uploads/chart.png"
FAQ = "## FAQ\n\n**Is there a trial?** Yes, ask our team."


# ---------------------------------------------------------------------------------------------------------- the fake model

class Models:
    """Stands in for every model call: Edit's planner and block writer (history in, one reply out) and Ask/Draft (streamed)."""

    def __init__(self):
        self.prompts, self.images, self.fail_edit = [], [], False

    def run(self, agent, prompt, stream=False, images=None, **kwargs):
        self.images.append(images)
        if isinstance(prompt, list):                         # AgnoLLM: Edit's plan and block calls
            content = prompt[0].content
            self.prompts.append(content)
            if self.fail_edit:
                raise RuntimeError("401 invalid api key")
            data = json.loads(content)
            if "allowed_block_ids" in data:
                intro = next(b["id"] for b in data["document"] if "introduction is" in b["text"])
                reply = json.dumps({"status": "ready", "message": "The introduction can be shorter.", "targets": [
                    {"block_id": intro, "request_quote": data["USER_INSTRUCTION"][:40], "reason": "Wordy."}]})
            else:
                reply = data["TARGET"].replace("very very long and ", "")
            return SimpleNamespace(content=reply, metrics=None, status=None, event=None)
        self.prompts.append(prompt)

        def events():
            yield SimpleNamespace(event=RunEvent.run_content, content=FAQ if "FAQ" in prompt else "It costs USD 20.", metrics=None)
            yield SimpleNamespace(event=RunEvent.run_completed, content=None, metrics={"input_tokens": 10, "output_tokens": 5})
        return events() if stream else SimpleNamespace(content="It costs USD 20.", metrics=None, status=None, event=None)


@pytest.fixture
def runtime(monkeypatch):
    models = Models()
    state = {"models": models, "charges": [], "reserves": [], "releases": []}
    monkeypatch.setattr(tpa.Agent, "run", lambda agent, prompt, **kw: models.run(agent, prompt, **kw))
    for flag in ["task_routing_enabled", "provider_fallback_enabled"]:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "document_context_enabled", True)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": True, "user_keys": {}})))
    monkeypatch.setattr(api, "credit_manager", None)
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", False)
    monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
    monkeypatch.setattr(api, "fire_credit_deduction", lambda *a, **kw: state["charges"].append((a, kw)))
    monkeypatch.setattr(api, "fire_credit_deduction_and_release", lambda **kw: state["charges"].append(kw))
    return TestClient(api.app, raise_server_exceptions=False), state


def body(query, **kw):
    return {"type": "general", "query": query, "model": "openai", "api_key": "k", "document_context": {"text": DOC}, **kw}


def named(events, name):
    return [e["data"] for e in events if e["event"] == name]


# ---------------------------------------------------------------------------------------------------------- 1. steps

class TestSplitAndPlan:
    @pytest.mark.parametrize("query, parts", [
        ("Shorten the introduction and add a FAQ section at the end", ["Shorten the introduction", "add a FAQ section at the end"]),
        ("Add a sentence about refunds below Pricing. Then translate the introduction to Spanish.",
         ["Add a sentence about refunds below Pricing", "translate the introduction to Spanish"]),
        ("What is the price? And add a note about discounts below Pricing", ["What is the price?", "add a note about discounts below Pricing"]),
    ])
    def test_several_tasks_split_at_request_verbs(self, query, parts):
        assert split_parts(query) == parts

    @pytest.mark.parametrize("query", ["Fix typos and grammar", "Translate to French and Spanish", "Make it shorter and friendlier",
                                       "Add a FAQ section and make it shorter", "Change 'add a FAQ and shorten it' to 'x'"])
    def test_one_task_is_not_split(self, query):
        assert len(split_parts(query)) == 1 and plan_steps(query, DOC) is None

    def test_plan_gives_each_part_its_action_and_merges_neighbouring_edits(self):
        steps = plan_steps("Shorten the introduction and add a FAQ section at the end", DOC)
        assert [(s.action, s.instruction) for s in steps] == [("edit", "Shorten the introduction"),
                                                             ("draft", "add a FAQ section at the end")]
        assert plan_steps("Rewrite the intro and make the pricing paragraph punchier", DOC) is None     # one Edit, as before

    def test_more_than_three_steps_keeps_the_single_action_path(self):
        assert plan_steps("Shorten the intro, add a FAQ at the end, and add a title note at the top, then fix typos", DOC) is None

    def test_an_unclear_part_without_a_classifier_is_not_split(self):
        assert plan_steps("Shorten the introduction and the vibe could be better somehow", DOC) is None


class TestRunSteps:
    def _nested(self, fail_first=False):
        calls = []

        def nested(query, **inputs):
            calls.append((query, inputs))
            if "document_text" in inputs:
                if fail_first:
                    raise RuntimeError("401 invalid api key")
                source = inputs["document_text"]
                after = source.replace("very very long and ", "")
                block_start = source.index("This introduction")
                block_end = source.index("\n\n## Pricing")
                from src.document_edit.blocks import fingerprint, utf16_length
                yield {"__event__": {"event": "document_edit_ready", "data": {
                    "status": "proposed", "message": "Shortened.", "base_sha256": fingerprint(source),
                    "result_sha256": fingerprint(after), "targets": [], "insertions": [], "patches": [{
                        "block_id": "b4_000000000000", "start_utf16": utf16_length(source[:block_start]),
                        "end_utf16": utf16_length(source[:block_end]), "before": source[block_start:block_end],
                        "after": after[block_start:after.index("\n\n## Pricing")], "reason": "Wordy."}]}}}
                yield after
            else:
                yield {"__event__": {"event": "document_context_ready", "data": inputs["prepared_context"].metadata.model_dump(mode="json")}}
                yield FAQ
        return nested, calls

    def _steps(self):
        return [Step(1, "edit", "Shorten the introduction", "Shorten the introduction", "rules"),
                Step(2, "draft", "add a FAQ section at the end", "Add a FAQ section at the end", "rules")]

    def test_each_step_runs_on_the_document_after_the_one_before(self):
        nested, calls = self._nested()
        chunks = list(run_steps(self._steps(), DOC, None, nested))
        events = [c["__event__"] for c in chunks if isinstance(c, dict) and "__event__" in c]
        assert [e["event"] for e in events] == ["document_step_started", "document_step_finished", "document_step_started",
                                                "document_step_finished", "document_steps_ready"]
        first, second, ready = events[1]["data"], events[3]["data"], events[4]["data"]
        assert first["status"] == "proposed" and first["document_edit"]["patches"]
        assert second["status"] == "proposed" and second["markdown"] == FAQ
        assert "very very" not in calls[1][1]["prepared_context"].prompt          # the draft saw the edited document
        assert ready["document_text"].endswith(FAQ + "\n") or ready["document_text"].rstrip().endswith(FAQ)
        assert second["base_sha256"] == first["result_sha256"]
        billing = next(c["__steps_billing__"] for c in chunks if isinstance(c, dict) and "__steps_billing__" in c)
        assert billing["output_chars"] == len(first["document_edit"]["patches"][0]["after"]) + len(FAQ)

    def test_a_failed_step_does_not_stop_the_others(self):
        nested, _ = self._nested(fail_first=True)
        events = [c["__event__"]["data"] for c in run_steps(self._steps(), DOC, None, nested)
                  if isinstance(c, dict) and c.get("__event__", {}).get("event") == "document_step_finished"]
        assert events[0]["status"] == "failed" and events[0]["error"]
        assert events[1]["status"] == "proposed" and events[1]["markdown"] == FAQ

    def test_photo_dict_is_handed_over_first(self):
        nested, _ = self._nested()
        first = next(iter(run_steps(self._steps(), DOC, None, nested)))
        assert "__pending_images__" in first


class TestStepsOnTheEndpoints:
    def test_stream_reports_each_step_and_its_own_result(self, runtime):
        client, state = runtime
        events = parse_sse_events(client.post("/agent/stream", json=body(
            "Shorten the introduction and add a FAQ section at the end", document_action="auto")).text)
        resolved, = named(events, "document_action_resolved")
        assert resolved["action"] == "steps" and [s["action"] for s in resolved["steps"]] == ["edit", "draft"]
        started, finished, ready = named(events, "document_step_started"), named(events, "document_step_finished"), \
            named(events, "document_steps_ready")
        assert [(s["step"], s["total"]) for s in started] == [(1, 2), (2, 2)]
        assert [f["status"] for f in finished] == ["proposed", "proposed"]
        assert finished[0]["document_edit"]["patches"] and finished[1]["markdown"] == FAQ
        assert "very very" not in ready[0]["document_text"] and FAQ in ready[0]["document_text"]
        assert not [e for e in events if e["data"].get("type") == "content"]          # results come in the step events
        assert events[-1]["data"]["type"] == "done"

    def test_single_task_auto_is_unchanged(self, runtime):
        client, _ = runtime
        events = parse_sse_events(client.post("/agent/stream", json=body("What is the price?", document_action="auto")).text)
        resolved, = named(events, "document_action_resolved")
        assert resolved["action"] == "ask" and not named(events, "document_step_started")

    def test_process_returns_every_step(self, runtime):
        client, _ = runtime
        result = client.post("/agent/process", json=body("Shorten the introduction and add a FAQ section at the end",
                                                         document_action="auto")).json()
        assert result["document_action_resolved"]["action"] == "steps"
        assert [s["action"] for s in result["document_steps"]["steps"]] == ["edit", "draft"]
        assert result["markdown_response"] == result["document_steps"]["document_text"]

    def test_paid_steps_are_charged_once_for_what_they_produced(self, runtime, monkeypatch):
        client, state = runtime
        monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
            "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
        monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
        monkeypatch.setattr(api, "credit_manager", SimpleNamespace(
            reserve=lambda user, amount: (state["reserves"].append(amount), (True, 99999))[1],
            schedule_release=lambda *a, **kw: state["releases"].append((a, kw))))
        events = parse_sse_events(client.post("/agent/stream", json=body(
            "Shorten the introduction and add a FAQ section at the end", document_action="auto")).text)
        finished = named(events, "document_step_finished")
        charge, = state["charges"]
        assert charge["success"] is True
        assert charge["output_chars"] == len(finished[0]["document_edit"]["patches"][0]["after"]) + len(FAQ)
        assert charge["reserved_scaled"] == sum(state["reserves"])


# ---------------------------------------------------------------------------------------------------------- 2. attachments

REF = ReferenceDocument(name="brief.docx", text="The launch date is 3 March. Offer a 14-day trial.")


class TestAttachments:
    def test_ask_and_draft_get_a_read_only_reference_block_and_count_it(self):
        prepared = prepare_context(DocumentContext(text=DOC), "What does the brief say about trials?", "ask")
        attached = with_attachments(prepared, [REF], 1)
        assert attached.prompt.startswith("REFERENCE_MATERIAL (read-only;") and "14-day trial" in attached.prompt
        assert attached.prompt.endswith(prepared.prompt)
        assert attached.metadata.billable_input_chars == prepared.metadata.billable_input_chars + len(REF.text) + VISION_INPUT_CHARS_PER_IMAGE
        assert attached.metadata.attachments == {"images": 1, "references": [{"name": "brief.docx", "chars": len(REF.text)}]}

    def test_stream_ask_with_an_image_and_a_reference(self, runtime):
        client, state = runtime
        events = parse_sse_events(client.post("/agent/stream", json=body(
            "what does this chart show?", document_action="ask", image_urls=[IMAGE],
            references=[REF.model_dump()])).text)
        context, = named(events, "document_context_ready")
        assert context["attachments"]["images"] == 1 and context["attachments"]["references"][0]["name"] == "brief.docx"
        prompt = state["models"].prompts[-1]
        assert "REFERENCE_MATERIAL" in prompt and "14-day trial" in prompt
        assert state["models"].images[-1]                                           # the image reached the model

    def test_edit_planner_and_writer_see_the_reference_and_the_image(self, runtime):
        client, state = runtime
        events = parse_sse_events(client.post("/agent/stream", json=body(
            "Shorten the introduction", document_action="edit", image_urls=[IMAGE], references=[REF.model_dump()])).text)
        assert named(events, "document_edit_ready")[0]["status"] == "proposed"
        plan = json.loads(state["models"].prompts[0])
        assert plan["REFERENCE_MATERIAL"][0]["name"] == "brief.docx"
        assert all(state["models"].images)                                          # every Edit call got the image

    def test_write_takes_reference_files(self, runtime):
        client, state = runtime
        events = parse_sse_events(client.post("/agent/stream", json={
            "type": "general", "query": "Summarise this brief", "model": "openai", "api_key": "k",
            "references": [REF.model_dump()]}).text)
        assert not named(events, "needs_context")                                   # "this brief" is the attachment
        assert "REFERENCE_MATERIAL" in state["models"].prompts[-1] and "USER_TASK:\nSummarise this brief" in state["models"].prompts[-1]

    @pytest.mark.parametrize("change, status", [
        ({"references": [{"name": f"f{i}", "text": "x"} for i in range(4)]}, 400),
        ({"references": [{"name": "big", "text": "x" * (MAX_REFERENCE_CHARS + 1)}]}, 400),
        ({"document_analysis": "summary", "image_urls": [IMAGE]}, 400),
        ({"type": "fix_grammar", "references": [{"name": "a", "text": "b"}], "document_context": None}, 400),
    ])
    def test_limits_are_enforced(self, runtime, change, status):
        client, _ = runtime
        request = {**body("What does it say?"), **change}
        if request.get("document_context") is None:
            request.pop("document_context")
        assert client.post("/agent/stream", json=request).status_code == status

    def test_an_image_read_by_ask_holds_no_image_generation_credits(self, runtime, monkeypatch):
        client, state = runtime
        monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
            "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
        monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
        monkeypatch.setattr(api, "credit_manager", SimpleNamespace(
            reserve=lambda user, amount: (state["reserves"].append(amount), (True, 99999))[1],
            schedule_release=lambda *a, **kw: state["releases"].append((a, kw))))
        client.post("/agent/stream", json=body("what does this chart show?", document_action="ask", image_urls=[IMAGE]))
        charge, = state["charges"]
        # the image is read, so it is billed as input text (VISION_INPUT_CHARS_PER_IMAGE); no 5-credit image-generation hold on top
        assert charge["input_chars"] >= VISION_INPUT_CHARS_PER_IMAGE
        assert state["reserves"] == [api.text_credits_scaled(charge["input_chars"], 0)]
