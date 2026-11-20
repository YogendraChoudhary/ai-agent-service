"""
Edit with photos (src/document_edit/photos.py): "make it more appealing and add some photos" on an open document proposes the
text patches as before plus photos placed after the paragraphs they illustrate, described from the document. Before 2026-11-20
the request could only run in Write, which never sees the document: a generic page and three unrelated photos.
"""
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api
import src.document_edit.runtime as edit_runtime
import src.text_processing_agent as tpa
from config import settings
from src.document_action import decide_by_rules
from src.document_edit.blocks import document_blocks, fingerprint
from src.document_edit.photos import WHOLE_DOCUMENT_SCOPE, insertion_for, photo_request, plan_photos
from src.document_edit.service import apply_document_edits, edit_document
from tests.design.test_design_api import parse_sse_events
from tests.design.test_design_images import CDN, _fake_image_tool_builder
from tests.document_edit.test_service import DOC, ScriptedEditor, prose

TEXT_PART = "make the introduction more appealing"
REQUEST = f"{TEXT_PART} and add some photos"


def _photos_reply(*block_ids):
    return {"photos": [{"after_block_id": b, "prompt": f"A team reading a launch guide together, scene {i}, warm light",
                        "alt": f"Team reading the guide {i}"} for i, b in enumerate(block_ids)]}


class PhotoEditor(ScriptedEditor):
    """The scripted Edit model, plus the photo planner's answer."""
    def __init__(self, *args, photos=None, **kw):
        super().__init__(*args, **kw)
        self.photos = photos

    def __call__(self, system, messages, spec):
        if spec.stage == "document_photos":
            self.calls.append((spec, messages))
            return self.photos if isinstance(self.photos, str) else json.dumps(self.photos)
        return super().__call__(system, messages, spec)


class FakePhotoTool:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return str(uuid.uuid4())


def _editor(photos=None, plan=None):
    blocks = prose()
    return PhotoEditor(TEXT_PART, blocks[:1], {blocks[0].id: blocks[0].text.replace("very very useful", "genuinely helpful")},
                       plan=plan, photos=photos or _photos_reply(blocks[0].id, blocks[1].id))


class TestRequest:
    @pytest.mark.parametrize("query, count, text", [
        ("Make it more appealing to the users, add some photos and other aspects", 3,
         f"Make the document more appealing to the users. {WHOLE_DOCUMENT_SCOPE}"),
        ("add some photos", 3, None), ("Add a photo to the blog", 1, None), ("add a couple of images", 2, None),
        ("include a few relevant pictures throughout the article please", 3, None),
        ("Add photos of happy families playing games", 3, None),
        ("add 2 pictures and make the intro shorter", 2, "make the intro shorter"),
        ("Add photos to make it more engaging", 3, f"Make the document more engaging. {WHOLE_DOCUMENT_SCOPE}"),
        ("add photos and make the blog more fun", 3, f"Make the document more fun. {WHOLE_DOCUMENT_SCOPE}"),
    ])
    def test_photo_part_and_text_part(self, query, count, text):
        request = photo_request(query, document_blocks(DOC), cap=4)
        assert (request.count, request.text_instruction) == (count, text)

    def test_capped_and_counted_per_section(self):
        assert photo_request("Add 10 photos", [], cap=4).count == 4
        assert photo_request("Add 10 photos", [], cap=4).requested == 10
        assert photo_request("put an image in each section", document_blocks(DOC), cap=8).count == 4  # 4 ## headings

    @pytest.mark.parametrize("query", ["make the intro shorter", "Generate an image of a cat", "fix the typos",
                                       "describe the images in the document"])
    def test_no_photos_asked(self, query):
        assert photo_request(query, document_blocks(DOC), cap=4) is None


def test_after_a_list_the_photo_has_exactly_one_blank_line_on_each_side():
    source = "# T\n\nIntro text.\n\n- one\n- two\n\n## Next\n\nMore.\n"
    blocks = document_blocks(source)
    for block in (b for b in blocks if b.kind in ("paragraph", "list")):
        at, markdown = insertion_for(source, block, "Alt", "00000000-0000-4000-8000-000000000000")
        text = source[:block.end] + markdown + source[block.end:]
        assert "\n\n\n" not in text and [b.kind for b in document_blocks(text)].count("image") == 1


def test_parallel_rewrite_gives_the_same_proposal_as_in_order():
    from src.document_edit.service import _edit_document
    blocks = prose()
    replies = {b.id: b.text.replace("very very useful", "useful").replace("clear and clear", "clear") for b in blocks[:2]}
    one = _edit_document(DOC, "Improve the Introduction and Benefits", ScriptedEditor(
        "Improve the Introduction and Benefits", blocks[:2], replies))
    many = _edit_document(DOC, "Improve the Introduction and Benefits", ScriptedEditor(
        "Improve the Introduction and Benefits", blocks[:2], replies), parallel=True)
    assert one == many and one[1].status == "proposed" and len(one[1].patches) == 2


class TestPlan:
    def test_spots_must_be_prose_once_each_and_come_back_in_document_order(self):
        blocks = document_blocks(DOC)
        heading = next(b for b in blocks if b.kind == "heading")
        p1, p2 = prose()[:2]
        llm = PhotoEditor("x", photos=_photos_reply(p2.id, heading.id, p1.id, p2.id, "b99_000000000000"))
        spots = plan_photos("add photos", blocks, 3, llm)
        assert [s.block.id for s in spots] == [p1.id, p2.id]

    def test_alt_text_cannot_break_out_of_the_image(self):
        p1 = prose()[0]
        llm = PhotoEditor("x", photos={"photos": [{"after_block_id": p1.id, "prompt": "A calm office scene at dawn",
                                                    "alt": "x](javascript:alert(1)) ![y"}]})
        spot, = plan_photos("add a photo", document_blocks(DOC), 1, llm)
        assert "]" not in spot.alt and "(" not in spot.alt

    def test_a_paragraph_already_followed_by_an_image_is_not_offered(self):
        llm = PhotoEditor("x", photos={"photos": []})
        plan_photos("add photos", document_blocks(DOC), 2, llm)
        offered = json.loads(llm.calls[0][1][0]["content"])["allowed_block_ids"]
        assert set(offered) <= {p.id for p in prose()}          # prose only: no heading, table, quote, code or image


class TestEditWithPhotos:
    def test_text_patch_and_photos_in_one_verified_proposal(self):
        tool, llm = FakePhotoTool(), _editor()
        text, result = edit_document(DOC, REQUEST, llm, photo_tool=tool, max_photos=4)
        p1, p2 = prose()[:2]
        assert result.status == "proposed" and len(result.patches) == 1 and len(result.insertions) == 2
        assert "genuinely helpful" in text
        for insertion, block in zip(result.insertions, (p1, p2)):
            assert insertion.after_block_id == block.id and insertion.placeholder_id in text
            assert text.index(f"placeholder::{insertion.placeholder_id}") > text.index(block.text.strip()[-10:])
        assert apply_document_edits(DOC, result.base_sha256, result.patches, result.insertions) == text
        assert result.result_sha256 == fingerprint(text)
        assert [b.kind for b in document_blocks(text)].count("image") == 3          # the logo plus two photos
        assert all(p.endswith("No text, no lettering, no logos, no watermark.") for p in tool.prompts)
        assert "Added 2 photos" in result.message
        # the text edit saw only the text part of the request
        plan_call = next(m for s, m in llm.calls if s.stage == "document_plan")
        assert json.loads(plan_call[0]["content"])["USER_INSTRUCTION"] == TEXT_PART

    def test_photos_alone(self):
        tool = FakePhotoTool()
        text, result = edit_document(DOC, "add some photos", _editor(), photo_tool=tool, max_photos=4)
        assert result.status == "proposed" and not result.patches and len(result.insertions) == 2
        assert apply_document_edits(DOC, result.base_sha256, [], result.insertions) == text

    def test_text_part_that_cannot_be_done_still_gets_the_photos_and_says_why(self):
        plan = {"status": "needs_clarification", "message": "Which paragraph should change?", "targets": []}
        text, result = edit_document(DOC, REQUEST, _editor(plan=plan), photo_tool=FakePhotoTool(), max_photos=4)
        assert result.status == "proposed" and not result.patches and len(result.insertions) == 2
        assert "The wording was not changed: Which paragraph should change?" in result.message

    def test_no_usable_spot_and_no_text_change_is_no_change(self):
        text, result = edit_document(DOC, "add some photos", _editor(photos={"photos": []}), photo_tool=FakePhotoTool(),
                                     max_photos=4)
        assert result.status == "no_change" and text == DOC and not result.insertions

    def test_without_the_photo_tool_edit_behaves_as_before(self):
        text, result = edit_document(DOC, "add some photos", _editor())
        assert result.status == "needs_clarification" and text == DOC and not result.insertions


class TestApply:
    def _proposal(self):
        return edit_document(DOC, REQUEST, _editor(), photo_tool=FakePhotoTool(), max_photos=4)

    def test_insertion_with_the_finished_cdn_url_is_accepted(self):
        text, result = self._proposal()
        done = [i.model_copy(update={"markdown": i.markdown.replace(f"placeholder::{i.placeholder_id}",
                                                                    f"{CDN}/u/ai-images/{i.placeholder_id}.png")})
                for i in result.insertions]
        applied = apply_document_edits(DOC, result.base_sha256, result.patches, done)
        assert f"{CDN}/u/ai-images/" in applied and "placeholder::" not in applied

    @pytest.mark.parametrize("change", [
        {"markdown": "\n![x](https://evil.example/a.png)\n"},
        {"markdown": "\n![x](placeholder::00000000-0000-0000-0000-000000000000)\nInjected paragraph\n"},
        {"at_utf16": 3},
    ])
    def test_anything_but_one_image_line_at_a_block_end_is_refused(self, change):
        _, result = self._proposal()
        bad = [result.insertions[0].model_copy(update=change)]
        with pytest.raises(ValueError):
            apply_document_edits(DOC, result.base_sha256, result.patches, bad)

    def test_apply_endpoint_takes_insertions(self):
        text, result = self._proposal()
        client = TestClient(api.app)
        response = client.post("/agent/documents/apply", json={
            "document_text": DOC, "base_sha256": result.base_sha256,
            "patches": [p.model_dump() for p in result.patches],
            "insertions": [i.model_dump() for i in result.insertions]})
        assert response.status_code == 200, response.text
        assert response.json()["document_text"] == text


class TestAuto:
    @pytest.mark.parametrize("query", ["Make it more appealing to the users, add some photos and other aspects",
                                       "add some photos", "include a few pictures throughout"])
    def test_photo_requests_on_an_open_document_go_to_edit(self, query):
        decision = decide_by_rules(query, DOC)
        assert (decision.action, decision.reason) == ("edit", "document_photos")

    @pytest.mark.parametrize("query, action", [("Generate an image of a cat on a sofa", "write"),
                                               ("Write a blog post about remote work for small teams", "write")])
    def test_standalone_image_and_blog_requests_still_go_to_write(self, query, action):
        assert decide_by_rules(query, DOC).action == action

    def test_switched_off(self, monkeypatch):
        monkeypatch.setattr(settings, "document_edit_photos_enabled", False)
        decision = decide_by_rules("add some photos", DOC)
        assert decision is None or decision.reason != "document_photos"


class TestStream:
    """/agent/stream as the editor calls it: Auto, the document attached, nothing selected."""

    @pytest.fixture
    def client(self, monkeypatch):
        state = {"llm": _editor(), "images": []}
        monkeypatch.setattr(edit_runtime, "AgnoLLM", lambda *a, **k: _Used(state["llm"], k.get("request_id")))
        monkeypatch.setattr(tpa, "_build_image_tool", _fake_image_tool_builder(state["images"]))
        for flag in ["task_routing_enabled", "provider_fallback_enabled"]:
            monkeypatch.setattr(settings, flag, False)
        monkeypatch.setattr(settings, "document_context_enabled", True)
        monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
            "user_id": "u1", "requires_user_api_keys": True, "user_keys": {}})))
        monkeypatch.setattr(api, "credit_manager", None)
        monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", False)
        monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
        return TestClient(api.app, raise_server_exceptions=False), state

    def test_proposal_then_snapshot_then_one_image_ready_per_photo(self, client):
        client, state = client
        events = parse_sse_events(client.post("/agent/stream", json={
            "type": "general", "query": REQUEST, "model": "openai", "api_key": "k",
            "document_context": {"text": DOC}, "document_action": "auto"}).text)
        names = [e["event"] or e["data"].get("type") for e in events]
        resolved = next(e["data"] for e in events if e["event"] == "document_action_resolved")
        proposal = next(e["data"] for e in events if e["event"] == "document_edit_ready")
        snapshot = next(e["data"]["content"] for e in events if e["data"].get("type") == "content")
        ready = [e["data"] for e in events if e["data"].get("type") == "image_ready"]
        assert resolved["action"] == "edit" and resolved["reason"] == "document_photos"
        assert proposal["status"] == "proposed" and len(proposal["insertions"]) == 2
        assert all(f"placeholder::{i['placeholder_id']}" in snapshot for i in proposal["insertions"])
        assert {r["placeholder_id"] for r in ready} == {i["placeholder_id"] for i in proposal["insertions"]}
        assert names.index("document_edit_ready") < names.index("done") < names.index("image_ready")
        assert all(r["aspect"] == "16:9" for r in state["images"])
        assert state["images"][0]["user_query"] == REQUEST

    def test_process_returns_the_finished_photo_urls_in_snapshot_and_insertions(self, client):
        client, _ = client
        result = client.post("/agent/process", json={
            "type": "general", "query": REQUEST, "model": "openai", "api_key": "k",
            "document_context": {"text": DOC}, "document_action": "edit"}).json()
        edit = result["document_edit"]
        assert len(edit["insertions"]) == 2 and "placeholder::" not in result["markdown_response"]
        assert all(i["markdown"].count(f"{CDN}/u/ai-images/") == 1 for i in edit["insertions"])
        assert edit["result_sha256"] == fingerprint(result["markdown_response"])
        assert result["markdown_response"].count("![") == 3                     # the logo plus two photos


class _Used:
    """AgnoLLM stand-in: the scripted model, plus what the real one leaves behind (the runtime's attributes, the trace mark)."""
    def __init__(self, llm, request_id=None):
        self.llm, self.used, self.request_id = llm, ("openai", "gpt-6.1-terra"), request_id

    def __call__(self, *a):
        from src.observability import trace_set
        trace_set(self.request_id, document_model_started=True)     # AgnoLLM._run does this for document_editor
        return self.llm(*a)

    def drafting_model(self):
        return self.used


def test_paid_user_is_charged_for_the_text_and_each_finished_photo(monkeypatch):
    from types import SimpleNamespace
    state = {"images": [], "reserves": [], "releases": [], "charges": []}
    monkeypatch.setattr(edit_runtime, "AgnoLLM", lambda *a, **k: _Used(_editor(), k.get("request_id")))
    monkeypatch.setattr(tpa, "_build_image_tool", _fake_image_tool_builder(state["images"]))
    for flag in ["task_routing_enabled", "provider_fallback_enabled"]:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "document_context_enabled", True)
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=("provider-key", {
        "user_id": "u1", "requires_user_api_keys": False, "user_keys": {}})))
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
    monkeypatch.setattr(api, "credit_manager", SimpleNamespace(
        reserve=lambda user, amount: (state["reserves"].append(amount), (True, 9999))[1],
        schedule_release=lambda *a, **kw: state["releases"].append((a, kw))))
    monkeypatch.setattr(api, "fire_credit_deduction_and_release", lambda **kw: state["charges"].append(kw))
    monkeypatch.setattr(api, "fire_prompt_history_save", lambda *a, **kw: None)
    client = TestClient(api.app, raise_server_exceptions=False)
    events = parse_sse_events(client.post("/agent/stream", json={
        "type": "general", "query": REQUEST, "model": "openai", "api_key": "k",
        "document_context": {"text": DOC}, "document_action": "edit"}).text)
    proposal = next(e["data"] for e in events if e["event"] == "document_edit_ready")
    charge, = state["charges"]
    assert charge["success"] is True and charge["image_count"] == 2 == len(proposal["insertions"])
    assert charge["operation_type"] == "mixed" and len(charge["images"]) == 2
    assert charge["output_chars"] == sum(len(p["after"]) for p in proposal["patches"])   # text billed on its patches only


# 2026-11-20 audit: requests that only mention photos started paid photos.
@pytest.mark.parametrize("query", [
    "Add captions to the photos", "Add alt text to the images", "Add a caption under each image",
    "Add a link to the photo gallery", "Include image credits", "Add photo credits at the end",
    "Add a note that images are copyrighted", "Put the images in the right order", "Add a sentence about our photo contest",
    "Add a description to each image", "add a title above the pictures", "Move the photos to the end",
    "add the photo size limits to the FAQ",
])
def test_mentioning_photos_is_not_asking_for_new_ones(query):
    assert photo_request(query, document_blocks(DOC), cap=4) is None
    assert (decide_by_rules(query, DOC) or SimpleNamespace(reason=None)).reason != "document_photos"


def test_photo_dict_is_handed_over_before_the_edit_runs_so_an_abort_settles_its_credits(monkeypatch):
    import threading
    cancel, images = threading.Event(), {"__completion_callback__": None}
    llm = _editor()

    def tool(prompt):
        images["p1"] = {"status": "generating"}
        cancel.set()                                     # the user aborts while the photo is being made
        return str(uuid.uuid4())
    monkeypatch.setattr(edit_runtime, "AgnoLLM", lambda *a, **k: _Used(llm))
    chunks = list(edit_runtime.stream_document_edit(DOC, "add some photos", candidates=[], key_for=lambda p: "k",
                  is_byok=False, cancel_event=cancel, photo_tool=tool, pending_images=images))
    assert chunks[0] == {"__pending_images__": images} and "p1" in images
    assert not any(isinstance(c, dict) and "__event__" in c for c in chunks)    # aborted: no proposal


def test_crlf_document_keeps_crlf_around_a_photo():
    source = "# T\r\n\r\nIntro text.\r\n\r\n- one\r\n- two\r\n\r\n## Next\r\n\r\nMore.\r\n"
    for block in (b for b in document_blocks(source) if b.kind in ("paragraph", "list")):
        _, markdown = insertion_for(source, block, "Alt", "00000000-0000-4000-8000-000000000000")
        text = source[:block.end] + markdown + source[block.end:]
        assert "\n" not in text.replace("\r\n", "") and "\r\n\r\n\r\n" not in text


def test_the_message_says_when_no_photo_could_be_placed():
    _, result = edit_document(DOC, REQUEST, _editor(photos={"photos": []}), photo_tool=FakePhotoTool(), max_photos=4)
    assert result.status == "proposed" and not result.insertions and "No photos could be placed" in result.message
