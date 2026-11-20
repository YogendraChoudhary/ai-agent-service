"""
Regression tests for the design-canvas and image defects found by the 2026-11-20 audit: auto-heal recolouring text the
user did not touch, ids taken from the user's nodes, renamed selections, 1.0 section splices, image memory, image src
checks, photo fill for 1.0 clients, blog picture settings, and questions about a selected design.
"""

import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.image_gen as ig
import src.text_processing_agent as tpa
from config import settings
from src.design_canvas.images import check_image_srcs
from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.schema_v11 import DesignDocumentASTV11, ImageNode
from src.design_canvas.section_edit import prepare_section_edit, splice_section
from src.design_canvas.service import execute_design_canvas_harness
from src.design_canvas.validate import auto_heal_design_ast, validate_design_ast

CDN = settings.cdn_url.rstrip("/")
SELECT = "\n\n(Apply this to the selected section, id {}. Keep everything else unchanged.)"


def _text(node_id, text, color):
    return {"node_type": "text", "id": node_id, "tag": "p", "spans": [{"text": text, "color": color}]}


def _button(node_id, label, bg="#C8743A"):
    return {"node_type": "button", "id": node_id, "label": label, "text_color": "#FFFFFF",
            "style": {"background_color": bg}}


def _design(version="1.0"):
    """A dark page whose caption the user set to a low-contrast grey themselves."""
    return {"version": version, "design_id": "page", "title": "Page", "theme_palette": ["#18100C", "#F5EDE6"],
            "root": {"node_type": "container", "id": "root", "style": {"background_color": "#18100C"}, "children": [
                {"node_type": "container", "id": "hero", "children": [
                    _text("hero_title", "Fresh roast", "#F5EDE6"),
                    _text("hero_sub", "Every morning", "#D8C8BA")]},
                {"node_type": "container", "id": "action_row", "layout": "flex_row", "children": [
                    _button("btn_order_now", "Order Fresh Roast"), _button("btn_menu", "Menu")]},
                _text("caption", "Roast-to-Cup Window", "#3A3A40")]}}


def _walk(node):
    yield node
    for child in node.get("children") or []:
        yield from _walk(child)


def _find(design, node_id):
    return next(n for n in _walk(design["root"]) if n["id"] == node_id)


class _Runner:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    async def run_json_completion(self, messages, image_bytes=None, temperature=0.2):
        self.calls += 1
        return json.dumps(self.answers[min(self.calls - 1, len(self.answers) - 1)])


def _edit(answer, query, existing, *more):
    runner, report = _Runner(answer, *more), {}
    out = asyncio.run(execute_design_canvas_harness(runner, user_query=query, sub_mode="edit_existing_ast",
                                                    existing_ast_json=json.dumps(existing), report=report))
    return out.model_dump(), runner.calls, report


# ------------------------------------------------------------------------------------- 1. heal only what the edit made


def test_full_edit_leaves_the_users_own_low_contrast_text_alone():
    existing = _design()
    answer = copy.deepcopy(existing)
    _find(answer, "btn_order_now")["style"]["background_color"] = "#1D4ED8"
    out, calls, report = _edit(answer, "make the button blue", existing)
    assert _find(out, "caption")["spans"][0]["color"] == "#3A3A40"
    assert _find(out, "btn_order_now")["style"]["background_color"] == "#1D4ED8"
    assert calls == 1 and report["auto_healed"] is False


def test_full_edit_still_heals_contrast_the_edit_made():
    existing = _design()
    answer = copy.deepcopy(existing)
    _find(answer, "hero_title")["spans"][0]["color"] = "#222222"           # the edit's own unreadable colour
    answer["root"]["children"].append(_text("new_note", "New", "#202020"))   # a new node in the user's dark grey
    out, calls, _ = _edit(answer, "make the title darker and add a note", existing)
    assert _find(out, "hero_title")["spans"][0]["color"] != "#222222"
    assert _find(out, "new_note")["spans"][0]["color"] != "#202020"
    assert _find(out, "caption")["spans"][0]["color"] == "#3A3A40"
    assert calls == 1


def test_contrast_on_a_new_background_is_the_edits_doing():
    existing = _design()
    existing["root"]["children"][2] = _text("caption", "Roast", "#EEEEEE")    # fine on the dark page
    answer = copy.deepcopy(existing)
    answer["root"]["style"]["background_color"] = "#FFFFFF"                  # the edit makes the page white
    errors = validate_design_ast(DesignDocumentAST.model_validate(answer), existing_ast_json=json.dumps(existing))
    assert any("`caption`" in e for e in errors if e.startswith("WCAG_LOW_CONTRAST"))
    assert validate_design_ast(DesignDocumentAST.model_validate(existing), existing_ast_json=json.dumps(existing)) == []


def test_section_splice_heals_the_section_only():
    user = _design()
    scope = prepare_section_edit(json.dumps(user), "make the heading grey" + SELECT.format("hero"))
    section = json.loads(scope.section_json)
    _find(section, "hero_title")["spans"][0]["color"] = "#444444"
    out = splice_section(scope, DesignDocumentAST.model_validate(section)).model_dump()
    assert _find(out, "hero_title")["spans"][0]["color"] != "#444444"
    assert _find(out, "caption")["spans"][0]["color"] == "#3A3A40"


def test_without_an_original_every_low_contrast_text_is_healed_as_before():
    healed = auto_heal_design_ast(DesignDocumentAST.model_validate(_design())).model_dump()
    assert _find(healed, "caption")["spans"][0]["color"] != "#3A3A40"


# ------------------------------------------------------------------------- 2. a new node never takes an existing id


def test_new_node_reusing_an_existing_id_is_the_one_renamed():
    existing = _design()
    answer = copy.deepcopy(existing)
    row = _find(answer, "action_row")
    row["children"].insert(0, _button("btn_order_now", "Book a tasting"))
    out, calls, _ = _edit(answer, 'add a "Book a tasting" button before the first button', existing)
    labels = {n["id"]: n.get("label") for n in _walk(out["root"]) if n["node_type"] == "button"}
    assert labels["btn_order_now"] == "Order Fresh Roast"
    assert labels["btn_order_now_2"] == "Book a tasting"
    assert calls == 1


def test_edited_original_keeps_its_id_over_an_identical_copy_by_position():
    existing = _design()
    answer = copy.deepcopy(existing)
    row = _find(answer, "action_row")
    row["children"][0]["label"] = "Order now"                                  # the user's button, edited
    row["children"].append(copy.deepcopy(row["children"][0]))                  # and a copy after the last button
    healed = auto_heal_design_ast(DesignDocumentAST.model_validate(answer), existing_ast_json=json.dumps(existing))
    ids = [c.id for c in healed.root.children[1].children]
    assert ids == ["btn_order_now", "btn_menu", "btn_order_now_2"]


def test_duplicates_in_a_new_design_keep_the_first_occurrence():
    design = _design()
    design["root"]["children"].append(_text("caption", "Second", "#F5EDE6"))
    healed = auto_heal_design_ast(DesignDocumentAST.model_validate(design))
    assert [c.id for c in healed.root.children][-2:] == ["caption", "caption_2"]


# --------------------------------------------------------------------------------- 3. the selected node keeps its id


def test_section_edit_restores_the_selected_id_when_the_model_renames_it():
    user = _design()
    scope = prepare_section_edit(json.dumps(user), "more padding here" + SELECT.format("hero"))
    section = json.loads(scope.section_json)
    node = section["root"]["children"][0]
    node["id"] = "hero_new"
    node["style"] = {"padding": {"top": 40, "right": 40, "bottom": 40, "left": 40}}
    out = splice_section(scope, DesignDocumentAST.model_validate(section)).model_dump()
    ids = {n["id"] for n in _walk(out["root"])}
    assert "hero" in ids and "hero_new" not in ids
    assert _find(out, "hero")["style"]["padding"]["top"] == 40


def test_full_edit_with_a_selection_restores_the_selected_id():
    existing = _design()
    answer = copy.deepcopy(existing)
    hero = _find(answer, "hero")
    hero["id"] = "hero_section"
    hero["style"] = {"background_color": "#2A1A12"}
    out, calls, _ = _edit(answer, "give this a warmer background" + SELECT.format("hero"), existing)
    assert _find(out, "hero")["style"]["background_color"] == "#2A1A12"
    assert "hero_section" not in {n["id"] for n in _walk(out["root"])}
    assert calls == 1


def test_full_edit_without_a_selection_does_not_rename_new_nodes():
    existing = _design()
    answer = copy.deepcopy(existing)
    _find(answer, "hero")["id"] = "hero_section"
    _find(answer, "hero_section")["style"] = {"background_color": "#2A1A12"}
    out, _, _ = _edit(answer, "give the hero a warmer background", existing)
    assert "hero_section" in {n["id"] for n in _walk(out["root"])}


# ------------------------------------------------------------------- 4. a renamed id is healed in code, no repair call


def test_renamed_id_costs_no_repair_call():
    existing = _design()
    answer = copy.deepcopy(existing)
    _find(answer, "hero_title")["id"] = "hero_title_x"
    _find(answer, "btn_order_now")["style"]["background_color"] = "#1D4ED8"
    out, calls, report = _edit(answer, "make the button blue", existing, existing)
    assert calls == 1
    assert report["attempts"][0]["outcome"] == "pass_auto_healed"
    assert "hero_title" in {n["id"] for n in _walk(out["root"])}


# ------------------------------------------------------------------- 5. section edits of 1.0 designs answer in 1.1


def test_section_edit_on_a_v10_design_keeps_v11_features(monkeypatch):
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    user = _design("1.0")
    scope = prepare_section_edit(json.dumps(user), "make the headline a gold gradient" + SELECT.format("hero"))
    section = json.loads(scope.section_json)
    section["version"] = "1.1"
    _find(section, "hero_title")["spans"][0]["gradient"] = {"angle_deg": 90, "stops": ["#F5C451", "#FFE29A"]}
    edited = DesignDocumentASTV11.model_validate(section)
    out = splice_section(scope, edited)
    assert isinstance(out, DesignDocumentASTV11) and out.version == "1.1"
    assert _find(out.model_dump(), "hero_title")["spans"][0]["gradient"]["stops"] == ["#F5C451", "#FFE29A"]


def test_section_edit_on_a_v10_design_answered_in_v10_stays_v10():
    user = _design("1.0")
    scope = prepare_section_edit(json.dumps(user), "make the heading grey" + SELECT.format("hero"))
    out = splice_section(scope, DesignDocumentAST.model_validate(json.loads(scope.section_json)))
    assert type(out) is DesignDocumentAST and out.version == "1.0"


# --------------------------------------------------------------------------- 6. design and blog pictures: no memory


@pytest.fixture
def fake_image_backend(monkeypatch):
    saved = []

    class FakeRedis:
        def set(self, key, value, ex=None):
            saved.append(key)

        def get(self, key):
            return None

    monkeypatch.setattr(ig, "_get_redis_client", lambda: FakeRedis())
    for name in ("_call_gemini_with_fallback", "_call_openai_with_fallback", "_call_grok_with_fallback"):
        monkeypatch.setattr(ig, name, lambda *a, **k: (b"\x89PNG fake", "image/png", "stub"))
    monkeypatch.setattr(ig, "upload_image_from_bytes_detailed", lambda **k: {
        "cdn_url": f"{CDN}/u/ai-images/x.png", "width": 1, "height": 1, "format": "png", "content_type": "image/png"})
    return saved


def _wait(pending):
    for _ in range(100):
        if all(v.get("status") != "generating" for k, v in pending.items() if not str(k).startswith("__")):
            return
        time.sleep(0.02)


def _photo_design():
    return {"version": "1.1", "design_id": "l", "title": "L", "theme_palette": ["#FFFFFF", "#1B1B1F"],
            "root": {"node_type": "container", "id": "root", "style": {"background_color": "#FFFFFF"}, "children": [
                {"node_type": "image", "id": "img1", "alt": "Burger", "prompt": "a burger on a board"},
                _text("t1", "Hello", "#1B1B1F")]}}


def _fill(monkeypatch, schema_version=None):
    monkeypatch.setattr(settings, "design_canvas_enabled", True)
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    monkeypatch.setattr(settings, "design_image_fill_enabled", True)
    monkeypatch.setattr(settings, "task_routing_enabled", False)
    monkeypatch.setattr(settings, "provider_fallback_enabled", False)
    return list(tpa.process_user_query_stream(
        "add real photos", model_provider="gemini", api_key="k", user_id="user_7", document_id="doc_42",
        is_byok=True, existing_design_ast=_photo_design(), schema_version=schema_version))


def test_design_photo_fill_does_not_become_the_previous_image(monkeypatch, fake_image_backend):
    out = _fill(monkeypatch)
    pending = next(c["__pending_images__"] for c in out if isinstance(c, dict) and "__pending_images__" in c)
    _wait(pending)
    assert [v["status"] for k, v in pending.items() if not str(k).startswith("__")] == ["done"]
    assert fake_image_backend == []


def test_without_the_wrapper_the_image_flow_still_remembers(fake_image_backend):
    pending = {}
    tool = tpa._build_image_tool("gemini", "k", pending, "r", "user_7", None, True, {"document_id": "doc_42"}, None)
    tool("a burger")
    _wait(pending)
    assert "image_context:user_7:doc:doc_42" in fake_image_backend
    fake_image_backend.clear()
    ig.without_image_memory(tool)("a burger")
    _wait(pending)
    assert fake_image_backend == []
    assert ig._REMEMBER_IMAGES.get() is True                     # the caller's own context is left as it was


def _run_blog(monkeypatch, query, image_params=None):
    from src.blog import runtime
    from src.task_router import Candidate

    seen = {}

    def build(provider, key, pending, request_id, user_id, credit_manager, is_byok, ctx, budget):
        seen["params"] = ctx["explicit_params"]
        return lambda prompt: seen.setdefault("remember", ig._REMEMBER_IMAGES.get()) and "ERROR: stub"

    def fake_pipeline(request, plan, llm, media, **kwargs):
        media.fulfil(SimpleNamespace(prompt="a laptop on a desk"))
        yield "# Title"

    monkeypatch.setattr(tpa, "_build_image_tool", build)
    monkeypatch.setattr(tpa, "_image_tool_available", lambda *a: True)
    monkeypatch.setattr(runtime, "generate_blog_stream", fake_pipeline)
    monkeypatch.setattr(runtime, "plan_blog", lambda request, image_available: SimpleNamespace(
        target_words=800, sections=3, tables=0, quotes=0, images=1, use_web=False, reasons={}))
    monkeypatch.setattr(settings, "blog_image_source", "generate")
    list(runtime.stream_blog(query, None, None, candidates=[Candidate("gemini", True)], key_for=lambda p: "k",
                             is_byok=True, user_id="user_7", credit_manager=None, pending_images={},
                             image_params=image_params, document_id="doc_42", request_id="r"))
    return seen


def test_blog_pictures_do_not_become_the_previous_image(monkeypatch):
    assert _run_blog(monkeypatch, "Write a blog post about remote work with 2 images")["remember"] is False


# --------------------------------------------------------------------------------------------- 7. image src checks


@pytest.mark.parametrize("bad", [
    "https://attacker.example\\@{host}/a.png", "https://user:pass@{host}/a.png", "https://{host}:8443/a.png",
    "https://{host}@attacker.example/a.png"])
def test_image_src_rejects_backslash_userinfo_and_ports(bad):
    host = CDN.split("://")[1]
    with pytest.raises(ValidationError):
        ImageNode(id="img1", alt="B", prompt="a burger", src=bad.format(host=host))
    assert ImageNode(id="img1", alt="B", prompt="a burger", src=f"{CDN}/u/ai-images/a.png").src


def test_model_written_image_src_is_cleared_unless_the_design_had_it():
    existing = _photo_design()
    existing["root"]["children"][0]["src"] = f"{CDN}/u/ai-images/mine.png"
    edited = copy.deepcopy(existing)
    edited["root"]["children"].append({"node_type": "image", "id": "img2", "alt": "Other", "prompt": "fries",
                                       "src": f"{CDN}/someone-else/ai-images/theirs.png"})
    ast = DesignDocumentASTV11.model_validate(edited)
    warnings = check_image_srcs(ast, json.dumps(existing))
    assert [w["kind"] for w in warnings] == ["image_src_removed"] and warnings[0]["node_id"] == "img2"
    assert ast.root.children[0].src == f"{CDN}/u/ai-images/mine.png"
    assert ast.root.children[2].src is None
    fresh = DesignDocumentASTV11.model_validate(existing)
    assert len(check_image_srcs(fresh, None)) == 1 and fresh.root.children[0].src is None   # a new design has none


def test_design_edit_reports_a_removed_image_src(monkeypatch):
    existing = _photo_design()
    answer = copy.deepcopy(existing)
    answer["root"]["children"][0]["src"] = f"{CDN}/someone-else/ai-images/theirs.png"

    async def harness(*args, **kwargs):
        return DesignDocumentASTV11.model_validate(answer)

    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness", harness)
    monkeypatch.setattr(settings, "design_canvas_enabled", True)
    monkeypatch.setattr(settings, "design_schema_v11_enabled", True)
    out = list(tpa.process_user_query_stream("move the photo to the right", model_provider="claude", api_key="k",
                                             existing_design_ast=existing, operation_type="edit_design"))
    events = {c["__event__"]["event"]: c["__event__"]["data"] for c in out if isinstance(c, dict) and "__event__" in c}
    assert events["design_ready"]["root"]["children"][0]["src"] is None
    assert [w["kind"] for w in events["complete"]["warnings"]] == ["image_src_removed"]


# ----------------------------------------------------------------------------------- 8. no photo fill for 1.0 clients


def test_photo_fill_request_from_a_v10_client_runs_the_layout_edit(monkeypatch):
    calls = []

    async def harness(*args, **kwargs):
        calls.append(kwargs)
        return DesignDocumentAST.model_validate({**_design(), "version": "1.0"})

    def no_image_tool(*args, **kwargs):
        raise AssertionError("photo fill must not run for a 1.0 client")

    monkeypatch.setattr("src.design_canvas.service.execute_design_canvas_harness", harness)
    monkeypatch.setattr(tpa, "_build_image_tool", no_image_tool)
    out = _fill(monkeypatch, schema_version="1.0")
    assert [c["sub_mode"] for c in calls] == ["edit_existing_ast"] and calls[0]["schema_version"] == "1.0"
    design = next(c["__event__"]["data"] for c in out if isinstance(c, dict) and c.get("__event__", {}).get("event") == "design_ready")
    assert design["version"] == "1.0"


# --------------------------------------------------------------------------- 9. blog pictures follow the user's words


@pytest.mark.parametrize("query, image_params, expected", [
    ("Write a blog post about remote work with square, high quality images", None, ("1:1", "high")),
    ("Write a blog post about coffee with photos in portrait", None, ("9:16", "medium")),
    ("Write a blog post about vertical farming with 3 images", None, ("16:9", "medium")),
    ("Write a blog post about Times Square nightlife", None, ("16:9", "medium")),
    ("Write a blog post with square images", {"aspect_ratio": "4:3"}, ("4:3", "medium")),   # API fields still win
])
def test_blog_picture_settings_precedence(query, image_params, expected):
    from src.blog.runtime import picture_params
    params = picture_params(query, None, image_params)
    assert (params["aspect_ratio"], params["quality"]) == expected


def test_blog_image_tool_gets_the_users_picture_words(monkeypatch):
    seen = _run_blog(monkeypatch, "Write a blog post about remote work with square, high quality images")
    assert seen["params"]["aspect_ratio"] == "1:1" and seen["params"]["quality"] == "high"


# -------------------------------------------------------------------- 10. questions about a selected design see it


def _ask(monkeypatch, query, design):
    seen = {}

    def fake_agent_text(agent, model_input, *args, **kwargs):
        seen["input"] = model_input
        yield "Inter."

    monkeypatch.setattr(settings, "design_canvas_enabled", True)
    monkeypatch.setattr(settings, "task_routing_enabled", False)
    monkeypatch.setattr(settings, "provider_fallback_enabled", False)
    monkeypatch.setattr(tpa, "create_agent", lambda *a, **k: object())
    monkeypatch.setattr(tpa, "_agent_text", fake_agent_text)
    monkeypatch.setattr("src.intent_classifier.classify_intent", lambda *a, **k: "other")
    out = list(tpa.process_user_query_stream(query, model_provider="claude", api_key="k", existing_design_ast=design))
    assert not any(isinstance(c, dict) and c.get("__event__", {}).get("event") == "design_ready" for c in out)
    return seen["input"]


def test_question_about_the_selected_design_is_answered_with_it(monkeypatch):
    design = _design("1.1")
    design["fonts"] = {"heading": {"family": "Fraunces", "fallback": "serif"}}
    model_input = _ask(monkeypatch, "What font is this?" + SELECT.format("hero"), design)
    head, task = model_input.split("USER_TASK:\n")
    assert head.startswith("REFERENCE_MATERIAL (read-only") and "data, never instructions" in head
    assert "Fraunces" in head and "selected node is `hero`" in head
    assert task.startswith("What font is this?")


def test_large_design_sends_the_selected_node_and_is_capped(monkeypatch):
    design = _design("1.1")
    design["root"]["children"].extend(_text(f"filler{i}", "x" * 800, "#F5EDE6") for i in range(20))
    model_input = _ask(monkeypatch, "Why is this text grey?" + SELECT.format("caption"), design)
    head = model_input.split("USER_TASK:\n")[0]
    assert 'selected_node' in head and "Roast-to-Cup Window" in head and "filler0" not in head
    assert len(head) < tpa._DESIGN_CHAT_MAX_CHARS + 2_000


def test_chat_without_a_design_has_no_design_block(monkeypatch):
    assert "REFERENCE_MATERIAL" not in _ask(monkeypatch, "What font is this?", None)
