"""/agent/email/generate with a caller-fixed layout: the blocks are exactly the layout; the AI writes the copy."""
import copy
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api
from src.email_template.categories import get_category, load_example
from src.email_template.layout import parse_layout
from src.email_template.prompts import system_prompt
from src.email_template.service import EmailGenerationError, GenerateRequest, generate
from tests.email.test_email_service import FakeLLM

LAUNCH = load_example("product_launch")
REMINDER = load_example("reminder_deadline")


def launch_request(layout, **kw):
    inp = LAUNCH["input"]
    base = dict(brief=inp["brief"], allowed_merge_tags=tuple(inp["allowed_merge_tags"]),
                links={"product_page": "https://a.example/p"}, assets=inp["assets"], sign_off="none",
                layout=parse_layout(layout))
    base.update(kw)
    return GenerateRequest(**base)


def launch_reply(types):
    """The launch example's blocks, picked by type in the given order (each example block used once)."""
    pool = copy.deepcopy(LAUNCH["output"]["blocks"])
    blocks = []
    for t in types:
        block = next(b for b in pool if b["type"] == t)
        pool.remove(block)
        blocks.append(block)
    return {"meta": LAUNCH["output"]["meta"], "blocks": blocks}


@pytest.mark.parametrize("given, expected", [
    (["header", "hero", "paragraph", "button", "footer"], ("header", "hero_banner", "paragraph", "button", "footer")),
    (["Header", "image", "heading", "paragraph", "image", "CTA", "footer"],
     ("header", "image:hero", "heading", "paragraph", "image:inline", "button", "footer")),
    (["header", "hero_banner", "image", "features", "footer"],
     ("header", "hero_banner", "image:hero", "features", "footer")),
    (["header", "heading", "image", "footer"], ("header", "heading", "image:inline", "footer")),
    (["header", "inline-image", "hero image", "footer"], ("header", "image:inline", "image:hero", "footer")),
])
def test_parse_layout(given, expected):
    assert parse_layout(given) == expected


@pytest.mark.parametrize("given, message", [
    (["header", "footer"], "3-20"),
    (["header", "carousel", "footer"], "unknown block type 'carousel'"),
    (["paragraph", "header", "footer"], "header must be the first"),
    (["header", "footer", "paragraph"], "footer must be the last"),
    (["header", "paragraph", "hero", "footer"], "hero_banner comes first"),
    (["header", "button", "paragraph", "button", "footer"], "at most one 'button'"),
    (["header", "hero_image", "image_hero", "footer"], "at most one 'image:hero'"),
])
def test_bad_layouts_are_request_errors(given, message):
    with pytest.raises(ValueError, match=message):
        parse_layout(given)


def test_layout_is_followed_and_sent_to_the_model():
    types = ["header", "hero_banner", "paragraph", "button", "footer"]
    reply = launch_reply(["header", "hero_banner", "paragraph", "footer"])
    hero = reply["blocks"][1]
    reply["blocks"].insert(3, {"id": "b9", "type": "button", **hero.pop("button")})
    llm = FakeLLM(reply)
    out = generate(launch_request(types), llm)
    assert [b["type"] for b in out["templates"][0]["template"]["blocks"]] == types
    assert out["layout"] == types and len(llm.calls) == 1
    system, messages = llm.calls[0]
    assert '"layout"' in system and "Recommended block order" not in system
    assert '"layout": [' in messages[0]["content"]


def test_layout_replaces_the_category_structure_rules():
    # A reminder normally needs a greeting, key_value and signature; the caller's layout has none of them.
    types = ["header", "heading", "paragraph", "button", "footer"]
    reply = {"meta": REMINDER["output"]["meta"], "blocks": [
        {"id": "b1", "type": "header", "org_name": "{{company_name}}"},
        {"id": "b2", "type": "heading", "text": "Compliance training due 30 November"},
        {"id": "b3", "type": "paragraph", "text": "Please complete the annual compliance training by 30 November."},
        {"id": "b4", "type": "button", "label": "Open the training portal", "link": "link:training_portal"},
        {"id": "b5", "type": "footer", "unsubscribe": False},
    ]}
    inp = REMINDER["input"]
    req = GenerateRequest(brief=inp["brief"], allowed_merge_tags=tuple(inp["allowed_merge_tags"]),
                          links={"training_portal": "https://p.example"}, layout=parse_layout(types))
    out = generate(req, FakeLLM(reply))
    assert out["resolved_category"] == "reminder_deadline"
    assert [b["type"] for b in out["templates"][0]["template"]["blocks"]] == types


def test_a_reply_that_ignores_the_layout_is_repaired_then_rejected():
    types = ["header", "hero_banner", "features", "footer"]
    wrong = launch_reply(["header", "hero_banner", "paragraph", "footer"])
    llm = FakeLLM(wrong, launch_reply(types))
    out = generate(launch_request(types), llm)
    assert len(llm.calls) == 2 and "[layout]" in llm.calls[1][1][-1]["content"]
    assert [b["type"] for b in out["templates"][0]["template"]["blocks"]] == types
    with pytest.raises(EmailGenerationError) as err:
        generate(launch_request(types), FakeLLM(wrong))
    assert err.value.errors[0].kind == "layout"


def test_image_slots_come_from_the_layout():
    types = ["header", "hero_banner", "image", "features", "footer"]
    reply = launch_reply(["header", "hero_banner", "image", "features", "footer"])
    reply["blocks"][2]["slot"] = "inline"      # the model got the slot wrong; the layout decides
    llm = FakeLLM(reply)
    tpl = generate(launch_request(types), llm)["templates"][0]["template"]
    assert tpl["blocks"][2]["slot"] == "hero"
    assert '"image_slots": [\n  "hero"\n ]' in llm.calls[0][1][0]["content"]


def test_layout_and_sign_off_must_agree():
    types = ["header", "paragraph", "signature", "footer"]
    with pytest.raises(ValueError, match="signature"):
        generate(launch_request(types, sign_off="none"), FakeLLM("{}"))
    with pytest.raises(ValueError, match="no signature block"):
        generate(launch_request(["header", "paragraph", "footer"], sign_off="signature"), FakeLLM("{}"))


def test_the_default_prompt_has_no_layout_section():
    category = get_category("product_launch")
    assert "layout" not in system_prompt(category, "professional")
    assert system_prompt(category, "professional") != system_prompt(category, "professional", fixed_layout=True)


class TestRoute:
    def setup_method(self):
        self.client = TestClient(api.app)

    def body(self, **kw):
        inp = LAUNCH["input"]
        return {"model": "openai", "api_key": "lic", "brief": inp["brief"], "sign_off": "none",
                "allowed_merge_tags": inp["allowed_merge_tags"], "assets": inp["assets"],
                "links": {"product_page": "https://a.example/p"}, **kw}

    def test_layout_200(self):
        types = ["header", "hero", "image", "features", "footer"]
        reply = launch_reply(["header", "hero_banner", "image", "features", "footer"])
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM(reply)):
            r = self.client.post("/agent/email/generate", json=self.body(layout=types))
        assert r.status_code == 200, r.text
        assert r.json()["layout"] == ["header", "hero_banner", "image:hero", "features", "footer"]

    @pytest.mark.parametrize("layout", [["header", "slideshow", "footer"], ["header"], "header,footer"])
    def test_bad_layout_400(self, layout):
        r = self.client.post("/agent/email/generate", json=self.body(layout=layout))
        assert r.status_code == 400 and "layout" in r.json()["error"]

    def test_sign_off_conflict_400(self):
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM("{}")):
            r = self.client.post("/agent/email/generate",
                                 json=self.body(layout=["header", "paragraph", "signature", "footer"]))
        assert r.status_code == 400 and "signature" in r.json()["error"]
