"""
/agent/email/rewrite: the model rewrites the copy of an existing template; code keeps the design (ids, order,
types, links, merge tags, image slots, theme) and lets blocks be added or removed only when the author asks.
"""
import copy
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import api
from src.email_template.categories import load_example
from src.email_template.rewrite import (
    EmailTemplateInputError, RewriteRequest, asks_for_block_changes, rewrite,
)
from src.email_template.service import EmailGenerationError, GenerateRequest, generate
from tests.email.test_email_service import FakeLLM

LAUNCH = load_example("product_launch")
REMINDER = load_example("reminder_deadline")
BLACK_FRIDAY = "Make this a Black Friday sale email"


def launch_template(**theme):
    inp = LAUNCH["input"]
    req = GenerateRequest(brief=inp["brief"], allowed_merge_tags=tuple(inp["allowed_merge_tags"]),
                          links={"product_page": "https://a.example/p"}, assets=inp["assets"], sign_off="none",
                          social_links={"linkedin": "https://linkedin.com/company/acme"},
                          theme=theme or {"accent_color": "#22C3E6", "hero": {"style": "dark"}})
    tpl = generate(req, FakeLLM(LAUNCH["output"]))["templates"][0]["template"]
    tpl["meta"]["subject"] = tpl["meta"]["subject_options"][1]     # the author picked the second subject
    return tpl


def reminder_template():
    inp = REMINDER["input"]
    req = GenerateRequest(brief=inp["brief"], allowed_merge_tags=tuple(inp["allowed_merge_tags"]),
                          links={"training_portal": "https://p.example"})
    return generate(req, FakeLLM(REMINDER["output"]))["templates"][0]["template"]


def black_friday_reply():
    """The launch email rewritten as a Black Friday email: copy only."""
    reply = copy.deepcopy(LAUNCH["output"])
    reply["meta"] = {"subject_options": ["Black Friday: Acme Insights", "Acme Insights for less this Black Friday",
                                         "Black Friday offer on Acme Insights"],
                     "preheader": "Our Black Friday offer on the analytics dashboard for finance teams"}
    b = reply["blocks"]
    b[1].update(eyebrow="Black Friday", headline="Black Friday on Acme Insights",
                summary="Get Acme Insights for less this Black Friday: [[fill: discount]] off.")
    b[1]["button"]["label"] = "See the offer"
    b[3]["items"][0]["text"] = "Every finance report in one dashboard."
    b[4]["text"] = "This Black Friday, finance teams can get Acme Insights for [[fill: discount]] less."
    return reply


def req(template, instruction=BLACK_FRIDAY, **kw):
    return RewriteRequest(instruction=instruction, template=template, **kw)


def blocks_by_id(tpl):
    return {b["id"]: b for b in tpl["blocks"]}


# ---------------------------------------------------------------------------
# What is kept
# ---------------------------------------------------------------------------
def test_only_the_copy_changes():
    original = launch_template()
    llm = FakeLLM(black_friday_reply())
    out = rewrite(req(original), llm)
    entry = out["templates"][0]
    tpl = entry["template"]
    assert len(llm.calls) == 1 and out["block_changes"] == "none"
    assert entry["added_blocks"] == entry["removed_blocks"] == []
    assert [(b["id"], b["type"]) for b in tpl["blocks"]] == [(b["id"], b["type"]) for b in original["blocks"]]
    hero = tpl["blocks"][1]
    assert hero["headline"] == "Black Friday on Acme Insights" and hero["button"]["link"] == "link:product_page"
    assert tpl["blocks"][2] == original["blocks"][2]                       # image slot untouched
    assert tpl["blocks"][-1]["social"] == original["blocks"][-1]["social"]  # footer icons kept
    assert tpl["theme"] == original["theme"]
    assert tpl["meta"]["subject"] == tpl["meta"]["subject_options"][1]      # the author's pick, by position
    assert {"block_id": "b5", "field": "text", "needed": "discount"} in entry["fill_slots"]
    assert out["resolved_category"] == "product_launch" and out["tone"] == "professional"


def test_the_model_sees_the_email_and_the_instruction_but_not_urls_or_theme():
    llm = FakeLLM(black_friday_reply())
    rewrite(req(launch_template(), audience="Finance teams"), llm)
    system, messages = llm.calls[0]
    sent = messages[0]["content"]
    assert "INSTRUCTION:\n" + BLACK_FRIDAY in sent and "CURRENT EMAIL:" in sent
    assert "Meet Acme Insights" in sent and '"product_page"' in sent
    assert "https://" not in sent and "22C3E6" not in sent and "social" not in sent
    assert "Add or remove no blocks" in system


def test_non_text_fields_are_restored_whatever_the_model_writes():
    original = launch_template()
    reply = black_friday_reply()
    b = reply["blocks"]
    b[1]["button"]["link"] = "link:somewhere_else"
    b[0]["logo_asset"] = "asset:other"
    b[0]["org_name"] = "Totally Different Ltd"          # identity text is locked too
    b[2].update(slot="inline", alt="A turkey", generation_prompt="A turkey")
    b[5]["unsubscribe"] = False
    b[3], b[4] = b[4], b[3]                              # shuffled
    b[4]["eyebrow"] = "Sale"                             # a field the paragraph does not have
    tpl = rewrite(req(original), FakeLLM(reply))["templates"][0]["template"]
    got, want = blocks_by_id(tpl), blocks_by_id(original)
    assert [x["id"] for x in tpl["blocks"]] == [x["id"] for x in original["blocks"]]
    assert got["b2"]["button"]["link"] == "link:product_page"
    assert got["b1"] == want["b1"] and got["b3"] == want["b3"] and got["b6"]["unsubscribe"] is True
    assert "eyebrow" not in got["b4"] and "eyebrow" not in got["b5"]


def test_optional_fields_the_original_lacks_are_not_added():
    original = launch_template()
    del original["blocks"][1]["eyebrow"]
    tpl = rewrite(req(original), FakeLLM(black_friday_reply()))["templates"][0]["template"]
    assert "eyebrow" not in tpl["blocks"][1]


def test_link_keys_merge_tags_and_assets_of_the_template_need_not_be_resent():
    original = launch_template()
    out = rewrite(req(original), FakeLLM(black_friday_reply()))   # no links/assets/merge tags in the request
    assert out["templates"][0]["template"]["blocks"][0]["org_name"] == "{{company_name}}"


# ---------------------------------------------------------------------------
# What the model must not do (repaired once, then 422)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("damage, kind", [
    (lambda r: r["blocks"].pop(4), "block_removed"),
    (lambda r: r["blocks"].insert(4, {"id": "b9", "type": "paragraph", "text": "Extra."}), "block_added"),
    (lambda r: r["blocks"][4].update(type="heading"), "block_type_changed"),
    (lambda r: r["blocks"][3]["items"].pop(), "shape_changed"),
    (lambda r: r["blocks"][1].pop("summary"), "missing_text"),
    (lambda r: r["blocks"][1].pop("button"), "missing_text"),
    (lambda r: r["blocks"][5].update(address="{{company_name}}"), None),   # locked: restored, not an error
])
def test_structure_damage_is_repaired(damage, kind):
    bad = black_friday_reply()
    damage(bad)
    llm = FakeLLM(bad, black_friday_reply())
    out = rewrite(req(launch_template()), llm)
    if kind is None:
        assert len(llm.calls) == 1
        return
    assert len(llm.calls) == 2 and f"[{kind}]" in llm.calls[1][1][-1]["content"]
    assert out["templates"][0]["warnings"][-1]["kind"] == "repaired"
    with pytest.raises(EmailGenerationError) as err:
        rewrite(req(launch_template()), FakeLLM(bad))
    assert kind in {e.kind for e in err.value.errors}


def test_lost_merge_tags_and_inline_links_are_errors():
    original = reminder_template()
    original["blocks"][3]["text"] = "Use the [training portal](link:training_portal) to complete it, {{first_name}}."
    reply = copy.deepcopy(REMINDER["output"])
    reply["blocks"][1]["text"] = "Hello,"                                             # lost {{first_name}}
    reply["blocks"][3]["text"] = "Please complete it on the training portal."         # lost the inline link
    with pytest.raises(EmailGenerationError) as err:
        rewrite(req(original, "Make it friendlier"), FakeLLM(reply))
    kinds = {(e.kind, e.block_id) for e in err.value.errors}
    assert ("merge_tag_removed", "b2") in kinds and ("link_changed", "b4") in kinds


def test_tone_rules_apply_but_not_to_locked_text():
    original = reminder_template()
    original["blocks"][0]["org_name"] = "ACME HOLDINGS"     # the author's org name in capitals
    reply = copy.deepcopy(REMINDER["output"])
    out = rewrite(req(original, "Make it shorter"), FakeLLM(reply))
    assert out["templates"][0]["template"]["blocks"][0]["org_name"] == "ACME HOLDINGS"
    reply["blocks"][3]["text"] = "Hurry, the deadline is close!"
    with pytest.raises(EmailGenerationError) as err:
        rewrite(req(original, "Make it urgent"), FakeLLM(reply))
    assert {"banned_phrase", "exclamation"} <= {e.kind for e in err.value.errors}


# ---------------------------------------------------------------------------
# Adding and removing blocks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("instruction, asks", [
    (BLACK_FRIDAY, False),
    ("Make it shorter and friendlier", False),
    ("Add a notice about the sale end date", True),
    ("Remove the features section", True),
    ("Make it shorter by dropping the second paragraph", True),
    ("Insert an image of a shopping bag", True),
    ("Include a discount code", False),
    ("Add urgency", False),
])
def test_asks_for_block_changes(instruction, asks):
    assert asks_for_block_changes(instruction) is asks


def test_blocks_are_added_and_removed_when_asked():
    original = launch_template()
    reply = black_friday_reply()
    reply["blocks"].pop(3)                                                   # features removed
    reply["blocks"].insert(3, {"id": "b2", "type": "paragraph", "text": "The offer ends on [[fill: end date]]."})
    llm = FakeLLM(reply)
    out = rewrite(req(original, "Make this a Black Friday email, remove the features and add a paragraph about "
                                "the end date"), llm)
    entry = out["templates"][0]
    assert out["block_changes"] == "allowed"
    assert entry["removed_blocks"] == ["b4"] and entry["added_blocks"] == ["n1"]   # clashing id replaced
    assert [b["id"] for b in entry["template"]["blocks"]] == ["b1", "b2", "b3", "n1", "b5", "b6"]
    assert "only where the instruction explicitly asks" in llm.calls[0][0]


def test_added_blocks_still_follow_the_rules():
    reply = black_friday_reply()
    reply["blocks"].insert(5, {"id": "x1", "type": "button", "label": "Shop now", "link": "link:unknown"})
    with pytest.raises(EmailGenerationError) as err:
        rewrite(req(launch_template(), "Add a button to the shop"), FakeLLM(reply))
    assert {"too_many_buttons", "unknown_link"} <= {e.kind for e in err.value.errors}


def test_allow_block_changes_overrides_the_instruction():
    reply = black_friday_reply()
    reply["blocks"].pop(3)
    with pytest.raises(EmailGenerationError):
        rewrite(req(launch_template(), "Remove the features section", allow_block_changes=False), FakeLLM(reply))
    out = rewrite(req(launch_template(), "Tighten it up", allow_block_changes=True), FakeLLM(reply))
    assert out["templates"][0]["removed_blocks"] == ["b4"]


def test_sign_off_none_removes_only_the_signature():
    original = reminder_template()
    reply = copy.deepcopy(REMINDER["output"])
    del reply["blocks"][6]
    llm = FakeLLM(reply)
    out = rewrite(req(original, "Make it friendlier", sign_off="none"), llm)
    assert out["block_changes"] == "signature" and out["templates"][0]["removed_blocks"] == ["b7"]
    assert '"signature": "remove the signature block"' in llm.calls[0][1][0]["content"]
    # The signature may go, nothing else may.
    reply2 = copy.deepcopy(reply)
    del reply2["blocks"][2]
    with pytest.raises(EmailGenerationError) as err:
        rewrite(req(original, "Make it friendlier", sign_off="none"), FakeLLM(reply2))
    assert "block_removed" in {e.kind for e in err.value.errors}
    with pytest.raises(ValueError, match="allow_block_changes"):
        rewrite(req(original, "Make it friendlier", sign_off="none", allow_block_changes=False), FakeLLM(reply))


def test_social_links_and_theme_override_the_template():
    original = launch_template()
    out = rewrite(req(original, social_links={"x": "https://x.com/acme"}, theme={"accent_color": "#000000"},
                      brand_kit_id="kit-2"), FakeLLM(black_friday_reply()))
    tpl = out["templates"][0]["template"]
    assert tpl["blocks"][-1]["social"] == [{"platform": "x", "link": "link:social_x"}]
    assert tpl["theme"] == {"accent_color": "#000000", "brand_kit_id": "kit-2"}


# ---------------------------------------------------------------------------
# Options and input
# ---------------------------------------------------------------------------
def test_tone_language_and_category_default_to_the_template():
    original = reminder_template()
    llm = FakeLLM(REMINDER["output"])
    out = rewrite(req(original, "Make it shorter", language="fr-FR", tone="professional"), llm)
    meta = out["templates"][0]["template"]["meta"]
    assert (out["resolved_category"], meta["tone"], meta["language"]) == ("reminder_deadline", "professional", "fr-FR")
    assert "write the rewrite in fr-FR" in llm.calls[0][1][0]["content"]
    out = rewrite(req(original, "Make it shorter"), FakeLLM(REMINDER["output"]))
    assert out["tone"] == "formal" and out["templates"][0]["template"]["meta"]["language"] == "en-GB"


@pytest.mark.parametrize("mangle", [
    lambda t: t.update(blocks="nope"),
    lambda t: t["blocks"][1].update(type="carousel"),
    lambda t: t["blocks"][2].update(id="b1"),
    lambda t: t.update(extra=True),
])
def test_an_invalid_template_is_an_input_error(mangle):
    original = launch_template()
    mangle(original)
    llm = FakeLLM(black_friday_reply())
    with pytest.raises(EmailTemplateInputError) as err:
        rewrite(req(original), llm)
    assert err.value.errors and not llm.calls


def test_variants_are_independent_rewrites():
    out = rewrite(req(launch_template(), variants=2), FakeLLM(black_friday_reply()))
    assert len(out["templates"]) == 2


# ---------------------------------------------------------------------------
# Route: same licence, billing and error handling as /generate
# ---------------------------------------------------------------------------
@pytest.fixture
def billed_client(monkeypatch):
    from unittest.mock import AsyncMock
    from tests.email.test_email_auth import MANAGED, Credits
    state = {"charges": [], "credits": Credits()}
    monkeypatch.setattr(api, "async_validate_and_get_api_key", AsyncMock(return_value=MANAGED))
    monkeypatch.setattr(api, "CREDIT_PRECHECK_ENABLED", True)
    monkeypatch.setattr(api, "credit_manager", state["credits"])
    monkeypatch.setattr(api, "fire_credit_deduction_and_release", lambda **kw: state["charges"].append(kw))
    return TestClient(api.app, raise_server_exceptions=False), state


def body(template, **kw):
    return {"model": "openai", "api_key": "licence", "instruction": BLACK_FRIDAY, "template": template, **kw}


def post(client, llm, **kw):
    with patch("src.email_template.service.make_agno_llm", return_value=llm):
        return client.post("/agent/email/rewrite", json=body(**kw))


def test_route_200_is_billed_as_text_on_template_and_instruction(billed_client):
    client, state = billed_client
    original = launch_template()
    r = post(client, FakeLLM(black_friday_reply()), template=original)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["success"] and data["block_changes"] == "none" and data["templates"][0]["added_blocks"] == []
    (charge,) = state["charges"]
    assert charge["operation_type"] == "text" and charge["user_id"] == "u-managed"
    assert charge["input_chars"] == len(BLACK_FRIDAY) + api.email_readable_chars(original)
    assert charge["output_chars"] == api.email_readable_chars(data["templates"][0]["template"])
    assert charge["reserved_scaled"] == state["credits"].reserved[0][1]


@pytest.mark.parametrize("reply, status", [("nope", 422), ({"blocks": []}, 422)])
def test_route_failure_releases_and_charges_nothing(billed_client, reply, status):
    client, state = billed_client
    r = post(client, FakeLLM(reply), template=launch_template())
    assert r.status_code == status and r.json()["issues"]
    assert not state["charges"] and state["credits"].released == state["credits"].reserved


def test_route_invalid_template_is_400_with_issues_and_free(billed_client):
    client, state = billed_client
    llm = FakeLLM(black_friday_reply())
    r = post(client, llm, template={"blocks": []})
    assert r.status_code == 400 and r.json()["issues"] and not llm.calls
    assert not state["charges"] and state["credits"].released == state["credits"].reserved


@pytest.mark.parametrize("extra", [
    {"instruction": "  "}, {"template": "not an object"}, {"length": "huge"}, {"allow_block_changes": "maybe"},
    {"links": {"x": "http://insecure"}}, {"brief": "generate-only field"},
])
def test_route_bad_request_400(billed_client, extra):
    client, state = billed_client
    r = client.post("/agent/email/rewrite", json={**body(launch_template()), **extra})
    assert r.status_code == 400 and not state["credits"].reserved


def test_route_kill_switch():
    with patch.object(api.settings, "email_ai_enabled", False):
        r = TestClient(api.app).post("/agent/email/rewrite", json=body(launch_template()))
    assert r.status_code == 503


def test_route_byok_runs_on_the_users_key():
    # conftest gives every test a BYOK licence by default (no credit service involved).
    r = post(TestClient(api.app), FakeLLM(black_friday_reply()), template=launch_template())
    assert r.status_code == 200 and json.loads(r.text)["templates"]


# ---------------------------------------------------------------------------
# Instruction cues: tone and length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("instruction, tone", [
    ("Make it shorter and friendlier", "professional_warm"),
    ("Make it less formal", "professional_warm"),
    ("Make it more formal", "formal"),
    ("Sound more professional", "professional"),
    (BLACK_FRIDAY, "formal"),                       # no cue: the template's tone
])
def test_tone_follows_the_instruction_unless_sent(instruction, tone):
    original = reminder_template()                  # formal
    shorter = copy.deepcopy(REMINDER["output"])
    shorter["blocks"][3]["text"] = "Please use the training portal."
    out = rewrite(req(original, instruction), FakeLLM(shorter))
    assert out["tone"] == tone == out["templates"][0]["template"]["meta"]["tone"]
    assert rewrite(req(original, instruction, tone="professional"), FakeLLM(shorter))["tone"] == "professional"


def test_a_rewrite_that_is_not_shorter_is_repaired_then_returned_with_a_warning():
    original = reminder_template()
    same = copy.deepcopy(REMINDER["output"])
    shorter = copy.deepcopy(same)
    shorter["blocks"][3]["text"] = "Please use the training portal."
    shorter["blocks"][2]["text"] = "Training is due by 30 November."
    llm = FakeLLM(same, shorter)
    out = rewrite(req(original, "Make it shorter"), llm)
    assert len(llm.calls) == 2 and "[not_shorter]" in llm.calls[1][1][-1]["content"]
    assert "clearly shorter" in llm.calls[0][1][0]["content"]
    assert "not_shorter" not in {w["kind"] for w in out["templates"][0]["warnings"]}
    out = rewrite(req(original, "Make it shorter"), FakeLLM(same))      # still not shorter after the repair
    assert "not_shorter" in {w["kind"] for w in out["templates"][0]["warnings"]}


def test_longer_is_checked_too_and_no_cue_means_no_check():
    original = reminder_template()
    same = copy.deepcopy(REMINDER["output"])
    out = rewrite(req(original, "Expand it with more detail"), FakeLLM(same))
    assert "not_longer" in {w["kind"] for w in out["templates"][0]["warnings"]}
    llm = FakeLLM(same)
    rewrite(req(original, "Make it friendlier"), llm)
    assert len(llm.calls) == 1
