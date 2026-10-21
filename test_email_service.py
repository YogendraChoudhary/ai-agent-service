"""
generate(): routing, image slots, the single repair pass, variants, and the /agent/email/* routes.
The model is a scripted fake, so no provider key is needed.
"""

import base64
import copy
import io
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import api
from src.email_template.categories import ALL_CATEGORIES, load_example
from src.email_template.service import (
    EmailGenerationError, EmailProviderError, GenerateRequest, generate, is_failed_run, requested_image_slots,
)

REMINDER = load_example("reminder_deadline")
INVITATION = load_example("invitation")


def request_from_example(example, **kw):
    inp = example["input"]
    base = dict(
        brief=inp["brief"], audience=inp.get("audience"), language=inp["language"], length=inp["length"],
        allowed_merge_tags=tuple(inp["allowed_merge_tags"]), links={k: f"https://x.test/{k}" for k in inp["links"]},
        assets=inp["assets"],
    )
    base.update(kw)
    return GenerateRequest(**base)


class FakeLLM:
    """Returns the scripted replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages):
        self.calls.append((system, copy.deepcopy(messages)))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply if isinstance(reply, str) else json.dumps(reply)


class TestGenerate(unittest.TestCase):
    def test_malformed_blocks_are_repaired(self):
        for blocks in (7, True, "invalid", {"type": "footer"}):
            with self.subTest(blocks=blocks):
                llm = FakeLLM({"blocks": blocks}, REMINDER["output"])
                out = generate(request_from_example(REMINDER), llm)
                self.assertEqual(len(llm.calls), 2)
                self.assertIn("[schema]", llm.calls[1][1][-1]["content"])
                self.assertEqual(len(out["templates"]), 1)

    def test_missing_button_links_are_fill_slots(self):
        for category in ("reminder_deadline", "invitation", "product_launch", "feature_update"):
            for hero_button in ((False, True) if category in ("product_launch", "feature_update") else (False,)):
                with self.subTest(category=category, hero_button=hero_button):
                    example = load_example(category)
                    reply = copy.deepcopy(example["output"])
                    button = next((b for b in reply["blocks"] if b["type"] == "button"), None)
                    hero = next((b for b in reply["blocks"] if b["type"] == "hero_banner"), None)
                    if hero_button and button:
                        hero["button"] = {k: button[k] for k in ("label", "link")}
                        reply["blocks"].remove(button)
                    elif not hero_button and hero and hero.get("button"):
                        reply["blocks"].insert(-1, {"id": "cta", "type": "button", **hero.pop("button")})
                    owner = next(b for b in reply["blocks"] if b["type"] == ("hero_banner" if hero_button else "button"))
                    target = owner["button"] if hero_button else owner
                    target["link"] = "[[fill: action URL]]"
                    llm = FakeLLM(reply)
                    entry = generate(request_from_example(example, links={}), llm)["templates"][0]
                    self.assertEqual(len(llm.calls), 1)
                    self.assertIn({"block_id": owner["id"], "field": "button.link" if hero_button else "link",
                                   "needed": "action URL"}, entry["fill_slots"])
                    self.assertIn('[[fill: action URL]]', llm.calls[0][0])

    def test_valid_first_reply(self):
        llm = FakeLLM(REMINDER["output"])
        out = generate(request_from_example(REMINDER), llm)
        self.assertEqual((out["resolved_category"], out["routing"], out["tone"]),
                         ("reminder_deadline", "matched", "formal"))
        self.assertEqual(len(llm.calls), 1)
        tpl = out["templates"][0]["template"]
        self.assertEqual(tpl["meta"]["subject"], REMINDER["output"]["meta"]["subject_options"][0])

    def test_urls_never_reach_the_model(self):
        llm = FakeLLM(REMINDER["output"])
        generate(request_from_example(REMINDER), llm)
        system, messages = llm.calls[0]
        self.assertNotIn("https://x.test", system + json.dumps(messages))
        self.assertIn("training_portal", messages[0]["content"])

    def test_one_repair_pass(self):
        bad = copy.deepcopy(REMINDER["output"])
        bad["blocks"][3]["text"] = "Please complete it ASAP!!! Act now 🎉"
        llm = FakeLLM(bad, REMINDER["output"])
        out = generate(request_from_example(REMINDER), llm)
        self.assertEqual(len(llm.calls), 2)
        repair_turn = llm.calls[1][1][-1]["content"]
        self.assertIn("[emoji]", repair_turn)
        self.assertIn("[banned_phrase]", repair_turn)
        self.assertIn("repaired", [w["kind"] for w in out["templates"][0]["warnings"]])

    def test_still_invalid_after_repair_raises(self):
        llm = FakeLLM("not json at all")
        with self.assertRaises(EmailGenerationError) as cm:
            generate(request_from_example(REMINDER), llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(cm.exception.errors[0].kind, "json")

    def test_repair_can_be_switched_off(self):
        llm = FakeLLM("not json")
        with patch.object(api.settings, "email_ai_repair_enabled", False), self.assertRaises(EmailGenerationError):
            generate(request_from_example(REMINDER), llm)
        self.assertEqual(len(llm.calls), 1)

    def test_variants_drop_failures(self):
        llm = FakeLLM(REMINDER["output"])
        out = generate(request_from_example(REMINDER, variants=3), llm)
        self.assertEqual(len(out["templates"]), 3)

    def test_selected_category_and_tone(self):
        llm = FakeLLM(REMINDER["output"])
        out = generate(request_from_example(REMINDER, category="reminder_deadline", tone="professional"), llm)
        self.assertEqual((out["routing"], out["tone"]), ("selected", "professional"))
        self.assertIn("Professional:", llm.calls[0][0])

    def test_image_request_opens_hero_slot(self):
        llm = FakeLLM(INVITATION["output"])
        out = generate(request_from_example(INVITATION), llm)
        self.assertIn('"image_slots": [\n  "hero"\n ]', llm.calls[0][1][0]["content"])
        blocks = out["templates"][0]["template"]["blocks"]
        self.assertEqual(blocks[1]["type"], "image")


class TestFailedRunDetection(unittest.TestCase):
    def test_agno_error_output_is_a_failure(self):
        from agno.run.agent import RunOutput
        from agno.run.base import RunStatus

        self.assertTrue(is_failed_run(RunOutput(content="Error code: 401", status=RunStatus.error)))
        self.assertFalse(is_failed_run(RunOutput(content="{}", status=RunStatus.completed)))


class TestSignatureOrSocial(unittest.TestCase):
    def test_social_links_mean_no_signature(self):
        no_sig = copy.deepcopy(REMINDER["output"])
        no_sig["blocks"] = [b for b in no_sig["blocks"] if b["type"] != "signature"]
        llm = FakeLLM(no_sig)
        out = generate(request_from_example(REMINDER, social_links={"linkedin": "https://l.example"}), llm)
        self.assertIn('"sign_off": "none"', llm.calls[0][1][0]["content"])
        blocks = out["templates"][0]["template"]["blocks"]
        self.assertNotIn("signature", [b["type"] for b in blocks])
        self.assertEqual(blocks[-1]["social"][0]["platform"], "linkedin")

    def test_asking_for_both_is_rejected(self):
        llm = FakeLLM(REMINDER["output"])
        with self.assertRaises(ValueError):
            generate(request_from_example(REMINDER, sign_off="signature",
                                          social_links={"linkedin": "https://l.example"}), llm)
        self.assertEqual(llm.calls, [])


class TestRequestedImageSlots(unittest.TestCase):
    INV = ALL_CATEGORIES["invitation"]

    def test_cases(self):
        cases = [
            ("Invite clients to the summit. Add a banner image of a conference hall.", self.INV, ("hero",)),
            ("Invite clients and include an image of the venue in the body", self.INV, ("inline",)),
            ("Invite clients and add an image of the venue", self.INV, ("hero",)),
            ("Invite clients with two images of the venue", self.INV, ("hero", "inline")),
            ("Invite clients to the summit", self.INV, ()),
            ("Remind staff and generate an image of a calendar", ALL_CATEGORIES["reminder_deadline"], ("inline",)),
            ("Explain our policy on brand images", ALL_CATEGORIES["policy_notice"], ()),
        ]
        for brief, cat, expected in cases:
            with self.subTest(brief=brief):
                self.assertEqual(requested_image_slots(brief, cat), expected)


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def _body(self, **kw):
        inp = REMINDER["input"]
        body = {"model": "claude", "api_key": "sk-test", "brief": inp["brief"],
                "allowed_merge_tags": inp["allowed_merge_tags"],
                "links": {"training_portal": "https://portal.example.com"}}
        body.update(kw)
        return body

    def test_generate_ok(self):
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM(REMINDER["output"])):
            r = self.client.post("/agent/email/generate", json=self._body())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["resolved_category"], "reminder_deadline")
        self.assertIn("X-Request-ID", r.headers)

    def test_generate_422_with_issues(self):
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM("nope")):
            r = self.client.post("/agent/email/generate", json=self._body())
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["issues"][0]["kind"], "json")

    def test_provider_failure_is_502_not_a_validation_error(self):
        def broken(system, messages):
            raise EmailProviderError("claude request failed: bad key")
        with patch("src.email_template.service.make_agno_llm", return_value=broken):
            r = self.client.post("/agent/email/generate", json=self._body())
        self.assertEqual(r.status_code, 502)
        self.assertIn("bad key", r.json()["error"])

    def test_generate_unknown_category_400(self):
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM("{}")):
            r = self.client.post("/agent/email/generate", json=self._body(category="nope"))
        self.assertEqual(r.status_code, 400)

    def test_generate_rejects_non_https_links(self):
        r = self.client.post("/agent/email/generate", json=self._body(links={"portal": "javascript:alert(1)"}))
        self.assertEqual(r.status_code, 400)

    def test_kill_switch(self):
        with patch.object(api.settings, "email_ai_enabled", False):
            r = self.client.post("/agent/email/generate", json=self._body())
        self.assertEqual(r.status_code, 503)

    def test_launch_with_sign_off_none_social_and_theme(self):
        launch = load_example("product_launch")
        body = {"model": "openai", "api_key": "sk-test", "brief": launch["input"]["brief"],
                "allowed_merge_tags": launch["input"]["allowed_merge_tags"],
                "links": {"product_page": "https://acme.example.com/insights"},
                "assets": launch["input"]["assets"], "sign_off": "none",
                "social_links": {"linkedin": "https://linkedin.com/company/acme", "email": "mailto:hi@acme.example.com"},
                "theme": {"accent_color": "#22C3E6", "footer": {"style": "dark"}}}
        with patch("src.email_template.service.make_agno_llm", return_value=FakeLLM(launch["output"])) as mk:
            r = self.client.post("/agent/email/generate", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        tpl = out["templates"][0]["template"]
        self.assertEqual(out["resolved_category"], "product_launch")
        self.assertEqual([s["platform"] for s in tpl["blocks"][-1]["social"]], ["linkedin", "email"])
        self.assertEqual(tpl["theme"]["footer"], {"style": "dark"})
        self.assertNotIn("signature", [b["type"] for b in tpl["blocks"]])
        sent = mk.return_value.calls[0][1][0]["content"]
        self.assertIn('"sign_off": "none"', sent)
        self.assertNotIn("linkedin.com", sent)

    def test_bad_social_and_theme_inputs_are_400(self):
        for extra in ({"social_links": {"myspace": "https://m.example"}},
                      {"social_links": {"linkedin": "http://insecure.example"}},
                      {"social_links": {k: "https://s.example" for k in ("linkedin", "x", "facebook", "instagram", "youtube")}},
                      {"theme": {"footer": {"style": "custom"}}},
                      {"theme": {"accent_color": "cyan"}},
                      {"sign_off": "maybe"},
                      {"sign_off": "signature", "social_links": {"x": "https://x.com/acme"}}):
            with self.subTest(extra=extra):
                self.assertEqual(self.client.post("/agent/email/generate", json=self._body(**extra)).status_code, 400)

    def test_image_fit(self):
        buf = io.BytesIO()
        Image.new("RGB", (1536, 1024), (10, 120, 200)).save(buf, "PNG")
        uploaded = {"cdn_url": "https://cdn.test/x.jpg"}
        with patch("src.s3_service.upload_image_from_bytes_detailed", return_value=uploaded) as up:
            r = self.client.post("/agent/email/image", json={
                "slot": "hero", "image_base64": base64.b64encode(buf.getvalue()).decode()})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["width"], body["height"], body["display_width"]), (1200, 600, 600))
        self.assertEqual(body["url"], "https://cdn.test/x.jpg")
        self.assertEqual(up.call_args.args[1], "image/jpeg")

    def test_image_bad_base64(self):
        r = self.client.post("/agent/email/image", json={"slot": "hero", "image_base64": "!!!!not-base64!!!!"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
