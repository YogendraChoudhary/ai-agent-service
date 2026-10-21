"""
The validator is the only gate between the model and the editor: schema, references, structure, images, tone.
"""

import copy
import json
import unittest

from src.email_template.categories import ALL_CATEGORIES
from src.email_template.tone import lint
from src.email_template.validate import Context, parse_json, validate

BRIEF = "Remind employees that the annual compliance training must be completed by 30 November."


def ctx(category="reminder_deadline", **kw):
    base = dict(
        category=ALL_CATEGORIES[category], routing="selected", tone="formal", language="en-GB",
        allowed_merge_tags=frozenset({"first_name", "company_name", "sender_name"}),
        link_keys=frozenset({"training_portal"}), asset_keys=frozenset({"logo_primary"}),
        image_slots=(), source_texts=(BRIEF,),
    )
    base.update(kw)
    return Context(**base)


GOOD = {
    "meta": {
        "subject_options": ["Reminder: training due 30 November", "Complete your training", "Training deadline"],
        "preheader": "Please complete the annual compliance training before 30 November",
    },
    "blocks": [
        {"id": "b1", "type": "header", "logo_asset": "asset:logo_primary", "org_name": "{{company_name}}"},
        {"id": "b2", "type": "greeting", "text": "Dear {{first_name}},"},
        {"id": "b3", "type": "paragraph", "text": "Please complete the training. See the [portal](link:training_portal)."},
        {"id": "b4", "type": "key_value", "rows": [{"label": "Deadline", "value": "30 November"}]},
        {"id": "b5", "type": "button", "label": "Open the portal", "link": "link:training_portal"},
        {"id": "b6", "type": "signature", "closing": "Kind regards,", "name": "{{sender_name}}"},
        {"id": "b7", "type": "footer"},
    ],
}


def with_blocks(*blocks, base=GOOD):
    t = copy.deepcopy(base)
    t["blocks"] = list(blocks)
    return t


def kinds(result):
    return sorted({e.kind for e in result.errors})


class TestParse(unittest.TestCase):
    def test_fences_and_prose_are_tolerated(self):
        raw = "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps"
        self.assertEqual(parse_json(raw)["meta"]["preheader"], GOOD["meta"]["preheader"])

    def test_not_json(self):
        r = validate("I cannot help with that.", ctx())
        self.assertEqual(kinds(r), ["json"])
        self.assertIsNone(r.template)


class TestValid(unittest.TestCase):
    def test_good_template_passes(self):
        r = validate(json.dumps(GOOD), ctx())
        self.assertTrue(r.ok, r.errors)
        meta = r.template["meta"]
        self.assertEqual(meta["subject"], GOOD["meta"]["subject_options"][0])
        self.assertEqual((meta["category"], meta["tone"], r.template["schema_version"]),
                         ("reminder_deadline", "formal", 1))

    def test_code_owned_fields_override_the_model(self):
        t = copy.deepcopy(GOOD)
        t["meta"].update(category="newsletter", tone="professional_warm", routing="general")
        r = validate(t, ctx())
        self.assertEqual((r.template["meta"]["category"], r.template["meta"]["tone"], r.template["meta"]["routing"]),
                         ("reminder_deadline", "formal", "selected"))

    def test_fill_slots_are_reported(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][3]["rows"].append({"label": "Venue", "value": "[[fill: training room]]"})
        r = validate(t, ctx())
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.fill_slots, [{"block_id": "b4", "field": "rows[1].value", "needed": "training room"}])

    def test_markup_is_stripped_with_a_warning(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][2]["text"] = "Please <b>complete</b> the training.<script>x()</script>"
        r = validate(t, ctx())
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.template["blocks"][2]["text"], "Please complete the training.x()")
        self.assertIn("sanitised", [w.kind for w in r.warnings])


class TestSchema(unittest.TestCase):
    def test_unknown_field_rejected(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][2]["html"] = "<p>hi</p>"
        self.assertEqual(kinds(validate(t, ctx())), ["schema"])

    def test_unknown_block_type_rejected(self):
        t = copy.deepcopy(GOOD)
        t["blocks"].insert(2, {"id": "x", "type": "video", "src": "a.mp4"})
        self.assertEqual(kinds(validate(t, ctx())), ["schema"])

    def test_two_subject_options_rejected(self):
        t = copy.deepcopy(GOOD)
        t["meta"]["subject_options"] = t["meta"]["subject_options"][:2]
        self.assertEqual(kinds(validate(t, ctx())), ["schema"])

    def test_button_needs_link_reference_not_url(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][4]["link"] = "https://evil.example"
        self.assertEqual(kinds(validate(t, ctx())), ["schema"])


class TestReferences(unittest.TestCase):
    def test_unknown_merge_tag(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][1]["text"] = "Dear {{full_name}},"
        self.assertIn("unknown_merge_tag", kinds(validate(t, ctx())))

    def test_unknown_link_key(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][4]["link"] = "link:signup"
        self.assertIn("unknown_link", kinds(validate(t, ctx())))

    def test_inline_link_must_be_a_known_key(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][2]["text"] = "See the [portal](https://example.com)."
        self.assertIn("unknown_link", kinds(validate(t, ctx())))

    def test_raw_url_in_text(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][2]["text"] = "Go to www.example.com to start."
        self.assertIn("raw_url", kinds(validate(t, ctx())))

    def test_unknown_logo_asset(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][0]["logo_asset"] = "asset:other_logo"
        self.assertIn("unknown_asset", kinds(validate(t, ctx())))


class TestStructure(unittest.TestCase):
    def test_one_call_to_action(self):
        b = GOOD["blocks"]
        t = with_blocks(*b[:5], {"id": "b9", "type": "button", "label": "Read more", "link": "link:training_portal"},
                        *b[5:])
        self.assertIn("too_many_buttons", kinds(validate(t, ctx())))

    def test_header_first_footer_last(self):
        b = GOOD["blocks"]
        t = with_blocks(b[1], b[0], *b[2:])
        self.assertIn("header_position", kinds(validate(t, ctx())))

    def test_required_blocks_for_category(self):
        b = GOOD["blocks"]
        t = with_blocks(*[x for x in b if x["type"] != "key_value"])
        self.assertIn("missing_block", kinds(validate(t, ctx())))

    def test_bulk_needs_unsubscribe(self):
        t = with_blocks(
            {"id": "b1", "type": "header"}, {"id": "b2", "type": "heading", "text": "News"},
            {"id": "b3", "type": "paragraph", "text": "Our news."}, {"id": "b4", "type": "footer"},
        )
        self.assertIn("unsubscribe", kinds(validate(t, ctx("newsletter", tone="professional"))))

    def test_duplicate_ids(self):
        t = copy.deepcopy(GOOD)
        t["blocks"][3]["id"] = "b3"
        self.assertIn("duplicate_id", kinds(validate(t, ctx())))


class TestImages(unittest.TestCase):
    HERO = {"id": "i1", "type": "image", "slot": "hero", "source": None, "alt": "A conference hall",
            "generation_prompt": "Bright modern conference hall, no text"}

    def _invitation(self, *images):
        return with_blocks(
            {"id": "b1", "type": "header"}, {"id": "b2", "type": "heading", "text": "Summit"},
            {"id": "b3", "type": "paragraph", "text": "Join us at the summit."}, *images,
            {"id": "b4", "type": "key_value", "rows": [{"label": "Venue", "value": "Grand Hall"}]},
            {"id": "b5", "type": "button", "label": "Reply to confirm", "link": "link:training_portal"},
            {"id": "b6", "type": "footer"},
        )

    def _ctx(self, slots):
        return ctx("invitation", tone="professional_warm", image_slots=slots, source_texts=("Grand Hall summit",))

    def test_hero_is_moved_under_the_header(self):
        r = validate(self._invitation(self.HERO), self._ctx(("hero",)))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual([b["type"] for b in r.template["blocks"][:2]], ["header", "image"])

    def test_inline_is_moved_before_the_button(self):
        inline = dict(self.HERO, id="i2", slot="inline")
        r = validate(self._invitation(inline), self._ctx(("inline",)))
        self.assertTrue(r.ok, r.errors)
        types = [b["type"] for b in r.template["blocks"]]
        self.assertEqual(types.index("image") + 1, types.index("button"))

    def test_image_without_request_is_rejected(self):
        self.assertIn("image_not_requested", kinds(validate(self._invitation(self.HERO), self._ctx(()))))

    def test_slot_not_in_category(self):
        t = copy.deepcopy(GOOD)
        t["blocks"].insert(1, dict(self.HERO))
        self.assertIn("image_slot", kinds(validate(t, ctx(image_slots=("hero",)))))

    def test_model_cannot_claim_a_generated_file(self):
        img = dict(self.HERO, source="generated", asset="asset:fake")
        self.assertIn("image_source", kinds(validate(self._invitation(img), self._ctx(("hero",)))))


LAUNCH_BASE = {
    "meta": {"subject_options": ["Introducing Acme Insights", "Meet Acme Insights", "Acme Insights is here"],
             "preheader": "A new analytics dashboard that brings your finance reports together"},
    "blocks": [
        {"id": "b1", "type": "header", "org_name": "{{company_name}}"},
        {"id": "b2", "type": "hero_banner", "eyebrow": "New", "headline": "Meet Acme Insights",
         "summary": "Your finance reports in one place.",
         "button": {"label": "See Acme Insights", "link": "link:training_portal"}},
        {"id": "b3", "type": "features", "items": [{"title": "One place", "text": "All reports together."},
                                                    {"title": "Shareable", "text": "Share with a link."}]},
        {"id": "b4", "type": "footer", "unsubscribe": True},
    ],
}


def launch_ctx(**kw):
    return ctx("product_launch", tone="professional", source_texts=("Acme Insights launch",), **kw)


class TestLaunchBlocks(unittest.TestCase):
    def test_launch_template_passes(self):
        r = validate(copy.deepcopy(LAUNCH_BASE), launch_ctx())
        self.assertTrue(r.ok, r.errors)

    def test_hero_banner_button_is_the_one_cta(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"].insert(3, {"id": "b9", "type": "button", "label": "Learn more", "link": "link:training_portal"})
        self.assertIn("too_many_buttons", kinds(validate(t, launch_ctx())))

    def test_cta_may_be_a_button_block_instead(self):
        t = copy.deepcopy(LAUNCH_BASE)
        del t["blocks"][1]["button"]
        self.assertIn("missing_block", kinds(validate(t, launch_ctx())))
        t["blocks"].insert(3, {"id": "b9", "type": "button", "label": "Learn more", "link": "link:training_portal"})
        self.assertTrue(validate(t, launch_ctx()).ok)

    def test_hero_banner_button_link_must_be_known(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"][1]["button"]["link"] = "link:elsewhere"
        self.assertIn("unknown_link", kinds(validate(t, launch_ctx())))

    def test_hero_banner_directly_after_header(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"][1], t["blocks"][2] = t["blocks"][2], t["blocks"][1]
        self.assertIn("hero_banner_position", kinds(validate(t, launch_ctx())))

    def test_hero_image_goes_under_the_banner(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"].insert(3, {"id": "i1", "type": "image", "slot": "hero", "alt": "Dashboard on a laptop",
                               "generation_prompt": "Laptop with abstract charts, no text"})
        r = validate(t, launch_ctx(image_slots=("hero",)))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual([b["type"] for b in r.template["blocks"][:3]], ["header", "hero_banner", "image"])

    def test_feature_text_is_tone_checked(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"][2]["items"][0]["text"] = "A total game changer!"
        self.assertTrue({"banned_phrase", "exclamation"} <= set(kinds(validate(t, launch_ctx()))))

    def test_launch_blocks_not_in_letter_categories(self):
        t = copy.deepcopy(GOOD)
        t["blocks"].insert(1, {"id": "h1", "type": "hero_banner", "headline": "Reminder"})
        self.assertIn("block_not_allowed", kinds(validate(t, ctx())))


class TestSignOffSocialTheme(unittest.TestCase):
    def test_sign_off_none_forbids_signature_even_where_required(self):
        r = validate(copy.deepcopy(GOOD), ctx(sign_off="none"))
        self.assertIn("signature_not_wanted", kinds(r))
        t = with_blocks(*[b for b in GOOD["blocks"] if b["type"] != "signature"])
        self.assertTrue(validate(t, ctx(sign_off="none")).ok)

    def test_sign_off_signature_requires_one(self):
        t = copy.deepcopy(LAUNCH_BASE)
        self.assertIn("missing_block", kinds(validate(t, launch_ctx(sign_off="signature"))))

    def test_signature_and_social_icons_are_exclusive(self):
        # Even a type that normally requires a signature drops it when the footer carries social icons.
        r = validate(copy.deepcopy(GOOD), ctx(social=("linkedin", "x")))
        self.assertIn("signature_not_wanted", kinds(r))
        t = with_blocks(*[b for b in GOOD["blocks"] if b["type"] != "signature"])
        r = validate(t, ctx(social=("linkedin", "x")))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(len(r.template["blocks"][-1]["social"]), 2)

    def test_social_links_come_from_the_request_only(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["blocks"][-1]["social"] = [{"platform": "facebook", "link": "link:made_up"}]
        r = validate(t, launch_ctx(social=("linkedin", "x")))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.template["blocks"][-1]["social"],
                         [{"platform": "linkedin", "link": "link:social_linkedin"},
                          {"platform": "x", "link": "link:social_x"}])

    def test_theme_is_the_users_not_the_models(self):
        t = copy.deepcopy(LAUNCH_BASE)
        t["theme"] = {"footer": {"style": "light"}}
        theme = {"accent_color": "#22C3E6", "footer": {"style": "dark"}, "hero": {"style": "dark"}}
        r = validate(t, launch_ctx(theme=theme))
        self.assertEqual(r.template["theme"], theme)


class TestTone(unittest.TestCase):
    def _lint(self, text, tone="formal"):
        t = copy.deepcopy(GOOD)
        t["blocks"][2]["text"] = text
        return lint(t, tone, (BRIEF,))

    def test_clean_text(self):
        self.assertEqual(self._lint("Please complete the training before the deadline.").errors, [])

    def test_emoji_caps_hype_and_exclamations(self):
        res = self._lint("This is AMAZING! Act now 🎉")
        self.assertEqual(sorted({e.kind for e in res.errors}), ["all_caps", "banned_phrase", "emoji", "exclamation"])

    def test_warm_tone_allows_one_exclamation(self):
        self.assertEqual([e.kind for e in self._lint("We look forward to seeing you!", "professional_warm").errors], [])
        self.assertEqual([e.kind for e in self._lint("Great! See you!", "professional_warm").errors], ["exclamation"])

    def test_subject_alternatives_have_independent_exclamation_limits(self):
        for tone, subjects, body, preheader, rejected in (
            ("professional_warm", ["Welcome!", "Join us!", "You are invited!"], "See you.", "Details below.", False),
            ("professional_warm", ["Welcome!", "Join us", "You are invited"], "See you!", "Details below.", True),
            ("professional_warm", ["Welcome", "Join us!!", "You are invited"], "See you.", "Details below.", True),
            ("professional_warm", ["Welcome!", "Join us", "You are invited"], "See you.", "Details below!", True),
            ("professional", ["Welcome!", "Join us", "You are invited"], "See you.", "Details below.", True),
            ("formal", ["Welcome", "Join us!", "You are invited"], "See you.", "Details below.", True),
        ):
            with self.subTest(tone=tone, subjects=subjects, body=body, preheader=preheader):
                t = copy.deepcopy(GOOD)
                t["meta"].update(subject_options=subjects, preheader=preheader)
                t["blocks"][2]["text"] = body
                errors = [e for e in lint(t, tone, (BRIEF,)).errors if e.kind == "exclamation"]
                self.assertEqual(bool(errors), rejected)

    def test_acronyms_are_fine(self):
        self.assertEqual(self._lint("Contact HR or read the GDPR FAQ.").errors, [])

    def test_invented_number_is_a_warning(self):
        res = self._lint("Over 5000 colleagues have finished already.")
        self.assertEqual(res.errors, [])
        self.assertIn("possible_invented_fact", [w.kind for w in res.warnings])

    def test_number_from_brief_is_fine(self):
        res = self._lint("The deadline is 30 November.")
        self.assertNotIn("possible_invented_fact", [w.kind for w in res.warnings])


if __name__ == "__main__":
    unittest.main()
