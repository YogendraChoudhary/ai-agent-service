"""
Category routing for AI email templates: selected -> regex-matched -> general.

The briefs below include the ones that misrouted under the first keyword table ("update staff on...",
"in order to... confirm") and must stay green whenever a pattern changes.
"""

import unittest

from src.email_template.categories import ALL_CATEGORIES, GENERAL, enabled_categories
from src.email_template.routing import match_category, resolve_category, score_brief

# (brief, category expected when every registered category takes part)
FULL_REGISTRY_CASES = [
    ("Write an email to update staff on the new leave policy", "policy_notice"),
    ("In order to comply, please confirm your attendance at the seminar", "invitation"),
    ("Send a reminder about the team party", "reminder_deadline"),
    ("Our offices will be closed on Friday", "announcement"),
    ("We are sorry, your order is delayed", "incident_apology"),
    ("Tell our team the office is moving to the Bannerghatta Road campus from next month", None),
    ("You are invited to our annual gala, please RSVP by 10 May", "invitation"),
    ("Welcome our new joiners starting Monday", "onboarding_welcome"),
    ("Payment received for invoice 2231", "confirmation"),
    ("Quarterly update for our investors", "newsletter"),
    ("Following up on yesterday's meeting with the vendor", "meeting_request"),
    ("Reminder: annual compliance training deadline is Friday", "reminder_deadline"),
    ("Write an email", None),
    ("Thank you for attending our webinar", "follow_up"),
    ("We are launching our new analytics dashboard next month", "product_launch"),
    ("Introducing our new mobile app", "product_launch"),
    ("Introducing Eddyter Templates: ready-made email layouts", "product_launch"),
    ("We introduce a new pricing calculator for teams", "product_launch"),
    ("Rolling out dark mode to all users this week", "feature_update"),
    ("Product update: the editor now supports tables", "feature_update"),
    ("Announce our new feature: scheduled exports", "feature_update"),
    ("Invite customers to our product launch event", "invitation"),
]


class TestRoutingPatterns(unittest.TestCase):
    def test_full_registry(self):
        for brief, expected in FULL_REGISTRY_CASES:
            with self.subTest(brief=brief):
                self.assertEqual(match_category(brief, ALL_CATEGORIES), expected)

    def test_repeated_phrase_counts_once(self):
        scores = score_brief("Reminder reminder REMINDER", ALL_CATEGORIES)
        self.assertEqual(scores["reminder_deadline"], 1)

    def test_general_has_no_pattern(self):
        self.assertIsNone(ALL_CATEGORIES[GENERAL].pattern)


class TestResolve(unittest.TestCase):
    def test_selected_wins_over_regex(self):
        cat, routing = resolve_category("Reminder: the deadline is Friday", selected="announcement")
        self.assertEqual((cat.id, routing), ("announcement", "selected"))

    def test_matched(self):
        cat, routing = resolve_category("You are invited to our gala")
        self.assertEqual((cat.id, routing), ("invitation", "matched"))

    def test_no_match_is_general(self):
        cat, routing = resolve_category("Tell the team about the parking changes")
        self.assertEqual((cat.id, routing), (GENERAL, "general"))

    def test_disabled_category_is_never_matched(self):
        # confirmation is defined but not enabled until it has an example
        self.assertNotIn("confirmation", enabled_categories())
        cat, routing = resolve_category("Payment received for invoice 2231")
        self.assertEqual((cat.id, routing), (GENERAL, "general"))

    def test_unknown_or_disabled_selection_is_rejected(self):
        for bad in ("nope", "confirmation"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_category("anything", selected=bad)


if __name__ == "__main__":
    unittest.main()
