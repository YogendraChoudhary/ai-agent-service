"""
Evaluates the image intent router against the labelled corpus.

Hard requirement: zero false positives. A wrongly routed request generates (and charges for)
an image the user never asked for, whereas a miss only falls back to the slower agent path.
"""

import unittest
from collections import Counter

from src.intent import detect_image_intent
from tests.fixtures.image_intent_eval_cases import CASES


class TestImageIntentEval(unittest.TestCase):

    def _run(self):
        rows = []
        for group, query, has_ref, has_prev, expected in CASES:
            got = detect_image_intent(query, has_reference=has_ref, has_previous_image=has_prev)
            rows.append((group, query, has_ref, has_prev, expected, got))
        return rows

    def test_no_false_positives(self):
        misroutes = [
            f"[{g}] {q!r} (ref={r}, prev={p}) -> {got['kind']} via {got['rule']}"
            for g, q, r, p, exp, got in self._run()
            if exp == "none" and got["kind"] != "none"
        ]
        self.assertEqual(misroutes, [], "Text requests routed to image generation:\n" + "\n".join(misroutes))

    def test_image_requests_are_recognised(self):
        wanted = [row for row in self._run() if row[4] != "none"]
        misses = [
            f"[{g}] {q!r} (ref={r}, prev={p}) expected {exp}, got {got['kind']} via {got['rule']}"
            for g, q, r, p, exp, got in wanted
            if got["kind"] != exp
        ]
        self.assertEqual(misses, [], "Image requests the router no longer recognises:\n" + "\n".join(misses))

    def test_followup_fields(self):
        redo = detect_image_intent("regenerate", has_previous_image=True)
        self.assertEqual((redo["kind"], redo["followup_mode"], redo["is_followup"]), ("generate", "redo", True))
        delta = detect_image_intent("make it night", has_previous_image=True)
        self.assertEqual((delta["kind"], delta["followup_mode"], delta["uses_previous_image"]), ("edit", "delta", True))
        none = detect_image_intent("make it night")
        self.assertEqual(none["kind"], "none")
        self.assertTrue(none["requires_context"])

    def test_attached_reference_beats_followup_logic(self):
        res = detect_image_intent("make it look like an oil painting", has_reference=True)
        self.assertEqual(res["kind"], "edit")
        self.assertFalse(res["uses_previous_image"])
        self.assertFalse(res["is_followup"])

    def test_corpus_is_not_tiny(self):
        self.assertGreaterEqual(len(CASES), 120)
        groups = Counter(c[0] for c in CASES)
        for needed in ("text_payload", "text_followup", "followup_no_prev", "edit_no_image_available"):
            self.assertIn(needed, groups)


if __name__ == "__main__":
    unittest.main()
