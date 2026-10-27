"""
Integration tests for Mode 8 Editable UI Design Engine endpoints:
  - POST /agent/stream (SSE stream with named events: init, design_ready, complete)
  - POST /agent/process (Synchronous parity with serialized AST + design payload)
  - Upfront credit reservation (3.0 credits generation, 0.5 credits edit)
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
import src.text_processing_agent as tpa
from config import settings
from src.design_canvas.schema import DesignDocumentAST

GOLDEN_AST_DICT = {
    "block_type": "eddyter_design_canvas",
    "version": "1.0",
    "design_id": "dsn_test_api_01",
    "title": "Onboarding Card",
    "theme_palette": ["#000000", "#34F5A6", "#FFFFFF"],
    "root": {
        "node_type": "container",
        "id": "root_card",
        "layout": "flex_col",
        "children": [
            {
                "node_type": "text",
                "id": "title_text",
                "tag": "h1",
                "spans": [{"text": "Welcome", "color": "#FFFFFF", "font_weight": "bold"}],
            }
        ],
    },
}

MOCK_USER_META = {"user_id": "user_123", "requires_user_api_keys": False, "user_keys": {}}


def parse_sse_events(response_text: str) -> list[dict]:
    """Parse raw SSE text into list of {event: ..., data: ...} dicts."""
    events = []
    current_event = None
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_raw = line[6:].strip()
            try:
                data_obj = json.loads(data_raw)
            except Exception:
                data_obj = data_raw
            events.append({"event": current_event, "data": data_obj})
            current_event = None
    return events


class TestDesignApi(unittest.TestCase):
    def setUp(self):
        self.mock_ast = DesignDocumentAST.model_validate(GOLDEN_AST_DICT)
        self.mock_harness = AsyncMock(return_value=self.mock_ast)

        self.patches = [
            patch("src.design_canvas.service.execute_design_canvas_harness", self.mock_harness),
            patch.object(settings, "design_canvas_enabled", True),
            patch.object(settings, "task_routing_enabled", False),
            patch.object(settings, "provider_fallback_enabled", False),
            patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("fake_key", MOCK_USER_META))),
            patch.object(api, "credit_manager", None),
            patch.object(api, "CREDIT_PRECHECK_ENABLED", False),
            patch.object(api, "fire_prompt_history_save"),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_stream_design_canvas_emits_contract_events(self):
        """POST /agent/stream must emit init, design_ready, and complete events."""
        payload = {
            "type": "general",
            "query": "design an onboarding card for users",
            "model": "claude",
            "api_key": "lic_test_123",
        }
        res = self.client.post("/agent/stream", json=payload)
        self.assertEqual(res.status_code, 200, res.text)

        events = parse_sse_events(res.text)
        named_events = {e["event"]: e["data"] for e in events if e["event"] is not None}

        # 1. Event 'init' must precede long generation
        self.assertIn("init", named_events)
        self.assertEqual(named_events["init"]["mode"], "design_canvas")
        self.assertEqual(named_events["init"]["provider"], "claude")

        # 2. Event 'design_ready' carries valid DesignDocumentAST
        self.assertIn("design_ready", named_events)
        self.assertEqual(named_events["design_ready"]["block_type"], "eddyter_design_canvas")
        self.assertEqual(named_events["design_ready"]["design_id"], "dsn_test_api_01")
        self.assertEqual(named_events["design_ready"]["root"]["id"], "root_card")

        # 3. Event 'complete' carries metadata
        self.assertIn("complete", named_events)
        self.assertEqual(named_events["complete"]["mode"], "design_canvas")

    def test_process_design_canvas_synchronous_parity(self):
        """POST /agent/process must return serialized AST in markdown_response and design payload."""
        payload = {
            "type": "general",
            "query": "create a pricing card with three tiers",
            "model": "claude",
            "api_key": "fake_provider_key",
        }
        res = self.client.post("/agent/process", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertTrue(data["success"])
        self.assertIn("markdown_response", data)
        # Parse markdown_response as JSON DesignDocumentAST
        parsed_ast = json.loads(data["markdown_response"])
        self.assertEqual(parsed_ast["block_type"], "eddyter_design_canvas")
        self.assertEqual(parsed_ast["design_id"], "dsn_test_api_01")

        # Check structured design field
        self.assertIn("design", data)
        self.assertEqual(data["design"]["block_type"], "eddyter_design_canvas")
        self.assertEqual(data["design"]["root"]["id"], "root_card")

    def test_upfront_credit_reservation_new_generation(self):
        """Generating a new design reserves 3.0 scaled credits (30 units)."""
        from src.credit_manager import DESIGN_CANVAS_GENERATION_CREDITS_SCALED

        reserved = []
        mock_credits = type("CreditManagerMock", (), {})()
        mock_credits.reserve = lambda user_id, scaled: (reserved.append(scaled) or True, 10_000)

        with patch.object(api, "credit_manager", mock_credits), patch.object(api, "CREDIT_PRECHECK_ENABLED", True):
            payload = {
                "type": "general",
                "query": "design a hero section for our home page",
                "model": "claude",
                "api_key": "lic_test_123",
            }
            res = self.client.post("/agent/stream", json=payload)
            self.assertEqual(res.status_code, 200, res.text)

        self.assertEqual(reserved[0], DESIGN_CANVAS_GENERATION_CREDITS_SCALED)
        self.assertEqual(reserved[0], 30)

    def test_upfront_credit_reservation_existing_ast_edit(self):
        """Editing an existing AST reserves 0.5 scaled credits (5 units)."""
        from src.credit_manager import DESIGN_CANVAS_EDIT_CREDITS_SCALED

        reserved = []
        mock_credits = type("CreditManagerMock", (), {})()
        mock_credits.reserve = lambda user_id, scaled: (reserved.append(scaled) or True, 10_000)

        with patch.object(api, "credit_manager", mock_credits), patch.object(api, "CREDIT_PRECHECK_ENABLED", True):
            payload = {
                "type": "general",
                "query": "update the title color to green",
                "model": "claude",
                "api_key": "lic_test_123",
                "existing_design_ast": GOLDEN_AST_DICT,
            }
            res = self.client.post("/agent/stream", json=payload)
            self.assertEqual(res.status_code, 200, res.text)

        self.assertEqual(reserved[0], DESIGN_CANVAS_EDIT_CREDITS_SCALED)
        self.assertEqual(reserved[0], 5)

    def test_edit_existing_ast_passes_sub_mode_and_json_to_harness(self):
        """When existing_design_ast is passed, harness receives sub_mode=edit_existing_ast."""
        payload = {
            "type": "general",
            "query": "change the text to goodbye",
            "model": "claude",
            "api_key": "lic_test_123",
            "existing_design_ast": GOLDEN_AST_DICT,
        }
        res = self.client.post("/agent/stream", json=payload)
        self.assertEqual(res.status_code, 200, res.text)

        self.assertTrue(self.mock_harness.called)
        call_kwargs = self.mock_harness.call_args.kwargs
        self.assertEqual(call_kwargs["sub_mode"], "edit_existing_ast")
        self.assertIsNotNone(call_kwargs["existing_ast_json"])
        passed_ast = json.loads(call_kwargs["existing_ast_json"])
        self.assertEqual(passed_ast["design_id"], "dsn_test_api_01")


class TestDesignBillingAndTypes(unittest.TestCase):
    """Gap 3 (explicit design types accepted) and gap 4 (the flat design price reaches the settlement webhook)."""

    def setUp(self):
        self.mock_harness = AsyncMock(return_value=DesignDocumentAST.model_validate(GOLDEN_AST_DICT))
        self.reserved: list[int] = []
        self.settled: list[dict] = []
        credits = type("CreditManagerMock", (), {})()
        credits.reserve = lambda user_id, scaled: (self.reserved.append(scaled) or True, 10_000)
        credits.schedule_release = lambda *a, **k: None
        credits.release = lambda *a, **k: None

        self.patches = [
            patch("src.design_canvas.service.execute_design_canvas_harness", self.mock_harness),
            patch.object(settings, "design_canvas_enabled", True),
            patch.object(settings, "task_routing_enabled", False),
            patch.object(settings, "provider_fallback_enabled", False),
            patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("fake_key", MOCK_USER_META))),
            patch.object(api, "credit_manager", credits),
            patch.object(api, "CREDIT_PRECHECK_ENABLED", True),
            patch.object(api, "fire_prompt_history_save"),
            patch.object(api, "fire_credit_deduction_and_release", lambda **kw: self.settled.append(kw)),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _stream(self, **payload):
        body = {"model": "claude", "api_key": "lic_test_123", **payload}
        res = self.client.post("/agent/stream", json=body)
        self.assertEqual(res.status_code, 200, res.text)
        return res

    def test_new_design_settles_at_the_generation_price(self):
        self._stream(type="general", query="design a hero section for our home page")
        self.assertEqual(self.reserved, [30])
        self.assertEqual(self.settled[0]["design"], {"sub_mode": "create_from_prompt", "credits_scaled": 30})

    def test_edit_settles_at_the_edit_price(self):
        self._stream(type="general", query="update the title color to green", existing_design_ast=GOLDEN_AST_DICT)
        self.assertEqual(self.reserved, [5])
        self.assertEqual(self.settled[0]["design"], {"sub_mode": "edit_existing_ast", "credits_scaled": 5})

    def test_switch_wording_with_a_design_attached_reserves_the_edit_price(self):
        self._stream(type="general", query="Switch this to a light theme: white background and dark text.",
                     existing_design_ast=GOLDEN_AST_DICT)
        self.assertEqual(self.reserved, [5])
        self.assertEqual(self.mock_harness.call_args.kwargs["sub_mode"], "edit_existing_ast")

    def test_failed_design_sends_no_design_price(self):
        self.mock_harness.side_effect = ValueError("DesignDocumentAST failed schema validation")
        self._stream(type="general", query="design a hero section for our home page")
        self.assertEqual(self.settled, [])  # the error path releases the hold; nothing is charged at the design price

    def test_typed_operation_with_design_words_reserves_text_credits(self):
        def plain_text(*args, **kwargs):
            yield "Fixed."

        with patch.object(api, "process_user_query_stream", plain_text):
            self._stream(type="fix_grammar", query="Make a hero section UI please")
        self.assertEqual(self.reserved, [10])               # 1 text credit, not the 3.0 design price
        self.assertIsNone(self.settled[0]["design"])

    def test_generate_design_type_is_accepted_and_runs_the_engine(self):
        res = self._stream(type="generate_design", query="something calm for the pricing page")
        named = {e["event"]: e["data"] for e in parse_sse_events(res.text) if e["event"]}
        self.assertIn("design_ready", named)
        self.assertEqual(self.mock_harness.call_args.kwargs["sub_mode"], "create_from_prompt")
        self.assertEqual(self.reserved, [30])

    def test_edit_design_type_is_accepted_and_edits(self):
        self._stream(type="edit_design", query="greener", existing_design_ast=GOLDEN_AST_DICT)
        self.assertEqual(self.mock_harness.call_args.kwargs["sub_mode"], "edit_existing_ast")
        self.assertEqual(self.reserved, [5])

    def test_process_accepts_generate_design(self):
        res = self.client.post("/agent/process", json={
            "type": "generate_design", "query": "something calm", "model": "claude", "api_key": "k"})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["design"]["design_id"], "dsn_test_api_01")


def test_webhook_payload_carries_the_design_price():
    from src.webhook import _payload

    body = _payload("u", 10, 0, "text", 0, "claude", "claude-opus-5", 5, True, None, None,
                    {"sub_mode": "create_from_prompt", "credits_scaled": 30})
    assert body["design"] == {"sub_mode": "create_from_prompt", "credits_scaled": 30}
    assert "design" not in _payload("u", 10, 0, "text", 0, None, None, 5, True, None, None)


def test_stream_hands_its_abort_event_to_the_pipeline():
    """/agent/abort sets this event; the design canvas watches it while it generates."""
    import threading

    seen = {}

    def pipeline(*args, **kwargs):
        seen.update(kwargs)
        yield "ok"

    with patch.object(api, "process_user_query_stream", pipeline), \
            patch.object(api, "async_validate_and_get_api_key", AsyncMock(return_value=("k", MOCK_USER_META))), \
            patch.object(api, "credit_manager", None), patch.object(api, "CREDIT_PRECHECK_ENABLED", False), \
            patch.object(api, "fire_prompt_history_save"), \
            patch.object(settings, "task_routing_enabled", False), \
            patch.object(settings, "provider_fallback_enabled", False):
        res = TestClient(api.app).post("/agent/stream", json={
            "type": "generate_design", "query": "a card", "model": "claude", "api_key": "lic"})
    assert res.status_code == 200, res.text
    assert isinstance(seen["cancel_event"], threading.Event)
