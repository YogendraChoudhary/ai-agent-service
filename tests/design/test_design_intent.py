"""
Tests for Mode 8 deterministic design intent classifier.

Coverage:
  - 40+ positive/negative cases across all four rule branches.
  - Verified zero cross-contamination with code and raster-photo negative guards.
  - Edge cases: operation_type overrides, existing_ast_json mutations, mixed signals.
  - Follow-up edits with the design attached, worded with any edit verb, a design part or a style change
    (live run 2026-10-27: "Switch this to a light theme" went to chat).
"""

from __future__ import annotations

import pytest

from src.design_intent import DesignIntentDecision, detect_design_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positive(query: str, **kwargs) -> DesignIntentDecision:
    d = detect_design_intent(query, **kwargs)
    assert d.is_design_request, (
        f"Expected is_design_request=True for: {query!r}\n  reason={d.reason}"
    )
    return d


def _negative(query: str, **kwargs) -> DesignIntentDecision:
    d = detect_design_intent(query, **kwargs)
    assert not d.is_design_request, (
        f"Expected is_design_request=False for: {query!r}\n  reason={d.reason}"
    )
    return d


# ===========================================================================
# Rule 0 — Explicit operation_type overrides
# ===========================================================================

class TestExplicitOperationType:
    def test_generate_design_prompt_mode(self):
        d = _positive("anything", operation_type="generate_design")
        assert d.sub_mode == "create_from_prompt"
        assert d.reason == "explicit_operation_type"

    def test_generate_design_vision_mode(self):
        d = _positive("anything", operation_type="generate_design", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_generate_design_edit_mode(self):
        d = _positive("anything", operation_type="generate_design", has_existing_design_ast=True)
        assert d.sub_mode == "edit_existing_ast"

    def test_edit_design_explicit(self):
        d = _positive("change the header", operation_type="edit_design")
        assert d.sub_mode == "create_from_prompt"  # no existing AST → prompt

    def test_general_operation_not_forced(self):
        # "general" operation_type must not force design mode
        d = detect_design_intent("hello world", operation_type="general")
        assert not d.is_design_request

    def test_active_ast_plus_action_verb(self):
        d = _positive("update the hero section", has_existing_design_ast=True)
        assert d.sub_mode == "edit_existing_ast"
        assert d.reason == "active_design_ast_mutation"

    def test_active_ast_no_verb_returns_false(self):
        d = _negative("hello", has_existing_design_ast=True)


# ===========================================================================
# Rule 1b — edit wording while a design is attached
# ===========================================================================

EDIT_WORDING = [
    "Switch this to a light theme: white background and dark text. Keep the mint accents.",
    "Swap the two cards",
    "Remove the illustration.",
    "Delete the divider",
    "Add a second button",
    "Replace the arrows with chevrons",
    "Translate everything to Spanish",
    "Set the background to #101010",
    "Use a serif font",
    "Now make the title bigger",
    "also, center the headline",
    "the button should be green",
    "bigger title please",
    "more padding",
    "Could the title be bigger?",
    "I want the background darker",
    "dark mode please",
    "too busy, simplify it",
    "let's tighten the spacing",
    "I want you to round the corners",
]

NOT_EDITS = [
    "hello",
    "thanks!",
    "ok",
    "looks great",
    "what font is this?",
    "why is the button green?",
    "how does this look?",
    "is the contrast ok?",
]


class TestEditWordingWithDesignAttached:
    @pytest.mark.parametrize("query", EDIT_WORDING)
    def test_edit_wording_is_an_edit(self, query):
        d = _positive(query, has_existing_design_ast=True)
        assert d.sub_mode == "edit_existing_ast"

    @pytest.mark.parametrize("query", NOT_EDITS)
    def test_chat_and_questions_stay_chat(self, query):
        _negative(query, has_existing_design_ast=True)

    @pytest.mark.parametrize("query", EDIT_WORDING)
    def test_same_wording_without_a_design_is_not_a_design(self, query):
        # "Remove the duplicate paragraph", "use a formal tone": ordinary chat when no design is selected
        _negative(query)

    def test_code_guard_still_wins(self):
        _negative("write a python function to validate the form", has_existing_design_ast=True)

    def test_raster_guard_still_wins(self):
        _negative("a photorealistic portrait of a cat for the background", has_existing_design_ast=True)

    def test_start_anchored_verbs_keep_their_original_reason(self):
        assert detect_design_intent("change the header", has_existing_design_ast=True).reason == "active_design_ast_mutation"
        assert detect_design_intent("Swap the cards", has_existing_design_ast=True).reason == "design_ast_edit_wording"


# ===========================================================================
# Rule 1 — Negative guards (code & raster photo)
# ===========================================================================

class TestNegativeCodeGuard:
    def test_python_blocks_design(self):
        _negative("create a python function that draws a ui card")

    def test_typescript_blocks_design(self):
        _negative("build a typescript react hook for the modal")

    def test_javascript_blocked(self):
        _negative("generate javascript for the landing section")

    def test_sql_blocked(self):
        _negative("create an sql schema for the user interface table")

    def test_regex_blocked(self):
        _negative("design a regex to match the card number format")

    def test_bash_blocked(self):
        _negative("make a bash script for the ui deployment")

    def test_unit_test_blocked(self):
        _negative("create unit tests for the card layout module")

    def test_api_endpoint_blocked(self):
        _negative("design an api endpoint that returns card data")

    def test_algorithm_blocked(self):
        _negative("build an algorithm for the ui search feature")


class TestNegativeRasterPhotoGuard:
    def test_photorealistic_blocked(self):
        _negative("create a photorealistic hero section image")

    def test_oil_painting_blocked(self):
        _negative("generate an oil painting of a ui card")

    def test_anime_blocked(self):
        _negative("design an anime style onboarding card")

    def test_cinematic_blocked(self):
        _negative("make a cinematic lighting hero section photo")

    def test_watercolor_blocked(self):
        _negative("create a watercolor landing section")


# ===========================================================================
# Rule 2 — Multimodal reference screenshot + UI cue
# ===========================================================================

class TestVisionAttachment:
    def test_like_this_image(self):
        d = _positive("make it look like this image", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_from_this_screenshot(self):
        d = _positive("recreate from this screenshot", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_matching_this_design(self):
        d = _positive("make matching this design please", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_make_it_editable(self):
        d = _positive("make it editable", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_vision_without_cue_returns_false(self):
        # Image attached but no UI/recreation cue in query
        d = _negative("what is this?", has_image_attachment=True)

    def test_vision_replicate_this(self):
        d = _positive("replicate this", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_vision_clone_this(self):
        d = _positive("clone this", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"

    def test_vision_artifact_keyword_without_cue(self):
        # Artifact keyword alone with image → should still trigger
        d = _positive("design a hero section", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"


# ===========================================================================
# Rule 3 — Action verb + UI artifact keyword
# ===========================================================================

class TestActionVerbPlusArtifact:
    def test_design_ui(self):
        d = _positive("design a ui for our mobile app")
        assert d.sub_mode == "create_from_prompt"

    def test_create_hero_section(self):
        d = _positive("create a hero section for the homepage")
        assert d.sub_mode == "create_from_prompt"

    def test_build_onboarding_card(self):
        d = _positive("build an onboarding card for new users")
        assert d.sub_mode == "create_from_prompt"

    def test_make_pricing_card(self):
        d = _positive("make a pricing card layout with three tiers")
        assert d.sub_mode == "create_from_prompt"

    def test_generate_wireframe(self):
        d = _positive("generate a wireframe for the checkout flow")
        assert d.sub_mode == "create_from_prompt"

    def test_generate_mockup(self):
        d = _positive("generate a mockup of the dashboard widget")
        assert d.sub_mode == "create_from_prompt"

    def test_edit_modal_dialog(self):
        d = _positive("edit the modal dialog to have a green CTA")
        assert d.sub_mode == "create_from_prompt"

    def test_update_banner_design(self):
        d = _positive("update the banner design with our new colors")
        assert d.sub_mode == "create_from_prompt"

    def test_change_carousel_slide(self):
        d = _positive("change the carousel slide layout")
        assert d.sub_mode == "create_from_prompt"

    def test_modify_screen_design(self):
        d = _positive("modify the screen design to be dark themed")
        assert d.sub_mode == "create_from_prompt"

    def test_please_design_user_interface(self):
        d = _positive("please design a clean user interface")
        assert d.sub_mode == "create_from_prompt"

    def test_can_you_create_app_screen(self):
        d = _positive("can you create an app screen for the login flow")
        assert d.sub_mode == "create_from_prompt"

    def test_could_you_draft_web_section(self):
        d = _positive("could you draft a web section with a feature card")
        assert d.sub_mode == "create_from_prompt"

    def test_verb_without_artifact_returns_false(self):
        # Verb but no UI artifact keyword
        _negative("create a blog post about designing cards")

    def test_artifact_without_verb_returns_false(self):
        # Artifact keyword but no action verb at start
        _negative("what makes a good hero section?")

    def test_generate_editable_ui(self):
        d = _positive("generate an editable ui for the onboarding screen")
        assert d.sub_mode == "create_from_prompt"

    def test_build_popup_layout(self):
        d = _positive("build a popup layout with a dismiss button")
        assert d.sub_mode == "create_from_prompt"

    def test_vision_with_verb_plus_artifact(self):
        # Both verb+artifact AND image → vision mode
        d = _positive("design a hero section", has_image_attachment=True)
        assert d.sub_mode == "create_from_vision"


# ===========================================================================
# Fallthrough — no design pattern matched
# ===========================================================================

class TestFallthrough:
    def test_plain_question(self):
        _negative("what is the best font for a mobile app?")

    def test_general_chat(self):
        _negative("hello, how are you doing today?")

    def test_write_article(self):
        _negative("write an article about ui design principles")

    def test_explain_code(self):
        _negative("explain this code block to me")

    def test_summarize(self):
        _negative("summarize this document for me")

    def test_empty_query(self):
        _negative("")

    def test_whitespace_only(self):
        _negative("   ")
