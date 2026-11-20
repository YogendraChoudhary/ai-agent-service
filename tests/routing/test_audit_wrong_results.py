"""
Wrong results found by the 2026-11-20 audit: requests that landed in the wrong mode or did the wrong thing. Each list is
the audit's own reproduction, plus the requests that must keep working the way they did.
"""
import time

import pytest

from src.design_intent import detect_design_intent
from src.document_action import decide_by_rules
from src.document_edit.in_place import in_place_request
from src.document_edit.service import edit_document
from src.intent import detect_image_intent
from src.request_plan import image_count_requested


# --- the image router -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "Create a slogan for my logo", "Create a React component that displays images", "Make a poem about a painting",
    "Generate interview questions for a graphic designer about logos", "Create a function to resize images",
    "Create a photo gallery page in HTML", "Make a CSS grid for my photos", "Generate a quiz about famous paintings",
    "Create a tagline for our new banner", "Create a contract for selling my paintings", "Make a recipe card with a photo",
    "Craft a speech about an iconic photograph", "Make a tutorial on drawing cats", "create a watercolor landing section",
])
def test_text_about_pictures_is_not_a_picture(query):
    assert detect_image_intent(query)["kind"] == "none"


@pytest.mark.parametrize("query", [
    "I'd like a photo of a dog on the beach", "Can I get a picture of a dragon?", "Give me a photo of a mountain lake",
    "Generate a pic of a sunset", "Create a cartoon of a dog", "Make a meme of a cat", "Photorealistic portrait of an old fisherman",
    "A watercolor of a fox in the snow", "Can you whip up a quick sketch of a lighthouse",
    "Crea una imagen de un gato con sombrero", "Dibuja un perro en la playa", "Erstelle ein Bild von einem Hund",
    "Dessine un chat sur un toit", "Crie uma imagem de um castelo", "Genera un'immagine di una montagna", "画一只猫", "猫の画像を生成して",
    "draw a mosque at dusk", "design a modern minimalist logo for a coffee shop",
])
def test_picture_requests_make_a_picture(query):
    assert detect_image_intent(query)["kind"] == "generate"


@pytest.mark.parametrize("query", ["More background on the history please", "Add more background about the company",
                                   "Make it more dramatic and add a plot twist", "More highlights from the match please",
                                   "Less contrast between the two arguments"])
def test_text_follow_ups_never_redraw_the_last_picture(query):
    assert detect_image_intent(query, has_previous_image=True)["kind"] == "none"


@pytest.mark.parametrize("query", ["make it night", "make it darker and rainier", "change the background to a beach", "more contrast"])
def test_picture_follow_ups_still_edit_the_last_picture(query):
    assert detect_image_intent(query, has_previous_image=True)["kind"] == "edit"


def test_a_huge_prompt_is_classified_fast_and_raw_query_is_whole():
    text = "Write a post for our newsletter with a short intro. The article should explain " * 1300
    started = time.perf_counter()
    result = detect_image_intent(text)
    assert time.perf_counter() - started < 1.0          # was 9.3 s at 8,000 characters and ~57 s at 100,000
    assert result["raw_query"] == text.strip()


# --- pictures granted inside a Write answer ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, count", [
    ("Make a list of tips for editing photos", 0), ("Explain how diffusion models create images", 0),
    ("Create a course outline about illustrations", 0), ("Write an essay about photography and why images matter", 0),
    ("Draft an email and add a note about the photos we took", 0), ("Create a checklist for a photo shoot", 0),
    ("how do I create a logo in Figma?", 0),
    ("an article with 3 images", 3), ("Write a blog post about cats with images", 2), ("Write an article and create 3 images", 3),
    ("write an article and 2 images", 2), ("can you create a logo for my bakery?", 1), ("Write a post with pics of the beach", 2),
])
def test_pictures_are_granted_only_when_asked_for(query, count):
    assert image_count_requested(query) == count


# --- the design gate --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "Draft an email to the team about the new UI", "Make a list of UX interview questions",
    "Create a job description for a senior UX designer", "Create a blog post about UI design trends for 2026",
    "Edit my cover letter for a UX role", "Change my resume to highlight UX skills", "Draft release notes for the new UI",
    "Write a landing page copy for my bakery", "Create a landing page in HTML and CSS", "Build a React login form",
])
def test_text_and_code_that_mention_ui_are_not_designs(query):
    assert not detect_design_intent(query).is_design_request


@pytest.mark.parametrize("query", ["Design a landing page for a coffee shop", "Create a website for my dental clinic",
                                   "Design a login screen", "Make a signup form", "Create a UI for a pricing page"])
def test_pages_and_screens_are_designs(query):
    assert detect_design_intent(query).sub_mode == "create_from_prompt"


@pytest.mark.parametrize("query", [
    "Add an FAQ about our JavaScript bootcamp", "Add a watercolor style to the header", "In French please", "Less clutter",
    "More modern", "Hmm, more playful", "Pizzas instead of burgers", "Same thing but for a sushi place", "Ok now for a coffee shop",
    "The price should read $12", "Could it be more minimal?", "Is it possible to make the button green?",
    "How about a darker background?",
])
def test_with_a_design_attached_short_edits_and_request_questions_edit_it(query):
    assert detect_design_intent(query, has_existing_design_ast=True).sub_mode == "edit_existing_ast"


@pytest.mark.parametrize("query", ["What font is this?", "thanks", "looks great, thank you!", "Can we meet at 4 PM tomorrow?",
                                   "write a python function to validate the form", "Write a blog post about dogs",
                                   "a photorealistic portrait of a cat for the background",
                                   "As a customer of three years, I was disappointed that"])
def test_with_a_design_attached_questions_and_other_tasks_stay_chat(query):
    assert not detect_design_intent(query, has_existing_design_ast=True).is_design_request


# --- Auto keeps the document for tasks about it -----------------------------------------------------------------------

STUDY = "# Onboarding study\n\n## Findings\n\nUsers drop off at screen 3 (42%).\n\n## Notes\n\nMost users skip the tour.\n"


@pytest.mark.parametrize("query", [
    "Make a summary of the UX findings", "Convert these notes into UX recommendations", "Write a blog post about these findings",
    "Write a blog post about the findings above", "Generate a chart of drop-off by screen", "Create a chart from this data",
    "Make a chart of these numbers", "Write a Python script that tracks the drop-off described here",
])
def test_auto_does_not_drop_the_document_for_tasks_about_it(query):
    decision = decide_by_rules(query, STUDY)
    assert decision is None or decision.action != "write"


@pytest.mark.parametrize("query", ["Write a blog post about remote work for small teams",
                                   "Make a bar chart of quarterly revenue: Q1 10, Q2 14, Q3 9, Q4 17",
                                   "Generate an image of a cat on a sofa"])
def test_auto_still_sends_standalone_tasks_to_write(query):
    assert decide_by_rules(query, STUDY).action == "write"


# --- Edit -------------------------------------------------------------------------------------------------------------

class NoModel:
    def __call__(self, *args):
        raise AssertionError("a literal edit must not call the model")


@pytest.mark.parametrize("query, result", [
    ("Change Zoom to Teams please", "Meetings use Teams.\n"), ("Change Zoom to Teams.", "Meetings use Teams.\n"),
    ("Change Zoom to Teams, thanks", "Meetings use Teams.\n"), ("Change Zoom to Teams too", "Meetings use Teams.\n"),
])
def test_literal_replace_never_writes_the_users_extra_words(query, result):
    text, proposal = edit_document("Meetings use Zoom.\n", query, NoModel())
    assert proposal.status == "proposed" and text == result


def test_abbreviations_keep_their_period():
    assert in_place_request("Change Acme to Acme Inc.").replacement == "Acme Inc."


def test_replace_everywhere_changes_every_editable_occurrence_and_no_code():
    source = "Meetings use Zoom. Zoom calls start at 9.\n\nWe pay for Zoom yearly.\n\n```\nZoom.start()\n```\n"
    text, proposal = edit_document(source, "Replace Zoom with Teams everywhere", NoModel())
    assert proposal.status == "proposed" and len(proposal.patches) == 2
    assert text.count("Teams") == 3 and "Zoom.start()" in text and "1 in protected content" in proposal.message


class PlanRecorder:
    def __init__(self):
        self.instructions = []

    def __call__(self, system, messages, spec):
        import json
        self.instructions.append(json.loads(messages[0]["content"])["USER_INSTRUCTION"])
        return json.dumps({"status": "no_change", "message": "Nothing to fix.", "targets": []})


@pytest.mark.parametrize("query", ["Fix the grammar", "Proofread", "Please proofread", "Fix typos", "Can you fix the grammar?",
                                   "Fix the punctuation", "Fix capitalization", "Polish it", "Improve it", "make it shorter"])
def test_bare_whole_document_requests_reach_the_planner(query):
    planner = PlanRecorder()
    _, proposal = edit_document("# Doc\n\nOur team are happy to help.\n\nIt work well.\n", query, planner)
    assert planner.instructions and proposal.message != "Which text should I change? Select one passage or quote the exact phrase to edit."


def test_a_heading_name_means_its_section():
    planner = PlanRecorder()
    edit_document("# Doc\n\n## Pricing\n\nThe price are 20.\n", "Fix Pricing", planner)
    assert planner.instructions == ["Fix the Pricing section: correct its mistakes and unclear wording."]


def test_a_plan_of_more_than_12_paragraphs_is_proposed_whole():
    import json
    paragraphs = [f"Paragraph number {i} is very very long." for i in range(20)]
    source = "# Doc\n\n" + "\n\n".join(paragraphs) + "\n"

    class Model:
        def __call__(self, system, messages, spec):
            data = json.loads(messages[0]["content"])
            if spec.stage == "document_plan":
                ids = data["allowed_block_ids"]
                return json.dumps({"status": "ready", "message": "Tighten every paragraph.", "targets": [
                    {"block_id": b, "request_quote": "Make every paragraph shorter", "reason": "Wordy."} for b in ids]})
            return data["TARGET"].replace("very very ", "")
    text, proposal = edit_document(source, "Make every paragraph shorter", Model())
    assert proposal.status == "proposed" and len(proposal.patches) == 20 and "very very" not in text


# --- second round (2026-11-20): Write check, attached images, code and blog routers -------------------------------------

from src.blog.intent import detect_blog_intent
from src.code_intent import detect_code_intent
from src.document_intent import write_needs_document


@pytest.mark.parametrize("query", [
    "Add photos to the blog", "Add some images", "Add some relevant pictures throughout", "Fix the grammar", "Fix all the typos",
    "Proofread", "Correct the spelling mistakes", "Translate to French", "Translate everything into Spanish",
    "Summarize the findings above", "Turn this into a blog post", "Write a blog post based on this", "Make a chart of these numbers",
    "Add a conclusion", "Add headings", "Keep going", "Write the ending", "Tone it down", "Make the opening punchier", "TL;DR",
    "Resume este texto", "Traduce esto al inglés", "Résume ce texte", "Fasse den Text zusammen",
])
def test_write_stops_prompts_about_an_unsent_document(query):
    assert write_needs_document(query) is not None


@pytest.mark.parametrize("query", [
    "Fix the grammar: their going home", "Translate to French: Hello, how are you?", "Add some images of cats",
    "Write the introduction for an article about solar power", "Can you fix this code: x = 1",
])
def test_write_answers_prompts_that_carry_their_own_material(query):
    assert write_needs_document(query) is None


@pytest.mark.parametrize("query", ["Rebuild this page", "Copy this layout", "Same layout for my gym",
                                   "Use this as a template for my bakery", "Make this editable"])
def test_a_screenshot_to_copy_is_an_editable_design(query):
    assert detect_design_intent(query, has_image_attachment=True).sub_mode == "create_from_vision"


@pytest.mark.parametrize("query", ["Turn this whiteboard photo into meeting notes", "Make a recipe from this picture",
                                   "Fix the typos in this screenshot", "Make a poem inspired by this image"])
def test_reading_an_attached_picture_into_text_is_not_an_image_edit(query):
    assert detect_image_intent(query, has_reference=True)["kind"] == "none"


@pytest.mark.parametrize("query", ["make the sky purple", "remove the background", "turn this into a watercolor", "add a hat to the cat"])
def test_real_picture_edits_still_edit_the_attached_picture(query):
    assert detect_image_intent(query, has_reference=True)["kind"] == "edit"


@pytest.mark.parametrize("query", ["Create a code name for our project", "Generate a code word for our team retreat",
                                   "Make a function plan for my wedding reception", "Create a function invitation for our company gala",
                                   "Write a code of conduct for staff"])
def test_code_words_that_are_not_programming_are_not_sent_straight_to_the_code_writer(query):
    assert detect_code_intent(query)["kind"] != "code"


@pytest.mark.parametrize("query", ["Write a function to reverse a string", "Create code for a login page", "Write Python code to read a CSV"])
def test_programming_requests_are_still_code(query):
    assert detect_code_intent(query)["kind"] == "code"


@pytest.mark.parametrize("query, kind", [
    ("Write an SEO blog post about running shoes", "blog"), ("Write a blog post for my bakery about sourdough", "blog"),
    ("Write a blog article about hiking in Norway", "blog"), ("Write an informative blog post about electric cars", "blog"),
    ("Write a blog post titled Ten Ways to Save", "blog"), ("Write a blog post explaining how SQL joins work", "blog"),
    ("My blog post about X needs a fix", "other"),
])
def test_blog_router(query, kind):
    assert detect_blog_intent(query)["kind"] == kind
