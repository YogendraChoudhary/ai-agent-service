import pytest

from src.voice_flow.formatter.post_processor import PostProcessor
from src.voice_flow.schemas import AppContext, CursorContext

P = PostProcessor


def cur(before="", after=""):
    return CursorContext(preceding_text=before, following_text=after)


class TestSanitize:
    def test_space_before_punctuation(self):
        assert P.sanitize("Hello , world !") == "Hello, world!"

    def test_dotfiles_and_decimals_are_not_touched(self):
        assert P.sanitize("Edit the .env file and .gitignore") == "Edit the .env file and .gitignore"

    def test_collapses_double_spaces(self):
        assert P.sanitize("We  need   more   RAM") == "We need more RAM"

    @pytest.mark.parametrize("indent", [2, 3, 4, 8])
    def test_indentation_survives(self, indent):
        code = "def f():\n" + " " * indent + "return  1"
        assert P.sanitize(code) == "def f():\n" + " " * indent + "return 1"

    def test_blank_lines_inside_a_list_are_removed_including_the_first_gap(self):
        assert P.sanitize("1. A\n\n2. B\n\n3. C") == "1. A\n2. B\n3. C"

    def test_blank_line_before_first_item_is_removed(self):
        assert P.sanitize("We have two tasks:\n\n- A\n- B") == "We have two tasks:\n- A\n- B"

    def test_paragraph_break_after_a_list_is_kept(self):
        assert P.sanitize("- A\n- B\n\nThanks,\nAlex") == "- A\n- B\n\nThanks,\nAlex"

    def test_lead_in_gets_a_colon(self):
        assert P.sanitize("We have two tasks.\n1. update\n2. deploy") == "We have two tasks:\n1. Update\n2. Deploy"
        assert P.sanitize("We have two tasks\n- update\n- deploy") == "We have two tasks:\n- Update\n- Deploy"

    def test_question_lead_in_keeps_its_question_mark(self):
        assert P.sanitize("Can we cover these?\n- a\n- b").startswith("Can we cover these?\n")

    def test_greeting_comma_before_list_is_left_alone(self):
        assert P.sanitize("Hi team,\n\n- a\n- b") == "Hi team,\n\n- A\n- B"

    def test_items_are_not_capitalized_in_code_context(self):
        assert P.sanitize("Steps:\n1. npm install", AppContext.CODE_COMMENT) == "Steps:\n1. npm install"

    def test_model_artifacts_are_removed(self):
        assert P.sanitize("<formatted_text>Hi there</formatted_text>") == "Hi there"
        assert P.sanitize("```\nHi there\n```") == "Hi there"
        assert P.sanitize("Hi​ there") == "Hi there"

    def test_runs_of_blank_lines_become_one_paragraph_break(self):
        assert P.sanitize("A\n\n\n\nB") == "A\n\nB"

    def test_empty(self):
        assert P.sanitize("") == "" and P.sanitize("   \n ") == ""


class TestLeadingSpace:
    @pytest.mark.parametrize("before,text,expected", [
        ("", "Hello", False),
        ("We need to focus on ", "the API", False),          # caret already after a space
        ("Hello", "world", True),
        ("Hello", ", world", False),
        ("Hello", ".", False),
        ("Call (", "the API", False),
        ('He said "', "hello", False),                       # opening quote
        ('He said "hi"', "then left", True),                 # closing quote
        ("line one\n", "line two", False),
        ("Done.", "Next up", True),
    ])
    def test_cases(self, before, text, expected):
        assert P.stitch(text, cur(before)).leading_space is expected

    def test_insert_text_contains_the_space(self):
        s = P.stitch("world", cur("Hello"))
        assert s.insert_text == " world" and s.text == "world"


class TestCaretAwareText:
    def test_sentence_start_is_capitalized(self):
        assert P.stitch("next up", cur("Done. ")).text == "Next up"
        assert P.stitch("hello", cur("")).text == "Hello"
        assert P.stitch("hello", cur("Line one\n")).text == "Hello"

    def test_mid_sentence_is_not_capitalized_by_us(self):
        assert P.stitch("the API", cur("We need to focus on ")).text == "the API"

    def test_mixed_case_first_word_is_left_alone(self):
        assert P.stitch("iPhone support is next.", cur("")).text == "iPhone support is next."

    def test_code_context_does_not_capitalize(self):
        assert P.stitch("npm run build", cur(""), AppContext.CODE_COMMENT).text == "npm run build"

    def test_a_list_starts_on_its_own_line_when_the_caret_is_mid_line(self):
        s = P.stitch("1. The logs\n2. The metrics", cur("Please check "))
        assert s.insert_text == "\n1. The logs\n2. The metrics" and s.leading_space is False

    def test_a_list_after_a_line_break_gets_no_extra_newline(self):
        assert P.stitch("- a\n- b", cur("Tasks:\n")).insert_text == "- A\n- B"

    def test_trailing_space_before_a_word(self):
        s = P.stitch("Hello", cur("", "world"))
        assert s.trailing_space and s.insert_text == "Hello "

    def test_no_trailing_space_before_punctuation_or_space(self):
        assert not P.stitch("Hello", cur("", ", world")).trailing_space
        assert not P.stitch("Hello", cur("", " world")).trailing_space

    def test_closing_period_dropped_before_a_continuing_sentence(self):
        assert P.stitch("the logs.", cur("Please check ", " and the metrics")).text == "the logs"
        assert P.stitch("the logs.", cur("Please check ", ". Then")).text == "the logs"

    def test_closing_period_kept_before_a_new_sentence(self):
        assert P.stitch("the logs.", cur("Please check ", " Then do X")).text == "the logs."

    def test_empty_text(self):
        s = P.stitch("", cur("Hello"))
        assert (s.text, s.leading_space, s.trailing_space, s.insert_text) == ("", False, False, "")
