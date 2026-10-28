import pytest

from src.voice_flow.formatter.rules import apply_dictionary, rule_format
from src.voice_flow.schemas import CursorContext


def cur(before):
    return CursorContext(preceding_text=before)


TWO_TASKS = "We have two pending tasks:\n1. Update the schema\n2. Deploy the service"


class TestWhisperQuirks:
    """Strings Whisper actually produced for 'mark it point 1 ... point 2' (see the benchmark notes in whisper_client.py)."""

    @pytest.mark.parametrize("heard", [
        "We have two pending tasks, Market.1 update the schema and .2 deploy the service.",                    # base, no prompt
        "We have two pending tasks. Market Point 1 update the schema and Point 2 deploy the service.",        # small, no prompt
        "We have two pending tasks, Mark it point 1, update the schema and point 2, deploy the service.",     # with the prompt
        "we have two pending tasks mark it point 1 update the schema and point to deploy the service",        # 'point to' = 2
        "we have two pending tasks mark it point one update the schema and point two deploy the service",
    ])
    def test_numbered_list(self, heard):
        assert rule_format(heard) == TWO_TASKS


class TestFalseTriggers:
    @pytest.mark.parametrize("text,expected", [
        ("at this point 1 developer is enough", "At this point 1 developer is enough."),
        ("The main point is that we need more RAM.", "The main point is that we need more RAM."),
        ("it lasted for a period of two weeks", "It lasted for a period of two weeks."),
        ("the number 1 priority is uptime", "The number 1 priority is uptime."),
        ("I'd point to the docs and point to the logs", "I'd point to the docs and point to the logs."),
    ])
    def test_stays_prose(self, text, expected):
        assert rule_format(text) == expected


class TestBullets:
    def test_make_it_bullets(self):
        assert rule_format("make it bullets first review q3 metrics second finalize the hiring plan third book the room") == (
            "- Review Q3 metrics\n- Finalize the hiring plan\n- Book the room")

    def test_lead_in_and_bullets(self):
        out = rule_format("I want to cover two things make it bullets first review q3 metrics second finalize the hiring plan")
        assert out == "I want to cover two things:\n- Review Q3 metrics\n- Finalize the hiring plan"

    def test_ordinals_without_a_bullet_cue_are_prose(self):
        assert "\n" not in rule_format("first we ship then we celebrate and second we rest")


class TestSelfCorrection:
    @pytest.mark.parametrize("heard,expected", [
        ("Set the timeout to 5 seconds. Wait no, 10 seconds.", "Set the timeout to 10 seconds."),
        ("set the timeout to 5 seconds wait no 10 seconds", "Set the timeout to 10 seconds."),
        ("send it at 3 pm wait no 4 pm", "Send it at 4 PM."),
        ("can we meet at 3pm? Wait, actually 4pm tomorrow?", "Can we meet at 4 PM tomorrow?"),
        # Not a number on both sides: left as spoken for the model to resolve, never guessed at.
        ("deploy on Tuesday actually make that 3 days later", "Deploy on Tuesday actually make that 3 days later."),
    ])
    def test_numeric(self, heard, expected):
        assert rule_format(heard) == expected


class TestCleaning:
    def test_fillers_and_stutters(self):
        assert rule_format("um we need to we need to ship the the build uh today") == "We need to ship the build today."

    def test_question_without_punctuation(self):
        assert rule_format("how do I center a div in css") == "How do I center a div in CSS?"

    def test_imperative_starting_with_do_is_not_a_question(self):
        assert rule_format("do the dishes before dinner") == "Do the dishes before dinner."

    def test_times_and_pronoun(self):
        assert rule_format("i think i'll go at 9 am") == "I think I'll go at 9 AM."

    def test_empty(self):
        assert rule_format("") == "" and rule_format("   ") == ""


class TestCaret:
    def test_continues_a_sentence_in_lower_case(self):
        out = rule_format("That the Redis connection pool exhausted its max limit", cur("The root cause of the outage is "))
        assert out == "that the Redis connection pool exhausted its max limit."

    def test_new_sentence_keeps_the_capital(self):
        assert rule_format("That works", cur("Sounds good. ")).startswith("That")

    def test_proper_noun_is_not_lowered(self):
        assert rule_format("Sarah will follow up", cur("Thanks, ")).startswith("Sarah")


def test_dictionary_respells_whole_words_only():
    assert apply_dictionary("we run kubernetes and kubernetesque", ["Kubernetes"]) == "we run Kubernetes and kubernetesque"
    assert apply_dictionary("ask sarah okafor", ["Sarah Okafor"]) == "ask Sarah Okafor"
    assert apply_dictionary("a\\1 b", ["a\\1"]) == "a\\1 b"               # backslashes in a term are literal


def test_dictionary_applies_inside_rule_format():
    assert rule_format("deploy to kubernetes", dictionary=["Kubernetes"]) == "Deploy to Kubernetes."
