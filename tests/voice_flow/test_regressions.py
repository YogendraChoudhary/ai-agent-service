"""
Regression matrix for the review (docs/VOICE_FLOW_CHECKLIST.md). Each test names the issue number it locks in.
Rows that belong to the email / Slack / code / Markdown formats are covered where the code is kept, but those formats are off.
"""

import asyncio
import json
import logging
import threading

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from config import Settings, settings
from src.voice_flow import router as router_module
from src.voice_flow.asr import vocabulary
from src.voice_flow.asr.audio import resample_audio, resample_linear
from src.voice_flow.asr.whisper_client import AsrResult
from src.voice_flow.evals import run_evals as ev
from src.voice_flow.formatter import gate, patterns, post_processor
from src.voice_flow.formatter.claims import check_claims, key_terms
from src.voice_flow.formatter.engine import (
    Breaker, LlmError, OpenAIBackend, VoiceFormattingService, _max_output_tokens, check_output)
from src.voice_flow.formatter.post_processor import PostProcessor
from src.voice_flow.formatter.rules import detect_list, rule_format
from src.voice_flow.schemas import (
    AppContext, ContextUpdateFrame, CursorContext, FormatRequest, StartFrame, effective_context)
from tests.voice_flow.test_engine import FakeBackend

RATE = 16000


def cur(before="", after=""):
    return CursorContext(preceding_text=before, following_text=after)


# ---------------------------------------------------------------------------------------------------------------------
# Phase 1: shared registry and rules
# ---------------------------------------------------------------------------------------------------------------------
class TestRegistry:
    def test_calendar_completeness_4_4(self):
        assert rule_format("meeting on march 15 and august 3") == "Meeting on March 15 and August 3."
        for name in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
                     "april", "june", "july", "august", "september", "october", "november", "december"):
            assert patterns.DAYS_AND_MONTHS[name] == name.capitalize()
        assert len(patterns.DAYS_AND_MONTHS) == 18

    def test_may_is_a_month_only_where_a_month_is_meant(self):
        assert rule_format("we may ship in may") == "We may ship in May."
        assert rule_format("the review is may 15") == "The review is May 15."
        assert rule_format("it may rain") == "It may rain."

    def test_days_and_months_are_not_acronyms_4_1(self):
        assert "monday" not in patterns.TECH_ACRONYMS and "monday" in patterns.DAYS_AND_MONTHS

    def test_acronyms_and_products_beyond_the_old_list_4_1(self):
        out = rule_format("deploy the sdk to gke with grpc and graphql behind a cdn on s3")
        assert out == "Deploy the SDK to GKE with gRPC and GraphQL behind a CDN on S3."
        assert rule_format("the apis and the sdks") == "The APIs and the SDKs."

    def test_words_that_are_also_english_are_not_forced_to_capitals(self):
        for word in ("bun", "linear", "remix"):
            assert word not in patterns.PROPER_NOUNS
        assert rule_format("a linear regression and a bun") == "A linear regression and a bun."

    def test_the_users_dictionary_beats_the_registry(self):
        assert rule_format("send the api docs", dictionary=["api"]) == "Send the api docs."

    @pytest.mark.parametrize("text", ["when does the pod restart", "where is the env file", "do you know the answer",
                                      "have you seen the logs", "will you join us", "how do i reset it", "isn't it done"])
    def test_question_starters_4_3(self, text):
        assert rule_format(text).endswith("?"), text

    @pytest.mark.parametrize("text", ["do the dishes before dinner", "have a good day", "will send it tomorrow",
                                      "when we deployed it broke", "may the best team win"])
    def test_imperatives_and_statements_are_not_questions(self, text):
        assert rule_format(text).endswith("."), text

    def test_the_review_list_had_a_capitalised_wont(self):
        assert all(w == w.lower() for w in patterns.QUESTION_STARTERS)

    def test_greeting_and_signoff_patterns_are_shared_6_1_6_2(self):
        assert gate.GREETING_DETECT_RE is patterns.GREETING_DETECT_RE and gate.SIGN_OFF_CLOSING_RE is patterns.SIGN_OFF_CLOSING_RE
        assert post_processor.GREETING_SPLIT_RE is patterns.GREETING_SPLIT_RE and post_processor.SIGN_OFF_RE is patterns.SIGN_OFF_RE
        assert gate.SPOKEN_COMMANDS_RE is patterns.SPOKEN_COMMANDS_RE
        for greeting in ("Good morning team, let's begin", "Hello there, how are you", "Dear Dr. Okafor, thank you", "Greetings everyone, welcome"):
            assert patterns.GREETING_DETECT_RE.search(greeting), greeting

    def test_lowercase_signoff_name_is_title_cased_6_1(self):
        assert PostProcessor.stitch("The build is green. thanks, alex.", None, AppContext.EMAIL).text == "The build is green.\n\nThanks,\nAlex"
        assert PostProcessor.stitch("Done. Regards, McDonald", None, AppContext.EMAIL).text.endswith("Regards,\nMcDonald")
        assert PostProcessor.stitch("that is the best idea, really", None, AppContext.EMAIL).text == "That is the best idea, really."

    def test_greeting_split_in_email_only_and_capitalises_what_follows(self):
        assert PostProcessor.stitch("Good morning team, let's begin", None, AppContext.EMAIL).text == "Good morning team,\n\nLet's begin"
        assert PostProcessor.stitch("Good morning team, let's begin", None, AppContext.GENERAL).text == "Good morning team, let's begin."


class TestListNumbers:
    def test_homophone_for_is_not_a_number_4_2(self):
        assert not detect_list("point 1 latency point for improvement")
        assert rule_format("point 1 latency point for improvement") == "Point 1 latency point for improvement."

    def test_homophones_count_only_as_the_next_number(self):
        assert detect_list("point 1 the logs point to the metrics")
        assert detect_list("point 1 a point to b point 3 c point for d")           # 'for' is 4 only after 3
        assert not detect_list("point for the first one and point to the second")  # nothing starts at 1

    def test_strict_and_homophone_numbers_are_kept_apart(self):
        from src.voice_flow.formatter import rules
        assert set(rules.HOMOPHONE_NUMBERS) == {"to", "too", "for"} and not set(rules.STRICT_NUMBERS) & set(rules.HOMOPHONE_NUMBERS)

    def test_step_and_item_markers(self):
        assert rule_format("step 1 open the file step 2 save it") == "1. Open the file\n2. Save it"
        assert rule_format("item one milk item two eggs") == "1. Milk\n2. Eggs"

    def test_point_number_one_is_one_cue_not_two_words(self):
        """The whole phrase is the cue; 'point' used to be left behind: 'Point\\n2. Is, we also...'."""
        out = rule_format("the plan is point number one we ship it point number two is we test it")
        assert out == "The plan is:\n1. We ship it\n2. We test it"

    def test_the_copula_after_a_cue_and_a_dangling_article_are_dropped(self):
        out = rule_format("so now the point number 1 would be to build it point number 2 is we also handle html")
        assert out.split("\n")[1:] == ["1. To build it", "2. We also handle HTML"] and not out.split("\n")[0].endswith("the:")


class TestStutter:
    @pytest.mark.parametrize("text,expected", [
        ("Yes, yes, that works", "Yes, yes, that works."),
        ("very very fast", "Very very fast."),
        ("bye bye now", "Bye bye now."),
        ("the the dog barked", "The dog barked."),
        ("i i think so", "I think so."),
        ("we need to we need to ship", "We need to ship."),
    ])
    def test_only_plain_stutters_are_collapsed_4_5(self, text, expected):
        assert rule_format(text) == expected


class TestVocabularyLanguages:
    def test_seed_is_english_only_8_1(self):
        assert vocabulary.get_structural_seed("en") == vocabulary.STRUCTURAL_SEED == vocabulary.get_structural_seed(None)
        assert vocabulary.get_structural_seed("en-GB") == vocabulary.STRUCTURAL_SEED
        assert vocabulary.get_structural_seed("es") == "" and vocabulary.get_structural_seed("hi") == ""
        assert vocabulary.build_initial_prompt(["Kubernetes"], "es") == "Kubernetes."
        assert vocabulary.build_initial_prompt([], "de") == ""
        assert vocabulary.build_initial_prompt(["Kubernetes"], "en").startswith(vocabulary.STRUCTURAL_SEED)

    def test_hallucinations_per_language_8_2(self):
        assert vocabulary.is_hallucination("Thank you.", "en") and vocabulary.is_hallucination("Gracias.", "es")
        assert vocabulary.is_hallucination("धन्यवाद", "hi") and not vocabulary.is_hallucination("gracias", "en")
        assert vocabulary.is_hallucination("thanks for watching", "es")                 # English is always checked as well
        assert not vocabulary.is_hallucination("Thank you for the update on the build", "en")


# ---------------------------------------------------------------------------------------------------------------------
# Phase 2: schemas
# ---------------------------------------------------------------------------------------------------------------------
class TestSchemas:
    def test_cursor_window_is_trimmed_from_the_far_end_9_2(self):
        c = CursorContext(preceding_text="a" * 100 + "b" * 500, following_text="c" * 400)
        assert c.preceding_text == "b" * 150 and c.following_text == "c" * 150
        with pytest.raises(ValidationError):
            CursorContext(preceding_text=123)            # a wrong type is an error, not silently ""

    def test_cursor_limits_are_the_documented_ones(self):
        info = CursorContext.model_fields
        assert "150" in info["preceding_text"].description and "150" in info["following_text"].description
        assert info["preceding_text"].metadata[0].max_length == 150

    def test_app_context_is_general_while_formats_are_off(self):
        assert effective_context("email") is AppContext.GENERAL and effective_context(AppContext.CODE_COMMENT) is AppContext.GENERAL
        assert FormatRequest(raw_text="hi", app_context="slack_chat").app_context is AppContext.GENERAL
        assert StartFrame(app_context="markdown_docs").app_context is AppContext.GENERAL

    def test_app_context_is_kept_when_formats_are_on(self, monkeypatch):
        monkeypatch.setattr(settings, "voice_app_contexts_enabled", True)
        assert effective_context("email") is AppContext.EMAIL
        assert FormatRequest(raw_text="hi", app_context="slack_chat").app_context is AppContext.SLACK

    def test_every_context_is_documented(self):
        assert AppContext.__doc__ and "SWITCHED OFF" in AppContext.__doc__

    def test_context_update_frame_carries_only_what_changed_7_2(self):
        u = ContextUpdateFrame.model_validate({"type": "context", "cursor_context": {"preceding_text": "Hello "}})
        assert u.custom_dictionary is None and u.language is None and u.app_context is None and u.style is None
        assert u.cursor_context.preceding_text == "Hello "
        assert ContextUpdateFrame.model_validate({"type": "context", "custom_dictionary": ["A", "a", " B "]}).custom_dictionary == ["A", "B"]
        with pytest.raises(ValidationError):
            ContextUpdateFrame.model_validate({"type": "context", "language": "english-us!"})

    def test_style_and_language_are_validated(self):
        assert StartFrame(style="polished", language="en-US").style == "polished"
        with pytest.raises(ValidationError):
            StartFrame(style="fancy")
        assert FormatRequest(raw_text="hi").style is None


# ---------------------------------------------------------------------------------------------------------------------
# Phase 3 + 4: guards, engine, post-processing
# ---------------------------------------------------------------------------------------------------------------------
class TestGuards:
    def test_homophone_chain_passes_5_2(self):
        assert check_output("their going to there house", "They're going to their house.") is None
        raw = "their going to there house because its to late and they're not ready for the move"
        assert check_output(raw, "They're going to their house because it's too late and they're not ready for the move.") is None

    def test_chatbot_answer_is_blocked_5_2(self):
        assert check_output("what is docker", "Sure! Docker is a container platform that packages software.") == "answered_instead_of_formatting"
        assert check_output("what is docker", "I can't help with that.") == "answered_instead_of_formatting"
        assert check_output("sure that works for me", "Sure, that works for me.") is None       # dictated, not an answer

    def test_a_long_answer_with_no_chatbot_words_is_still_blocked(self):
        raw = "can you tell me how to reset the database"
        out = "To reset the database, stop the service, run the migration tool with the reset flag, and restart everything."
        assert check_output(raw, out) == "answered_instead_of_formatting"

    def test_short_outputs_are_never_called_answers(self):
        assert check_output("their going there", "They're going there.") is None

    def test_token_floor_5_3(self):
        assert _max_output_tokens("hi") == 512 and _max_output_tokens("word " * 5) >= 512
        assert _max_output_tokens("word " * 2000) == 4096 and _max_output_tokens("word " * 400) == 1500

    def test_breaker_logs_name_the_backend_5_1(self, caplog):
        caplog.set_level(logging.WARNING)
        Breaker("openai").failure(LlmError("openai 402", permanent=True))
        assert "openai backend disabled" in caplog.text and "Gemini" not in caplog.text
        service = VoiceFormattingService(backend=FakeBackend("x"), mode="always")
        assert service.breaker.backend_name == "fake"


class TestOpenAICapabilityCache:
    """5.4: model names are not stable, so 'does this model take temperature' is learned from the API and remembered."""

    @staticmethod
    def error_400(message):
        import httpx
        import openai
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return openai.APIStatusError("x", response=httpx.Response(400, request=request), body={"error": {"message": message}})

    def backend(self, model, *outcomes):
        b = OpenAIBackend(api_key="k", model=model)
        calls = []

        async def create(**kw):
            calls.append(kw)
            outcome = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        b._client = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
        return b, calls

    @staticmethod
    def ok():
        from types import SimpleNamespace
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Hi."))])

    async def test_an_unknown_model_that_rejects_temperature_is_learned_once(self, monkeypatch):
        monkeypatch.setattr(OpenAIBackend, "_no_temperature_models", set())
        b, calls = self.backend("zeta-9", self.error_400("Unsupported parameter: 'temperature' is not supported with this model."), self.ok())
        assert await b.generate("s", [{"role": "user", "content": "x"}], max_output_tokens=10, timeout=2) == "Hi."
        assert "temperature" in calls[0] and "temperature" not in calls[1]
        assert "zeta-9" in OpenAIBackend._no_temperature_models
        other, other_calls = self.backend("zeta-9", self.ok())                      # a new instance: no second 400
        await other.generate("s", [{"role": "user", "content": "x"}], max_output_tokens=10, timeout=2)
        assert "temperature" not in other_calls[0]

    async def test_known_prefixes_never_send_temperature(self):
        b, calls = self.backend("gpt-5.9-nano", self.ok())
        await b.generate("s", [{"role": "user", "content": "x"}], max_output_tokens=10, timeout=2)
        assert "temperature" not in calls[0]

    async def test_an_unrelated_400_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(OpenAIBackend, "_no_temperature_models", set())
        b, calls = self.backend("zeta-9", self.error_400("bad request"))
        with pytest.raises(LlmError):
            await b.generate("s", [{"role": "user", "content": "x"}], max_output_tokens=10, timeout=2)
        assert len(calls) == 1


class TestSentenceClosing:
    def test_never_closes_lists_headings_or_code_6_3(self):
        for text in ("- Review the Q3 metrics", "1. Update the schema now", "# Release notes today", "`pip install fastapi` first"):
            assert PostProcessor.stitch(text).text == text, text

    def test_mid_sentence_before_existing_text_gets_no_period_2_2(self):
        s = PostProcessor.stitch("the Redis connection pool exhausted its max limit.", cur("The cause is ", " which failed."))
        assert s.text == "the Redis connection pool exhausted its max limit" and s.leading_space is False

    def test_a_new_sentence_after_the_caret_still_gets_its_period(self):
        assert PostProcessor.stitch("the logs.", cur("Please check ", " Then do X")).text == "the logs."


class TestMidSentenceThroughTheWholePipeline:
    async def test_the_model_path_starts_lower_case_and_has_no_trailing_period_2_2(self):
        s = VoiceFormattingService(backend=FakeBackend("The Redis connection pool exhausted its max limit."), mode="always", timeout=1)
        cursor = cur("The cause is ", " which failed.")
        r = await s.compile("um the redis connection pool exhausted its max limit", AppContext.GENERAL, cursor, ["Redis"])
        stitched = PostProcessor.stitch(r.formatted_text, cursor)
        assert r.source == "llm" and stitched.insert_text == "the Redis connection pool exhausted its max limit"

    async def test_a_new_sentence_keeps_its_capital(self):
        s = VoiceFormattingService(backend=FakeBackend("The build is green."), mode="always", timeout=1)
        r = await s.compile("the build is green", AppContext.GENERAL, cur("Done. "))
        assert PostProcessor.stitch(r.formatted_text, cur("Done. ")).text == "The build is green."

    async def test_the_rule_path_continues_the_sentence_too(self):
        cursor = cur("The cause is ", " which failed.")
        text = PostProcessor.stitch(rule_format("The Redis pool exhausted its max limit.", cursor, ["Redis"]), cursor).text
        assert text == "the Redis pool exhausted its max limit"


# ---------------------------------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------------------------------
class TestGateCommands:
    @pytest.mark.parametrize("text", ["Run it with a colon after the name.", "Open bracket the value close bracket.", "Type forward slash api.",
                                      "Add a semicolon here.", "Open quote hello close quote."])
    def test_spoken_symbols_go_to_the_model_3_3(self, text):
        assert gate.needs_model(text, AppContext.GENERAL) == "spoken command"

    def test_heading_is_no_longer_a_command_while_markdown_is_off(self):
        assert gate.needs_model("The heading of the page is blue.", AppContext.GENERAL) is None

    def test_fast_word_limit_is_documented_3_2(self):
        assert gate.MAX_FAST_WORDS == 45 and "30 s window" in gate.__doc__ + open(gate.__file__).read()


# ---------------------------------------------------------------------------------------------------------------------
# Claims and Polished style
# ---------------------------------------------------------------------------------------------------------------------
RAW_DICTATION = ("So the point is, uh, now the point number 1 would be to build the AI inside it. Point number 2 is we handle the HTML "
                 "rendering. And that's why the Editor is called a plug-and-play experience. And point number 5 is we are 3 times lower "
                 "in pricing. In the USP area it takes 10 minutes to get running. These are the points for the editor SDKs.")


class TestClaims:
    def test_key_terms_of_a_real_dictation(self):
        terms = key_terms(RAW_DICTATION)
        assert {"AI", "HTML", "USP", "SDKs", "plug-and-play", "3", "10"} <= set(terms)
        assert "5" not in terms and "1" not in terms and "2" not in terms           # list-cue numbers are not claims

    def test_retracted_numbers_are_not_claims(self):
        assert "5" not in key_terms("Set the timeout to 5 seconds. Wait no, 10 seconds.") and "10" in key_terms("Set the timeout to 5 seconds. Wait no, 10 seconds.")

    def test_a_dropped_hyphenated_term_is_caught(self):
        assert check_claims(RAW_DICTATION, "AI built in. HTML handled. 3 times lower. USP: 10 minutes. SDKs.") == "plug-and-play"

    def test_spelling_a_number_out_is_not_dropping_it(self):
        out = "AI. HTML. plug and play. three times lower. USP. ten minutes. SDKs. Editor."
        assert check_claims(RAW_DICTATION, out) is None

    def test_nothing_missing_passes(self):
        assert check_claims("the build is green", "The build is green.") is None
        assert check_claims("we use the APIs", "We use the API.") is None            # plural tolerance


class TestPolishedFlow:
    RAW = "point number 1 is we ship the editor point number 2 is it is plug-and-play and takes 10 minutes"

    def svc(self, *replies):
        class Seq(FakeBackend):
            async def generate(self, system, turns, **kw):
                self.calls.append((system, turns, kw["max_output_tokens"], kw["timeout"]))
                return replies[min(len(self.calls), len(replies)) - 1]
        backend = Seq("")
        return VoiceFormattingService(backend=backend, mode="always", timeout=2, breaker=Breaker()), backend

    async def test_polished_is_used_when_it_keeps_every_claim(self):
        s, backend = self.svc("1. Editor: We ship the editor.\n2. Setup: It is plug-and-play and takes 10 minutes.")
        r = await s.compile(self.RAW, style="polished")
        assert r.source == "llm" and r.fallback_reason is None and len(backend.calls) == 1
        assert "Polished style" in backend.calls[0][0]

    async def test_a_refused_polished_answer_falls_back_to_clean_not_to_the_rules(self):
        """Gemini opened a polished answer with 'Here are...' (a new introduction); that is refused, and the clean style answers."""
        raw = "we ship the editor and it is plug-and-play and takes 10 minutes"
        s, backend = self.svc("Here are the points we covered: ship the editor, plug-and-play, 10 minutes.",
                              "We ship the editor and it is plug-and-play and takes 10 minutes.")
        r = await s.compile(raw, style="polished")
        assert r.source == "llm" and len(backend.calls) == 2 and "refused" in r.fallback_reason and "answered" in r.fallback_reason
        assert r.formatted_text.startswith("We ship the editor")

    async def test_an_invented_lead_in_above_a_list_is_removed_not_refused(self):
        s, backend = self.svc("Here are the points we covered:\n1. Editor: We ship the editor.\n2. Setup: It is plug-and-play and takes 10 minutes.")
        r = await s.compile(self.RAW, style="polished")
        assert r.source == "llm" and len(backend.calls) == 1 and r.formatted_text.startswith("1. Editor:")

    async def test_clean_failing_too_ends_in_the_rules(self):
        s, backend = self.svc("Sure! Here is the formatted text: ok", "Sure! Here is the formatted text: ok")
        r = await s.compile("we ship the editor and it is plug-and-play", style="polished")      # no list cues: one call per attempt
        assert r.source == "rules" and "answered" in r.fallback_reason and len(backend.calls) == 2

    async def test_a_wrong_item_count_is_asked_again_once_with_the_exact_correction(self):
        s, backend = self.svc("1. We ship the editor and it is plug-and-play and takes 10 minutes.",
                              "1. We ship the editor.\n2. It is plug-and-play and takes 10 minutes.")
        r = await s.compile(self.RAW, style="clean")
        assert r.source == "llm" and len(backend.calls) == 2
        retry_turns = backend.calls[1][1]
        assert retry_turns[-2]["role"] == "assistant" and "has 1 items but the speaker counted 2" in retry_turns[-1]["content"]
        assert "<list_items>2</list_items>" in backend.calls[0][1][-1]["content"]       # the count was in the request from the start

    async def test_still_wrong_after_the_retry_ends_in_the_rules_which_get_it_right(self):
        s, backend = self.svc("We ship the editor and it is plug-and-play.", "We ship the editor and it is plug-and-play.")
        r = await s.compile(self.RAW, style="clean")
        assert r.source == "rules" and len(backend.calls) == 2 and r.formatted_text.count("\n") == 1

    async def test_a_dropped_claim_gets_one_repair_that_names_it(self):
        s, backend = self.svc("1. Editor: We ship the editor.\n2. Setup: It is easy and quick.",
                              "1. Editor: We ship the editor.\n2. Setup: It is plug-and-play and takes 10 minutes.")
        r = await s.compile(self.RAW, style="polished")
        assert r.source == "llm" and len(backend.calls) == 2 and "repaired" in r.fallback_reason
        assert all("Polished style" in c[0] for c in backend.calls)                       # still the polished style
        note = backend.calls[1][1][-1]["content"]
        assert "dropped 'plug-and-play'" in note and "'10'" in note and backend.calls[1][1][-2]["role"] == "assistant"

    async def test_a_claim_still_missing_after_the_repair_falls_back_to_clean(self):
        s, backend = self.svc("1. Editor: We ship the editor.\n2. Setup: It is easy.", "1. Editor: We ship the editor.\n2. Setup: Quick.",
                              "1. We ship the editor.\n2. It is plug-and-play and takes 10 minutes.")
        r = await s.compile(self.RAW, style="polished")
        assert len(backend.calls) == 3 and "Polished style" in backend.calls[1][0] and "Polished style" not in backend.calls[2][0]
        assert r.source == "llm" and "plug-and-play" in r.fallback_reason and "used clean" in r.fallback_reason
        assert "plug-and-play" in r.formatted_text

    async def test_clean_style_only_checks_acronyms_and_hyphenated_terms(self):
        """Numbers and names are often retracted ("make that Alice", "5 seconds wait no 10"), so Clean never asks to keep them."""
        s, backend = self.svc("1. We ship the editor.\n2. It is easy.")
        r = await s.compile("point 1 we ship the editor point 2 it takes 10 minutes for Bob", style="clean")
        assert len(backend.calls) == 1 and r.fallback_reason is None

    async def test_clean_style_repairs_a_dropped_hyphenated_term(self):
        s, backend = self.svc("1. We ship the editor.\n2. It is easy to start.", "1. We ship the editor.\n2. It is plug-and-play.")
        r = await s.compile("point 1 we ship the editor point 2 it is plug-and-play", style="clean")
        assert len(backend.calls) == 2 and "repaired" in r.fallback_reason and "plug-and-play" in r.formatted_text
        assert "Polished style" not in backend.calls[1][0] and "dropped 'plug-and-play'" in backend.calls[1][1][-1]["content"]

    async def test_a_repair_that_makes_it_worse_keeps_the_first_answer(self):
        s, backend = self.svc("1. We ship the editor.\n2. It is easy to start.", "Sure! Here is the formatted text: ok")
        r = await s.compile("point 1 we ship the editor point 2 it is plug-and-play", style="clean")
        assert r.source == "llm" and "kept despite dropping" in r.fallback_reason and r.formatted_text.startswith("1. We ship")

    async def test_the_server_default_style_applies(self, monkeypatch):
        monkeypatch.setattr(settings, "voice_default_style", "polished")
        s, backend = self.svc("1. Editor: We ship the editor.\n2. Setup: It is plug-and-play and takes 10 minutes.")
        await s.compile(self.RAW)
        assert "Polished style" in backend.calls[0][0]

    def test_default_style_setting_is_validated(self):
        assert Settings(VOICE_DEFAULT_STYLE="polished").voice_default_style == "polished"
        assert Settings(VOICE_DEFAULT_STYLE="fancy").voice_default_style == "clean"
        assert Settings().voice_app_contexts_enabled is False


class TestLongFormChecks:
    GOOD = ("Hey, hi there. Good morning. On this call, we are going to demonstrate a few of the tricks we have learned while building the "
            "editor. The points are:\n1. Build the AI inside it so people can write right within the editor.\n"
            "2. We handle the HTML rendering inside the editor.\n3. Server maintenance is handled by the editor itself, which is why it is a "
            "plug-and-play experience.\n4. We are three times lower in pricing, and the USP is 10 minutes to get running.\n\n"
            "So these are the points that make the editor different.")

    def test_a_good_answer_passes_every_check(self):
        case = next(c for c in ev.ALL_CASES if c["id"] == "lf-01") if hasattr(ev, "ALL_CASES") else None
        cases = json.loads(ev.CASES_PATH.read_text())["cases"]
        case = next(c for c in cases if c["id"] == "lf-01")
        assert ev.check_long(self.GOOD, case["checks"]) == []

    def test_the_reported_defects_are_caught(self):
        cases = json.loads(ev.CASES_PATH.read_text())["cases"]
        checks = next(c for c in cases if c["id"] == "lf-01")["checks"]
        stacked = self.GOOD.replace("The points are:", "So the point is, now the points are:")
        assert any("should not contain" in f for f in ev.check_long(stacked, checks))
        five_items = self.GOOD.replace("\n\nSo these", "\n5. And that is why it is plug-and-play.\n\nSo these")
        assert any("5 list items" in f for f in ev.check_long(five_items, checks))
        assert any("plug" in f for f in ev.check_long(self.GOOD.replace("plug-and-play", "easy"), checks))
        inside = self.GOOD.replace("\n\nSo these are the points that make the editor different.", "")
        assert any("closing" in f for f in ev.check_long(inside, checks))


# ---------------------------------------------------------------------------------------------------------------------
# Phase 5: router and audio
# ---------------------------------------------------------------------------------------------------------------------
class FakeTranscriber:
    model_name, compute_type, _model = "fake-whisper", "float32", object()

    def __init__(self, text="we have two tasks mark it point 1 update the schema and point to deploy the service"):
        self.text, self.calls, self.languages = text, [], []
        self.release = threading.Event()
        self.block = False

    async def transcribe(self, audio, custom_vocab=(), language=None):
        self.calls.append((audio.size, tuple(custom_vocab)))
        self.languages.append(language)
        while self.block and not self.release.is_set():
            await asyncio.sleep(0.01)
        return AsrResult(self.text if audio.size else "", audio.size / RATE * 1000, 12.0, "en")


def pcm(seconds=1.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * 0.1 * 32767).astype("<i2").tobytes()


START = {"type": "start", "sample_rate": RATE, "mode": "push_to_talk"}


@pytest.fixture
def ws_setup(monkeypatch):
    backend = FakeBackend("We have two tasks:\n1. Update the schema\n2. Deploy the service")
    service = VoiceFormattingService(backend=backend, timeout=1.0, breaker=Breaker(), mode="always")
    transcriber = FakeTranscriber()
    monkeypatch.setattr(router_module, "_service", service)
    monkeypatch.setattr(router_module, "get_transcriber", lambda: transcriber)
    monkeypatch.setattr(settings, "voice_flow_token", "")
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app), backend, transcriber


class TestPartialContextUpdates:
    def test_a_cursor_only_update_keeps_the_dictionary_7_2(self, ws_setup):
        client, backend, transcriber = ws_setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "custom_dictionary": ["Redis", "Kubernetes"], "language": "en", "style": "polished"})
            ws.receive_json()
            ws.send_json({"type": "context", "cursor_context": {"preceding_text": "Please check ", "following_text": " tomorrow"}})
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            assert ws.receive_json()["type"] == "final"
        assert transcriber.calls == [(RATE // 2, ("Redis", "Kubernetes"))] and transcriber.languages == ["en"]
        system, turns = backend.calls[0][0], backend.calls[0][1]
        assert "Polished style" in system                                   # the style set at start survived the update
        assert "<preceding_text>Please check </preceding_text>" in turns[-1]["content"] and "<dictionary>Redis, Kubernetes</dictionary>" in turns[-1]["content"]

    def test_a_dictionary_update_keeps_the_cursor(self, ws_setup):
        client, backend, _ = ws_setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "cursor_context": {"preceding_text": "Hello "}})
            ws.receive_json()
            ws.send_json({"type": "context", "custom_dictionary": ["Pydantic"], "language": "es"})
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            ws.receive_json()
        content = backend.calls[0][1][-1]["content"]
        assert "<preceding_text>Hello </preceding_text>" in content and "<dictionary>Pydantic</dictionary>" in content

    def test_an_invalid_update_changes_nothing_and_is_reported(self, ws_setup):
        client, _, transcriber = ws_setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json({**START, "custom_dictionary": ["Redis"]})
            ws.receive_json()
            ws.send_json({"type": "context", "language": "english-us!"})
            assert ws.receive_json()["code"] == "bad_context"
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            ws.receive_json()
        assert transcriber.calls[0][1] == ("Redis",) and transcriber.languages == [None]


class TestEmptyUtterancesAndBackpressure:
    def test_an_end_with_no_audio_never_reaches_whisper_7_3(self, ws_setup):
        client, _, transcriber = ws_setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_json({"type": "end"})
            final = ws.receive_json()
            ws.send_bytes(b"\x00\x01" * 100)                                  # 6 ms: a key bounce
            ws.send_json({"type": "end"})
            bounce = ws.receive_json()
        assert final["raw_text"] == "" and final["fallback_reason"] == "no speech detected" and bounce["raw_text"] == ""
        assert transcriber.calls == []

    def test_empty_results_keep_their_place_in_line(self, ws_setup):
        client, _, transcriber = ws_setup
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_bytes(pcm(0.5))
            ws.send_json({"type": "end"})
            ws.send_json({"type": "end"})                                     # nothing left: answered after the first, not before
            first, second = ws.receive_json(), ws.receive_json()
        assert first["raw_text"] != "" and second["raw_text"] == ""

    def test_the_utterance_queue_is_bounded_7_4(self, ws_setup):
        client, _, transcriber = ws_setup
        transcriber.block = True                                              # the speech model is stuck on the first utterance
        with client.websocket_connect("/v1/voice/stream") as ws:
            ws.send_json(START)
            ws.receive_json()
            for _ in range(router_module.MAX_QUEUED_UTTERANCES + 2):         # 1 in the worker + 8 waiting = 9 accepted, the 10th refused
                ws.send_bytes(pcm(0.2))
                ws.send_json({"type": "end"})
            refused = ws.receive_json()
            assert refused["type"] == "error" and refused["code"] == "rate_limited"
            transcriber.release.set()
            finals = [ws.receive_json() for _ in range(router_module.MAX_QUEUED_UTTERANCES + 1)]
        assert all(f["type"] == "final" for f in finals)

    def test_limits_are_what_the_review_asked_for(self):
        assert router_module.MAX_QUEUED_UTTERANCES == 8 and router_module.MIN_UTTERANCE_SECONDS == 0.05


class TestBearerToken:
    def test_constant_time_and_utf8_safe_7_1(self):
        v = router_module._verify_bearer_token
        assert v("", "") and v("anything", "")                                 # no token configured: open
        assert v("s3cret", "s3cret") and not v("s3cret!", "s3cret") and not v("", "s3cret")
        assert v("pässwörd-ключ", "pässwörd-ключ") and not v("pässwörd", "pässwörd-ключ")
        assert not hasattr(router_module, "_token_ok") and not hasattr(router_module, "hmac")


class TestResampling:
    @staticmethod
    def level_at(y, hz, rate=RATE):
        spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) / (y.size / 4)
        return spec[np.argmin(abs(np.fft.rfftfreq(y.size, 1 / rate) - hz))]

    def test_frequencies_above_the_new_nyquist_are_removed_not_folded_8_3(self):
        sr = 48000
        t = np.arange(sr) / sr
        loud = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)        # 12 kHz folds to 4 kHz at 16 kHz
        assert self.level_at(resample_linear(loud, sr), 4000) > 0.2            # the old method: a loud ghost at 4 kHz
        assert self.level_at(resample_audio(loud, sr), 4000) < 0.01

    def test_speech_band_content_is_kept(self):
        sr = 44100
        t = np.arange(sr) / sr
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        out = resample_audio(tone, sr)
        assert abs(out.size - RATE) < 50 and self.level_at(out, 1000) == pytest.approx(0.5, abs=0.03)

    def test_16k_and_empty_pass_through(self):
        x = np.ones(100, dtype=np.float32)
        assert resample_audio(x, RATE) is x and resample_audio(np.zeros(0, dtype=np.float32), 48000).size == 0

    def test_a_filter_failure_falls_back_to_linear(self, monkeypatch):
        import av
        monkeypatch.setattr(av, "AudioResampler", lambda **kw: (_ for _ in ()).throw(RuntimeError("no swr")))
        out = resample_audio(np.ones(4800, dtype=np.float32) * 0.1, 48000)
        assert out.size == 1600


class TestOpeningGreeting:
    """A clean answer to the spoken dictation dropped "Hey. Hi there. Good morning." in 1 run of 8."""

    def test_the_check(self):
        from src.voice_flow.formatter.engine import _keeps_opening_greeting as keeps
        assert keeps("Hey. Hi there. Good morning. So on this call we begin.", "Hey. Hi there. Good morning. On this call we begin.")
        assert keeps("Good morning team, let's begin.", "Good morning team, let's begin.")
        assert not keeps("Hey. Hi there. Good morning. So on this call we begin.", "On this call we begin.")
        assert keeps("The build is green.", "The build is green.")            # not a greeting: nothing to protect

    async def test_a_dropped_greeting_is_repaired_once(self):
        raw = "Hey. Hi there. Good morning. We ship the editor and it is plug-and-play."
        class Seq(FakeBackend):
            async def generate(self, system, turns, **kw):
                self.calls.append((system, turns, kw["max_output_tokens"], kw["timeout"]))
                return ["We ship the editor and it is plug-and-play.", "Hey. Hi there. Good morning. We ship the editor and it is plug-and-play."][len(self.calls) - 1]
        backend = Seq("")
        r = await VoiceFormattingService(backend=backend, mode="always", timeout=2, breaker=Breaker()).compile(raw, style="clean")
        assert r.formatted_text.startswith("Hey.") and "repaired" in r.fallback_reason and "opening greeting" in r.fallback_reason
        assert "opening greeting" in backend.calls[1][1][-1]["content"]
