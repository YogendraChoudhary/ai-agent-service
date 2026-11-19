from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config import settings
from src import text_processing_agent as writer
from src.length_spec import count_words, measure, parse_length, size_word, strip_length, to_int, trim_to
from tests.blog.fakes import filler


@pytest.mark.parametrize(("query", "expected"), [
    ("Write an article on Weather Forcast in 100 Words.", ("words", 100, 90, 110, "about")),
    ("Write an article on weather in 50 words", ("words", 50, 45, 55, "about")),
    ("Write a paragraph about dogs in only 100 words", ("words", 100, 90, 110, "about")),
    ("Write a 1,500-word blog post about remote work", ("words", 1500, 1350, 1650, "about")),
    ("Write a blog post about SEO under 150 words", ("words", 150, 90, 150, "at_most")),
    ("Write about the sea in no more than 80 words", ("words", 80, 48, 80, "at_most")),
    ("Write an essay of at least 300 words on climate", ("words", 300, 300, 900, "at_least")),
    ("Write a poem about rain, 40 words.", ("words", 40, 35, 45, "about")),
    ("Summarise this in 3 sentences", ("sentences", 3, 3, 3, "exact")),
    ("Write two paragraphs about Rome", ("paragraphs", 2, 2, 2, "exact")),
    ("Give me 5 bullet points on safety", ("bullets", 5, 5, 5, "exact")),
    ("Give me at most 5 bullets", ("bullets", 5, 1, 5, "at_most")),
    ("Write a 280-character tweet about our launch", ("characters", 280, 168, 280, "at_most")),
    ("Write about cats in one hundred words", ("words", 100, 90, 110, "about")),
    ("Write about cats in a hundred and fifty words", ("words", 150, 135, 165, "about")),
    ("Write between 80 and 120 words about tea", ("words", 100, 80, 120, "range")),
    ("Write about tea in 100-150 words", ("words", 125, 100, 150, "range")),
    ("Write a note. 180–220 words, subject first.", ("words", 200, 180, 220, "range")),
    ("Write a product description, word limit of 60", ("words", 60, 36, 60, "at_most")),
    ("Write exactly 50 words about the moon", ("words", 50, 48, 52, "exact")),
    ("Write an article on weather in about 1.5k words", ("words", 1500, 1350, 1650, "about")),
    ("Write about moss in 100 words max.", ("words", 100, 60, 100, "at_most")),
    ("Draft a synopsis in ~100 words", ("words", 100, 90, 110, "about")),
    ("List 10 words that rhyme with cat", None),
    ("Give me 5 words for happy", None),
    ("Write about the 100 words every child should know", None),
    ("Write an article on 100-word stories", None),
    ("A list of 20 words related to winter", None),
    ("Write a list of 20 words.", None),
    ("Translate these 3 sentences into French.", None),
    ("Explain this with two words", None),
    ("Write a short article on weather", None),
])
def test_length_phrases(query, expected):
    spec = parse_length(query)
    actual = (spec.unit, spec.target, spec.low, spec.high, spec.kind) if spec else None
    assert actual == expected


@pytest.mark.parametrize(("text", "number"), [("one hundred and fifty", 150), ("1.5k", 1500), ("2,000", 2000)])
def test_numbers(text, number):
    assert to_int(text) == number


@pytest.mark.parametrize(("query", "topic"), [
    ("Write an article about rain in 100 words. Include an example.", "Write an article about rain. Include an example."),
    ("Write a 100-word post on weather", "Write a post on weather"),
    ("Write two paragraphs about Rome", "Write two paragraphs about Rome"),
    ("Write a 2000 word article about the 1990s", "Write an article about the 1990s"),
    ("Write an 800-word blog post on tea", "Write a blog post on tea"),
    ("A 100-word intro for my site", "An intro for my site"),
])
def test_strip_length_preserves_the_rest_of_the_request(query, topic):
    assert strip_length(query, parse_length(query)) == topic


@pytest.mark.parametrize(("query", "size"), [
    ("Write a short article about the long run", "short"),
    ("Please draft a concise blog post about memory", "short"),
    ("A lengthy guide about running", "long"),
    ("Write an article on short-term memory", None),
    ("Write a blog post about long-distance running", None),
    ("Write an article about being resilient in the long run", None),
    ("Write an essay about pitching a short guide", None),
])
def test_size_is_an_adjective_in_the_request(query, size):
    assert size_word(query) == size


def test_editor_word_count_includes_structure_and_excludes_attached_captions():
    # Verified against the Eddyter live demo's Markdown import.
    markdown = ("# Heading alone\n\n*Subtitle text*\n\nwell-known state-of-the-art can’t X—Y A/B 1,500 -- ...\n\n"
                "| Table head | Second |\n| --- | --- |\n| cell text | extra |\n\n> Quoted text\n\n"
                "- [Linked words](https://example.com/)\n")
    assert count_words(markdown) == 22
    for separator in ("\n", "\n\n"):
        figure = f"![Alt text has five words](placeholder::image-id){separator}*Caption has four words*"
        assert count_words(f"Seed word\n\n{figure}\n\nFinal word") == 4


@pytest.mark.parametrize(("query", "text", "count"), [
    ("In 10 characters", "[Tea](https://example.com) and me.", 11),
    ("In 2 paragraphs", "# Title\n\nFirst one.\n\nSecond one.\n\n- Item", 2),
    ("In 3 sentences", "Dr. Green arrived. She asked a question. A final fragment", 3),
    ("In 2 lines", "One\n\nTwo\n", 2),
    ("Give me 3 bullets", "- first\n  - nested\n- last\n\n```text\n- code\n```", 3),
])
def test_measure_in_the_requested_unit(query, text, count):
    assert measure(parse_length(query), text) == count


def test_trim_cuts_whole_trailing_sections_when_the_last_paragraph_is_not_enough():
    spec = parse_length("under 30 words")
    text = ("# Plan\n\n" + filler(14) + "\n\n## Packing\n\n" + filler(14, 3) + "\n\n## Pack-down\n\n" + filler(10, 5) + "\n")
    trimmed = trim_to(spec, text)
    assert count_words(trimmed) <= 30 and count_words(trimmed) >= spec.low
    assert not trimmed.rstrip().endswith("Pack-down")          # no heading left with nothing under it
    assert trimmed.rstrip()[-1] in ".!?"
    # Never below the request's own floor: then the text is kept and the caller warns.
    assert trim_to(parse_length("under 10 words"), text) == text


def test_trim_keeps_only_complete_sentences():
    spec = parse_length("under 12 words")
    text = "A first short line. A second short line. Here is one more line. The end."
    assert trim_to(spec, text) == "A first short line. A second short line.\n"
    assert trim_to(spec, "one " * 13 + "sentence.") == "one " * 13 + "sentence."
    with_figure = text + "\n\n![A photo](placeholder::keep-me)\n*A caption*"
    assert trim_to(spec, with_figure) == with_figure


@pytest.fixture
def repair(monkeypatch):
    monkeypatch.setattr(settings, "length_repair_enabled", True)
    run = Mock()
    factory = Mock(return_value=SimpleNamespace(run=run))
    monkeypatch.setattr(writer, "Agent", factory)
    return factory, run


@pytest.mark.parametrize(("query", "first", "fixed"), [
    ("Write under 100 words", 105, 85),
    ("Write at least 300 words", 290, 350),
    ("Write in 100 words", 150, 100),
])
def test_repair_accepts_closer_to_the_window_and_uses_only_text(repair, query, first, fixed):
    factory, run = repair
    run.return_value = SimpleNamespace(content=filler(fixed))
    original = SimpleNamespace(model=object(), run=Mock(side_effect=AssertionError("original agent called")))
    chunks = list(writer._enforce_length(parse_length(query), filler(first), original, "test", None))
    assert chunks == [{"__content_replace__": filler(fixed) + "\n"}]
    factory.assert_called_once_with(model=original.model, telemetry=False)
    assert run.call_args.kwargs == {"stream": False, "images": None}
    assert filler(first) in run.call_args.args[0]


def test_no_repair_when_in_window_or_disabled(repair, monkeypatch):
    factory, _ = repair
    spec = parse_length("Write under 100 words")
    assert list(writer._enforce_length(spec, filler(85), object(), "test", None)) == []
    monkeypatch.setattr(settings, "length_repair_enabled", False)
    (warning,) = list(writer._enforce_length(parse_length("in 100 words"), filler(150), object(), "test", None))
    assert warning["__warnings__"][0]["kind"] == "length"
    # The sentence trim is local and free, so an "at most" is still cut when rewrites are off.
    (replacement,) = list(writer._enforce_length(spec, filler(105), object(), "test", None))
    assert count_words(replacement["__content_replace__"]) == 98
    factory.assert_not_called()


def test_repair_prompt_carries_the_task_and_direction(repair):
    _, run = repair
    run.return_value = SimpleNamespace(content=filler(100))
    spec = parse_length("Write about tea in 100 words")
    list(writer._enforce_length(spec, filler(150), SimpleNamespace(model=object()), "test", None, task="Write about tea"))
    prompt = run.call_args.args[0]
    assert "TASK" in prompt and "Write about tea" in prompt and "cutting words" in prompt
    list(writer._enforce_length(spec, filler(50), SimpleNamespace(model=object()), "test", None, task="Write about tea"))
    assert "Do not invent" in run.call_args.args[0]


def test_draft_is_never_lengthened_without_its_document(repair):
    factory, run = repair
    run.return_value = SimpleNamespace(content=filler(100))
    spec = parse_length("Draft a note in 100 words")
    (warning,) = list(writer._enforce_length(spec, filler(30), SimpleNamespace(model=object()), "test", None,
                                            allow_lengthen=False))
    assert warning["__warnings__"][0]["kind"] == "length"
    factory.assert_not_called()
    # Shortening needs no new facts, so Draft still gets it.
    (replacement,) = list(writer._enforce_length(spec, filler(150), SimpleNamespace(model=object()), "test", None,
                                                allow_lengthen=False))
    assert replacement == {"__content_replace__": filler(100) + "\n"}


def test_models_that_write_short_are_told_more_but_judged_the_same(repair):
    from src.length_spec import aim_factor
    spec = parse_length("in 300 words")
    assert "270-330 words (aim for 300)" in spec.instruction()
    assert "297-363 words (aim for 330)" in spec.instruction(aim_factor("gemini-3.5-flash-lite"))
    assert spec.instruction(aim_factor("gpt-6.1-terra")) == spec.instruction()
    # Ceilings and counts are never moved.
    assert parse_length("under 100 words").instruction(1.1) == parse_length("under 100 words").instruction()
    assert parse_length("in 3 sentences").instruction(1.1) == parse_length("in 3 sentences").instruction()
    _, run = repair
    run.return_value = SimpleNamespace(content=filler(300))
    (replacement,) = list(writer._enforce_length(spec, filler(250), SimpleNamespace(model=object()),
                                                "gemini-3.5-flash-lite", None))
    assert "aim for 330" in run.call_args.args[0]
    assert replacement == {"__content_replace__": filler(300) + "\n"}


def test_draft_lengthens_only_with_its_source(repair):
    _, run = repair
    run.return_value = SimpleNamespace(content=filler(100))
    spec = parse_length("Draft a note in 100 words")
    list(writer._enforce_length(spec, filler(30), SimpleNamespace(model=object()), "test", None, allow_lengthen=False,
                                source="DOC TEXT\n\nUSER_TASK:\nDraft a note in 100 words"))
    assert "DOC TEXT" in run.call_args.args[0] and "finish it" in run.call_args.args[0]
    list(writer._enforce_length(spec, filler(150), SimpleNamespace(model=object()), "test", None, allow_lengthen=False,
                                source="DOC TEXT"))
    assert "DOC TEXT" not in run.call_args.args[0]       # shortening does not re-send the document


def test_long_texts_are_warned_not_rewritten(repair, monkeypatch):
    factory, _ = repair
    monkeypatch.setattr(settings, "length_repair_max_words", 1000)
    (warning,) = list(writer._enforce_length(parse_length("in 1450 words"), filler(1700),
                                            SimpleNamespace(model=object()), "test", None))
    assert warning["__warnings__"][0]["kind"] == "length"
    (warning,) = list(writer._enforce_length(parse_length("in 900 words"), filler(1200),
                                            SimpleNamespace(model=object()), "test", None))
    assert warning["__warnings__"][0]["kind"] == "length"
    factory.assert_not_called()


def test_write_rewrite_passes_the_output_guard(repair):
    _, run = repair
    run.return_value = SimpleNamespace(content="Here is the rewritten text:\n\n" + filler(100))
    (replacement,) = list(writer._enforce_length(parse_length("in 100 words"), filler(150), SimpleNamespace(model=object()),
                                                "test", None, guard_query="Write about tea in 100 words"))
    assert replacement["__content_replace__"].startswith(filler(100)[:20])


def test_refuses_repair_that_drops_an_image(repair):
    _, run = repair
    run.return_value = SimpleNamespace(content=filler(100))
    original = filler(150) + "\n\n![Alt](placeholder::keep-me)"
    (warning,) = list(writer._enforce_length(parse_length("in 100 words"), original,
                                            SimpleNamespace(model=object()), "test", None))
    assert warning["__warnings__"][0]["kind"] == "length"


def test_failed_repair_preserves_the_draft_and_tries_sentence_trim(repair):
    _, run = repair
    run.side_effect = TimeoutError()
    (replacement,) = list(writer._enforce_length(parse_length("under 100 words"), filler(105),
                                                SimpleNamespace(model=object()), "test", None))
    assert count_words(replacement["__content_replace__"]) == 98
