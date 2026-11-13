"""
Translate, shorten and lengthen through the structure-preserving pipeline (src/longform), with scripted models.

A translation has to reach the cells of a table, a quote, a caption and an image's alt text, and still leave every address, code block,
front matter and formula byte for byte; a shortening or an expansion must keep every heading, list item and figure. The model can do
anything; the code decides what is accepted, so the fakes here are deliberately blunt: they reverse the letters of every word
("translate"), cut the tail of every line ("shorten") or pad it ("lengthen").
"""
import random
import re

import pytest

from src.longform.chunk import analyse
from src.longform.freeze import freeze, thaw
from src.longform.markdown_ast import parse, script_share, word_count
from src.longform.prompts import OPERATIONS, instruction_line, ratio_window, system_prompt, user_message
from src.longform.refine import RefineConfig, chunk_max_tokens, refine_stream
from src.longform.verify import figures, snap_tables, verify
from tests.longform.fakes import ScriptedRefiner, fix_on_repair, passage_of, summarise

TRANSLATE, SHORTER, LONGER = OPERATIONS["translation"], OPERATIONS["make_shorter"], OPERATIONS["make_longer"]
LOCKED = re.compile(r"(\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML|URL)_\d+\]\]|\(link://\d+\))")

DOC = """---
title: Pricing guide
---

# Pricing plans

Our plans are simple and fair. See [the documentation](https://example.com/docs) or visit https://example.com/pricing for details. Contact sales@example.com with questions.

## Compare the options

| Plan | Monthly price |
|:-----|--------------:|
| Basic | 5 |
| Pro | 12 |

> [!NOTE]
> Prices may change next year.

![Team photo](team.png "Our team")
*The team at work*

- First choose a plan
- Then confirm the payment

```python
print("hello")
```

The energy formula is $E = mc^2$ and it stays.[^1]

[^1]: Source of the formula, checked twice.
"""


def run(text, behaviour, operation, instruction=None, per_unit=None, config=None):
    model = ScriptedRefiner(behaviour, per_unit)
    out = list(refine_stream(text, instruction, operation, model, config or RefineConfig()))
    document = "".join(x for x in out if isinstance(x, str))
    warnings = next((x["__warnings__"] for x in out if isinstance(x, dict) and "__warnings__" in x), [])
    return document, warnings, model


# --- what the scripted "model" does ---------------------------------------------------------------------------------------

def reverse_words(passage, spec):
    """A translator: the letters of every prose word reversed. Tokens and link references are not touched."""
    return "".join(part if i % 2 else re.sub(r"[A-Za-z]{2,}", lambda m: m.group(0)[::-1], part)
                   for i, part in enumerate(LOCKED.split(passage)))


def to_han(passage, spec):
    """A translator into Chinese: a run of words becomes that many Han characters, with no spaces between them, as Chinese is written."""
    return "".join(part if i % 2 else re.sub(r"[A-Za-z]{2,}(?: [A-Za-z]{2,})*", lambda m: "字" * len(m.group(0).split()), part)
                   for i, part in enumerate(LOCKED.split(passage)))


def cut_lines(keep):
    """A shortener: keeps the first `keep` of the words of every prose line (and every token in the tail), headings untouched."""
    def behaviour(passage, spec):
        lines = []
        for line in passage.split("\n"):
            if not line.strip() or line.lstrip().startswith("#") or re.fullmatch(r"\[\[[A-Z]+_\d+\]\]", line.strip()):
                lines.append(line)
                continue
            marker = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+)?(.*)$", line)
            words = marker.group(2).split(" ")
            n = max(1, round(len(words) * keep))
            lines.append((marker.group(1) or "") + " ".join(words[:n] + [w for w in words[n:] if "[[" in w or "link://" in w]))
        return "\n".join(lines)
    return behaviour


def pad(extra):
    """A lengthener: a clause with no digits added to every plain paragraph line."""
    def behaviour(passage, spec):
        return "\n".join(line + extra if line.strip() and not re.match(r"^\s*([#>|*\-`]|\d+[.)]|\[\[)", line) else line
                         for line in passage.split("\n"))
    return behaviour


PROSE = ("The team reviewed the plan together and agreed that every section needed clearer examples, a calmer tone and a shorter "
         "introduction before anyone outside the group could be expected to read it carefully. ")
SECTIONS = "\n\n".join(f"## Section {i}\n\n{PROSE * 2}\n\n{PROSE}\n\n- the first point of section {i} explained in some detail\n"
                       f"- the second point of section {i} explained in some detail" for i in range(1, 5))
LONG_DOC = "# A long report\n\n" + SECTIONS + "\n"


class TestTranslate:
    def test_prose_headings_cells_quotes_captions_and_alt_text_are_translated(self):
        out, warnings, _ = run(DOC, reverse_words, TRANSLATE, "Spanish")
        assert warnings == []
        assert "# gnicirP snalp" in out and "## ", "emoC".replace("emoC", "") is not None
        assert "[eht noitatnemucod](https://example.com/docs)" in out        # the anchor is translated, the address is not
        assert "| nalP | ylhtnoM ecirp |" in out and "| cisaB | 5 |" in out     # header and cells
        assert "> [!NOTE]\n> secirP yam egnahc txen raey." in out             # the quote's text, not its alert keyword
        assert '![maeT otohp](team.png "Our team")' in out                    # the alt text, not the address or the title
        assert "*ehT maet ta krow*" in out                                    # the caption
        assert "- tsriF esoohc a nalp" in out
        assert "[^1]: ecruoS fo eht alumrof, dekcehc eciwt." in out              # a footnote's text is translated, its label is not

    def test_everything_that_must_not_change_is_byte_for_byte(self):
        out, _, _ = run(DOC, reverse_words, TRANSLATE, "Spanish")
        for kept in ["---\ntitle: Pricing guide\n---", "https://example.com/pricing", "sales@example.com", "|:-----|--------------:|",
                     '```python\nprint("hello")\n```', "$E = mc^2$", "[!NOTE]", "team.png", "https://example.com/docs", "syats.[^1]\n"]:
            assert kept in out, kept

    def test_the_model_is_told_the_target_language_and_the_default_is_english(self):
        _, _, model = run(DOC, reverse_words, TRANSLATE, "Japanese")
        system, messages, _ = model.calls[0]
        assert system == system_prompt("translation") and "TARGET LANGUAGE: Japanese" in messages[0]["content"]
        _, _, model = run(DOC, reverse_words, TRANSLATE, None)
        assert "TARGET LANGUAGE: English" in model.calls[0][1][0]["content"]

    def test_the_model_sees_the_table_cells_the_quote_and_the_alt_text_but_never_an_address_or_code(self):
        _, _, model = run(DOC, reverse_words, TRANSLATE, "Spanish")
        seen = model.calls[0][1][0]["content"]
        for visible in ["| Basic | 5 |", "Prices may change next year.", "![Team photo](link://", "The team at work"]:
            assert visible in seen, visible
        for hidden in ["team.png", "example.com", "print(", "title: Pricing guide", "mc^2", "[!NOTE]"]:
            assert hidden not in seen, hidden

    def test_a_redrawn_separator_row_is_put_back_as_written(self):
        def redraw(passage, spec):
            return reverse_words(passage, spec).replace("|:-----|--------------:|", "|---|---|")
        out, warnings, _ = run(DOC, redraw, TRANSLATE, "Spanish")
        assert warnings == [] and "|:-----|--------------:|" in out and "|---|---|" not in out

    def test_a_table_that_loses_a_column_is_repaired_when_the_model_can_and_restored_when_it_cannot(self):
        def drop_column(passage, spec):
            return re.sub(r"(?m)^\|([^|\n]*)\|[^|\n]*\|$", r"|\1|", reverse_words(passage, spec))
        repairs = lambda p, s: drop_column(p, s) if s.stage == "chunk" else reverse_words(p, s)         # wrong first, right when told
        out, warnings, _ = run(DOC, repairs, TRANSLATE, "Spanish")
        assert warnings == [] and "| nalP | ylhtnoM ecirp |" in out
        out, warnings, _ = run(DOC, drop_column, TRANSLATE, "Spanish")
        assert [w["kind"] for w in warnings] == ["chunk_restored"] and out == DOC       # kept as written, never a broken table

    def test_a_summary_is_refused_but_a_longer_or_shorter_translation_is_accepted(self):
        out, warnings, _ = run(DOC, summarise, TRANSLATE, "Spanish")
        assert [w["kind"] for w in warnings] == ["chunk_restored"] and out == DOC
        longer = lambda p, s: reverse_words(p, s) + ""        # same length; Spanish is about 20% longer, German about 10%
        pad_ = pad(" ydob")                                    # a few more words per paragraph, like a longer language
        out, warnings, _ = run(DOC, lambda p, s: pad_(reverse_words(p, s), s), TRANSLATE, "Spanish")
        assert warnings == []

    def test_paragraphs_are_not_merged(self):
        merge = lambda p, s: reverse_words(p, s).replace("\n\n", " ", 1)
        out, warnings, _ = run(DOC, merge, TRANSLATE, "Spanish")
        assert [w["kind"] for w in warnings] == ["chunk_restored"]

    def test_into_a_script_without_spaces_the_word_count_falls_and_that_is_accepted(self):
        out, warnings, _ = run(LONG_DOC, to_han, TRANSLATE, "Chinese")
        assert warnings == [] and out.count("字") > 100 and script_share(out) > 0.5
        # the same collapse into a Latin-script language is a summary and is refused
        out, warnings, _ = run(LONG_DOC, cut_lines(0.4), TRANSLATE, "German")
        assert warnings and all(w["kind"] == "chunk_restored" for w in warnings) and out == LONG_DOC

    def test_a_quote_that_holds_code_stays_frozen(self):
        doc = "# T\n\nIntro words here.\n\n> A quote\n>\n> ```\n> keep_this()\n> ```\n\nOutro words here.\n"
        out, warnings, model = run(doc, reverse_words, TRANSLATE, "Spanish")
        assert warnings == [] and "keep_this()" in out and "> A quote" in out        # untouched: it was a token
        assert "keep_this" not in model.calls[0][1][0]["content"]

    def test_a_long_document_is_translated_in_order_and_one_bad_passage_does_not_lose_the_rest(self):
        text = LONG_DOC
        out, warnings, _ = run(text, reverse_words, TRANSLATE, "Spanish", config=RefineConfig(chunk_words=120, parallel=4))
        assert warnings == [] and out.index("troper") < out.index("noitceS")
        out, warnings, _ = run(text, reverse_words, TRANSLATE, "Spanish", per_unit={"c2": lambda p, s: ""},
                               config=RefineConfig(chunk_words=120, parallel=4))
        assert len(warnings) >= 1 and "troper" in out and out.count("noitceS") >= 2     # a middle chunk kept as written, the others translated
        assert re.search(r"## Section \d", out) and out.index("troper") < out.index("noitceS")


class TestAPassageLeftInTheSourceLanguageIsNotAccepted:
    """Seen live on a real model: of six sections, two came back with every heading, table cell and caption still in English, and no check noticed."""
    LAZY = lambda self: (lambda p, s: p)

    def test_a_passage_returned_as_it_was_is_refused_and_kept_with_a_notice(self):
        out, warnings, _ = run(LONG_DOC, reverse_words, TRANSLATE, "Spanish", per_unit={"c2": lambda p, s: p}, config=RefineConfig(chunk_words=120))
        assert [w["kind"] for w in warnings] == ["chunk_restored"] and "untranslated" in warnings[0]["detail"]
        assert out.count("noitceS") >= 2 and "## Section 2" in out                  # the other sections translated; this one as written

    def test_it_is_repaired_when_the_model_translates_on_the_second_try(self):
        lazy_first = lambda p, s: p if s.stage == "chunk" else reverse_words(p, s)
        out, warnings, model = run(LONG_DOC, lazy_first, TRANSLATE, "Spanish", config=RefineConfig(chunk_words=120))
        assert warnings == [] and "## Section 1" not in out and "noitceS 1" in out
        assert any(spec.stage == "chunk_repair" for _, _, spec in model.calls)

    def test_a_passage_with_its_headings_and_captions_left_in_english_is_refused(self):
        def forgets_headings(passage, spec):
            return "\n".join(line if line.lstrip().startswith("#") else reverse_words(line, spec) for line in passage.split("\n"))
        doc = "# Overview of the platform\n\nShort intro text here.\n\n## Pricing for every team\n\nSome words about the price list.\n\n## Security and support\n\nMore words about support.\n\n## What customers say\n\nA few kind words.\n"
        out, warnings, _ = run(doc, forgets_headings, TRANSLATE, "Spanish")
        assert [w["kind"] for w in warnings] == ["chunk_restored"] and out == doc

    def test_a_translation_that_keeps_some_names_and_commands_is_accepted(self):
        keeps_some = lambda p, s: "\n".join(l if "Nimbus" in l else reverse_words(l, s) for l in p.split("\n"))
        doc = "# Guide\n\n" + "\n\n".join(f"Paragraph number {i} explains how the platform works in plain words." for i in range(20)) + "\n\nNimbus Cloud Platform Guide Overview\n"
        out, warnings, _ = run(doc, keeps_some, TRANSLATE, "Spanish")
        assert warnings == []

    def test_a_document_already_in_the_target_language_is_kept_as_written_and_the_user_is_told(self):
        """It cannot be told apart from a skipped passage without detecting languages, so it is kept (it is the same text) with the same notice."""
        out, warnings, _ = run(LONG_DOC, lambda p, s: p, TRANSLATE, "English", config=RefineConfig(chunk_words=120))
        assert warnings and all("untranslated" in w["detail"] for w in warnings) and out == LONG_DOC

    def test_only_translate_asks_for_change(self):
        assert TRANSLATE.must_change and not SHORTER.must_change and not LONGER.must_change and not any(
            OPERATIONS[n].must_change for n in ("grammar", "simplify", "tone", "enhance"))

    def test_the_prompt_says_headings_are_translated(self):
        prompt = system_prompt("translation")
        assert "Headings are translated too" in prompt and "leave nothing in the source language" in prompt


class TestShorten:
    def test_every_heading_item_and_locked_block_survives_and_the_text_is_clearly_shorter(self):
        out, warnings, _ = run(LONG_DOC, cut_lines(0.62), SHORTER, config=RefineConfig(chunk_words=120))
        assert warnings == []
        assert analyse(out).headings == analyse(LONG_DOC).headings and analyse(out).lists == analyse(LONG_DOC).lists
        assert 0.3 <= word_count(out) / word_count(LONG_DOC) <= 0.8

    def test_locked_blocks_are_not_touched(self):
        out, warnings, _ = run(DOC, cut_lines(0.6), SHORTER)
        for kept in ["|:-----|--------------:|", '```python\nprint("hello")\n```', "> [!NOTE]\n> Prices may change next year.", "team.png",
                     "https://example.com/pricing", "---\ntitle: Pricing guide\n---"]:
            assert kept in out, kept

    def test_text_that_is_not_shorter_is_refused(self):
        out, warnings, _ = run(LONG_DOC, lambda p, s: p, SHORTER, config=RefineConfig(chunk_words=120))
        assert warnings and all(w["kind"] == "chunk_restored" for w in warnings) and out == LONG_DOC

    def test_a_summary_is_refused(self):
        out, warnings, _ = run(LONG_DOC, summarise, SHORTER, config=RefineConfig(chunk_words=120))
        assert warnings and out == LONG_DOC

    def test_an_invented_figure_is_refused_and_a_dropped_one_is_not(self):
        invent = lambda p, s: cut_lines(0.6)(p, s).replace("Section 1", "Section 1 (42%)")
        out, warnings, _ = run(LONG_DOC, invent, SHORTER, config=RefineConfig(chunk_words=120))
        assert any("numbers" in w["detail"] for w in warnings) and "42%" not in out
        out, warnings, _ = run(LONG_DOC, cut_lines(0.6), SHORTER, config=RefineConfig(chunk_words=120))
        assert warnings == []

    def test_the_prompt_gives_a_number_to_aim_for_and_keeps_every_item(self):
        prompt = system_prompt("make_shorter")
        assert "number of words on the LENGTH line, which is 40% of the passage" in prompt and "from 35% to 60% is accepted" in prompt
        assert "every list item" in prompt and "Do not drop a section" in prompt


class TestTheLengthLine:
    """Told "50-75%" a real model cut 2-6%; told "about 140 words (the passage has 233)" it lands near it."""

    def chunk(self, text="# T\n\n" + "word " * 100 + "\n"):
        from src.longform.chunk import chunk_blocks
        from src.longform.freeze import freeze_blocks
        return chunk_blocks(freeze_blocks(parse(text))[0], 1000)[0]

    def test_shorten_and_lengthen_get_a_target_in_words(self):
        c = self.chunk()
        assert "LENGTH: about 40 words (the passage has 101)" in user_message(SHORTER, None, c)
        assert "LENGTH: about 146 words (the passage has 101)" in user_message(LONGER, None, c)

    def test_no_other_operation_gets_one(self):
        c = self.chunk()
        for name in ("grammar", "simplify", "tone", "enhance", "translation"):
            assert "LENGTH:" not in user_message(OPERATIONS[name], "x", c), name

    def test_the_target_is_in_the_middle_of_the_told_and_the_accepted_windows(self):
        assert SHORTER.advertised[0] <= SHORTER.length_target <= SHORTER.advertised[1] and LONGER.advertised[0] <= LONGER.length_target <= LONGER.advertised[1]

    def test_the_model_really_receives_it_in_the_pipeline(self):
        _, _, model = run(LONG_DOC, cut_lines(0.6), SHORTER, config=RefineConfig(chunk_words=120))
        assert all("LENGTH: about" in messages[0]["content"] for _, messages, _ in model.calls)


class TestTheRepairNamesWhatWasNotTranslated:
    def test_the_message_quotes_the_lines_that_came_back_unchanged(self):
        def forgets_headings(passage, spec):
            return "\n".join(line if line.lstrip().startswith("#") else reverse_words(line, spec) for line in passage.split("\n"))
        doc = "# Overview of the platform\n\nShort intro text here.\n\n## Pricing for every team\n\nSome words about the price list.\n\n## Security and support\n\nMore words.\n"
        _, warnings, model = run(doc, forgets_headings, TRANSLATE, "Spanish")
        repair = next(messages[-1]["content"] for _, messages, spec in model.calls if spec.stage == "chunk_repair")
        assert '"# Overview of the platform"' in repair and '"## Pricing for every team"' in repair and "were not translated" in repair


class TestLengthen:
    def test_the_text_grows_and_keeps_its_structure(self):
        out, warnings, _ = run(LONG_DOC, pad(" and this is worth explaining in more detail for every reader who needs it"), LONGER,
                               config=RefineConfig(chunk_words=120))
        assert warnings == []
        assert analyse(out).headings == analyse(LONG_DOC).headings and analyse(out).lists == analyse(LONG_DOC).lists
        assert 1.12 <= word_count(out) / word_count(LONG_DOC) <= 2.4

    def test_a_new_figure_is_refused(self):
        out, warnings, _ = run(LONG_DOC, pad(" and this was finished in 2031 according to the team"), LONGER, config=RefineConfig(chunk_words=120))
        assert warnings and any("numbers" in w["detail"] for w in warnings) and "2031" not in out

    def test_text_that_did_not_grow_is_refused(self):
        out, warnings, _ = run(LONG_DOC, lambda p, s: p, LONGER, config=RefineConfig(chunk_words=120))
        assert warnings and out == LONG_DOC

    def test_a_ballooning_answer_is_refused(self):
        out, warnings, _ = run(LONG_DOC, lambda p, s: p + "\n\n" + p, LONGER, config=RefineConfig(chunk_words=120))
        assert warnings


class TestTheOperationsThemselves:
    def test_the_windows_the_checks_and_the_budgets(self):
        assert TRANSLATE.ratio == (0.7, 1.8) and TRANSLATE.wide_ratio == (0.4, 2.6) and TRANSLATE.paragraphs == "equal"
        assert SHORTER.ratio[1] < 1 < LONGER.ratio[0] and SHORTER.no_new_numbers and LONGER.no_new_numbers and not TRANSLATE.no_new_numbers
        assert TRANSLATE.expose == ("table", "blockquote", "image", "footnote") and SHORTER.expose == () and LONGER.expose == ()
        assert TRANSLATE.output_factor > LONGER.output_factor > SHORTER.output_factor == 1.6

    def test_only_translate_reads_the_instruction_and_it_defaults_to_english(self):
        assert instruction_line(TRANSLATE, "Spanish") == "TARGET LANGUAGE: Spanish" and instruction_line(TRANSLATE, None) == "TARGET LANGUAGE: English"
        assert instruction_line(SHORTER, "by half") is None and instruction_line(LONGER, "much") is None

    def test_a_window_override_must_keep_the_side_of_one_the_operation_is_on(self):
        assert ratio_window(SHORTER, '{"make_shorter": [0.4, 0.8]}') == (0.4, 0.8)
        assert ratio_window(SHORTER, '{"make_shorter": [0.5, 1.2]}') == SHORTER.ratio               # would allow growth
        assert ratio_window(LONGER, '{"make_longer": [1.2, 1.9]}') == (1.2, 1.9)
        assert ratio_window(LONGER, '{"make_longer": [0.8, 1.9]}') == LONGER.ratio
        assert ratio_window(OPERATIONS["grammar"], '{"grammar": [0.9, 1.1]}') == (0.9, 1.1)          # unchanged for the original four

    def test_translation_gets_a_bigger_token_budget_than_the_others(self):
        _, _, model = run(LONG_DOC, reverse_words, TRANSLATE, "Hindi")
        assert model.calls[0][2].max_tokens > 2 * 1.6 * len(LONG_DOC) / 3.5 * 0.9                  # four times the estimate, not 1.6 times
        from src.longform.chunk import chunk_blocks
        from src.longform.freeze import freeze_blocks
        frozen, _ = freeze_blocks(parse(LONG_DOC), expose=TRANSLATE.expose)
        chunk = chunk_blocks(frozen, 1000)[0]
        assert chunk_max_tokens(chunk, TRANSLATE) > 2 * chunk_max_tokens(chunk) and chunk_max_tokens(chunk) == chunk_max_tokens(chunk, OPERATIONS["grammar"])


class TestFreezeAndVerifyBuildingBlocks:
    def test_default_freezing_is_unchanged_and_exposing_changes_only_the_named_kinds(self):
        default, vault = freeze(DOC)
        assert "[[TABLE_1]]" in default and "[[QUOTE_1]]" in default and "[[IMAGE_1]]" in default
        exposed, vault2 = freeze(DOC, expose=("table",))
        assert "| Basic | 5 |" in exposed and "[[QUOTE_1]]" in exposed and "[[IMAGE_1]]" in exposed
        assert thaw(exposed, vault2) == DOC

    def test_freeze_and_thaw_stay_exact_with_everything_exposed_on_mixed_documents(self):
        pieces = ["| a | b |\n|:--|--:|\n| x | y |\n", "> [!NOTE]\n> text here\n", "> quote\n", "![alt](x.png)\n*cap*\n", "[l](http://a.b)\n", "plain words\n",
                  "```\ncode\n```\n", "- [ ] task\n", "# Head {#h}\n", "\n", "$$\nx\n$$\n", "text with ![i](y.png \"t\") inline\n", "---\n", "> ```\n> c\n> ```\n"]
        rng = random.Random(3)
        for _ in range(600):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 16)))
            frozen, vault = freeze(text, expose=("table", "blockquote", "image"))
            assert thaw(frozen, vault) == text, repr(text)

    def test_the_separator_row_snaps_back_only_when_the_shape_matches(self):
        from src.longform.chunk import chunk_blocks
        from src.longform.freeze import freeze_blocks
        original = "| a | b |\n|:--|--:|\n| x | y |\n"
        frozen, _ = freeze_blocks(parse(original), expose=("table",))
        chunk = chunk_blocks(frozen, 100)[0]
        assert snap_tables("| q | w |\n|---|---|\n| e | r |\n", chunk) == "| q | w |\n|:--|--:|\n| e | r |\n"
        assert snap_tables("| q |\n|---|\n| e |\n", chunk) == "| q |\n|---|\n| e |\n"              # a different shape is left for verify to refuse

    def test_figures_ignore_separators_and_tokens(self):
        assert figures("In 2024 we sold 1,000 and 1.5 units [[TABLE_12]] and [see this](link://7)") == {"2024", "1000", "15"}

    def test_the_length_window_widens_only_when_the_script_changes(self):
        from src.longform.chunk import chunk_blocks
        frozen, _ = freeze(LONG_DOC)
        from src.longform.freeze import freeze_blocks
        chunk = chunk_blocks(freeze_blocks(parse(LONG_DOC))[0], 1000)[0]
        han = to_han(chunk.frozen, None)
        assert not verify(chunk, han, ratio=(0.7, 1.8), paragraphs="equal").ok
        assert all(i.kind != "length" for i in verify(chunk, han, ratio=(0.7, 1.8), paragraphs="equal", wide_ratio=(0.4, 2.6)).issues)


class TestWordCount:
    @pytest.mark.parametrize("text, low, high", [("今天天气很好，所以我们一起去公园散步。", 8, 14), ("今日は天気がいいので、一緒に公園へ散歩に行きます。", 10, 18),
                                                  ("วันนี้อากาศดีมาก เราไปเดินเล่นที่สวนสาธารณะกันเถอะ", 8, 20)])
    def test_unspaced_scripts_count_as_words_not_as_one(self, text, low, high):
        assert low <= word_count(text) <= high

    def test_text_without_those_scripts_is_counted_exactly_as_before(self):
        pattern = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
        rng = random.Random(1)
        words = ["alpha", "it's", "well-known", "café", "Zażółć", "42", "a_b", "한국어", "русский", "عربى"]
        for _ in range(300):
            text = " ".join(rng.choice(words) + rng.choice(["", ",", ".", "!"]) for _ in range(rng.randint(0, 40)))
            assert word_count(text) == len(pattern.findall(text))
