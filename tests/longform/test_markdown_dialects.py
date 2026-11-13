"""
Markdown beyond CommonMark: front matter, display math, footnotes, task lists, wikilinks, strikethrough, :emoji: codes, {#attributes}.

They must be found in code, never left to the model: found as opaque blocks (kind "other" with a flavor, the kind Edit never touches and
freeze already turns into a token), frozen to tokens inside prose, and never seen by the model. Plain Markdown must come out exactly as
it did before (the differential against the original code is recorded in docs/DOCUMENT_UPLOAD_API_HANDOFF.md).
"""
import random
import re
import time

import pytest

from src.document_placement import resolve_placement
from src.longform.freeze import collides, freeze, thaw
from src.longform.markdown_ast import heading_lines, is_front_matter, join, parse, scan
from src.longform.prompts import OPERATIONS
from src.longform.refine import qualifies, refine_stream, RefineConfig
from tests.longform.fakes import ScriptedRefiner, passage_of, polish

GRAMMAR = OPERATIONS["grammar"]

FRONT = "---\ntitle: My Post\ntags: [a, b]\ndate: 2026-11-13\n---\n\n# Real Heading\n\nBody text here.\n"


def kinds(text):
    return [(b.kind, b.flavor) for b in parse(text) if b.kind != "blank"]


class TestFrontMatter:
    def test_yaml_front_matter_is_one_opaque_block_and_not_a_heading(self):
        assert kinds(FRONT) == [("other", "front_matter"), ("heading", ""), ("paragraph", "")]
        assert heading_lines(FRONT) == [(1, "# Real Heading")]            # it was: H2 "title: My Post" plus the real heading

    def test_toml_front_matter(self):
        text = '+++\ntitle = "My Post"\n[params]\nx = 1\n+++\n\n# H\n'
        assert kinds(text)[0] == ("other", "front_matter") and heading_lines(text) == [(1, "# H")]

    def test_crlf_and_trailing_spaces_on_the_delimiters(self):
        text = "---  \r\ntitle: x\r\n---\r\n\r\n# H\r\n"
        assert kinds(text)[0] == ("other", "front_matter") and join(parse(text)) == text

    def test_lists_nested_values_and_comments_are_front_matter(self):
        text = "---\n# a comment\ntags:\n  - a\n  - b\nauthor:\n  name: X\n---\nBody\n"
        assert kinds(text)[0] == ("other", "front_matter")

    @pytest.mark.parametrize("yaml", [
        '{title: "My Post", tags: [docs, finance]}\n',
        '? "custom key"\n: Value\n',
        'defaults: &defaults {draft: true}\npost: {<<: *defaults, title: "My Post"}\n',
        '{}\n',
    ])
    def test_yaml_mappings_are_kept_as_original_text(self, yaml):
        front_matter = "---\n" + yaml + "...\n"
        text = front_matter + "\n# Body\n"
        block, *_ = parse(text)
        assert block.flavor == "front_matter" and block.text == front_matter
        assert heading_lines(text) == [(1, "# Body")]
        assert join(parse(text)) == text

    @pytest.mark.parametrize("text", [
        "---\nJust some prose in a rule pair\n---\n\nBody\n",          # prose between two rules
        "---\n---\nBody\n",                                           # two rules, nothing between
        "---\ntitle: never closed\n\nBody text and more\n",
        "Intro\n\n---\ntitle: not at the very start\n---\n\nBody\n",   # front matter is only ever the first thing in the file
        "---\n**Note:** emphasis is not a key\n---\n\nBody\n",
        "---\n- a\n- b\n---\n\nBody\n",
        "---\ntitle: [unclosed\n---\n\nBody\n",
    ])
    def test_a_pair_of_rules_is_not_front_matter(self, text):
        assert all(flavor != "front_matter" for _, flavor in kinds(text))

    def test_front_matter_inside_a_code_fence_is_code(self):
        text = "```yaml\n---\ntitle: x\n---\n```\n"
        assert kinds(text) == [("fence", "")]

    def test_helper(self):
        assert is_front_matter("---\ntitle: x\n---\n") and not is_front_matter("---\nprose\n---\n") and not is_front_matter("")


class TestMathAndFootnotes:
    def test_display_math_is_one_opaque_block(self):
        text = "Energy is:\n\n$$\nE = mc^2\n$$\n\nDone.\n"
        assert kinds(text) == [("paragraph", ""), ("other", "math"), ("paragraph", "")]

    def test_one_line_math_and_bracket_math(self):
        assert ("other", "math") in kinds("Before.\n\n$$ x^2 + y^2 = z^2 $$\n\nAfter.\n")
        assert ("other", "math") in kinds("Before.\n\n\\[\nx = 1\n\\]\n\nAfter.\n")

    def test_a_heading_looking_line_inside_math_is_not_a_heading(self):
        assert heading_lines("$$\n# not a heading\nx\n$$\n") == []

    def test_math_inside_code_is_code_and_unclosed_math_is_left_alone(self):
        assert kinds("```\n$$\nx\n$$\n```\n") == [("fence", "")]
        assert all(flavor != "math" for _, flavor in kinds("Price $$ is high and never closed\n\nMore text\n"))

    def test_footnote_definitions_with_continuation_lines(self):
        text = "A claim.[^1]\n\n[^1]: First line of the note,\n    and its second line.\n\nNext paragraph.\n"
        assert kinds(text) == [("paragraph", ""), ("other", "footnote"), ("paragraph", "")]
        assert [b.text for b in parse(text) if b.flavor == "footnote"] == ["[^1]: First line of the note,\n    and its second line.\n\n"[:-1]] \
            or [b.text.rstrip() for b in parse(text) if b.flavor == "footnote"] == ["[^1]: First line of the note,\n    and its second line."]


INLINE = [
    ("The value $x_1 + y_2$ is small", 1),
    ("Display $$a^2$$ inline", 1),
    ("See [[Project Plan]] and ![[diagram.png]] here", 2),
    ("A claim.[^note] and another.[^2]", 2),
    ("This is ~~old~~ new", 1),
    ("Shipped :rocket: and :tada: today", 2),
    ("## Install {#install .wide}", 1),
]
NOT_INLINE = [
    "Price is $5 and $10 total.",
    "Suffix style: 5$ and 10$ and 7$ more.",
    "Paid US$5 and then $6 more",
    "Time 10:30:45 and key:value:other",
    "A [x] note and a [y] other",
    'JSON {"a": 1} and {x} and {}',
    "Costs ~5 or ~~ nothing",
    "Use the ~/path and a_b_c",
]


class TestInline:
    @pytest.mark.parametrize("text, count", INLINE)
    def test_each_construct_becomes_a_token_and_comes_back_exactly(self, text, count):
        frozen, vault = freeze(text + "\n")
        assert len(re.findall(r"\[\[HTML_\d+\]\]", frozen)) == count, frozen
        assert thaw(frozen, vault) == text + "\n"

    @pytest.mark.parametrize("text", NOT_INLINE)
    def test_ordinary_text_is_left_for_the_model(self, text):
        assert freeze(text + "\n")[0] == text + "\n"

    def test_task_list_boxes_are_tokens_and_the_items_keep_their_words(self):
        frozen, vault = freeze("- [ ] write the draft\n- [x] outline\n  - [ ] nested\n1. [X] numbered\n")
        assert re.findall(r"\[\[HTML_\d+\]\]", frozen) and "write the draft" in frozen and "[ ]" not in frozen and "[x]" not in frozen
        assert thaw(frozen, vault) == "- [ ] write the draft\n- [x] outline\n  - [ ] nested\n1. [X] numbered\n"

    def test_code_fences_are_not_searched(self):
        assert scan("```\n$x$ and [[a]] and ~~b~~\n```\n").features == []

    def test_a_document_that_already_contains_our_token_syntax_still_collides(self):
        assert collides("It literally says [[TABLE_1]] in prose.\n")
        assert "[[TABLE_1]]" not in "".join(freeze("A [[TABLE_1]] note\n")[0].split("[[TABLE_1]]")[:0])


class TestScan:
    def test_features_in_a_fixed_order(self):
        text = (FRONT + "\nMath $x$, a note.[^1] and a [[wiki]] and ~~old~~ :tada: {#id}\n\n- [ ] task\n\n[^1]: the note\n")
        assert scan(text).features == ["front_matter", "math", "footnotes", "task_list", "wikilinks", "strikethrough", "attributes", "emoji_codes"]

    def test_plain_markdown_reports_nothing(self):
        assert scan("# T\n\nPlain paragraph with a [link](http://a.b) and $5.\n\n- item\n").features == []

    def test_a_checkbox_looking_paragraph_is_not_a_task_list(self):
        assert scan("[ ] not in a list\n").features == [] and scan("- [ ] in a list\n").features == ["task_list"]


class TestEverythingStillRoundTrips:
    FRAGMENTS = ["---\ntitle: x\n---\n", "$$\nx^2\n$$\n", "$$ y $$\n", "\\[\nz\n\\]\n", "[^1]: note\n    more\n", "Text[^1] here.\n",
                 "- [ ] a\n", "- [x] b\n", "See [[Page]] now.\n", "~~old~~ new\n", ":smile: ok\n", "## H {#h}\n", "# H\n", "plain text\n", "\n",
                 "```\n$$\n---\n```\n", "> $x$ quote\n", "| a | b |\n|---|---|\n| $x$ | y |\n", "***\n", "---\n", "1. one\n", "  - nested\n",
                 "$5 and $6\n", "[link](http://a.b)\n", "![img](x.png)\n", "<div>$$</div>\n", "\r\n", "$$\n", "[^9]:\n"]

    def test_parse_and_freeze_are_lossless_on_a_thousand_mixed_documents(self):
        rng = random.Random(5)
        for _ in range(1000):
            text = "".join(rng.choice(self.FRAGMENTS) for _ in range(rng.randint(1, 22)))
            assert join(parse(text)) == text, repr(text)
            frozen, vault = freeze(text)
            assert thaw(frozen, vault) == text, repr(text)


DOC = ("---\ntitle: Quarterly plan\ntags: [plan, q4]\n---\n\n# Quarterly plan\n\n"
       "Our target is $x_1 + y_2$ per team, see [[Project Plan]] for details.[^1] The old number was ~~ten~~ twelve :tada:\n\n"
       "$$\nE = mc^2\n$$\n\n- [ ] write the draft\n- [x] review the numbers\n\n[^1]: Source: finance, Nov 2026.\n")
PROTECTED_STRINGS = ["---\ntitle: Quarterly plan\ntags: [plan, q4]\n---", "$x_1 + y_2$", "[[Project Plan]]", "[^1]", "~~ten~~", ":tada:",
                     "$$\nE = mc^2\n$$", "- [ ] write", "- [x] review", "[^1]: Source: finance, Nov 2026."]


class TestTheModelNeverSeesOrChangesThem:
    def test_the_dialect_text_is_not_in_what_the_model_is_sent(self):
        model = ScriptedRefiner(polish)
        list(refine_stream(DOC, None, GRAMMAR, model, RefineConfig()))
        assert model.calls
        sent = " ".join(m["content"] for _, messages, _ in model.calls for m in messages)
        for fragment in ["Quarterly plan\ntags", "$x_1", "[[Project Plan]]", "~~ten~~", ":tada:", "E = mc^2", "[x] review", "Nov 2026"]:
            assert fragment not in sent, fragment

    def test_a_model_that_rewrites_the_prose_leaves_every_one_of_them_byte_for_byte(self):
        def reword(passage, spec):
            return passage.replace("Our target", "Our goal").replace("The old number", "The previous figure").replace("write the", "draft the")
        out = "".join(x for x in refine_stream(DOC, None, GRAMMAR, ScriptedRefiner(reword), RefineConfig()) if isinstance(x, str))
        assert "Our goal" in out and "The previous figure" in out                     # the prose really was reworded
        assert "- [ ] draft the draft" in out                                          # the words of a task item are prose, its box is not
        for fragment in PROTECTED_STRINGS:
            if fragment != "- [ ] write":
                assert fragment in out, fragment
        assert "- [ ] " in out and "- [x] review" in out

    def test_short_documents_with_such_markup_take_the_protected_path_and_plain_ones_do_not(self):
        assert qualifies("Short note with $x^2$ math.\n")
        assert qualifies("- [ ] one\n- [x] two\n")
        assert qualifies("See [[Page]] for more.\n")
        assert qualifies("---\ntitle: x\n---\n\nHello there.\n")
        assert not qualifies("A short plain note with a [link](http://a.b) and $5.\n")
        assert not qualifies("Shipped :rocket: today\n")                       # emoji codes alone are not worth the chunked path


class TestDraftPlacementKeepsFrontMatterFirst:
    @pytest.mark.parametrize("task", ["Add a short intro paragraph at the beginning", "add a summary at the top of the document"])
    def test_the_top_is_just_below_the_front_matter(self, task):
        placement, question = resolve_placement(FRONT, task, None)
        assert question is None and placement.position == "before"
        assert FRONT[placement.start_utf16:placement.end_utf16].startswith("# Real Heading")

    def test_front_matter_only(self):
        doc = "---\ntitle: X\n---\n"
        placement, _ = resolve_placement(doc, "add a summary at the top", None)
        assert placement.position == "after" and doc[placement.start_utf16:placement.end_utf16] == doc

    def test_draft_starts_below_yaml_mapping_in_flow_style(self):
        doc = "---\n{title: Quarterly report, tags: [docs, finance]}\n---\n\n# Sales\n"
        placement, question = resolve_placement(doc, "add a summary at the top", None)
        assert question is None and placement.position == "before" and placement.text == "# Sales\n"

    def test_a_document_without_front_matter_is_unchanged(self):
        doc = "# Heading\n\nFirst.\n"
        placement, _ = resolve_placement(doc, "add an intro at the very beginning", None)
        assert placement.position == "before" and doc[placement.start_utf16:].startswith("# Heading")


class TestCraftedInputsStayFast:
    """Every pattern repetition is bounded. The original link-label pattern took ~23 s on the first of these (100,000 characters)."""
    N = 100_000
    CASES = {
        "escaped brackets": ("\\[\nx\n" * 20000)[:N], "unclosed math": ("$a " * 40000)[:N], "unclosed display math": ("$$ a\n" * 20000)[:N],
        "unclosed wikilinks": ("[[a " * 25000)[:N], "unclosed strike": ("~~a " * 25000)[:N], "emoji starts": (":a " * 33000)[:N],
        "attribute starts": ("{#a " * 25000)[:N], "footnote refs": ("[^a " * 25000)[:N], "task list": ("- [ ] a\n" * 12000)[:N],
        "front matter never closed": "---\n" + "k: v\n" * 20000,
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_freeze_and_scan_finish_quickly(self, name):
        text = self.CASES[name]
        started = time.perf_counter()
        frozen, vault = freeze(text)
        scan(text)
        assert time.perf_counter() - started < 4.0, name
        assert thaw(frozen, vault) == text
