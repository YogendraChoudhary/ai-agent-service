"""
Autocomplete spacing. Measured 2026-11-12 on 74 labelled contexts x 3 runs: after a finished word the old prompt produced a suggestion without
the space it needed on 62% of calls for gpt-4o-mini and 86% for Claude Haiku 5.5 ("toschedule"), and after "," or "." on 92-100%. The model now
puts its answer in <new>...</new> (a new word: the system adds the space) or <cont>...</cont> (continues straight on); text outside the tag is dropped.
"""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.models as M
import src.text_processing_agent as tpa
from src.autocomplete_join import apply_join_tag, fix_join, is_commentary

post = tpa._postprocess_autocomplete


class TestTags(unittest.TestCase):

    def test_a_new_word_gets_its_space(self):
        self.assertEqual(apply_join_tag("Thanks for your", "<new>help with this.</new>"), (" help with this.", True))

    def test_each_spelling_of_each_tag(self):
        for tag in ("new", "w", "NEW"):
            with self.subTest(tag=tag):
                self.assertEqual(apply_join_tag("so", f"<{tag}>on</{tag}>"), (" on", True))
        for tag in ("cont", "c", "CONT"):
            with self.subTest(tag=tag):
                self.assertEqual(apply_join_tag("tomor", f"<{tag}>row</{tag}>"), ("row", True))

    def test_the_space_is_added_only_where_a_space_belongs(self):
        self.assertEqual(apply_join_tag("Hi Sam,", "<new>hope</new>")[0], " hope")             # after punctuation
        self.assertEqual(apply_join_tag("Hi Sam, ", "<new>hope</new>")[0], "hope")             # the text already ends with a space
        self.assertEqual(apply_join_tag("line\n", "<new>next</new>")[0], "next")
        self.assertEqual(apply_join_tag("see (", "<new>this</new>")[0], "this")                # after an opening bracket
        self.assertEqual(apply_join_tag("", "<new>First words</new>")[0], "First words")       # empty text before the cursor

    def test_anything_outside_the_tag_is_dropped_so_an_explanation_cannot_leak(self):
        reply = 'The text ends with "make", a complete word, so I start a new word. <new>sure it works.</new> I hope that helps.'
        self.assertEqual(apply_join_tag("run it to make", reply), (" sure it works.", True))

    def test_a_reply_cut_off_inside_or_before_its_closing_tag(self):
        self.assertEqual(apply_join_tag("go", "<new>on and on")[0], " on and on")
        self.assertEqual(apply_join_tag("go", "<new>on and on</")[0], " on and on")
        self.assertEqual(apply_join_tag("go", "<new>on and on</ne")[0], " on and on")
        self.assertEqual(apply_join_tag("go", "<new>on and on</new")[0], " on and on")

    def test_an_empty_tag_is_no_suggestion(self):
        for reply in ("<cont></cont>", "<new></new>", "<new>   </new>"):
            with self.subTest(reply=reply):
                self.assertEqual(post("Thanks for", reply), "")

    def test_an_untagged_reply_is_left_alone(self):
        self.assertEqual(apply_join_tag("Thanks", " for everything"), (" for everything", False))
        self.assertEqual(apply_join_tag("Thanks", ""), ("", False))
        self.assertEqual(apply_join_tag("Thanks", None), (None, False))


class TestFixJoin(unittest.TestCase):

    def test_cases(self):
        cases = [
            ("Hello, ", " world", "world"),                  # never a double space
            ("Hello,\n", "world", "world"),
            ("Hi Sam,", "hope you are well", " hope you are well"),        # a space after punctuation
            ("Wait!", "Then", " Then"),
            ("It was (see below)", "and more", " and more"),
            ("end of it.", "Next steps", " Next steps"),                   # a new sentence
            ("3.", "5 million", "5 million"),                              # a decimal is not a sentence
            ("example.", "com", "com"),                                    # nor a domain name
            ("It costs 1,", "000 dollars", "000 dollars"),                 # 1,000
            ("At 10:", "30", "30"),
            ("Thanks so much", "! Really", "! Really"),                    # attaching punctuation stays attached
            ("Thanks", " for that", " for that"),                           # a leading space stays
            ("tomor", "row", "row"),                                       # mid-word: no space invented
            ("Thanks", "", ""),
            ("", "Hello", "Hello"),
        ]
        for before, suggestion, expected in cases:
            with self.subTest(before=before, suggestion=suggestion):
                self.assertEqual(fix_join(before, suggestion), expected)


class TestCommentary(unittest.TestCase):

    def test_what_is_and_is_not_commentary(self):
        for text in ('The text ends mid-word ("dep"), so I finish the word.', "Text ends with a complete word", "The user's text ends with a period",
                     "the line ends here, so", "This is a mid-word fragment", "only the missing letters"):
            with self.subTest(text=text):
                self.assertTrue(is_commentary(text))
        for text in ("row at noon.", " help with this.", "! I really appreciate it.", "the budget figures are confirmed by finance", "", None):
            with self.subTest(text=text):
                self.assertFalse(is_commentary(text))

    def test_untagged_commentary_is_never_shown_but_tagged_text_is(self):
        self.assertEqual(post("The end", "The text ends with a complete word, so I start with a space."), "")
        self.assertEqual(post("The end", "<new>The text ends here.</new>"), " The text ends here.")        # inside the tag it is the answer


class TestPostprocess(unittest.TestCase):

    def test_tagged_joins(self):
        cases = [
            ("Thanks for your", "<new>help with this.</new>", "Thanks for your help with this."),
            ("See you tomor", "<cont>row at noon.</cont>", "See you tomorrow at noon."),
            ("Hi Sam,", "<new>hope you are well.</new>", "Hi Sam, hope you are well."),
            ("Hi Sam,", "<cont>hope you are well.</cont>", "Hi Sam, hope you are well."),     # fix_join still adds the space
            ("Dear Sam, ", "<new>Thank you</new>", "Dear Sam, Thank you"),
            ("Thanks so much", "<cont>! I appreciate it</cont>", "Thanks so much! I appreciate it"),
            ("I will be an", "<new>amazing addition</new>", "I will be an amazing addition"),
        ]
        for before, reply, expected in cases:
            with self.subTest(before=before, reply=reply):
                self.assertEqual(before + post(before, reply), expected)

    def test_the_echo_strip_no_longer_eats_a_new_word_that_starts_like_the_last_word(self):
        for before, new_word in [("We went to the", " theory lab"), ("She is in", " interesting company"), ("He gave it to", " Tony"),
                                 ("We live on", " one street"), ("They are at", " attention now")]:
            with self.subTest(before=before):
                self.assertEqual(before + post(before, new_word), before + new_word)

    def test_a_real_echo_without_a_space_is_still_stripped(self):
        self.assertEqual(post("everyo", "everyone! Hope you are well"), "ne! Hope you are well")

    def test_a_tagged_answer_is_never_echo_stripped(self):
        self.assertEqual("an" + post("an", "<new>answer</new>"), "an answer")                # "an" + "swer" would be wrong

    def test_the_old_untagged_behaviour_is_kept(self):
        self.assertEqual(post("morning", "How are you?"), " How are you?")                  # the capitalised join fix
        self.assertEqual(post("Really?", "? I think so"), " I think so")                    # duplicated terminal punctuation
        self.assertEqual(post("Hello", '"friend"'), "friend")                                # wrapping quotes
        self.assertEqual(post("Hello", " friend  \n "), " friend")                           # trailing whitespace

    def test_no_text_before_the_cursor(self):
        self.assertEqual(post("", "<new>Hello there</new>"), "Hello there")


class TestPrompt(unittest.TestCase):

    def test_it_asks_for_the_tags_and_has_no_leftover_symbol(self):
        prompt = tpa.AUTOCOMPLETE_SYSTEM_PROMPT
        for needle in ("<new>", "</new>", "<cont>", "</cont>", "UNFINISHED", "[Text after cursor]"):
            self.assertIn(needle, prompt)
        self.assertNotIn("␣", prompt)

    def test_every_example_in_the_prompt_round_trips_through_the_parser(self):
        examples = re.findall(r"Text: '([^']*)' -> <(new|cont)>(.*?)</\2>", tpa.AUTOCOMPLETE_SYSTEM_PROMPT)
        self.assertGreaterEqual(len(examples), 10)
        for before, tag, body in examples:
            with self.subTest(before=before):
                joined = before + post(before, f"<{tag}>{body}</{tag}>")
                self.assertTrue(joined.startswith(before) and joined.endswith(body))
                gap = joined[len(before):-len(body)]
                self.assertEqual(gap, " " if (tag == "new" and not before.endswith(" ")) else "")


class _Anthropic:
    def __init__(self, text):
        self.sent = {}
        rec = self

        class _Messages:
            def create(self, **kwargs):
                rec.sent.update(kwargs)
                return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])

        self.messages = _Messages()


class TestEveryProviderPath(unittest.TestCase):

    def suggest(self, provider, reply, before="Thanks for your"):
        if provider == "claude":
            client = _Anthropic(reply)
            patcher = patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client)
        elif provider == "gemini":
            client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text=reply)))
            patcher = patch.object(tpa, "_get_cached_gemini_client", lambda *a, **k: client)
        else:
            msg = SimpleNamespace(content=reply)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)]))))
            patcher = patch.object(tpa, "_get_cached_openai_client", lambda *a, **k: client)
        with patcher:
            return tpa.get_autocomplete_suggestion(before, "", provider, "k")

    def test_every_provider_gets_the_join_applied(self):
        for provider in ("claude", "openai", "gemini", "grok", "deepseek"):
            with self.subTest(provider=provider):
                self.assertEqual(self.suggest(provider, "<new>help with this.</new>"), " help with this.")
                self.assertEqual(self.suggest(provider, "<cont>row at noon.</cont>", "See you tomor"), "row at noon.")

    def test_the_prompt_reaches_the_model_unchanged_in_shape(self):
        client = _Anthropic("<new>x</new>")
        with patch.object(tpa, "_get_cached_anthropic_client", lambda *a, **k: client):
            tpa.get_autocomplete_suggestion("Thanks for your", "", "claude", "k")
        self.assertIs(client.sent["system"][0]["text"], tpa.AUTOCOMPLETE_SYSTEM_PROMPT)
        self.assertEqual(client.sent["max_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
