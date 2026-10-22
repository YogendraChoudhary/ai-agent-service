"""
Documents for the long-form tests. They are built, not stored: deterministic, readable here, exact in size, and every
structural element the plan's acceptance document names is in them (docs/rich_content_engine_plan.md, Phase 1).
"""

import random
from typing import Dict, List

_WORDS = (
    "people team work time project plan result review customer market product design system process change report "
    "decision meeting budget quality support growth manager leader skill habit health energy focus task goal reason "
    "example question answer problem solution method option risk cost value trust feedback story detail pattern "
    "simple clear steady careful early later often rarely really mostly usually slowly quickly together carefully "
    "build write read check share keep move start finish choose notice repeat protect measure explain improve "
    "daily weekly small large short long first second final common useful honest practical specific recent current"
).split()
_STARTS = ("Most", "Every", "Teams", "A good", "When", "Because", "Often", "Before", "After", "Without", "People", "Still")


def sentence(rng: random.Random, words: int) -> str:
    body = [rng.choice(_WORDS) for _ in range(words - 1)]
    return f"{rng.choice(_STARTS)} " + " ".join(body) + rng.choice([".", ".", ".", "; and then it holds."])


def paragraph(rng: random.Random, words: int) -> str:
    """About `words` words in sentences of 8-20 words."""
    out, count = [], 0
    while count < words:
        n = min(rng.randint(8, 20), max(2, words - count))
        out.append(sentence(rng, n))
        count += n
    return " ".join(out)


TABLE = "| Option | Cost | Best for |\n|---|---|---|\n| Plan A | $10 | Small teams |\n| Plan B | $25 \\| month | Larger teams |"
QUOTE = "> A pull quote that must stay exactly as written.\n> It has two lines."
NESTED_QUOTE = "> Outer quote.\n>\n> > Inner quote with *emphasis*.\n>\n> Back to the outer one."
FIGURE = "![A desk with a laptop](https://cdn.example.com/img/desk.jpg)\n*A caption under the picture*"
FENCE = "```python\nfor x in range(3):\n    print(x)  # > not a quote\n\n> still code\n```"
HTML = "<div class=\"note\">Raw <b>HTML</b> stays as it is.</div>"


def acceptance_document(words: int = 5000, seed: int = 7) -> str:
    """~`words` words with 4 tables, 6 blockquotes, 5 images, 10 links and 2 code fences (the plan's acceptance document)."""
    rng = random.Random(seed)
    sections = 10
    per = max(60, (words - 60) // sections // 3)
    parts: List[str] = ["# A long working document", "", "*A subtitle line that is italic.*", "", paragraph(rng, 60), ""]
    tables = [TABLE] * 4
    quotes = [QUOTE, NESTED_QUOTE, QUOTE, QUOTE, "> One.\n>\n> Two paragraphs in one quote.", QUOTE]
    links = ["[the guide](https://example.com/guide)", "[a study](https://example.org/study \"Study title\")",
             "[the docs](https://docs.example.com/a_(b))", "[our policy][pol]", "https://example.net/plain.", "<https://auto.example.com/x>",
             "[ref two](https://example.com/two?a=1&b=2)", "[three](https://example.com/3)", "[four](https://example.com/4)",
             "[![logo](https://cdn.example.com/logo.png)](https://example.com/home)"]
    for i in range(sections):
        parts += [f"## Section {i + 1}: {rng.choice(_WORDS).title()} {rng.choice(_WORDS)}", ""]
        parts += [paragraph(rng, per), ""]
        parts += [f"{paragraph(rng, per // 2)} See {links[i % len(links)]} for details. {paragraph(rng, per // 2)}", ""]
        if i < 4:
            parts += [TABLE if i % 2 == 0 else TABLE.replace("Plan", "Tier"), ""]
        if i < 6:
            parts += [quotes[i], ""]
        if i < 3:
            parts += [FIGURE.replace("A desk with", f"Desk {i} with"), ""]
        if i == 3:
            parts += [f"A picture ![inline icon](https://cdn.example.com/i.png) sits in the middle of this sentence. {paragraph(rng, 30)}", ""]
        if i == 5:
            parts += [f"Another one: ![second inline](https://cdn.example.com/j.png) and then the text carries on. {paragraph(rng, 30)}", ""]
        if i in (2, 7):
            parts += [FENCE, ""]
        if i % 3 == 0:
            parts += [f"- {paragraph(rng, 12)}", f"- {paragraph(rng, 10)}", f"  - {paragraph(rng, 8)}", f"- {paragraph(rng, 9)}", ""]
        parts += [paragraph(rng, per), ""]
    parts += ["[pol]: https://example.com/policy \"Policy\"", ""]
    return "\n".join(parts)


def edge_documents() -> Dict[str, str]:
    """The cases the plan lists, plus a few it implies."""
    rng = random.Random(3)
    return {
        "pipes_in_cells": "Intro paragraph here.\n\n| a | b |\n|---|---|\n| x \\| y | `a|b` |\n\nAfter.\n",
        "quote_marks_inside_fences": "Text.\n\n```\n> not a quote\n| not | a table |\n```\n\n~~~\n> tilde fence\n~~~\n\nMore text.\n",
        "nested_quotes": f"Before.\n\n{NESTED_QUOTE}\n\nAfter.\n",
        "inline_image_mid_sentence": "A sentence with ![an icon](https://x.example/i.png \"Icon\") in the middle of it, then more.\n",
        "reference_links": "See [the guide][g] and [another][] and ![logo][l].\n\n[g]: https://example.com/g\n[another]: https://example.com/a \"A\"\n[l]: https://example.com/l.png\n",
        "one_huge_paragraph": paragraph(rng, 2000) + "\n",
        "crlf": "# Title\r\n\r\nParagraph one.\r\n\r\n| a | b |\r\n|---|---|\r\n| 1 | 2 |\r\n\r\n> quote\r\n\r\nEnd.\r\n",
        "only_a_table": TABLE + "\n",
        "setext_headings": "Title\n=====\n\nSub\n---\n\nText.\n",
        "list_with_code": "- item\n  ```sh\n  echo hi\n  ```\n- next\n  - nested [l](https://a.example)\n",
        "linked_image": "[![alt](https://cdn.example/a.png)](https://example.com/target)\n",
        "html_and_rules": f"Intro.\n\n{HTML}\n\n---\n\nOutro.\n\n***\n",
        "bare_urls": "Visit https://example.com/a, or (https://example.org/b), or www.example.net/c. Not an email: a@b.co.\n",
        "no_final_newline": "Last line without a newline",
        "empty": "",
        "blank_only": "\n\n\n",
        "heading_only": "# Just a heading\n",
        "inline_html_and_emails": ("Press <kbd>Ctrl</kbd>+<kbd>C</kbd>.<br>Write to <team@example.com> or ops+x@mail.example.org, "
                                   "a < b and c > d. <!-- note --> An <img src=\"a.png\" alt='x y' /> too.\n"),
        "unicode": "Café “quoted” — naïve résumé 日本語 and emoji 🎉 with [a link](https://例え.jp/パス).\n",
    }
