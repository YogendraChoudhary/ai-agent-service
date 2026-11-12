"""Accuracy and latency of document_action "auto": rules alone, the light-tier model alone, and the two together (live provider calls).

  .venv/bin/python scripts/document_actions/eval_auto.py --provider openai --out <file.json>

Local-only measurement script. The labels are the intended mode of each phrasing against the sample "Launch guide" document.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.update({"REDIS_URL": "", "BACKEND_WEBHOOK_URL": "", "VOICE_FLOW_ENABLED": "false", "MODEL_PREFLIGHT_ON_START": "false"})

from config import settings  # noqa: E402
from src.document_action import CLASSIFIER_SYSTEM, LABELS, decide_by_rules, decide_with_model  # noqa: E402
from src.intent_classifier import classify_label  # noqa: E402

DOC = ("# Launch guide\n\n## Introduction\n\nThis introduction is very very useful for readers who want to understand the launch.\n\n"
       "## Benefits\n\nThe benefits are clear and clear for every team that adopts the new workflow.\n\n"
       "## Pricing\n\nThe price is USD 20 per seat per month in 2026.\n\n## Conclusion\n\nWe recommend launching in May with a small cohort.\n")

ITEMS = {
    "ask": [
        "What does the Team plan cost?", "Summarize the document", "Is the pricing section consistent with the introduction?",
        "Does the document mention refunds?", "Which section talks about onboarding?", "Review the conclusion and tell me what's weak",
        "Can you explain what the Benefits section is saying?", "How many sections does this document have?", "What's the main argument here?",
        "Are there any contradictions?", "Give me feedback on the introduction", "Is the tone too formal?",
        "tell me what is missing from this proposal", "list the action items in this document", "Why is the second paragraph so long?",
        "Check the document for repeated points", "Who is the target audience of this guide?", "Could you find where we mention the deadline?",
        "Should I add a section about risks?", "¿Qué dice el documento sobre los precios?", "Anything unclear in the pricing?",
    ],
    "draft": [
        "add a short note about annual billing below the Pricing heading", "Continue writing", "write the next paragraph",
        "add a new section about security after Pricing", "Draft a summary below the Pricing heading",
        "add three bullet points about benefits at the end of the Benefits section", "write a transition between Pricing and Conclusion",
        "add a caption below the table", "Write a closing sentence for the Conclusion", "add a paragraph about our refund policy",
        "insert a quote after the first paragraph", "add more detail to the Benefits paragraph", "include two examples in the Introduction",
        "Add a disclaimer at the end", "Write an intro paragraph for this document", "Add a sentence thanking the reader",
        "put a call to action below the pricing table", "Add a FAQ section", "add Acme Inc. below the Contact heading",
        "add an example of how the discount works after the Discounts paragraph", "expand on the risks in a new paragraph below Pricing",
        "continue the paragraph with the next steps", "Draft a LinkedIn post announcing our launch",
    ],
    "edit": [
        "make the Introduction more concise", "fix the typos", "rewrite the conclusion to sound more confident", "shorten the Benefits paragraph",
        "translate the introduction to Spanish", "add (2024) next to the title", "add a phone number after the address",
        "rename the Contact heading to Get in touch", "make the quote shorter", "change the tone of the intro to friendly",
        "remove the repetition in Benefits", "proofread the document", "make this paragraph clearer",
        "turn the Benefits paragraph into simpler language", "tighten the text under Pricing", "use active voice throughout",
        "make the whole thing sound more professional", "make the title more engaging", "fix the grammar in the second paragraph",
        "Reword the first sentence", "polish the pricing paragraph", "make it shorter",
    ],
    "write": [
        "Write a blog post about remote work for small teams", "Write a Python function that checks if an email address is valid",
        "Generate an image of a cat on a sofa", "Make a bar chart of quarterly revenue: Q1 10, Q2 14, Q3 9, Q4 17",
        "Write an email asking the team for their availability next week", "Write a short essay about cats",
        "Write a product description for our app", "Create a 3-item checklist for onboarding a new engineer",
        "Design an onboarding card UI for a meditation app with a dark background", "write a poem about autumn",
        "Write a cover letter for a marketing role", "Give me 10 name ideas for a coffee shop",
        "Write a SQL query that returns the top 5 customers by revenue", "Write a press release about our new office",
        "Write a limerick about debugging", "Compose a thank-you note to a customer",
    ],
}

# Phrased after the classifier prompt was written and sharing no wording with its examples: the honest accuracy number.
HELDOUT = {
    "ask": [
        "hey, could you tell me whether the pricing is stated anywhere?", "what's the weakest part of this write-up",
        "Does anything in here contradict the launch date?", "I'm not sure the Benefits section lands. Thoughts?", "how long is this document",
        "Dime si el tono es demasiado formal", "Count the number of headings", "is it worth splitting the introduction in two?",
        "What would a skeptical reader object to?", "Where do we talk about the cohort size?",
    ],
    "draft": [
        "Please add a paragraph on supported browsers under Pricing", "write a short bio of the author at the top", "I need a conclusion paragraph at the end",
        "Add 3 bullets listing the key features after the intro", "can you put a table of plan comparisons below the price paragraph", "write what comes next",
        "Add a section called Risks between Benefits and Pricing", "append a short 'next steps' list", "Insert a pull quote after the Benefits paragraph",
        "escribe un párrafo nuevo después de la introducción",
    ],
    "edit": [
        "the intro is wordy, trim it", "Can you make the pricing paragraph sound less salesy?", "swap 'utilize' for 'use' everywhere",
        "capitalize the headings properly", "Rewrite the conclusion in the first person", "make the second sentence of the intro shorter",
        "put the year in brackets after the title", "reduce the Benefits paragraph by half", "Convert the intro to British English",
        "Reescribe la introducción en un tono más cálido",
    ],
    "write": [
        "Write a thank-you email to a client who just signed", "make me a pie chart: apples 30, pears 20, plums 10",
        "Create a landing page hero section with a headline and a sign-up button", "Write a JavaScript function that debounces another function",
        "Compose a tweet announcing our beta", "Brainstorm slogans for a bike repair shop", "Write a 200-word story about a lighthouse keeper",
        "give me a packing list for a weekend camping trip", "Generate a picture of a red bicycle leaning on a wall", "Write a job description for a junior designer",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "claude", "gemini", "grok"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", default="main", choices=["main", "heldout"])
    args = parser.parse_args()
    key = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key,
           "gemini": settings.default_gemini_api_key, "grok": settings.default_grok_api_key}[args.provider]
    cases = [(label, text) for label, texts in (ITEMS if args.set == 'main' else HELDOUT).items() for text in texts]

    def classify(text, system, labels):
        return classify_label(text, args.provider, key, system=system, labels=labels, is_byok=True,
                              timeout=settings.document_action_classifier_timeout_seconds)

    def run(case):
        label, text = case
        rules = decide_by_rules(text, DOC)
        t = time.perf_counter()
        model = decide_with_model(text, DOC, classify=classify)
        model_s = time.perf_counter() - t
        return {"expected": label, "query": text, "rules": rules and rules.action, "rules_reason": rules and rules.reason,
                "model": model.action, "model_source": model.source, "model_seconds": round(model_s, 2),
                "combined": (rules or model).action, "combined_source": (rules or model).source}

    with ThreadPoolExecutor(4) as pool:     # the label classifier has 8 slots; 4 workers keep the timing honest
        rows = list(pool.map(run, cases))
    decided = [r for r in rows if r["rules"]]
    out = {
        "provider": args.provider, "items": len(rows),
        "rules_coverage": round(len(decided) / len(rows), 3),
        "rules_precision": round(sum(r["rules"] == r["expected"] for r in decided) / len(decided), 3),
        "rules_wrong": [(r["query"], r["expected"], r["rules"]) for r in decided if r["rules"] != r["expected"]],
        "model_accuracy_all": round(sum(r["model"] == r["expected"] for r in rows) / len(rows), 3),
        "model_fallbacks": sum(r["model_source"] == "fallback" for r in rows),
        "model_accuracy_where_rules_unsure": None,
        "combined_accuracy": round(sum(r["combined"] == r["expected"] for r in rows) / len(rows), 3),
        "combined_wrong": [(r["query"], r["expected"], r["combined"], r["combined_source"]) for r in rows if r["combined"] != r["expected"]],
        "model_seconds_median": sorted(r["model_seconds"] for r in rows)[len(rows) // 2],
        "model_seconds_p95": sorted(r["model_seconds"] for r in rows)[int(len(rows) * 0.95) - 1],
        "rows": rows,
    }
    unsure = [r for r in rows if not r["rules"]]
    out["unsure_items"] = len(unsure)
    out["model_accuracy_where_rules_unsure"] = round(sum(r["model"] == r["expected"] for r in unsure) / len(unsure), 3) if unsure else None
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
