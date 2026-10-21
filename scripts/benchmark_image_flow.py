"""
Image router evaluation (computed, nothing hardcoded).

Reports, on the labelled corpus in tests/fixtures/image_intent_eval_cases.py:
  * routing precision / recall and every misroute, per group
  * how the previous keyword router (determine_agent_type) handled the same image requests
  * parameter extraction accuracy on a small labelled prompt set
  * regex decision time and MIME sniff / transcode time

What this does NOT measure: provider latency, LLM latency, or end-to-end time to image. Those
need live API keys (see scripts/image_smoke_test.py). The numbers here describe routing quality
only, on a hand-written corpus, so they are an upper bound on real-world accuracy.

Usage: python scripts/benchmark_image_flow.py
"""

import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from PIL import Image

from tests.fixtures.image_intent_eval_cases import CASES
from src.image_gen import ImageParams
from src.intent import detect_image_intent
from src.s3_service import convert_image_format, sniff_image_bytes
from src.text_processing_agent import determine_agent_type

# Prompt -> expected settings. Includes phrases that used to set the wrong value.
PARAM_CASES = [
    ("a sunset over mountains", {"aspect_ratio": "1:1", "quality": "medium", "format": "jpeg"}),
    ("a landscape image of mountains", {"aspect_ratio": "16:9"}),
    ("draw it in portrait", {"aspect_ratio": "9:16"}),
    ("an instagram story background", {"aspect_ratio": "9:16"}),
    ("a widescreen wallpaper", {"aspect_ratio": "16:9"}),
    ("a children's story illustration of a fox", {"aspect_ratio": "1:1"}),
    ("wide-angle shot of a mountain, cinematic", {"aspect_ratio": "1:1", "quality": "medium"}),
    ("a detailed masterpiece of a castle", {"quality": "medium"}),
    ("high quality render of a car", {"quality": "high"}),
    ("a draft of the layout", {"quality": "low"}),
    ("a poster in 4k", {"resolution": "4K"}),
    ("an app icon with transparent background", {"format": "png", "transparent_bg": True}),
    ("a logo in webp format", {"format": "webp"}),
    ("a banner in 16:9", {"aspect_ratio": "16:9"}),
    ("a square avatar", {"aspect_ratio": "1:1"}),
]

# Image types with no image tool in the legacy path: a request routed there cannot produce a picture.
LEGACY_TEXT_ONLY = {
    "grammar", "simplify", "completion", "emojify", "translation", "table", "list",
    "summarize", "tone", "make_longer", "make_shorter", "enhance",
}


def evaluate_router():
    by_group = defaultdict(lambda: {"total": 0, "correct": 0})
    misroutes, misses = [], []
    tp = fp = fn = tn = 0
    legacy_text_only = legacy_image_cases = 0

    for group, query, has_ref, has_prev, expected in CASES:
        got = detect_image_intent(query, has_reference=has_ref, has_previous_image=has_prev)
        by_group[group]["total"] += 1
        if got["kind"] == expected:
            by_group[group]["correct"] += 1

        wants_image = expected != "none"
        routed_image = got["kind"] != "none"
        if wants_image and routed_image and got["kind"] == expected:
            tp += 1
        elif wants_image and not (routed_image and got["kind"] == expected):
            fn += 1
            misses.append({"group": group, "query": query, "expected": expected, "got": got["kind"], "rule": got["rule"]})
        elif not wants_image and routed_image:
            fp += 1
            misroutes.append({"group": group, "query": query, "got": got["kind"], "rule": got["rule"]})
        else:
            tn += 1

        if wants_image and not has_prev:  # legacy router had no notion of previous images
            legacy_image_cases += 1
            agent_type, _ = determine_agent_type(query.lower())
            if agent_type in LEGACY_TEXT_ONLY:
                legacy_text_only += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "corpus_size": len(CASES),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "by_group": {g: f"{v['correct']}/{v['total']}" for g, v in sorted(by_group.items())},
        "misroutes_to_image": misroutes,
        "missed_image_requests": misses,
        "legacy_keyword_router": {
            "image_requests_without_prior_image": legacy_image_cases,
            "sent_to_a_text_only_agent": legacy_text_only,
        },
    }


def evaluate_params():
    wrong = []
    for prompt, expected in PARAM_CASES:
        got = ImageParams.from_prompt_and_overrides(prompt).to_dict()
        bad = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
        if bad:
            wrong.append({"prompt": prompt, "expected_vs_got": bad})
    return {
        "cases": len(PARAM_CASES),
        "correct": len(PARAM_CASES) - len(wrong),
        "accuracy": round((len(PARAM_CASES) - len(wrong)) / len(PARAM_CASES), 4),
        "wrong": wrong,
    }


def time_router(repeats=20):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _g, query, has_ref, has_prev, _e in CASES:
            detect_image_intent(query, has_reference=has_ref, has_previous_image=has_prev)
        samples.append((time.perf_counter() - t0) / len(CASES) * 1e6)
    return {"median_microseconds_per_query": round(statistics.median(samples), 1), "repeats": repeats}


def time_image_ops():
    img = Image.new("RGB", (1024, 1024), (120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    t0 = time.perf_counter()
    for _ in range(50):
        sniff_image_bytes(png)
    sniff_ms = (time.perf_counter() - t0) / 50 * 1000

    out = {"image": "1024x1024 PNG", "sniff_ms": round(sniff_ms, 4)}
    for fmt in ("jpeg", "webp"):
        t0 = time.perf_counter()
        for _ in range(5):
            data, _mime, _ext = convert_image_format(png, fmt)
        out[f"png_to_{fmt}_ms"] = round((time.perf_counter() - t0) / 5 * 1000, 1)
        out[f"png_to_{fmt}_bytes"] = len(data)
    out["png_bytes"] = len(png)
    return out


def main():
    results = {
        "note": "Routing quality on a hand-written corpus. Excludes provider/LLM latency.",
        "router": evaluate_router(),
        "params": evaluate_params(),
        "router_decision_time": time_router(),
        "image_ops": time_image_ops(),
    }
    out_path = root_dir / "benchmarks" / "benchmark_image_router.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
