"""Paid live eval of editable screenshot recreation. Use the configured provider key.

Run python scripts/design_fidelity_eval.py --help for options. The server and
billing backend are bypassed, while the production image adapter and harness run.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from config import settings
from src.design_canvas.links import check_links
from src.design_canvas.reference import ReferenceSpec, check_fidelity
from src.design_canvas.service import execute_design_canvas_harness
from src.models import design_model_id
from src.observability import get_trace, start_trace
from src.provider_work import run_provider_call
from src.text_processing_agent import _design_measure_image, _design_reference_image, _run_design_json_completion

FIXTURES = ROOT / "tests/design/fixtures/reference"
QUERY = "Recreate this screenshot faithfully as an editable design."


def render_runs(results, output):
    from playwright.sync_api import sync_playwright

    from scripts.design_reference_render import render_html

    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        for result in results:
            if "ast" not in result:
                continue
            image_path = FIXTURES / result["fixture"]
            with Image.open(image_path) as ref:
                width = min(ref.width, 1200)
                reference = ref.convert("RGB")
                reference.thumbnail((width, ref.height))
            page.set_viewport_size({"width": width, "height": 900})
            html = output / f'{image_path.stem}_{result["run"]}.html'
            html.write_text(render_html(result["ast"], width), encoding="utf-8")
            page.goto(html.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            render = html.with_suffix(".png")
            page.locator("#canvas").screenshot(path=render)
            with Image.open(render) as generated:
                pair = Image.new("RGB", (width + generated.width, max(reference.height, generated.height)), "white")
                pair.paste(reference, (0, 0))
                pair.paste(generated, (width, 0))
                pair.save(html.with_name(html.stem + "_comparison.png"))


async def evaluate(args, images, key):
    async def one(path, run):
        expected = json.loads(path.with_suffix(".expect.json").read_text())
        spec = ReferenceSpec.model_validate(expected["spec"])
        reference = _design_reference_image(path.read_bytes())
        report = {}
        request_id = f"fidelity-{args.label}-{path.stem}-{run}"
        start_trace(request_id, "design_fidelity_eval")
        started = time.perf_counter()

        class Runner:
            async def run_json_completion(self, messages, image_bytes=None, temperature=0.2):
                return await run_provider_call(
                    _run_design_json_completion, messages, image_bytes, temperature, args.provider, key, request_id,
                )

        result = {"fixture": path.name, "run": run, "report": report}
        try:
            ast = await execute_design_canvas_harness(Runner(), QUERY, "create_from_vision", reference, report=report,
                                                      measure_image_bytes=_design_measure_image(path.read_bytes()))
            report["link_warnings"] = check_links(ast, QUERY)
            result["ast"] = ast.model_dump()
            scores, findings = check_fidelity(result["ast"], spec, expected_accents=expected.get("accents", ()))
            result.update(scores=scores, findings=findings)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            result["error"] = f"{type(exc).__name__} (HTTP {status})" if status else type(exc).__name__
        result["seconds"] = round(time.perf_counter() - started, 2)
        trace = get_trace(request_id)
        result["usage"] = {k: trace.attrs.get(k) for k in ("tokens_in", "tokens_out", "est_cost_usd")} if trace else {}
        print(f'{path.stem} #{run} {result["seconds"]}s {result.get("scores", result.get("error"))}', flush=True)
        return result

    return [await one(path, run) for path in images for run in range(1, args.runs + 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="*", help="Fixture names without extension. Default: all.")
    parser.add_argument("--provider", choices=("openai", "claude", "gemini"), default="claude")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--label", default="live")
    parser.add_argument("--render", action="store_true", help="Save HTML and side-by-side Playwright screenshots.")
    args = parser.parse_args()
    images = sorted(path for path in FIXTURES.iterdir() if path.suffix in (".webp", ".png", ".jpg")
                    and (not args.fixtures or path.stem in args.fixtures))
    if args.runs < 1 or not images or set(args.fixtures) - {path.stem for path in images}:
        parser.error("Provide known fixture names and a positive run count.")
    key = {"openai": settings.default_openai_api_key, "claude": settings.default_anthropic_api_key,
           "gemini": settings.default_gemini_api_key}[args.provider]
    if not key:
        parser.error(f"No configured API key for {args.provider}.")
    results = asyncio.run(evaluate(args, images, key))
    summaries = {}
    for image in images:
        successful = [row for row in results if row["fixture"] == image.name and "scores" in row]
        if successful:
            recalls = [row["scores"]["text_recall"] for row in successful if row["scores"]["text_recall"] is not None]
            accent_coverages = [row["scores"]["accent_coverage"] for row in successful
                                if row["scores"]["accent_coverage"] is not None]
            summaries[image.stem] = {
                "completed_runs": len(successful),
                "median_seconds": median(row["seconds"] for row in successful),
                "median_text_recall": median(recalls) if recalls else None,
                "layout_matches": sum(row["scores"]["layout_matches"] for row in successful),
                "layout_regions": sum(row["scores"]["layout_regions"] for row in successful),
                "median_accent_coverage": median(accent_coverages) if accent_coverages else None,
                "fonts_set_runs": sum(row["scores"]["fonts_set"] for row in successful),
            }
    out = ROOT / "local_run/design_fidelity" / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label, "provider": args.provider, "model": design_model_id(args.provider),
        "schema_v11_enabled": settings.design_schema_v11_enabled, "two_pass": settings.design_vision_two_pass,
        "vision_openai_effort": settings.design_vision_openai_effort,
        "vision_output_cap": settings.design_vision_max_output_tokens,
        "summary": summaries, "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    if args.render:
        render_runs(results, out.with_suffix(""))
    return int(any("error" in result for result in results))


if __name__ == "__main__":
    sys.exit(main())
