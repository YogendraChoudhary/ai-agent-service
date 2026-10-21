"""Code intent regression check, or opt-in live quality/latency evaluation (never executes generated code)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fixtures.code_intent_eval_cases import CASES
from config import settings
from src.code_intent import detect_code_intent
from src.code_validate import validate_code
from src.observability import start_trace, finish_trace
from src.models import ADVANCED_MODEL_IDS
from src.provider_errors import status_code
from src.text_processing_agent import process_user_query_stream

TASKS = {
    'python': ['Parse CSV records', 'Sort records by date', 'Group expenses by category', 'Validate email addresses',
               'Merge sorted lists', 'Count words in text', 'Deduplicate objects by ID', 'Paginate records',
               'Parse ISO dates', 'Compute moving averages'],
    'typescript': ['Build a React menu component', 'Validate form input', 'Sort records by date', 'Debounce a callback',
                   'Fetch paginated API results', 'Group records by category', 'Merge configuration objects',
                   'Deduplicate records by ID', 'Build an accessible React dialog', 'Parse CSV input'],
    'html': ['Build an accessible contact form', 'Build a responsive navigation bar', 'Build a product card',
             'Build a pricing table', 'Build a login form', 'Build an FAQ page', 'Build a photo gallery',
             'Build a documentation sidebar', 'Build a newsletter signup form', 'Build a dashboard layout'],
    'sql': ['Find duplicate email addresses', 'Rank products by monthly revenue', 'Find customers without orders',
            'Compute running sales totals', 'List the latest order per customer', 'Group expenses by category',
            'Paginate users by ID', 'Find gaps in daily records', 'Calculate customer retention', 'Compute median order value'],
}


def quality_prompts():
    for language, tasks in TASKS.items():
        for task in tasks:
            for constraint in ('Handle empty inputs.', 'Include a short usage example in code.',
                               'Explain edge cases using concise code comments.'):
                yield language, f'{task} in {language}. {constraint} Return complete code only.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make paid provider calls using configured keys')
    parser.add_argument('--providers', nargs='+', choices=['claude', 'openai', 'gemini'], default=['claude', 'openai', 'gemini'])
    parser.add_argument('--limit', type=int, default=0, help='Prompts per provider; 0 runs all 120')
    parser.add_argument('--out', type=Path, default=Path('local_run/code_evaluation.json'))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be non-negative')
    mismatches = [(q, expected, detect_code_intent(q)['kind']) for _, q, expected in CASES
                  if detect_code_intent(q)['kind'] != expected]
    print(f'Intent corpus: {len(CASES)} cases, {len(mismatches)} mismatches')
    if mismatches:
        print(json.dumps(mismatches, indent=2))
        return 1
    if not args.live:
        return 0
    keys = {'claude': settings.default_anthropic_api_key, 'openai': settings.default_openai_api_key,
            'gemini': settings.default_gemini_api_key}
    rows = []
    for provider in args.providers:
        if not keys[provider]:
            rows.append({'provider': provider, 'error': 'No configured key'})
            continue
        prompts = list(quality_prompts())
        for i, (language, prompt) in enumerate(prompts[:args.limit or None]):
            request_id = f'eval-code-{provider}-{i}'
            start_trace(request_id, 'code_evaluation', task='code')
            started, first, chunks, failure = time.perf_counter(), None, [], None
            try:
                for chunk in process_user_query_stream(prompt, provider, keys[provider], request_id=request_id,
                                                      force_agent_type='code_writer', is_byok=True):
                    if isinstance(chunk, str):
                        first = first if first is not None else time.perf_counter() - started
                        chunks.append(chunk)
                    elif isinstance(chunk, dict) and '__content_replace__' in chunk:
                        chunks = [chunk['__content_replace__']]
            except Exception as error:
                failure = error
            answer = ''.join(chunks)
            summary = finish_trace(request_id, 'error' if failure else 'ok') or {}
            rows.append(dict(provider=provider, model=ADVANCED_MODEL_IDS[provider], language=language, prompt=prompt,
                             content=answer, errors=validate_code(answer), ttft=first,
                             seconds=round(time.perf_counter() - started, 3), error=str(failure) if failure else None,
                             tokens_in=summary.get('tokens_in'), tokens_out=summary.get('tokens_out'),
                             est_cost_usd=summary.get('est_cost_usd')))
            # Save each result so an interrupted paid evaluation loses no completed work.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
            if failure and status_code(failure) in (400, 401, 403, 404):
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} results to {args.out}')
    return int(any(row.get('error') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
