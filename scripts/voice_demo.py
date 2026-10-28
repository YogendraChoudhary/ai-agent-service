"""
Start voice dictation on its own and open the test page in your browser.

  python scripts/voice_demo.py                 # Whisper small + Gemini, opens http://127.0.0.1:8000/v1/voice/demo
  python scripts/voice_demo.py --gpt           # format with GPT (gpt-4o-mini) instead of Gemini
  python scripts/voice_demo.py --whisper base  # faster speech recognition, less accurate
  python scripts/voice_demo.py --always-model  # send every utterance to the model (default: plain sentences skip it)
  python scripts/voice_demo.py --no-open       # do not open a browser

Only the /v1/voice routes are served (no Redis, credits or other endpoints), on this machine only. Press Ctrl+C to stop.
"""

import argparse
import logging
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                       # config.py finds ../.env relative to here


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpt", action="store_true", help="format with OpenAI instead of Gemini")
    parser.add_argument("--whisper", help="Whisper model: tiny, base, small (default), ...")
    parser.add_argument("--always-model", action="store_true", help="do not use the fast path for plain sentences")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"Port {args.port} is already in use (another server, or an earlier copy of this one).\n"
                  f"Stop it, or run with --port {args.port + 1}.")
            return 1

    # Environment first: config.py reads it once at import. An explicit value here beats .env.
    os.environ["VOICE_FLOW_ENABLED"] = "true"
    if args.gpt:
        os.environ["VOICE_FORMAT_PROVIDER"] = "openai"
    if args.whisper:
        os.environ["VOICE_WHISPER_MODEL"] = args.whisper
    if args.always_model:
        os.environ["VOICE_LLM_MODE"] = "always"

    import uvicorn
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from config import settings
    from src.voice_flow.router import get_service, router, warm_up

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    url = f"http://127.0.0.1:{args.port}/v1/voice/demo"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        print(f"\nLoading Whisper '{settings.voice_whisper_model}' and checking the {settings.voice_format_provider} formatter ...", flush=True)
        await warm_up()
        service = get_service()
        problem = "" if service.breaker.allow() else f"\n  !! formatter unavailable ({service.breaker.last_reason}): plain rule-based formatting will be used"
        print(f"\n  Ready. Open: {url}\n  Whisper {settings.voice_whisper_model} | formatter {settings.voice_format_provider}:{getattr(service.backend, 'model', '')}"
              f" | mode {service.mode}{problem}\n  Ctrl+C to stop.\n", flush=True)
        if not args.no_open:
            threading.Timer(0.3, webbrowser.open, args=(url,)).start()
        yield

    app = FastAPI(title="Voice dictation test", lifespan=lifespan)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def home() -> RedirectResponse:
        return RedirectResponse("/v1/voice/demo")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
