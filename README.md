# Production Setup & Architecture Guide for FastAPI Agent API

A high-performance asynchronous AI agent service built with **FastAPI**, **Uvicorn**, **Agno**, and **Redis** for HTML editor integration (eddyter).

The service provides multi-provider LLM inference (OpenAI, Anthropic Claude, Grok xAI, Google Gemini, DeepSeek), Server-Sent Events (SSE) streaming, low-latency keystroke autocomplete, concurrent web research tools, and credit reservation/settlement tracking.

## Project layout

```text
api.py               Application entry point
config.py            Application settings
src/                 Service implementation
  blog/              Visually rich blog generation (outline -> sections -> lint)
  longform/          Structure-preserving refinement of long documents (freeze -> chunk -> refine -> verify -> restore),
                     plus the Markdown block parser, ordered parallel runner and model call the blog modules share
tests/
  blog/              Blog generation tests (a scripted model stands in for the providers)
  code/              Code generation and validation tests
  core/              Observability, latency, and security tests
  email/             Email template tests
  images/            Image generation and routing tests
  longform/          Document refinement, Markdown parser and ordered runner tests (a scripted model stands in for the providers)
  routing/           Request routing, classification, and fallback tests
  fixtures/          Shared labelled evaluation cases
  conftest.py        Shared test guards
scripts/             Local tools and opt-in live checks
benchmarks/          Saved benchmark reports
docs/                API documentation, plans, and benchmark notes
```

Install test dependencies with `python -m pip install -r requirements-dev.txt`, then run `python -m pytest` from the
project root. Test discovery is limited to `tests/`; the shared guards block real Redis access.
Live provider checks remain opt-in scripts. Run one area with `python -m pytest tests/code`,
or run the code intent corpus with `python scripts/evaluate_code.py`.
The latency runner is `python scripts/benchmark_suite.py <phase>`; it writes reports to `benchmarks/`.

---

## 1. Environment Configuration

The application centralizes configuration using `pydantic-settings` via `config.py`. Create a `.env` file in the project root:

```bash
# ==============================================================================
# Redis (Session management, credit tracking, license validation)
# ==============================================================================
REDIS_URL=redis://127.0.0.1:6379/0

# ==============================================================================
# Default Platform AI API Keys (Used when user does not provide BYOK)
# ==============================================================================
DEFAULT_OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEFAULT_GROK_API_KEY=your_grok_api_key_here
DEFAULT_GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_DEEPSEEK_API_KEY=your_deepseek_api_key_here

# ==============================================================================
# Backend Webhooks & Security
# ==============================================================================
BACKEND_WEBHOOK_URL=https://api.yourdomain.com/api/agent/webhook
WEBHOOK_SECRET=your_webhook_hmac_secret_here
# Enable only after the receiver atomically deduplicates Idempotency-Key.
BILLING_WEBHOOK_IDEMPOTENT=false
API_KEY_ENCRYPTION_SECRET=your_32_byte_encryption_secret_here

# ==============================================================================
# CDN / S3 Storage (Generated Images & Assets)
# ==============================================================================
CDN_URL=https://cdn.yourdomain.com
AWS_S3_BUCKET=your-s3-bucket-name
AWS_REGION=us-east-1

# ==============================================================================
# Credit Precheck & Reservation Settings
# ==============================================================================
CREDIT_PRECHECK_ENABLED=true
CREDIT_SETTLE_DELAY_SECONDS=3.0
MAX_RESERVATION_PER_REQUEST=100
MAX_REFERENCE_IMAGES=3
```

Credit billing uses a persistent Redis queue and retains credit holds after an unacknowledged delivery.
Automatic payment retries stay disabled until receiver-side duplicate protection is confirmed. See
[billing delivery and cancellation](docs/BILLING_DELIVERY.md) for the contract, deployment requirements and tests.

---

## 2. Installation & Virtual Environment

Navigate to the project directory and set up a Python virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Key Libraries in `requirements.txt`:**
> - `fastapi` & `uvicorn[standard]`: ASGI high-concurrency web server.
> - `agno==2.8.5`: Maintained successor to `phidata` orchestrating agent flows.
> - `pydantic-settings`: Type-safe startup environment configuration.
> - `ddgs` & `beautifulsoup4`: Web search and HTML scraping engine.
> - `redis` & `cryptography`: State caching and key encryption.

---

## 3. Running in Development

Start the development server with live reload enabled:

```bash
# Run directly via Uvicorn CLI
uvicorn api:app --host 127.0.0.1 --port 5000 --reload

# Or run the entrypoint script
python api.py
```

Once running:
- **Interactive OpenAPI (Swagger) UI**: [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:5000/redoc](http://127.0.0.1:5000/redoc)
- **Health Check Endpoint**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

---

## 4. Production Deployment with Systemd

For Linux servers (e.g., Ubuntu), manage the FastAPI application as a systemd service using **Gunicorn with Uvicorn worker class** (or native Uvicorn).

### Service File: `/etc/systemd/system/agent.service`

```ini
[Unit]
Description=FastAPI Agent Service for Eddyter
After=network.target

[Service]
Type=simple
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/agent
EnvironmentFile=/home/ubuntu/agent/.env

# Gunicorn managing async Uvicorn workers
ExecStart=/home/ubuntu/.venvs/aienv/bin/gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:5000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info

Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

## 5. Enable, Start, and Monitor Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable agent

# Start or restart the service
sudo systemctl start agent
sudo systemctl restart agent

# Check service status
sudo systemctl status agent

# View live streaming application logs
sudo journalctl -u agent -f
```

> **📌 Note on Existing Servers (`flaskagent` vs `agent`):**
> - If updating an existing production server that was previously running `flaskagent.service`, you have two choices:
>   1. **Keep the existing service name (Zero-friction)**: Update `/etc/systemd/system/flaskagent.service` with the new `ExecStart` (`--worker-class uvicorn.workers.UvicornWorker`), reload systemd (`sudo systemctl daemon-reload`), and restart (`sudo systemctl restart flaskagent`).
>   2. **Migrate to `agent.service`**: Disable the old service and start the new one:
>      ```bash
>      sudo systemctl stop flaskagent && sudo systemctl disable flaskagent
>      sudo systemctl enable --now agent
>      ```

---

## 6. Core API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health probe checking API status, Redis connectivity, and credit system state. |
| `POST` | `/agent/process` | Synchronous endpoint for single-turn text processing and formatting. |
| `POST` | `/agent/stream` | Server-Sent Events (SSE) streaming endpoint for real-time text generation and tool executions. |
| `POST` | `/agent/autocomplete` | Low-latency (<1s) keystroke continuation engine with client connection pooling. |
| `POST` | `/agent/abort` | Cancels an in-flight stream generator using client request ID. |

---

## 7a. Which agent handles a request

The declared `type` field decides the agent, through the fixed table `AGENT_FOR_REQUEST_TYPE`
(`src/text_processing_agent.py`). Nothing is inferred from the user's text: a `simplify` request that
contains the words "fix grammar" still runs the simplify agent.

| `type` | Agent | | `type` | Agent |
|---|---|---|---|---|
| `fix_grammar` | `grammar` | | `make_shorter` | `make_shorter` |
| `highlight_key_points` | `key_point_highlighter` | | | |
| `simplify` | `simplify` | | `enhance_with_ai` | `enhance` |
| `complete` | `completion` | | `change_tone` | `tone` |
| `emojify` | `emojify` | | `generate_email_template` | `email_template` |
| `translate` | `translation` | | `extract_image_text` | `text_extraction` |
| `make_longer` | `make_longer` | | **`general`** | **`smart_global`** (default) |
| `generate_code` | `code_writer` | | `explain_code` | `code_explainer` |
| `generate_blog` | `blog_writer` | | | |

`general` defaults to the free-form agent, which plans its own tools from the user's words
(`src/request_plan.py`: image count, web access, model tier) and may take the direct image route.
Tables, lists and summaries requested through `general` are handled by that agent; the dedicated `table`,
`list` and `summarize` agents have no API `type` and are no longer reachable through the API.

`highlight_key_points` accepts a paragraph/small text portion (up to 10,000 characters). It produces a concise,
paraphrased summary: one lead sentence followed by supporting bullet points, preserving the source's figures,
categories and qualifiers. See [the frontend handoff](docs/KEY_POINTS_API_HANDOFF.md) or run
`python scripts/demo_key_points.py` for a provider-free example (`--live` uses the configured OpenAI key).

`enhance_with_ai` can additionally suggest an editable graph from the original text or Markdown table when
`CHART_SUGGESTIONS_ENABLED=true` (the default; an Enhance instruction asking for a chart always gets the chart or a "can't chart this" event). Suggestions are separate from the improved prose
and require the writer to choose Insert. The graph stores exact decimal values separately from its appearance;
`POST /agent/charts/edit` changes only explicit fields, and `POST /agent/charts/preview` renders a saved graph.
Type/color changes preserve the full dataset. See [the frontend handoff](docs/GRAPH_API_HANDOFF.md) for the SSE/JSON
contracts and editor changes. Run `python scripts/demo_editable_graph.py` for a provider-free demonstration.

Document-aware questions and drafting are enabled by default (`DOCUMENT_CONTEXT_ENABLED=true`). Send
`document_context` with `document_action="ask"` or `"draft"` on `type="general"`. Ask answers in chat; Draft
returns only a new suggested passage. Coverage and billing metadata distinguish full context, partial drafting
context and requests that need a smaller source without running a model. See the
[context handoff](docs/DOCUMENT_CONTEXT_API_HANDOFF.md) and run `python scripts/demo_document_context.py`
for a provider-free example. Explicit `document_scope="focused"|"selection"` enables local retrieval;
`document_analysis="summary"|"contradictions"` runs bounded full-source analysis. Explicit context Edit uses
untargeted prose as a style reference, and `suggest_document_intent=true` returns advisory action suggestions.
The handoff describes limits, staged analysis billing and editor integration.
Live synthetic accuracy, Edit preservation and streamed latency results are in
[the provider evaluation report](docs/DOCUMENT_CONTEXT_LIVE_RESULTS.md); `scripts/evaluate_document_context.py`
repeats them with the configured provider keys.

To edit existing parts of a document without a selection, send `document_text` with the full current Markdown/plain text
and use `query` for the editing instruction (`general` or `enhance_with_ai`). The backend locates the relevant
paragraphs, returns verified patches and preserves every untargeted source region exactly. Proposals require
frontend review and stale-snapshot checks; `/agent/documents/apply` replays accepted patches without model calls.
See [the document-editing handoff](docs/DOCUMENT_EDIT_API_HANDOFF.md) and run `python scripts/demo_document_edit.py`
for a provider-free example, or add `--live` to test the configured OpenAI model on synthetic documents.
New clients can also use `document_context` with `document_action="edit"` when the new feature is enabled;
this delegates to the same patch engine. Keep the existing `complete` selection-replacement contract separate from Draft insertion.
`determine_agent_type` (keyword matching) remains only for callers that name no agent.

`generate_blog` is not one agent run: `src/blog` plans the post in code (length, sections, tables, quotes, images), has the
model write a validated outline, drafts the sections in parallel while the pictures generate, and lints each part before
it is streamed. It is on by default (`BLOG_AI_ENABLED`); the contract, events and flags are in
[the API handoff](docs/API_DOCUMENTATION.md#visually-rich-blog-generation-generate_blog). Try it locally with
`python scripts/run_local_server.py` then `python scripts/local_test_requests.py` (the blog scenarios), or compare providers
with `python scripts/evaluate_blog.py --live`. To read saved posts as a reader would, `python scripts/render_blog_preview.py`
writes one self-contained HTML page; `--compare LABEL OLD.md NEW.md` puts two generations of a brief side by side.

`fix_grammar`, `simplify`, `change_tone` and `enhance_with_ai` keep their single model call for ordinary text (the chunked path needs Python 3.11 or newer; on an older interpreter it logs why and the single call is used). A long or
structured document (600+ words, or one with a table, quote, picture or code block) can instead go through `src/longform`:
tables, quotes, pictures, code and links are replaced by locked tokens, the prose is refined in ~1,000-word chunks in
parallel, every chunk is checked in code (tokens, headings, lists, paragraphs, length), repaired once if it broke a rule,
and restored exactly as written if it still does. It is on by default (`LONGFORM_REFINE_ENABLED`); the contract, events and
flags are in [the API handoff](docs/API_DOCUMENTATION.md#long-document-refinement-enhance_with_ai-fix_grammar-change_tone-simplify).
Check it for free with `python scripts/evaluate_longform.py` (10 documents x 4 operations, scripted models), or compare
providers on real documents with `python scripts/evaluate_longform.py --live --corpus docs`.

Optional code intent routing, task defaults, provider fallback, and code validation ship disabled.
See [the API handoff](docs/API_DOCUMENTATION.md#code-generation-and-task-routing) for flags, `model_source`,
BYOK restrictions and the streaming `content_replace` contract. Image routing takes priority.

## 7. Model Routing Architecture

The agent automatically routes operations across two model tiers to minimize latency and token cost without sacrificing quality:

1. **Basic / Fast Tier (`BASIC_MODEL_IDS`)**:
   - Models: `gpt-4o-mini`, `gemini-3.5-flash-lite`, `grok-non-reasoning`, `claude-sonnet-5-5`.
   - Used for deterministic tasks: `grammar`, `simplify`, `completion`, `emojify`, `table`, `list`, `summarize`, `tone`, `make_shorter`, and `autocomplete`.
   - Tuned with low temperatures (`0.0`–`0.3`) for deterministic formatting and low TTFT (<1s).
2. **Advanced / Flagship Tier (`ADVANCED_MODEL_IDS`)**:
   - Models: `gpt-6.1-terra` (`gpt-5.2` for the blog and code agents), `gemini-3.8-flash` (thinking budget tuned), `claude-opus-5-5`, `grok-4.6`.
   - Used for complex generative tasks: `smart_global`, `enhance`, `make_longer`, and deep web research agents.

---

## 8. Security Guardrails & Prompt Injection Firewall

All untrusted external data retrieved via web tools (`read_web_page`, `read_multiple_web_pages`, `search`, `fallback_search`) passes through a dedicated sub-millisecond security sanitizer (`src/security_guardrails.py`) before entering the LLM context:

1. **Zero-Width Character Scrubber**: Eliminates invisible Unicode and bidirectional override codes used in prompt-hiding attacks.
2. **Tokenizer Delimiter Defanger**: Defangs model-control tokens (`<|im_start|>`, `[INST]`, `<<SYS>>`, `### System:`) into harmless escaped text.
3. **Instruction Override Defanger**: Neutralizes indirect prompt injection directives (e.g. `ignore previous instructions`, `DAN mode`, `exfiltrate data`) while preserving factual page content.
4. **Sandboxed XML Isolation**: Fences external web content inside `<untrusted_web_content>` tags, reinforced by system prompt rules prohibiting the agent from executing instructions found in external data.
5. **Sub-Millisecond Performance**: Regex pre-filtering executes in **< 0.8 ms** per 24,000-character page, adding zero perceivable latency.
