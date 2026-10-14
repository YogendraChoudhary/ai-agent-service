# Production Setup & Architecture Guide for FastAPI Agent API

A high-performance asynchronous AI agent service built with **FastAPI**, **Uvicorn**, **Agno**, and **Redis** for HTML editor integration (eddyter).

The service provides multi-provider LLM inference (OpenAI, Anthropic Claude, Grok xAI, Google Gemini, DeepSeek), Server-Sent Events (SSE) streaming, low-latency keystroke autocomplete, concurrent web research tools, and credit reservation/settlement tracking.

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

## 7. Model Routing Architecture

The agent automatically routes operations across two model tiers to minimize latency and token cost without sacrificing quality:

1. **Basic / Fast Tier (`BASIC_MODEL_IDS`)**:
   - Models: `gpt-4o-mini`, `gemini-3.5-flash-lite`, `grok-non-reasoning`, `claude-sonnet-5`.
   - Used for deterministic tasks: `grammar`, `simplify`, `completion`, `emojify`, `table`, `list`, `summarize`, `tone`, `make_shorter`, and `autocomplete`.
   - Tuned with low temperatures (`0.0`–`0.3`) for deterministic formatting and low TTFT (<1s).
2. **Advanced / Flagship Tier (`ADVANCED_MODEL_IDS`)**:
   - Models: `gpt-5.2`, `gemini-3.8-flash` (thinking budget tuned), `claude-opus-5`, `grok-4.6`.
   - Used for complex generative tasks: `smart_global`, `enhance`, `make_longer`, and deep web research agents.

---

## 8. Security Guardrails & Prompt Injection Firewall

All untrusted external data retrieved via web tools (`read_web_page`, `read_multiple_web_pages`, `search`, `fallback_search`) passes through a dedicated sub-millisecond security sanitizer (`src/security_guardrails.py`) before entering the LLM context:

1. **Zero-Width Character Scrubber**: Eliminates invisible Unicode and bidirectional override codes used in prompt-hiding attacks.
2. **Tokenizer Delimiter Defanger**: Defangs model-control tokens (`<|im_start|>`, `[INST]`, `<<SYS>>`, `### System:`) into harmless escaped text.
3. **Instruction Override Defanger**: Neutralizes indirect prompt injection directives (e.g. `ignore previous instructions`, `DAN mode`, `exfiltrate data`) while preserving factual page content.
4. **Sandboxed XML Isolation**: Fences external web content inside `<untrusted_web_content>` tags, reinforced by system prompt rules prohibiting the agent from executing instructions found in external data.
5. **Sub-Millisecond Performance**: Regex pre-filtering executes in **< 0.8 ms** per 24,000-character page, adding zero perceivable latency.
