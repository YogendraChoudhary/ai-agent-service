# Eddyter AI Agent Service — System Architecture & Developer Manual (`agent.md`)

Welcome to the definitive architectural, operational, and integration manual for the **Eddyter AI Agent Service**. This document provides an exhaustive, end-to-end breakdown of how Eddyter’s AI engine is built, how its internal subsystems communicate, how requests transition from user keystrokes into rich document structures, and how third-party AI providers, Redis, S3/CDN, and webhook billing are orchestrated.

---

## Table of Contents

1. [Executive Summary & System Vision](#1-executive-summary--system-vision)
2. [End-to-End System Topology](#2-end-to-end-system-topology)
3. [Repository Layout & Directory Inventory](#3-repository-layout--directory-inventory)
4. [Model Routing & Provider Architecture](#4-model-routing--provider-architecture)
5. [The Core Operational Modes](#5-the-core-operational-modes)
   - [Mode 1: Text Operations & Quick Edits](#mode-1-text-operations--quick-edits)
   - [Mode 2: Structure-Preserving Long Document Refinement (`src/longform`)](#mode-2-structure-preserving-long-document-refinement-srclongform)
   - [Mode 3: Visually Rich Editorial Blog Generation (`src/blog`)](#mode-3-visually-rich-editorial-blog-generation-srcblog)
   - [Mode 4: AI Email Template Engine (`src/email_template`)](#mode-4-ai-email-template-engine-srcemail_template)
   - [Mode 5: Code Generation, Validation & Repair (`src/code_*`)](#mode-5-code-generation-validation--repair-srccode_)
   - [Mode 6: Low-Latency Keystroke Autocomplete](#mode-6-low-latency-keystroke-autocomplete)
   - [Mode 7: Multimodal Vision & Direct Image Generation (`src/image_gen.py`, `src/intent.py`)](#mode-7-multimodal-vision--direct-image-generation)
6. [Deterministic Request Planning (`src/request_plan.py`)](#6-deterministic-request-planning-srcrequest_planpy)
7. [Credit Reservation, Settlement & License Engine (`src/credit_manager.py`, `src/redis_manager.py`)](#7-credit-reservation-settlement--license-engine)
8. [Security Guardrails & Prompt Injection Firewall (`src/security_guardrails.py`, `src/encryption.py`)](#8-security-guardrails--prompt-injection-firewall)
9. [Autonomous Web Tools & SSRF-Protected Research Engine](#9-autonomous-web-tools--ssrf-protected-research-engine)
10. [Observability, Tracing & Performance Metrics (`src/observability.py`)](#10-observability-tracing--performance-metrics-srcobservabilitypy)
11. [API Surface & Streaming Protocol Reference](#11-api-surface--streaming-protocol-reference)
12. [Configuration Inventory (`config.py` & `.env`)](#12-configuration-inventory-configpy--env)
13. [Local Development, Testing & Production Deployment](#13-local-development-testing--production-deployment)

---

## 1. Executive Summary & System Vision

**Eddyter** is a modern, next-generation rich content and block editor (built upon Facebook's Lexical framework). Unlike conventional editors that simply paste raw LLM text into a textarea, Eddyter features a native, asynchronous AI agent service designed around six foundational engineering principles:

1. **Code Owns Structure, Models Own Prose**: The LLM writes text, titles, and explanations. Code (not the model) decides headings, visual spacing, table schemas, image placements, and token limits.
2. **Zero Layout Destruction**: When enhancing, editing, or translating complex existing documents, tables, code blocks, blockquotes, URLs, and images are byte-frozen before reaching the LLM and mathematically verified upon return.
3. **Deterministic Planning over Stochastic Tool Use**: The agent service extracts user intent using compiled, start-anchored regex and stateful guards before LLM invocation. Images and web tools are granted only when necessary, preventing hallucinated costs and latency bloat.
4. **Atomic Multi-Tenant Credit Safety**: Users never overspend or starve balances during long async runs. A Redis Lua-based reservation and settlement system holds credits during processing and settles only on success.
5. **Zero-Latency Multi-Provider Resilience**: Seamless fallback between OpenAI, Anthropic Claude, xAI Grok, Google Gemini, and DeepSeek, paired with model tiering (fast vs. advanced) based on cognitive complexity.
6. **Strict Output Formatting**: Never emit unrequested conversational preamble (e.g., *"Here is your article..."*). Clean Markdown or typed JSON blocks are delivered directly to the editor canvas.

---

## 2. End-to-End System Topology

The Eddyter ecosystem operates in a three-tier architecture:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             BROWSER / CLIENT (Eddyter UI)                         │
│  - Lexical Block Editor / HtmlCanvasNode                                         │
│  - Keystroke-level Autocomplete Listener                                         │
│  - SSE Consumer (Streams Markdown chunks, handles image_ready / content_replace) │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ HTTP / SSE / REST
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             EDDYTER APPLICATION BACKEND                           │
│  - User Authentication & Session Verification                                    │
│  - Resolves BYOK (Bring-Your-Own-Key) or Platform API Keys                       │
│  - Supabase Database (Users, Workspaces, Credit Balances)                        │
│  - Webhook Receiver (`/api/agent-webhook/credit-deduction`, `/track-file`)       │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Server-to-Server Internal HTTP
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   EDDYTER AI AGENT SERVICE (This Service: FastAPI)               │
│                                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │   Request Router     │  │ Deterministic Plan   │  │   Security Firewall    │  │
│  │  (Task / Fallback)   │  │  (Images/Web/Tier)   │  │  (Zero-width, Delim)   │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └───────────┬────────────┘  │
│             │                         │                          │               │
│  ┌──────────▼─────────────────────────▼──────────────────────────▼────────────┐  │
│  │                         Agno-Powered Agent Engine                          │  │
│  │  - Single Turn Tasks (grammar, simplify, tone, enhance)                    │  │
│  │  - Mode 1: Longform Refinement (AST Freeze -> Chunk -> Parallel -> Verify) │  │
│  │  - Mode 2: Visually Rich Blog (Outline -> Parallel Sections -> Lint)       │  │
│  │  - Mode 3: Email Generation (Pydantic Typed JSON Blocks -> Validation)     │  │
│  │  - Mode 4: Code Engine (AST/Structure Balance Check -> Auto-Repair)        │  │
│  └────────────────────────────────────┬───────────────────────────────────────┘  │
│                                       │                                          │
│        ┌──────────────────────────────┼──────────────────────────────┐           │
│        ▼                              ▼                              ▼           │
│  ┌───────────┐                 ┌─────────────┐                ┌─────────────┐    │
│  │ Upstash / │                 │   AWS S3    │                │ CloudFront  │    │
│  │   Redis   │                 │   Bucket    │                │     CDN     │    │
│  │ (Credits) │                 │  (Images)   │                │  (Assets)   │    │
│  └───────────┘                 └─────────────┘                └─────────────┘    │
└────────┬──────────────────────────────┬──────────────────────────────┬───────────┘
         │                              │                              │
         ▼                              ▼                              ▼
  ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
  │ OpenAI / xAI  │              │ Anthropic     │              │ Google Gemini │
  │ (GPT-5, Grok) │              │ (Opus/Sonnet) │              │ & DeepSeek    │
  └───────────────┘              └───────────────┘              └───────────────┘
```

---

## 3. Repository Layout & Directory Inventory

The core service code lives under `agent/`:

```text
agent/
├── api.py                    # Main FastAPI application entry point, routes, SSE endpoints, exception handlers
├── config.py                 # Centralized settings via pydantic-settings; all env vars and feature flags
├── requirements.txt          # Python dependencies (fastapi, uvicorn, agno, redis, boto3, etc.)
├── package.json              # Node dependencies / local runner scripts
├── pytest.ini                # Pytest configuration
│
├── src/                      # Core business logic & agent implementations
│   ├── blog/                 # Mode 3: Visually rich editorial blog generation engine
│   │   ├── plan.py           # Quotas (tables, quotes, images) derived deterministically from length
│   │   ├── outline.py        # Outline generation, JSON parser, retry loop, streaming title extractor
│   │   ├── draft.py          # Parallel drafting of section bodies and visual elements
│   │   ├── validate.py       # Linting engine (word runs, visual density, list limits, sentence length)
│   │   ├── media.py          # Unified image placeholder injector and resolution handler
│   │   ├── prompts.py        # Byte-stable system prompts for outline & section drafting
│   │   ├── schema.py         # Pydantic schemas for blog outlines and sections
│   │   ├── service.py        # Top-level orchestration service for blog generation
│   │   └── text.py           # Cleaners and text statistics helpers
│   │
│   ├── longform/             # Mode 2: Structure-preserving long document refinement
│   │   ├── markdown_ast.py   # Markdown block boundary slicer (CommonMark + GFM tables)
│   │   ├── freeze.py         # Freezes tables, quotes, code, images, links into tokens (<frozen:...>)
│   │   ├── chunk.py          # Intelligent block chunking (~1,000 words, split at H2/H3 boundaries)
│   │   ├── refine.py         # Parallel chunk refinement engine
│   │   ├── verify.py         # Deterministic verifier (checks tokens, headings, lists, paragraphs, ratios)
│   │   ├── prompts.py        # Operation-specific prompts with strict token-preservation instructions
│   │   ├── ordered_stream.py # Concurrency queue emitting chunks in strict document order
│   │   └── runtime.py        # Longform feature flag guards and fallback delegator
│   │
│   ├── email_template/       # Mode 4: Structured email template generation engine
│   │   ├── categories.py     # Registry of supported email types (product launch, reminder, etc.)
│   │   ├── schema.py         # Pydantic schemas for strictly typed email component blocks
│   │   ├── prompts.py        # Prompts strictly enforcing valid JSON output with zero markdown wrappers
│   │   ├── service.py        # Orchestrates email generation, parallel variant generation, and repair
│   │   ├── validate.py       # Business logic validator (merge tags, link keys, button mandates)
│   │   ├── images.py         # Auto-crop, fit, and image upload pipeline for email assets
│   │   └── tone.py           # Category tone mapping and style definitions
│   │
│   ├── text_processing_agent.py # The monolithic Agno engine: agent definitions, tool setups, streaming loops
│   ├── models.py             # Single source of truth for all LLM IDs, token headroom, vision capabilities
│   ├── request_plan.py       # Deterministic regex request planner (image count, web tools gate, model tier)
│   ├── task_router.py        # Intelligent task router (code vs. writing vs. content) & fallback trees
│   ├── code_intent.py        # Start-anchored regex engine for code generation intent detection
│   ├── code_validate.py      # Structural & syntax validator for generated code (Python AST, JS/TS balance)
│   ├── credit_manager.py     # Credit reservation logic and Lua scripts for atomic precheck
│   ├── redis_manager.py      # Redis client manager, connection pooling, user license caching
│   ├── image_gen.py          # Unified image generation adapter (Gemini, OpenAI, Grok, Stability)
│   ├── s3_service.py         # AWS S3 image uploader, storage quota checker, CloudFront CDN mapper
│   ├── security_guardrails.py# Sub-millisecond prompt injection firewall & tokenizer defanger
│   ├── observability.py      # Request tracing, latency markers, token usage tracking, secret redaction
│   ├── webhook.py            # Async webhooks for credit settlement and prompt history archival
│   ├── encryption.py         # AES-256-GCM encryption for stored API keys
│   └── provider_errors.py    # Standardized error classifier across all LLM providers
│
├── docs/                     # Project architectural plans and integration handoffs
│   ├── API_DOCUMENTATION.md  # Comprehensive HTTP & SSE API reference
│   ├── EMAIL_API_HANDOFF.md  # Frontend handoff document for the Email Template Engine
│   ├── rich_content_engine_plan.md # Architectural plan for Blog & Longform systems
│   └── task_routing_code_generation_plan.md # Plan for Code Routing and Fallbacks
│
├── tests/                    # Robust test suite (300+ unit and integration tests)
│   ├── blog/                 # Blog generation and linting tests
│   ├── code/                 # Code detection, validation, and repair tests
│   ├── core/                 # Observability, credit manager, and security tests
│   ├── email/                # Email schema and edge-case tests
│   ├── images/               # Image generation and routing tests
│   ├── longform/             # Longform AST freeze, chunk, and verify tests
│   └── routing/              # Task and fallback routing tests
│
└── scripts/                  # Operational tools, benchmarks, and evaluation scripts
    ├── run_local_server.py   # Spawns mock Redis / local dev server
    ├── benchmark_suite.py    # Latency and TTFT benchmark runner
    ├── render_blog_preview.py# HTML preview renderer for generated blog Markdown
    └── render_email_preview.py# Visual renderer for generated email JSON templates
```

---

## 4. Model Routing & Provider Architecture

All AI model IDs are centralized inside [`src/models.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/models.py). No model IDs are hardcoded in prompt strings or handler logic.

### 4.1 Model Tiers

The system partitions text operations across two primary tiers to maximize speed, minimize token burn, and maintain editorial quality:

| Tier | Target Latency (TTFT) | Typical Providers / Model IDs | Use Cases |
|---|---|---|---|
| **Basic / Fast Tier** (`BASIC_MODEL_IDS`) | **< 1.0s** | `gpt-4o-mini`, `gemini-3.5-flash-lite`, `grok-4.20-non-reasoning`, `deepseek-chat`, `claude-sonnet-5` | `fix_grammar`, `simplify`, `complete`, `emojify`, `tone`, `make_shorter`, inline edits |
| **Advanced Tier** (`ADVANCED_MODEL_IDS`) | **1.5s – 3.0s** | `gpt-5.2`, `gemini-3.8-flash`, `grok-4.6`, `deepseek-reasoner`, `claude-opus-5` | `general` (free-form), `enhance_with_ai`, `make_longer`, `generate_blog`, deep research |
| **Autocomplete Tier** (`AUTOCOMPLETE_MODEL_IDS`) | **< 500ms** | `gpt-4o-mini`, `gemini-3.5-flash-lite`, `claude-haiku-4-5`, `deepseek-chat` | Keystroke-level continuation |

### 4.2 Special Provider Considerations

- **DeepSeek Tool Limitations**: `deepseek-reasoner` (R1) does not support function calling. Any agent requiring web tools or image tools automatically falls back to `deepseek-chat` (`DEEPSEEK_TOOL_MODEL_ID`).
- **Hidden Reasoning Headroom**: For reasoning models (e.g. OpenAI o-series/GPT-5, Grok 4 reasoning), output limits include hidden "thought" tokens. [`output_token_budget()`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/models.py#L84) dynamically appends `REASONING_HEADROOM_TOKENS` (4,096 tokens) so reasoning output does not truncate the visible text.
- **Blocked Models**: Legacy or problematic model snapshots (e.g., `claude-fable-5-1`) are permanently blocked by [`BLOCKED_MODELS`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/models.py#L29).

### 4.3 Task Routing & Provider Fallback Trees (`src/task_router.py`)

When the user specifies no model preference (`model_source: "default"`), Eddyter routes the request by cognitive task:
- **Code Tasks** (`generate_code`, `explain_code`): Route to **Claude** (`claude-opus-5`). Fallback tree: `claude` ➔ `openai` ➔ `gemini`.
- **Writing Tasks** (`make_longer`, `generate_email_template`, `generate_blog`): Route to **OpenAI** (`gpt-5.2`). Fallback tree: `openai` ➔ `<requested>` ➔ `claude`.
- **Content Tasks** (Standard edits): Route to **Gemini** (`gemini-3.8-flash` / `gemini-3.5-flash-lite`).

Fallback triggers only on **HTTP 429 (rate limits)**, **HTTP 5xx (outages)**, or **timeouts**, strictly **before** the first SSE token is streamed. If an error occurs after streaming starts, the error is returned to avoid output corruption.

---

## 5. The Core Operational Modes

### Mode 1: Text Operations & Quick Edits
Designed for quick, single-pass edits in the editor.
- **Operations**: `fix_grammar`, `simplify`, `complete`, `emojify`, `translate`, `make_longer`, `make_shorter`, `enhance_with_ai`, `change_tone`.
- **Execution**: Handled directly by [`src/text_processing_agent.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py). Temperature is kept strictly low (`0.0`–`0.3`) for deterministic formatting. No conversational filler is returned.

---

### Mode 2: Structure-Preserving Long Document Refinement (`src/longform`)
*Activated when `LONGFORM_REFINE_ENABLED=true` for inputs ≥ 600 words or documents containing tables, blockquotes, code, or images.*

When a user pastes a 4,000-word document with custom markdown tables, diagrams, and pull quotes, a standard LLM will often summarize, mangle table pipes, or drop image tags. The Longform engine guarantees structural preservation:

```
[User Document: 5,000 words + Tables + Images]
                         │
                         ▼
             1. AST Block Parse (markdown_ast.py)
   (Slices raw document lines by top-level block boundaries)
                         │
                         ▼
             2. Token Freeze (freeze.py)
   (Tables, quotes, images, fences, URLs replaced by <frozen:type:id>)
                         │
                         ▼
             3. Smart Chunking (chunk.py)
   (~1,000-word chunks, split strictly at H2/H3 boundaries)
                         │
                         ▼
             4. Parallel Refinement (refine.py)
   (ThreadPool concurrency = 5; ordered queue preserves sequence)
                         │
                         ▼
             5. Code Verification (verify.py)
   - Are all frozen tokens present in exact order?
   - Is heading count & level intact?
   - Are paragraph counts preserved?
   - Is output/input word count ratio within ±15%?
         │                               │
       [PASS]                          [FAIL]
         │                               │
         │                         One Repair Call
         │                         (Explains exact discrepancy)
         │                               │
         │                         [STILL FAILS?]
         │                               │
         │                         Fallback to original chunk
         │                         (Zero data loss guarantee)
         ▼                               ▼
             6. Token Defrost & SSE Emission (ordered_stream.py)
```

---

### Mode 3: Visually Rich Editorial Blog Generation (`src/blog`)
*Activated when `BLOG_AI_ENABLED=true` via request type `generate_blog`.*

Traditional AI blog generators return a wall of unformatted text. The Eddyter Blog Engine behaves like an editorial art director:

1. **Deterministic Planning ([`src/blog/plan.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/plan.py))**:
   - Calculates visual quotas based on target word count (e.g. 1,200 words ➔ minimum 1 comparison table, 3 pull quotes, 2 section images, max 220 words between visual breaks).
2. **Outline Generation & Title Streaming ([`src/blog/outline.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/outline.py))**:
   - The LLM writes a strictly typed JSON outline adhering to [`BlogOutline`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/schema.py).
   - If `BLOG_STREAM_TITLE=true`, the title and subtitle are extracted from the raw stream and emitted to the UI immediately, providing near-instant visual feedback while the outline completes.
3. **Image Placeholder Injection ([`src/blog/media.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/media.py))**:
   - Code places image markdown: `![alt](placeholder::<uuid>)`. The model never invents or writes image URLs.
   - Image generation launches in parallel with section drafting.
4. **Parallel Section Drafting ([`src/blog/draft.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/draft.py))**:
   - Up to 3 sections draft concurrently in thread pools.
5. **Linting & Visual Density Guard ([`src/blog/validate.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/blog/validate.py))**:
   - Verifies prose runs do not exceed `BLOG_MAX_PROSE_RUN` (220 words).
   - Verifies maximum 1 bullet list per section, paragraphs ≤ 4 sentences, and validates quotes against facts.
   - Failing sections receive one repair retry before being emitted with warnings.

---

### Mode 4: AI Email Template Engine (`src/email_template`)
*Mounted at `/agent/email/*`.*

Designed for professional email marketing and transactional campaigns.
- **Never returns HTML**: The agent outputs a structured JSON AST containing typed block components (`header`, `hero_banner`, `paragraph`, `button`, `social_links`, `footer`). The Eddyter frontend compiles this JSON to MJML and HTML.
- **Strict Brand Kits & Theming**: The theme (colors, fonts, border radii) is supplied by the user and echoed unchanged. The AI cannot tamper with brand guidelines.
- **Merge Tag Safety**: Only explicitly whitelisted merge tags (e.g. `{{first_name}}`, `{{company_name}}`) are permitted.
- **Link Key Decoupling**: Raw URLs are never sent to the LLM prompt. The frontend sends `{ "cta_link": "https://company.com/signup" }`. The LLM refers only to `"cta_link"`. Eddyter’s backend reconnects the actual URL upon render.
- **Asset Fitting Endpoint (`/agent/email/image`)**: Crops, resizes, and compresses user-uploaded email images, uploading them to S3 and returning a CDN link formatted for high-DPI email clients.

---

### Mode 5: Code Generation, Validation & Repair (`src/code_*`)
*Handles `generate_code` and `explain_code`.*

- **Regex & Classifier Intent Gate ([`src/code_intent.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/code_intent.py))**: Detects if a free-form `general` query is actually asking for code (e.g., *"write a react hook for localstorage"*). Start-anchored regex classifies high-confidence prompts. Ambiguous queries run through an ultra-fast classifier (~1.5s timeout).
- **Non-Execution Structural Validation ([`src/code_validate.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/code_validate.py))**:
  - **Python**: Validated via Python's native `ast.parse()`.
  - **JSON**: Validated via `json.loads()`.
  - **JS / TS / JSX / HTML / CSS**: Structural parser validates bracket balancing, unclosed strings, template literal braces `${...}`, and JSX tag closings.
- **Automated Repair Loop**: If validation fails, a repair prompt highlights the exact syntax error. In non-streaming mode, unrepairable code returns HTTP 422. In streaming mode, the `content_replace` SSE event delivers the repaired block.

---

### Mode 6: Low-Latency Keystroke Autocomplete
*Endpoint: `/agent/autocomplete`.*

- Fires as the user pauses mid-sentence in the editor.
- **Target Latency**: **< 500ms** round-trip.
- Utilizes dedicated, high-speed lightweight models (`AUTOCOMPLETE_MODEL_IDS`).
- Uses strict prompt truncation: looks back ~500 characters of editor context and continues the sentence seamlessly without echoing previous text.
- Billed at a micro-rate (0.2 credits).

---

### Mode 7: Multimodal Vision & Direct Image Generation
*Orchestrated by [`src/image_gen.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/image_gen.py) and [`src/intent.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/intent.py).*

- **Direct Image Bypass**: If a user enters *"Create a photo of a cyberpunk skyline"*, the request bypasses the text LLM entirely and routes directly to the image adapter, saving 2-4 seconds of chat LLM latency.
- **Default Image Model**: Google Gemini (`gemini-3.1-flash-lite-image` at ~$0.033 / 3.5s latency). High-resolution requests scale to `gemini-hires` or OpenAI `gpt-image-1`.
- **Permanent Storage & CDN**: Image bytes or temporary URLs are downloaded, processed via Pillow, and uploaded to AWS S3 (`s3_service.py`), returning permanent CloudFront CDN URLs.
- **Multimodal Vision Input**: Supports image attachment (`max_reference_images: 3`). Users can upload screenshots or diagrams and ask the editor to summarize, OCR transcribe, or edit visual assets.

---

## 6. Deterministic Request Planning (`src/request_plan.py`)

To prevent model hallucination and runaway token costs, Eddyter enforces deterministic planning in code **before** calling the LLM:

```python
# Extracted from src/request_plan.py
plan = plan_request(query)
# plan.image_count   -> 0, 1, 2... (capped at IMAGE_MAX_PER_REQUEST, default 4)
# plan.needs_web     -> True / False (freshness keywords, news, research queries)
# plan.model_tier    -> 'basic' vs. 'advanced'
```

1. **Image Quota**: Phrases like *"write an article with 2 images"* deterministically reserve exactly 2 images. If no images are requested, the image generation tools are omitted from the agent's context entirely.
2. **Web Tools Gate**: DuckDuckGo search and web scraper tools are attached **only** if the query contains freshness cues (dates, *"current"*, *"latest"*, *"news"*, *"weather"*, *"who is"*). If the query is an evergreen writing prompt (e.g. *"write a story about a dragon"*), web tools are stripped. This eliminates tool-call latency overhead.
3. **Tool Budgets**: The `ToolBudget` class enforces hard caps per request (maximum 6 web searches and 8 page reads) to prevent agent infinite loops.

---

## 7. Credit Reservation, Settlement & License Engine

The credit engine ([`src/credit_manager.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/credit_manager.py), [`src/redis_manager.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/redis_manager.py)) manages atomic balance protection across distributed requests.

### 7.1 Scaled Credit Units
Because autocomplete costs 0.2 credits, all internal Redis accounting uses integer math scaled by 10 (`CREDIT_SCALE = 10`):
- `1.0 credit` = `10 scaled units`
- `0.2 credit` (autocomplete) = `2 scaled units`
- `5.0 credits` (image) = `50 scaled units`
- Text processing: 1 credit per 1,000 characters (input + output).

### 7.2 Atomic Lua Reservation
Before any agent runs, the service invokes `_RESERVE_LUA` in Redis:
```lua
-- Reads user balance: subscription:{userId} -> remaining_credits + purchased_credits
-- Checks active holds in: credits_reserved:{userId}
-- If (available - reserved) >= required:
--   INCRBY credits_reserved:{userId}, required
--   Sets 600s TTL (prevents permanent lock on crash)
--   Returns 1 (Success)
```

### 7.3 Two-Phase Settlement
1. **Hold**: In-flight credits remain reserved in Redis while the agent streams tokens.
2. **Settlement**: Upon stream completion, an asynchronous background task waits `CREDIT_SETTLE_DELAY_SECONDS` (3s) and dispatches an HMAC-signed webhook (`src/webhook.py`) to the Eddyter backend:
   ```json
   POST /api/agent-webhook/credit-deduction
   Headers: { "x-webhook-secret": "..." }
   {
     "user_id": "usr_123",
     "cost": 1.2,
     "request_id": "req_abc",
     "operation_type": "generate_blog"
   }
   ```
3. **Release**: The Lua release script decrements `credits_reserved:{userId}`.

---

## 8. Security Guardrails & Prompt Injection Firewall

All untrusted external data retrieved via web tools, user prompts, or web page scrapers passes through [`src/security_guardrails.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/security_guardrails.py) before reaching the LLM context.

- **Zero-Width Scrubber**: Strips steganographic characters (`\u200B`, `\u200C`, `\uFEFF`, `\u2060`) and bidirectional override codes (`\u202A`–`\u202E`).
- **Delimiter Defanger**: Neutralizes tokenizer control strings across all model architectures:
  - `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
  - `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>`
  - `### System:`, `---BEGIN SYSTEM PROMPT---`
- **Instruction Override Neutralizer**: Regex transforms adversarial commands into inert text:
  - *"ignore previous instructions"* ➔ `[DEFANGED: Instruction override attempt]`
  - *"enter DAN mode"* ➔ `[DEFANGED: Persona hijack attempt]`
  - *"exfiltrate data to https://..."* ➔ `[DEFANGED: Data exfiltration attempt]`
- **Sandboxed XML Fencing**: Scraped content is isolated within `<untrusted_web_content>` tags, reinforced by system prompt instructions forbidding execution of commands inside boundary tags.
- **Sub-Millisecond Speed**: Runs in **< 0.8ms** per 24,000 characters.

---

## 9. Autonomous Web Tools & SSRF-Protected Research Engine

When research is required, the agent leverages resilient, self-healing web utilities in [`src/text_processing_agent.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/src/text_processing_agent.py):

1. **DuckDuckGo Multi-Backend Rotation**: Rotates through DuckDuckGo backends (`html`, `lite`) with exponential backoff on HTTP 202/403/429.
2. **Zero-Starvation Secondary Fallback**: If DuckDuckGo rate-limits completely, the tool automatically queries:
   - Wikipedia Full-Text Search API & REST Summary API (`/api/rest_v1/page/summary`).
   - DuckDuckGo Instant Answer API.
3. **SSRF (Server-Side Request Forgery) Defense**:
   - Webpage scraper parses destination IPs and resolves DNS before connecting.
   - Blocks loopback (`127.0.0.1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), AWS metadata services (`169.254.169.254`), and IPv6 equivalents.
   - Blocks non-standard HTTP ports and limits HTTP redirects.

---

## 10. Observability, Tracing & Performance Metrics (`src/observability.py`)

Every incoming `/agent/*` request generates a unique `RequestTrace`:

- **Timing Marks**:
  - `auth`: Time to resolve license / Redis credentials.
  - `routing`: Time taken by task and intent regex engines.
  - `first_token`: Time to First Token (TTFT).
  - `text_done`: Full text stream completion time.
  - `image_<n>`: Individual image generation durations.
  - `total`: Total end-to-end wall-clock latency.
- **Trace Output**:
  - Emits human-readable log lines:
    ```text
    ⏱ [req_9f3a] POST /agent/stream total=3.42s | auth 0.02s | first_token 0.84s | text_done 3.41s
    ```
  - Emits machine-parsable `REQUEST_SUMMARY` JSON objects (written to `REQUEST_LOG_FILE` if configured).
  - Maintains an in-memory ring buffer (up to 500 traces) queryable via `GET /agent/metrics`.
- **Zero-Data-Leak Redaction**: All API keys, authorization headers, bearer tokens, and passwords are automatically stripped before logging.

---

## 11. API Surface & Streaming Protocol Reference

### 11.1 Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health probe returning Redis connection status, credit system state, and active worker health. |
| `POST` | `/agent/process` | Synchronous request execution. Returns full markdown response in a single JSON payload. |
| `POST` | `/agent/stream` | Primary Server-Sent Events (SSE) streaming endpoint for live text generation and inline media. |
| `POST` | `/agent/autocomplete` | Sub-second inline text completion for the editor keystroke listener. |
| `POST` | `/agent/abort` | Cancels an active in-flight stream using `request_id`. |
| `GET` | `/agent/email/categories`| Retrieves active email categories and metadata for the frontend email picker. |
| `POST` | `/agent/email/generate`  | Generates 1–3 structured JSON email template variants from a brief. |
| `POST` | `/agent/email/image`     | Auto-crops and uploads email assets to S3/CDN. |
| `GET` | `/agent/metrics`         | Diagnostic endpoint returning the ring buffer of recent traces. |

### 11.2 SSE Event Stream Contract (`POST /agent/stream`)

The streaming endpoint communicates via Server-Sent Events with structured event types:

```text
event: init
data: {"request_id": "req_123", "model": "gemini-3.8-flash", "provider": "gemini"}

event: token
data: {"token": "## Introduction\n\nArtificial Intelligence is"}

event: placeholder
data: {"placeholder_id": "placeholder::550e8400-e29b-41d4-a716-446655440000", "prompt": "Futuristic AI neural network"}

event: image_ready
data: {"placeholder_id": "placeholder::550e8400-e29b-41d4-a716-446655440000", "url": "https://d13is3h8ehkzsw.cloudfront.net/images/ai-net.jpg"}

event: content_replace
data: {"old_content": "...", "new_content": "..."}

event: complete
data: {"total_tokens": 842, "elapsed_seconds": 4.12, "warnings": []}
```

---

## 12. Configuration Inventory (`config.py` & `.env`)

All settings are strongly typed via Pydantic in [`config.py`](file:///Users/codecords/Desktop/Codebase/Agent/agent/config.py):

| Variable | Type | Default | Description |
|---|---|---|---|
| `REDIS_URL` | string | `None` | Upstash / Redis connection string for credit and license management. |
| `DEFAULT_OPENAI_API_KEY` | string | `None` | Platform fallback API key for OpenAI. |
| `DEFAULT_ANTHROPIC_API_KEY` | string | `None` | Platform fallback API key for Claude. |
| `DEFAULT_GROK_API_KEY` | string | `None` | Platform fallback API key for xAI Grok. |
| `DEFAULT_GEMINI_API_KEY` | string | `None` | Platform fallback API key for Google Gemini. |
| `DEFAULT_DEEPSEEK_API_KEY` | string | `None` | Platform fallback API key for DeepSeek. |
| `BACKEND_WEBHOOK_URL` | string | `""` | Webhook target for credit settlement (`/api/agent-webhook/credit-deduction`). |
| `WEBHOOK_SECRET` | string | `""` | Shared HMAC secret for verifying webhook authenticity. |
| `AWS_S3_BUCKET` | string | `""` | AWS S3 bucket name for image storage. |
| `AWS_REGION` | string | `"us-east-1"` | AWS S3 region. |
| `CDN_URL` | string | `""` | CloudFront CDN base URL mapped to the S3 bucket. |
| `CREDIT_PRECHECK_ENABLED` | boolean | `true` | Master kill switch for agent-side credit reservation. |
| `CREDIT_SETTLE_DELAY_SECONDS`| float | `3.0` | Delay before releasing credit reservations. |
| `MAX_RESERVATION_PER_REQUEST`| integer | `100` | Hard cap on credit reservations per single request. |
| `TASK_ROUTING_ENABLED` | boolean | `true` | Enables task-specific model routing (code ➔ Claude, etc.). |
| `PROVIDER_FALLBACK_ENABLED` | boolean | `true` | Enables automatic fallback to secondary LLMs on 429/5xx. |
| `CODE_VALIDATION_ENABLED` | boolean | `true` | Validates syntax of generated code without executing it. |
| `CODE_REPAIR_ENABLED` | boolean | `true` | Performs automatic one-pass repair on syntax errors. |
| `BLOG_AI_ENABLED` | boolean | `false` | Rollout flag for Mode 3 rich editorial blog generation. |
| `LONGFORM_REFINE_ENABLED` | boolean | `false` | Rollout flag for Mode 2 structure-preserving longform editing. |
| `EMAIL_AI_ENABLED` | boolean | `true` | Enables `/agent/email/*` template endpoints. |
| `IMAGE_DEFAULT_PROVIDER` | string | `"gemini"` | Default image generator provider (`gemini`, `openai`, `grok`). |

---

## 13. Local Development, Testing & Production Deployment

### 13.1 Local Setup

1. **Activate Environment & Install Dependencies**:
   ```bash
   cd agent
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure `.env`**:
   Ensure `.env` exists in the `agent/` folder (or project root). Populate required AI keys and Redis URL.

3. **Run Development Server**:
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 5000 --reload
   ```
   - OpenAPI Docs: [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)
   - Health Probe: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

### 13.2 Running Tests

The test suite includes mock guards to prevent inadvertent live Redis modifications or unbudgeted API spend:

```bash
# Run all automated tests
pytest

# Test specific subsystems
pytest tests/blog/
pytest tests/email/
pytest tests/longform/
pytest tests/code/
```

### 13.3 Production Systemd Configuration

In production (e.g. Ubuntu Linux), the service is managed via `systemd` using Gunicorn with Uvicorn workers:

File: `/etc/systemd/system/agent.service`
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

To enable and inspect:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now agent
sudo systemctl status agent
sudo journalctl -u agent -f
```

---

*This document is the authoritative architectural baseline for the Eddyter Agent Service. When extending models, adding routes, or tuning credit math, update this file and its associated tests.*
