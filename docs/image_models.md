# Image Models & API Capabilities Specification

*Written from provider docs and mocked-SDK tests. Re-run `scripts/image_smoke_test.py` with live keys to verify; the Grok JSON edit and Gemini `image_size` paths have not been exercised against the live APIs.*

## Executive Summary

| Provider | Default Model ID | Fallback Model ID | Generate Call Shape | Image-to-Image (Edit) Support | Output Format | Supported Dimensions / Sizes |
|---|---|---|---|---|---|---|
| **Google Gemini** (default) | `gemini-3.1-flash-lite-image` | `gemini-2.5-flash-image` / `gemini-3.1-flash-image` / `gemini-3.1-flash-image-preview` | `client.models.generate_content(contents=prompt, config=GenerateContentConfig(response_modalities=['IMAGE']))` | **Yes** — pass `[image_part, prompt]` in `contents` | Inline bytes (`image/jpeg` or `image/png`) | Aspect ratio (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`), `image_size` (`1K`, `2K`) |
| **OpenAI** | `gpt-image-1` | `gpt-image-1` (dall-e-3 deprecated on key) | `client.images.generate(model='gpt-image-1', prompt=prompt, size=size, quality=quality)` | **Yes** — `client.images.edit(model='gpt-image-1', image=buf, prompt=prompt, size=size)` | `b64_json` (raw image bytes) | `1024x1024`, `1024x1536`, `1536x1024`, `auto` |
| **Grok (xAI)** | `grok-imagine-image-2.0` | `grok-imagine-image` | `client.images.generate(model=model, prompt=prompt)` via `https://api.x.ai/v1` | **Yes, via JSON** — `POST /v1/images/edits` with `{"model","prompt","image":{"url":<data URI>,"type":"image_url"}}` (not the OpenAI SDK's multipart `images.edit`) | Temporary URL (`image/jpeg`) | aspect ratio / resolution documented but field names unpublished, so not sent |

---

## Provider Details

### 1. Google Gemini
- **Model IDs**:
  - Primary: `gemini-3.1-flash-lite-image` (default since 2026-10-19; override with `IMAGE_MODEL_GEMINI`).
    Chosen after a same-prompt side-by-side: 3.5-4.7s vs 8.4-10.2s for `gemini-2.5-flash-image`, equal or better
    quality, $0.0336/image vs $0.039. Google marks 2.5 deprecated and recommends the lite model.
  - Fallback: `gemini-2.5-flash-image`, `gemini-3.1-flash-image`, `gemini-3.1-flash-image-preview` (tried in this order when the primary is not available: retired/not found (404) or temporarily unavailable (503 overloaded, 429 quota). Safety blocks and bad-parameter 400s do not fall back. A fallback never downgrades the requested size)
  - **Sizes really delivered** (probed live 2026-10-19): lite = `1K` only (rejects 512/2K/4K with a 400);
    `gemini-2.5-flash-image` accepts 2K/4K but returns 1024x1024; `gemini-3.1-flash-image` = 512/1K/2K/4K
    (about 9.5s / 12s / 18s / 26s). Table: `GEMINI_IMAGE_SIZES` in `src/models.py`.
  - **Resolution routing:** with no model named, a `2K`/`4K` request goes to `GEMINI_HIRES_MODEL`
    (`gemini-3.1-flash-image`, env `IMAGE_MODEL_GEMINI_HIRES`). A model the user named is kept and an
    unsupported size is clamped to 1K instead of failing. Credits are priced for the model that actually runs.
- **Method**: Google GenAI SDK (`google.genai.Client`)
- **Generation**:
  ```python
  response = client.models.generate_content(
      model=model_id,
      contents=prompt,
      config=types.GenerateContentConfig(
          response_modalities=["IMAGE"],
      )
  )
  ```
- **Edit / Refinement**:
  ```python
  image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
  response = client.models.generate_content(
      model=model_id,
      contents=[image_part, prompt],
      config=types.GenerateContentConfig(
          response_modalities=["IMAGE"],
      )
  )
  ```
- **Response handling**: Extract `candidate.content.parts[0].inline_data.data`.

---

### 2. OpenAI
- **Model IDs**:
  - Primary: `gpt-image-1`
- **Method**: OpenAI SDK (`openai.OpenAI`)
- **Generation**:
  ```python
  response = client.images.generate(
      model="gpt-image-1",
      prompt=prompt,
      size="1024x1024",  # or "1024x1536", "1536x1024", "auto"
      quality="medium",  # "low" | "medium" | "high"
      output_format="jpeg",  # "png" | "jpeg" | "webp"
      n=1,
  )
  ```
- **Edit / Refinement**:
  ```python
  response = client.images.edit(
      model="gpt-image-1",
      image=image_file_or_bytes_io,
      prompt=prompt,
      size="1024x1024",
      n=1,
  )
  ```
- **Response handling**: Extract base64 `response.data[0].b64_json`.

---

### 3. Grok (xAI)
- **Model IDs**:
  - Primary: `grok-imagine-image-2.0`
  - Fallback: `grok-imagine-image`
- **Method**: OpenAI SDK with base URL `https://api.x.ai/v1`
- **Generation**:
  ```python
  response = client.images.generate(
      model="grok-imagine-image-2.0",
      prompt=prompt,
      n=1,
  )
  ```
- **Edit / Refinement**: `POST https://api.x.ai/v1/images/edits`, JSON body:
  ```json
  {"model": "grok-imagine-image-2.0", "prompt": "...",
   "image": {"url": "data:image/png;base64,...", "type": "image_url"}}
  ```
  Input must be jpg/jpeg/png and at most 20 MiB (other formats are transcoded to PNG first).
  An earlier version of this file said xAI had no edit endpoint. That came from calling it through
  the OpenAI SDK, which sends multipart/form-data and gets HTTP 415; the documented endpoint
  takes JSON. If an edit still fails, `generate_image_unified` falls back to Gemini when a platform
  key exists and otherwise raises. It never regenerates from text alone.
- **Response handling**: Download image bytes from temporary URL in `response.data[0].url`.

---

## Multi-Model & Cross-Provider Routing

Image generation requests can be routed to any supported provider and specific model, regardless of the chat model being used (e.g. Gemini, Grok, Claude, OpenAI, DeepSeek).

### 1. API Parameters (Direct Overrides)
Clients can specify `image_provider` and `image_model` at the top level of the request or inside `image_params`:
```json
{
  "type": "general",
  "query": "Draw a cybernetic tiger in a futuristic neon city",
  "model": "gemini",
  "image_provider": "grok",
  "image_model": "grok-imagine-image-2.0"
}
```
Or inside `image_params`:
```json
{
  "image_params": {
    "provider": "gemini",
    "model": "gemini-2.5-flash-image",
    "resolution": "1K",
    "format": "jpeg"
  }
}
```

### 2. Natural Language Prompt Cues (regex only; the LLM never picks the model)
The model is chosen from the **user's own words** by regex (`detect_image_model_request` in
`src/image_gen.py`). The chat LLM only writes the picture prompt: the LLM-facing image tools pass the
user's original query as `selection_text`, so wording the LLM adds to its own prompt cannot switch models.
Precedence: explicit API field > user's words > previous image in the document (follow-ups) > default.

**Default (no cue): Gemini `gemini-3.1-flash-lite-image`, whatever the chat model is.** Bring-your-own-key
users never use platform keys, so they stay on their own provider's key. Configurable with
`IMAGE_DEFAULT_PROVIDER` and `IMAGE_MODEL_GEMINI`.

| User says | Routes to |
|---|---|
| `using gemini`, `with gemini`, `gemini model` | Gemini, default model (lite) |
| `gemini lite`, `gemini 3.1 lite` | `gemini-3.1-flash-lite-image` |
| `gemini 2.5`, `nano banana` | `gemini-2.5-flash-image` |
| `gemini 3.1` | `gemini-3.1-flash-image` |
| `using grok`, `with xai` | Grok, default model (`grok-imagine-image-2.0`) |
| `grok imagine` | `grok-imagine-image` |
| `grok 2`, `grok imagine 2` | `grok-imagine-image-2.0` |
| `using gpt`, `with openai`, `via chatgpt`, `dall-e` | OpenAI, default model (`gpt-image-1`) |
| `gpt-image-1` / `gpt-image-1-mini` | that exact model |

Brand words alone only count with a lead-in (`using / with / via / through / by / from / on`) or a
trailing `model`, so "a poster with the Gemini zodiac sign", "a robot called Grok" or "Google Maps"
do not switch models. The phrase ("using grok") is removed from the picture prompt. If several
models are named, the first mention wins. A follow-up edit keeps the previous image's model unless the
user names another.

Measured latency (live, 2026-10-19, 16:9, provider call + download): `gemini-3.1-flash-lite-image` ~3.6s,
`grok-imagine-image` ~6.7s, `gemini-2.5-flash-image` ~8-9.5s, `gpt-image-1-mini` (low) ~8.7s, `gemini-3.1-flash-image` ~12s,
`grok-imagine-image-2.0` ~12.8s, `gpt-image-1` low ~12s / medium ~25s.

### 3. Key Resolution
- If the image provider matches the chat model provider, the resolved chat model key is used.
- For cross-provider generation (e.g., chat is Gemini/Claude and image provider is Grok):
  - On managed accounts (non-BYOK), the backend resolves the platform's default API key for that image provider.
  - On BYOK accounts, the client can pass an explicit `api_key` inside `image_params` or configure the provider key in their settings.


---

## Deterministic Request Planning (code decides, the LLM writes)

`src/request_plan.py` plans the free-form ("general") agent from the **user's own words** before any LLM
call. Both `/agent/stream` and `/agent/process` use it (the non-stream endpoint now runs the same pipeline,
waits for the images, and writes their real URLs into the text; a failed image becomes
`*(image could not be generated)*`, never a dead link).

| Decision | Made by | Rule |
|---|---|---|
| How many images | `extract_image_count` | Number in the user's words ("2 images", "a hero image", "several"), only after a request verb ("with", "include", "add"…). `with images` = `IMAGE_DEFAULT_PLURAL_COUNT` (2). Capped at `IMAGE_MAX_PER_REQUEST` (4); the prompt tells the model about a cap. 0 = no image tool is attached. |
| Image tool for Claude/DeepSeek | `create_agent` | Attached for **every** chat provider when an image provider can be resolved (platform Gemini key, or the user's own). Otherwise the prompt says images are unavailable. |
| Web search/read | `needs_web` | Attached only on freshness/research cues ("latest", a 20xx year, "statistics", "research", "sources", a URL…). `WEB_TOOLS_GATE_ENABLED=false` restores always-on. |
| Tool limits | `ToolBudget` + `tool_call_limit` | `WEB_SEARCH_CALL_LIMIT` (6 queries), `WEB_READ_CALL_LIMIT` (8 pages), images = planned count; agno `tool_call_limit` is the backstop. |
| Model tier | `needs_advanced_model` | Fast tier for short conversational writes with no tools; advanced for tools, long-form/code requests, a 500+ word target, or input over `TIER_LONG_QUERY_CHARS` (1500). `MODEL_TIER_ROUTING_ENABLED=false` restores always-advanced. |
| Picture settings (shape, quality, resolution, format, transparency) | `ImageParams` | Read from the user's text (`selection_text`), never from the LLM-written picture prompt. The LLM-facing image tools no longer take a `size` argument. |

The system prompt is assembled from the plan (`build_smart_global_instructions`): it describes only the
tools that were granted and the exact number of images, so the model has no routing decision to make.

### Image credits per model
Each image is priced with `credit_manager.image_credits_scaled(provider, model)`: exact `provider/model`,
then `provider`, then the flat `IMAGE_CREDITS_SCALED` (50). The table is empty by default (**no price
change**); set prices in `IMAGE_CREDITS_BY_MODEL` in `src/credit_manager.py` or with the env var
`IMAGE_CREDITS_BY_MODEL='{"openai/gpt-image-1": 80}'`. The agent reserves the price of the model that will
actually run, releases **every** image reservation at settlement (failed images no longer hold credits),
and sends `images: [{provider, model, credits_scaled}]` in the credit webhook. Image-only requests are
attributed to the image provider/model instead of the unused chat model.
**The backend still computes the actual deduction from `image_count`**; it must read `images[].credits_scaled`
before per-model prices change what users are charged.

### Environment variables
`IMAGE_MAX_PER_REQUEST`, `IMAGE_DEFAULT_PLURAL_COUNT`, `WEB_TOOLS_GATE_ENABLED`, `WEB_SEARCH_CALL_LIMIT`,
`WEB_READ_CALL_LIMIT`, `MODEL_TIER_ROUTING_ENABLED`, `TIER_LONG_QUERY_CHARS`, `IMAGE_WAIT_SECONDS`
(how long `/agent/process` waits for images, default 120), `IMAGE_CREDITS_BY_MODEL`.
