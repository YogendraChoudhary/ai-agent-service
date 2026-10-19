# Image Models & API Capabilities Specification

*Written from provider docs and mocked-SDK tests. Re-run `scripts/image_smoke_test.py` with live keys to verify; the Grok JSON edit and Gemini `image_size` paths have not been exercised against the live APIs.*

## Executive Summary

| Provider | Default Model ID | Fallback Model ID | Generate Call Shape | Image-to-Image (Edit) Support | Output Format | Supported Dimensions / Sizes |
|---|---|---|---|---|---|---|
| **Google Gemini** (default) | `gemini-2.5-flash-image` | `gemini-3.1-flash-image` / `gemini-3.1-flash-image-preview` | `client.models.generate_content(contents=prompt, config=GenerateContentConfig(response_modalities=['IMAGE']))` | **Yes** — pass `[image_part, prompt]` in `contents` | Inline bytes (`image/jpeg` or `image/png`) | Aspect ratio (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`), `image_size` (`1K`, `2K`) |
| **OpenAI** | `gpt-image-1` | `gpt-image-1` (dall-e-3 deprecated on key) | `client.images.generate(model='gpt-image-1', prompt=prompt, size=size, quality=quality)` | **Yes** — `client.images.edit(model='gpt-image-1', image=buf, prompt=prompt, size=size)` | `b64_json` (raw image bytes) | `1024x1024`, `1024x1536`, `1536x1024`, `auto` |
| **Grok (xAI)** | `grok-imagine-image-2.0` | `grok-imagine-image` | `client.images.generate(model=model, prompt=prompt)` via `https://api.x.ai/v1` | **Yes, via JSON** — `POST /v1/images/edits` with `{"model","prompt","image":{"url":<data URI>,"type":"image_url"}}` (not the OpenAI SDK's multipart `images.edit`) | Temporary URL (`image/jpeg`) | aspect ratio / resolution documented but field names unpublished, so not sent |

---

## Provider Details

### 1. Google Gemini
- **Model IDs**:
  - Primary: `gemini-2.5-flash-image` (default since 2026-10-19; override with `IMAGE_MODEL_GEMINI`)
  - Fallback: `gemini-3.1-flash-image`, `gemini-3.1-flash-image-preview` (used only if the primary is retired/not found)
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

**Default (no cue): Gemini `gemini-2.5-flash-image`, whatever the chat model is.** Bring-your-own-key
users never use platform keys, so they stay on their own provider's key. Configurable with
`IMAGE_DEFAULT_PROVIDER` and `IMAGE_MODEL_GEMINI`.

| User says | Routes to |
|---|---|
| `using gemini`, `with gemini`, `gemini model` | Gemini, default model (2.5) |
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

Measured latency (live, 2026-10-19, 16:9, provider call + download): `grok-imagine-image` ~6.7s,
`gemini-2.5-flash-image` ~9.5s, `gpt-image-1-mini` (low) ~8.7s, `gemini-3.1-flash-image` ~12s,
`grok-imagine-image-2.0` ~12.8s, `gpt-image-1` low ~12s / medium ~25s.

### 3. Key Resolution
- If the image provider matches the chat model provider, the resolved chat model key is used.
- For cross-provider generation (e.g., chat is Gemini/Claude and image provider is Grok):
  - On managed accounts (non-BYOK), the backend resolves the platform's default API key for that image provider.
  - On BYOK accounts, the client can pass an explicit `api_key` inside `image_params` or configure the provider key in their settings.

