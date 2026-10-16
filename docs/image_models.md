# Image Models & API Capabilities Specification

*Written from provider docs and mocked-SDK tests. Re-run `scripts/image_smoke_test.py` with live keys to verify; the Grok JSON edit and Gemini `image_size` paths have not been exercised against the live APIs.*

## Executive Summary

| Provider | Default Model ID | Fallback Model ID | Generate Call Shape | Image-to-Image (Edit) Support | Output Format | Supported Dimensions / Sizes |
|---|---|---|---|---|---|---|
| **Google Gemini** | `gemini-3.1-flash-image` | `gemini-3.1-flash-image-preview` / `gemini-2.5-flash-image` | `client.models.generate_content(contents=prompt, config=GenerateContentConfig(response_modalities=['IMAGE']))` | **Yes** — pass `[image_part, prompt]` in `contents` | Inline bytes (`image/jpeg` or `image/png`) | Aspect ratio (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`), `image_size` (`1K`, `2K`) |
| **OpenAI** | `gpt-image-1` | `gpt-image-1` (dall-e-3 deprecated on key) | `client.images.generate(model='gpt-image-1', prompt=prompt, size=size, quality=quality)` | **Yes** — `client.images.edit(model='gpt-image-1', image=buf, prompt=prompt, size=size)` | `b64_json` (raw image bytes) | `1024x1024`, `1024x1536`, `1536x1024`, `auto` |
| **Grok (xAI)** | `grok-imagine-image-2.0` | `grok-imagine-image` | `client.images.generate(model=model, prompt=prompt)` via `https://api.x.ai/v1` | **Yes, via JSON** — `POST /v1/images/edits` with `{"model","prompt","image":{"url":<data URI>,"type":"image_url"}}` (not the OpenAI SDK's multipart `images.edit`) | Temporary URL (`image/jpeg`) | aspect ratio / resolution documented but field names unpublished, so not sent |

---

## Provider Details

### 1. Google Gemini
- **Model IDs**:
  - Primary: `gemini-3.1-flash-image`
  - Fallback: `gemini-3.1-flash-image-preview`, `gemini-2.5-flash-image`
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
