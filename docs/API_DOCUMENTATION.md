# Text Processing Agent API Documentation

A powerful AI-powered text processing API that supports multiple AI providers and various text operations for HTML editor integration.

## Base URL

```
http://localhost:5000
```

## Supported AI Models

| Provider | Model Key | Fast Model | Advanced Model |
|----------|-----------|------------|----------------|
| OpenAI | `openai` | gpt-3.5-turbo | gpt-4o |
| Claude | `claude` | claude-3-haiku-20240307 | claude-3-5-sonnet-20241022 |
| Grok | `grok` | grok-2-1212 | grok-2-latest |
| DeepSeek | `deepseek` | deepseek-chat | deepseek-reasoner |
| Gemini | `gemini` | gemini-2.0-flash-lite | gemini-2.0-flash |

## Endpoint

### POST `/agent/process`

Process text using AI with various operations.

#### Request Headers

```
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | The operation type (see supported types below) |
| `query` | string | Yes | The text to process |
| `model` | string | Yes | AI model provider: `openai`, `claude`, `grok`, `deepseek`, `gemini` |
| `api_key` | string | Yes | API key for the selected model provider |
| `secondary_prompt` | string | No | Additional instruction for specific operations |
| `blog_options` | object | No | `generate_blog` only: `length`, `images`, `tone`, `audience` (see [blog generation](#visually-rich-blog-generation-generate_blog)) |

#### Supported Operation Types

| Type | Description | Uses `secondary_prompt` |
|------|-------------|------------------------|
| `fix_grammar` | Fix grammar, spelling, and punctuation errors | No |
| `simplify` | Simplify complex text for easier understanding | No |
| `complete` | Complete partial sentences and paragraphs | No |
| `emojify` | Add relevant emojis to text | No |
| `translate` | Translate text to another language | Yes (target language) |
| `make_longer` | Expand and elaborate text | No |
| `make_shorter` | Condense text while preserving meaning | No |
| `enhance_with_ai` | Enhance and improve text quality | Yes (enhancement instructions) |
| `change_tone` | Change the tone/style of text | Yes (tone description) |
| `general` | General content creation (articles, images, etc.) | No |
| `generate_blog` | A visually rich blog post: tables, pull quotes and inline images ([details](#visually-rich-blog-generation-generate_blog)) | Yes (style notes) |

#### Response

**Success (200)**
```json
{
  "success": true,
  "markdown_response": "Processed Markdown content...",
  "html_response": "Processed Markdown content...",
  "warnings": []
}
```

`html_response` repeats `markdown_response` (kept for existing clients). `warnings` is always present and empty unless the
server fixed or could not do something (today only `generate_blog` reports any; see below).

**Error (400/500)**
```json
{
  "error": "Error message description"
}
```

---

## Node.js Integration

### Installation

```bash
npm install axios
```

### Basic Setup

```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:5000';

// Configuration
const config = {
  model: 'openai', // or 'claude', 'grok', 'deepseek', 'gemini'
  api_key: 'your-api-key-here'
};

/**
 * Process text using the AI agent
 * @param {string} type - Operation type
 * @param {string} query - Text to process
 * @param {string} secondaryPrompt - Optional secondary prompt
 * @returns {Promise<string>} - Processed HTML response
 */
async function processText(type, query, secondaryPrompt = '') {
  try {
    const response = await axios.post(`${API_BASE_URL}/agent/process`, {
      type,
      query,
      model: config.model,
      api_key: config.api_key,
      secondary_prompt: secondaryPrompt
    });

    return response.data.html_response;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.error);
    }
    throw error;
  }
}

module.exports = { processText, config };
```

---

## Usage Examples

### 1. Fix Grammar

```javascript
const { processText } = require('./textProcessor');

async function fixGrammar() {
  const text = 'Their going to the store tommorow and they dont have no money.';

  const result = await processText('fix_grammar', text);
  console.log(result);
  // Output: <p>They're going to the store tomorrow and they don't have any money.</p>
}

fixGrammar();
```

### 2. Simplify Text

```javascript
async function simplifyText() {
  const text = 'The implementation of quantum computing paradigms necessitates a comprehensive understanding of superposition and entanglement phenomena.';

  const result = await processText('simplify', text);
  console.log(result);
  // Output: <p>To use quantum computers, you need to understand how particles can be in multiple states at once and how they can be connected.</p>
}

simplifyText();
```

### 3. Complete Text

```javascript
async function completeText() {
  const text = 'The benefits of regular exercise include improved cardiovascular health,';

  const result = await processText('complete', text);
  console.log(result);
  // Output: <p>The benefits of regular exercise include improved cardiovascular health, enhanced mental well-being, increased energy levels, and better sleep quality.</p>
}

completeText();
```

### 4. Add Emojis

```javascript
async function emojifyText() {
  const text = 'I am so happy today! The weather is beautiful and I got a promotion at work.';

  const result = await processText('emojify', text);
  console.log(result);
  // Output: <p>I am so happy today! 😊 The weather is beautiful ☀️ and I got a promotion at work! 🎉</p>
}

emojifyText();
```

### 5. Translate Text

```javascript
async function translateText() {
  const text = 'Hello, how are you? I hope you are having a wonderful day.';
  const targetLanguage = 'Spanish';

  const result = await processText('translate', text, targetLanguage);
  console.log(result);
  // Output: <p>Hola, ¿cómo estás? Espero que estés teniendo un día maravilloso.</p>
}

translateText();
```

### 6. Make Text Longer

```javascript
async function makeLonger() {
  const text = 'Climate change is a serious problem.';

  const result = await processText('make_longer', text);
  console.log(result);
  // Output: <p>Climate change represents one of the most serious and pressing problems facing our planet today, with far-reaching consequences that affect ecosystems, weather patterns, sea levels, and the overall health of our environment.</p>
}

makeLonger();
```

### 7. Make Text Shorter

```javascript
async function makeShorter() {
  const text = 'In the current day and age, it has become increasingly apparent and abundantly clear that the utilization of artificial intelligence technologies has the potential to revolutionize and fundamentally transform the way in which businesses operate and conduct their day-to-day activities.';

  const result = await processText('make_shorter', text);
  console.log(result);
  // Output: <p>AI technologies can revolutionize how businesses operate.</p>
}

makeShorter();
```

### 8. Enhance with AI

```javascript
async function enhanceText() {
  const text = 'The product is good. It works well. People like it.';
  const enhancement = 'Make it more professional and compelling for a marketing brochure';

  const result = await processText('enhance_with_ai', text, enhancement);
  console.log(result);
  // Output: <p>Our exceptional product delivers outstanding performance that consistently exceeds expectations. Trusted by customers worldwide, it has earned a reputation for reliability and excellence.</p>
}

enhanceText();
```

### 9. Change Tone

```javascript
async function changeTone() {
  const text = 'Hey! Just wanted to let you know the project is done. It was pretty tough but we made it work somehow.';
  const tone = 'Professional: Formal and business-appropriate';

  const result = await processText('change_tone', text, tone);
  console.log(result);
  // Output: <p>I am pleased to inform you that the project has been successfully completed. Despite encountering several challenges during the development process, our team was able to deliver the expected results.</p>
}

changeTone();
```

### 10. General Content Creation

```javascript
async function createArticle() {
  const prompt = 'Write a short article about the benefits of meditation for mental health';

  const result = await processText('general', prompt);
  console.log(result);
  // Output: Full HTML article with headings, paragraphs, etc.
}

createArticle();
```

---

## Complete Node.js Class Implementation

```javascript
const axios = require('axios');

class TextProcessingAgent {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:5000';
    this.model = options.model || 'openai';
    this.apiKey = options.apiKey;

    if (!this.apiKey) {
      throw new Error('API key is required');
    }
  }

  async _request(type, query, secondaryPrompt = '') {
    try {
      const response = await axios.post(`${this.baseUrl}/agent/process`, {
        type,
        query,
        model: this.model,
        api_key: this.apiKey,
        secondary_prompt: secondaryPrompt
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 60000 // 60 second timeout
      });

      if (response.data.success) {
        return response.data.html_response;
      }

      throw new Error(response.data.error || 'Unknown error occurred');
    } catch (error) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      }
      throw error;
    }
  }

  // Fix grammar, spelling, and punctuation
  async fixGrammar(text) {
    return this._request('fix_grammar', text);
  }

  // Simplify complex text
  async simplify(text) {
    return this._request('simplify', text);
  }

  // Complete partial text
  async complete(text) {
    return this._request('complete', text);
  }

  // Add emojis to text
  async emojify(text) {
    return this._request('emojify', text);
  }

  // Translate text to target language
  async translate(text, targetLanguage) {
    return this._request('translate', text, targetLanguage);
  }

  // Make text longer
  async makeLonger(text) {
    return this._request('make_longer', text);
  }

  // Make text shorter
  async makeShorter(text) {
    return this._request('make_shorter', text);
  }

  // Enhance text with AI
  async enhance(text, instructions = '') {
    return this._request('enhance_with_ai', text, instructions);
  }

  // Change tone of text
  async changeTone(text, toneDescription) {
    return this._request('change_tone', text, toneDescription);
  }

  // General content creation
  async generate(prompt) {
    return this._request('general', prompt);
  }

  // Change the AI model
  setModel(model) {
    const validModels = ['openai', 'claude', 'grok', 'deepseek', 'gemini'];
    if (!validModels.includes(model.toLowerCase())) {
      throw new Error(`Invalid model. Must be one of: ${validModels.join(', ')}`);
    }
    this.model = model.toLowerCase();
  }

  // Update API key
  setApiKey(apiKey) {
    this.apiKey = apiKey;
  }
}

module.exports = TextProcessingAgent;
```

---

## Using the Class

```javascript
const TextProcessingAgent = require('./TextProcessingAgent');

// Initialize with OpenAI
const agent = new TextProcessingAgent({
  model: 'openai',
  apiKey: 'sk-your-openai-api-key'
});

async function main() {
  try {
    // Fix grammar
    const corrected = await agent.fixGrammar('Their going to the store.');
    console.log('Corrected:', corrected);

    // Translate to Spanish
    const translated = await agent.translate('Hello, how are you?', 'Spanish');
    console.log('Translated:', translated);

    // Switch to Claude
    agent.setModel('claude');
    agent.setApiKey('sk-ant-your-anthropic-key');

    // Enhance with Claude
    const enhanced = await agent.enhance(
      'The product is good.',
      'Make it more professional'
    );
    console.log('Enhanced:', enhanced);

    // Switch to Gemini
    agent.setModel('gemini');
    agent.setApiKey('your-google-api-key');

    // Generate content with Gemini
    const article = await agent.generate('Write a haiku about programming');
    console.log('Generated:', article);

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```

---

## Express.js Integration Example

```javascript
const express = require('express');
const TextProcessingAgent = require('./TextProcessingAgent');

const app = express();
app.use(express.json());

// Initialize agents for different providers
const agents = {
  openai: new TextProcessingAgent({
    model: 'openai',
    apiKey: process.env.OPENAI_API_KEY
  }),
  claude: new TextProcessingAgent({
    model: 'claude',
    apiKey: process.env.ANTHROPIC_API_KEY
  }),
  gemini: new TextProcessingAgent({
    model: 'gemini',
    apiKey: process.env.GOOGLE_API_KEY
  })
};

// Endpoint to process text
app.post('/process', async (req, res) => {
  const { text, operation, model = 'openai', secondaryPrompt } = req.body;

  try {
    const agent = agents[model];
    if (!agent) {
      return res.status(400).json({ error: 'Invalid model specified' });
    }

    let result;
    switch (operation) {
      case 'fix_grammar':
        result = await agent.fixGrammar(text);
        break;
      case 'simplify':
        result = await agent.simplify(text);
        break;
      case 'translate':
        result = await agent.translate(text, secondaryPrompt);
        break;
      case 'enhance':
        result = await agent.enhance(text, secondaryPrompt);
        break;
      case 'change_tone':
        result = await agent.changeTone(text, secondaryPrompt);
        break;
      case 'make_longer':
        result = await agent.makeLonger(text);
        break;
      case 'make_shorter':
        result = await agent.makeShorter(text);
        break;
      default:
        result = await agent.generate(text);
    }

    res.json({ success: true, html: result });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

---

## Error Handling

```javascript
const TextProcessingAgent = require('./TextProcessingAgent');

async function processWithRetry(agent, operation, text, maxRetries = 3) {
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await agent[operation](text);
    } catch (error) {
      lastError = error;
      console.warn(`Attempt ${attempt} failed: ${error.message}`);

      if (attempt < maxRetries) {
        // Wait before retrying (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  }

  throw lastError;
}

// Usage
const agent = new TextProcessingAgent({
  model: 'openai',
  apiKey: 'your-api-key'
});

processWithRetry(agent, 'fixGrammar', 'Their going to the store.')
  .then(result => console.log(result))
  .catch(error => console.error('All retries failed:', error.message));
```

---

## TypeScript Definitions

```typescript
interface TextProcessingAgentOptions {
  baseUrl?: string;
  model: 'openai' | 'claude' | 'grok' | 'deepseek' | 'gemini';
  apiKey: string;
}

interface ProcessResponse {
  success: boolean;
  html_response: string;
}

interface ErrorResponse {
  error: string;
}

type OperationType =
  | 'fix_grammar'
  | 'simplify'
  | 'complete'
  | 'emojify'
  | 'translate'
  | 'make_longer'
  | 'make_shorter'
  | 'enhance_with_ai'
  | 'change_tone'
  | 'general';

declare class TextProcessingAgent {
  constructor(options: TextProcessingAgentOptions);

  fixGrammar(text: string): Promise<string>;
  simplify(text: string): Promise<string>;
  complete(text: string): Promise<string>;
  emojify(text: string): Promise<string>;
  translate(text: string, targetLanguage: string): Promise<string>;
  makeLonger(text: string): Promise<string>;
  makeShorter(text: string): Promise<string>;
  enhance(text: string, instructions?: string): Promise<string>;
  changeTone(text: string, toneDescription: string): Promise<string>;
  generate(prompt: string): Promise<string>;

  setModel(model: TextProcessingAgentOptions['model']): void;
  setApiKey(apiKey: string): void;
}

export = TextProcessingAgent;
```

---

## Rate Limiting Considerations

Each AI provider has different rate limits. Consider implementing rate limiting in your application:

```javascript
const Bottleneck = require('bottleneck');

// Create a limiter (e.g., 10 requests per minute)
const limiter = new Bottleneck({
  minTime: 6000, // Minimum time between requests (ms)
  maxConcurrent: 1
});

const agent = new TextProcessingAgent({
  model: 'openai',
  apiKey: 'your-api-key'
});

// Wrap agent methods with rate limiting
const rateLimitedFixGrammar = limiter.wrap(agent.fixGrammar.bind(agent));

// Use the rate-limited version
rateLimitedFixGrammar('Their going to the store.')
  .then(result => console.log(result));
```

---

## Image Generation & Editing Flow

`/agent/stream` can send simple image requests straight to the image adapter, skipping the agent's LLM turns.

### 1. When a request goes straight to the image adapter
Only when **all** of these hold:
- the request `type` is `general` (text operations such as `fix_grammar`, `simplify`, `enhance_with_ai` are never routed to images, because the text they carry can mention images freely);
- the query *starts* with an image request, e.g. *"draw a mosque at dusk"*, *"please generate an image of a red barn"*, *"remove background from this photo"* (with an attached image);
- for edits and follow-ups, there is an image to work on: an attached `image_urls[0]`, or the user's previous generated image.

Everything else, including *"write an article with 2 images"*, stays on the agent. Edits and follow-ups with no image available also stay on the agent.

The response is a markdown placeholder `![alt](placeholder::<uuid>)` followed by an `image_ready` / `image_failed` event, the same contract the agent path uses.

Providers: with no model named, images are drawn by **Gemini (`gemini-3.1-flash-lite-image`) whatever the chat model is**, using the platform Gemini key. Accounts that bring their own keys (BYOK) never use platform keys: they stay on their own provider's key when it has an image endpoint (OpenAI, Gemini, Grok), otherwise the request stays on the agent path. The user can pick another model in the prompt ("using grok", "with gpt", "gemini 3.1") or with `image_provider` / `image_model` (see "Request fields"). The prompt wording is matched by regex; the chat LLM never chooses the image model.

### 2. Settings and how they are chosen
Precedence: **explicit `image_params` field > words in the prompt > previous image's settings (follow-ups) > default.**

| Setting | Default | Prompt words that change it |
|---|---|---|
| `aspect_ratio` | `1:1` | `16:9`, `widescreen`, `banner`, `horizontal`, `in landscape`, `landscape image/photo/format…` → `16:9`; `9:16`, `vertical`, `instagram story`, `reel`, `tiktok`, `in portrait`, `portrait image/photo/format…` → `9:16`; `4:3`, `3:4`, `square` |
| `quality` | `medium` | `draft`, `low quality` → `low`; `high quality`, `best quality`, `ultra-detailed`, `hq` → `high` |
| `resolution` | `1K` | `2k`, `high-res` → `2K`; `4k`, `uhd` → `4K` (Gemini only; OpenAI uses fixed sizes, Grok ignores it). The default Gemini model only draws 1K, so 2K/4K requests are drawn by `gemini-3.1-flash-image` (about 18s / 26s) |
| `format` | `jpeg` (`IMAGE_DEFAULT_FORMAT`) | `png`, `webp`, `jpeg`/`jpg`; `transparent background` forces `png` |

Style words an LLM habitually writes ("cinematic", "detailed", "story", "wide-angle") do **not** change settings. Transparent output is requested natively only from OpenAI; Gemini and Grok return an opaque background.

### 3. Request fields
```json
{
  "type": "general",
  "query": "draw a mosque at dusk",
  "model": "gemini",
  "api_key": "<licence key>",
  "document_id": "doc-123",
  "image_params": { "aspect_ratio": "16:9", "quality": "high", "format": "webp" }
}
```
- `document_id` (optional, `[A-Za-z0-9_.:-]`, max 128): scopes the "previous image" memory. When sent, follow-ups only ever see images from that document.
- `image_params` (optional): `aspect_ratio` (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`), `quality` (`low`/`medium`/`high`), `resolution` (`1K`/`2K`/`4K`), `format` (`jpeg`/`png`/`webp`), `transparent_bg` (bool), `provider` (`gemini`/`openai`/`grok`), `model` (a model id of that provider). Invalid keys or values return **HTTP 400** `{"error": "image_params: ..."}` (earlier versions of this document said 422; the server has always answered 400). A `model` that does not belong to the given `provider` (for example `provider: gemini`, `model: gpt-image-1`) is also a 400.
- `image_provider`, `image_model` (optional, top level, on both `/agent/stream` and `/agent/process`): same as `image_params.provider` / `.model`. Unlike the `image_params` form, a mismatched pair here is not rejected: the model that does not belong to the provider is ignored and that provider's default is used.

### 4. Server-Sent Events
`image_ready` carries dimension and format metadata (fields are omitted when the image could not be measured):

```json
data: {
  "type": "image_ready",
  "placeholder_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "url": "https://cdn.example.com/user_1/ai-images/3fa85f64.jpg",
  "prompt": "draw a mosque at dusk",
  "width": 1536,
  "height": 1024,
  "format": "jpg"
}
```
`format` is the file extension of the stored image (`jpg`, `png`, `webp`).

### 5. Follow-ups (Redis, 1 hour TTL)
Each successful image is stored per user (and per `document_id` when sent). A follow-up only counts when a stored image exists:
- **Redo** (*"regenerate"*, *"again"*, *"another version"*): same prompt and settings.
- **Delta** (*"make it night"*, *"same but in autumn colors"*, *"change the mood to dramatic"*) and **edits of the last image** (*"refine the image with a Taj Mahal vibe"*): previous image sent as the reference, prompt = original prompt + the last 3 modifications, settings inherited unless the new prompt overrides them.

Without a stored image these phrases are handled by the agent as ordinary text. An attached `image_urls[0]` always takes priority over the stored image.

---

## Changes for API consumers (2026-10-19)

**No breaking change.** Request fields, response fields, status codes and Server-Sent Events are the same.
Nothing has to change in a client that works today; the items below are behaviour changes it should know about.

**Unchanged**
- Request bodies of `/agent/process`, `/agent/stream`, `/agent/autocomplete`, `/agent/abort`.
- Stream events: `start`, `content`, `done`, `image_ready`, `image_failed`, `stream_end`, `credit_limit_reached`, `error`, with the same fields.
- `/agent/process` response shape: `{ success, markdown_response, html_response }`.

**Additive**
- Every response carries `X-Response-Time-Ms` (next to `X-Request-ID`). For a stream it is the time until the headers were sent, not the whole stream.
- `GET /agent/metrics` exists but answers 404 unless the server sets `METRICS_ENDPOINT_ENABLED=true` (local/staging only).
- `image_params` accepts `provider` and `model` (see "Request fields").

**Behaviour changes a client may notice**
1. **The `type` field decides the agent.** `fix_grammar`, `simplify`, `translate`, ... always run their own agent, whatever words the text contains. `type: "general"` always runs the free-form agent: tables, lists and summaries requested through `general` are written by that agent. If the editor sends `type: "general"` for operations that have their own `type`, send the specific `type`.
2. **`/agent/process` now returns finished images.** It waits (up to 120s, `IMAGE_WAIT_SECONDS`) and puts the real image URLs in `markdown_response` / `html_response`. Before, it returned `placeholder::<uuid>` links that never resolved. An image that fails or times out is replaced by the text `*(image could not be generated)*`. **Set the HTTP client timeout for this endpoint above 2 minutes** (a 4K image alone takes about 25s).
3. **Model failures are now visible.** If the AI provider rejects the call (bad key, quota, outage) the server returns HTTP 500 `{"error": "<provider> request failed: ..."}` on `/agent/process`, and a `{"type": "error", "error": "..."}` event on `/agent/stream` (no `done` follows). Before, these came back as an empty, successful response. An empty model answer with no images is also an error now. Make sure both the 500 and the `error` event are shown to the user.
4. **Image defaults.** Default model `gemini-3.1-flash-lite-image` (about 3.5s per image), whatever the chat model. Claude and DeepSeek chats can now produce images too.
5. **Images inside written content ("write a post with 2 images").** The number of images comes from the user's words ("2 images", "a hero image", "with images" = 2), capped at 4 per request, and only when the request asks for them. The model can no longer add extra images. The placeholder markdown and `image_ready` events are the same as before.
6. **Model cue is removed from the picture prompt.** "draw a cat using grok" draws "draw a cat"; the `prompt` in `image_ready` and the alt text do not contain "using grok".
7. **Web access** is only granted when the request needs fresh or sourced facts (words such as "latest", a year, "research", a URL). Other requests are answered from the model's own knowledge.

---

## Environment Variables

For production, store API keys in environment variables:

```bash
# .env file
DEFAULT_OPENAI_API_KEY=sk-your-openai-key
DEFAULT_ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
DEFAULT_GROK_API_KEY=xai-your-grok-key
DEFAULT_DEEPSEEK_API_KEY=sk-your-deepseek-key
DEFAULT_GEMINI_API_KEY=your-google-api-key
TEXT_AGENT_URL=http://localhost:5000
```

```javascript
require('dotenv').config();

const TextProcessingAgent = require('./TextProcessingAgent');

const agent = new TextProcessingAgent({
  baseUrl: process.env.TEXT_AGENT_URL,
  model: 'openai',
  apiKey: process.env.OPENAI_API_KEY
});
```

---

## Support

For issues or questions, please refer to the main repository or contact the development team.

## Code generation and task routing

`/agent/process` and `/agent/stream` accept `generate_code` (tagged Markdown code fences only) and
`explain_code` (Markdown prose). Code generation has budgeted web search/page reading and no image tools.
Both endpoints and `/agent/email/generate` accept `model_source: "default" | "user"`; omission means
`"user"`. The frontend sends `"default"` until the user changes the model dropdown, then `"user"`.
Email continues to use the submitted provider. Existing image selection is unchanged.

When task routing is enabled, `/agent/stream` selects Claude for default code requests, OpenAI (GPT-5.2) for
default `make_longer` and `generate_email_template` requests, and preserves the submitted provider for all
other content, including `change_tone`. A user-selected provider takes priority. Code candidates are Claude →
OpenAI → Gemini; writing candidates are OpenAI → submitted provider → Claude; content candidates are submitted
provider → OpenAI → Claude. Providers without keys,
providers without vision for reference-image requests, and blocked model IDs are removed. The original
model tier is preserved. BYOK requests, including accounts with `use_own_keys`, never use platform keys
for routing or classification.

`/agent/process` always uses the submitted provider/key; the backend still owns its billing. Both endpoints
can retry a 429 once when provider fallback is enabled, honoring Retry-After up to two seconds. Cross-provider
fallback only occurs on streaming, for rate limits, server failures, overload, connection timeouts/resets, and
an unusable provider (401/403 invalid key or permission, 404 retired model), before content or background
image work begins. Unusable providers are logged at error level. Bad requests and safety failures stop
immediately because every provider would reject them.
Streaming text is billed to the provider/model that actually answered; character-based prices are unchanged.

Code validation checks fences, Python syntax, JSON syntax, HTML tags, and JS/TS/CSS structure. It never
executes code or proves runtime correctness. One repair can replace invalid code. Before enabling streaming
repair, the frontend must support this SSE event:

```json
{"type":"content_replace","content":"```python\ndef add(a, b):\n    return a + b\n```"}
```

Replace all previously streamed content, then process `done` normally. Until that support ships, streaming
validation only logs errors; `/agent/process` can repair its result in place.
When validation is enabled, a non-streaming answer that remains invalid returns HTTP 422 with
`{"error": "The generated code did not pass validation", "issues": ["..."]}`. This also applies when
repair is disabled. The evaluation runner scores the final replacement when a streaming repair succeeds.

Rollout flags (set any to `false` to switch that piece off without a deploy):

| Environment variable | Default | Purpose |
|---|---|---|
| `TASK_ROUTING_ENABLED` | `true` | Apply task defaults and filter eligible providers |
| `PROVIDER_FALLBACK_ENABLED` | `true` | Retry 429 once and switch providers before output begins |
| `CODE_INTENT_REGEX_ENABLED` | `true` | Route explicit code requests inside `general` |
| `CODE_INTENT_CLASSIFIER_ENABLED` | `true` | Resolve ambiguous code mentions with a bounded lightweight model |
| `CODE_INTENT_SHADOW_MODE` | `false` | Log intent without changing the selected agent |
| `CODE_VALIDATION_ENABLED` | `true` | Check generated code after completion |
| `CODE_REPAIR_ENABLED` | `true` | Allow one repair |
| `CODE_REPAIR_STREAM_EVENT` | `false` | Allow streaming repair and `content_replace` (enable once the frontend handles it) |

`CODE_CLASSIFIER_TIMEOUT_SECONDS` defaults to `1.5`. Platform classification uses the currently configured
Claude autocomplete model; BYOK uses the user's provider's autocomplete model and key. Ordinary content
without code cues never calls the classifier. Timeouts/errors/malformed answers fall back to content.
Mixed requests such as an article containing a code example stay with the writing agent. Image intent wins.

`TASK_ROUTE_DEFAULTS` and `TASK_FALLBACK_TREES` accept JSON objects with `code`/`writing`/`content` keys and provider
names (`openai`, `claude`, `gemini`, `grok`, `deepseek`). A default of `null` keeps the submitted provider;
`<requested>` in a fallback tree refers to that provider. Invalid overrides are logged and ignored at startup.
Fable (`claude-fable-5-1`) remains blocked regardless of overrides.

Traces record task, model source, intent, candidates, actual model, fallbacks, validation and available token
usage. Configure `TEXT_PRICES_PER_MTOK` as `{"model-id": [input_USD_per_million, output_USD_per_million]}` using
verified prices. Unknown prices are `null`, not zero. Metrics group latency by task and actual model.

Run `python scripts/evaluate_code.py` for the 195-case intent corpus. For paid quality evaluation, run
`python scripts/evaluate_code.py --live`; it has 30 prompts per language group (Python, JS/TS, HTML/CSS, SQL)
per selected provider and saves code, validation, timing, tokens and estimated cost to
`local_run/code_evaluation.json`. Use `--limit 1` for a smoke check. Generated code is never executed.
Review those outputs side by side before production rollout. Claude IDs remain unchanged until new IDs can
be verified against the live model list.

## Visually rich blog generation (`generate_blog`)

`type: "generate_blog"` on `/agent/stream` and `/agent/process` writes an editorial post with at least one table,
several pull quotes and real inline images, instead of a wall of prose. It ships **disabled**: with
`BLOG_AI_ENABLED=false` both endpoints answer **HTTP 503** `{"error": "AI blog generation is disabled"}`.

### Request

| Field | Type | Notes |
|---|---|---|
| `type` | string | `generate_blog` |
| `query` | string | The topic or a short brief, at most 5,000 characters (longer: HTTP 400) |
| `secondary_prompt` | string | Optional style notes, for example "friendly, no jargon" |
| `blog_options` | object | Optional, only valid with this type (any other type: HTTP 400). Unknown keys: HTTP 400 |
| `image_params`, `image_provider`, `image_model`, `document_id` | | As for chat images. They override the blog's picture defaults (16:9, medium quality) |

`blog_options` (every field optional):

| Field | Values | Meaning |
|---|---|---|
| `length` | `short` / `standard` / `long` | About 800 / 1,450 / 2,200 words (default `standard`) |
| `images` | `0`-`10` or `"auto"` | Pictures wanted. `auto` is one plus one per 700 words. Capped by `IMAGE_MAX_PER_REQUEST` (4) and by the section count |
| `tone`, `audience` | string | Passed to the writer as context |

The user's own words in `query` win over the options, because code reads them before the model writes anything:
"a 2,000-word guide" sets the length (clamped to 300-4,000), "with 3 images" the picture count,
"without images" or "text-only" turns pictures off. When no image provider is usable for the account the post is
planned with no pictures. If the user explicitly asked for more pictures than the post gets, `done.warnings` says so.

### What the post contains

Plain GitHub-flavoured Markdown, in this order: `# Title`, an italic subtitle line, the introduction, then one `##`
heading per section and a final `##` conclusion. Code, not the model, decides the visuals before drafting:

| Visual | Rule | Markdown |
|---|---|---|
| Table | at least 1 (2 from 2,000 words), every row complete | `\| a \| b \|` with a header and delimiter row |
| Pull quote | at least 2 (3 from 2,000 words) | `> one key sentence` |
| Image | exactly the planned count, at most one per section, after the section's first paragraph | `![alt](placeholder::<uuid>)` then a line `*caption*` |

Pull quotes are the author's own unattributed sentences. A quote is attributed to a source only when the research
tools actually read that page in this request, and then it carries a link to it; any other attribution is removed.
URLs appear only for pages that were read. No part of the post contains more than `BLOG_MAX_PROSE_RUN` (220) words of
prose without a visual break, unless a part could not be fixed (reported in `warnings`).

### `/agent/stream` events

`start` -> `content` (the title block, alone) -> **`outline`** -> `content`... -> `done` -> `image_ready` / `image_failed`... -> `stream_end`

The title block opens the post the moment the model has written it, while the rest of the outline is still being generated
(`BLOG_STREAM_TITLE`, default on; with `gpt-5.2` that is about 2 s after the request instead of 13-15 s). It is the same
title and subtitle the finished post carries, so nothing is retracted. With `BLOG_STREAM_TITLE=false` the order is
`start` -> `outline` -> `content`..., as before.

`outline` is additive and arrives as soon as the structure is validated, so the editor can draw a skeleton at once:

```json
{"type": "outline", "title": "Remote meetings people actually want to join",
 "sections": [{"id": "s1", "heading": "Why most remote meetings lose people early"}, {"id": "conclusion", "heading": "Make every meeting earn its place"}]}
```

`content` chunks follow in document order: first the title block, then the introduction, each section, the
conclusion. Every chunk has already passed the checks above, so none is ever retracted. Pictures start generating
right after the outline, in parallel with drafting, and arrive through the existing `image_ready` swap
(`placeholder_id` is the uuid in the Markdown). `stream_end` is sent after the pictures, as for any request that
started pictures.

`done` carries `warnings` only when there is something to report:

```json
{"type": "done", "duration": 31.2, "pending_images": 3,
 "warnings": [{"kind": "unit_lint", "unit": "s2", "detail": "table_missing: The assigned table is missing. ..."}]}
```

| `kind` | Meaning |
|---|---|
| `unit_lint` | A part still broke a rule after its one repair (`unit` names it) and was delivered as written |
| `outline_filled` | A table or picture the plan requires was still missing after the repair, so code added a generic one (lossless fixes, such as an extra pull-quote directive or rescaled section lengths, cost no repair and are not reported) |
| `outline_rules` | The outline still broke a rule after the fill (should not happen) |
| `quota_tables`, `quota_quotes`, `heading_mismatch` | The finished post is short of the plan |
| `image_skipped` | A picture could not be started (`unit` names the section; the text carries no placeholder for it) |
| `images_limited` | More pictures were asked for than the post gets |

Clients that ignore `outline` and `warnings` keep working unchanged. If the outline cannot be produced even after a
repair, or a part comes back empty, the stream ends with an `error` event; nothing is billed for pictures that never
started, and credits reserved for pictures that did are released.

### `/agent/process`

Same pipeline, same provider key contract as the other types (the backend owns billing). The response waits for the
pictures (up to `IMAGE_WAIT_SECONDS`, 120 s), writes their URLs into `markdown_response`, and returns `warnings`. A post
with pictures can take well over a minute: keep the client timeout above 3 minutes. A failed picture becomes
`*(image could not be generated)*`, never a dead link. An unusable outline is HTTP 500 with the reason in `error`.

### Billing

Text is billed per streamed character exactly as before. Each picture is priced for the model that will draw it and its
credits are reserved right after the outline (platform keys only), then settled by the existing image billing; BYOK
accounts reserve nothing. The post is a single request: `operation_type` is `mixed` when pictures finished.

### Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `BLOG_AI_ENABLED` | `false` | `generate_blog` answers 503 when off |
| `BLOG_IMAGE_SOURCE` | `generate` | `generate` draws pictures with the image adapter; `none` (or `stock`, not built yet) means no pictures |
| `BLOG_SECTION_PARALLEL` | `3` | Parts drafted at the same time per request (1-8) |
| `BLOG_MAX_PROSE_RUN` | `220` | Words allowed between visual breaks |
| `BLOG_REPAIR_ENABLED` | `true` | One repair call per failing outline or part |
| `BLOG_STREAM_TITLE` | `true` | Open the post with its title and subtitle as soon as the model has written them, while the rest of the outline is still streaming. The post is identical either way; `false` sends the title only after the whole outline has validated (and the `outline` event first) |
| `BLOG_OUTLINE_MODEL` | *(empty)* | Model for the **first** outline attempt only: `provider` or `provider:tier` (for example `gemini:basic`; tier `basic` or `advanced`, default `basic`). Empty means the drafting model. It is used only if the request may use that provider (a BYOK account needs its own key for it; `/agent/process` has just the one key it was given); if it is unavailable or fails, the outline falls back to the drafting providers. The outline repair and every part always run on the drafting model, and billing/attribution follow the model that writes the post. Invalid values are ignored with a warning |
| `BLOG_INTENT_REGEX_ENABLED` | `false` | Route "write a blog post about ..." typed into `general` chat to this pipeline |
| `BLOG_INTENT_SHADOW_MODE` | `false` | Log that decision without acting on it (run this for a week first) |

The writing task defaults to OpenAI (advanced tier) with the usual fallback tree, unless the user chose a model. Each
call runs the same retry and fallback rules as the other streams, and a provider that failed once is not retried by the
parts that follow. Every model call states its output budget (see `get_model(max_tokens=...)`), so long outputs are not
cut at a provider default.

Run `python scripts/evaluate_blog.py` for a free check of the structure rules over 30 briefs, and
`python scripts/evaluate_blog.py --live --providers openai claude --limit 3 --images` to compare providers on real posts
(paid; the posts are written to `local_run/blog_eval/` for side-by-side reading and pictures to `local_run/images/`).

To judge saved posts by eye, `python scripts/render_blog_preview.py --out local_run/blog_preview.html --compare "Sleep routine" old.md new.md`
(repeat `--compare` for more briefs, or `--post LABEL post.md` for one post) writes a single self-contained page: pictures
inlined, a page-rhythm bar and an at-a-glance table beside each post, and a table whose delimiter row is wrong shown as the
plain text a reader would see, flagged. It uses a generic CommonMark renderer, not the editor's, so it says nothing about
how the editor draws a picture followed by an italic caption line (open item O2 of the plan).
