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

#### Response

**Success (200)**
```json
{
  "success": true,
  "html_response": "<p>Processed HTML content...</p>"
}
```

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

Providers: OpenAI, Gemini and Grok use the user's own key. Claude and DeepSeek have no image endpoint, so they use the platform's `DEFAULT_GEMINI_API_KEY`; if that is unset, or the user brings their own keys, the request stays on the agent path instead of failing.

### 2. Settings and how they are chosen
Precedence: **explicit `image_params` field > words in the prompt > previous image's settings (follow-ups) > default.**

| Setting | Default | Prompt words that change it |
|---|---|---|
| `aspect_ratio` | `1:1` | `16:9`, `widescreen`, `banner`, `horizontal`, `in landscape`, `landscape image/photo/format…` → `16:9`; `9:16`, `vertical`, `instagram story`, `reel`, `tiktok`, `in portrait`, `portrait image/photo/format…` → `9:16`; `4:3`, `3:4`, `square` |
| `quality` | `medium` | `draft`, `low quality` → `low`; `high quality`, `best quality`, `ultra-detailed`, `hq` → `high` |
| `resolution` | `1K` | `2k`, `high-res` → `2K`; `4k`, `uhd` → `4K` (Gemini only; OpenAI uses fixed sizes, Grok ignores it) |
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
- `image_params` (optional): `aspect_ratio` (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`), `quality` (`low`/`medium`/`high`), `resolution` (`1K`/`2K`/`4K`), `format` (`jpeg`/`png`/`webp`), `transparent_bg` (bool). Invalid keys or values return HTTP 422.

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
