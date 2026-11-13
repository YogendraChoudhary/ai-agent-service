"""
Centralised application configuration via pydantic-settings.

Every environment variable the API needs is declared here with:
  - a type  (bool, float, int, str | None ...) so coercion happens at startup
  - a default so the app degrades gracefully where possible
  - a docstring comment explaining what the variable does

Import the singleton `settings` object everywhere instead of calling
os.getenv() directly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Populate os.environ so all modules using os.getenv() get values regardless of cwd
_dotenv_path = find_dotenv()
if _dotenv_path:
    load_dotenv(_dotenv_path)


# BLOG_OUTLINE_MODEL: "provider" or "provider:tier". One pattern for the setting's validator and the parser in
# src/blog/runtime.py, so a value that loads is a value that parses.
OUTLINE_MODEL_PATTERN = r'(openai|claude|gemini|grok|deepseek)(?::(basic|advanced))?'


class Settings(BaseSettings):
    task_routing_enabled: bool = Field(True, alias="TASK_ROUTING_ENABLED")
    provider_fallback_enabled: bool = Field(True, alias="PROVIDER_FALLBACK_ENABLED")
    task_route_defaults: str = Field('{}', alias="TASK_ROUTE_DEFAULTS")
    task_fallback_trees: str = Field('{}', alias="TASK_FALLBACK_TREES")
    code_intent_regex_enabled: bool = Field(True, alias="CODE_INTENT_REGEX_ENABLED")
    code_intent_classifier_enabled: bool = Field(True, alias="CODE_INTENT_CLASSIFIER_ENABLED")
    code_intent_shadow_mode: bool = Field(False, alias="CODE_INTENT_SHADOW_MODE")

    # ------------------------------------------------------------------
    # Mode 8: Editable UI Design Engine (src/design_canvas)
    # ------------------------------------------------------------------
    design_canvas_enabled: bool = Field(True, alias="DESIGN_CANVAS_ENABLED")
    design_max_repair_attempts: int = Field(2, ge=0, le=5, alias="DESIGN_MAX_REPAIR_ATTEMPTS")
    # How hard Claude's advanced tier (claude-opus-5-5) thinks: low | medium | high | xhigh | max. Thinking cannot be switched off on
    # this model and is billed as output, so this is the speed/cost/quality dial. "low" measured as good as "medium" on writing, design
    # canvas and code, and faster and cheaper (docs/MODEL_UPGRADE_BENCHMARK.md, section 12). The model's own default would be medium.
    claude_advanced_effort: str = Field("low", alias="CLAUDE_ADVANCED_EFFORT")
    # reasoning_effort for OpenAI's advanced tier (gpt-6.1-terra): none | low | medium | high | xhigh. "none" measured fastest and as good
    # as "low" on writing; at the model's default effort the tail was far slower (docs/MODEL_UPGRADE_BENCHMARK.md, section 14).
    openai_advanced_effort: str = Field("none", alias="OPENAI_ADVANCED_EFFORT")
    # At startup, look up every configured model and platform key in a background thread (free calls) and log ROLLOUT_ALERT lines for
    # anything missing or rejected; the result is served at GET /agent/health/providers. scripts/preflight_models.py runs the same check.
    model_preflight_on_start: bool = Field(True, alias="MODEL_PREFLIGHT_ON_START")
    # Same dial for the basic tier (claude-sonnet-5-5: quick edits, summaries, long-document passages). Empty = the model's own default.
    claude_basic_effort: str = Field("", alias="CLAUDE_BASIC_EFFORT")
    code_classifier_timeout_seconds: float = Field(1.5, gt=0, alias="CODE_CLASSIFIER_TIMEOUT_SECONDS")
    code_validation_enabled: bool = Field(True, alias="CODE_VALIDATION_ENABLED")
    code_repair_enabled: bool = Field(True, alias="CODE_REPAIR_ENABLED")
    # On: the frontend must handle the content_replace SSE event (a repaired answer replaces the streamed one).
    code_repair_stream_event: bool = Field(True, alias="CODE_REPAIR_STREAM_EVENT")
    text_prices_per_mtok: str = Field('{}', alias="TEXT_PRICES_PER_MTOK")

    @field_validator('voice_llm_mode')
    @classmethod
    def validate_voice_llm_mode(cls, value):
        value = (value or '').strip().lower()
        if value not in {'smart', 'always'}:
            logging.getLogger(__name__).warning('Invalid VOICE_LLM_MODE %r; using smart', value)
            return 'smart'
        return value

    @field_validator('voice_default_style')
    @classmethod
    def validate_voice_style(cls, value):
        value = (value or '').strip().lower()
        if value not in {'clean', 'polished'}:
            logging.getLogger(__name__).warning('Invalid VOICE_DEFAULT_STYLE %r; using clean', value)
            return 'clean'
        return value

    @field_validator('claude_advanced_effort')
    @classmethod
    def validate_claude_effort(cls, value):
        value = (value or '').strip().lower()
        if value not in {'low', 'medium', 'high', 'xhigh', 'max'}:
            logging.getLogger(__name__).warning('Invalid CLAUDE_ADVANCED_EFFORT %r; using low', value)
            return 'low'
        return value

    @field_validator('openai_advanced_effort')
    @classmethod
    def validate_openai_effort(cls, value):
        value = (value or '').strip().lower()
        if value not in {'none', 'low', 'medium', 'high', 'xhigh'}:
            logging.getLogger(__name__).warning('Invalid OPENAI_ADVANCED_EFFORT %r; using none', value)
            return 'none'
        return value

    @field_validator('claude_basic_effort')
    @classmethod
    def validate_claude_basic_effort(cls, value):
        value = (value or '').strip().lower()
        if value and value not in {'low', 'medium', 'high', 'xhigh', 'max'}:
            logging.getLogger(__name__).warning('Invalid CLAUDE_BASIC_EFFORT %r; using the model default', value)
            return ''
        return value

    @field_validator('voice_format_provider')
    @classmethod
    def validate_voice_provider(cls, value):
        value = (value or '').strip().lower()
        if value not in {'gemini', 'openai'}:
            logging.getLogger(__name__).warning('Invalid VOICE_FORMAT_PROVIDER %r; using gemini', value)
            return 'gemini'
        return value

    @field_validator('blog_outline_model')
    @classmethod
    def validate_outline_model(cls, value):
        value = (value or '').strip().lower()
        if value and not re.fullmatch(OUTLINE_MODEL_PATTERN, value):
            logging.getLogger(__name__).warning('Invalid BLOG_OUTLINE_MODEL %r; using the drafting model', value)
            return ''
        return value

    @field_validator('longform_word_ratio')
    @classmethod
    def validate_word_ratio(cls, value):
        """{"grammar": [0.9, 1.1]}: operation -> [lowest, highest] words out / words in. Anything else is ignored."""
        value = (value or '').strip()
        if not value:
            return ''
        try:
            data = json.loads(value)
            ok = isinstance(data, dict) and all(
                k in {'enhance', 'grammar', 'tone', 'simplify'} and isinstance(v, list) and len(v) == 2
                and all(type(n) in (int, float) for n in v) and 0 < v[0] <= 1 <= v[1] for k, v in data.items())
            if not ok:
                raise ValueError('invalid entries')
        except (ValueError, TypeError):
            logging.getLogger(__name__).warning('Invalid LONGFORM_WORD_RATIO %r; using the per-operation windows', value)
            return ''
        return value

    @field_validator('task_route_defaults', 'task_fallback_trees', 'text_prices_per_mtok')
    @classmethod
    def validate_route_json(cls, value, info):
        providers = {'openai', 'claude', 'gemini', 'grok', 'deepseek'}
        try:
            data = json.loads(value)
            if not isinstance(data, dict):
                raise ValueError('expected an object')
            if info.field_name == 'task_route_defaults':
                valid = all(k in {'code', 'writing', 'content'} and (v is None or v in providers) for k, v in data.items())
            elif info.field_name == 'task_fallback_trees':
                valid = all(k in {'code', 'writing', 'content'} and isinstance(v, list)
                            and all(isinstance(p, str) and p in providers | {'<requested>'} for p in v)
                            for k, v in data.items())
            else:
                valid = all(isinstance(v, list) and len(v) == 2
                            and all(type(n) in (int, float) and 0 <= n < float('inf') for n in v)
                            for v in data.values())
            if not valid:
                raise ValueError('invalid entries')
        except (ValueError, TypeError):
            logging.getLogger(__name__).warning('Invalid %s; using defaults', info.field_name)
            return '{}'
        return value

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")

    # ------------------------------------------------------------------
    # Backend webhooks
    # ------------------------------------------------------------------
    backend_webhook_url: str = Field("", alias="BACKEND_WEBHOOK_URL")
    webhook_secret: str = Field("", alias="WEBHOOK_SECRET")
    # Stays off until the receiver dedupes Idempotency-Key with the charge and answers repeats with 2xx (docs/BILLING_DELIVERY.md).
    billing_webhook_idempotent: bool = Field(False, alias="BILLING_WEBHOOK_IDEMPOTENT")

    # ------------------------------------------------------------------
    # CDN / S3
    # ------------------------------------------------------------------
    cdn_url: str = Field("", alias="CDN_URL")
    aws_s3_bucket: str = Field("", alias="AWS_S3_BUCKET")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")

    # ------------------------------------------------------------------
    # Default (platform-funded) AI API keys
    # ------------------------------------------------------------------
    default_openai_api_key: Optional[str] = Field(None, alias="DEFAULT_OPENAI_API_KEY")
    default_anthropic_api_key: Optional[str] = Field(None, alias="DEFAULT_ANTHROPIC_API_KEY")
    default_grok_api_key: Optional[str] = Field(None, alias="DEFAULT_GROK_API_KEY")
    default_gemini_api_key: Optional[str] = Field(None, alias="DEFAULT_GEMINI_API_KEY")
    default_deepseek_api_key: Optional[str] = Field(None, alias="DEFAULT_DEEPSEEK_API_KEY")

    # ------------------------------------------------------------------
    # Credit system
    # ------------------------------------------------------------------
    credit_precheck_enabled: bool = Field(True, alias="CREDIT_PRECHECK_ENABLED")
    credit_settle_delay_seconds: float = Field(3.0, alias="CREDIT_SETTLE_DELAY_SECONDS")
    max_reservation_per_request: int = Field(100, alias="MAX_RESERVATION_PER_REQUEST")

    # ------------------------------------------------------------------
    # Image handling & adapter
    # ------------------------------------------------------------------
    max_reference_images: int = Field(3, alias="MAX_REFERENCE_IMAGES")
    image_adapter_enabled: bool = Field(True, alias="IMAGE_ADAPTER_ENABLED")
    image_router_regex_enabled: bool = Field(True, alias="IMAGE_ROUTER_REGEX_ENABLED")
    image_router_shadow_mode: bool = Field(False, alias="IMAGE_ROUTER_SHADOW_MODE")
    image_memory_enabled: bool = Field(True, alias="IMAGE_MEMORY_ENABLED")
    image_default_format: str = Field("jpeg", alias="IMAGE_DEFAULT_FORMAT")
    # Image provider used when the user's request names none (chat model never decides this).
    image_default_provider: str = Field("gemini", alias="IMAGE_DEFAULT_PROVIDER")
    image_default_provider_fallback: str = Field("gemini", alias="IMAGE_DEFAULT_PROVIDER_FALLBACK")
    image_memory_ttl_seconds: int = Field(3600, alias="IMAGE_MEMORY_TTL_SECONDS")

    # ------------------------------------------------------------------
    # Deterministic request planning (src/request_plan.py): the code, not the LLM, decides how many
    # images to make, whether web tools exist, and which model tier answers.
    # ------------------------------------------------------------------
    image_max_per_request: int = Field(4, alias="IMAGE_MAX_PER_REQUEST")
    image_default_plural_count: int = Field(2, alias="IMAGE_DEFAULT_PLURAL_COUNT")  # "with images" and no number
    web_tools_gate_enabled: bool = Field(True, alias="WEB_TOOLS_GATE_ENABLED")      # False = always attach web tools
    web_search_call_limit: int = Field(6, alias="WEB_SEARCH_CALL_LIMIT")
    web_read_call_limit: int = Field(8, alias="WEB_READ_CALL_LIMIT")
    model_tier_routing_enabled: bool = Field(True, alias="MODEL_TIER_ROUTING_ENABLED")  # False = always advanced
    tier_long_query_chars: int = Field(1500, alias="TIER_LONG_QUERY_CHARS")
    document_context_enabled: bool = Field(True, alias="DOCUMENT_CONTEXT_ENABLED")
    image_wait_seconds: int = Field(120, alias="IMAGE_WAIT_SECONDS")                # /agent/process waits this long
    # JSON object of per-image credit overrides in scaled units, e.g. {"openai/gpt-image-1": 80}
    image_credits_by_model: str = Field("", alias="IMAGE_CREDITS_BY_MODEL")

    # ------------------------------------------------------------------
    # Document actions: document_action "auto", exact-text Draft and in-line Edit additions (src/document_action.py,
    # src/document_literal.py, src/document_edit/, src/intent_classifier.py). Every number those features added is here; the defaults are
    # the measured ones (docs/DOCUMENT_CONTEXT_API_HANDOFF.md, "Auto mode"). Blank values are not accepted: leave a variable unset to
    # keep the default.
    # ------------------------------------------------------------------
    # The light-tier call that picks Ask / Draft / Edit / Write when the wording alone does not. Off = unclear wording resolves to Ask.
    document_action_classifier_enabled: bool = Field(True, alias="DOCUMENT_ACTION_CLASSIFIER_ENABLED")
    # Longer than the code classifier's: a wrong mode costs the user more than a slower right one.
    document_action_classifier_timeout_seconds: float = Field(4.0, gt=0, le=30, alias="DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS")
    # Decisions that can be waiting on the provider at once (worker threads and slots; read once at start-up). A burst waits up to the
    # timeout for a slot before the request falls back to Ask.
    document_action_classifier_concurrency: int = Field(16, ge=1, le=64, alias="DOCUMENT_ACTION_CLASSIFIER_CONCURRENCY")
    # How much of a task the Auto rules and the classifier read: its first and last half. Bounds the work done before the licence check.
    # The classifier call is cut at 1,000 characters including a ~50-character hint, so this may not exceed 940.
    document_action_digest_chars: int = Field(900, ge=200, le=940, alias="DOCUMENT_ACTION_DIGEST_CHARS")
    # Draft: a request longer than this is never treated as "exact text" (such a request is one short sentence).
    document_literal_max_request_chars: int = Field(400, ge=50, le=2000, alias="DOCUMENT_LITERAL_MAX_REQUEST_CHARS")
    # Edit, applying a stored patch: new numbers are accepted only in a pure insertion of at most this many characters.
    document_edit_small_insertion_chars: int = Field(80, ge=1, le=500, alias="DOCUMENT_EDIT_SMALL_INSERTION_CHARS")
    # Edit, an in-line addition ("add (2024) next to the title"): at most this many words may be added to the block.
    document_edit_small_addition_max_words: int = Field(12, ge=1, le=100, alias="DOCUMENT_EDIT_SMALL_ADDITION_MAX_WORDS")

    # ------------------------------------------------------------------
    # Document upload: POST /agent/documents/upload (src/document_upload/router.py). Blank values are not accepted.
    # ------------------------------------------------------------------
    # Extractions (decode, validate, headings, hash) that may run at once, each on one of this many dedicated threads (read once at
    # start-up). The work is pure-Python CPU, so extra threads add no speed: they only take the GIL from the event loop that serves
    # every other request and stream.
    document_upload_max_concurrent: int = Field(2, ge=1, le=16, alias="DOCUMENT_UPLOAD_MAX_CONCURRENT")
    # Files up to this many bytes are extracted on one extra thread of their own, so a small upload never waits behind large ones
    # (a 100 KB file can cost ~0.4 s of CPU, a 16 KB one at most ~60 ms). Read at request time; bytes, so 4-byte characters count in full.
    document_upload_small_file_bytes: int = Field(16_000, ge=1_000, le=400_000, alias="DOCUMENT_UPLOAD_SMALL_FILE_BYTES")
    # A request that has waited this long for a free extraction thread is answered 503 + Retry-After instead of queueing behind a burst.
    document_upload_queue_wait_seconds: float = Field(10.0, gt=0, le=60, alias="DOCUMENT_UPLOAD_QUEUE_WAIT_SECONDS")

    # ------------------------------------------------------------------
    # Long-document Ask (src/document_retrieval.py, src/document_context.py). Blank values are not accepted.
    # ------------------------------------------------------------------
    # The most UTF-8 bytes of context (the JSON envelope: excerpts, references, framing) a scoped Ask (document_scope "focused" or "auto") sends
    # to the model. It replaced a fixed 6,000, which fits about three passages. Measured on 39 questions over real 26-88 KB documents (share of questions
    # whose answer-bearing passage was in the context): 72% before (6,000, no stemming), 79% at 6,000 with stemming, 90% at 8,000, 95% at 10,000, 97% at
    # 11,000 and again at 12,000 or 24,000, so 10,000 is the knee. Credits are priced by the characters sent: this is also the most a scoped Ask can cost
    # in input. Never above the 24,000-byte context ceiling.
    document_retrieval_budget_bytes: int = Field(10_000, ge=4_000, le=24_000, alias="DOCUMENT_RETRIEVAL_BUDGET_BYTES")
    # What an Ask with the default scope does when the document is too large to send whole. "refuse" (the documented contract): answer needs_context
    # without a model call. "auto": behave as document_scope "auto" (retrieval for a question about a part, still refusing one that needs the whole
    # document), reporting partial coverage. Lets every existing client get long-document Q&A without sending a new field.
    document_large_ask: Literal["refuse", "auto"] = Field("refuse", alias="DOCUMENT_LARGE_ASK")

    # ------------------------------------------------------------------
    # Document import: POST /agent/documents/import (src/document_import/router.py): DOCX, PDF and HTML files, and web addresses, to Markdown.
    # Every conversion runs in its own locked-down child process. Blank values are not accepted. Pool sizes and the size cap are read once at start-up.
    # ------------------------------------------------------------------
    # false = the endpoint answers 503 (switches the whole feature off without a deploy).
    document_import_enabled: bool = Field(True, alias="DOCUMENT_IMPORT_ENABLED")
    # false = files can still be imported but a "url" is refused (the part of the feature that makes outgoing requests).
    document_import_urls_enabled: bool = Field(True, alias="DOCUMENT_IMPORT_URLS_ENABLED")
    # Conversions that may run at once. Each is a child process (a PDF can take ~150 MB and several seconds of one CPU), so this is also the cap on memory.
    document_import_max_concurrent: int = Field(2, ge=1, le=8, alias="DOCUMENT_IMPORT_MAX_CONCURRENT")
    # Web addresses fetched at once. Fetching waits on the network and holds no conversion slot, so it has its own, larger, pool.
    document_import_fetch_concurrent: int = Field(4, ge=1, le=16, alias="DOCUMENT_IMPORT_FETCH_CONCURRENT")
    # Largest DOCX or PDF (and largest body of the request), in bytes. A web page itself is capped at 5 MB.
    document_import_max_file_bytes: int = Field(15_000_000, ge=100_000, le=50_000_000, alias="DOCUMENT_IMPORT_MAX_FILE_BYTES")
    # A conversion that runs longer than this is killed and the request answered 422.
    document_import_timeout_seconds: float = Field(30.0, ge=3, le=120, alias="DOCUMENT_IMPORT_TIMEOUT_SECONDS")
    # Pages read from a PDF; a longer one is imported up to this page and says so.
    document_import_max_pdf_pages: int = Field(200, ge=1, le=1000, alias="DOCUMENT_IMPORT_MAX_PDF_PAGES")
    # Memory limit of a conversion's child process (enforced where the OS supports RLIMIT_AS, i.e. Linux; not macOS).
    document_import_memory_mb: int = Field(1536, ge=256, le=8192, alias="DOCUMENT_IMPORT_MEMORY_MB")

    # ------------------------------------------------------------------
    # AI email templates (src/email_template). The model writes JSON blocks; code validates them.
    # ------------------------------------------------------------------
    email_ai_enabled: bool = Field(True, alias="EMAIL_AI_ENABLED")                # False = /agent/email/* return 503
    email_ai_repair_enabled: bool = Field(True, alias="EMAIL_AI_REPAIR_ENABLED")  # one "fix these errors" retry
    email_ai_max_variants: int = Field(3, alias="EMAIL_AI_MAX_VARIANTS")
    # Shared secret for server-to-server callers, sent in the X-Agent-Backend-Secret header.
    # * /agent/process (provider key, billed by the caller): when set, every call must carry it (401 otherwise). When
    #   empty the route stays open as before and startup logs a warning: set it on the backend first, then here.
    # * /agent/email/*: these take the Eddyter licence key; only a caller with this secret may pass a provider key
    #   instead (no licence check, no billing here). Empty = that mode is off.
    agent_backend_secret: str = Field("", alias="AGENT_BACKEND_SECRET")

    # ------------------------------------------------------------------
    # Visually rich blog generation (src/blog). The model writes an outline and section bodies; code decides
    # the visuals, writes the headings and image markup, and lints the result. On by default.
    # ------------------------------------------------------------------
    blog_ai_enabled: bool = Field(True, alias="BLOG_AI_ENABLED")                 # False = generate_blog returns 503
    blog_image_source: str = Field("generate", alias="BLOG_IMAGE_SOURCE")          # generate | stock | none
    blog_section_parallel: int = Field(3, ge=1, le=8, alias="BLOG_SECTION_PARALLEL")  # units in flight per request
    blog_max_prose_run: int = Field(220, ge=60, alias="BLOG_MAX_PROSE_RUN")        # words allowed between visual breaks
    blog_repair_enabled: bool = Field(True, alias="BLOG_REPAIR_ENABLED")           # one repair per failing unit / outline
    # Model for the FIRST outline attempt only, "provider" or "provider:tier" (tier basic|advanced, default basic), e.g.
    # "gemini:basic". Empty = the drafting model. The outline is ~1,000 tokens of JSON, so its time is generation speed.
    # Used only if the request may use that provider (BYOK accounts: their own key); the repair always runs on the drafting model.
    blog_outline_model: str = Field("", alias="BLOG_OUTLINE_MODEL")
    # Open the post with its title and subtitle the moment the model has written them, while the rest of the outline is still
    # streaming in (the outline is ~1,000 tokens, so first text no longer waits for all of it). The content is the same either
    # way; False opens the post only once the whole outline has validated.
    blog_stream_title: bool = Field(True, alias="BLOG_STREAM_TITLE")
    blog_intent_regex_enabled: bool = Field(True, alias="BLOG_INTENT_REGEX_ENABLED")  # promote "write a blog post about..." from general
    blog_intent_shadow_mode: bool = Field(False, alias="BLOG_INTENT_SHADOW_MODE")  # log the promotion decision only

    # ------------------------------------------------------------------
    # Structure-preserving refinement of long documents (src/longform) for enhance_with_ai, fix_grammar, change_tone and
    # simplify. A qualifying document is frozen (tables, quotes, images, code and links become tokens), cut into chunks,
    # refined in parallel, checked in code and restored where the model moved something. On by default.
    # ------------------------------------------------------------------
    # Additive graph suggestions for Enhance; an Enhance instruction that asks for a chart ("add a bar chart") always
    # gets the editable chart or a "can't chart this" event, never a table. Write mode keeps its Mermaid output. Enable when the editor
    # handles graph_suggestion events and persists eddyter_chart JSON (docs/GRAPH_API_HANDOFF.md). On by default: the
    # editor team asked for it in production. No charts are automatically inserted into the prose.
    chart_suggestions_enabled: bool = Field(True, alias="CHART_SUGGESTIONS_ENABLED")
    chart_extraction_timeout_seconds: float = Field(25.0, gt=0, le=60, alias="CHART_EXTRACTION_TIMEOUT_SECONDS")
    longform_refine_enabled: bool = Field(True, alias="LONGFORM_REFINE_ENABLED")
    longform_refine_min_words: int = Field(600, ge=1, alias="LONGFORM_REFINE_MIN_WORDS")   # shorter and unstructured: the single call
    longform_chunk_words: int = Field(1000, ge=100, le=3000, alias="LONGFORM_CHUNK_WORDS")  # target chunk size (window 0.6x-1.4x)
    longform_first_chunk_words: int = Field(300, ge=0, le=3000, alias="LONGFORM_FIRST_CHUNK_WORDS")  # smaller first chunk = text sooner; 0 = off
    longform_parallel: int = Field(5, ge=1, le=8, alias="LONGFORM_PARALLEL")                # chunks in flight per request (measured: 5 beats 3 by a third)
    longform_freeze_quotes: bool = Field(True, alias="LONGFORM_FREEZE_QUOTES")             # False = quotes are polished, still verified
    longform_repair_enabled: bool = Field(True, alias="LONGFORM_REPAIR_ENABLED")           # one repair call per failing chunk
    longform_word_ratio: str = Field("", alias="LONGFORM_WORD_RATIO")                      # JSON override of the ratio windows
    # Shorten (make_shorter) runs on the advanced-tier model: measured on a real document, the basic tier shortened a passage by 2-9% whatever it
    # was told, the advanced tier by 15-30% with every structure check passing. Credits are priced per character, not per tier, so this costs the
    # platform more per request on managed plans (not the user). False = every operation uses the tier its single call uses.
    longform_tier_overrides: bool = Field(True, alias="LONGFORM_TIER_OVERRIDES")
    longform_call_timeout_seconds: float = Field(150.0, ge=0, le=900, alias="LONGFORM_CALL_TIMEOUT_SECONDS")   # one model call; 0 = no limit

    # ------------------------------------------------------------------
    # Voice dictation (src/voice_flow). Local Whisper (faster-whisper) turns speech into text; a low-end Gemini model
    # compiles it (lists, self-corrections, fillers, cursor-aware casing); code does the final spacing.
    # ------------------------------------------------------------------
    # Off by default: with an empty VOICE_FLOW_TOKEN the routes are unauthenticated and run on the
    # platform Gemini key and this host's CPU (Whisper). Enable only together with a token.
    voice_flow_enabled: bool = Field(False, alias="VOICE_FLOW_ENABLED")            # True = /v1/voice/* is mounted
    voice_flow_token: str = Field("", alias="VOICE_FLOW_TOKEN")                      # non-empty = clients must send it
    voice_whisper_model: str = Field("small", alias="VOICE_WHISPER_MODEL")           # tiny|base|small|... or a local path
    # float32 beat int8 on an Apple M4 (small: 675 ms vs 860-940 ms); on x86 servers int8 is the faster choice.
    voice_whisper_compute_type: str = Field("float32", alias="VOICE_WHISPER_COMPUTE_TYPE")
    voice_whisper_threads: int = Field(8, ge=0, le=64, alias="VOICE_WHISPER_THREADS")   # 0 = library default
    voice_whisper_language: str = Field("en", alias="VOICE_WHISPER_LANGUAGE")         # "" = detect per utterance (slower)
    # smart: plain sentences skip the model (it costs ~1 s per call whatever the prompt); cues, corrections, fillers, email, code go to it.
    voice_llm_mode: str = Field("smart", alias="VOICE_LLM_MODE")                       # smart | always
    # Writing style of the formatted text. clean: keeps the speaker's wording, removes fillers, false starts and repeats.
    # polished: also tightens each list point to "Label: one sentence"; a dropped claim falls back to clean. Per request override: `style`.
    voice_default_style: str = Field("clean", alias="VOICE_DEFAULT_STYLE")             # clean | polished
    # Email / Slack / code / Markdown specific formatting. Off for now: every app_context is treated as "general".
    voice_app_contexts_enabled: bool = Field(False, alias="VOICE_APP_CONTEXTS")
    voice_format_provider: str = Field("gemini", alias="VOICE_FORMAT_PROVIDER")       # gemini | openai (openai: while Gemini is out)
    voice_format_model: str = Field("", alias="VOICE_FORMAT_MODEL")                   # "" = the project's low-end model of that provider
    voice_gemini_api_key: Optional[str] = Field(None, alias="VOICE_GEMINI_API_KEY")   # None = DEFAULT_GEMINI_API_KEY
    voice_format_timeout_seconds: float = Field(4.0, gt=0, le=30, alias="VOICE_FORMAT_TIMEOUT_SECONDS")
    voice_max_utterance_seconds: int = Field(600, ge=5, le=1800, alias="VOICE_MAX_UTTERANCE_SECONDS")   # a longer recording is refused (413)

    # ------------------------------------------------------------------
    # Observability (src/observability.py). Bodies contain user text, so they are OFF by default.
    # ------------------------------------------------------------------
    request_log_file: str = Field("", alias="REQUEST_LOG_FILE")                  # JSONL, one summary per request
    log_request_bodies: bool = Field(False, alias="LOG_REQUEST_BODIES")          # log request + response text (api keys always redacted)
    log_body_max_chars: int = Field(2000, alias="LOG_BODY_MAX_CHARS")
    metrics_endpoint_enabled: bool = Field(False, alias="METRICS_ENDPOINT_ENABLED")  # GET /agent/metrics (local/dev only)
    trace_ring_size: int = Field(500, alias="TRACE_RING_SIZE")

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------
    api_key_encryption_secret: str = Field(
        "default-32-byte-encryption-key!!",
        alias="API_KEY_ENCRYPTION_SECRET",
    )

    model_config = {
        "env_file": (".env", "../.env"),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


# Singleton — import `settings` everywhere; never call os.getenv() directly.
settings = Settings()
