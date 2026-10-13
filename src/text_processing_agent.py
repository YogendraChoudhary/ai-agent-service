"""
Global Text Processing Agent for HTML Editor Integration
Supports ALL operations: grammar, simplify, complete, emojify, translate, articles, images, tables, etc.
Supports dynamic model selection: OpenAI, Claude, Grok, DeepSeek, Gemini
"""

from agno.agent import Agent, RunEvent
from agno.media import Image as AgnoImage
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.xai import xAI
from agno.models.deepseek import DeepSeek
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from .models import (
    ADVANCED_MODEL_IDS,
    AUTOCOMPLETE_MODEL_IDS,
    BASIC_MODEL_IDS,
    DEEPSEEK_TOOL_MODEL_ID,
    IMAGE_MODEL_IDS,
    VISION_CAPABLE_PROVIDERS,
    supports_vision,
)
from dotenv import load_dotenv
import os
import logging
import time
import uuid
import threading
import re
import ipaddress
import socket
from urllib.parse import urlparse

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# ============================================
# DUCKDUCKGO SEARCH WITH RETRY (handles rate limits)
# ============================================

class DuckDuckGoWithRetry(DuckDuckGoTools):
    """
    Wraps agno's DuckDuckGoTools with retry + exponential backoff to handle
    DuckDuckGo rate limiting (HTTP 202 Ratelimit).

    NOTE: agno registers ``web_search`` / ``search_news`` as the actual tool
    functions (``duckduckgo_search`` / ``duckduckgo_news`` are only backward-
    compat aliases assigned in ``__init__`` and are never registered). The
    retry wrapper therefore MUST override ``web_search`` / ``search_news`` —
    overriding the alias names has no effect.
    """

    max_retries: int = 3
    base_delay: float = 2.0  # seconds

    def web_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search the web for a query.

        Args:
            query (str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The search results from the web.
        """
        for attempt in range(self.max_retries):
            try:
                return super().web_search(query, max_results)
            except Exception as e:
                if "Ratelimit" in str(e) and attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"DuckDuckGo search rate-limited, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    raise

    def search_news(self, query: str, max_results: int = 5) -> str:
        """Use this function to get the latest news from the web.

        Args:
            query (str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The latest news from the web.
        """
        for attempt in range(self.max_retries):
            try:
                return super().search_news(query, max_results)
            except Exception as e:
                if "Ratelimit" in str(e) and attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"DuckDuckGo news rate-limited, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    raise


# ============================================
# MARKDOWN CONVERSION INSTRUCTION (Global)
# ============================================
markdown_conversion_instruction = (
    "CRITICAL: Always format your response in clean, well-structured Markdown. "
    "Use appropriate Markdown elements: # for headings, ** for bold, * for italic, - or * for bullet lists, 1. for numbered lists. "
    "For tables, use proper Markdown table syntax with | pipes and --- for headers. "
    "Use ``` for code blocks with language specification (e.g., ```python). "
    "Use > for blockquotes. "
    "Keep formatting clean and readable - this will be streamed directly to the user. "
    "Return ONLY the Markdown content, no HTML tags."
)

# ============================================
# VISION SUPPORT
# ============================================

# VISION_CAPABLE_PROVIDERS and supports_vision() are defined in models.py
# alongside the model IDs they describe, and re-exported here so existing
# callers (api.py) keep working.


def build_agno_images(image_urls, request_id: str = None):
    """
    Convert a list of image URLs into agno Image objects.

    agno hands these to each provider in its own native vision format, so the
    same list works for OpenAI, Claude, Gemini and Grok without branching.

    Returns None (not []) when there are no images — agno treats an empty list
    as "a multimodal turn with no media", which some providers reject.
    """
    if not image_urls:
        return None

    log_prefix = f"[{request_id}] " if request_id else ""
    images = [AgnoImage(url=url) for url in image_urls]
    logger.info(f"{log_prefix}Attached {len(images)} reference image(s) to the run")
    return images


# ============================================
# MODEL FACTORY FUNCTION
# ============================================

def get_model(model_provider: str, api_key: str, use_advanced: bool = True, request_id: str = None, needs_tools: bool = False):
    """
    Factory function to create the appropriate model instance based on provider.

    Args:
        model_provider: The model provider ('openai', 'claude', 'grok', 'deepseek', 'gemini')
        api_key: The API key for the model provider
        use_advanced: Whether to use advanced/larger model (for complex tasks)
        request_id: Request ID for logging
        needs_tools: Whether this agent needs tool/function calling support

    Returns:
        Model instance for agno Agent
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    model_provider = model_provider.lower()

    tier = ADVANCED_MODEL_IDS if use_advanced else BASIC_MODEL_IDS
    model_id = tier.get(model_provider)

    if model_id is None:
        logger.error(f"{log_prefix}Unsupported model provider: {model_provider}")
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # deepseek-reasoner (R1) does NOT support function/tool calling, so any
    # agent that needs tools has to drop to deepseek-chat.
    if model_provider == 'deepseek' and needs_tools and model_id != DEEPSEEK_TOOL_MODEL_ID:
        logger.info(
            f"{log_prefix}Creating DeepSeek model: {DEEPSEEK_TOOL_MODEL_ID} "
            f"(downgraded from {model_id} - no tool support)"
        )
        return DeepSeek(id=DEEPSEEK_TOOL_MODEL_ID, api_key=api_key)

    logger.info(f"{log_prefix}Creating {model_provider} model: {model_id} (advanced={use_advanced})")

    if model_provider == 'openai':
        return OpenAIChat(id=model_id, api_key=api_key)
    if model_provider == 'claude':
        return Claude(id=model_id, api_key=api_key)
    if model_provider == 'grok':
        return xAI(id=model_id, api_key=api_key)
    if model_provider == 'deepseek':
        return DeepSeek(id=model_id, api_key=api_key)
    if model_provider == 'gemini':
        return Gemini(id=model_id, api_key=api_key)

    # Unreachable: the tier lookup above already rejects unknown providers.
    raise ValueError(f"Unsupported model provider: {model_provider}")

# ============================================
# CUSTOM TOOLS
# ============================================

# Cap on returned page text. Enough for the tables and figures this is used
# for, small enough not to blow out the context window on a long article.
MAX_PAGE_CHARS = 20000


def _is_publicly_routable(hostname: str) -> bool:
    """
    Reject anything that resolves to a private, loopback, link-local or
    otherwise reserved address.

    The model chooses these URLs, and the fetch happens from our EC2 host, so an
    unguarded fetcher is an SSRF primitive: http://169.254.169.254/ returns IAM
    credentials, and the VPC's private ranges are reachable too. Resolution is
    done here rather than trusting the hostname string, since a public-looking
    name can resolve to a private address.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def create_web_read_tool(request_id: str = None):
    """
    A guarded page reader.

    web_search alone returns titles, URLs and snippets — never the page body —
    so an agent asked for figures could find the right source but not read it,
    and would fall back on recalled numbers while citing a URL it never opened.
    This closes that gap.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    def read_web_page(url: str) -> str:
        """
        Read the text content of a web page.

        Use this after web_search to read the actual figures, tables or facts on
        a page. Search results alone give you only a title and a snippet — never
        quote numbers from a page you have not read with this tool.

        Args:
            url: Full http(s) URL of the page to read.

        Returns:
            The page's text content, or an error string beginning with "ERROR:".
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return "ERROR: only http and https URLs can be read."
            if not parsed.hostname:
                return "ERROR: malformed URL."
            if not _is_publicly_routable(parsed.hostname):
                logger.warning(f"{log_prefix}Blocked read_web_page for non-public host")
                return "ERROR: that host is not publicly routable and cannot be read."

            from agno.knowledge.reader.website_reader import WebsiteReader

            logger.info(f"{log_prefix}Reading page: {parsed.scheme}://{parsed.hostname}{parsed.path[:60]}")
            docs = WebsiteReader(max_links=1, max_depth=1).read(url=url)
            text = "\n\n".join((d.content or "") for d in docs).strip()

            if not text:
                return "ERROR: no readable text found at that URL (it may be JavaScript-rendered or blocking automated access). Say so rather than substituting remembered figures."
            if len(text) > MAX_PAGE_CHARS:
                text = text[:MAX_PAGE_CHARS] + "\n\n[truncated]"
            return text
        except Exception as e:
            logger.warning(f"{log_prefix}read_web_page failed: {type(e).__name__}")
            return (f"ERROR: could not read that page ({type(e).__name__}). "
                    "Try another source, or say the figure could not be verified.")

    return read_web_page



def create_image_tool(
    api_key: str,
    pending_images: dict = None,
    request_id: str = None,
    user_id: str = None,
    credit_manager=None,
    is_byok: bool = False,
):
    """
    Create an async image generation tool with the provided API key.
    Returns a placeholder immediately and generates the image in a background thread.

    Args:
        api_key: OpenAI API key for GPT Image
        pending_images: Shared dict to track background image generation status
        request_id: Request ID for logging
        user_id: User ID for organizing uploaded images under user's S3 directory
        credit_manager: CreditManager instance for per-image credit reservation
        is_byok: If True, skip credit reservation (BYOK users pay no credits)

    Returns:
        Function that triggers async image generation and returns a placeholder
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    def generate_image_tool(prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image using GPT Image and return a placeholder URL immediately.
        The actual image is generated in the background and uploaded to S3.

        Args:
            prompt: Description of the image to generate
            size: Image size - "1024x1024" (square), "1024x1536" (portrait), or "1536x1024" (landscape)

        Returns:
            Placeholder URL string (format: "Image URL: placeholder::{uuid}")
        """
        placeholder_id = str(uuid.uuid4())

        # Credit gate (skipped for BYOK users — they pay no credits)
        if credit_manager is not None and not is_byok and user_id:
            from src.credit_manager import IMAGE_CREDITS_SCALED
            ok, _available = credit_manager.reserve(user_id, IMAGE_CREDITS_SCALED)
            if not ok:
                logger.warning(
                    f"{log_prefix}Image generation skipped — insufficient credits "
                    f"(user={user_id})"
                )
                return "Image generation skipped: insufficient credits"

        if pending_images is not None:
            pending_images[placeholder_id] = {
                "status": "generating",
                "cdn_url": None,
                "error": None,
                "prompt": prompt,
            }

            def _background_generate():
                try:
                    import base64
                    from openai import OpenAI
                    from src.s3_service import upload_image_from_bytes

                    logger.info(f"{log_prefix}Background image generation started for placeholder {placeholder_id}")

                    client = OpenAI(api_key=api_key)
                    response = client.images.generate(
                        model=IMAGE_MODEL_IDS["openai"],
                        prompt=prompt,
                        size=size,
                        quality="auto",
                        n=1,
                    )

                    image_bytes = base64.b64decode(response.data[0].b64_json)
                    logger.info(f"{log_prefix}GPT Image generated, uploading to S3...")

                    cdn_url = upload_image_from_bytes(image_bytes, content_type="image/png", user_id=user_id, request_id=request_id)

                    pending_images[placeholder_id]["cdn_url"] = cdn_url
                    pending_images[placeholder_id]["status"] = "done"
                    logger.info(f"{log_prefix}Background image complete: {cdn_url}")

                except Exception as e:
                    logger.error(f"{log_prefix}Background image generation failed: {str(e)}")
                    pending_images[placeholder_id]["error"] = str(e)
                    pending_images[placeholder_id]["status"] = "failed"

            thread = threading.Thread(target=_background_generate, daemon=True)
            thread.start()

        return f"Image URL: placeholder::{placeholder_id}"

    return generate_image_tool


def create_grok_image_tool(
    api_key: str,
    pending_images: dict = None,
    request_id: str = None,
    user_id: str = None,
    credit_manager=None,
    is_byok: bool = False,
):
    """
    Create an async image generation tool using xAI's Grok (Aurora) model.
    Uses the OpenAI-compatible xAI API endpoint.
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    def generate_image_tool(prompt: str) -> str:
        """
        Generate an image using Grok's Aurora model and return a placeholder URL immediately.
        The actual image is generated in the background and uploaded to S3.

        Args:
            prompt: Description of the image to generate

        Returns:
            Placeholder URL string (format: "Image URL: placeholder::{uuid}")
        """
        placeholder_id = str(uuid.uuid4())

        if credit_manager is not None and not is_byok and user_id:
            from src.credit_manager import IMAGE_CREDITS_SCALED
            ok, _available = credit_manager.reserve(user_id, IMAGE_CREDITS_SCALED)
            if not ok:
                logger.warning(
                    f"{log_prefix}Grok image generation skipped — insufficient credits "
                    f"(user={user_id})"
                )
                return "Image generation skipped: insufficient credits"

        if pending_images is not None:
            pending_images[placeholder_id] = {
                "status": "generating",
                "cdn_url": None,
                "error": None,
                "prompt": prompt,
            }

            def _background_generate():
                try:
                    from openai import OpenAI
                    from src.s3_service import upload_image_from_url

                    logger.info(f"{log_prefix}Background Grok image generation started for placeholder {placeholder_id}")

                    client = OpenAI(
                        base_url="https://api.x.ai/v1",
                        api_key=api_key,
                    )
                    response = client.images.generate(
                        model=IMAGE_MODEL_IDS["grok"],
                        prompt=prompt,
                        n=1,
                    )

                    grok_url = response.data[0].url
                    logger.info(f"{log_prefix}Grok image generated, uploading to S3...")

                    cdn_url = upload_image_from_url(grok_url, user_id=user_id, request_id=request_id)

                    pending_images[placeholder_id]["cdn_url"] = cdn_url
                    pending_images[placeholder_id]["status"] = "done"
                    logger.info(f"{log_prefix}Background Grok image complete: {cdn_url}")

                except Exception as e:
                    logger.error(f"{log_prefix}Background Grok image generation failed: {str(e)}")
                    pending_images[placeholder_id]["error"] = str(e)
                    pending_images[placeholder_id]["status"] = "failed"

            thread = threading.Thread(target=_background_generate, daemon=True)
            thread.start()

        return f"Image URL: placeholder::{placeholder_id}"

    return generate_image_tool


def create_gemini_image_tool(
    api_key: str,
    pending_images: dict = None,
    request_id: str = None,
    user_id: str = None,
    credit_manager=None,
    is_byok: bool = False,
):
    """
    Create an async image generation tool using Google Gemini's native image generation.
    Uses the Google Gen AI SDK with response_modalities=["IMAGE"].
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    def generate_image_tool(prompt: str) -> str:
        """
        Generate an image using Gemini's native image generation and return a placeholder URL immediately.
        The actual image is generated in the background and uploaded to S3.

        Args:
            prompt: Description of the image to generate

        Returns:
            Placeholder URL string (format: "Image URL: placeholder::{uuid}")
        """
        placeholder_id = str(uuid.uuid4())

        if credit_manager is not None and not is_byok and user_id:
            from src.credit_manager import IMAGE_CREDITS_SCALED
            ok, _available = credit_manager.reserve(user_id, IMAGE_CREDITS_SCALED)
            if not ok:
                logger.warning(
                    f"{log_prefix}Gemini image generation skipped — insufficient credits "
                    f"(user={user_id})"
                )
                return "Image generation skipped: insufficient credits"

        if pending_images is not None:
            pending_images[placeholder_id] = {
                "status": "generating",
                "cdn_url": None,
                "error": None,
                "prompt": prompt,
            }

            def _background_generate():
                try:
                    from google import genai
                    from google.genai import types
                    from src.s3_service import upload_image_from_bytes

                    logger.info(f"{log_prefix}Background Gemini image generation started for placeholder {placeholder_id}")

                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=IMAGE_MODEL_IDS["gemini"],
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                        ),
                    )

                    # Validate response structure
                    if not response.candidates:
                        block_reason = getattr(response, 'prompt_feedback', None)
                        raise Exception(f"Gemini image generation blocked by safety filters: {block_reason}")

                    candidate = response.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                        raise Exception(f"Gemini returned no image data (finish_reason: {finish_reason}). The prompt may have been blocked by safety filters.")

                    # Extract image bytes from response parts
                    image_uploaded = False
                    for part in candidate.content.parts:
                        if part.inline_data:
                            image_bytes = part.inline_data.data
                            mime_type = part.inline_data.mime_type or "image/png"
                            logger.info(f"{log_prefix}Gemini image generated ({mime_type}), uploading to S3...")

                            cdn_url = upload_image_from_bytes(
                                image_bytes, content_type=mime_type,
                                user_id=user_id, request_id=request_id,
                            )

                            pending_images[placeholder_id]["cdn_url"] = cdn_url
                            pending_images[placeholder_id]["status"] = "done"
                            logger.info(f"{log_prefix}Background Gemini image complete: {cdn_url}")
                            image_uploaded = True
                            break

                    if not image_uploaded:
                        raise Exception("Gemini response did not contain image data")

                except Exception as e:
                    logger.error(f"{log_prefix}Background Gemini image generation failed: {str(e)}")
                    pending_images[placeholder_id]["error"] = str(e)
                    pending_images[placeholder_id]["status"] = "failed"

            thread = threading.Thread(target=_background_generate, daemon=True)
            thread.start()

        return f"Image URL: placeholder::{placeholder_id}"

    return generate_image_tool


# ============================================
# AGENT CONFIGURATIONS
# ============================================

AGENT_CONFIGS = {
    "grammar": {
        "name": "Grammar Corrector",
        "role": "Fix grammar, spelling, and punctuation errors",
        "instructions": [
            "Your task is to correct grammar, spelling, and punctuation errors in the provided text.",
            "Maintain the original meaning, tone, and style of the text.",
            "Do NOT change the structure or add new content.",
            "Only fix errors - don't rewrite or rephrase unnecessarily.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the corrected text. Do NOT include any headers, labels, or prefixes like 'Corrected Text:', 'Here is the corrected version:', etc. Just output the final corrected text directly.",
            markdown_conversion_instruction,
        ],
    },
    "simplify": {
        "name": "Text Simplifier",
        "role": "Simplify complex text for easier understanding",
        "instructions": [
            "Your task is to simplify complex sentences and vocabulary without losing the core meaning.",
            "Use simpler words and shorter sentences where appropriate.",
            "Make the text more accessible to a general audience.",
            "Maintain the key information and facts.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the simplified text. Do NOT include any headers, labels, or prefixes like 'Simplified Text:', 'Here is the simplified version:', etc. Just output the final simplified text directly.",
            markdown_conversion_instruction,
        ],
    },
    "completion": {
        "name": "Text Completer",
        "role": "Complete partial sentences and paragraphs",
        "instructions": [
            "Your task is to complete partial or incomplete sentences and paragraphs.",
            "Maintain the style, tone, and direction of the original text.",
            "Make logical and coherent completions that flow naturally.",
            "Don't change what's already written - only add to it.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the completed text (original + completion). Do NOT include any headers, labels, or prefixes like 'Completed Text:', 'Here is the completion:', etc. Just output the final text directly.",
            markdown_conversion_instruction,
        ],
    },
    "emojify": {
        "name": "Emoji Enhancer",
        "role": "Add relevant emojis to text",
        "instructions": [
            "Your task is to add appropriate and relevant emojis to the text.",
            "Place emojis at natural points that enhance meaning and engagement.",
            "Don't overdo it - use emojis tastefully (1-3 per paragraph is usually enough).",
            "Match emoji sentiment to the text context.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the text with emojis added. Do NOT include any headers, labels, or prefixes like 'Emojified Text:', 'Here is the text with emojis:', etc. Just output the final text directly.",
            markdown_conversion_instruction,
        ],
    },
    "translation": {
        "name": "Language Translator",
        "role": "Translate text between languages",
        "instructions": [
            "Your task is to translate text to the specified target language.",
            "First, detect the source language automatically.",
            "Maintain the meaning, tone, and style of the original text.",
            "Preserve any existing formatting in the input.",
            "For language detection, look for explicit instructions like 'translate to Spanish' or 'convert to French'.",
            "CRITICAL: Output ONLY the translated text. Do NOT include any headers, labels, or prefixes like 'Translation:', 'Translated Text:', 'Here is the translation:', etc. Just output the final translated text directly.",
            markdown_conversion_instruction,
        ],
    },
    "table": {
        "name": "Table Creator",
        "role": "Organize information into Markdown tables",
        "instructions": [
            "Your task is to organize information into clean, well-structured Markdown tables.",
            "Use proper Markdown table syntax with | pipes and --- separators.",
            "Example: | Header 1 | Header 2 |\\n|---|---|\\n| Data 1 | Data 2 |",
            "Add appropriate headers based on the content.",
            "Ensure data is logically organized in rows and columns.",
            "Align columns properly for readability.",
            "CRITICAL: Output ONLY the table. Do NOT include any headers, labels, or prefixes like 'Table:', 'Here is the table:', etc. Just output the Markdown table directly.",
            markdown_conversion_instruction,
        ],
    },
    "list": {
        "name": "List Organizer",
        "role": "Convert text into organized lists",
        "instructions": [
            "Your task is to organize information into Markdown lists.",
            "Use - or * for unordered lists and 1. 2. 3. for ordered lists.",
            "Identify list items logically from the content.",
            "Use indentation (2 spaces) for nested lists.",
            "Keep list items concise and clear.",
            "CRITICAL: Output ONLY the list. Do NOT include any headers, labels, or prefixes like 'List:', 'Here is the list:', etc. Just output the Markdown list directly.",
            markdown_conversion_instruction,
        ],
    },
    "summarize": {
        "name": "Content Summarizer",
        "role": "Create concise summaries of text",
        "instructions": [
            "Your task is to create a concise summary of the provided text.",
            "Extract key points and main ideas.",
            "Keep the summary 20-30% of the original length.",
            "Maintain factual accuracy and important details.",
            "Structure the summary with proper Markdown formatting.",
            "CRITICAL: Output ONLY the summary. Do NOT include any headers, labels, or prefixes like 'Summary:', 'Here is the summary:', 'Key Points:', etc. Just output the final summary directly.",
            markdown_conversion_instruction,
        ],
    },
    "tone": {
        "name": "Tone Adjuster",
        "role": "Change the tone/style of text",
        "instructions": [
            "Your task is to rewrite text in a different tone or style.",
            "Common tones: professional, casual, friendly, formal, persuasive, technical, creative.",
            "Maintain the core information and facts.",
            "Adjust vocabulary, sentence structure, and phrasing to match the requested tone.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the rewritten text in the new tone. Do NOT include any headers, labels, or prefixes like 'Professional Tone:', 'Casual Version:', 'Here is the text in a different tone:', etc. Just output the final text directly.",
            markdown_conversion_instruction,
        ],
    },
    "make_longer": {
        "name": "Text Expander",
        "role": "Expand and elaborate sentences to make them longer",
        "instructions": [
            "Your task is to make the provided text longer by expanding and elaborating on it.",
            "Add relevant details, examples, explanations, or context to enrich the content.",
            "Use descriptive language and additional clauses to extend sentences naturally.",
            "Maintain the original meaning and tone while expanding.",
            "Do NOT add irrelevant or filler content - all additions should be meaningful.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the expanded text. Do NOT include any headers, labels, or prefixes like 'Expanded Text:', 'Longer Version:', 'Here is the expanded text:', etc. Just output the final expanded text directly.",
            markdown_conversion_instruction,
        ],
    },
    "make_shorter": {
        "name": "Text Condenser",
        "role": "Condense and shorten sentences while preserving meaning",
        "instructions": [
            "Your task is to make the provided text shorter and more concise.",
            "Remove unnecessary words, redundancies, and verbose expressions.",
            "Keep only the essential information and core meaning.",
            "Use simpler, more direct sentence structures.",
            "Maintain clarity and readability while reducing length.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the shortened text. Do NOT include any headers, labels, or prefixes like 'Shortened Text:', 'Condensed Version:', 'Here is the shorter text:', etc. Just output the final condensed text directly.",
            markdown_conversion_instruction,
        ],
    },
    "enhance": {
        "name": "AI Enhancer",
        "role": "Enhance and improve text quality using AI",
        "instructions": [
            "Your task is to enhance and improve the provided text using AI capabilities.",
            "Improve clarity, flow, and readability of the content.",
            "Enhance vocabulary with more precise and impactful words.",
            "Improve sentence structure for better engagement.",
            "Add polish and professionalism to the writing.",
            "Fix any grammar or spelling issues while enhancing.",
            "Maintain the original intent and meaning of the text.",
            "Preserve any existing formatting in the input.",
            "CRITICAL: Output ONLY the enhanced text. Do NOT include any headers, labels, or prefixes like 'Enhanced Text:', 'Improved Version:', 'Here is the enhanced text:', etc. Just output the final enhanced text directly.",
            markdown_conversion_instruction,
        ],
    },
    "smart_global": {
        "name": "Smart Content Assistant",
        "role": "Handle all content creation, editing, and enhancement tasks",
        "instructions": [
            "You are a comprehensive content assistant integrated into a text editor.",
            "You can handle ALL text operations and content creation tasks.",
            "",
            "CAPABILITIES:",
            "- Write articles, blog posts, essays, reports",
            "- Generate images using the available image generation tool (GPT Image, Grok Aurora, or Gemini depending on the provider)",
            "- Create tables, lists, and structured content",
            "- Search the web for current information",
            "- Format content in clean Markdown",
            "",
            "DECISION MAKING FOR IMAGES:",
            "1. If user asks ONLY for an image ('draw X', 'create picture of Y'):",
            "   - Use generate_image_tool to create the image",
            "   - The tool returns a placeholder URL in the format: 'Image URL: placeholder::{uuid}'",
            "   - Return ONLY: ![description](placeholder::{uuid})",
            "",
            "2. If user asks for content WITH images ('article about X with images'):",
            "   - IMPORTANT: Call generate_image_tool for ALL needed images FIRST, before writing any content.",
            "   - Collect all the placeholder URLs returned by the tool (format: 'Image URL: placeholder::{uuid}').",
            "   - THEN write the complete content in Markdown, embedding the placeholder URLs using ![alt text](placeholder::{uuid}) syntax.",
            "   - Do NOT invent or fabricate image URLs. Only use URLs returned by the tool.",
            "   - Do NOT write the content twice. Write it once using the placeholder URLs from the tool.",
            "",
            "3. If user asks for content WITHOUT images ('write article about X'):",
            "   - Write the content in Markdown",
            "   - Do NOT generate images unless explicitly requested",
            "",
            "TABLE CREATION:",
            "- When user asks to 'create a table', 'organize as table', 'make a comparison table'",
            "- Use proper Markdown table syntax with | pipes and --- separators",
            "- Example: | Header 1 | Header 2 |\\n|---|---|\\n| Data 1 | Data 2 |",
            "- Add appropriate headers and organize data logically",
            "",
            "LIST CREATION:",
            "- When user asks to 'make a list', 'bullet points', 'numbered list'",
            "- Use - or * for unordered lists, 1. 2. 3. for ordered lists",
            "- Use indentation (2 spaces) for nested lists",
            "",
            "WEB SEARCH:",
            "- Use DuckDuckGo tool when you need current information or facts",
            "- Search for statistics, recent events, or verification of information",
            "- You MUST search before answering any question about exchange rates, prices, market data, statistics, company or product facts, or anything that changes over time. Your training data is stale; searching is the only way to be current.",
            "- Search results give you only titles, URLs and snippets. When you need actual figures, follow up with read_web_page on the most promising result and take the numbers from the page itself.",
            "- Never quote a figure from a page you have not read. If read_web_page fails or returns nothing usable, try one other source; if that also fails, say the figure could not be verified and label your answer as approximate.",
            "- Resolve relative dates ('this year', 'the last 5 years', 'recently') against the current date given above, NOT against your training data.",
            "- A relative range INCLUDES the current year. 'The last 5 years' asked during 2026 means 2022-2026, not 2021-2025. Do not silently narrow it to the last complete years — the current period is usually the one the user cares about most.",
            "- Where the current period is incomplete, still include it and label it (e.g. '2026 (year to date)'), noting the cut-off. Omitting it entirely loses the most relevant data point.",
            "- If a search returns nothing usable, say so plainly and give your best answer clearly labelled as approximate. Never present recalled figures as though they were retrieved.",
            "",
            "CITING SOURCES:",
            "- Only cite a source you actually retrieved in this conversation. Never reconstruct a citation, URL, dataset name or series identifier from memory — a plausible-looking but invented reference is worse than no reference.",
            "- When you do cite, link the specific page the figures came from.",
            "",
            "MARKDOWN FORMATTING:",
            markdown_conversion_instruction,
            "",
            "TOOL USAGE:",
            "- Always extract the placeholder URL from generate_image_tool responses",
            "- Tool returns format: 'Image URL: placeholder::{uuid}'",
            "- Extract just the 'placeholder::{uuid}' part for ![alt](placeholder::{uuid}) syntax",
            "- NEVER fabricate image URLs like 'https://example.com/...' — only use placeholder URLs from the tool",
            "",
            "QUALITY STANDARDS:",
            "- Use proper Markdown formatting",
            "- Ensure proper heading hierarchy (# > ## > ###)",
            "- Include alt text for all images",
            "- Keep content clean and well-structured",
            "- Make content readable and accessible",
        ],
        "use_tools": True,
        "use_advanced": True,
    },
    # Transcribes what is written in an attached image. Deliberately narrow:
    # it reports what it can see and nothing else, because the failure mode
    # that matters here is confident invention of text that was not there.
    "text_extraction": {
        "name": "Image Text Extractor",
        "role": "Transcribe the text contained in an image",
        "instructions": [
            "Your only job is to transcribe text that appears in the attached image.",
            "",
            "RULES:",
            "- Transcribe verbatim. Do not correct spelling, grammar, punctuation or capitalisation, even if something looks like a mistake.",
            "- Do not translate, summarise, rephrase or explain. Output the text itself.",
            "- Preserve the reading order and the visual grouping: keep headings as headings, lists as lists, and paragraph breaks where they appear.",
            "- If the image contains a table, reproduce it as a Markdown table with the same rows and columns.",
            "- If a word is genuinely illegible, write [illegible] in its place rather than guessing at it.",
            "- If the image contains no readable text at all, say exactly: No text found in this image.",
            "- Never add commentary, headers or preamble such as 'Here is the text:'. Output only the transcription.",
            "",
            "CRITICAL: Do not invent text. Anything you output must be visibly present in the image. Inventing plausible-looking copy is the worst possible failure here, because the user cannot tell it apart from a correct transcription.",
            markdown_conversion_instruction,
        ],
        # No tools: this is a single-shot read of the image. Web search and
        # image generation would only add latency and opportunities to drift.
        "use_tools": False,
        "use_advanced": True,
    },
    # Email templates are the one agent that must NOT emit Markdown. Its output
    # goes straight into an HtmlCanvasNode (rendered in a sandboxed iframe via
    # srcdoc), so the raw HTML survives byte-for-byte instead of being
    # normalised by Lexical's node system.
    "email_template": {
        "name": "Email Template Designer",
        "role": "Turn a design reference into production-ready HTML email markup",
        "instructions": [
            "You produce HTML email templates that render correctly across Gmail, Outlook, Apple Mail and mobile clients.",
            "",
            "WHEN A REFERENCE IMAGE IS ATTACHED:",
            "- Read the design from the image: layout structure, section order, colour palette, typography scale, spacing, imagery placement and calls to action.",
            "- Reproduce that design as faithfully as the email HTML subset allows.",
            "- Sample actual hex colours from the image rather than approximating with named colours.",
            "- If the reference contains imagery you cannot reproduce in markup (logos, photographs, illustrations), emit an <img> with the exact literal src=\"placeholder\" — the editor swaps that token for a real placeholder graphic, so write it verbatim and never invent a URL such as https://example.com/logo.png.",
            "- Every placeholder <img> MUST carry explicit width and height attributes matching the size in the reference, plus a descriptive alt attribute, so the layout holds before the user swaps the image in.",
            "- Transcribe any text visible in the reference verbatim; do not invent replacement copy unless the user asks for it.",
            "",
            "REQUIRED STRUCTURE:",
            "- Use table-based layout only. Nest <table> elements for structure; never use flexbox, grid, float or position.",
            "- Every <table> must carry role='presentation' cellpadding='0' cellspacing='0' border='0'.",
            "- Wrap everything in a 100%-width outer table, with an inner container table fixed at 600px max width.",
            "- Use <td> padding for all spacing. Never use margin.",
            "",
            "REQUIRED STYLING:",
            "- All CSS must be inline via the style attribute. No <style> blocks and no external stylesheets — most clients strip them.",
            "- Set explicit width, height and alt on every <img>, plus style='display:block;' to avoid gaps in Gmail.",
            "- Use web-safe font stacks with fallbacks (e.g. Arial, Helvetica, sans-serif).",
            "- Specify colours as full six-digit hex values.",
            "- Build buttons as a bordered <table> cell with a padded <a> inside, not as a styled <button>.",
            "",
            "OUTPUT CONTRACT:",
            "- Return a complete, standalone HTML document beginning with <!DOCTYPE html> and ending with </html>.",
            "- Include <meta name='viewport' content='width=device-width, initial-scale=1'> and <meta charset='utf-8'> in the head.",
            "- CRITICAL: Output ONLY raw HTML. Do NOT wrap it in Markdown code fences, do NOT prefix it with explanation, and do NOT append commentary after </html>.",
            "- Do NOT convert the output to Markdown under any circumstances. This agent's output is rendered as HTML, not parsed as Markdown.",
        ],
        # Tools stay off: web search and image generation would interrupt a
        # single-shot HTML emission, and disabling them also puts Claude on the
        # same streaming path as every other provider (see create_agent).
        "use_tools": False,
        "use_advanced": True,
    },
}

# ============================================
# AGENT FACTORY FUNCTION
# ============================================

def create_agent(
    agent_type: str,
    model_provider: str,
    api_key: str,
    request_id: str = None,
    pending_images: dict = None,
    user_id: str = None,
    credit_manager=None,
    is_byok: bool = False,
):
    """
    Factory function to create an agent with the specified model.

    Args:
        agent_type: Type of agent to create (from AGENT_CONFIGS keys)
        model_provider: The model provider ('openai', 'claude', 'grok')
        api_key: The API key for the model provider
        request_id: Request ID for logging
        pending_images: Shared dict for tracking async image generation
        user_id: User ID for organizing uploaded images under user's S3 directory
        credit_manager: CreditManager instance for per-image credit reservation
        is_byok: If True, skip credit reservation (BYOK users pay no credits)

    Returns:
        Configured Agent instance
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if agent_type not in AGENT_CONFIGS:
        logger.error(f"{log_prefix}Unknown agent type: {agent_type}")
        raise ValueError(f"Unknown agent type: {agent_type}")

    config = AGENT_CONFIGS[agent_type]
    use_advanced = config.get("use_advanced", True)
    use_tools = config.get("use_tools", True)

    logger.info(f"{log_prefix}Creating agent: {config['name']} (type={agent_type}, advanced={use_advanced}, tools={use_tools})")

    model = get_model(model_provider, api_key, use_advanced, request_id, needs_tools=use_tools)

    agent_kwargs = {
        "name": config["name"],
        "role": config["role"],
        "model": model,
        "instructions": config["instructions"],
        "markdown": True,
        # Server-side service: disable agno's anonymous usage telemetry so we
        # don't make an outbound call on every agent run.
        "telemetry": False,
        # Tell the model what day it is. Without this (agno defaults it off) the
        # model anchors relative dates to its training cutoff — "the last 5
        # years" came back as 2020-2024 well into 2026, and "this year" was
        # simply wrong. UTC keeps it unambiguous across our user base.
        "add_datetime_to_context": True,
        "timezone_identifier": "Etc/UTC",
    }

    # Add tools for smart_global agent
    # Claude: tools currently disabled. This was originally a phidata limitation
    #         (no streaming + tools for Claude; the non-streaming fallback was
    #         ~18s and hurt UX). agno DOES support Claude streaming with tools,
    #         so this exclusion can likely be lifted — verify latency with a live
    #         key before enabling image/web-search tools for Claude.
    if use_tools and model_provider != 'claude':
        # backend="auto" lets ddgs rotate across multiple search engines
        # (Google/Bing/Brave/DuckDuckGo/...) instead of only DuckDuckGo. The
        # DuckDuckGo-only backend intermittently 403s from datacenter IPs
        # (e.g. the news endpoint on our EC2 host), starving the model of
        # results; "auto" falls back to other engines and is far more reliable.
        # Search finds the page; read_web_page reads it. Without the reader the
        # model can only see snippets, so it cites a source it never opened and
        # fills the numbers in from memory.
        tools = [DuckDuckGoWithRetry(backend="auto"), create_web_read_tool(request_id)]
        if model_provider == 'openai':
            tools.insert(0, create_image_tool(
                api_key, pending_images, request_id, user_id,
                credit_manager=credit_manager, is_byok=is_byok,
            ))
            logger.info(f"{log_prefix}Added tools: GPT Image generation (async), DuckDuckGo search")
        elif model_provider == 'grok':
            tools.insert(0, create_grok_image_tool(
                api_key, pending_images, request_id, user_id,
                credit_manager=credit_manager, is_byok=is_byok,
            ))
            logger.info(f"{log_prefix}Added tools: Grok Aurora image generation (async), DuckDuckGo search")
        elif model_provider == 'gemini':
            tools.insert(0, create_gemini_image_tool(
                api_key, pending_images, request_id, user_id,
                credit_manager=credit_manager, is_byok=is_byok,
            ))
            logger.info(f"{log_prefix}Added tools: Gemini image generation (async), DuckDuckGo search")
        else:
            logger.info(f"{log_prefix}Added tools: DuckDuckGo search")
        agent_kwargs["tools"] = tools

        # DeepSeek tool calling is unstable and can loop indefinitely.
        # Limit to 1 tool call per turn to prevent runaway loops.
        if model_provider == 'deepseek':
            agent_kwargs["tool_call_limit"] = 1
            logger.info(f"{log_prefix}Set tool_call_limit=1 for DeepSeek (prevents tool call loops)")
    elif use_tools and model_provider == 'claude':
        logger.info(f"{log_prefix}Tools skipped for Claude (streaming + tools not supported)")

    return Agent(**agent_kwargs)

def determine_agent_type(query_lower: str) -> tuple[str, str]:
    """
    Determine which agent to use based on query keywords.
    Single source of truth used by both streaming and non-streaming query execution (Bug #3 fix).
    """
    if "fix grammar" in query_lower or "correct grammar" in query_lower or "fix spelling" in query_lower:
        return "grammar", "detected grammar/spelling fix keywords"
    elif "simplify" in query_lower and "image" not in query_lower:
        return "simplify", "detected 'simplify' keyword"
    elif "complete" in query_lower and "image" not in query_lower:
        return "completion", "detected 'complete' keyword"
    elif "emojify" in query_lower or "add emoji" in query_lower:
        return "emojify", "detected emoji-related keywords"
    elif "translate" in query_lower and "image" not in query_lower:
        return "translation", "detected 'translate' keyword"
    elif "create table" in query_lower or "make table" in query_lower or "organize as table" in query_lower or "table format" in query_lower:
        return "table", "detected table creation keywords"
    elif ("create list" in query_lower or "make list" in query_lower or
          "bullet points" in query_lower or "numbered list" in query_lower):
        return "list", "detected list creation keywords"
    elif "summarize" in query_lower or "summary" in query_lower or "tldr" in query_lower:
        return "summarize", "detected summarization keywords"
    elif "change tone" in query_lower or "change_tone" in query_lower:
        return "tone", "detected tone change keywords"
    elif "make longer" in query_lower or "expand" in query_lower or "elaborate" in query_lower:
        return "make_longer", "detected 'make longer' keywords"
    elif "make shorter" in query_lower or "condense" in query_lower or "shorten" in query_lower:
        return "make_shorter", "detected 'make shorter' keywords"
    elif "enhance" in query_lower or "improve" in query_lower or "polish" in query_lower:
        return "enhance", "detected enhancement keywords"
    else:
        return "smart_global", "no specific keywords matched, using smart global agent"


# ============================================
# GLOBAL QUERY PROCESSING FUNCTION
# ============================================

def process_user_query_stream(
    query: str,
    model_provider: str = "openai",
    api_key: str = None,
    request_id: str = None,
    user_id: str = None,
    credit_manager=None,
    is_byok: bool = False,
    image_urls=None,
    force_agent_type: str = None,
):
    """
    Stream query processor for text editor integration.
    Routes queries to appropriate specialized agents and streams the response.

    Args:
        query: User's request/command
        model_provider: The model provider ('openai', 'claude', 'grok', 'deepseek', 'gemini')
        api_key: The API key for the model provider
        request_id: Request ID for logging
        user_id: User ID for organizing uploaded images under user's S3 directory
        credit_manager: CreditManager instance for per-image credit reservation
        is_byok: If True, skip credit reservation (BYOK users pay no credits)
        image_urls: Reference images the user attached, as CDN URLs. Passed to
            the model as vision input — distinct from the image *generation*
            tool, which produces images rather than reading them.
        force_agent_type: Skip keyword routing and use this agent type directly.
            Used by callers that already know the intent (e.g. the email
            template flow), where sniffing the query text would be unreliable.

    Yields:
        Markdown content chunks as they are generated
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not api_key:
        logger.error(f"{log_prefix}API key is required but not provided")
        raise ValueError("API key is required")

    query_lower = query.lower()

    if force_agent_type:
        if force_agent_type not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {force_agent_type}")
        agent_type, reason = force_agent_type, "explicitly requested by caller"
    else:
        agent_type, reason = determine_agent_type(query_lower)
    logger.info(f"{log_prefix}Query routing: {reason} -> {agent_type} agent")

    # Reference images are vision input. Silently dropping them on a text-only
    # provider would produce a confidently wrong answer with no clue why, so
    # callers are expected to gate on supports_vision() before getting here.
    images = None
    if image_urls:
        if not supports_vision(model_provider):
            raise ValueError(
                f"Provider '{model_provider}' does not support image input"
            )
        images = build_agno_images(image_urls, request_id)

    # Shared dict for tracking async image generation
    pending_images = {}

    try:
        agent = create_agent(
            agent_type, model_provider, api_key, request_id,
            pending_images=pending_images, user_id=user_id,
            credit_manager=credit_manager, is_byok=is_byok,
        )
        start_time = time.time()

        config = AGENT_CONFIGS[agent_type]
        use_tools = config.get("use_tools", False)

        # Non-streaming fallback for DeepSeek with tools:
        # DeepSeek streaming + tools is unstable and can loop or produce errors.
        # Run non-streaming (tool use loop works correctly), then yield in chunks.
        # Note: Claude tools are disabled in create_agent() so it always streams natively.
        if use_tools and model_provider == 'deepseek':
            logger.info(f"{log_prefix}Running agent: {agent_type} (non-streaming fallback for {model_provider} + tools)...")
            response = agent.run(query, stream=False, images=images)
            total_chars = 0
            if hasattr(response, 'content') and response.content:
                content = response.content
                total_chars = len(content)
                # Yield in chunks to maintain SSE streaming behavior
                chunk_size = 80
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]

            duration = time.time() - start_time
            logger.info(f"{log_prefix}Agent {agent_type} completed in {duration:.2f}s (total chars: {total_chars})")
        else:
            logger.info(f"{log_prefix}Running agent: {agent_type} with streaming...")
            # Stream the response.
            # agno yields typed RunOutputEvent objects; only run_content events
            # carry incremental text deltas. Filtering on run_content avoids
            # emitting tool-call chatter or duplicating the final completed
            # content (which arrives as a separate run_content_completed event).
            response_stream = agent.run(query, stream=True, images=images)
            total_chars = 0

            for event in response_stream:
                if getattr(event, "event", None) == RunEvent.run_content and event.content:
                    total_chars += len(event.content)
                    yield event.content

            duration = time.time() - start_time
            logger.info(f"{log_prefix}Agent {agent_type} streaming completed in {duration:.2f}s (total chars: {total_chars})")

        # Yield sentinel with pending images so the SSE endpoint can poll for them
        if pending_images:
            logger.info(f"{log_prefix}Pending background images: {len(pending_images)}")
            yield {"__pending_images__": pending_images}

    except Exception as e:
        logger.error(f"{log_prefix}Agent {agent_type} failed with error: {str(e)}")
        raise


def process_user_query(
    query: str,
    model_provider: str = "openai",
    api_key: str = None,
    request_id: str = None,
    force_agent_type: str = None,
) -> str:
    """
    Global query processor for HTML editor integration.
    Routes queries to appropriate specialized agents or uses the smart global agent.

    Supported operations:
    - fix_grammar: Grammar and spelling correction
    - simplify: Simplify complex text
    - complete: Complete partial text
    - emojify: Add emojis to text
    - translate: Translate between languages
    - create_table: Organize content into tables
    - create_list: Organize content into lists
    - summarize: Create summary of text
    - change_tone: Change writing tone/style
    - make_longer: Expand and elaborate sentences to make them longer
    - make_shorter: Condense sentences while preserving meaning
    - enhance_with_ai: Enhance and improve text quality using AI
    - general: Articles, images, complex tasks (uses smart agent)

    Args:
        query: User's request/command
        model_provider: The model provider ('openai', 'claude', 'grok')
        api_key: The API key for the model provider
        request_id: Request ID for logging
        force_agent_type: Direct agent override

    Returns:
        HTML formatted response
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not api_key:
        logger.error(f"{log_prefix}API key is required but not provided")
        raise ValueError("API key is required")

    query_lower = query.lower()

    def run_agent(agent_type: str, reason: str):
        """Helper function to run agent with logging"""
        logger.info(f"{log_prefix}Query routing: {reason} -> {agent_type} agent")
        try:
            agent = create_agent(agent_type, model_provider, api_key, request_id)
            logger.info(f"{log_prefix}Running agent: {agent_type}...")
            start_time = time.time()
            response = agent.run(query)
            duration = time.time() - start_time
            response_length = len(response.content) if response.content else 0
            logger.info(f"{log_prefix}Agent {agent_type} completed in {duration:.2f}s (response length: {response_length} chars)")
            return response.content
        except Exception as e:
            logger.error(f"{log_prefix}Agent {agent_type} failed with error: {str(e)}")
            raise

    # Route to specialized agents (fixes Bug #3: single source of truth via determine_agent_type)
    if force_agent_type:
        if force_agent_type not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {force_agent_type}")
        agent_type, reason = force_agent_type, "explicitly requested by caller"
    else:
        agent_type, reason = determine_agent_type(query_lower)

    return run_agent(agent_type, reason)


# ============================================
# INLINE AUTOCOMPLETE (Direct API, no Phidata)
# ============================================

AUTOCOMPLETE_SYSTEM_PROMPT = (
    "You are an inline text autocomplete engine integrated into a text editor. "
    "Given the text the user has typed so far, predict the most likely next words or sentence they will type. "
    "Your output will be CONCATENATED directly to the end of the user's text with no extra characters in between.\n\n"
    "Rules:\n"
    "- Output ONLY the completion text that comes AFTER what the user already typed.\n"
    "- Do NOT repeat or echo ANY part of the text the user already typed.\n"
    "- SPACING IS CRITICAL: your output is appended character-by-character to the existing text.\n"
    "  * If the user's text ends mid-word (e.g. 'mor'), start with the remaining letters to finish that word (e.g. 'ning! How are you?').\n"
    "  * If the user's text ends with a complete word (e.g. 'and'), you MUST start your output with a space (e.g. ' I hope you are well').\n"
    "  * If the user's text ends with a space or punctuation, do NOT add an extra leading space.\n"
    "- Keep completions short: 1-2 sentences maximum.\n"
    "- Match the style, tone, and language of the existing text.\n"
    "- If you cannot predict a meaningful completion, respond with an empty string.\n"
    "- Do NOT add any labels, prefixes, quotation marks, or markdown formatting.\n"
    "- Do NOT explain or comment on the text. Just output the predicted continuation."
)

# Model IDs for fast/basic variants (used for autocomplete)
# Autocomplete model IDs live in models.py; aliased here for the existing
# call sites. Do not reintroduce a literal map — that is how this path ended up
# pinned to claude-opus-5 and the withdrawn gemini-2.0-flash-lite.
FAST_MODEL_IDS = AUTOCOMPLETE_MODEL_IDS

# OpenAI-compatible API base URLs for providers that support it
OPENAI_COMPATIBLE_BASES = {
    'grok': 'https://api.x.ai/v1',
    'deepseek': 'https://api.deepseek.com',
}


def _strip_echo(text_before: str, suggestion: str) -> str:
    """
    If the AI echoed back the trailing partial word of text_before, strip it.
    e.g. text_before="...everyo", suggestion="everyone! Hope..." → "ne! Hope..."
    Only checks the last word fragment, not arbitrary substrings.
    """
    if not suggestion or not text_before:
        return suggestion

    m = re.search(r'(\w+)$', text_before)
    if not m:
        return suggestion

    partial = m.group(1)

    # Skip leading whitespace before checking for echo
    j = 0
    while j < len(suggestion) and suggestion[j].isspace():
        j += 1

    stripped = suggestion[j:]
    if stripped.lower().startswith(partial.lower()):
        return suggestion[:j] + stripped[len(partial):]

    return suggestion



def _postprocess_autocomplete(text_before: str, suggestion: str) -> str:
    """
    Minimal, deterministic cleanup applied to every autocomplete suggestion.
    This mirrors what large providers typically do:
    - rely on the model + prompt for most behaviour
    - keep post-processing small and focused.

    Steps:
      1) strip trailing whitespace and wrapping quotes (keep leading space)
      2) remove a simple trailing-word echo
      3) insert a space for the common \"word\" + \"CapitalizedWord\" join
      4) remove duplicated terminal punctuation like \"??\" / \"!!\" at the join.
    """
    if not suggestion:
        return ""

    # 1) Trailing whitespace and wrapping quotes
    suggestion = suggestion.rstrip()
    if len(suggestion) >= 2:
        if (suggestion[0] == '"' and suggestion[-1] == '"') or (
            suggestion[0] == "'" and suggestion[-1] == "'"
        ):
            suggestion = suggestion[1:-1]

    # 2) Simple echo strip on trailing word fragment
    suggestion = _strip_echo(text_before, suggestion)

    if not text_before or not suggestion:
        return suggestion

    tb = text_before.rstrip()
    if not tb:
        return suggestion

    last = tb[-1]

    # Find first non-space char in suggestion
    i = 0
    while i < len(suggestion) and suggestion[i].isspace():
        i += 1

    first = suggestion[i] if i < len(suggestion) else ""

    # 3) Fix missing space before a capitalised word after a word character.
    #    e.g. "...morningHow are" → "...morning How are"
    #    Only insert when suggestion has NO leading whitespace (i == 0),
    #    otherwise a space already exists and we'd create a double-space.
    if last.isalnum() and first.isupper() and i == 0:
        suggestion = " " + suggestion

    # 4) Deduplicate terminal punctuation when suggestion starts with the same
    #    mark that text_before already ends with (e.g. "...?" + "? I've...").
    if last in ("?", "!", ".") and first == last:
        suggestion = suggestion[:i] + suggestion[i + 1:]

    return suggestion


def get_autocomplete_suggestion(
    text_before: str,
    text_after: str,
    model_provider: str,
    api_key: str,
    request_id: str = None,
) -> str:
    """
    Get an inline autocomplete suggestion using direct API calls (no Phidata).
    Uses fast/basic model variants for speed.

    Args:
        text_before: Text before the cursor (last ~500 chars)
        text_after: Text after the cursor (next ~200 chars)
        model_provider: 'openai', 'claude', 'grok', 'deepseek', 'gemini'
        api_key: API key for the provider
        request_id: Request ID for logging

    Returns:
        Suggested completion text, or empty string if none
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    model_provider = model_provider.lower()

    model_id = FAST_MODEL_IDS.get(model_provider)
    if not model_id:
        logger.error(f"{log_prefix}Unsupported model provider for autocomplete: {model_provider}")
        return ""

    # Build the user prompt with context
    user_prompt = text_before
    if text_after and text_after.strip():
        user_prompt += f"\n\n[Text after cursor]: {text_after}"

    logger.info(f"{log_prefix}Autocomplete request: provider={model_provider}, model={model_id}, context_len={len(text_before)}")
    start_time = time.time()

    try:
        suggestion = ""

        if model_provider == 'openai' or model_provider in OPENAI_COMPATIBLE_BASES:
            # OpenAI and OpenAI-compatible providers (Grok, DeepSeek)
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if model_provider in OPENAI_COMPATIBLE_BASES:
                client_kwargs["base_url"] = OPENAI_COMPATIBLE_BASES[model_provider]

            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": AUTOCOMPLETE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            suggestion = response.choices[0].message.content or ""

        elif model_provider == 'claude':
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_id,
                system=AUTOCOMPLETE_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            suggestion = response.content[0].text if response.content else ""

        elif model_provider == 'gemini':
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AUTOCOMPLETE_SYSTEM_PROMPT,
                    max_output_tokens=100,
                    temperature=0.3,
                ),
            )
            suggestion = response.text if response.text else ""

        # Minimal deterministic cleanup (whitespace, simple echo, spacing and
        # punctuation at the join). Kept in one place so behaviour stays clear.
        suggestion = _postprocess_autocomplete(text_before, suggestion)

        duration = time.time() - start_time
        logger.info(f"{log_prefix}Autocomplete completed in {duration:.2f}s (suggestion_len={len(suggestion)})")

        return suggestion

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{log_prefix}Autocomplete failed in {duration:.2f}s: {str(e)}")
        return ""
