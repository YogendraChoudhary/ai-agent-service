"""Keep provider cache counters through Agno's per-model usage aggregation."""

from agno.models.anthropic import Claude as AgnoClaude
from agno.models.deepseek import DeepSeek as AgnoDeepSeek
from agno.models.google import Gemini as AgnoGemini
from agno.models.openai import OpenAIChat as AgnoOpenAIChat
from agno.models.xai import xAI as AgnoXAI


def _tokens(data, field):
    value = data.get(field) if isinstance(data, dict) else getattr(data, field, None)
    return value if type(value) is int and value >= 0 else None


def provider_usage(provider, usage):
    """Return total input, output, and counters that Agno can sum across model calls."""
    read = write = miss = None
    if provider == "claude":
        tokens_in = _tokens(usage, "input_tokens")
        tokens_out = _tokens(usage, "output_tokens")
        read = _tokens(usage, "cache_read_input_tokens")
        write = _tokens(usage, "cache_creation_input_tokens")
        if tokens_in is not None:
            tokens_in += (read or 0) + (write or 0)
    elif provider == "gemini":
        tokens_in = _tokens(usage, "prompt_token_count")
        tokens_out = _tokens(usage, "candidates_token_count")
        read = _tokens(usage, "cached_content_token_count")
    else:
        tokens_in = _tokens(usage, "prompt_tokens")
        tokens_out = _tokens(usage, "completion_tokens")
        if provider == "deepseek":
            read = _tokens(usage, "prompt_cache_hit_tokens")
            miss = _tokens(usage, "prompt_cache_miss_tokens")
            if tokens_in is None and read is not None and miss is not None:
                tokens_in = read + miss
        else:
            details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
            read = _tokens(details, "cached_tokens")
            write = _tokens(details, "cache_write_tokens")

    counters = {
        "cache_provider": provider,
        "cache_calls": 1,
        "cache_read_reported_calls": int(read is not None),
        "cache_hit_calls": int(read is not None and read > 0),
        "cache_write_reported_calls": int(write is not None),
    }
    if tokens_in is not None:
        counters["cache_input_tokens"] = tokens_in
    if read is not None:
        counters["cache_read_tokens"] = read
        if tokens_in is not None:
            counters["cache_input_tokens_with_read_data"] = tokens_in
    if write is not None:
        counters["cache_write_tokens"] = write
    if miss is not None:
        counters["cache_miss_tokens"] = miss
    return tokens_in, tokens_out, counters


class _CacheMetrics:
    def _get_metrics(self, response_usage):
        metrics = super()._get_metrics(response_usage)
        _, _, counters = provider_usage(self.cache_provider, response_usage)
        metrics.provider_metrics = {**(metrics.provider_metrics or {}), **counters}
        return metrics


class Claude(_CacheMetrics, AgnoClaude):
    cache_provider = "claude"


class OpenAIChat(_CacheMetrics, AgnoOpenAIChat):
    cache_provider = "openai"


class Gemini(_CacheMetrics, AgnoGemini):
    cache_provider = "gemini"


class Grok(_CacheMetrics, AgnoXAI):
    cache_provider = "grok"


class DeepSeek(_CacheMetrics, AgnoDeepSeek):
    cache_provider = "deepseek"
