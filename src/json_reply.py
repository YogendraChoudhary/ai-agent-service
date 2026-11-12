"""
Models sometimes wrap a JSON answer in a Markdown code fence (```json ... ```) even when told not to. Claude Opus 5.5 does it on
about one chart-extraction call in six (live, 2026-11-12: 2 of 12), and a strict JSON parser then rejects the whole reply.
"""
import re

# the body may not contain another fence marker, so a reply with two blocks (or prose between blocks) is left alone
_WHOLE_REPLY_IS_ONE_FENCE = re.compile(r"\A\s*```[A-Za-z0-9_-]*[ \t]*\r?\n((?:(?!```).)*?)\r?\n?[ \t]*```\s*\Z", re.DOTALL)


def strip_json_fence(reply: str) -> str:
    """The JSON inside a reply that is exactly one fenced block; any other reply is returned unchanged."""
    match = _WHOLE_REPLY_IS_ONE_FENCE.match(reply or "")
    return match.group(1).strip() if match else reply
