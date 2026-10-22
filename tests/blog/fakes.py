"""
A scripted stand-in for the model, shared by the blog tests and scripts/evaluate_blog.py (offline mode).

ScriptedLLM behaves like a competent writer: it READS the request it is sent (the plan numbers for the outline,
the part's assignments for a section), so a green test also proves the pipeline handed the model the right data.
Individual replies can be overridden to script a failure: a bad outline, a section without its table, an
attributed quote, a slow part.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from typing import Callable, Dict, List, Optional, Union

from src.blog.llm import CallSpec
from src.blog.prompts import load_example

Reply = Union[str, dict, Callable[[dict, CallSpec], Union[str, dict]]]

_WORDS = ("clear purpose shared notes simple habit steady rhythm honest feedback small steps careful planning "
          "useful context calm review practical choice real example plain language short cycle open question "
          "fair trade focused effort quiet progress better outcome").split()


def request_of(message: str) -> dict:
    """The JSON after 'REQUEST:' in a user message."""
    return json.loads(message.split("REQUEST:\n", 1)[1])


def filler(words: int, offset: int = 0) -> str:
    """`words` words in sentences of at most 14 words, each paragraph-safe."""
    out, sentence = [], []
    for i in range(words):
        sentence.append(_WORDS[(i + offset) % len(_WORDS)])
        if len(sentence) == 14 or i == words - 1:
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    return " ".join(out)


def good_outline(request: dict) -> dict:
    """A valid outline for the plan in the request: the matching few-shot example, adjusted to the plan's images."""
    plan = request["plan"]
    length = "short" if plan["target_words"] <= 1000 else "standard" if plan["target_words"] <= 1600 else "long"
    outline = copy.deepcopy(load_example(length)["output"])
    if plan["images"] < sum(v["kind"] == "image" for s in outline["sections"] for v in s["visuals"]):
        keep = plan["images"]
        for section in outline["sections"]:
            kept = []
            for v in section["visuals"]:
                if v["kind"] == "image":
                    if keep <= 0:
                        continue
                    keep -= 1
                kept.append(v)
            section["visuals"] = kept
        # dropping pictures can leave two neighbours bare: give the longer one a quote
        for a, b in zip(outline["sections"], outline["sections"][1:]):
            if not a["visuals"] and not b["visuals"]:
                b["visuals"].append({"kind": "quote", "mode": "pull"})
    return outline


def good_body(request: dict) -> str:
    """Markdown for one part that satisfies the lint: right length, its table and quote, runs broken by a list."""
    write, assignments = request["write"], request.get("assignments", {})
    words, limit = write["target_words"], request["limits"]["max_words_without_visual_break"]
    parts: List[str] = []
    table, quote, takeaways = assignments.get("table"), assignments.get("quote"), assignments.get("takeaways")
    reserved = 20 * bool(quote) + 15 * bool(takeaways) + (20 if words > limit else 0)
    prose = max(30, words - reserved)
    half = prose // 2
    parts.append(filler(half))
    if table:
        cols = table["columns"]
        rows = [f"| {' | '.join(cols)} |", f"|{'---|' * len(cols)}"]
        rows += [f"| {' | '.join(f'{c} {r + 1}' for c in cols)} |" for r in range(table.get("rows_hint", 3))]
        parts += [filler(12, 3), "\n".join(rows)]
    if quote:
        parts.append("> " + filler(14, 5))
    if takeaways:
        parts.append("\n".join(f"- {filler(6, i)}" for i in range(takeaways)))
    elif words > limit and not table:
        parts.append("\n".join(f"- {filler(6, i)}" for i in range(3)))
    parts.append(filler(prose - half, 9))
    return "\n\n".join(parts)


def with_quote(body: str, quote: str) -> str:
    """`body` with its pull quote replaced by `quote` (a sourced quote REPLACES the assigned one; it is not an extra)."""
    import re
    return re.sub(r"^> .*$", quote, body, count=1, flags=re.M)


class ScriptedLLM:
    """
    outline: a reply, or a list of replies for the outline then the repair (the last one repeats).
    units:   {unit id: reply or [replies for the call, then the repair]}; a reply may be a callable
             (request, spec) -> text, for a slow or failing part.
    A call made with `spec.on_text` is streamed: the listener gets the reply `chunk` characters at a time, as a prefix.
    """

    def __init__(self, outline: Optional[Union[Reply, List[Reply]]] = None, units: Optional[Dict[str, Union[Reply, List[Reply]]]] = None,
                 chunk: int = 24):
        self.chunk = chunk
        self.outline = outline if isinstance(outline, list) else ([outline] if outline is not None else None)
        self.units = {k: (v if isinstance(v, list) else [v]) for k, v in (units or {}).items()}
        self.calls: List[tuple] = []
        self._served: Dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, system: str, messages: List[dict], spec: CallSpec) -> str:
        request = request_of(messages[0]["content"])
        with self._lock:
            self.calls.append((system, copy.deepcopy(messages), spec))
            key = f"{spec.stage.split('_')[0]}:{spec.unit or ''}"
            n = self._served[key] = self._served.get(key, 0) + 1
        scripted = self.outline if spec.stage.startswith("outline") else self.units.get(spec.unit)
        if scripted is not None:
            reply = scripted[min(n - 1, len(scripted) - 1)]
        else:
            reply = good_outline(request) if spec.stage.startswith("outline") else good_body(request)
        if callable(reply):
            reply = reply(request, spec)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        if spec.on_text is not None:
            for end in range(self.chunk, len(text) + self.chunk, self.chunk):
                spec.on_text(text[:end])
        return text

    def calls_for(self, stage: str, unit: Optional[str] = None) -> List[tuple]:
        return [c for c in self.calls if c[2].stage == stage and (unit is None or c[2].unit == unit)]


class FakeImageTool:
    """The app's image tool as the pipeline sees it: prompt in, 'Image URL: placeholder::<uuid>' (or an error) out."""

    def __init__(self, fail: Optional[Callable[[str], Optional[str]]] = None):
        self.prompts: List[str] = []
        self.placeholders: List[str] = []
        self.fail = fail

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail and (message := self.fail(prompt)):
            return message
        pid = str(uuid.uuid4())
        self.placeholders.append(pid)
        return f"Image URL: placeholder::{pid}"


# ---------------------------------------------------------------------------
# The app's agent factory, faked: the unmodified AgnoLLM / runtime / stream hook run against it
# ---------------------------------------------------------------------------

def spec_from_agent_call(kw: dict, messages: List[dict]) -> CallSpec:
    """What the pipeline's CallSpec was, recovered from what create_agent received."""
    request = request_of(messages[0]["content"])
    outline = kw["instructions"].startswith("You plan editorial blog posts")
    stage = ("outline" if outline else "unit") + ("_repair" if len(messages) > 1 else "")
    return CallSpec(stage, kw["temperature"], kw["max_tokens"], use_web=kw["plan"].use_web,
                    unit=None if outline else request["write"]["id"])


class FakeAgentFactory:
    """
    Stands in for text_processing_agent.create_agent. Every agent it builds answers through `llm` (a ScriptedLLM) unless
    `fail` says the provider is down: fail = {provider: exception or callable(call number) -> exception or None}. A run
    made with stream=True answers in `chunk`-character content events, as agno's streaming does.
    Records every creation in `.created` as (agent_type, provider, key, kwargs).
    """

    def __init__(self, llm: Optional["ScriptedLLM"] = None, fail: Optional[dict] = None, chunk: int = 24):
        from types import SimpleNamespace
        self._ns = SimpleNamespace
        self.llm = llm or ScriptedLLM()
        self.fail = fail or {}
        self.chunk = chunk                              # characters per streamed content event
        self.streamed: List[bool] = []                  # per run, in order: was it made with stream=True
        self.mid_stream: Dict[str, Exception] = {}      # {provider: error}: a streamed run dies after its first chunk
        self.created: List[tuple] = []
        self.runs: Dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, agent_type, provider, key, request_id=None, **kw):
        with self._lock:
            self.created.append((agent_type, provider, key, kw))
        factory = self

        class Agent:
            model = factory._ns(id=f"fake-{provider}")

            def run(self, history, stream=False, stream_events=False):
                messages = [{"role": m.role, "content": m.content} for m in history]
                with factory._lock:
                    n = factory.runs[provider] = factory.runs.get(provider, 0) + 1
                    factory.streamed.append(bool(stream))
                failure = factory.fail.get(provider)
                failure = failure(n) if callable(failure) else failure
                if failure is not None:
                    raise failure
                text = factory.llm(kw["instructions"], messages, spec_from_agent_call(kw, messages))
                if not stream:
                    return factory._ns(content=text, status=None, event=None, metrics=None)
                return self._events(text, provider)

            @staticmethod
            def _events(text, provider):
                """What agno streams: content events in small chunks, then a completed event."""
                from agno.agent import RunEvent
                for n, i in enumerate(range(0, len(text), factory.chunk)):
                    if n == 1 and provider in factory.mid_stream:
                        raise factory.mid_stream[provider]
                    yield factory._ns(event=RunEvent.run_content, content=text[i:i + factory.chunk], metrics=None)
                yield factory._ns(event=RunEvent.run_completed, content=None, metrics=None)
        return Agent()

    def providers(self) -> List[str]:
        return [c[1] for c in self.created]
