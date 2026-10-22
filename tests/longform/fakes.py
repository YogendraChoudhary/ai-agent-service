"""A scripted stand-in for the model the refinement pipeline calls: no provider key, and every way of misbehaving on demand."""

import copy
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from src.longform.llm import CallSpec

_TOKEN = re.compile(r"\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML|URL)_\d+\]\]")


def passage_of(user_message: str) -> str:
    """The passage between the markers of the user message."""
    return user_message.split("<<<PASSAGE\n", 1)[1].rsplit("\nPASSAGE>>>", 1)[0]


# --- what a model can do to a passage -------------------------------------------------------------------------------------

def identity(passage: str, spec: CallSpec) -> str:
    return passage


def polish(passage: str, spec: CallSpec) -> str:
    """Rewords prose only: a model doing its job."""
    return passage.replace("Most ", "Nearly all ").replace(" and ", " plus ").replace("Teams ", "Groups ")


def drop_a_token(passage: str, spec: CallSpec) -> str:
    token = _TOKEN.search(passage)
    return passage.replace(token.group(0), "", 1) if token else passage


def summarise(passage: str, spec: CallSpec) -> str:
    """Keeps the first 35% of the lines: tokens, headings and most of the words are gone."""
    lines = passage.split("\n")
    return "\n".join(lines[:max(1, int(len(lines) * 0.35))])


def merge_paragraphs(passage: str, spec: CallSpec) -> str:
    return re.sub(r"(?<=[a-z.])\n\n(?=[A-Z])", " ", passage)


def add_a_table(passage: str, spec: CallSpec) -> str:
    return passage + "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"


def space_the_tokens(passage: str, spec: CallSpec) -> str:
    """Two trailing spaces after every block token (a Markdown hard break): a model's whitespace habit, seen live on a table."""
    return re.sub(r"(?m)^([ \t]*\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML)_\d+\]\])$", r"\1  ", passage)


def glue_to_the_blocks(passage: str, spec: CallSpec) -> str:
    """Drops the blank lines around every block token, which would glue the next paragraph to a quote or a table."""
    return re.sub(r"\n\n(?=[ \t]*\[\[(?:TABLE|QUOTE|IMAGE|CODE|HTML)_\d+\]\]$)|(?<=\]\])\n\n", "\n", passage, flags=re.M)


def add_a_lead_in(passage: str, spec: CallSpec) -> str:
    """A friendly tone: a new paragraph right under every heading that is followed by a list (seen live on Grok)."""
    return re.sub(r"(?m)^(#{1,6} [^\n]*\n\n)(?=[-*] |\d+\. )", r"\1Hey there, let us look at this together.\n\n", passage)


def wrap(passage: str, spec: CallSpec) -> str:
    return "Here is the refined passage:\n\n```markdown\n" + passage.strip() + "\n```"


def fix_on_repair(bad: Callable[[str, CallSpec], str]) -> Callable[[str, CallSpec], str]:
    """Misbehaves first, then does it right when it is told what was wrong."""
    return lambda passage, spec: bad(passage, spec) if spec.stage == "chunk" else passage


class ScriptedRefiner:
    """
    LLM for refine_stream. `behaviour` is the default; `per_unit` overrides it for a unit ("c2") or for one stage of a unit
    (("c2", "chunk_repair")). `delays` holds a unit back, to prove the output stays in document order.
    """

    def __init__(self, behaviour: Callable[[str, CallSpec], str] = identity,
                 per_unit: Optional[Dict[object, Callable[[str, CallSpec], str]]] = None,
                 delays: Optional[Dict[str, float]] = None):
        self.behaviour = behaviour
        self.per_unit = per_unit or {}
        self.delays = delays or {}
        self.calls: List[Tuple[str, List[dict], CallSpec]] = []
        self._lock = threading.Lock()

    def __call__(self, system: str, messages: List[dict], spec: CallSpec) -> str:
        with self._lock:
            self.calls.append((system, copy.deepcopy(messages), spec))
        if spec.unit in self.delays:
            import time
            time.sleep(self.delays[spec.unit])
        fn = self.per_unit.get((spec.unit, spec.stage)) or self.per_unit.get(spec.unit) or self.behaviour
        return fn(passage_of(messages[0]["content"]), spec)

    def stages(self) -> List[Tuple[str, Optional[str]]]:
        return sorted((c[2].stage, c[2].unit) for c in self.calls)


class FakeAgents:
    """
    Stands in for text_processing_agent.create_agent in the runtime and API tests, so the real routing, hook, pipeline and model
    call code run without a provider.

      - a "longform_refiner" agent is one model call of the chunk pipeline: it answers through `refiner` (a ScriptedRefiner) with the
        system prompt, messages, temperature and output budget the pipeline gave it;
      - any other agent is the single call, the path that runs without the pipeline: it streams `single(query)`.

    fail = {provider: exception | callable(run number) -> exception | None} makes that provider's runs raise;
    hang = {provider: seconds} makes them wait that long before answering (a call that never comes back).
    Records `.created` as (agent_type, provider, key, kwargs), `.runs` as {provider: count} and `.queries` (single calls only).
    """

    SINGLE_ANSWER = "The single call answered."

    def __init__(self, refiner: Optional[ScriptedRefiner] = None, single: Optional[Callable[[str], str]] = None,
                 fail: Optional[dict] = None):
        from types import SimpleNamespace
        self._ns = SimpleNamespace
        self.refiner = refiner or ScriptedRefiner()
        self.single = single or (lambda query: self.SINGLE_ANSWER)
        self.fail = fail or {}
        self.hang: Dict[str, float] = {}
        self.created: List[tuple] = []
        self.runs: Dict[str, int] = {}
        self.queries: List[str] = []
        self._lock = threading.Lock()

    def __call__(self, agent_type, provider, key, request_id=None, **kw):
        with self._lock:
            self.created.append((agent_type, provider, key, kw))
        factory = self

        class Agent:
            model = factory._ns(id=f"fake-{provider}")

            def run(self, history, stream=False, stream_events=False, images=None):
                with factory._lock:
                    n = factory.runs[provider] = factory.runs.get(provider, 0) + 1
                if factory.hang.get(provider):
                    time.sleep(factory.hang[provider])
                failure = factory.fail.get(provider)
                failure = failure(n) if callable(failure) else failure
                if failure is not None:
                    raise failure
                if agent_type == "longform_refiner":
                    messages = [{"role": m.role, "content": m.content} for m in history]
                    stage = "chunk_repair" if len(messages) > 1 else "chunk"
                    text = factory.refiner(kw["instructions"], messages, CallSpec(stage, kw["temperature"], kw["max_tokens"]))
                    return factory._ns(content=text, status=None, event=None, metrics=None)
                with factory._lock:
                    factory.queries.append(history)
                text = factory.single(history)
                if not stream:
                    return factory._ns(content=text, status=None, event=None, metrics=None)
                return self._events(text)

            @staticmethod
            def _events(text):
                from agno.agent import RunEvent
                for i in range(0, len(text), 24):
                    yield factory._ns(event=RunEvent.run_content, content=text[i:i + 24], metrics=None)
                yield factory._ns(event=RunEvent.run_completed, content=None, metrics=None)
        return Agent()

    def agent_types(self) -> List[str]:
        return [c[0] for c in self.created]

    def providers(self) -> List[str]:
        return [c[1] for c in self.created]
