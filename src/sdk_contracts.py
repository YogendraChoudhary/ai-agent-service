"""
Does the INSTALLED provider SDK accept every argument this app sends it? Checked against the real SDK, offline: no keys, no network.

Why: on 2026-11-20 every Claude request failed in production with "Messages.stream() got an unexpected keyword argument
'output_config'". The model upgrade started sending `output_config` (the effort dial); the server had an older anthropic SDK whose
Messages.stream() has no such parameter. Tests passed (they fake the SDK) and the model preflight passed (it only looks models up).

How: each request path is built the way production builds it -- agno paths through get_model() and agno's own request builder, the
direct SDK calls through the same helpers their call sites use -- and the argument names are compared with the signature of the SDK
method that path calls. Also every keyword passed to a google.genai `types.X(...)` model in src/ must be a field of X.

Run at startup (api.py lifespan: logs CRITICAL, optionally refuses to start), before a deploy (scripts/preflight_models.py), and in
tests (tests/core/test_sdk_contracts.py) against the real installed packages.
"""
from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

SDK_PACKAGES = ("anthropic", "openai", "google-genai", "agno")
_DUMMY_KEY = "sdk-contract-check"          # never sent anywhere: nothing here makes a request
_SAMPLE_TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look something up.",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
_SRC = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Finding:
    path: str                 # which app request path, e.g. "agno claude advanced (tools) -> stream"
    method: str               # the SDK method, e.g. "Messages.stream"
    unsupported: Tuple[str, ...]
    detail: str = ""

    def message(self) -> str:
        if self.unsupported:
            return f"{self.path}: {self.method}() does not accept {', '.join(self.unsupported)}"
        return f"{self.path}: {self.detail}"


def unsupported_params(method: Callable, params: Iterable[str]) -> Tuple[str, ...]:
    """Argument names `method` would reject. A method taking **kwargs cannot reject a name statically."""
    sig = inspect.signature(method)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return ()
    return tuple(sorted(set(params) - set(sig.parameters) - {"self"}))


def sdk_versions() -> Dict[str, Optional[str]]:
    out = {}
    for name in SDK_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Request paths: (label, SDK method, argument names) -- built as production builds them
# ---------------------------------------------------------------------------
Path_ = Tuple[str, Callable, Iterable[str], str]     # label, SDK method, argument names, method name


def _agno_paths() -> Iterator[Path_]:
    from anthropic.resources.beta.messages import Messages as BetaMessages
    from anthropic.resources.messages import Messages
    from google.genai.models import Models
    from openai.resources.chat.completions import Completions

    from . import models as M
    from .intent_classifier import build_light_model
    from .text_processing_agent import get_model

    def variants():
        for provider in ("claude", "openai", "gemini", "grok", "deepseek"):
            for use_advanced in (True, False):
                for tools in (False, True):
                    tier = "advanced" if use_advanced else "basic"
                    yield (f"agno {provider} {tier}{' (tools)' if tools else ''}",
                           lambda p=provider, a=use_advanced, t=tools: get_model(p, _DUMMY_KEY, use_advanced=a, needs_tools=t,
                                                                                  cache_model=False), tools)
        for (provider, agent), _model_id in M.AGENT_MODEL_PINS.items():
            yield (f"agno {provider} pinned:{agent}",
                   lambda p=provider, ag=agent: get_model(p, _DUMMY_KEY, use_advanced=True, agent_type=ag, cache_model=False), False)
        for provider in ("claude", "openai", "gemini"):
            yield f"agno {provider} light (classifier)", lambda p=provider: build_light_model(p, _DUMMY_KEY), False

    for label, build, with_tools in variants():
        tools = _SAMPLE_TOOLS if with_tools else None
        model = build()
        cls = type(model).__name__
        if cls == "Claude":
            kwargs = {"model": model.id, "messages": [], **model._prepare_request_kwargs("system", tools=tools)}
            beta = model._has_beta_features(tools=tools)
            api = BetaMessages if beta else Messages
            prefix = "beta.Messages" if beta else "Messages"
            yield f"{label} -> stream", api.stream, kwargs, f"{prefix}.stream"
            yield f"{label} -> create", api.create, kwargs, f"{prefix}.create"
        elif cls == "Gemini":
            kwargs = {"model": model.id, "contents": [], **model.get_request_params("system", tools=tools)}
            yield f"{label} -> stream", Models.generate_content_stream, kwargs, "Models.generate_content_stream"
            yield f"{label} -> create", Models.generate_content, kwargs, "Models.generate_content"
        else:   # OpenAIChat and the OpenAI-compatible Grok / DeepSeek models
            kwargs = {"model": model.id, "messages": [], "stream": True, "stream_options": {},
                      **model.get_request_params(tools=tools)}
            yield f"{label} -> create", Completions.create, kwargs, "chat.Completions.create"


def _direct_paths() -> Iterator[Path_]:
    """SDK calls the app makes itself, with the same argument builders their call sites use."""
    from anthropic.resources.messages import AsyncMessages, Messages
    from openai.resources.chat.completions import AsyncCompletions, Completions

    from config import settings
    from . import models as M
    from .text_processing_agent import FAST_MODEL_IDS, design_openai_kwargs

    base_claude = {"model", "system", "messages", "max_tokens"}
    design_claude = M.DESIGN_MODEL_IDS.get("claude") or M.ADVANCED_MODEL_IDS["claude"]
    yield ("design canvas claude", Messages.create,
           base_claude | set(M.claude_effort_kwargs(design_claude, 0.3, settings.claude_advanced_effort)), "Messages.create")
    yield ("autocomplete claude", Messages.create,
           base_claude | set(M.claude_light_kwargs(FAST_MODEL_IDS["claude"], 0.3)), "Messages.create")
    yield ("voice formatter claude", AsyncMessages.create,
           base_claude | {"timeout"} | set(M.claude_light_kwargs(M.AUTOCOMPLETE_MODEL_IDS["claude"], 0)), "AsyncMessages.create")

    for provider in ("openai", "grok", "deepseek"):
        model_id = M.DESIGN_MODEL_IDS.get(provider) or M.ADVANCED_MODEL_IDS.get(provider)
        if model_id:
            for has_image in (False, True):
                yield (f"design canvas {provider}{' (image)' if has_image else ''}", Completions.create,
                       {"model", "messages"} | set(design_openai_kwargs(provider, model_id, 8000, 0.3, has_image)),
                       "chat.Completions.create")
    yield ("autocomplete openai", Completions.create, {"model", "messages", "max_tokens", "temperature"}, "chat.Completions.create")
    yield ("voice formatter openai", AsyncCompletions.create,
           {"model", "messages", "max_completion_tokens", "max_tokens", "reasoning_effort", "temperature"}, "chat.AsyncCompletions.create")


def _genai_type_findings() -> List[Finding]:
    """Every keyword passed to `types.X(...)` (google.genai.types) in src/ must be a field of X in the installed SDK."""
    from google.genai import types as genai_types

    findings = []
    for file in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "types"):
                continue
            cls = getattr(genai_types, node.func.attr, None)
            fields = getattr(cls, "model_fields", None)
            if fields is None:
                continue
            names = {k.arg for k in node.keywords if k.arg}
            aliases = {f.alias for f in fields.values() if getattr(f, "alias", None)}
            bad = tuple(sorted(names - set(fields) - aliases))
            if bad:
                where = f"{file.relative_to(_SRC.parent)}:{node.lineno}"
                findings.append(Finding(f"{where} types.{node.func.attr}", f"types.{node.func.attr}", bad))
    return findings


def check_sdk_contracts() -> Dict[str, Any]:
    """{"ok", "versions", "checked", "findings": [...]}. Never raises: a path that cannot be built is itself a finding."""
    findings: List[Finding] = []
    checked = 0
    for source in (_agno_paths, _direct_paths):
        try:
            for label, method, params, method_name in source():
                checked += 1
                bad = unsupported_params(method, params)
                if bad:
                    findings.append(Finding(label, method_name, bad))
        except Exception as error:                                     # noqa: BLE001 - an unbuildable path is a finding
            findings.append(Finding(source.__name__.strip("_"), "", (), f"could not build the request: "
                                                                        f"{type(error).__name__}: {error}"))
    try:
        findings += _genai_type_findings()
        checked += 1
    except Exception as error:                                         # noqa: BLE001
        findings.append(Finding("google.genai types", "", (), f"{type(error).__name__}: {error}"))
    return {"ok": not findings, "versions": sdk_versions(), "checked": checked,
            "findings": [{**asdict(f), "message": f.message()} for f in findings]}


def run_startup_check(on_mismatch: str = "warn") -> Dict[str, Any]:
    """
    The startup gate. Findings are logged CRITICAL with event=sdk_contract_mismatch (one line each, easy to alarm on) and kept for
    /agent/health/providers. on_mismatch "fail" raises, so the process refuses to serve; "warn" keeps serving the paths that work.
    """
    from .provider_health import HEALTH

    report = check_sdk_contracts()
    HEALTH.sdk_contracts = report
    versions = ", ".join(f"{k}={v}" for k, v in report["versions"].items())
    if report["ok"]:
        logger.info("SDK contract check OK (%d request paths; %s)", report["checked"], versions)
        return report
    for f in report["findings"]:
        logger.critical("ROLLOUT_ALERT event=sdk_contract_mismatch %s (%s)", f["message"], versions)
    if on_mismatch == "fail":
        raise RuntimeError(f"SDK contract mismatch ({len(report['findings'])} request paths would fail): "
                           f"{report['findings'][0]['message']}. Install requirements.lock.")
    return report
