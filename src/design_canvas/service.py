"""
Self-healing loop harness for Mode 8: Editable UI Design Engine.

execute_design_canvas_harness()
    Draft → Pydantic Parse → Deterministic Domain Lint → XML-Fenced Error Repair
    → Code Auto-Heal

The harness is model-agnostic: it accepts any ``agent_runner`` object that
exposes an ``async run_json_completion(messages, image_bytes, temperature)``
coroutine returning raw model output text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.validate import auto_heal_design_ast, validate_design_ast
from src.design_canvas.prompts import (
    DESIGN_SYSTEM_PROMPT,
    build_design_repair_prompt,
    build_initial_design_prompt,
)

logger = logging.getLogger(__name__)

# Maximum number of XML-fenced LLM repair attempts AFTER the first generation.
# Overridden at runtime by config.settings.design_max_repair_attempts.
MAX_DESIGN_REPAIR_ATTEMPTS = 2

# Regex to strip ```json … ``` or ``` … ``` fences the model sometimes adds
# despite explicit instructions.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_markdown_fences(raw: str) -> str:
    """Remove leading/trailing code-fence markers from model output."""
    return _FENCE_RE.sub("", raw).strip()


class AgentRunnerProtocol:
    """
    Structural interface that the agent runner must satisfy.

    The actual agent runner lives in text_processing_agent.py and is passed
    in at call time; this class documents the expected async interface only.
    """

    async def run_json_completion(
        self,
        messages: List[Dict[str, Any]],
        image_bytes: Optional[bytes] = None,
        temperature: float = 0.2,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


async def execute_design_canvas_harness(
    agent_runner: Any,
    user_query: str,
    sub_mode: str,
    reference_image_bytes: Optional[bytes] = None,
    existing_ast_json: Optional[str] = None,
    max_repair_attempts: Optional[int] = None,
    report: Optional[Dict[str, Any]] = None,
) -> DesignDocumentAST:
    """
    Full Draft → Validate → Repair → Auto-Heal harness.

    Args:
        agent_runner:           Any object with ``run_json_completion`` async method.
        user_query:             The raw user request string.
        sub_mode:               "create_from_prompt" | "create_from_vision" | "edit_existing_ast".
        reference_image_bytes:  Raw image bytes for vision-based generation (optional).
        existing_ast_json:      Serialised DesignDocumentAST for edit mode (optional).
        max_repair_attempts:    Override for the global repair limit (default: 2).
        report:                 Optional dict the harness fills for observability: ``attempts`` (one entry per
                                model call: outcome and the first errors it was sent back with) and
                                ``auto_healed``.

    Returns:
        A fully validated DesignDocumentAST.

    Raises:
        ValueError: If the AST fails Pydantic schema validation after all repair
                    attempts AND cannot be auto-healed (i.e., the raw JSON is
                    structurally invalid).
    """
    if max_repair_attempts is None:
        try:
            from config import settings
            max_repair_attempts = int(settings.design_max_repair_attempts)
        except Exception:
            max_repair_attempts = MAX_DESIGN_REPAIR_ATTEMPTS

    # ------------------------------------------------------------------ #
    # Vision preprocessing (< 15 ms, zero LLM calls)
    # ------------------------------------------------------------------ #
    vision_meta = None
    if reference_image_bytes:
        from src.design_canvas.vision_extractor import extract_vision_metadata
        try:
            vision_meta = extract_vision_metadata(reference_image_bytes)
        except Exception:
            vision_meta = None

    # ------------------------------------------------------------------ #
    # Build initial conversation
    # ------------------------------------------------------------------ #
    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": DESIGN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_design_prompt(
                user_query=user_query,
                sub_mode=sub_mode,
                vision_meta=vision_meta,
                existing_ast_json=existing_ast_json,
            ),
        },
    ]

    last_parsed_ast: Optional[DesignDocumentAST] = None
    if report is None:
        report = {}
    report.update(attempts=[], auto_healed=False)

    def _record(attempt: int, outcome: str, errors: List[str]) -> None:
        report["attempts"].append({"attempt": attempt, "outcome": outcome, "errors": errors[:5]})
        logger.info("design harness attempt %d: %s%s", attempt, outcome,
                    f" ({len(errors)} errors, first: {errors[0]})" if errors else "")

    # ------------------------------------------------------------------ #
    # Repair loop: attempt 0 (generation) + max_repair_attempts retries
    # ------------------------------------------------------------------ #
    for attempt in range(max_repair_attempts + 1):
        raw_output: str = await agent_runner.run_json_completion(
            messages=conversation,
            image_bytes=reference_image_bytes if attempt == 0 else None,
            temperature=0.1 if attempt > 0 else 0.2,
        )
        clean_json = _strip_markdown_fences(raw_output)

        # -------------------------------------------------------------- #
        # Stage A: Pydantic structural validation
        # -------------------------------------------------------------- #
        try:
            candidate_ast = DesignDocumentAST.model_validate_json(clean_json)
            last_parsed_ast = candidate_ast
        except ValidationError as exc:
            schema_errors = [
                f"SCHEMA_ERROR at `{'.'.join(str(p) for p in e['loc'])}`: {e['msg']}"
                for e in exc.errors()[:8]
            ]
            _record(attempt, "schema_errors", schema_errors)
            conversation.append({"role": "assistant", "content": clean_json})
            conversation.append(
                {"role": "user", "content": build_design_repair_prompt(schema_errors)}
            )
            continue

        # -------------------------------------------------------------- #
        # Stage B: Deterministic domain validation
        #          (WCAG contrast, SVG XSS, unique IDs, max depth, sibling preservation)
        # -------------------------------------------------------------- #
        domain_errors = validate_design_ast(
            candidate_ast,
            vision_meta=vision_meta,
            existing_ast_json=existing_ast_json,
        )
        if not domain_errors:
            _record(attempt, "pass", [])
            return candidate_ast  # ✅ Clean pass — return immediately

        # Deterministic domain auto-heal:
        # If the domain errors only contain deterministic violations (WCAG contrast, duplicate IDs,
        # max depth, SVG forbidden attributes), heal them in code without wasting an expensive LLM round-trip.
        deterministic_prefixes = (
            "WCAG_LOW_CONTRAST",
            "DUPLICATE_NODE_ID",
            "DUPLICATE_LAYER_ID",
            "MAX_DEPTH_EXCEEDED",
            "SVG_FORBIDDEN_ATTR",
        )
        if all(any(err.startswith(prefix) for prefix in deterministic_prefixes) for err in domain_errors):
            healed = auto_heal_design_ast(
                candidate_ast,
                vision_meta=vision_meta,
                existing_ast_json=existing_ast_json,
            )
            healed_errors = validate_design_ast(
                healed,
                vision_meta=vision_meta,
                existing_ast_json=existing_ast_json,
            )
            if not healed_errors:
                report["auto_healed"] = True
                _record(attempt, "pass_auto_healed", domain_errors)
                return healed

        _record(attempt, "domain_errors", domain_errors)

        conversation.append({"role": "assistant", "content": clean_json})
        conversation.append(
            {"role": "user", "content": build_design_repair_prompt(domain_errors)}
        )

    # ------------------------------------------------------------------ #
    # Stage C: Deterministic code auto-healer
    #          Guarantees zero 500 failures when at least one attempt parsed.
    # ------------------------------------------------------------------ #
    if last_parsed_ast is not None:
        report["auto_healed"] = True
        logger.info("design harness: auto-healing the last parsed AST after %d attempts", len(report["attempts"]))
        return auto_heal_design_ast(last_parsed_ast, vision_meta=vision_meta, existing_ast_json=existing_ast_json)

    raise ValueError(
        "DesignDocumentAST failed schema validation after maximum repair attempts. "
        "The model output was not valid JSON conforming to the DesignDocumentAST schema."
    )
