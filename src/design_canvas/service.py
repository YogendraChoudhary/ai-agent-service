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

import asyncio
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.design_canvas.schema import DesignDocumentAST
from src.design_canvas.schema_v11 import DesignDocumentASTV11, Version, downgrade_to_v10
from src.design_canvas.reference import ReferenceSpec, check_fidelity, description_coverage, heal_reference_details
from src.design_canvas.typography_match import match_typography
from src.design_canvas.text_edit import split_selection
from src.design_canvas.validate import auto_heal_design_ast, keep_selected_id, validate_design_ast
from src.design_canvas.patch import compact_design_json
from src.design_canvas.prompts import (
    DESIGN_SYSTEM_PROMPT,
    DESIGN_SYSTEM_PROMPT_V11,
    REFERENCE_SYSTEM_PROMPT,
    build_design_repair_prompt,
    build_reference_coverage_prompt,
    build_reference_repair_prompt,
    build_initial_design_prompt,
)
from src.provider_work import ProviderCancelled

logger = logging.getLogger(__name__)

# Regex to strip ```json … ``` or ``` … ``` fences the model sometimes adds
# despite explicit instructions.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Findings that earn the extra fidelity call (~20 s on Claude): stacked panels, missing copy, wrong or missing fonts.
# Tracking, gradient, raised-symbol and off-palette findings alone only become warnings; when a repair does run, they
# are sent along with the triggering finding.
FIDELITY_REPAIR_TRIGGERS = (
    "FIDELITY_LAYOUT_STACKED", "FIDELITY_TEXT_MISSING", "FIDELITY_FONTS_MISSING", "FIDELITY_FONT_CLASS",
)


def _strip_markdown_fences(raw: str) -> str:
    """Remove leading/trailing code-fence markers from model output."""
    return _FENCE_RE.sub("", raw).strip()


def _parse_design(raw: str, version: Version) -> DesignDocumentAST:
    if version == "1.1":
        return DesignDocumentASTV11.model_validate_json(raw).model_copy(update={"version": "1.1"})
    try:
        return DesignDocumentAST.model_validate_json(raw)
    except ValidationError as legacy_error:
        try:
            return downgrade_to_v10(DesignDocumentASTV11.model_validate_json(raw))
        except ValidationError:
            raise legacy_error from None


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
    schema_version: Optional[Version] = None,
    measure_image_bytes: Optional[bytes] = None,
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
                                ``auto_healed``, plus fidelity scores for vision requests.
        schema_version:         Optional request for the legacy 1.0 wire format.
        measure_image_bytes:    The original upload, for measuring typography at full resolution (default: the
                                reference image the model sees).

    Returns:
        A fully validated DesignDocumentAST.

    Raises:
        ValueError: If the AST fails Pydantic schema validation after all repair
                    attempts AND cannot be auto-healed (i.e., the raw JSON is
                    structurally invalid).
    """
    from config import settings

    if max_repair_attempts is None:
        max_repair_attempts = settings.design_max_repair_attempts
    version = "1.1" if settings.design_schema_v11_enabled and schema_version != "1.0" else "1.0"
    if existing_ast_json:
        try:
            incoming = json.loads(existing_ast_json)
        except (ValueError, RecursionError):        # a deeply nested body raises RecursionError, not ValueError
            incoming = None
        if isinstance(incoming, dict) and incoming.get("version") == "1.1":
            if not settings.design_schema_v11_enabled:
                raise ValueError("Editing a schema 1.1 design requires DESIGN_SCHEMA_V11_ENABLED.")
            if version == "1.0":
                existing_ast_json = downgrade_to_v10(DesignDocumentASTV11.model_validate(incoming)).model_dump_json()

    if report is None:
        report = {}
    report.update(attempts=[], auto_healed=False)

    image = reference_image_bytes if sub_mode == "create_from_vision" else None
    # Copy the screenshot, or use it as a template for new content ("like this, but for pizzas").
    from src.design_intent import wants_new_content
    reference_use = "adapt" if image and wants_new_content(user_query) else "replicate"
    if image:
        report["reference_use"] = reference_use
    vision_meta = None
    if image:
        from src.design_canvas.vision_extractor import extract_vision_metadata
        try:
            vision_meta = extract_vision_metadata(image)
        except Exception:
            logger.warning("Could not extract the design reference palette")

    spec = None
    analysis_warning = None
    if image and settings.design_vision_two_pass:
        describe = [{"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": "Describe the attached design."}]
        # One repair for the description, used for whichever comes first: a slip that makes it invalid (live:
        # '"aspect":"3:2".length?...') or a description that stops short of the page (live: header only, so every
        # later check passed vacuously). Still short after that: it is used, and the result says what went unchecked.
        short_spec = None
        for describe_attempt in range(2):
            try:
                description = await agent_runner.run_json_completion(messages=describe, image_bytes=image)
                candidate = ReferenceSpec.model_validate_json(_strip_markdown_fences(description))
            except ProviderCancelled:
                raise
            except ValidationError as exc:
                logger.warning("Design reference analysis invalid (attempt %d): %s", describe_attempt,
                               exc.errors()[0]["msg"][:120] if exc.errors() else "?")
                if describe_attempt:
                    break
                errors = [f"{'.'.join(str(p) for p in e['loc']) or 'document'}: {e['msg']}" for e in exc.errors()[:8]]
                describe = describe + [
                    {"role": "assistant", "content": _strip_markdown_fences(description)},
                    {"role": "user", "content": build_reference_repair_prompt(errors)}]
                continue
            except Exception as exc:
                logger.warning("Design reference analysis failed (%s)", type(exc).__name__)
                break
            coverage = description_coverage(candidate, image)
            if coverage is not None:
                report["reference_coverage"] = coverage
            if coverage is None or coverage["complete"] or describe_attempt:
                spec = candidate
                break
            logger.warning("Design reference description stops at y=%s of %s px; asking once for the whole page",
                           coverage["described_bottom"], coverage["height"])
            short_spec = candidate
            describe = describe + [
                {"role": "assistant", "content": _strip_markdown_fences(description)},
                {"role": "user", "content": build_reference_coverage_prompt(coverage)}]
        if spec is None and short_spec is not None:
            spec = short_spec                        # the repair failed: the short description is still worth using
        coverage = report.get("reference_coverage")
        partial = spec is not None and coverage is not None and not coverage["complete"]
        if spec is None:
            analysis_warning = "FIDELITY_ANALYSIS_UNAVAILABLE: The image was used directly. The reference structure could not be checked."
        elif partial:
            analysis_warning = (f"FIDELITY_ANALYSIS_PARTIAL: The reference was described down to y={coverage['described_bottom']} "
                                f"of {coverage['height']} px; layout, text and photo checks below that were not run.")
        report["reference_analysis"] = ("unavailable" if spec is None else "partial" if partial
                                        else "pass" if describe_attempt == 0 else "pass_after_repair")

    # Measured typography: family, exact weight and spacing of each boxed display/heading line, from the pixels.
    typography = None
    measured_prompt = None
    if spec and version == "1.1" and any(text.box for text in spec.texts):
        try:
            from PIL import Image
            with Image.open(io.BytesIO(image)) as supplied:
                supplied_width, supplied_ratio = supplied.width, supplied.width / supplied.height
            measure = measure_image_bytes or image
            if measure is not image:
                with Image.open(io.BytesIO(measure)) as original:
                    if abs(original.width / original.height - supplied_ratio) > 0.01 * supplied_ratio:
                        measure = image          # not the same pixels (orientation, crop): boxes would not line up
            typography = await asyncio.to_thread(match_typography, measure, spec, supplied_width)
            report["typography"] = typography.as_report()
            if typography.fonts:
                spec = spec.model_copy(update={"font_guess": {**spec.font_guess, **typography.fonts}})
                texts = {text.id: text for text in spec.texts}
                measured_prompt = {"fonts": typography.fonts, "texts": [
                    {"id": m.text_id, "text": texts[m.text_id].text.split("\n")[0], "font_role": m.role,
                     "font_weight_value": m.weight, "letter_spacing_em": m.letter_spacing_em}
                    for m in typography.texts.values()]}
        except Exception as exc:
            logger.warning("Typography measurement failed (%s)", type(exc).__name__)
            report["typography"] = {"error": type(exc).__name__}
            typography = None

    # ------------------------------------------------------------------ #
    # Build initial conversation
    # ------------------------------------------------------------------ #
    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": DESIGN_SYSTEM_PROMPT_V11 if version == "1.1" else DESIGN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_design_prompt(
                user_query=user_query,
                sub_mode=sub_mode,
                vision_meta=vision_meta,
                existing_ast_json=compact_design_json(existing_ast_json) if existing_ast_json else None,
                reference_spec=spec,
                schema_version=version,
                measured_typography=measured_prompt,
                reference_use=reference_use,
            ),
        },
    ]

    last_parsed_ast: Optional[DesignDocumentAST] = None
    # The editor's selection ("(Apply this to the selected section, id X. ...)"): that node keeps its id in the result.
    selected_id = split_selection(user_query)[1] if existing_ast_json and sub_mode == "edit_existing_ast" else None
    best: Optional[tuple[DesignDocumentAST, dict, list[str], bool]] = None
    quantized = vision_meta.quantized_hex_colors if vision_meta else ()
    expected_accents = vision_meta.roles.accents if vision_meta and vision_meta.roles else ()

    def _evaluate(ast: DesignDocumentAST) -> tuple[dict, list[str]]:
        return check_fidelity(ast.model_dump(), spec, quantized, expected_accents, typography=typography,
                              content="new" if reference_use == "adapt" else "reference")

    def _rank(scores: dict, findings: list[str]) -> tuple:
        return (scores["layout_matches"], scores["text_recall"] or 0, scores["fonts_set"],
                -len(findings), scores["accent_coverage"] or 0)

    def _record(attempt: int, outcome: str, errors: List[str], heal: Optional[List[str]] = None) -> None:
        report["attempts"].append({"attempt": attempt, "outcome": outcome, "errors": errors[:5]})
        if heal:
            report["attempts"][-1]["reference_heal"] = heal
        logger.info("design harness attempt %d: %s%s", attempt, outcome,
                    f" ({len(errors)} errors, first: {errors[0]})" if errors else "")

    # ------------------------------------------------------------------ #
    # Repair loop: attempt 0 (generation) + max_repair_attempts retries
    # ------------------------------------------------------------------ #
    for attempt in range(max_repair_attempts + 1):
        try:
            raw_output = await agent_runner.run_json_completion(
                messages=conversation, image_bytes=image, temperature=0.1 if attempt else 0.2,
            )
        except ProviderCancelled:
            raise
        except Exception as exc:
            if best is None:
                raise
            _record(attempt, "fidelity_provider_error", [type(exc).__name__])
            break
        clean_json = _strip_markdown_fences(raw_output)

        # -------------------------------------------------------------- #
        # Stage A: Pydantic structural validation
        # -------------------------------------------------------------- #
        try:
            candidate_ast = _parse_design(clean_json, version)
            if selected_id:
                candidate_ast = keep_selected_id(candidate_ast, existing_ast_json, selected_id)
            last_parsed_ast = candidate_ast
        except ValidationError as exc:
            schema_errors = [
                f"SCHEMA_ERROR at `{'.'.join(str(p) for p in e['loc'])}`: {e['msg']}"
                for e in exc.errors()[:8]
            ]
            _record(attempt, "schema_errors", schema_errors)
            if best is not None:
                break
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
        # Deterministic domain auto-heal:
        # If the domain errors only contain deterministic violations (WCAG contrast, duplicate IDs,
        # max depth, SVG forbidden attributes), heal them in code without wasting an expensive LLM round-trip.
        deterministic_prefixes = (
            "WCAG_LOW_CONTRAST",
            "DUPLICATE_NODE_ID",
            "DUPLICATE_LAYER_ID",
            "MAX_DEPTH_EXCEEDED",
            "SVG_FORBIDDEN_ATTR",
            # auto_heal_design_ast restores the original id; a model round cost ~45 s for it (audit 2026-11-20).
            "RENAMED_NODE_ID",
        )
        healed_this_attempt = False
        if domain_errors and all(err.startswith(deterministic_prefixes) for err in domain_errors):
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
                healed_this_attempt = True
                candidate_ast = healed

        if not domain_errors or healed_this_attempt:
            outcome = "pass_auto_healed" if healed_this_attempt else "pass"
            if not (spec or quantized or analysis_warning):
                report["auto_healed"] = healed_this_attempt
                _record(attempt, outcome, domain_errors)
                return candidate_ast
            # Tracking and inline rules the reference spec measures are fixed in code: no model call for them.
            heal_changes: List[str] = []
            if spec and version == "1.1":
                raw = candidate_ast.model_dump()
                heal_changes = heal_reference_details(raw, spec, typography)
                if heal_changes:
                    try:
                        healed_ref = type(candidate_ast).model_validate(raw)
                    except ValidationError:
                        healed_ref = None
                    if healed_ref is None or validate_design_ast(
                            healed_ref, vision_meta=vision_meta, existing_ast_json=existing_ast_json):
                        logger.warning("design harness: reference heal discarded (would not validate)")
                        heal_changes = []
                    else:
                        candidate_ast = healed_ref
            scores, findings = _evaluate(candidate_ast)
            try_fidelity_repair = (best is None and settings.design_max_fidelity_repairs
                                   and any(f.startswith(FIDELITY_REPAIR_TRIGGERS) for f in findings)
                                   and attempt < max_repair_attempts)
            if best is None or _rank(scores, findings) > _rank(best[1], best[2]):
                best = candidate_ast, scores, findings, healed_this_attempt
            if not try_fidelity_repair:
                _record(attempt, "pass_with_fidelity_warnings" if findings else outcome, findings or domain_errors,
                        heal_changes)
                break
            _record(attempt, "fidelity_repair", findings, heal_changes)
            conversation.append({"role": "assistant", "content": candidate_ast.model_dump_json()})
            conversation.append({"role": "user", "content": build_design_repair_prompt(
                findings[:8], fidelity=True, has_spec=spec is not None)})
            continue

        _record(attempt, "domain_errors", domain_errors)
        if best is not None:
            break

        conversation.append({"role": "assistant", "content": clean_json})
        conversation.append(
            {"role": "user", "content": build_design_repair_prompt(domain_errors)}
        )

    # ------------------------------------------------------------------ #
    # Stage C: Deterministic code auto-healer
    #          Guarantees zero 500 failures when at least one attempt parsed.
    # ------------------------------------------------------------------ #
    if best is not None:
        result, scores, findings, report["auto_healed"] = best
    elif last_parsed_ast is not None:
        report["auto_healed"] = True
        logger.info("design harness: auto-healing the last parsed AST after %d attempts", len(report["attempts"]))
        result = auto_heal_design_ast(last_parsed_ast, vision_meta=vision_meta, existing_ast_json=existing_ast_json)
        if not (spec or quantized or analysis_warning):
            return result
        scores, findings = _evaluate(result)
    else:
        raise ValueError(
            "DesignDocumentAST failed schema validation after maximum repair attempts. "
            "The model output was not valid JSON conforming to the DesignDocumentAST schema."
        )

    if analysis_warning:
        findings.append(analysis_warning)
    report["fidelity"] = {**scores, "findings": findings}
    return result
