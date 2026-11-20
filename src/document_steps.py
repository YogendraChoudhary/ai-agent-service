"""
Several steps in one Auto prompt ("Shorten the introduction and add a FAQ section at the end").

Auto used to pick ONE action, so the second half of such a prompt was dropped. Here the prompt is split into at most
MAX_STEPS parts, each part gets its action from Auto's own rules (or its light classifier), and the steps run in order,
each on the document as it is after the steps before it. Every step is an ordinary nested run of the pipeline (Edit, Draft
or Ask, with their checks and fallbacks), and comes back in the format that action already uses, so the editor can keep or
discard each part. A step that fails does not stop the others.

A prompt that is one task, or that cannot be split safely, is not a plan: the caller runs it exactly as before.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

logger = logging.getLogger(__name__)

MAX_STEPS = 3

# A new part starts only where a request verb starts it: "Shorten the intro and add a FAQ" splits, "Fix typos and grammar",
# "Translate to French and Spanish" and "Make it shorter and friendlier" do not.
_STEP_VERBS = (r"(?:add|insert|append|include|write|draft|create|shorten|lengthen|expand|rewrite|reword|rephrase|translate|"
               r"fix|correct|proofread|polish|improve|tighten|simplify|summari[sz]e|make|change|replace|rename|update|turn|"
               r"convert|put|answer|explain|tell\s+me|list|check|review|suggest|generate|give\s+me)")
_SPLIT = re.compile(
    r"(?:\s*[;.!?]\s+|\s*\n+\s*|\s*,\s*(?:and\s+)?(?:then\s+)?(?:also\s+)?|\s+and\s+(?:then\s+|also\s+)?|\s+then\s+|"
    r"\s+after\s+that,?\s+|\s+next,?\s+)(?=(?:please\s+)?" + _STEP_VERBS + r"\b)", re.IGNORECASE)
# A later part that points back at an earlier one's result ("add a FAQ and make it shorter") is one task, not two.
_POINTS_BACK = re.compile(r"^(?:please\s+)?\w+(?:\s+\w+)?\s+(?:it|this|that|them|these|those)\b", re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]{2,}'|‘[^’]*’|`[^`]*`")
_LEAD = re.compile(r"^(?:and|then|also|next|after\s+that|please)[\s,]+", re.IGNORECASE)


@dataclass(frozen=True)
class Step:
    number: int
    action: str          # "ask" | "draft" | "edit"
    instruction: str
    label: str           # short, for "Step 1 of 2: Shorten the introduction"
    source: str          # how the action was decided: "rules" | "model"

    def public(self) -> dict:
        return {"step": self.number, "action": self.action, "label": self.label}


def split_parts(query: str) -> List[str]:
    """The prompt's parts at request-verb boundaries (quoted text is never split). One part when it is one task."""
    text = (query or "").strip()
    if not text or len(text) > 2000:
        return [text]
    masked = _QUOTED.sub(lambda m: "\x01" * len(m.group(0)), text)      # same length: offsets still line up
    cuts, at = [], 0
    for match in _SPLIT.finditer(masked):
        cuts.append(text[at:match.start()])
        at = match.end()
    cuts.append(text[at:])
    parts = [_LEAD.sub("", part.strip(" ,;.")).strip() for part in cuts]
    parts = [p for p in parts if p]
    if len(parts) < 2 or any(len(p.split()) < 2 for p in parts) or any(_POINTS_BACK.match(p) for p in parts[1:]):
        return [text]
    return parts


def _label(instruction: str) -> str:
    words = instruction.split()
    label = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
    return label[:1].upper() + label[1:]


def plan_steps(query: str, document_text: str, *, has_selection: bool = False,
               classify: Optional[Callable[[str], Optional[str]]] = None) -> Optional[List[Step]]:
    """
    The steps of a several-task Auto prompt, or None to run it as one action (as before). `classify(part)` is Auto's light
    classifier for a part the rules cannot settle; without it, or when it cannot tell, the prompt is not split.
    """
    from config import settings
    from .document_action import _standalone_task, decide_by_rules
    if has_selection or not getattr(settings, "document_steps_enabled", True):
        return None
    parts = split_parts(query)
    if len(parts) < 2:
        return None
    decided = []
    for part in parts:
        decision = decide_by_rules(part, document_text)
        action, source = (decision.action, "rules") if decision is not None else (None, "model")
        if action is None and classify is not None:
            action = classify(part)
        if action is None or (decision is not None and decision.source == "fallback"):
            return None
        if action == "write":
            # A standalone piece inside a several-step prompt is written into the document (Draft); pictures, designs, code
            # and charts are Write-only, so such a prompt keeps the single-action path.
            if _standalone_task(part) not in (None, "blog"):
                return None
            action = "draft"
        decided.append((action, part, source))
    merged: List[tuple] = []
    for action, part, source in decided:
        if merged and merged[-1][0] == action and action in ("edit", "ask"):
            # One Edit run changes several paragraphs; one Ask answers several questions. Drafts stay apart: each has a place.
            previous = merged[-1]
            merged[-1] = (action, f"{previous[1]}; and {part[:1].lower()}{part[1:]}", previous[2] if previous[2] == source else "model")
        else:
            merged.append((action, part, source))
    if len(merged) < 2 or len(merged) > MAX_STEPS:
        return None
    return [Step(i, action, part, _label(part), source) for i, (action, part, source) in enumerate(merged, 1)]


def steps_resolution(steps: List[Step]) -> dict:
    """The document_action_resolved payload of a several-step prompt."""
    return {"requested_action": "auto", "action": "steps", "source": "model" if any(s.source == "model" for s in steps) else "rules",
            "reason": "several_tasks", "ambiguous": False, "steps": [s.public() for s in steps]}


# ----------------------------------------------------------------------------------------------------------------- running

def _utf16_to_index(text: str, offset: int) -> int:
    from .document_context import source_index
    return source_index(text, offset)


def _insert_draft(text: str, markdown: str, placement, cursor: Optional[int]) -> tuple[str, int, int]:
    """(new text, character index where the draft went, UTF-16 offset of that index) for one Draft result."""
    from .document_edit.blocks import utf16_length
    if placement is not None:
        at = _utf16_to_index(text, placement.end_utf16 if placement.position == "after" else placement.start_utf16)
    elif cursor is not None:
        at = min(cursor, len(text))
    else:
        at = len(text)
    before, after = text[:at], text[at:]
    lead = "" if not before or before.endswith("\n\n") else "\n" if before.endswith("\n") else "\n\n"
    tail = "" if not after else "\n\n" if not after.startswith("\n") else "\n" if not after.startswith("\n\n") else ""
    block = markdown.strip("\n")
    new_text = f"{before}{lead}{block}{tail}{after}" if block else text
    return new_text, at + len(lead), utf16_length(before + lead)


def _shift_cursor(cursor: Optional[int], source: str, patches) -> Optional[int]:
    """The cursor's place after an Edit: moved by the patches that end before it."""
    if cursor is None:
        return None
    delta = 0
    for patch in patches:
        start = _utf16_to_index(source, patch["start_utf16"])
        end = _utf16_to_index(source, patch["end_utf16"])
        if end <= cursor:
            delta += len(patch["after"]) - (end - start)
    return cursor + delta


def run_steps(steps: List[Step], document_text: str, cursor_utf16: Optional[int], nested: Callable[..., Iterator],
              *, references=None, image_count: int = 0, cancel_event=None) -> Iterator:
    """
    Yields the pipeline chunks of a several-step prompt. `nested(query, **inputs)` runs the normal pipeline for one step
    (inputs: document_text for Edit, prepared_context for Ask/Draft). Events: document_step_started / document_step_finished
    per step, then document_steps_ready; one {"__pending_images__"} dict shared by every step that makes photos; one
    {"__steps_billing__"} with the characters to bill (each step as if it were sent alone; nothing for a failed step).
    """
    from .attachments import VISION_INPUT_CHARS_PER_IMAGE, reference_chars
    from .document_context import CONTEXT_ERROR, DocumentContext, prepare_context, with_attachments
    from .document_edit.blocks import fingerprint
    from .provider_errors import public_error_message
    from .provider_work import ProviderCancelled

    shared_images: dict = {}
    nested_dicts: list = []
    yield {"__pending_images__": shared_images}
    current = document_text
    cursor = _utf16_to_index(document_text, cursor_utf16) if cursor_utf16 is not None else None
    total, summaries, billed_in, billed_out, model_used = len(steps), [], 0, 0, None
    attached_chars = reference_chars(references) + VISION_INPUT_CHARS_PER_IMAGE * image_count
    try:
        for step in steps:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield {"__event__": {"event": "document_step_started", "data": {**step.public(), "total": total}}}
            base = fingerprint(current)
            finished = {**step.public(), "total": total, "base_sha256": base}
            try:
                if step.action == "edit":
                    edit, snapshot = None, ""
                    for chunk in nested(step.instruction, document_text=current):
                        if isinstance(chunk, dict) and "__pending_images__" in chunk:
                            nested_dicts.append(chunk["__pending_images__"])
                        elif isinstance(chunk, dict) and "__model_used__" in chunk:
                            model_used = model_used or chunk["__model_used__"]
                        elif isinstance(chunk, dict) and chunk.get("__event__", {}).get("event") == "document_edit_ready":
                            edit = chunk["__event__"]["data"]
                        elif isinstance(chunk, str):
                            snapshot += chunk
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if edit is None:
                        raise RuntimeError("The edit did not return a proposal")
                    finished.update(status=edit["status"], message=edit["message"], document_edit=edit)
                    if edit["status"] == "proposed":
                        billed_in += len(step.instruction) + len(current) + attached_chars
                        billed_out += sum(len(p["after"]) for p in edit["patches"])
                        cursor = _shift_cursor(cursor, current, edit["patches"])
                        current = snapshot
                else:
                    context = DocumentContext(text=current, cursor_utf16=None)
                    prepared = with_attachments(prepare_context(context, step.instruction, step.action), references, image_count)
                    metadata = prepared.metadata
                    if prepared.message is not None:
                        literal = metadata.source == "literal"
                        text, status = (prepared.message, "proposed") if literal else ("", metadata.status)
                        finished.update(status=status, message=None if literal else prepared.message)
                    else:
                        text = ""
                        for chunk in nested(step.instruction, prepared_context=prepared):
                            if isinstance(chunk, dict) and "__content_replace__" in chunk:
                                text = chunk["__content_replace__"]
                            elif isinstance(chunk, dict) and "__model_used__" in chunk:
                                model_used = model_used or chunk["__model_used__"]
                            elif isinstance(chunk, dict) and chunk.get("__event__", {}).get("event") == "document_context_ready":
                                metadata = metadata.model_validate(chunk["__event__"]["data"])
                            elif isinstance(chunk, str):
                                text += chunk
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        if not text.strip():
                            raise RuntimeError("The model returned an empty answer")
                        billed_in += metadata.billable_input_chars
                        billed_out += len(text)
                        status = "answered" if step.action == "ask" else "proposed"
                        finished.update(status=status, message=None)
                    finished["document_context"] = metadata.model_dump(mode="json")
                    if step.action == "ask":
                        finished["answer"] = text
                    else:
                        finished["markdown"] = text
                        if finished["status"] == "proposed" and text.strip():
                            current, cursor, at_utf16 = _insert_draft(current, text, metadata.placement, cursor)
                            cursor += len(text.strip("\n"))
                            finished["insert_at_utf16"] = at_utf16
            except (ProviderCancelled, GeneratorExit):
                raise
            except Exception as error:
                if cancel_event is not None and cancel_event.is_set():
                    return
                logger.warning("Document step %d (%s) failed: %s", step.number, step.action, type(error).__name__)
                finished.update(status="failed", error=public_error_message(error) or CONTEXT_ERROR)
            finished["result_sha256"] = fingerprint(current)
            summaries.append({**step.public(), "status": finished["status"]})
            yield {"__event__": {"event": "document_step_finished", "data": finished}}
        if model_used:
            yield {"__model_used__": model_used}
        yield {"__steps_billing__": {"input_chars": billed_in, "output_chars": billed_out}}
        yield {"__event__": {"event": "document_steps_ready", "data": {
            "steps": summaries, "base_sha256": fingerprint(document_text), "result_sha256": fingerprint(current),
            "document_text": current}}}
    finally:
        # Photos started by any step reserve credits: they are all settled with the request, aborted or not.
        for images in nested_dicts:
            for key, entry in images.items():
                if not str(key).startswith("__"):
                    shared_images[key] = entry
