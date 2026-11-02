"""Reuse request-scoped provider eligibility, fallback and usage accounting."""
from __future__ import annotations

import logging

from config import settings
from src.longform.llm import AgnoLLM
from src.observability import trace_set
from .blocks import fingerprint
from .schema import DocumentEditResult
from .service import EditCancelled, edit_document

logger = logging.getLogger(__name__)


def stream_document_edit(source, instruction, *, candidates, key_for, is_byok, user_id=None,
                         request_id=None, report_model=False, cancel_event=None):
    llm = AgnoLLM(candidates, key_for, is_byok=is_byok, user_id=user_id, request_id=request_id,
                  agent_type="document_editor", timeout=float(settings.longform_call_timeout_seconds) or None)
    trace_set(request_id, mode="document_edit", agent_type="document_editor")
    try:
        text, result = edit_document(source, instruction, llm, cancel_event=cancel_event)
    except EditCancelled:
        return
    except Exception as error:
        logger.warning("[%s] Document proposal failed (%s)", request_id, type(error).__name__)
        text = source
        result = DocumentEditResult(status="failed", base_sha256=fingerprint(source), result_sha256=fingerprint(source),
                    message="A safe document edit could not be produced. Your document was kept unchanged. Please retry or specify the sections.")
    if cancel_event is not None and cancel_event.is_set():
        return
    if report_model and (llm.used or llm.drafting_model()):
        yield {"__model_used__": llm.used or llm.drafting_model()}
    trace_set(request_id, document_edit_status=result.status, edited_blocks=len(result.patches))
    yield {"__event__": {"event": "document_edit_ready", "data": result.model_dump(mode="json")}}
    # A complete verified snapshot, never a partially generated replacement. The frontend buffers this mode.
    yield text
