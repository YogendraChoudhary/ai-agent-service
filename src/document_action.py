"""
document_action "auto": decide Ask / Draft / Edit / Write for a request that was sent with a document.

Order of play, cheapest first:
  1. The request itself: a scoped or analysis request only Ask can serve.
  2. Wording rules (no model, instant). They answer only when the wording points one way, never on a mix.
     Write is chosen by the same detectors the Write pipeline routes on (blog, code, image, design, chart), and only when the task
     does not point at the open document.
  3. One light-tier call (code-intent classifier pattern) when the rules cannot tell. Platform users' text goes through the platform's
     Claude key and BYOK users' through their own, exactly as for the code classifier.
  4. Ask, if that call fails or times out. Ask only reads: a wrong guess there costs a re-ask, where a wrong Draft or Edit costs a
     change the user has to reject.

The decision is returned to the editor (document_action_resolved) so it can show which mode ran.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from config import settings
from .document_intent import asks_to_insert, imperative_edit, intent_signals, refers_to_document

logger = logging.getLogger(__name__)

ResolvedAction = Literal["ask", "draft", "edit", "write"]
LABELS = ("ask", "draft", "edit", "write")
EDIT_QUERY_LIMIT = 2000      # Edit takes an instruction of at most this many characters (explicit Edit has the same limit)

# Words that tie a task to the open document even when no pattern in document_intent caught it ("…the pricing section below").
_DOCUMENT_PART = re.compile(r"\b(?:section|paragraph|heading|chapter|document|doc|selection|cursor|above|below)\b", re.I)

_REASON = {"ask": "question_or_advice", "draft": "new_content", "edit": "existing_prose"}

CLASSIFIER_SYSTEM = """A user typed a request into a writing assistant while a document is open. Decide which mode runs it. Reply with ONE lowercase word: ask, draft, edit or write.

ask - the user wants an ANSWER about the document: a question, a summary, feedback, a check, advice. Nothing is written into the document.
  e.g. "How much does the Pro tier cost?", "Is this document too long?", "Summarize the key points", "What is the deadline mentioned?"
draft - the user wants NEW text added to the document: a sentence, paragraph, section, list or table, at a named place or at the cursor.
  e.g. "add a sentence about shipping times under the Delivery heading", "write the next section", "insert a list of requirements after the overview"
edit - the user wants text that is ALREADY in the document changed: rewrite, shorten, fix, rephrase, translate, rename a heading, or a small item added inside an existing line.
  e.g. "make the opening paragraph punchier", "correct the spelling mistakes", "change the Overview heading to Summary", "add Ltd. after the company name"
write - the user wants a NEW standalone piece composed about something else, not about this document: an essay, email, poem, letter, product description, list of ideas, checklist, blog post, code, image, chart or design.
  e.g. "Write a haiku about the sea", "Write an email inviting the team to the kickoff", "Suggest 8 names for a bakery", "Write a short story about a robot"

Judge by what the user wants, not by the first word. A request that points at the document ("this", "the intro", "below the heading") is ask, draft or edit. A question is ask. Only when you truly cannot tell, reply ask."""


@dataclass(frozen=True)
class ActionDecision:
    action: ResolvedAction
    source: Literal["rules", "model", "request", "fallback"]
    reason: str
    ambiguous: bool = False

    def event(self) -> dict:
        """The payload of the document_action_resolved event / the process response field."""
        return {"requested_action": "auto", "action": self.action, "source": self.source, "reason": self.reason,
                "ambiguous": self.ambiguous}


def digest(text: str, limit: int | None = None) -> str:
    """The task itself when short; otherwise its beginning and its end, where the instruction usually is ("write …" / "… , then shorten it")."""
    limit = limit or settings.document_action_digest_chars
    text = text.strip()
    if len(text) <= limit:
        return text
    half = (limit - 3) // 2
    return text[:half].rstrip() + " … " + text[-half:].lstrip()


def _standalone_task(query: str) -> Optional[str]:
    """The Write specialist this task belongs to (the detectors Write itself routes on), or None."""
    from .blog.intent import detect_blog_intent
    from .charts.request import asks_for_chart
    from .code_intent import detect_code_intent
    from .design_intent import detect_design_intent
    from .intent import detect_image_intent
    if detect_blog_intent(query)["kind"] == "blog":
        return "blog"
    if detect_code_intent(query)["kind"] == "code":
        return "code"
    if detect_image_intent(query)["kind"] == "generate":
        return "image"
    if settings.design_canvas_enabled and detect_design_intent(query).is_design_request:
        return "design"
    if asks_for_chart(query):
        return "chart"
    return None


def decide_by_rules(query: str, document_text: str, *, scope: str = "document", analysis=None) -> Optional[ActionDecision]:
    """The action when the request and its wording settle it, else None (the caller then asks the classifier)."""
    if scope != "document" or analysis is not None:
        return ActionDecision("ask", "request", "scope_requires_ask")
    task = digest(query)
    blank = not document_text.strip()
    if not blank and len(query.strip()) <= EDIT_QUERY_LIMIT:
        from .document_edit.in_place import in_place_request
        in_place = in_place_request(query)
        if (in_place is not None and (in_place.phrase is not None or in_place.replacement is not None or in_place.ambiguous)
                and not _question_shaped(task)):
            return ActionDecision("edit", "rules", _REASON["edit"])
    points_at_document = refers_to_document(task) or asks_to_insert(task) or bool(_DOCUMENT_PART.search(task))
    if not points_at_document and _standalone_task(task):
        return ActionDecision("write", "rules", "standalone_task")
    signals = intent_signals(task)
    hits = [name for name in ("ask", "draft", "edit") if signals[name]]
    if len(hits) != 1:
        return None
    action = hits[0]
    if action == "edit" and blank:
        # Nothing to change yet: Ask says so ("There is no document content…") instead of failing.
        return ActionDecision("ask", "rules", "empty_document")
    if action == "edit" and (len(query.strip()) > EDIT_QUERY_LIMIT or not imperative_edit(task) or _question_shaped(task)):
        # An edit word somewhere inside a sentence ("Polish bakery traditions", "correct me if I am wrong, but does …") is not an
        # instruction to edit; the classifier reads it.
        return None
    if action == "ask" and _GENERATIVE_LEAD.match(task) and not points_at_document:
        return None      # "Suggest names for a new sandwich" asks for something written, not for advice about the document
    return ActionDecision(action, "rules", _REASON[action])


_POLITE_LEAD = re.compile(r"(?:please\b|(?:can|could|would|will)\s+you\b|i\s+(?:want|need|would\s+like)\b)", re.I)
_GENERATIVE_LEAD = re.compile(r"(?:(?:please|kindly)\s+)?(?:suggest|brainstorm|come\s+up\s+with)\b", re.I)


def _question_shaped(task: str) -> bool:
    """A '?' in a task that is not a polite request ("Can you shorten the intro?") makes it a question until the classifier says otherwise."""
    return "?" in task and not _POLITE_LEAD.match(task)


def _hint(document_text: str, has_selection: bool) -> str:
    words = len(document_text.split())
    return f"[document: {'empty' if not words else f'{words} words'}; {'text is selected' if has_selection else 'no selection'}]"


def decide_with_model(query: str, document_text: str, *, has_selection: bool = False, classify: Optional[Callable] = None) -> ActionDecision:
    """
    The light-tier answer, or Ask when there is none. `classify(text, system, labels)` returns one of LABELS or None; the endpoints
    bind it to the request's provider and key.
    """
    label = None
    if classify is not None and settings.document_action_classifier_enabled:
        try:
            label = classify(f"{_hint(document_text, has_selection)}\n{digest(query)}", CLASSIFIER_SYSTEM, LABELS)
        except Exception as error:      # a classifier problem must never fail the request
            logger.warning("Document action classifier failed (%s)", type(error).__name__)
    if label is None:
        return ActionDecision("ask", "fallback", "classifier_unavailable" if classify is not None else "mixed_or_unclear_task", True)
    if label in ("edit",) and not document_text.strip():
        return ActionDecision("ask", "model", "empty_document")
    return ActionDecision(label, "model", "model_choice")
