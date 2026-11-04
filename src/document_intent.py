"""Advisory action suggestions from the task alone; suggestions never execute editor changes."""
import re
from typing import Literal

from pydantic import BaseModel

from .document_context import DocumentAction


class IntentSuggestion(BaseModel):
    suggested_action: DocumentAction
    requested_action: DocumentAction | None
    effective_action: DocumentAction
    advisory: Literal[True] = True
    ambiguous: bool
    reason: str


def suggest_document_intent(query, requested_action=None):
    text = re.sub(r'^(?:(?:please|can you|could you|would you)\s+)+', '', query.strip(), flags=re.I)
    ask = re.match(r"(?:what|why|how|where|when|should|is|are|suggest|critique|review|summari[sz]e|explain|analy[sz]e|résume|resume|qué|cómo|इसका सारांश)\b", text, re.I)
    draft = re.search(r'\b(?:continue(?: writing)?|draft|write (?:the |a )?(?:next|new)|add (?:a |the )?new section)\b|continúa|अगला अनुच्छेद', text, re.I)
    edit = re.search(r'\b(?:rewrite|rephrase|shorten|improve|simplify|make (?:this|it|the .+?) (?:shorter|clearer)|remove repeated|fix grammar|reescribe)\b|दोबारा लिख', text, re.I)
    ambiguous = not ask and bool(draft) == bool(edit)
    suggested = 'ask' if ask or ambiguous else 'draft' if draft else 'edit'
    return IntentSuggestion(suggested_action=suggested, requested_action=requested_action,
        effective_action=requested_action or 'ask', ambiguous=bool(ambiguous),
        reason='question_or_advice' if ask else 'mixed_or_unclear_task' if ambiguous else 'new_content' if draft else 'existing_prose')
