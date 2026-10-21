"""
Which category builds an email. Code decides; the model only ever sees the resolved category.

Precedence: the category the user selected -> a regex match on the user's brief -> "general".
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping, Optional

from .categories import GENERAL, Category, enabled_categories, get_category


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def score_brief(brief: str, categories: Mapping[str, Category]) -> dict[str, int]:
    """Distinct matched phrases per category (a phrase repeated in the brief counts once)."""
    scores = {}
    for cid, cat in categories.items():
        if not cat.pattern:
            continue
        hits = {m.group(0).lower() for m in _compiled(cat.pattern).finditer(brief or "")}
        scores[cid] = len(hits)
    return scores


def match_category(brief: str, categories: Optional[Mapping[str, Category]] = None) -> Optional[str]:
    """Highest-scoring category, ties broken by registry order; None when nothing matches."""
    categories = enabled_categories() if categories is None else categories
    scores = score_brief(brief, categories)
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    return next(cid for cid in categories if scores.get(cid) == best)


def resolve_category(brief: str, selected: Optional[str] = None) -> tuple[Category, str]:
    """(category, routing) where routing is "selected", "matched" or "general"."""
    if selected:
        return get_category(selected), "selected"
    matched = match_category(brief)
    if matched:
        return get_category(matched), "matched"
    return get_category(GENERAL), "general"
