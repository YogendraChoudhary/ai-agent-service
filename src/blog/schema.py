"""
Blog outline format, schema v1.

The model writes an Outline as JSON; code validates it (outline.py), overwrites the fields it owns (section ids,
image positions) and fills any visual the model left out. Every model forbids unknown fields, so a field the model
invents is an error rather than silently kept. Limits live in this module only: the prompt spec (prompts.py) is
generated from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
TITLE_MAX = 120
SUBTITLE_MAX = 200
HEADING_MAX = 90
ANGLE = (10, 500)             # guidance for the writer; one past this is cut by code (outline.trim_guidance), never repaired
HOOK_MAX = 500                # same: guidance, trimmed by code
SECTIONS = (3, 8)
SECTION_WORDS = (60, 800)
EDGE_WORDS = (40, 300)          # intro and conclusion
TAKEAWAYS = (1, 5)
VISUALS_PER_SECTION_MAX = 3     # at most one image, one table and one quote
ALT = (3, 150)
CAPTION_MAX = 160
IMAGE_PROMPT = (10, 600)
TABLE_TITLE = (3, 100)
TABLE_COLUMNS = (2, 5)
TABLE_COLUMN_MAX = 40
TABLE_ROWS_HINT = (2, 8)

SECTION_ID = r"^s[1-9][0-9]?$"
ImagePosition = Literal["after_heading", "after_first_paragraph", "end"]
IMAGE_POSITION: ImagePosition = "after_first_paragraph"   # code-owned: always overwritten


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImageDirective(_Strict):
    kind: Literal["image"]
    prompt: str = Field(min_length=IMAGE_PROMPT[0], max_length=IMAGE_PROMPT[1])
    alt: str = Field(min_length=ALT[0], max_length=ALT[1])
    caption: str = Field("", max_length=CAPTION_MAX)
    position: Optional[ImagePosition] = None


class TableDirective(_Strict):
    kind: Literal["table"]
    title: str = Field(min_length=TABLE_TITLE[0], max_length=TABLE_TITLE[1])
    columns: List[Annotated[str, Field(min_length=1, max_length=TABLE_COLUMN_MAX)]] = Field(
        min_length=TABLE_COLUMNS[0], max_length=TABLE_COLUMNS[1])
    rows_hint: int = Field(4, ge=TABLE_ROWS_HINT[0], le=TABLE_ROWS_HINT[1])


class QuoteDirective(_Strict):
    kind: Literal["quote"]
    mode: Literal["pull"] = "pull"        # an unattributed key sentence; sourced quotes are decided by the lint


Visual = Annotated[Union[ImageDirective, TableDirective, QuoteDirective], Field(discriminator="kind")]


class Intro(_Strict):
    target_words: int = Field(ge=EDGE_WORDS[0], le=EDGE_WORDS[1])
    hook: str = Field(min_length=3, max_length=HOOK_MAX)


class Section(_Strict):
    id: str = Field(pattern=SECTION_ID)
    heading: str = Field(min_length=3, max_length=HEADING_MAX)
    angle: str = Field(min_length=ANGLE[0], max_length=ANGLE[1])
    target_words: int = Field(ge=SECTION_WORDS[0], le=SECTION_WORDS[1])
    visuals: List[Visual] = Field(default_factory=list, max_length=VISUALS_PER_SECTION_MAX)

    def of_kind(self, kind: str) -> list:
        return [v for v in self.visuals if v.kind == kind]


class Conclusion(_Strict):
    heading: str = Field(min_length=3, max_length=HEADING_MAX)
    target_words: int = Field(ge=EDGE_WORDS[0], le=EDGE_WORDS[1])
    takeaways: int = Field(3, ge=TAKEAWAYS[0], le=TAKEAWAYS[1])


class Outline(_Strict):
    title: str = Field(min_length=3, max_length=TITLE_MAX)
    subtitle: Optional[str] = Field(None, max_length=SUBTITLE_MAX)
    intro: Intro
    sections: List[Section] = Field(min_length=SECTIONS[0], max_length=SECTIONS[1])
    conclusion: Conclusion

    def count(self, kind: str) -> int:
        return sum(len(s.of_kind(kind)) for s in self.sections)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    """A finding on the model's output. Errors go back to the model once (repair); warnings reach the client."""
    kind: str
    message: str
    unit: Optional[str] = None
    field: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def format_errors(errors) -> str:
    """Numbered list for the repair prompt."""
    lines = []
    for i, e in enumerate(errors, 1):
        where = " ".join(x for x in (f"unit {e.unit}" if e.unit else "", e.field or "") if x)
        lines.append(f"{i}. [{e.kind}]{' (' + where + ')' if where else ''} {e.message}")
    return "\n".join(lines)


class BlogError(RuntimeError):
    """The post cannot be produced (the model's structure is unusable even after repair, or a section is empty)."""

    def __init__(self, message: str, errors: Optional[List[Issue]] = None):
        super().__init__(message)
        self.errors = errors or []
