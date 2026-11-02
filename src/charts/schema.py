"""The dataset is the source of truth; presentation changes never regenerate it."""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

ChartType = Literal["bar", "line", "pie"]
Number = Annotated[Decimal, Field(allow_inf_nan=False, max_digits=24, decimal_places=6,
                                  ge=Decimal("-1000000000000000"), le=Decimal("1000000000000000"))]
Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
PointId = Annotated[str, Field(pattern=r"^p[1-9][0-9]?$", max_length=3)]
Label = Annotated[str, Field(min_length=1, max_length=120)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    excerpt: str = Field(min_length=1, max_length=800)
    label: Label
    value_text: str = Field(min_length=1, max_length=80)


class ChartPoint(StrictModel):
    id: PointId
    label: Label
    value: Number
    source: SourceReference

    @field_serializer("value")
    def serialize_value(self, value: Decimal) -> str:
        # Decimal strings survive JSON and type/color edits without floating-point rounding.
        return format(value, "f")


class ChartData(StrictModel):
    unit: str = Field(default="", max_length=64)
    points: list[ChartPoint] = Field(min_length=2, max_length=24)

    @model_validator(mode="after")
    def unique_points(self):
        if len({p.id for p in self.points}) != len(self.points):
            raise ValueError("Point IDs must be unique")
        if len({p.label.casefold() for p in self.points}) != len(self.points):
            raise ValueError("Category labels must be unique")
        return self


class ChartPresentation(StrictModel):
    chart_type: ChartType = "bar"
    color: Color = "#2563EB"
    point_colors: dict[PointId, Color] = Field(default_factory=dict, max_length=24)


class ChartDocument(StrictModel):
    block_type: Literal["eddyter_chart"] = "eddyter_chart"
    version: Literal["1.0"] = "1.0"
    chart_id: str = Field(pattern=r"^chart_[a-f0-9]{16}$")
    title: str = Field(min_length=1, max_length=160)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data: ChartData
    presentation: ChartPresentation = Field(default_factory=ChartPresentation)

    @model_validator(mode="after")
    def validate_presentation(self):
        if set(self.presentation.point_colors) - {p.id for p in self.data.points}:
            raise ValueError("Point colors reference an unknown point ID")
        if self.presentation.chart_type == "pie":
            values = [p.value for p in self.data.points]
            if any(v < 0 for v in values) or sum(values) <= 0:
                raise ValueError("A pie graph needs non-negative values and a positive total; data was not changed")
        return self


class ChartPatch(StrictModel):
    """Only supplied fields change. Data edits are explicit and keyed by stable point ID."""
    chart_type: ChartType | None = None
    color: Color | None = None
    point_colors: dict[PointId, Color] | None = Field(default=None, max_length=24)
    values: dict[PointId, Number] | None = Field(default=None, max_length=24)
    labels: dict[PointId, Label] | None = Field(default=None, max_length=24)
    title: str | None = Field(default=None, min_length=1, max_length=160)
    unit: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def nonempty_patch(self):
        if not self.model_fields_set or any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError("Supply at least one edit; supplied fields cannot be null")
        if any(getattr(self, key) == {} for key in self.model_fields_set):
            raise ValueError("Point edits cannot be empty")
        return self


class ChartView(StrictModel):
    chart: ChartDocument
    preview_svg: str


class GraphSuggestion(ChartView):
    status: Literal["suggested"] = "suggested"
    reason: str = "Comparable values were found in your original content. Review before inserting."
