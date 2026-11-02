"""Stateless preview/edit endpoints. They make no provider, storage, licence or billing calls."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import Field, model_validator

from .edit import apply_patch, patch_from_instruction
from .render import chart_view
from .schema import ChartDocument, ChartPatch, ChartView, StrictModel

router = APIRouter(prefix="/agent/charts", tags=["Editable graphs"])


class PreviewRequest(StrictModel):
    chart: ChartDocument


class EditRequest(PreviewRequest):
    patch: ChartPatch | None = None
    instruction: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def exactly_one_edit(self):
        if (self.patch is None) == (self.instruction is None):
            raise ValueError("Supply exactly one of patch or instruction")
        return self


@router.post("/preview", response_model=ChartView)
def preview(request: PreviewRequest) -> ChartView:
    return chart_view(request.chart)


@router.post("/edit", response_model=ChartView)
def edit(request: EditRequest) -> ChartView:
    try:
        patch = request.patch if request.patch is not None else patch_from_instruction(request.instruction)
        return chart_view(apply_patch(request.chart, patch))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
