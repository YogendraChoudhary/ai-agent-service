"""Stateless, stale-safe application of accepted document patches; no provider calls."""
from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator

from .blocks import document_blocks, fingerprint
from .schema import DocumentPatch, StrictModel
from .service import apply_document_patches

router = APIRouter(prefix="/agent/documents", tags=["Document edits"])


class ApplyRequest(StrictModel):
    document_text: str = Field(min_length=1, max_length=100_000)
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    patches: list[DocumentPatch] = Field(max_length=12)

    @field_validator("document_text")
    @classmethod
    def valid_source(cls, value):
        document_blocks(value)
        return value


@router.post("/apply")
def apply(request: ApplyRequest):
    try:
        result = apply_document_patches(request.document_text, request.base_sha256, request.patches)
    except ValueError as error:
        raise HTTPException(status_code=409 if fingerprint(request.document_text) != request.base_sha256 else 400,
                            detail=str(error)) from error
    return {"document_text": result, "result_sha256": fingerprint(result)}
