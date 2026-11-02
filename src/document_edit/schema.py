from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(StrictModel):
    block_id: str = Field(pattern=r"^b[0-9]+_[a-f0-9]{12}$")
    request_quote: str = Field(min_length=3, max_length=2000)
    reason: str = Field(min_length=1, max_length=300)


class EditPlan(StrictModel):
    status: Literal["ready", "needs_clarification", "no_change"]
    message: str = Field(min_length=1, max_length=500)
    targets: list[Target] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def coherent(self):
        if (self.status == "ready") != bool(self.targets):
            raise ValueError("Only a ready plan can contain targets, and a ready plan needs targets")
        if len({t.block_id for t in self.targets}) != len(self.targets):
            raise ValueError("A block can only be targeted once")
        return self


class DocumentPatch(StrictModel):
    block_id: str = Field(pattern=r"^b[0-9]+_[a-f0-9]{12}$")
    start_utf16: int = Field(ge=0, le=200_000)
    end_utf16: int = Field(gt=0, le=200_000)
    before: str = Field(min_length=1, max_length=12_000)
    after: str = Field(min_length=1, max_length=24_000)
    reason: str = Field(min_length=1, max_length=300)


class DocumentEditResult(StrictModel):
    status: Literal["proposed", "needs_clarification", "no_change", "failed"]
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    message: str = Field(min_length=1, max_length=500)
    targets: list[Target] = Field(default_factory=list, max_length=12)
    patches: list[DocumentPatch] = Field(default_factory=list, max_length=12)
