"""One exception for every "this file/page cannot be imported" outcome, carrying the HTTP status and a message a person can act on."""
from __future__ import annotations


class ImportRefused(Exception):
    def __init__(self, status: int, code: str, message: str, *, headers: "dict[str, str] | None" = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def as_dict(self) -> dict:
        return {"status": self.status, "code": self.code, "message": self.message, "headers": self.headers}

    @classmethod
    def from_dict(cls, d: dict) -> "ImportRefused":
        return cls(int(d.get("status", 422)), str(d.get("code", "import_failed")), str(d.get("message", "The document could not be imported.")),
                   headers=d.get("headers") or {})
