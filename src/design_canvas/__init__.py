"""
src/design_canvas — Mode 8: Editable UI Design Engine.

Public API:
    execute_design_canvas_harness  – orchestrator (service.py)
    DesignDocumentAST              – root Pydantic model (schema.py)
    detect_design_intent           – intent gate lives in src/design_intent.py
"""

from .schema import DesignDocumentAST  # noqa: F401
from .service import execute_design_canvas_harness  # noqa: F401

__all__ = ["DesignDocumentAST", "execute_design_canvas_harness"]
