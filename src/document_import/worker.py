"""
The isolated process that opens an untrusted file.

Run by sandbox.py as `python -I worker.py <kind> <input file> <output file>` with an empty environment, a private working directory, resource
limits, and no stdin. It converts one DOCX or PDF and writes ONE JSON document to the output file: either {"ok": true, ...} or {"ok": false,
"error": {...}}. It reads nothing else and writes nothing else, so a parser bug triggered by a hostile file can do no more than make this process
crash or run out of its limits, which the parent treats as "could not be read".

Not importable for any other purpose: it has side effects at import time only through main().
"""
from __future__ import annotations

import json
import os
import sys


def _limits() -> None:
    """Best effort: not every limit exists on every OS (macOS has no working RLIMIT_AS), so each is set on its own."""
    try:
        import resource
    except ImportError:
        return
    memory = int(os.environ.get("DOC_IMPORT_MEMORY_BYTES", "0") or 0)
    cpu = int(os.environ.get("DOC_IMPORT_CPU_SECONDS", "0") or 0)
    output = int(os.environ.get("DOC_IMPORT_OUTPUT_BYTES", "0") or 0)
    for name, value in (("RLIMIT_AS", memory), ("RLIMIT_CPU", cpu), ("RLIMIT_FSIZE", output), ("RLIMIT_NOFILE", 64), ("RLIMIT_CORE", 0)):
        if value or name == "RLIMIT_CORE":
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
            except (ValueError, OSError, AttributeError):
                pass


def main(argv: list) -> int:
    kind, source, target = argv[1], argv[2], argv[3]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, root)                                           # -I removes the script directory from sys.path; the package is imported by name
    _limits()
    from src.document_import.errors import ImportRefused

    result: dict
    try:
        with open(source, "rb") as f:
            data = f.read()
        if kind == "docx":
            from src.document_import.docx_conv import docx_to_markdown
            converted = docx_to_markdown(data, max_chars=int(os.environ.get("DOC_IMPORT_MAX_CHARS", "400000")))
        elif kind == "pdf":
            from src.document_import.pdf_conv import pdf_to_markdown
            converted = pdf_to_markdown(data, max_pages=int(os.environ.get("DOC_IMPORT_MAX_PAGES", "200")),
                                        time_budget=float(os.environ.get("DOC_IMPORT_TIME_BUDGET", "20")),
                                        max_chars=int(os.environ.get("DOC_IMPORT_MAX_CHARS", "400000")))
        elif kind == "html":
            from src.document_import.html_to_md import html_to_markdown
            converted = html_to_markdown(data, base_url=os.environ.get("DOC_IMPORT_BASE_URL", ""), charset=os.environ.get("DOC_IMPORT_CHARSET") or None,
                                         max_chars=int(os.environ.get("DOC_IMPORT_MAX_CHARS", "400000")))
        else:
            raise ImportRefused(500, "internal", "Unknown document kind.")
        result = {"ok": True, "markdown": converted.markdown, "title": converted.title, "stats": converted.stats, "warnings": converted.warnings}
    except ImportRefused as exc:
        result = {"ok": False, "error": exc.as_dict()}
    except MemoryError:
        result = {"ok": False, "error": {"status": 413, "code": "too_complex", "message": "This document needs more memory than the import allows."}}
    except BaseException as exc:                                       # includes RecursionError: nothing a bad file does may leave without an answer
        result = {"ok": False, "error": {"status": 422, "code": "unreadable", "message": "This document could not be read.",
                                         "detail": type(exc).__name__}}
    with open(target, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
