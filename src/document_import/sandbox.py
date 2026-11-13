"""
Run a converter on an untrusted file in a separate, locked-down process.

Opening a DOCX or PDF means running a large parser on a stranger's bytes. A bug there (a crash, a runaway loop, a memory bomb) should cost one
child process, not the API server that is also serving every other request. So the parent:

  * writes the file into a fresh private directory and starts `python -I worker.py` with an EMPTY environment (no API keys, no Redis URL, no cloud
    credentials), that directory as its working directory, no stdin, and its own session (so the whole process group can be killed);
  * sets limits inside the child before it parses anything: CPU seconds, address space (Linux), output file size, open files, no core dumps;
  * waits at most `timeout` seconds, then kills the group;
  * reads one JSON result file back (its size is bounded by the output limit) and removes the directory.

A child that is killed, crashes, or exceeds a limit gives the user "could not be read" (422) or "too complex" (413); it never gives a stack trace.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from typing import Optional

from .errors import ImportRefused
from .html_to_md import Converted

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
OUTPUT_LIMIT = 8 * 1024 * 1024                    # the JSON result; a 400 000-character document is about 1.2 MB once escaped


def run_converter(kind: str, data: bytes, *, timeout: float = 30.0, memory_bytes: int = 1536 * 1024 * 1024, max_chars: int = 400_000,
                  max_pages: int = 200, extra_env: Optional[dict] = None) -> Converted:
    """Convert `data` ("docx", "pdf" or "html") in a child process. Raises ImportRefused; never returns a partial result."""
    env = {
        "DOC_IMPORT_MEMORY_BYTES": str(memory_bytes),
        "DOC_IMPORT_CPU_SECONDS": str(max(2, int(timeout) + 2)),
        "DOC_IMPORT_OUTPUT_BYTES": str(OUTPUT_LIMIT),
        "DOC_IMPORT_MAX_CHARS": str(max_chars),
        "DOC_IMPORT_MAX_PAGES": str(max_pages),
        "DOC_IMPORT_TIME_BUDGET": str(max(2.0, timeout * 0.6)),         # the PDF reader stops early on its own before the kill
        "LANG": "C.UTF-8",
    }
    env.update(extra_env or {})
    with tempfile.TemporaryDirectory(prefix="docimport-") as workdir:
        os.chmod(workdir, 0o700)
        source, target = os.path.join(workdir, "input.bin"), os.path.join(workdir, "result.json")
        with open(source, "wb") as f:
            f.write(data)
        try:
            proc = subprocess.Popen([sys.executable, "-I", _WORKER, kind, source, target], cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        except OSError:
            raise ImportRefused(503, "import_unavailable", "Document import is not available right now. Try again later.",
                                headers={"Retry-After": "30"}) from None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            raise ImportRefused(422, "import_timeout", "This document took too long to read and was stopped. It may be too large or too complex.") from None
        finally:
            if proc.poll() is None:
                _kill(proc)
        try:
            if os.path.getsize(target) > OUTPUT_LIMIT:
                raise ValueError("oversized")
            with open(target, "r", encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, ValueError):
            code = proc.returncode
            if code is not None and code < 0 and -code in (signal.SIGXCPU, signal.SIGKILL, signal.SIGSEGV, signal.SIGABRT, signal.SIGXFSZ):
                raise ImportRefused(413, "too_complex", "This document is too large or too complex to read safely.") from None
            raise ImportRefused(422, "unreadable", "This document could not be read.") from None
    if not isinstance(result, dict):
        raise ImportRefused(422, "unreadable", "This document could not be read.")
    if not result.get("ok"):
        raise ImportRefused.from_dict(result.get("error") or {})
    return Converted(str(result.get("markdown", "")), title=str(result.get("title", "")), stats=dict(result.get("stats") or {}),
                     warnings=list(result.get("warnings") or []))


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
