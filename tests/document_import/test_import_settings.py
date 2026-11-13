"""Every tunable the import feature added lives in config.py: defaults, env overrides, bounds, that the code reads them, and that the docs list them."""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.document_import.router as import_router
from config import Settings, settings

DEFAULTS = {
    "DOCUMENT_IMPORT_ENABLED": ("document_import_enabled", "false", False),
    "DOCUMENT_IMPORT_URLS_ENABLED": ("document_import_urls_enabled", "false", False),
    "DOCUMENT_IMPORT_MAX_CONCURRENT": ("document_import_max_concurrent", "4", 4),
    "DOCUMENT_IMPORT_FETCH_CONCURRENT": ("document_import_fetch_concurrent", "8", 8),
    "DOCUMENT_IMPORT_MAX_FILE_BYTES": ("document_import_max_file_bytes", "5000000", 5_000_000),
    "DOCUMENT_IMPORT_TIMEOUT_SECONDS": ("document_import_timeout_seconds", "12.5", 12.5),
    "DOCUMENT_IMPORT_MAX_PDF_PAGES": ("document_import_max_pdf_pages", "50", 50),
    "DOCUMENT_IMPORT_MEMORY_MB": ("document_import_memory_mb", "512", 512),
}
DOCUMENTED = {"document_import_enabled": True, "document_import_urls_enabled": True, "document_import_max_concurrent": 2, "document_import_fetch_concurrent": 4,
              "document_import_max_file_bytes": 15_000_000, "document_import_timeout_seconds": 30.0, "document_import_max_pdf_pages": 200,
              "document_import_memory_mb": 1536}
ROOT = Path(__file__).resolve().parents[2]


def fresh(monkeypatch, **env):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_defaults(monkeypatch):
    built = fresh(monkeypatch)
    assert {field: getattr(built, field) for field in DOCUMENTED} == DOCUMENTED


@pytest.mark.parametrize("variable, field, raw, value", [(v, *spec) for v, spec in DEFAULTS.items()])
def test_each_setting_can_be_overridden(monkeypatch, variable, field, raw, value):
    assert getattr(fresh(monkeypatch, **{variable: raw}), field) == value


@pytest.mark.parametrize("variable, bad", [
    ("DOCUMENT_IMPORT_MAX_CONCURRENT", "0"), ("DOCUMENT_IMPORT_MAX_CONCURRENT", "9"), ("DOCUMENT_IMPORT_FETCH_CONCURRENT", "0"),
    ("DOCUMENT_IMPORT_FETCH_CONCURRENT", "17"), ("DOCUMENT_IMPORT_MAX_FILE_BYTES", "10"), ("DOCUMENT_IMPORT_MAX_FILE_BYTES", "999999999"),
    ("DOCUMENT_IMPORT_TIMEOUT_SECONDS", "1"), ("DOCUMENT_IMPORT_TIMEOUT_SECONDS", "999"), ("DOCUMENT_IMPORT_MAX_PDF_PAGES", "0"),
    ("DOCUMENT_IMPORT_MAX_PDF_PAGES", "5000"), ("DOCUMENT_IMPORT_MEMORY_MB", "64"), ("DOCUMENT_IMPORT_ENABLED", "maybe"),
    ("DOCUMENT_IMPORT_MAX_CONCURRENT", ""), ("DOCUMENT_IMPORT_TIMEOUT_SECONDS", ""),                       # blank is rejected, not defaulted
])
def test_a_bad_or_blank_value_stops_start_up(monkeypatch, variable, bad):
    with pytest.raises(ValidationError):
        fresh(monkeypatch, **{variable: bad})


def test_the_router_reads_the_settings():
    assert import_router._CONVERT_POOL._max_workers == settings.document_import_max_concurrent
    assert import_router._FETCH_POOL._max_workers == settings.document_import_fetch_concurrent
    assert import_router._MAX_FILE_BYTES == settings.document_import_max_file_bytes


def test_the_conversion_gets_the_configured_limits(monkeypatch):
    seen = {}

    class Done:
        markdown, title, stats, warnings = "# T\n\ntext\n", "T", {}, []

    def fake(kind, raw, **kw):
        seen.update(kind=kind, **kw)
        return Done
    monkeypatch.setattr(import_router, "run_converter", fake)
    monkeypatch.setattr(settings, "document_import_timeout_seconds", 12.0)
    monkeypatch.setattr(settings, "document_import_memory_mb", 700)
    monkeypatch.setattr(settings, "document_import_max_pdf_pages", 33)
    import_router._convert("pdf", b"%PDF-", "a.pdf")
    assert seen["timeout"] == 12.0 and seen["memory_bytes"] == 700 * 1024 * 1024 and seen["max_pages"] == 33


def test_env_example_lists_every_setting_commented_out():
    text = (ROOT / ".env.example").read_text()
    for variable in DEFAULTS:
        assert re.search(rf"^# {variable}=\S+$", text, re.M), variable
        assert not re.search(rf"^{variable}=\s*$", text, re.M), variable


def test_the_handoff_documents_every_setting_with_its_default():
    text = (ROOT / "docs" / "DOCUMENT_IMPORT_API_HANDOFF.md").read_text()
    for variable, (field, _, _) in DEFAULTS.items():
        row = next((line for line in text.splitlines() if line.startswith(f"| `{variable}`")), None)
        assert row is not None, variable
        assert f"`{str(DOCUMENTED[field]).lower()}`" in row.lower(), (variable, row)
