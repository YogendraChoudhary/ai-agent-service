"""Every tunable the document-action features added lives in config.py: defaults, env overrides, bounds, and that the code reads them."""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.document_action as document_action
import src.document_edit.service as edit_service
import src.document_literal as document_literal
import src.intent_classifier as intent_classifier
from config import Settings, settings

DEFAULTS = {
    "DOCUMENT_ACTION_CLASSIFIER_ENABLED": ("document_action_classifier_enabled", "true", True),
    "DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS": ("document_action_classifier_timeout_seconds", "2.5", 2.5),
    "DOCUMENT_ACTION_CLASSIFIER_CONCURRENCY": ("document_action_classifier_concurrency", "8", 8),
    "DOCUMENT_ACTION_DIGEST_CHARS": ("document_action_digest_chars", "500", 500),
    "DOCUMENT_LITERAL_MAX_REQUEST_CHARS": ("document_literal_max_request_chars", "200", 200),
    "DOCUMENT_EDIT_SMALL_INSERTION_CHARS": ("document_edit_small_insertion_chars", "40", 40),
    "DOCUMENT_EDIT_SMALL_ADDITION_MAX_WORDS": ("document_edit_small_addition_max_words", "6", 6),
}
DOCUMENTED_DEFAULTS = {"document_action_classifier_enabled": True, "document_action_classifier_timeout_seconds": 4.0,
                       "document_action_classifier_concurrency": 16, "document_action_digest_chars": 900,
                       "document_literal_max_request_chars": 400, "document_edit_small_insertion_chars": 80,
                       "document_edit_small_addition_max_words": 12}


def fresh(monkeypatch, **env):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_defaults_are_the_measured_ones(monkeypatch):
    built = fresh(monkeypatch)
    assert {field: getattr(built, field) for field in DOCUMENTED_DEFAULTS} == DOCUMENTED_DEFAULTS


@pytest.mark.parametrize("variable, field, raw, value", [(v, *spec) for v, spec in DEFAULTS.items()])
def test_each_setting_can_be_overridden_from_the_environment(monkeypatch, variable, field, raw, value):
    assert getattr(fresh(monkeypatch, **{variable: raw}), field) == value


@pytest.mark.parametrize("variable, bad", [
    ("DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS", "0"), ("DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS", "99"),
    ("DOCUMENT_ACTION_CLASSIFIER_CONCURRENCY", "0"), ("DOCUMENT_ACTION_CLASSIFIER_CONCURRENCY", "500"),
    ("DOCUMENT_ACTION_DIGEST_CHARS", "100"), ("DOCUMENT_ACTION_DIGEST_CHARS", "2000"),     # the classifier call is cut at 1,000 characters
    ("DOCUMENT_LITERAL_MAX_REQUEST_CHARS", "10"), ("DOCUMENT_EDIT_SMALL_INSERTION_CHARS", "0"),
    ("DOCUMENT_EDIT_SMALL_ADDITION_MAX_WORDS", "0"), ("DOCUMENT_ACTION_CLASSIFIER_ENABLED", "maybe"),
    ("DOCUMENT_ACTION_CLASSIFIER_TIMEOUT_SECONDS", ""), ("DOCUMENT_ACTION_DIGEST_CHARS", ""),         # blank is rejected, not defaulted
])
def test_a_bad_or_blank_value_stops_start_up(monkeypatch, variable, bad):
    with pytest.raises(ValidationError):
        fresh(monkeypatch, **{variable: bad})


def test_the_code_reads_the_settings_instead_of_its_own_numbers(monkeypatch):
    monkeypatch.setattr(settings, "document_action_digest_chars", 300)
    assert len(document_action.digest("a" * 2000)) <= 300
    monkeypatch.setattr(settings, "document_literal_max_request_chars", 30)
    assert document_literal.literal_draft_text("add Acme Inc. below the Contact heading") is None       # 39 characters
    monkeypatch.setattr(settings, "document_literal_max_request_chars", 400)
    assert document_literal.literal_draft_text("add Acme Inc. below the Contact heading") == "Acme Inc."
    monkeypatch.setattr(settings, "document_edit_small_insertion_chars", 5)
    assert edit_service.is_small_insertion("Report\n", "Report (1)\n") and not edit_service.is_small_insertion("Report\n", "Report (2024)\n")
    monkeypatch.setattr(settings, "document_edit_small_addition_max_words", 2)
    assert edit_service._words_preserved("Annual Report", "Annual Report two words") and not edit_service._words_preserved("Annual Report", "Annual Report three more words")


def test_the_classifier_workers_and_slots_follow_the_setting():
    assert intent_classifier.LABEL_CONCURRENCY == settings.document_action_classifier_concurrency
    held = [intent_classifier._LABEL_SLOTS.acquire(timeout=2) for _ in range(intent_classifier.LABEL_CONCURRENCY)]
    try:
        assert all(held) and not intent_classifier._LABEL_SLOTS.acquire(blocking=False)
    finally:
        for _ in filter(None, held):
            intent_classifier._LABEL_SLOTS.release()


def test_the_numbers_are_not_defined_a_second_time_in_the_modules():
    assert not hasattr(document_action, "DIGEST_CHARS")
    assert not hasattr(document_literal, "MAX_LITERAL_REQUEST_CHARS")
    assert not hasattr(edit_service, "SMALL_INSERTION_CHARS")


def test_env_example_lists_every_setting_commented_out():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    for variable in DEFAULTS:
        assert re.search(rf"^# {variable}=\S+$", text, re.M), variable            # commented, with its default
        assert not re.search(rf"^{variable}=\s*$", text, re.M), variable          # never a blank assignment: that stops start-up


def test_the_handoff_documents_every_setting_with_its_default():
    text = (Path(__file__).resolve().parents[1] / "docs" / "DOCUMENT_CONTEXT_API_HANDOFF.md").read_text()
    for variable, (field, _, _) in DEFAULTS.items():
        row = next((line for line in text.splitlines() if line.startswith(f"| `{variable}`")), None)
        assert row is not None, variable
        assert f"`{str(DOCUMENTED_DEFAULTS[field]).lower()}`" in row.lower(), (variable, row)
