"""Design tests run against the code defaults, not this machine's .env (which turns the 1.1 features on)."""

import pytest

from config import Settings, settings

_DESIGN_SETTINGS = ("design_schema_v11_enabled", "design_vision_two_pass", "design_image_fill_enabled",
                    "design_vision_max_output_tokens", "design_max_fill_images", "design_max_fidelity_repairs",
                    "design_vision_openai_effort", "design_text_edit_enabled", "design_text_edit_timeout_seconds",
                    "design_section_edit_enabled", "design_section_edit_max_fraction")


@pytest.fixture(autouse=True)
def _design_settings_at_code_defaults(monkeypatch):
    for name in _DESIGN_SETTINGS:
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)
