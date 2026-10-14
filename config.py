"""
Centralised application configuration via pydantic-settings.

Every environment variable the API needs is declared here with:
  - a type  (bool, float, int, str | None ...) so coercion happens at startup
  - a default so the app degrades gracefully where possible
  - a docstring comment explaining what the variable does

Import the singleton `settings` object everywhere instead of calling
os.getenv() directly.
"""

from __future__ import annotations

from typing import Optional

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Populate os.environ so all modules using os.getenv() get values regardless of cwd
_dotenv_path = find_dotenv()
if _dotenv_path:
    load_dotenv(_dotenv_path)


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")

    # ------------------------------------------------------------------
    # Backend webhooks
    # ------------------------------------------------------------------
    backend_webhook_url: str = Field("", alias="BACKEND_WEBHOOK_URL")
    webhook_secret: str = Field("", alias="WEBHOOK_SECRET")

    # ------------------------------------------------------------------
    # CDN / S3
    # ------------------------------------------------------------------
    cdn_url: str = Field("", alias="CDN_URL")
    aws_s3_bucket: str = Field("", alias="AWS_S3_BUCKET")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")

    # ------------------------------------------------------------------
    # Default (platform-funded) AI API keys
    # ------------------------------------------------------------------
    default_openai_api_key: Optional[str] = Field(None, alias="DEFAULT_OPENAI_API_KEY")
    default_anthropic_api_key: Optional[str] = Field(None, alias="DEFAULT_ANTHROPIC_API_KEY")
    default_grok_api_key: Optional[str] = Field(None, alias="DEFAULT_GROK_API_KEY")
    default_gemini_api_key: Optional[str] = Field(None, alias="DEFAULT_GEMINI_API_KEY")
    default_deepseek_api_key: Optional[str] = Field(None, alias="DEFAULT_DEEPSEEK_API_KEY")

    # ------------------------------------------------------------------
    # Credit system
    # ------------------------------------------------------------------
    credit_precheck_enabled: bool = Field(True, alias="CREDIT_PRECHECK_ENABLED")
    credit_settle_delay_seconds: float = Field(3.0, alias="CREDIT_SETTLE_DELAY_SECONDS")
    max_reservation_per_request: int = Field(100, alias="MAX_RESERVATION_PER_REQUEST")

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    max_reference_images: int = Field(3, alias="MAX_REFERENCE_IMAGES")

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------
    api_key_encryption_secret: str = Field(
        "default-32-byte-encryption-key!!",
        alias="API_KEY_ENCRYPTION_SECRET",
    )

    model_config = {
        "env_file": (".env", "../.env"),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


# Singleton — import `settings` everywhere; never call os.getenv() directly.
settings = Settings()
