"""Loads config/models.yaml and .env into typed settings objects.

api_key fields in the YAML may be either a literal string or an ${ENV_VAR}
placeholder resolved from the environment (populated from .env via
python-dotenv). Literal values always win over unresolved placeholders.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "models.yaml"
ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        match = ENV_VAR_PATTERN.match(value.strip())
        if match:
            return os.environ.get(match.group(1), "")
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


class AnthropicModels(BaseModel):
    intake: str
    eligibility_reasoning: str
    document_guidance: str
    simplification_fallback: str
    chat: str = "claude-sonnet-5"


class AnthropicConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.anthropic.com"
    max_retries: int = 3
    timeout_seconds: int = 60
    models: AnthropicModels


class HFModels(BaseModel):
    simplification: str
    embeddings: str
    speech_to_text: str


class HuggingFaceConfig(BaseModel):
    api_key: str = ""
    inference_api_url: str = "https://router.huggingface.co"
    max_retries: int = 3
    timeout_seconds: int = 60
    models: HFModels


class LLMProvidersConfig(BaseModel):
    anthropic: AnthropicConfig
    huggingface: HuggingFaceConfig


class SearchConfig(BaseModel):
    provider: str = "duckduckgo"
    max_results: int = 5
    tavily_api_key: str = ""
    serpapi_api_key: str = ""


class AppConfig(BaseModel):
    default_language: str = "en"
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])
    vector_store_path: str = "./data/vectorstore"
    seed_data_path: str = "./data/schemes/seed_schemes.json"
    request_timeout_seconds: int = 30
    max_schemes_returned: int = 8

    @property
    def vector_store_dir(self) -> Path:
        p = Path(self.vector_store_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def seed_data_file(self) -> Path:
        p = Path(self.seed_data_path)
        return p if p.is_absolute() else ROOT_DIR / p


class ModelSettings(BaseModel):
    """Typed view of config/models.yaml."""

    llm_providers: LLMProvidersConfig
    search: SearchConfig
    app: AppConfig

    @property
    def anthropic(self) -> AnthropicConfig:
        return self.llm_providers.anthropic

    @property
    def huggingface(self) -> HuggingFaceConfig:
        return self.llm_providers.huggingface


class ServerSettings(BaseSettings):
    """Server/process-level settings sourced directly from the environment."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"), extra="ignore", env_file_encoding="utf-8"
    )

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:8501"
    backend_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def load_model_settings(config_path: Path | str | None = None) -> ModelSettings:
    load_dotenv(ROOT_DIR / ".env", override=False)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env_placeholders(raw)
    return ModelSettings.model_validate(resolved)


@lru_cache(maxsize=1)
def get_model_settings() -> ModelSettings:
    return load_model_settings()


@lru_cache(maxsize=1)
def get_server_settings() -> ServerSettings:
    load_dotenv(ROOT_DIR / ".env", override=False)
    return ServerSettings()
