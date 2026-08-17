from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    override = os.getenv("LOA_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    return Path(__file__).resolve().parents[3]


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


@dataclass(frozen=True)
class Settings:
    project_root: Path
    env: str = "local"
    log_level: str = "INFO"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:4b-instruct"
    embedding_model: str = "nomic-embed-text"
    max_context_tokens: int = 4096
    concurrent_llm_requests: int = 1

    def __post_init__(self) -> None:
        if self.max_context_tokens < 512:
            raise ValueError("max_context_tokens must be at least 512")
        if self.concurrent_llm_requests != 1:
            raise ValueError("concurrent_llm_requests must stay at 1 on 8 GB RAM")

    @property
    def data_path(self) -> Path:
        return self.project_root / "data"

    @property
    def database_path(self) -> Path:
        return self.data_path / "app.db"

    @property
    def qdrant_path(self) -> Path:
        return self.data_path / "qdrant"

    @property
    def raw_data_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def obsidian_path(self) -> Path:
        return self.project_root / "obsidian"


def load_settings() -> Settings:
    return Settings(
        project_root=_project_root(),
        env=os.getenv("LOA_ENV", "local"),
        log_level=os.getenv("LOA_LOG_LEVEL", "INFO"),
        ollama_base_url=os.getenv("LOA_OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=os.getenv("LOA_CHAT_MODEL", "qwen3:4b-instruct"),
        embedding_model=os.getenv("LOA_EMBEDDING_MODEL", "nomic-embed-text"),
        max_context_tokens=_int_from_env("LOA_MAX_CONTEXT_TOKENS", 4096),
        concurrent_llm_requests=_int_from_env("LOA_CONCURRENT_LLM_REQUESTS", 1),
    )


def ensure_runtime_directories(settings: Settings) -> None:
    directories = [
        settings.data_path,
        settings.qdrant_path,
        settings.raw_data_path,
        settings.obsidian_path / "opportunities",
        settings.obsidian_path / "research",
        settings.obsidian_path / "companies",
        settings.obsidian_path / "people",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
