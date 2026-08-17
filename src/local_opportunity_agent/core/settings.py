from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    override = os.getenv("LOA_PROJECT_ROOT")

    if override:
        return Path(override).expanduser().resolve()

    return Path(__file__).resolve().parents[3]


def _settings_file(project_root: Path) -> Path:
    return project_root / "config" / "settings.toml"


def _load_toml(project_root: Path) -> dict[str, Any]:
    path = _settings_file(project_root)

    if not path.exists():
        return {}

    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML configuration: {path}") from error

    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a TOML table: {path}")

    return data


def _section(
    config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = config.get(name, {})

    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{name}' must be a table")

    return value


def _value(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    values = _section(config, section)
    return values.get(key, default)


def _env_or_toml(
    *,
    env_name: str,
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    environment_value = os.getenv(env_name)

    if environment_value is not None:
        return environment_value

    return _value(
        config,
        section,
        key,
        default,
    )


def _int_value(
    *,
    env_name: str,
    config: dict[str, Any],
    section: str,
    key: str,
    default: int,
) -> int:
    value = _env_or_toml(
        env_name=env_name,
        config=config,
        section=section,
        key=key,
        default=default,
    )

    if isinstance(value, bool):
        raise TypeError(f"{env_name} must be an integer")

    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{env_name} must be an integer") from error


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

    qdrant_collection: str = "opportunity_memory"
    qdrant_vector_size: int = 768

    def __post_init__(self) -> None:
        if self.max_context_tokens < 512:
            raise ValueError("max_context_tokens must be at least 512")

        if self.concurrent_llm_requests != 1:
            raise ValueError("concurrent_llm_requests must stay at 1 on 8 GB RAM")

        if not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection must not be empty")

        if self.qdrant_vector_size < 1:
            raise ValueError("qdrant_vector_size must be greater than 0")

    @property
    def config_path(self) -> Path:
        return self.project_root / "config" / "settings.toml"

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
    project_root = _project_root()
    config = _load_toml(project_root)

    return Settings(
        project_root=project_root,
        env=str(
            _env_or_toml(
                env_name="LOA_ENV",
                config=config,
                section="runtime",
                key="env",
                default="local",
            )
        ),
        log_level=str(
            _env_or_toml(
                env_name="LOA_LOG_LEVEL",
                config=config,
                section="runtime",
                key="log_level",
                default="INFO",
            )
        ),
        ollama_base_url=str(
            _env_or_toml(
                env_name="LOA_OLLAMA_BASE_URL",
                config=config,
                section="ollama",
                key="base_url",
                default="http://localhost:11434",
            )
        ),
        chat_model=str(
            _env_or_toml(
                env_name="LOA_CHAT_MODEL",
                config=config,
                section="ollama",
                key="chat_model",
                default="qwen3:4b-instruct",
            )
        ),
        embedding_model=str(
            _env_or_toml(
                env_name="LOA_EMBEDDING_MODEL",
                config=config,
                section="ollama",
                key="embedding_model",
                default="nomic-embed-text",
            )
        ),
        max_context_tokens=_int_value(
            env_name="LOA_MAX_CONTEXT_TOKENS",
            config=config,
            section="ollama",
            key="max_context_tokens",
            default=4096,
        ),
        concurrent_llm_requests=_int_value(
            env_name="LOA_CONCURRENT_LLM_REQUESTS",
            config=config,
            section="ollama",
            key="concurrent_llm_requests",
            default=1,
        ),
        qdrant_collection=str(
            _env_or_toml(
                env_name="LOA_QDRANT_COLLECTION",
                config=config,
                section="memory",
                key="qdrant_collection",
                default="opportunity_memory",
            )
        ),
        qdrant_vector_size=_int_value(
            env_name="LOA_QDRANT_VECTOR_SIZE",
            config=config,
            section="memory",
            key="qdrant_vector_size",
            default=768,
        ),
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
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
