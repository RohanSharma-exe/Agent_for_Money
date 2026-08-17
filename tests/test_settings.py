from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_opportunity_agent.core.settings import load_settings


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        self.project_root = Path(self.temp_dir.name)

        config_dir = self.project_root / "config"
        config_dir.mkdir()

        self.settings_file = config_dir / "settings.toml"

        self.settings_file.write_text(
            """
[runtime]
env = "test"
log_level = "DEBUG"

[ollama]
base_url = "http://toml-host:11434"
chat_model = "toml-chat"
embedding_model = "toml-embedding"
max_context_tokens = 2048
concurrent_llm_requests = 1

[memory]
qdrant_collection = "toml_memory"
qdrant_vector_size = 768
""".strip(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_defaults_are_low_ram_and_local_first(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOA_PROJECT_ROOT": str(self.project_root / "missing"),
            },
            clear=False,
        ):
            settings = load_settings()

        self.assertEqual(
            settings.env,
            "local",
        )
        self.assertEqual(
            settings.chat_model,
            "qwen3:4b-instruct",
        )
        self.assertEqual(
            settings.embedding_model,
            "nomic-embed-text",
        )
        self.assertEqual(
            settings.max_context_tokens,
            4096,
        )
        self.assertEqual(
            settings.concurrent_llm_requests,
            1,
        )
        self.assertEqual(
            settings.qdrant_collection,
            "opportunity_memory",
        )
        self.assertEqual(
            settings.qdrant_vector_size,
            768,
        )

    def test_toml_configuration_is_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOA_PROJECT_ROOT": str(self.project_root),
            },
            clear=False,
        ):
            settings = load_settings()

        self.assertEqual(
            settings.env,
            "test",
        )
        self.assertEqual(
            settings.log_level,
            "DEBUG",
        )
        self.assertEqual(
            settings.ollama_base_url,
            "http://toml-host:11434",
        )
        self.assertEqual(
            settings.chat_model,
            "toml-chat",
        )
        self.assertEqual(
            settings.embedding_model,
            "toml-embedding",
        )
        self.assertEqual(
            settings.max_context_tokens,
            2048,
        )
        self.assertEqual(
            settings.qdrant_collection,
            "toml_memory",
        )

    def test_environment_overrides_toml(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOA_PROJECT_ROOT": str(self.project_root),
                "LOA_CHAT_MODEL": "env-chat",
                "LOA_MAX_CONTEXT_TOKENS": "4096",
                "LOA_QDRANT_VECTOR_SIZE": "768",
            },
            clear=False,
        ):
            settings = load_settings()

        self.assertEqual(
            settings.chat_model,
            "env-chat",
        )
        self.assertEqual(
            settings.max_context_tokens,
            4096,
        )
        self.assertEqual(
            settings.qdrant_vector_size,
            768,
        )

        # Values not overridden by environment remain from TOML.
        self.assertEqual(
            settings.embedding_model,
            "toml-embedding",
        )
        self.assertEqual(
            settings.qdrant_collection,
            "toml_memory",
        )

    def test_environment_overrides_paths_and_models(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOA_PROJECT_ROOT": str(self.project_root),
                "LOA_OLLAMA_BASE_URL": "http://env-host:11434",
                "LOA_EMBEDDING_MODEL": "env-embedding",
            },
            clear=False,
        ):
            settings = load_settings()

        self.assertEqual(
            settings.ollama_base_url,
            "http://env-host:11434",
        )
        self.assertEqual(
            settings.embedding_model,
            "env-embedding",
        )
        self.assertEqual(
            settings.database_path,
            self.project_root / "data" / "app.db",
        )
        self.assertEqual(
            settings.qdrant_path,
            self.project_root / "data" / "qdrant",
        )

    def test_invalid_resource_settings_are_rejected(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "LOA_PROJECT_ROOT": str(self.project_root),
                    "LOA_MAX_CONTEXT_TOKENS": "256",
                },
                clear=False,
            ),
            self.assertRaises(ValueError),
        ):
            load_settings()

        with (
            patch.dict(
                os.environ,
                {
                    "LOA_PROJECT_ROOT": str(self.project_root),
                    "LOA_QDRANT_VECTOR_SIZE": "0",
                },
                clear=False,
            ),
            self.assertRaises(ValueError),
        ):
            load_settings()


if __name__ == "__main__":
    unittest.main()
