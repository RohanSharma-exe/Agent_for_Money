import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_opportunity_agent.core.settings import Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_low_ram_and_local_first(self) -> None:
        settings = load_settings()

        self.assertEqual(settings.chat_model, "qwen3:4b-instruct")
        self.assertEqual(settings.embedding_model, "nomic-embed-text")
        self.assertEqual(settings.max_context_tokens, 4096)
        self.assertEqual(settings.concurrent_llm_requests, 1)
        self.assertEqual(settings.project_root.name, "local-opportunity-agent")
        self.assertEqual(settings.database_path, settings.project_root / "data" / "app.db")
        self.assertEqual(settings.qdrant_path, settings.project_root / "data" / "qdrant")
        self.assertEqual(settings.obsidian_path, settings.project_root / "obsidian")

    def test_environment_overrides_paths_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {
                "LOA_PROJECT_ROOT": temp_dir,
                "LOA_CHAT_MODEL": "qwen3:test",
                "LOA_EMBEDDING_MODEL": "embed:test",
                "LOA_MAX_CONTEXT_TOKENS": "2048",
            }

            with patch.dict(os.environ, env, clear=False):
                settings = load_settings()

        self.assertEqual(settings.project_root, Path(temp_dir))
        self.assertEqual(settings.chat_model, "qwen3:test")
        self.assertEqual(settings.embedding_model, "embed:test")
        self.assertEqual(settings.max_context_tokens, 2048)

    def test_invalid_resource_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(project_root=Path.cwd(), max_context_tokens=0)

        with self.assertRaises(ValueError):
            Settings(project_root=Path.cwd(), concurrent_llm_requests=2)


if __name__ == "__main__":
    unittest.main()
