from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from local_opportunity_agent.llm import (
    OllamaModelError,
    OllamaService,
)


class OllamaServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = OllamaService(
            base_url="http://localhost:11434",
            chat_model="qwen3:4b-instruct",
            embedding_model="nomic-embed-text",
            context_tokens=4096,
        )

    def test_list_models_returns_model_names(self) -> None:
        self.service._client = MagicMock()

        self.service._client.list.return_value = {
            "models": [
                {"model": "qwen3:4b-instruct"},
                {"model": "nomic-embed-text"},
            ]
        }

        models = self.service.list_models()

        self.assertEqual(
            models,
            [
                "qwen3:4b-instruct",
                "nomic-embed-text",
            ],
        )

    def test_health_reports_available_models(self) -> None:
        self.service._client = MagicMock()

        self.service._client.list.return_value = {
            "models": [
                {"model": "qwen3:4b-instruct"},
                {"model": "nomic-embed-text"},
            ]
        }

        health = self.service.health()

        self.assertTrue(health.reachable)
        self.assertTrue(health.chat_model_available)
        self.assertTrue(health.embedding_model_available)

    def test_chat_returns_normalized_result(self) -> None:
        self.service._client = MagicMock()

        self.service._client.list.return_value = {
            "models": [
                {"model": "qwen3:4b-instruct"},
            ]
        }

        self.service._client.chat.return_value = {
            "message": {
                "content": "Hello from Qwen.",
            },
            "prompt_eval_count": 12,
            "eval_count": 7,
        }

        result = self.service.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )

        self.assertEqual(result.content, "Hello from Qwen.")
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(result.total_tokens, 19)

    def test_embed_returns_vector(self) -> None:
        self.service._client = MagicMock()

        self.service._client.list.return_value = {
            "models": [
                {"model": "nomic-embed-text"},
            ]
        }

        self.service._client.embed.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4],
            ]
        }

        result = self.service.embed("test document")

        self.assertEqual(
            result.embedding,
            [0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(result.dimensions, 4)

    def test_empty_embedding_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.service.embed("")

    def test_missing_model_raises_model_error(self) -> None:
        self.service._client = MagicMock()

        self.service._client.list.return_value = {
            "models": [],
        }

        with self.assertRaises(OllamaModelError):
            self.service.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
