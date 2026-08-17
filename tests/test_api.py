from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from local_opportunity_agent.api.app import create_app
from local_opportunity_agent.llm.ollama_client import OllamaChatResult


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_models_endpoint_returns_configured_chat_model(self) -> None:
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["object"], "list")
        self.assertEqual(
            payload["data"][0]["id"],
            "qwen3:4b-instruct",
        )
        self.assertEqual(
            payload["data"][0]["object"],
            "model",
        )
        self.assertEqual(
            payload["data"][0]["owned_by"],
            "local-opportunity-agent",
        )

    @patch(
        "local_opportunity_agent.api.routes.OllamaService.chat",
        return_value=OllamaChatResult(
            content="Hello from the local model.",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )
    def test_chat_completions_returns_ollama_response(
        self,
        mock_chat,
    ) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3:4b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ],
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(
            payload["object"],
            "chat.completion",
        )
        self.assertEqual(
            payload["model"],
            "qwen3:4b-instruct",
        )
        self.assertEqual(
            payload["choices"][0]["index"],
            0,
        )
        self.assertEqual(
            payload["choices"][0]["finish_reason"],
            "stop",
        )
        self.assertEqual(
            payload["choices"][0]["message"]["role"],
            "assistant",
        )
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "Hello from the local model.",
        )

        self.assertEqual(
            payload["usage"]["prompt_tokens"],
            10,
        )
        self.assertEqual(
            payload["usage"]["completion_tokens"],
            5,
        )
        self.assertEqual(
            payload["usage"]["total_tokens"],
            15,
        )

        mock_chat.assert_called_once()

    def test_chat_completions_rejects_empty_messages(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3:4b-instruct",
                "messages": [],
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_completions_rejects_unknown_model(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "some-other-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 404)

    def test_chat_completions_rejects_streaming_for_now(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3:4b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 501)

    def test_health_reports_ollama_state(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertIn(payload["status"], {"ok", "degraded"})
        self.assertIn("ollama", payload)

        self.assertIn(
            "reachable",
            payload["ollama"],
        )
        self.assertIn(
            "chat_model_available",
            payload["ollama"],
        )
        self.assertIn(
            "embedding_model_available",
            payload["ollama"],
        )


if __name__ == "__main__":
    unittest.main()
