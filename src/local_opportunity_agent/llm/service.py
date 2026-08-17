from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ollama


class LLMError(RuntimeError):
    """Base error for local LLM operations."""


class OllamaUnavailableError(LLMError):
    """Raised when Ollama cannot be reached."""


class OllamaModelError(LLMError):
    """Raised when a configured model is unavailable."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float]
    dimensions: int


@dataclass(frozen=True)
class OllamaHealth:
    reachable: bool
    chat_model_available: bool
    embedding_model_available: bool


class OllamaService:
    """Small, centralized wrapper around the local Ollama runtime."""

    def __init__(
        self,
        *,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        context_tokens: int,
    ) -> None:
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.context_tokens = context_tokens

        self._client = ollama.Client(host=base_url)

    def list_models(self) -> list[str]:
        """Return locally available Ollama model names."""
        try:
            response = self._client.list()
        except Exception as error:
            raise OllamaUnavailableError(
                f"Unable to connect to Ollama at {self.base_url}: {error}"
            ) from error

        models = response.get("models", [])

        names: list[str] = []

        for model in models:
            name = model.get("model") or model.get("name")

            if name:
                names.append(str(name))

        return names

    def health(self) -> OllamaHealth:
        """Check Ollama connectivity and configured model availability."""
        try:
            models = self.list_models()
        except OllamaUnavailableError:
            return OllamaHealth(
                reachable=False,
                chat_model_available=False,
                embedding_model_available=False,
            )

        return OllamaHealth(
            reachable=True,
            chat_model_available=self._model_is_available(
                self.chat_model,
                models,
            ),
            embedding_model_available=self._model_is_available(
                self.embedding_model,
                models,
            ),
        )

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Generate a response using the configured local chat model."""
        self._ensure_model_available(self.chat_model)

        options: dict[str, int | float] = {
            "num_ctx": self.context_tokens,
        }

        if temperature is not None:
            options["temperature"] = temperature

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            response = self._client.chat(
                model=self.chat_model,
                messages=messages,
                stream=False,
                options=options,
            )
        except Exception as error:
            raise LLMError(f"Ollama chat request failed: {error}") from error

        message: dict[str, Any] = response.get("message", {})
        content = str(message.get("content", ""))

        prompt_tokens = int(response.get("prompt_eval_count") or 0)
        completion_tokens = int(response.get("eval_count") or 0)

        return ChatResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def embed(self, text: str) -> EmbeddingResult:
        """Generate an embedding using the configured local embedding model."""
        if not text.strip():
            raise ValueError("text must not be empty")

        self._ensure_model_available(self.embedding_model)

        try:
            response = self._client.embed(
                model=self.embedding_model,
                input=text,
            )
        except Exception as error:
            raise LLMError(f"Ollama embedding request failed: {error}") from error

        embeddings = response.get("embeddings", [])

        if not embeddings:
            raise LLMError("Ollama returned no embeddings")

        embedding = [float(value) for value in embeddings[0]]

        return EmbeddingResult(
            embedding=embedding,
            dimensions=len(embedding),
        )

    @staticmethod
    def _model_is_available(
        configured_model: str,
        available_models: list[str],
    ) -> bool:
        if configured_model in available_models:
            return True

        if f"{configured_model}:latest" in available_models:
            return True

        if configured_model.endswith(":latest"):
            base_model = configured_model.removesuffix(":latest")

            if base_model in available_models:
                return True

        return False

    def _ensure_model_available(self, model: str) -> None:
        models = self.list_models()

        if self._model_is_available(model, models):
            return

        raise OllamaModelError(
            f"Configured Ollama model '{model}' is not installed. "
            f"Available models: {', '.join(models) or 'none'}"
        )
