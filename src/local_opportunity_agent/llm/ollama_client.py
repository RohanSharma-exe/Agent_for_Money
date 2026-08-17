from __future__ import annotations

from dataclasses import dataclass

import ollama


@dataclass(frozen=True)
class OllamaChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OllamaError(RuntimeError):
    """Raised when communication with Ollama fails."""


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        context_tokens: int,
    ) -> None:
        self.model = model
        self.context_tokens = context_tokens
        self.client = ollama.Client(host=base_url)

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> OllamaChatResult:
        options: dict[str, int | float] = {
            "num_ctx": self.context_tokens,
        }

        if temperature is not None:
            options["temperature"] = temperature

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False,
            )
        except Exception as error:
            raise OllamaError(
                f"Unable to communicate with Ollama at the configured endpoint: {error}"
            ) from error

        message = response.get("message", {})
        content = message.get("content", "")

        prompt_tokens = int(response.get("prompt_eval_count") or 0)
        completion_tokens = int(response.get("eval_count") or 0)

        return OllamaChatResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
