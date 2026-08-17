from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from local_opportunity_agent.core.settings import Settings
from local_opportunity_agent.llm import (
    LLMError,
    OllamaModelError,
    OllamaService,
    OllamaUnavailableError,
)

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local-opportunity-agent"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _ollama(request: Request) -> OllamaService:
    return request.app.state.ollama_service


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    service = _ollama(request)
    health_status = service.health()

    return {
        "status": (
            "ok"
            if health_status.reachable
            and health_status.chat_model_available
            and health_status.embedding_model_available
            else "degraded"
        ),
        "ollama": {
            "reachable": health_status.reachable,
            "chat_model_available": health_status.chat_model_available,
            "embedding_model_available": health_status.embedding_model_available,
        },
    }


@router.get("/v1/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    settings = _settings(request)

    return ModelsResponse(
        data=[
            ModelInfo(
                id=settings.chat_model,
                created=int(time.time()),
            )
        ]
    )


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse | JSONResponse:
    settings = _settings(request)
    service = _ollama(request)

    model = payload.model or settings.chat_model

    if model != settings.chat_model:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Model '{model}' is not configured.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    if payload.stream:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Streaming is not implemented yet.",
                    "type": "not_implemented",
                    "code": "streaming_not_implemented",
                }
            },
        )

    try:
        result = service.chat(
            messages=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in payload.messages
            ],
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except OllamaUnavailableError as error:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": str(error),
                    "type": "service_unavailable",
                    "code": "ollama_unavailable",
                }
            },
        )
    except OllamaModelError as error:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": str(error),
                    "type": "model_unavailable",
                    "code": "ollama_model_unavailable",
                }
            },
        )
    except LLMError as error:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(error),
                    "type": "server_error",
                    "code": "llm_error",
                }
            },
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.content,
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
    )
