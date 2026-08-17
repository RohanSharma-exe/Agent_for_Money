from __future__ import annotations

from fastapi import FastAPI

from local_opportunity_agent.api.routes import router
from local_opportunity_agent.core.settings import (
    ensure_runtime_directories,
    load_settings,
)
from local_opportunity_agent.llm import OllamaService
from local_opportunity_agent.memory.database import Database


def create_app() -> FastAPI:
    settings = load_settings()

    ensure_runtime_directories(settings)

    database = Database(settings.database_path)
    database.initialize()

    app = FastAPI(
        title="Local Opportunity Agent",
        version="0.1.0",
        description="Local-first OpenAI-compatible API for opportunity research.",
    )

    app.state.settings = settings
    app.state.database = database

    app.state.ollama_service = OllamaService(
        base_url=settings.ollama_base_url,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        context_tokens=settings.max_context_tokens,
    )

    app.include_router(router)

    return app


app = create_app()
