from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from local_opportunity_agent.agents.supervisor import Supervisor
from local_opportunity_agent.api.routes import router
from local_opportunity_agent.core.settings import (
    ensure_runtime_directories,
    load_settings,
)
from local_opportunity_agent.llm import OllamaService
from local_opportunity_agent.memory import (
    Database,
    ObsidianMemory,
)
from local_opportunity_agent.memory.vector_store import VectorStore
from local_opportunity_agent.runtime.graph import build_graph
from local_opportunity_agent.tools.memory import MemorySearchTool
from local_opportunity_agent.tools.web.research import ResearchTool


def create_app() -> FastAPI:
    settings = load_settings()

    ensure_runtime_directories(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = Database(settings.database_path)
        database.initialize()

        obsidian = ObsidianMemory(settings.obsidian_path)
        obsidian.initialize()

        ollama_service = OllamaService(
            base_url=settings.ollama_base_url,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            context_tokens=settings.max_context_tokens,
        )

        vector_store = VectorStore(
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
            vector_size=settings.qdrant_vector_size,
        )

        try:
            vector_store.initialize()

            memory_search_tool = MemorySearchTool(
                llm=ollama_service,
                vector_store=vector_store,
            )

            research_tool = ResearchTool()

            supervisor = Supervisor(ollama_service)

            graph = build_graph(
                supervisor,
                memory_search_tool,
                research_tool,
            )

            app.state.database = database
            app.state.obsidian = obsidian
            app.state.ollama_service = ollama_service
            app.state.vector_store = vector_store
            app.state.memory_search_tool = memory_search_tool
            app.state.supervisor = supervisor
            app.state.graph = graph
            app.state.research_tool = research_tool

            yield

        finally:
            vector_store.close()

    app = FastAPI(
        title="Local Opportunity Agent",
        version="0.1.0",
        description="Local-first OpenAI-compatible API for opportunity research.",
        lifespan=lifespan,
    )

    app.state.settings = settings

    app.include_router(router)

    return app


app = create_app()
