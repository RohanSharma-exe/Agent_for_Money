# Local Opportunity Agent

Local-first AI opportunity research agent for Windows machines with 8 GB RAM and no dedicated GPU.

## Phase 1 Status

This project currently contains the scaffold, package boundaries, and settings system. Later phases add FastAPI, Ollama, SQLite, Qdrant Edge, Obsidian memory, LangGraph, Crawl4AI, and Open WebUI connection instructions.

## Resource Rules

- Use one local LLM request at a time.
- Use `qwen3:4b-instruct` before larger models.
- Keep Ollama as the main local LLM runtime.
- Run Crawl4AI browsers only on demand.
- Keep Open WebUI separate from the core agent runtime.
- Store local runtime data outside the Python source package.

## Phase 1 Commands

```powershell
uv init C:/Python/local-opportunity-agent --package --python 3.12 --no-readme
uv run python -m unittest tests/test_settings.py
```
