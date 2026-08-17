from __future__ import annotations

import uvicorn


def run_api() -> None:
    uvicorn.run(
        "local_opportunity_agent.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
    )
