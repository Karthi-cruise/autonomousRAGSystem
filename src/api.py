"""FastAPI interface for Autonomous RAG System."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class IngestRequest(BaseModel):
    content: str = Field(min_length=1)
    source: str = "api"
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)


def create_app(system: dict[str, Any] | None = None):
    """Create FastAPI app with RAG endpoints."""
    from src.main import build_system, query as run_system_query

    app = FastAPI(
        title="Autonomous RAG System",
        description="Enterprise RAG with trust scoring, temporal decay, and hallucination detection",
        version="1.0.0",
    )

    _system = system or build_system()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        from src.utils.config import get_project_root

        return FileResponse(get_project_root() / "static" / "index.html")

    @app.post("/query")
    def run_query(req: QueryRequest):
        try:
            resp = run_system_query(_system, req.query)
            return resp.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        try:
            v = _system["updater"].ingest_documents(
                documents=[req.content],
                source=req.source,
                trust_score=req.trust_score,
            )
            return {"version": v, "status": "ingested"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/versions")
    def list_versions():
        return {"versions": _system["updater"].list_versions()}

    @app.get("/tools")
    def list_tools():
        tool_manager = _system.get("tools")
        return {"tools": tool_manager.describe() if tool_manager else []}

    return app
