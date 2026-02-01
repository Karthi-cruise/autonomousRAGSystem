"""FastAPI interface for Autonomous RAG System."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def create_app(system: dict[str, Any] | None = None):
    """Create FastAPI app with RAG endpoints."""
    from src.main import build_system

    app = FastAPI(
        title="Autonomous RAG System",
        description="Enterprise RAG with trust scoring, temporal decay, and hallucination detection",
        version="1.0.0",
    )

    _system = system or build_system()

    class QueryRequest(BaseModel):
        query: str

    class IngestRequest(BaseModel):
        content: str
        source: str = "api"
        trust_score: float = 0.5

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query")
    def run_query(req: QueryRequest):
        from src.orchestration.rag_graph import run_rag
        try:
            resp = run_rag(
                query=req.query,
                retriever=_system["retriever"],
                verifier=_system["verifier"],
                updater=_system["updater"],
            )
            return {
                "answer": resp.answer,
                "citations": resp.citations,
                "trust_explanations": resp.trust_explanations,
                "decay_explanations": resp.decay_explanations,
                "verification_explanation": resp.verification_explanation,
                "verdict": resp.verdict.value,
            }
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

    return app
