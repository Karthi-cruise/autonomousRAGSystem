"""Main entry point for Autonomous RAG System."""

from __future__ import annotations

from pathlib import Path

from src.retrieval.vector_store import VersionedVectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.scoring.trust_scoring import TrustScorer
from src.scoring.decay_model import DecayModel
from src.agents.retriever_agent import RetrieverAgent
from src.agents.verifier_agent import VerifierAgent
from src.agents.updater_agent import UpdaterAgent
from src.orchestration.rag_graph import run_rag
from src.tools.manager import ToolManager
from src.tools.rest_tool import RestTool
from src.tools.sql_tool import SQLTool
from src.utils.config import load_config, get_project_root
from src.utils.document_loader import load_documents_from_dir


def build_system(config: dict | None = None):
    """Build the full RAG system from config."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    cfg = config or load_config()
    root = get_project_root()

    # Use project-local cache for HuggingFace models (avoids permission issues)
    cache_dir = root / "data" / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    embedding_cfg = cfg.get("embedding", {})
    retrieval_cfg = cfg.get("retrieval", {})
    trust_cfg = cfg.get("trust_scoring", {})
    decay_cfg = cfg.get("decay_model", {})
    vs_cfg = cfg.get("vector_store", {})
    llm_cfg = cfg.get("llm", {})
    tools_cfg = cfg.get("tools", {})

    base_path = root / vs_cfg.get("path", "data/processed/faiss_index")

    vector_store = VersionedVectorStore(
        embedding_model=embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        base_path=base_path,
        version_prefix=vs_cfg.get("version_prefix", "v"),
        cache_dir=cache_dir,
    )
    vector_store.deduplicate(create_new_version=False)

    trust_scorer = TrustScorer(
        default_trust=trust_cfg.get("default_trust", 0.5),
        trusted_domains=trust_cfg.get("trusted_domains", []),
        manual_overrides=trust_cfg.get("manual_overrides", {}),
    )

    decay_model = DecayModel(
        half_life_days=decay_cfg.get("half_life_days", 180),
        min_decay_factor=decay_cfg.get("min_decay_factor", 0.1),
    )

    hybrid_search = HybridSearch(
        vector_store=vector_store,
        trust_scorer=trust_scorer,
        decay_model=decay_model,
        faiss_weight=retrieval_cfg.get("faiss_hybrid_weight", 0.6),
        bm25_weight=retrieval_cfg.get("bm25_weight", 0.4),
        top_k=retrieval_cfg.get("top_k", 5),
    )

    tools = []
    sql_cfg = tools_cfg.get("sql", {})
    if sql_cfg.get("enabled"):
        db_path = Path(sql_cfg.get("database_path", root / "data" / "metadata" / "knowledge.db"))
        if not db_path.is_absolute():
            db_path = root / db_path
        tools.append(SQLTool(
            database_path=db_path,
            tables=sql_cfg.get("tables", []),
            trust_score=sql_cfg.get("trust_score", 0.8),
            max_rows=sql_cfg.get("max_rows", 5),
        ))

    rest_cfg = tools_cfg.get("rest", {})
    if rest_cfg.get("enabled"):
        tools.append(RestTool.from_config(rest_cfg.get("endpoints", [])))

    tool_manager = ToolManager(tools=tools, top_k=retrieval_cfg.get("tool_top_k", 5)) if tools else None

    retriever = RetrieverAgent(hybrid_search, tool_manager=tool_manager)
    verifier = VerifierAgent(model=llm_cfg.get("verifier_model", "gpt-4o-mini"))
    updater = UpdaterAgent(vector_store=vector_store, hybrid_search=hybrid_search)

    return {
        "retriever": retriever,
        "verifier": verifier,
        "updater": updater,
        "vector_store": vector_store,
        "tools": tool_manager,
        "config": cfg,
    }


def ingest_documents(system: dict, data_dir: str | Path | None = None) -> int:
    """Ingest documents from data/raw into the KB."""
    root = get_project_root()
    data_dir = Path(data_dir or root / "data" / "raw")

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return 0

    docs = load_documents_from_dir(data_dir)
    if not docs:
        return 0

    contents = [d[0] for d in docs]
    metas = [d[1] for d in docs]

    version = system["updater"].ingest_documents(
        documents=contents,
        metadata_list=metas,
    )
    return version


def query(system: dict, question: str):
    """Run a query through the autonomous RAG pipeline."""
    llm_cfg = system.get("config", {}).get("llm", {})
    retrieval_cfg = system.get("config", {}).get("retrieval", {})
    return run_rag(
        query=question,
        retriever=system["retriever"],
        verifier=system["verifier"],
        updater=system["updater"],
        generator_model=llm_cfg.get("generator_model", "gpt-4o-mini"),
        max_retrieve_attempts=retrieval_cfg.get("max_retrieve_attempts", 2),
    )


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous RAG System")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from data/raw")
    parser.add_argument("--query", type=str, help="Query to run")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    args = parser.parse_args()

    system = build_system()

    if args.ingest:
        v = ingest_documents(system)
        print(f"Ingested documents. KB version: {v}")

    if args.query:
        resp = query(system, args.query)
        print("\n=== Answer ===")
        print(resp.answer)
        print("\n=== Citations ===")
        for c in resp.citations:
            print(f"  - {c}")
        print("\n=== Trust ===")
        for t in resp.trust_explanations:
            print(f"  {t}")
        print("\n=== Decay ===")
        for d in resp.decay_explanations:
            print(f"  {d}")
        print(f"\nVerdict: {resp.verdict.value}")

    if args.serve:
        from src.api import create_app
        import uvicorn
        app = create_app(system)
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
