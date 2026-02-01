"""Retriever Agent - hybrid search with trust and decay scoring."""

from __future__ import annotations

from src.retrieval.hybrid_search import HybridSearch
from src.utils.schema import RetrievalResult, RetrievedDocument


class RetrieverAgent:
    """Retrieves relevant documents using hybrid search with trust & decay."""
    
    def __init__(self, hybrid_search: HybridSearch):
        self.hybrid_search = hybrid_search
    
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve candidate documents with explainability."""
        docs = self.hybrid_search.search(query, top_k=top_k)
        
        explainability = {
            "query": query,
            "num_docs": len(docs),
            "sources": [
                {
                    "source": d.metadata.source,
                    "retrieval_score": d.retrieval_score,
                    "trust_score": d.metadata.trust_score,
                    "effective_score": d.effective_score,
                }
                for d in docs
            ],
        }
        
        return RetrievalResult(
            documents=docs,
            query=query,
            explainability=explainability,
        )
