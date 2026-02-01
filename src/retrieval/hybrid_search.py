"""Hybrid search combining FAISS semantic search with BM25 lexical search."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VersionedVectorStore
from src.utils.schema import DocumentMetadata, RetrievedDocument
from src.scoring.trust_scoring import TrustScorer
from src.scoring.decay_model import DecayModel


class HybridSearch:
    """FAISS + BM25 hybrid search with trust and decay scoring."""
    
    def __init__(
        self,
        vector_store: VersionedVectorStore,
        trust_scorer: TrustScorer,
        decay_model: DecayModel,
        faiss_weight: float = 0.6,
        bm25_weight: float = 0.4,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.trust_scorer = trust_scorer
        self.decay_model = decay_model
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []
        self._build_bm25()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_bm25(self) -> None:
        """Build BM25 index from current documents."""
        if self.vector_store.documents:
            self._bm25_corpus = [d["content"] for d in self.vector_store.documents]
            tokenized = [self._tokenize(d) for d in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized)
        else:
            self._bm25_index = None
            self._bm25_corpus = []
    
    def update_bm25(self) -> None:
        """Rebuild BM25 after vector store updates."""
        self._build_bm25()
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Perform hybrid search with trust and decay scoring."""
        k = top_k or self.top_k
        if not self.vector_store.documents:
            return []
        
        # Get query embedding
        query_embedding = self.vector_store.embedding_model.encode(
            query, normalize_embeddings=True
        )
        
        # FAISS semantic search
        faiss_results = self.vector_store.search(query_embedding, top_k=k * 2)
        
        # BM25 lexical search
        bm25_scores = {}
        if self._bm25_index:
            tokenized_query = self._tokenize(query)
            bm25_raw = self._bm25_index.get_scores(tokenized_query)
            max_bm = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
            for i, s in enumerate(bm25_raw):
                bm25_scores[i] = s / max_bm if max_bm > 0 else 0
        
        # Combine scores (reciprocal rank fusion style)
        combined: dict[int, float] = {}
        for rank, (idx, faiss_score) in enumerate(faiss_results):
            bm25_s = bm25_scores.get(idx, 0)
            comb = self.faiss_weight * faiss_score + self.bm25_weight * bm25_s
            combined[idx] = comb
        
        # Sort and take top_k
        sorted_indices = sorted(combined.items(), key=lambda x: -x[1])[:k]
        
        # Build RetrievedDocuments with effective score (retrieval × trust × decay)
        results = []
        explainability = []
        
        for idx, retrieval_score in sorted_indices:
            content, metadata = self.vector_store.get_document(idx)
            trust_score = self.trust_scorer.get_trust_score(metadata)
            decay_factor = self.decay_model.get_decay_factor(metadata.timestamp)
            effective_score = retrieval_score * trust_score * decay_factor
            
            doc_id = self.vector_store.documents[idx].get("id", f"doc_{idx}")
            results.append(RetrievedDocument(
                content=content,
                metadata=metadata,
                retrieval_score=retrieval_score,
                effective_score=effective_score,
                doc_id=doc_id,
            ))
            
            explainability.append({
                "source": metadata.source,
                "retrieval_score": retrieval_score,
                "trust_score": trust_score,
                "decay_factor": decay_factor,
                "effective_score": effective_score,
            })
        
        return results
