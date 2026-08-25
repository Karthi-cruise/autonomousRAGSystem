"""Hybrid search combining FAISS semantic search with BM25 lexical search."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VersionedVectorStore
from src.utils.schema import RetrievedDocument
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
        self._bm25_tokens: list[list[str]] = []
        self._build_bm25()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_bm25(self) -> None:
        """Build BM25 index from current documents."""
        if self.vector_store.documents:
            self._bm25_corpus = [d["content"] for d in self.vector_store.documents]
            self._bm25_tokens = [self._tokenize(d) for d in self._bm25_corpus]
            self._bm25_index = BM25Okapi(self._bm25_tokens)
        else:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_tokens = []
    
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
        
        candidate_k = max(k * 4, k)

        # Get query embedding
        query_embedding = self.vector_store.embedding_model.encode(
            query, normalize_embeddings=True
        )
        
        # FAISS semantic search
        faiss_results = self.vector_store.search(query_embedding, top_k=candidate_k)
        faiss_scores = {idx: max(0.0, score) for idx, score in faiss_results}
        
        # BM25 lexical search
        bm25_scores = {}
        if self._bm25_index:
            tokenized_query = self._tokenize(query)
            bm25_raw = self._bm25_index.get_scores(tokenized_query)
            max_bm = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
            if max(bm25_raw) > 0:
                for i, s in enumerate(bm25_raw):
                    bm25_scores[i] = s / max_bm if max_bm > 0 else 0
            else:
                query_terms = set(tokenized_query)
                overlap_scores = [
                    len(query_terms & set(doc_tokens)) / len(query_terms)
                    if query_terms else 0.0
                    for doc_tokens in self._bm25_tokens
                ]
                max_overlap = max(overlap_scores) if overlap_scores else 0.0
                if max_overlap > 0:
                    for i, s in enumerate(overlap_scores):
                        bm25_scores[i] = s / max_overlap
        
        # Combine semantic and lexical candidate pools. BM25-only documents must
        # be allowed into the final ranking for exact-term enterprise queries.
        candidate_indices = set(faiss_scores)
        candidate_indices.update(
            idx
            for idx, score in sorted(bm25_scores.items(), key=lambda x: -x[1])[:candidate_k]
            if score > 0
        )

        combined: dict[int, float] = {}
        for idx in candidate_indices:
            combined[idx] = (
                self.faiss_weight * faiss_scores.get(idx, 0.0)
                + self.bm25_weight * bm25_scores.get(idx, 0.0)
            )
        
        # Build RetrievedDocuments with effective score (retrieval × trust × decay)
        results = []
        
        for idx, retrieval_score in combined.items():
            content, metadata = self.vector_store.get_document(idx)
            trust_score = self.trust_scorer.get_trust_score(metadata)
            metadata.trust_score = trust_score
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
        
        results.sort(key=lambda d: (d.effective_score, d.retrieval_score), reverse=True)
        return results[:k]
