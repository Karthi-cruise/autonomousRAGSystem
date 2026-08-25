"""Updater Agent - Autonomous KB updates, versioning, safe rollback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrieval.vector_store import VersionedVectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.utils.schema import DocumentMetadata


class UpdaterAgent:
    """
    Autonomous KB Update:
    - Ingests new trusted docs
    - Removes outdated vectors (optional, via new version)
    - Creates new vector versions
    - Maintains rollback safety
    """
    
    def __init__(
        self,
        vector_store: VersionedVectorStore,
        hybrid_search: HybridSearch,
    ):
        self.vector_store = vector_store
        self.hybrid_search = hybrid_search
    
    def ingest_documents(
        self,
        documents: list[str],
        source: str = "unknown",
        author: str = "",
        publisher: str = "",
        trust_score: float = 0.5,
        metadata_list: list[DocumentMetadata] | None = None,
    ) -> int:
        """Ingest new documents with metadata. Returns new version ID."""
        now = datetime.now(timezone.utc)
        version = self.vector_store.current_version + 1

        if metadata_list is not None and len(metadata_list) == len(documents):
            pass  # use provided metadata
        else:
            metadata_list = [
                DocumentMetadata(
                    source=source,
                    timestamp=now,
                    author=author,
                    publisher=publisher,
                    trust_score=trust_score,
                    version_id=f"v{version}",
                )
                for _ in documents
            ]

        new_version = self.vector_store.add_documents(
            documents=documents,
            metadata_list=metadata_list,
            create_new_version=True,
        )

        self.hybrid_search.update_bm25()
        return new_version
    
    def remove_outdated(self, doc_indices: list[int]) -> int:
        """Create new version with specified documents removed."""
        import numpy as np
        from src.retrieval.vector_store import faiss

        to_remove = set(doc_indices)
        new_docs = []
        new_meta = []
        for i, doc in enumerate(self.vector_store.documents):
            if i not in to_remove:
                new_docs.append(doc["content"])
                new_meta.append(self.vector_store.metadata_list[i])

        if not new_docs:
            return self.vector_store.current_version

        # Rebuild index and documents
        emb = self.vector_store.embedding_model.encode(new_docs, normalize_embeddings=True)
        emb = np.array(emb, dtype=np.float32)
        if getattr(self.vector_store, "lightweight_mode", False):
            self.vector_store.index = emb
        else:
            self.vector_store.index = faiss.IndexFlatIP(emb.shape[1])
            self.vector_store.index.add(emb)
        self.vector_store.documents = [
            {
                "id": f"doc_{i}",
                "content": d,
                "content_hash": self.vector_store._content_hash(d),
            }
            for i, d in enumerate(new_docs)
        ]
        self.vector_store.metadata_list = new_meta
        self.vector_store.current_version += 1
        self.vector_store._save_version(self.vector_store.current_version)
        self.hybrid_search.update_bm25()
        return self.vector_store.current_version
    
    def rollback(self, version: int) -> bool:
        """Rollback to a previous version. Safe KB update."""
        success = self.vector_store.rollback(version)
        if success:
            self.hybrid_search.update_bm25()
        return success

    def handle_verification_failure(
        self,
        query: str,
        answer: str,
        reason: str,
        rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Record verifier failures and rollback the vector store when possible."""
        versions = self.list_versions()
        current_version = self.vector_store.current_version
        rolled_back_to = None

        if rollback_on_failure:
            previous_versions = [version for version in versions if version < current_version]
            if previous_versions and self.rollback(previous_versions[-1]):
                rolled_back_to = previous_versions[-1]

        return {
            "query": query,
            "answer_preview": answer[:300],
            "reason": reason,
            "current_version": current_version,
            "rolled_back_to": rolled_back_to,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def list_versions(self) -> list[int]:
        """List available versions."""
        return self.vector_store.list_versions()
