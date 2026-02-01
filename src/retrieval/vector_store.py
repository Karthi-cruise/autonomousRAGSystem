"""FAISS vector store with versioning and safe updates."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import get_project_root
from src.utils.schema import DocumentMetadata, RetrievedDocument


class VersionedVectorStore:
    """FAISS-backed vector store with versioning for safe KB updates."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_path: str | Path | None = None,
        version_prefix: str = "v",
        cache_dir: str | Path | None = None,
    ):
        import os
        if cache_dir:
            os.environ.setdefault("HF_HOME", str(cache_dir))
        self.embedding_model = SentenceTransformer(
            embedding_model,
            cache_folder=str(cache_dir) if cache_dir else None,
        )
        self.base_path = Path(base_path or get_project_root() / "data" / "processed" / "faiss_index")
        self.version_prefix = version_prefix
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict[str, Any]] = []
        self.metadata_list: list[DocumentMetadata] = []
        self.current_version = 0
        self._load_or_init()
    
    def _load_or_init(self) -> None:
        """Load existing index or initialize empty."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        versions = sorted(
            p for p in self.base_path.iterdir()
            if p.is_dir() and p.name.startswith(self.version_prefix)
        )
        if versions:
            self._load_version(versions[-1])
        else:
            self.index = None
            self.documents = []
            self.metadata_list = []
    
    def _load_version(self, version_path: Path) -> None:
        """Load a specific version of the index."""
        index_file = version_path / "index.faiss"
        docs_file = version_path / "documents.pkl"
        meta_file = version_path / "metadata.json"
        
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
        if docs_file.exists():
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
        if meta_file.exists():
            with open(meta_file) as f:
                meta_data = json.load(f)
            self.metadata_list = [DocumentMetadata.from_dict(m) for m in meta_data]
        
        version_num = version_path.name.replace(self.version_prefix, "")
        self.current_version = int(version_num) if version_num.isdigit() else 0
    
    def _save_version(self, version: int) -> Path:
        """Save current index as a new version (rollback-safe)."""
        version_path = self.base_path / f"{self.version_prefix}{version}"
        version_path.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(version_path / "index.faiss"))
        with open(version_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        with open(version_path / "metadata.json", "w") as f:
            json.dump([m.to_dict() for m in self.metadata_list], f, indent=2)
        
        return version_path
    
    def add_documents(
        self,
        documents: list[str],
        metadata_list: list[DocumentMetadata],
        create_new_version: bool = True,
    ) -> int:
        """Add documents to the store. Returns new version number."""
        if not documents:
            return self.current_version
        
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings)
        for i, (doc, meta) in enumerate(zip(documents, metadata_list)):
            doc_id = f"doc_{len(self.documents) + i}"
            self.documents.append({"id": doc_id, "content": doc})
            self.metadata_list.append(meta)
        
        if create_new_version:
            self.current_version += 1
            self._save_version(self.current_version)
        
        return self.current_version
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Search for similar documents. Returns [(doc_idx, score), ...]."""
        if self.index is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append((int(idx), float(scores[0][i])))
        return results
    
    def get_document(self, idx: int) -> tuple[str, DocumentMetadata]:
        """Get document content and metadata by index."""
        doc = self.documents[idx]
        meta = self.metadata_list[idx]
        return doc["content"], meta
    
    def rollback(self, version: int) -> bool:
        """Rollback to a previous version. Returns True if successful."""
        version_path = self.base_path / f"{self.version_prefix}{version}"
        if not version_path.exists():
            return False
        self._load_version(version_path)
        return True
    
    def list_versions(self) -> list[int]:
        """List available versions for rollback."""
        versions = []
        for p in self.base_path.iterdir():
            if p.is_dir() and p.name.startswith(self.version_prefix):
                num = p.name.replace(self.version_prefix, "")
                if num.isdigit():
                    versions.append(int(num))
        return sorted(versions)
