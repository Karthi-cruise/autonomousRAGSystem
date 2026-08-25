"""FAISS vector store with versioning and safe updates."""

from __future__ import annotations

import json
import os
import pickle
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

from src.utils.config import get_project_root
from src.utils.schema import DocumentMetadata


class VersionedVectorStore:
    """FAISS-backed vector store with versioning for safe KB updates."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_path: str | Path | None = None,
        version_prefix: str = "v",
        cache_dir: str | Path | None = None,
    ):
        if cache_dir:
            os.environ.setdefault("HF_HOME", str(cache_dir))
        self.lightweight_mode = os.environ.get("LIGHTWEIGHT_DEPLOY") == "1" or SentenceTransformer is None
        if self.lightweight_mode:
            self.embedding_model = LightweightEmbeddingModel()
        else:
            cached_model_path = self._cached_model_path(embedding_model, cache_dir)
            model_name_or_path = str(cached_model_path or embedding_model)
            self.embedding_model = SentenceTransformer(
                model_name_or_path,
                cache_folder=str(cache_dir) if cache_dir else None,
                local_files_only=bool(cached_model_path) or os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        self.base_path = Path(base_path or get_project_root() / "data" / "processed" / "faiss_index")
        self.version_prefix = version_prefix
        self.index: Any | None = None
        self.documents: list[dict[str, Any]] = []
        self.metadata_list: list[DocumentMetadata] = []
        self.current_version = 0
        self._load_or_init()

    def _version_number(self, version_path: Path) -> int:
        """Return numeric version suffix, or -1 for non-version directories."""
        version_num = version_path.name.replace(self.version_prefix, "", 1)
        return int(version_num) if version_num.isdigit() else -1

    def _version_dirs(self) -> list[Path]:
        """List version directories in numeric order."""
        return sorted(
            (
                p for p in self.base_path.iterdir()
                if p.is_dir()
                and p.name.startswith(self.version_prefix)
                and self._version_number(p) >= 0
            ),
            key=self._version_number,
        )

    def _content_hash(self, content: str) -> str:
        """Stable content fingerprint used to skip exact duplicate ingests."""
        return sha256(content.encode("utf-8")).hexdigest()

    def _cached_model_path(
        self,
        embedding_model: str,
        cache_dir: str | Path | None,
    ) -> Path | None:
        """Return a complete HuggingFace snapshot path if it is cached locally."""
        if not cache_dir or Path(embedding_model).exists():
            return None

        repo_dir = Path(cache_dir) / f"models--{embedding_model.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        ref_file = repo_dir / "refs" / "main"
        if ref_file.exists():
            ref = ref_file.read_text().strip()
            ref_snapshot = snapshots_dir / ref
            if (ref_snapshot / "modules.json").exists():
                return ref_snapshot

        snapshots = [
            path for path in snapshots_dir.iterdir()
            if path.is_dir() and (path / "modules.json").exists()
        ]
        if not snapshots:
            return None
        return max(snapshots, key=lambda path: path.stat().st_mtime)
    
    def _load_or_init(self) -> None:
        """Load existing index or initialize empty."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        versions = self._version_dirs()
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

        self.index = None
        self.documents = []
        self.metadata_list = []
        
        if index_file.exists() and faiss is not None and not getattr(self, "lightweight_mode", False):
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
        
        if self.index is not None and faiss is not None and not getattr(self, "lightweight_mode", False):
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
        if len(documents) != len(metadata_list):
            raise ValueError("documents and metadata_list must have the same length")

        existing_hashes = {
            doc.get("content_hash") or self._content_hash(doc.get("content", ""))
            for doc in self.documents
        }
        filtered: list[tuple[str, DocumentMetadata, str]] = []
        for doc, meta in zip(documents, metadata_list):
            content_hash = self._content_hash(doc)
            if content_hash in existing_hashes:
                continue
            existing_hashes.add(content_hash)
            filtered.append((doc, meta, content_hash))

        if not filtered:
            return self.current_version

        documents = [doc for doc, _, _ in filtered]
        
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if getattr(self, "lightweight_mode", False):
            self.index = embeddings if self.index is None else np.vstack([self.index, embeddings])
        elif self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        if not getattr(self, "lightweight_mode", False):
            self.index.add(embeddings)
        for doc, meta, content_hash in filtered:
            doc_id = f"doc_{len(self.documents)}"
            self.documents.append({
                "id": doc_id,
                "content": doc,
                "content_hash": content_hash,
            })
            self.metadata_list.append(meta)
        
        if create_new_version:
            self.current_version += 1
            self._save_version(self.current_version)
        
        return self.current_version

    def deduplicate(self, create_new_version: bool = False) -> bool:
        """Remove exact duplicate content while preserving first-seen metadata."""
        seen_hashes = set()
        unique: list[tuple[str, DocumentMetadata, str]] = []

        for doc, meta in zip(self.documents, self.metadata_list):
            content = doc.get("content", "")
            content_hash = doc.get("content_hash") or self._content_hash(content)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            unique.append((content, meta, content_hash))

        if len(unique) == len(self.documents):
            return False

        if unique:
            documents = [doc for doc, _, _ in unique]
            embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            if getattr(self, "lightweight_mode", False):
                self.index = embeddings
            else:
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)
        else:
            self.index = None

        self.documents = [
            {
                "id": f"doc_{i}",
                "content": content,
                "content_hash": content_hash,
            }
            for i, (content, _, content_hash) in enumerate(unique)
        ]
        self.metadata_list = [meta for _, meta, _ in unique]

        if create_new_version:
            self.current_version += 1
            self._save_version(self.current_version)

        return True
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Search for similar documents. Returns [(doc_idx, score), ...]."""
        if self.index is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        if getattr(self, "lightweight_mode", False):
            scores = np.dot(self.index, query_embedding[0])
            indices = np.argsort(-scores)[:top_k]
            return [(int(idx), float(scores[idx])) for idx in indices]

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
        return [self._version_number(p) for p in self._version_dirs()]


class LightweightEmbeddingModel:
    """Small deterministic embedding model for low-memory cloud deploys."""

    dimension = 256

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", text.lower())

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokenize(text):
            vector[hash(token) % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode(self, values, normalize_embeddings: bool = True):
        if isinstance(values, str):
            return self._embed_one(values)
        return np.array([self._embed_one(value) for value in values], dtype=np.float32)
