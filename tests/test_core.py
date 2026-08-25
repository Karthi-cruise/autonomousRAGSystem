from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

from src.agents.verifier_agent import VerifierAgent
from src.api import create_app
from src.orchestration.rag_graph import _extractive_answer
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.vector_store import VersionedVectorStore
from src.scoring.decay_model import DecayModel
from src.scoring.trust_scoring import TrustScorer
from src.tools.rest_tool import RestTool
from src.tools.sql_tool import SQLTool
from src.utils.schema import (
    DocumentMetadata,
    RAGResponse,
    RetrievalResult,
    RetrievedDocument,
    Verdict,
)


class FakeEmbeddingModel:
    def encode(self, values, normalize_embeddings=True):
        if isinstance(values, str):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.ones((len(values), 2), dtype=np.float32)


class FakeVectorStore:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.embedding_model = FakeEmbeddingModel()
        self.documents = [
            {"id": "doc_0", "content": "Employees may work remotely three days per week."},
            {"id": "doc_1", "content": "Rate limit: 100 requests per minute."},
        ]
        self.metadata_list = [
            DocumentMetadata("remote.md", now, trust_score=1.0),
            DocumentMetadata("api.md", now, trust_score=1.0),
        ]

    def search(self, query_embedding, top_k=5):
        return [(0, 0.2)]

    def get_document(self, idx):
        return self.documents[idx]["content"], self.metadata_list[idx]


class FakeRetriever:
    def __init__(self):
        self.hybrid_search = SimpleNamespace(
            trust_scorer=TrustScorer(),
            decay_model=DecayModel(),
        )

    def retrieve(self, query, top_k=None):
        now = datetime.now(timezone.utc)
        doc = RetrievedDocument(
            content="Rate limit: 100 requests per minute.",
            metadata=DocumentMetadata("api.md", now, trust_score=1.0),
            retrieval_score=1.0,
            effective_score=1.0,
            doc_id="doc_1",
        )
        return RetrievalResult(documents=[doc], query=query)


class TestCoreBehavior(unittest.TestCase):
    def test_hybrid_search_includes_bm25_only_matches(self):
        search = HybridSearch(
            vector_store=FakeVectorStore(),
            trust_scorer=TrustScorer(),
            decay_model=DecayModel(),
            faiss_weight=0.6,
            bm25_weight=0.4,
            top_k=1,
        )

        results = search.search("What is the API rate limit?", top_k=1)

        self.assertEqual(results[0].metadata.source, "api.md")
        self.assertIn("100 requests", results[0].content)

    def test_extractive_answer_targets_relevant_passage(self):
        now = datetime.now(timezone.utc)
        retrieval = RetrievalResult(
            query="What is the API rate limit?",
            documents=[
                RetrievedDocument(
                    content=(
                        "Remote Work Policy\nEmployees may work remotely up to 3 days per week.\n\n"
                        "API Documentation v2.1\nRate limit: 100 requests per minute."
                    ),
                    metadata=DocumentMetadata("sample_knowledge.md", now, trust_score=1.0),
                    retrieval_score=1.0,
                    effective_score=1.0,
                    doc_id="doc_0",
                )
            ],
        )

        answer = _extractive_answer("What is the API rate limit?", retrieval)

        self.assertIn("100 requests per minute", answer)
        self.assertIn("[Source: sample_knowledge.md]", answer)

    def test_vector_store_skips_exact_duplicate_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionedVectorStore.__new__(VersionedVectorStore)
            store.embedding_model = FakeEmbeddingModel()
            store.base_path = Path(tmp)
            store.version_prefix = "v"
            store.index = None
            store.documents = []
            store.metadata_list = []
            store.current_version = 0

            metadata = DocumentMetadata("source.md", datetime.now(timezone.utc))
            first_version = store.add_documents(["same content"], [metadata])
            second_version = store.add_documents(["same content"], [metadata])

            self.assertEqual(first_version, 1)
            self.assertEqual(second_version, 1)
            self.assertEqual(len(store.documents), 1)
            self.assertIn("content_hash", store.documents[0])

    def test_vector_store_deduplicates_loaded_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionedVectorStore.__new__(VersionedVectorStore)
            store.embedding_model = FakeEmbeddingModel()
            store.base_path = Path(tmp)
            store.version_prefix = "v"
            store.index = None
            store.documents = [
                {"id": "doc_0", "content": "same content"},
                {"id": "doc_1", "content": "same content"},
            ]
            store.metadata_list = [
                DocumentMetadata("one.md", datetime.now(timezone.utc)),
                DocumentMetadata("two.md", datetime.now(timezone.utc)),
            ]
            store.current_version = 3

            changed = store.deduplicate(create_new_version=False)

            self.assertTrue(changed)
            self.assertEqual(store.current_version, 3)
            self.assertEqual(len(store.documents), 1)
            self.assertEqual(store.metadata_list[0].source, "one.md")

    def test_version_listing_uses_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionedVectorStore.__new__(VersionedVectorStore)
            store.base_path = Path(tmp)
            store.version_prefix = "v"
            for name in ("v1", "v10", "v2", "notes"):
                (store.base_path / name).mkdir()

            self.assertEqual(store.list_versions(), [1, 2, 10])

    def test_cached_model_path_prefers_ref_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            repo_dir = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
            snapshot = repo_dir / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "modules.json").write_text("{}")
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text("abc123")

            store = VersionedVectorStore.__new__(VersionedVectorStore)
            path = store._cached_model_path(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_dir,
            )

            self.assertEqual(path, snapshot)

    def test_metadata_and_response_are_json_safe(self):
        metadata = DocumentMetadata.from_dict({
            "source": "doc.md",
            "timestamp": "2026-02-01",
            "trust_score": 0.7,
        })
        response = RAGResponse(
            answer="Answer",
            citations=["doc.md"],
            trust_explanations=[],
            decay_explanations=[],
            verification_explanation="ok",
            verdict=Verdict.ACCEPT,
        )

        self.assertIsInstance(metadata.timestamp, datetime)
        self.assertEqual(response.to_dict()["verdict"], "accept")

    def test_api_query_returns_serialized_response(self):
        system = {
            "retriever": FakeRetriever(),
            "verifier": VerifierAgent(api_key=None),
            "updater": SimpleNamespace(list_versions=lambda: []),
            "config": {},
        }
        client = TestClient(create_app(system))

        response = client.post("/query", json={"query": "What is the API rate limit?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "accept")
        self.assertIn("100 requests per minute", body["answer"])

    def test_sql_tool_returns_grounded_rows(self):
        from scripts.bootstrap_sample_db import create_sample_db

        with tempfile.TemporaryDirectory() as tmp:
            db_path = create_sample_db(Path(tmp) / "sample.db")
            tool = SQLTool(
                database_path=db_path,
                tables=["api_docs"],
                trust_score=0.9,
            )

            results = tool.search("API rate limit")

            self.assertTrue(results)
            self.assertIn("100 requests per minute", results[0].content)
            self.assertEqual(results[0].metadata.source, "sqlite:sample.db:api_docs")

    def test_rest_tool_from_config_describes_endpoint(self):
        tool = RestTool.from_config([
            {
                "name": "status",
                "url": "http://example.invalid/status",
                "trust_score": 0.6,
            }
        ])

        self.assertEqual(tool.endpoints[0].name, "status")
        self.assertEqual(tool.endpoints[0].trust_score, 0.6)

    def test_updater_records_and_rolls_back_verification_failure(self):
        from src.agents.updater_agent import UpdaterAgent

        vector_store = SimpleNamespace(
            current_version=3,
        )
        hybrid_search = SimpleNamespace(update_bm25=Mock())
        updater = UpdaterAgent(vector_store=vector_store, hybrid_search=hybrid_search)
        updater.list_versions = Mock(return_value=[1, 2, 3])
        updater.rollback = Mock(return_value=True)

        action = updater.handle_verification_failure(
            query="bad answer?",
            answer="unsupported",
            reason="flagged",
        )

        self.assertEqual(action["rolled_back_to"], 2)
        updater.rollback.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
