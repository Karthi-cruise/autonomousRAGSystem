"""Shared data schemas for the Autonomous RAG System."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Verdict(Enum):
    """Verifier agent decision."""
    ACCEPT = "accept"
    REJECT = "reject"
    RE_RETRIEVE = "re_retrieve"
    FLAG_KB_ISSUE = "flag_kb_issue"


@dataclass
class DocumentMetadata:
    """Metadata for each document in the knowledge base."""
    source: str
    timestamp: datetime
    author: str = ""
    publisher: str = ""
    trust_score: float = 0.5
    version_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "author": self.author,
            "publisher": self.publisher,
            "trust_score": self.trust_score,
            "version_id": self.version_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        ts = data.get("timestamp", "")
        if isinstance(ts, str) and "T" in ts:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            source=data.get("source", ""),
            timestamp=ts,
            author=data.get("author", ""),
            publisher=data.get("publisher", ""),
            trust_score=float(data.get("trust_score", 0.5)),
            version_id=data.get("version_id", ""),
        )


@dataclass
class RetrievedDocument:
    """A document retrieved with scoring."""
    content: str
    metadata: DocumentMetadata
    retrieval_score: float
    effective_score: float  # retrieval × trust × decay
    doc_id: str = ""
    
    def to_context_str(self) -> str:
        return f"[Source: {self.metadata.source} | Trust: {self.metadata.trust_score:.2f}]\n{self.content}"


@dataclass
class RetrievalResult:
    """Result from the retriever agent."""
    documents: list[RetrievedDocument]
    query: str
    explainability: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from the generator."""
    answer: str
    citations: list[str]
    context_used: list[str]


@dataclass
class VerificationResult:
    """Result from the verifier agent."""
    verdict: Verdict
    hallucination_score: float
    groundedness_score: float
    explanation: str
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Final response from the autonomous RAG system."""
    answer: str
    citations: list[str]
    trust_explanations: list[str]
    decay_explanations: list[str]
    verification_explanation: str
    verdict: Verdict
