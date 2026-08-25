"""Coordinator for optional SQL and REST retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.utils.schema import RetrievedDocument


class RetrievalTool(Protocol):
    def search(self, query: str) -> list[RetrievedDocument]:
        """Return documents grounded in an external tool."""


@dataclass
class ToolManager:
    """Run configured retrieval tools and merge their grounded documents."""

    tools: list[RetrievalTool] = field(default_factory=list)
    top_k: int = 5

    def search(self, query: str) -> list[RetrievedDocument]:
        results: list[RetrievedDocument] = []
        for tool in self.tools:
            results.extend(tool.search(query))

        results.sort(key=lambda doc: (doc.effective_score, doc.retrieval_score), reverse=True)
        return results[:self.top_k]

    def describe(self) -> list[str]:
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.__class__.__name__)
        return descriptions
