"""Configured REST API retrieval tool."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.utils.schema import DocumentMetadata, RetrievedDocument


@dataclass
class RestEndpoint:
    name: str
    url: str
    method: str = "GET"
    query_param: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    trust_score: float = 0.75


@dataclass
class RestTool:
    """Fetch configured REST endpoints and return response snippets as documents."""

    endpoints: list[RestEndpoint]
    timeout_seconds: float = 5.0
    max_chars: int = 2000

    @classmethod
    def from_config(cls, endpoint_configs: list[dict[str, Any]]) -> "RestTool":
        endpoints = [
            RestEndpoint(
                name=item["name"],
                url=item["url"],
                method=item.get("method", "GET").upper(),
                query_param=item.get("query_param"),
                headers=item.get("headers", {}),
                trust_score=float(item.get("trust_score", 0.75)),
            )
            for item in endpoint_configs
            if item.get("name") and item.get("url")
        ]
        return cls(endpoints=endpoints)

    def _format_payload(self, payload: bytes, content_type: str) -> str:
        text = payload.decode("utf-8", errors="replace")
        if "json" not in content_type.lower():
            return text[:self.max_chars]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text[:self.max_chars]
        return json.dumps(parsed, indent=2, sort_keys=True)[:self.max_chars]

    def _endpoint_url(self, endpoint: RestEndpoint, query: str) -> str:
        if not endpoint.query_param:
            return endpoint.url
        separator = "&" if "?" in endpoint.url else "?"
        return endpoint.url + separator + urlencode({endpoint.query_param: query})

    def search(self, query: str) -> list[RetrievedDocument]:
        """Call configured GET endpoints and return successful responses."""
        results: list[RetrievedDocument] = []
        for endpoint in self.endpoints:
            if endpoint.method != "GET":
                continue

            url = self._endpoint_url(endpoint, query)
            req = Request(url, headers=endpoint.headers, method="GET")
            try:
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    content_type = response.headers.get("content-type", "")
                    payload = self._format_payload(response.read(), content_type)
            except (OSError, URLError) as exc:
                payload = f"REST endpoint `{endpoint.name}` unavailable: {exc}"

            metadata = DocumentMetadata(
                source=f"rest:{endpoint.name}",
                timestamp=datetime.now(timezone.utc),
                publisher="rest",
                trust_score=endpoint.trust_score,
            )
            results.append(RetrievedDocument(
                content=f"REST endpoint `{endpoint.name}` response:\n{payload}",
                metadata=metadata,
                retrieval_score=1.0 if "unavailable" not in payload else 0.1,
                effective_score=endpoint.trust_score,
                doc_id=f"rest:{endpoint.name}",
            ))

        return results
