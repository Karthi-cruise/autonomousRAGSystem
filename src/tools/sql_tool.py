"""Read-only SQLite retrieval tool."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.schema import DocumentMetadata, RetrievedDocument


@dataclass
class SQLTool:
    """Search configured SQLite tables and return rows as retrieved documents."""

    database_path: Path
    tables: list[str]
    trust_score: float = 0.8
    max_rows: int = 5

    def __post_init__(self) -> None:
        self.database_path = Path(self.database_path)
        self.tables = [table for table in self.tables if self._is_identifier(table)]

    def _is_identifier(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))

    def _query_terms(self, query: str) -> list[str]:
        return [
            term
            for term in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower())
            if term not in {"what", "where", "when", "with", "from", "this", "that", "have"}
        ]

    def _table_columns(self, conn: sqlite3.Connection, table: str) -> list[str]:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [row[1] for row in rows if self._is_identifier(row[1])]

    def _row_to_text(self, table: str, row: sqlite3.Row) -> str:
        parts = [f"{key}: {row[key]}" for key in row.keys()]
        return f"SQL table `{table}` row: " + "; ".join(parts)

    def search(self, query: str) -> list[RetrievedDocument]:
        """Run a safe lexical search over configured SQLite tables."""
        if not self.database_path.exists() or not self.tables:
            return []

        terms = self._query_terms(query)
        results: list[RetrievedDocument] = []

        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            for table in self.tables:
                columns = self._table_columns(conn, table)
                if not columns:
                    continue

                where = ""
                params: list[Any] = []
                if terms:
                    clauses = []
                    for term in terms:
                        term_clauses = [f'CAST("{column}" AS TEXT) LIKE ?' for column in columns]
                        clauses.append("(" + " OR ".join(term_clauses) + ")")
                        params.extend([f"%{term}%"] * len(columns))
                    where = "WHERE " + " OR ".join(clauses)

                sql = f'SELECT * FROM "{table}" {where} LIMIT ?'
                params.append(self.max_rows)
                for row in conn.execute(sql, params).fetchall():
                    content = self._row_to_text(table, row)
                    overlap = sum(1 for term in terms if term in content.lower())
                    retrieval_score = min(1.0, 0.35 + overlap / max(len(terms), 1))
                    metadata = DocumentMetadata(
                        source=f"sqlite:{self.database_path.name}:{table}",
                        timestamp=datetime.now(timezone.utc),
                        publisher="sqlite",
                        trust_score=self.trust_score,
                    )
                    results.append(RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        retrieval_score=retrieval_score,
                        effective_score=retrieval_score * self.trust_score,
                        doc_id=f"sql:{table}:{len(results)}",
                    ))

        results.sort(key=lambda doc: doc.effective_score, reverse=True)
        return results[:self.max_rows]
