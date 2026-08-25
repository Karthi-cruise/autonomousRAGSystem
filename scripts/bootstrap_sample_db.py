#!/usr/bin/env python3
"""Create a sample SQLite database used by the SQL retrieval tool."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def create_sample_db(path: str | Path) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                details TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_docs (
                id INTEGER PRIMARY KEY,
                endpoint TEXT NOT NULL,
                details TEXT NOT NULL,
                rate_limit TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("DELETE FROM policies")
        conn.execute("DELETE FROM api_docs")
        conn.executemany(
            "INSERT INTO policies (name, details, updated_at) VALUES (?, ?, ?)",
            [
                (
                    "Remote Work Policy",
                    "Employees may work remotely up to 3 days per week with manager approval.",
                    "2026-02-01",
                ),
                (
                    "Security Guidelines",
                    "MFA is required for all company accounts and suspicious emails go to security@company.com.",
                    "2026-02-01",
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO api_docs (endpoint, details, rate_limit, updated_at) VALUES (?, ?, ?, ?)",
            [
                (
                    "https://api.company.com/v2",
                    "All requests require Authorization: Bearer <key>.",
                    "100 requests per minute",
                    "2026-02-01",
                )
            ],
        )
        conn.commit()

    return db_path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = create_sample_db(root / "data" / "metadata" / "sample_knowledge.db")
    print(f"Sample SQLite database ready: {path}")


if __name__ == "__main__":
    main()
