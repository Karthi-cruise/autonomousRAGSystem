#!/usr/bin/env python3
"""Run the Autonomous RAG demo - ingest sample docs and query."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import build_system, ingest_documents, query


def main():
    print("Building Autonomous RAG System...")
    system = build_system()

    print("Ingesting sample documents from data/raw...")
    v = ingest_documents(system)
    print(f"  KB version: {v}")

    if v == 0:
        print("\nNo documents found in data/raw. Add PDF, DOCX, or Markdown files.")
        print("Using inline sample for demo...")
        system["updater"].ingest_documents(
            documents=["Company remote work policy: Employees may work remotely up to 3 days per week. API endpoint is https://api.company.com/v2. Rate limit: 100 requests per minute."],
            source="inline_sample",
            trust_score=0.7,
        )

    demo_queries = [
        "What is the remote work policy?",
        "What is the API rate limit?",
    ]

    print("\n" + "=" * 60)
    print("DEMO: Autonomous RAG Queries")
    print("=" * 60)

    for q in demo_queries:
        print(f"\nQuery: {q}")
        try:
            resp = query(system, q)
            print(f"Answer: {resp.answer[:300]}...")
            print(f"Verdict: {resp.verdict.value}")
        except Exception as e:
            print(f"Error: {e}")
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                print("  Set OPENAI_API_KEY for full LLM/verification. Retrieval works without it.")

    print("\n" + "=" * 60)
    print("Demo complete. Use: python -m src.main --query 'Your question'")
    print("Or: python -m src.main --serve  (FastAPI on :8000)")
    print("=" * 60)


if __name__ == "__main__":
    main()
