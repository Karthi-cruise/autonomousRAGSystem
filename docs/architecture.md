# Autonomous RAG System — Architecture

## Overview

Multi-agent RAG with SQL/REST tool-calling, trust scoring, temporal decay, and hallucination detection.

## Data Flow

1. **Query** → Retriever Agent
2. **Retriever** → Hybrid search (FAISS + BM25) + SQL/REST tools → Candidate docs
3. **Scoring** → Trust × Decay → Effective scores
4. **Generator** → LLM answer from context
5. **Verifier** → Hallucination check → Verdict
6. **Updater** (on Flag KB) → Record issue / version / rollback

## Components

### Vector Store (FAISS)

- IndexFlatIP for cosine similarity
- Versioned snapshots for safe updates
- Metadata: source, timestamp, author, trust_score, version_id

### Hybrid Search

- FAISS for semantic similarity (sentence-transformers)
- BM25 for lexical matching
- Weighted combination

### Tool-Calling Grounding

- SQLite tool searches configured read-only tables and converts matching rows into grounded documents
- REST tool calls configured GET endpoints and converts successful responses into grounded documents
- Tool results are ranked with vector results using retrieval, trust, and freshness signals

### Trust Scorer

- Domain reputation (trusted domains)
- Citation frequency
- Manual overrides
- Output: score ∈ [0, 1]

### Decay Model

- Exponential decay: 0.5^(days/half_life)
- Configurable half-life (default 180 days)
- Prevents stale knowledge dominance

### Verifier

- LLM-as-judge
- Structured JSON output: verdict, hallucination_score, groundedness_score
- Actions: Accept | Reject | Re-retrieve | Flag KB

## LangGraph Workflow

```
retrieve → generate → verify
    ↑         |           |
    |         |           +→ format_response
    +---------+           +→ update_kb → format_response
    (re_retrieve)         (flag_kb)
```

## Explainability

- Trust: Why source was trusted (domain, citation, override)
- Decay: Document age and decay factor
- Verification: Verifier explanation for accept/reject
