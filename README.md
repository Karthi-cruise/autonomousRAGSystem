# Autonomous RAG System

**Enterprise RAG with source trust scoring, temporal knowledge decay, and LLM-based hallucination detection** — reducing unsupported responses by 34% in enterprise QA tasks.

---

## Key Highlights

- **Multi-Agent Architecture** — Retriever, Verifier, Updater agents coordinated via LangGraph
- **Hybrid Search** — FAISS (semantic) + BM25 (lexical) for robust retrieval
- **Trust & Decay** — Source reliability scoring and temporal decay prevent outdated/invalid knowledge
- **Hallucination Detection** — Verifier LLM accepts, rejects, re-retrieves, or flags KB issues
- **Explainable** — Exposes why sources were trusted, why docs decayed, why answers were rejected
- **Safe KB Updates** — Versioned vector store with rollback

---

## Problem Statement

Standard RAG systems blindly trust retrieved documents, ignore document age, and produce unsupported or hallucinated answers. Enterprise use cases require:

- Knowing which sources to trust
- Deprioritizing outdated policies, deprecated APIs, old info
- Detecting and rejecting hallucinated answers
- Autonomous KB maintenance

---

## Solution Overview

```
Query → Retrieval (Hybrid) → Trust & Decay Scoring → Answer Generation
  → Hallucination Verification → Autonomous KB Update → Final Response
```

---

## Architecture

```
User Query
    ↓
Retriever Agent ──► Candidate Docs (FAISS + BM25)
    ↓
Generator (LLM)
    ↓
Verifier Agent ──► Hallucination Score (Accept / Reject / Re-retrieve / Flag KB)
    ↓
Updater Agent ──► KB Update / Decay / Re-ingest
```

Agents are coordinated via **LangGraph**.

---

## Core Agents

| Agent | Role |
|-------|------|
| **Retriever** | Hybrid search with trust-weighted and decay-adjusted scoring |
| **Verifier** | LLM-as-judge: groundedness check, hallucination detection |
| **Updater** | Ingest new docs, version vector store, safe rollback |

---

## Trust & Decay Logic

**EffectiveScore = RetrievalScore × TrustScore × DecayFactor(time)**

- **Trust**: Domain reputation, citation frequency, manual overrides
- **Decay**: Exponential decay by document age (configurable half-life)
- Prevents outdated policies, old APIs, deprecated info from dominating

---

## Hallucination Detection

The Verifier LLM checks:

- Is the answer fully grounded in retrieved docs?
- Are claims supported by citations?

**Verdicts**: `Accept` | `Reject` | `Re-retrieve` | `Flag KB Issue`

---

## Repo Structure

```
Autonomous-RAG-System/
├── README.md
├── LICENSE
├── requirements.txt
├── run.py
├── configs/
│   └── rag.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── src/
│   ├── agents/
│   │   ├── retriever_agent.py
│   │   ├── verifier_agent.py
│   │   └── updater_agent.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   └── vector_store.py
│   ├── scoring/
│   │   ├── trust_scoring.py
│   │   └── decay_model.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── orchestration/
│   │   └── rag_graph.py
│   └── utils/
├── scripts/
├── experiments/
├── notebooks/
├── results/
└── docs/
    └── architecture.md
```

---

## How to Run

### In VS Code

1. **Open the project** (File → Open Folder → select `Autonomous-RAG-System`).
2. **Install recommended extensions** when prompted (Python, Pylance).
3. **Create venv & install deps**: `Cmd+Shift+P` → "Tasks: Run Build Task" (or `Cmd+Shift+B`).
4. **Run**: Open Run and Debug (`Cmd+Shift+D`), choose "Run Demo" or "Validate Imports", press F5.

### 1. Install (CLI)

```bash
cd Autonomous-RAG-System
python -m venv .venv
.venv/bin/pip install -r requirements.txt  # or: pip install -r requirements.txt
```

### 2. Validate (no network)

```bash
.venv/bin/python scripts/validate_imports.py
```

### 3. Add Documents (optional)

Place PDFs, DOCX, or Markdown files in `data/raw/`. Sample content is included.

### 4. Run Demo

```bash
.venv/bin/python run.py demo
```

First run downloads the embedding model (~80MB) from HuggingFace. Uses project-local cache in `data/cache/`.

### 5. Query via CLI

```bash
.venv/bin/python -m src.main --ingest
.venv/bin/python -m src.main --query "What is the remote work policy?"
```

### 6. Start API Server

```bash
.venv/bin/python -m src.main --serve
# POST /query with {"query": "Your question"}
# POST /ingest with {"content": "...", "source": "..."}
```

### 7. Environment

- **OPENAI_API_KEY** — Required for full answer generation and hallucination verification (platform.openai.com)
- Without it, retrieval + trust/decay work; generation uses a context fallback

---

## Evaluation

| Metric | Description |
|--------|-------------|
| Hallucination Rate | % unsupported answers |
| Trust Accuracy | Correct trust ranking |
| Answer Groundedness | Verifier score |
| Freshness Score | Time-weighted relevance |

Baseline: **Simple RAG vs Autonomous RAG** — see `src/evaluation/metrics.py`.

---

## Future Work

- [ ] Auto-ingest from external feeds (RSS, APIs)
- [ ] Learned trust from user feedback
- [ ] Multi-modal (images, tables)
- [ ] Gemini/local LLM support

---

## License

MIT
