"""Evaluation metrics: hallucination rate, trust accuracy, groundedness, freshness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.utils.schema import Verdict


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    hallucination_rate: float = 0.0
    trust_accuracy: float = 0.0
    answer_groundedness: float = 0.0
    freshness_score: float = 0.0
    retrieval_accuracy: float = 0.0
    total_queries: int = 0
    rejected_count: int = 0
    re_retrieve_count: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hallucination_rate": self.hallucination_rate,
            "trust_accuracy": self.trust_accuracy,
            "answer_groundedness": self.answer_groundedness,
            "freshness_score": self.freshness_score,
            "retrieval_accuracy": self.retrieval_accuracy,
            "total_queries": self.total_queries,
            "rejected_count": self.rejected_count,
            "re_retrieve_count": self.re_retrieve_count,
        }


def compute_hallucination_rate(verdicts: list[Verdict]) -> float:
    """% of unsupported/hallucinated answers (rejected)."""
    if not verdicts:
        return 0.0
    rejected = sum(1 for v in verdicts if v == Verdict.REJECT)
    return rejected / len(verdicts)


def compute_groundedness_scores(verification_results: list[dict]) -> float:
    """Average groundedness score from verifier."""
    if not verification_results:
        return 0.0
    scores = [r.get("groundedness_score", 0.5) for r in verification_results]
    return sum(scores) / len(scores)


def compute_freshness_score(doc_ages: list[float], decay_factors: list[float]) -> float:
    """Time-weighted relevance - higher decay = fresher content used."""
    if not decay_factors:
        return 0.0
    return sum(decay_factors) / len(decay_factors)


def evaluate_rag_run(
    query: str,
    response: Any,
    expected_answer: str | None = None,
    retrieval_docs: list[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a single RAG run."""
    result = {
        "query": query,
        "verdict": getattr(response, "verdict", Verdict.ACCEPT).value,
        "hallucination_detected": getattr(response, "verdict", Verdict.ACCEPT) == Verdict.REJECT,
        "groundedness": 1.0,
        "trust_explanations": getattr(response, "trust_explanations", []),
        "decay_explanations": getattr(response, "decay_explanations", []),
    }

    # If we have verification result with groundedness
    return result


def run_baseline_comparison(
    queries: list[str],
    autonomous_rag_fn: Callable[[str], Any],
    simple_rag_fn: Callable[[str], Any],
) -> dict[str, Any]:
    """
    Compare Simple RAG vs Autonomous RAG.
    Returns metrics for both and improvement delta.
    """
    auto_verdicts = []
    simple_verdicts = []

    for q in queries:
        try:
            auto_resp = autonomous_rag_fn(q)
            auto_verdicts.append(getattr(auto_resp, "verdict", Verdict.ACCEPT))
        except Exception:
            auto_verdicts.append(Verdict.REJECT)

        try:
            simple_resp = simple_rag_fn(q)
            simple_verdicts.append(getattr(simple_resp, "verdict", Verdict.ACCEPT))
        except Exception:
            simple_verdicts.append(Verdict.REJECT)

    auto_hr = compute_hallucination_rate(auto_verdicts)
    simple_hr = compute_hallucination_rate(simple_verdicts)
    improvement = (simple_hr - auto_hr) / simple_hr if simple_hr > 0 else 0

    return {
        "autonomous_rag": {
            "hallucination_rate": auto_hr,
            "total_queries": len(queries),
        },
        "simple_rag": {
            "hallucination_rate": simple_hr,
            "total_queries": len(queries),
        },
        "improvement_pct": improvement * 100,
    }
