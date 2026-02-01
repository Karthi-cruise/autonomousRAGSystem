#!/usr/bin/env python3
"""Validate that all modules import correctly (no network required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("Validating imports...")
    from src.utils.config import load_config, get_project_root
    from src.utils.schema import DocumentMetadata, Verdict, RAGResponse
    from src.scoring.trust_scoring import TrustScorer
    from src.scoring.decay_model import DecayModel
    from src.evaluation.metrics import compute_hallucination_rate
    print("  config, schema, scoring, evaluation OK")

    # Config loads
    cfg = load_config()
    assert "embedding" in cfg
    print("  config/rag.yaml OK")

    # Trust & decay logic
    from datetime import datetime, timezone
    ts = TrustScorer(default_trust=0.5)
    dm = DecayModel(half_life_days=180)
    meta = DocumentMetadata(source="test", timestamp=datetime.now(timezone.utc), trust_score=0.7)
    assert 0 <= ts.get_trust_score(meta) <= 1
    assert 0 < dm.get_decay_factor(meta.timestamp) <= 1
    print("  trust_scoring, decay_model OK")

    # Metrics
    rate = compute_hallucination_rate([Verdict.ACCEPT, Verdict.REJECT])
    assert rate == 0.5
    print("  metrics OK")

    print("\nAll imports and core logic validated successfully.")

if __name__ == "__main__":
    main()
