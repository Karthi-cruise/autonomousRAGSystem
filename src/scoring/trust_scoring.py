"""Source trust scoring - domain reputation, historical accuracy, citation frequency."""

from __future__ import annotations

from typing import Any

from src.utils.schema import DocumentMetadata


class TrustScorer:
    """
    Scores source reliability from:
    - Domain reputation (known trusted domains)
    - Historical accuracy (placeholder for learned scores)
    - Citation frequency (how often source is cited)
    - Manual seed trust (explicit overrides)
    Output: Trust score ∈ [0, 1]
    """
    
    def __init__(
        self,
        default_trust: float = 0.5,
        trusted_domains: list[str] | None = None,
        manual_overrides: dict[str, float] | None = None,
    ):
        self.default_trust = default_trust
        self.trusted_domains = set(d.lower() for d in (trusted_domains or []))
        self.manual_overrides = manual_overrides or {}
        self._citation_counts: dict[str, int] = {}
    
    def get_trust_score(self, metadata: DocumentMetadata) -> float:
        """Compute trust score for a document's metadata."""
        source = metadata.source.lower()
        
        # Manual override takes precedence
        for key, score in self.manual_overrides.items():
            if key.lower() in source or source in key.lower():
                return min(1.0, max(0.0, score))
        
        # Use pre-computed trust_score if available and non-default
        if metadata.trust_score != 0.5 and metadata.trust_score > 0:
            return min(1.0, max(0.0, metadata.trust_score))
        
        # Domain reputation
        domain_score = 0.5
        for td in self.trusted_domains:
            if td in source:
                domain_score = 0.9
                break
        
        # Citation frequency (normalized)
        citation_score = 0.5
        if source in self._citation_counts:
            count = self._citation_counts[source]
            citation_score = min(1.0, 0.5 + count * 0.05)
        
        # Weighted combination
        trust = 0.4 * domain_score + 0.3 * citation_score + 0.3 * self.default_trust
        return min(1.0, max(0.0, trust))
    
    def record_citation(self, source: str) -> None:
        """Record that a source was cited (for citation frequency scoring)."""
        s = source.lower()
        self._citation_counts[s] = self._citation_counts.get(s, 0) + 1
    
    def add_trusted_domain(self, domain: str) -> None:
        """Add a trusted domain."""
        self.trusted_domains.add(domain.lower())
    
    def set_manual_override(self, source_pattern: str, score: float) -> None:
        """Set manual trust override for sources matching pattern."""
        self.manual_overrides[source_pattern] = min(1.0, max(0.0, score))
    
    def explain_trust(self, metadata: DocumentMetadata) -> str:
        """Return human-readable explanation of trust score."""
        score = self.get_trust_score(metadata)
        reasons = []
        source = metadata.source.lower()
        
        for td in self.trusted_domains:
            if td in source:
                reasons.append(f"Domain '{td}' is in trusted list")
                break
        if metadata.trust_score != 0.5:
            reasons.append(f"Explicit trust score: {metadata.trust_score}")
        if source in self._citation_counts:
            reasons.append(f"Cited {self._citation_counts[source]} times")
        
        if not reasons:
            reasons.append("Using default trust score")
        
        return f"Trust score {score:.2f}: " + "; ".join(reasons)
