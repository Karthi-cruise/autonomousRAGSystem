"""Temporal knowledge decay - older docs lose relevance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.utils.schema import DocumentMetadata


class DecayModel:
    """
    Applies temporal decay:
    EffectiveScore = RetrievalScore × TrustScore × DecayFactor(time)
    Prevents outdated policies, old APIs, deprecated info from dominating.
    """
    
    def __init__(self, half_life_days: int = 180, min_decay_factor: float = 0.1):
        self.half_life_days = half_life_days
        self.min_decay_factor = min_decay_factor
    
    def get_decay_factor(self, timestamp: datetime | str) -> float:
        """Compute decay factor. 1.0 for recent, decreases for older docs."""
        if isinstance(timestamp, str):
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return 0.5  # Unknown date -> neutral
        else:
            ts = timestamp
        
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - ts
        days_old = delta.total_seconds() / 86400
        
        # Exponential decay: factor = 0.5^(days/half_life)
        import math
        factor = 0.5 ** (days_old / self.half_life_days)
        return max(self.min_decay_factor, min(1.0, factor))
    
    def explain_decay(self, metadata: DocumentMetadata) -> str:
        """Return human-readable explanation of decay."""
        factor = self.get_decay_factor(metadata.timestamp)
        ts = metadata.timestamp
        if isinstance(ts, str):
            ts_str = ts
        else:
            ts_str = ts.isoformat()
        
        if factor >= 0.9:
            return f"Document is recent ({ts_str}), decay factor: {factor:.2f} (minimal decay)"
        elif factor >= 0.5:
            return f"Document age reduces relevance ({ts_str}), decay factor: {factor:.2f}"
        else:
            return f"Document is outdated ({ts_str}), decay factor: {factor:.2f} (significant decay)"
