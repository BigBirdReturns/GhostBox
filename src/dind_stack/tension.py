from __future__ import annotations

from typing import Iterable

from .models import Claim


def compute_tension(claims: Iterable[Claim]) -> float:
    """
    Simple tension metric for a set of claims about one topic.

    Higher tension when:
    - There are more distinct sources
    - Average confidence is lower
    """
    claims = list(claims)
    if not claims:
        return 0.0

    avg_conf = sum(c.confidence for c in claims) / len(claims)
    sources = {c.source for c in claims}
    source_factor = min(2.0, max(0.0, len(sources) - 1) * 0.5)
    base = max(0.0, 3.0 - avg_conf * 2.0)

    return round(base + source_factor, 2)
