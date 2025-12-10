from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .session import Session


PostureLabel = Literal["stable", "contested", "critical"]


@dataclass
class ReversibilitySummary:
    """
    Simple summary of one oversight session.

    This is not the full GhostBox spec. It is a small, concrete metric
    that can be computed from the current demo and extended later.
    """

    reversible_ops: int
    irreversible_ops: int
    total_ops: int
    alert_ratio: float
    posture: PostureLabel


def summarize_session(session: Session) -> ReversibilitySummary:
    """
    Derive a coarse posture flag from the session counts.

    For the demo we treat:
    - claim creations as reversible (you can revisit or retract them)
    - alert creations as irreversible (they change the operator's context)

    This keeps the math simple but gives you a visible signal that can
    later incorporate your richer metrics and fork logic.
    """
    reversible = max(0, session.claims_created)
    irreversible = max(0, session.alerts_created)
    total = reversible + irreversible

    if total == 0:
        alert_ratio = 0.0
    else:
        alert_ratio = round(irreversible / total, 3)

    if alert_ratio < 0.25:
        posture: PostureLabel = "stable"
    elif alert_ratio < 0.6:
        posture = "contested"
    else:
        posture = "critical"

    return ReversibilitySummary(
        reversible_ops=reversible,
        irreversible_ops=irreversible,
        total_ops=total,
        alert_ratio=alert_ratio,
        posture=posture,
    )
