from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Session:
    """
    Tracks one oversight run of the DIND stack.

    This is intentionally simple:
    - counts how many events were ingested
    - counts how many claims and alerts were created
    - records timestamps for basic auditing
    """

    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    events_ingested: int = 0
    claims_created: int = 0
    alerts_created: int = 0
    topics_seen: int = 0

    def finish(self) -> None:
        self.ended_at = datetime.utcnow()


class SessionContext:
    """
    Context manager helper for running a DIND session.

    Example:
        with SessionContext() as session:
            ... do work and mutate session ...
    """

    def __init__(self) -> None:
        self.session = Session()

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.session.finish()
