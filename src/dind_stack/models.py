from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Claim:
    id: Optional[int]
    topic: str
    source: str
    text: str
    confidence: float
    created_at: datetime


@dataclass
class Alert:
    id: Optional[int]
    claim_id: int
    topic: str
    tension: float
    status: str
    created_at: datetime
