"""
Screen source abstraction.

This module defines a small interface that the DIND stack uses to obtain
events from an adversarial environment. The default implementation uses
the local simulated events. You can add a real ScreenGhost-backed source
that reads from your own logs or live runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from .simulated import Event, generate_demo_events


class ScreenSource(Protocol):
    def generate_events(self) -> List[Event]:
        """Return a batch of events for downstream processing."""
        raise NotImplementedError


@dataclass
class SimulatedScreenSource(ScreenSource):
    def generate_events(self) -> List[Event]:
        return generate_demo_events()


# Placeholder for a real ScreenGhost source. This keeps the interface clear
# without forcing any heavy dependencies at import time.
class ScreenGhostSource(ScreenSource):
    """
    Example adapter skeleton for your real ScreenGhost runs.

    You can implement this to:
    - read from the SQLite log that screenghost.py already writes
    - or stream events from a live automation run

    For now it raises NotImplementedError so the demo cannot accidentally
    depend on it.
    """

    def __init__(self, db_path: str = "screenghost.db") -> None:
        self.db_path = db_path

    def generate_events(self) -> List[Event]:
        raise NotImplementedError(
            "ScreenGhostSource.generate_events is a placeholder. "
            "Wire this to your real screenghost.py logs or live runs."
        )
