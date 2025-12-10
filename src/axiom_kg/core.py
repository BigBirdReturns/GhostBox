"""
Axiom KG core interface for the demo.

In the real system this would be a full semantic graph and reasoning layer.
For this integrated demo we keep it minimal but provide an optional adapter
into the real `axiom` package if it is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from screen_ghost.simulated import Event
from dind_stack import db as dind_db

try:  # optional import of your real Axiom-KG implementation
    from axiom.core import SemanticID, Node, Space  # type: ignore
    _AXIOM_AVAILABLE = True
except Exception:  # noqa: BLE001
    SemanticID = Node = Space = None  # type: ignore
    _AXIOM_AVAILABLE = False


def ingest_events(
    events: Iterable[Event],
    db_path: str | Path = "ghost.db",
    use_axiom: bool = True,
) -> Optional["Space"]:
    """
    Ingest events into the shared claim store and optionally into an Axiom Space.

    - Always writes each event as a claim into DIND's SQLite `ghost.db`.
    - If `use_axiom` is True and the `axiom` package is importable, it also:
        - constructs an in-memory `Space`
        - creates one `Node` per event with a simple SemanticID allocation
        - adds the node to the space

    Returns the Space instance if Axiom is available and `use_axiom` is True,
    otherwise returns None.
    """
    space: Optional["Space"] = None  # type: ignore

    if use_axiom and _AXIOM_AVAILABLE:
        space = Space()  # type: ignore

    # Simple instance counter for SemanticIDs
    instance = 0

    for e in events:
        instance += 1

        # Always persist to DIND claim store
        dind_db.insert_claim(
            topic=e.topic,
            source=e.source,
            text=e.text,
            confidence=e.confidence,
            db_path=db_path,
        )

        # Optionally populate Axiom space
        if space is not None and SemanticID is not None and Node is not None:  # type: ignore
            # For the demo we use a fixed coordinate band 01-01-01-XXXX.
            sid = SemanticID(major=1, type_=1, subtype=1, instance=instance)  # type: ignore
            node = Node(  # type: ignore
                id=sid,
                label=e.text,
                metadata={
                    "topic": e.topic,
                    "source": e.source,
                    "confidence": e.confidence,
                },
            )
            space.add(node)

    return space
