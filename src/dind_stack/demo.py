"""
Full chain demo for the integrated Digital Independence stack.

Flow:
1. Initialize the database.
2. Generate demo events via ScreenGhost simulated layer.
3. Ingest events into the shared claim store through Axiom KG (and optionally Axiom Space).
4. Compute tension and create alerts.
5. Show high tension alerts for human review.
6. Print a session-level reversibility summary.
"""

from __future__ import annotations

from pathlib import Path

from screen_ghost.source import SimulatedScreenSource
from axiom_kg.core import ingest_events
from . import db as dind_db
from .scribe import compute_and_alert_for_topic
from .viewer import show_alerts
from .session import SessionContext
from .metrics import summarize_session


def main() -> None:
    db_path: str | Path = "ghost.db"

    with SessionContext() as session:
        print("Step 1: Initialize database")
        dind_db.init_db(db_path=db_path)

        print("Step 2: Generate demo events from ScreenGhost (simulated)")
        source = SimulatedScreenSource()
        events = source.generate_events()
        session.events_ingested += len(events)
        print(f"  Generated {len(events)} events.")

        print("Step 3: Ingest events into Axiom KG adapter + DIND claim store")
        space = ingest_events(events, db_path=db_path, use_axiom=True)
        if space is not None:
            print("  Axiom Space integration active (nodes added).")
        else:
            print("  Axiom package not available or integration disabled.")

        print("Step 4: Compute tension and create alerts")
        topics = sorted({e.topic for e in events})
        for topic in topics:
            tension = compute_and_alert_for_topic(topic, db_path=db_path, session=session)
            print(f"  Topic '{topic}' tension score: {tension}")

        print("Step 5: Show alerts to human overseer")
        show_alerts(db_path=db_path, min_tension=3.0)

        summary = summarize_session(session)
        print("\nStep 6: Session summary")
        print(f"  Events ingested:     {session.events_ingested}")
        print(f"  Claims created:      {summary.reversible_ops}")
        print(f"  Alerts created:      {summary.irreversible_ops}")
        print(f"  Alert ratio:         {summary.alert_ratio:.3f}")
        print(f"  Posture:             {summary.posture}")


if __name__ == "__main__":
    main()
