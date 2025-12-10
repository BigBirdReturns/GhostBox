from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import db as dind_db
from .tension import compute_tension
from .session import Session


def compute_and_alert_for_topic(
    topic: str,
    db_path: str | Path = "ghost.db",
    session: Optional[Session] = None,
) -> float:
    """
    Compute tension for a topic and create an alert if needed.

    Returns the tension value.
    """
    claims = dind_db.list_claims_by_topic(topic, db_path=db_path)
    tension = compute_tension(claims)

    if session is not None:
        session.topics_seen += 1
        session.claims_created += len(claims)

    if claims and tension >= 3.0:
        first = claims[0]
        dind_db.create_alert(
            claim_id=first.id,
            topic=topic,
            tension=tension,
            db_path=db_path,
        )
        if session is not None:
            session.alerts_created += 1
    return tension


def main() -> None:
    db_path = "ghost.db"
    print("Initializing database (if needed)...")
    dind_db.init_db(db_path=db_path)

    topics = dind_db.list_topics(db_path=db_path)
    if not topics:
        print("No topics found. You probably want to run digital-independence-demo first.")
        return

    # Local session for this CLI run (not exported)
    from .session import Session
    session = Session()

    for topic in topics:
        print(f"Computing tension for topic '{topic}'")
        tension = compute_and_alert_for_topic(topic, db_path=db_path, session=session)
        print(f"  Tension score: {tension}")
        if tension >= 3.0:
            print("  High tension detected. Alert created.")
        else:
            print("  Tension below threshold. No alert created.")

    from .metrics import summarize_session
    summary = summarize_session(session)
    print("\nSession summary:")
    print(f"  Claims created:      {summary.reversible_ops}")
    print(f"  Alerts created:      {summary.irreversible_ops}")
    print(f"  Alert ratio:         {summary.alert_ratio:.3f}")
    print(f"  Posture:             {summary.posture}")
