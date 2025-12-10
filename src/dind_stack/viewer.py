from __future__ import annotations

from pathlib import Path

from . import db as dind_db


def show_alerts(
    db_path: str | Path = "ghost.db",
    min_tension: float = 3.0,
) -> None:
    alerts = dind_db.list_pending_alerts(min_tension=min_tension, db_path=db_path)

    if not alerts:
        print("No pending alerts at this tension threshold.")
        return

    print(f"{len(alerts)} alert(s) require review:\n")
    for alert in alerts:
        print(f"[ALERT #{alert.id}] Topic: {alert.topic}")
        print(f"  Tension: {alert.tension}")
        claims = dind_db.list_claims_by_topic(alert.topic, db_path=db_path)
        print("  Claims for this topic:")
        for c in claims:
            print(f"    - ({c.source}, {c.confidence:.2f}) {c.text}")
        print("")


def main() -> None:
    db_path = "ghost.db"
    print("Showing pending alerts...")
    show_alerts(db_path=db_path, min_tension=3.0)
