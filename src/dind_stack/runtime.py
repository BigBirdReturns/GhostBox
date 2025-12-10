
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from . import db as dind_db
from .metrics import summarize_session
from .session import Session


@dataclass
class KGNode:
    id: str
    kind: str
    topic: str | None
    source: str | None
    label: str
    confidence: float | None


@dataclass
class KGEdge:
    id: str
    kind: str
    src: str
    dst: str


def build_kg(db_path: str | Path = "ghost.db") -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a simple knowledge graph view from the DIND SQLite store.

    Nodes:
      - one node per claim (kind="claim")
      - one node per alert (kind="alert")

    Edges:
      - "alert_for" edges from alert -> claim
    """
    topics = dind_db.list_topics(db_path=db_path)
    pending_alerts = dind_db.list_pending_alerts(db_path=db_path)

    nodes: list[KGNode] = []
    edges: list[KGEdge] = []

    # Index alerts by claim id for quick linking
    alerts_by_claim: dict[int, list[int]] = {}
    for alert in pending_alerts:
        alerts_by_claim.setdefault(alert.claim_id, []).append(alert.id)

    seen_node_ids: set[str] = set()

    for topic in topics:
        claims = dind_db.list_claims_by_topic(topic, db_path=db_path)
        for claim in claims:
            node_id = f"claim:{claim.id}"
            if node_id not in seen_node_ids:
                nodes.append(
                    KGNode(
                        id=node_id,
                        kind="claim",
                        topic=claim.topic,
                        source=claim.source,
                        label=claim.text,
                        confidence=claim.confidence,
                    )
                )
                seen_node_ids.add(node_id)

            for alert_id in alerts_by_claim.get(claim.id or -1, []):
                alert_node_id = f"alert:{alert_id}"
                if alert_node_id not in seen_node_ids:
                    # We do not need full alert metadata here, only linkage
                    nodes.append(
                        KGNode(
                            id=alert_node_id,
                            kind="alert",
                            topic=claim.topic,
                            source=None,
                            label=f"Alert for topic {claim.topic}",
                            confidence=None,
                        )
                    )
                    seen_node_ids.add(alert_node_id)

                edges.append(
                    KGEdge(
                        id=f"alert_for:{alert_id}:{claim.id}",
                        kind="alert_for",
                        src=alert_node_id,
                        dst=node_id,
                    )
                )

    return {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }


def build_topic_view(
    topic: str,
    db_path: str | Path = "ghost.db",
) -> Dict[str, Any]:
    """
    Build a topic focused view including claims, alerts, and a tension score.
    """
    claims = dind_db.list_claims_by_topic(topic, db_path=db_path)
    pending_alerts = dind_db.list_pending_alerts(db_path=db_path)
    alerts = [a for a in pending_alerts if a.topic == topic]

    from .tension import compute_tension

    tension = compute_tension(claims)

    claim_dicts = [
        {
            "id": c.id,
            "source": c.source,
            "text": c.text,
            "confidence": c.confidence,
            "created_at": c.created_at.isoformat(),
        }
        for c in claims
    ]
    alert_dicts = [
        {
            "id": a.id,
            "claim_id": a.claim_id,
            "tension": a.tension,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]

    return {
        "topic": topic,
        "tension": tension,
        "claims": claim_dicts,
        "alerts": alert_dicts,
    }


def summarize(db_path: str | Path = "ghost.db", session: Session | None = None) -> Dict[str, Any]:
    """
    Return a compact summary view that mirrors the CLI metrics output.
    """
    if session is None:
        session = Session()

    summary = summarize_session(session)
    return {
        "events_ingested": session.events_ingested,
        "claims_created": summary.reversible_ops,
        "alerts_created": summary.irreversible_ops,
        "alert_ratio": summary.alert_ratio,
        "posture": summary.posture,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
    }
