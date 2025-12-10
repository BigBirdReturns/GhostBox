from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List

from .models import Claim, Alert


DEFAULT_DB_PATH = Path("ghost.db")


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            tension REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (claim_id) REFERENCES claims (id)
        )
        """
    )

    conn.commit()
    conn.close()


def insert_claim(
    topic: str,
    source: str,
    text: str,
    confidence: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO claims (topic, source, text, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (topic, source, text, confidence, datetime.utcnow().isoformat()),
    )
    conn.commit()
    claim_id = cur.lastrowid
    conn.close()
    return int(claim_id)


def list_claims_by_topic(topic: str, db_path: str | Path = DEFAULT_DB_PATH) -> List[Claim]:
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM claims WHERE topic = ? ORDER BY created_at ASC",
        (topic,),
    )
    rows = cur.fetchall()
    conn.close()

    claims: List[Claim] = []
    for row in rows:
        claims.append(
            Claim(
                id=row["id"],
                topic=row["topic"],
                source=row["source"],
                text=row["text"],
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        )
    return claims


def list_topics(db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT topic FROM claims ORDER BY topic ASC")
    rows = cur.fetchall()
    conn.close()
    return [row["topic"] for row in rows]


def create_alert(
    claim_id: int,
    topic: str,
    tension: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alerts (claim_id, topic, tension, status, created_at)
        VALUES (?, ?, ?, 'pending', ?)
        """,
        (claim_id, topic, tension, datetime.utcnow().isoformat()),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return int(alert_id)


def list_pending_alerts(
    min_tension: float = 0.0,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[Alert]:
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM alerts
        WHERE status = 'pending' AND tension >= ?
        ORDER BY tension DESC, created_at DESC
        """,
        (min_tension,),
    )
    rows = cur.fetchall()
    conn.close()

    alerts: list[Alert] = []
    for row in rows:
        alerts.append(
            Alert(
                id=row["id"],
                claim_id=row["claim_id"],
                topic=row["topic"],
                tension=row["tension"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        )
    return alerts
