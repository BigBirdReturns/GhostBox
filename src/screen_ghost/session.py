"""
Analyst Session Continuity for GhostBox v0.7

The problem: When analysts rotate off a target, the thread resets.
The next analyst starts from zero. Context is lost. Patterns are missed.

This module provides:
1. Session handoff protocol (what the outgoing analyst knows)
2. Thread persistence (state survives rotation)
3. Attention inheritance (where were they looking?)
4. Divergence detection (what changed since handoff?)

This is human-compatible oversight: the system supports analyst rotation
without forcing them to re-learn everything.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum, auto
import sqlite3

from .attention import AttentionMap, AttentionShape, ShapeType


class HandoffStatus(Enum):
    """Status of an analyst session."""
    ACTIVE = auto()      # Analyst is currently working
    PAUSED = auto()      # Temporarily away
    HANDOFF = auto()     # Preparing to transfer
    COMPLETED = auto()   # Session ended, handoff done
    ABANDONED = auto()   # Session ended without proper handoff


@dataclass
class AnalystNote:
    """A note left by an analyst for continuity."""
    analyst_id: str
    timestamp: datetime
    note_type: str  # "observation", "concern", "follow_up", "hypothesis"
    topic: str
    content: str
    priority: int = 0  # 0=low, 1=medium, 2=high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyst_id": self.analyst_id,
            "timestamp": self.timestamp.isoformat(),
            "note_type": self.note_type,
            "topic": self.topic,
            "content": self.content,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalystNote":
        return cls(
            analyst_id=d["analyst_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            note_type=d["note_type"],
            topic=d["topic"],
            content=d["content"],
            priority=d.get("priority", 0),
        )


@dataclass
class FocusArea:
    """A topic or region the analyst was focusing on."""
    topic: str
    attention_score: float
    time_spent_seconds: float
    last_viewed: datetime
    notes: List[AnalystNote] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "attention_score": self.attention_score,
            "time_spent_seconds": self.time_spent_seconds,
            "last_viewed": self.last_viewed.isoformat(),
            "notes": [n.to_dict() for n in self.notes],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FocusArea":
        return cls(
            topic=d["topic"],
            attention_score=d["attention_score"],
            time_spent_seconds=d["time_spent_seconds"],
            last_viewed=datetime.fromisoformat(d["last_viewed"]),
            notes=[AnalystNote.from_dict(n) for n in d.get("notes", [])],
        )


@dataclass
class AnalystSession:
    """
    Complete state of an analyst's oversight session.
    
    This is what gets handed off to the next analyst.
    """
    
    session_id: str
    analyst_id: str
    started_at: datetime
    status: HandoffStatus = HandoffStatus.ACTIVE
    ended_at: Optional[datetime] = None
    
    # What they were working on
    target_description: str = ""
    focus_areas: Dict[str, FocusArea] = field(default_factory=dict)
    
    # Their notes and observations
    notes: List[AnalystNote] = field(default_factory=list)
    
    # Attention state at last update
    last_attention_map: Optional[Dict[str, Any]] = None
    
    # Handoff metadata
    handoff_to: Optional[str] = None
    handoff_summary: Optional[str] = None
    
    # State hash for divergence detection
    state_hash: Optional[str] = None
    
    def add_note(
        self, 
        note_type: str, 
        topic: str, 
        content: str,
        priority: int = 0,
    ) -> None:
        """Add a note to the session."""
        note = AnalystNote(
            analyst_id=self.analyst_id,
            timestamp=datetime.utcnow(),
            note_type=note_type,
            topic=topic,
            content=content,
            priority=priority,
        )
        self.notes.append(note)
        
        if topic in self.focus_areas:
            self.focus_areas[topic].notes.append(note)
    
    def track_focus(self, topic: str, seconds: float, attention_score: float = 0.5) -> None:
        """Track time spent on a topic."""
        if topic not in self.focus_areas:
            self.focus_areas[topic] = FocusArea(
                topic=topic,
                attention_score=attention_score,
                time_spent_seconds=0,
                last_viewed=datetime.utcnow(),
            )
        
        self.focus_areas[topic].time_spent_seconds += seconds
        self.focus_areas[topic].last_viewed = datetime.utcnow()
        self.focus_areas[topic].attention_score = max(
            self.focus_areas[topic].attention_score,
            attention_score
        )
    
    def update_attention(self, attention_map: AttentionMap) -> None:
        """Update with latest attention geometry."""
        self.last_attention_map = attention_map.summary()
        self._update_hash()
    
    def _update_hash(self) -> None:
        """Update state hash for divergence detection."""
        state_str = json.dumps({
            "notes": len(self.notes),
            "focus_areas": list(self.focus_areas.keys()),
            "attention": self.last_attention_map,
        }, sort_keys=True)
        self.state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def prepare_handoff(self, summary: str, to_analyst: Optional[str] = None) -> Dict[str, Any]:
        """Prepare handoff package for next analyst."""
        self.status = HandoffStatus.HANDOFF
        self.handoff_summary = summary
        self.handoff_to = to_analyst
        
        return self.to_handoff_package()
    
    def complete_handoff(self) -> None:
        """Mark session as completed."""
        self.status = HandoffStatus.COMPLETED
        self.ended_at = datetime.utcnow()
    
    def to_handoff_package(self) -> Dict[str, Any]:
        """
        Generate the handoff package - everything the next analyst needs.
        
        This is the core of session continuity.
        """
        # Sort notes by priority and recency
        priority_notes = sorted(
            self.notes, 
            key=lambda n: (n.priority, n.timestamp),
            reverse=True
        )[:10]  # Top 10 most important
        
        # Sort focus areas by attention and time
        top_focus = sorted(
            self.focus_areas.values(),
            key=lambda f: (f.attention_score, f.time_spent_seconds),
            reverse=True
        )[:5]  # Top 5 areas
        
        return {
            "session_id": self.session_id,
            "from_analyst": self.analyst_id,
            "to_analyst": self.handoff_to,
            "duration_hours": (
                (self.ended_at or datetime.utcnow()) - self.started_at
            ).total_seconds() / 3600,
            
            "summary": self.handoff_summary,
            "target": self.target_description,
            
            "priority_notes": [n.to_dict() for n in priority_notes],
            "top_focus_areas": [f.to_dict() for f in top_focus],
            
            "attention_state": self.last_attention_map,
            "state_hash": self.state_hash,
            
            "handoff_at": datetime.utcnow().isoformat(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Full serialization for persistence."""
        return {
            "session_id": self.session_id,
            "analyst_id": self.analyst_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status.name,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "target_description": self.target_description,
            "focus_areas": {k: v.to_dict() for k, v in self.focus_areas.items()},
            "notes": [n.to_dict() for n in self.notes],
            "last_attention_map": self.last_attention_map,
            "handoff_to": self.handoff_to,
            "handoff_summary": self.handoff_summary,
            "state_hash": self.state_hash,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalystSession":
        """Deserialize from storage."""
        session = cls(
            session_id=d["session_id"],
            analyst_id=d["analyst_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
            status=HandoffStatus[d["status"]],
            target_description=d.get("target_description", ""),
        )
        
        if d.get("ended_at"):
            session.ended_at = datetime.fromisoformat(d["ended_at"])
        
        session.focus_areas = {
            k: FocusArea.from_dict(v) 
            for k, v in d.get("focus_areas", {}).items()
        }
        session.notes = [
            AnalystNote.from_dict(n) 
            for n in d.get("notes", [])
        ]
        session.last_attention_map = d.get("last_attention_map")
        session.handoff_to = d.get("handoff_to")
        session.handoff_summary = d.get("handoff_summary")
        session.state_hash = d.get("state_hash")
        
        return session


# =============================================================================
# Session Manager
# =============================================================================

@dataclass
class SessionManager:
    """
    Manages analyst sessions with persistence and handoff protocol.
    
    This is the central coordinator for session continuity.
    """
    
    db_path: Path = Path("sessions.db")
    active_sessions: Dict[str, AnalystSession] = field(default_factory=dict)
    
    def __post_init__(self):
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite storage."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                analyst_id TEXT,
                status TEXT,
                data JSON,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS handoffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_session TEXT,
                to_session TEXT,
                package JSON,
                created_at TEXT,
                FOREIGN KEY (from_session) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()
        conn.close()
    
    def start_session(
        self, 
        analyst_id: str, 
        target_description: str = "",
        inherit_from: Optional[str] = None,
    ) -> AnalystSession:
        """
        Start a new analyst session.
        
        If inherit_from is provided, loads the handoff package from that session.
        """
        session_id = f"{analyst_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        session = AnalystSession(
            session_id=session_id,
            analyst_id=analyst_id,
            started_at=datetime.utcnow(),
            target_description=target_description,
        )
        
        # Inherit from previous session if specified
        if inherit_from:
            handoff = self._load_handoff(inherit_from)
            if handoff:
                session.target_description = handoff.get("target", target_description)
                # Import notes as "inherited"
                for note_dict in handoff.get("priority_notes", []):
                    note = AnalystNote.from_dict(note_dict)
                    note.note_type = f"inherited_{note.note_type}"
                    session.notes.append(note)
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        return session
    
    def end_session(
        self, 
        session_id: str, 
        summary: str,
        to_analyst: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End a session and create handoff package.
        
        Returns the handoff package.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            session = self._load_session(session_id)
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Prepare handoff
        package = session.prepare_handoff(summary, to_analyst)
        session.complete_handoff()
        
        # Save everything
        self._save_session(session)
        self._save_handoff(session_id, package)
        
        # Remove from active
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        return package
    
    def get_session(self, session_id: str) -> Optional[AnalystSession]:
        """Get a session by ID."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        return self._load_session(session_id)
    
    def list_sessions(
        self, 
        analyst_id: Optional[str] = None,
        status: Optional[HandoffStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT session_id, analyst_id, status, created_at FROM sessions WHERE 1=1"
        params: List[Any] = []
        
        if analyst_id:
            query += " AND analyst_id = ?"
            params.append(analyst_id)
        if status:
            query += " AND status = ?"
            params.append(status.name)
        
        query += " ORDER BY created_at DESC"
        
        cursor = conn.execute(query, params)
        results = [dict(row) for row in cursor]
        conn.close()
        
        return results
    
    def check_divergence(self, session_id: str, current_hash: str) -> bool:
        """
        Check if state has diverged since last handoff.
        
        Returns True if divergence detected.
        """
        session = self.get_session(session_id)
        if not session or not session.state_hash:
            return False
        return session.state_hash != current_hash
    
    def _save_session(self, session: AnalystSession) -> None:
        """Persist session to SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, analyst_id, status, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.analyst_id,
            session.status.name,
            json.dumps(session.to_dict()),
            session.started_at.isoformat(),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
    
    def _load_session(self, session_id: str) -> Optional[AnalystSession]:
        """Load session from SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return AnalystSession.from_dict(json.loads(row["data"]))
    
    def _save_handoff(self, from_session: str, package: Dict[str, Any]) -> None:
        """Save handoff package."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO handoffs (from_session, to_session, package, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            from_session,
            package.get("to_analyst"),
            json.dumps(package),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
    
    def _load_handoff(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the most recent handoff from a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT package FROM handoffs 
            WHERE from_session = ?
            ORDER BY created_at DESC LIMIT 1
        """, (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return json.loads(row["package"])


# =============================================================================
# Demo
# =============================================================================

def demo_session_continuity() -> None:
    """Demonstrate analyst session handoff."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  ANALYST SESSION CONTINUITY DEMO                                  ║")
    print("║                                                                   ║")
    print("║  Thread doesn't reset on rotation.                                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Use temp path for demo
    manager = SessionManager(db_path=Path("/tmp/demo_sessions.db"))
    
    # Analyst A starts a session
    print("ANALYST A starts session")
    print("-" * 60)
    session_a = manager.start_session(
        analyst_id="analyst_a",
        target_description="Monitor election integrity feeds"
    )
    
    # Analyst A works for a while
    session_a.track_focus("election_integrity", seconds=1800, attention_score=0.8)
    session_a.track_focus("voting_machines", seconds=600, attention_score=0.6)
    
    session_a.add_note(
        note_type="observation",
        topic="election_integrity",
        content="Feed B consistently contradicts Feed A on remote access claims",
        priority=2,
    )
    session_a.add_note(
        note_type="hypothesis",
        topic="voting_machines",
        content="Possible coordinated narrative - check source ownership",
        priority=1,
    )
    session_a.add_note(
        note_type="follow_up",
        topic="election_integrity",
        content="Request raw audit logs from county registrar",
        priority=2,
    )
    
    print(f"  Session ID: {session_a.session_id}")
    print(f"  Focus areas: {list(session_a.focus_areas.keys())}")
    print(f"  Notes: {len(session_a.notes)}")
    print()
    
    # Analyst A hands off
    print("ANALYST A prepares handoff")
    print("-" * 60)
    handoff = manager.end_session(
        session_id=session_a.session_id,
        summary="Primary contradiction between Feed A and Feed B on remote access. "
                "High priority: verify claims with county registrar audit logs. "
                "Possible coordinated narrative on voting machines topic.",
        to_analyst="analyst_b",
    )
    
    print(f"  Summary: {handoff['summary'][:80]}...")
    print(f"  Priority notes: {len(handoff['priority_notes'])}")
    print(f"  Top focus areas: {len(handoff['top_focus_areas'])}")
    print()
    
    # Analyst B picks up
    print("ANALYST B starts session (inheriting from A)")
    print("-" * 60)
    session_b = manager.start_session(
        analyst_id="analyst_b",
        inherit_from=session_a.session_id,
    )
    
    print(f"  Session ID: {session_b.session_id}")
    print(f"  Inherited target: {session_b.target_description}")
    print(f"  Inherited notes: {len(session_b.notes)}")
    print()
    
    # Show what Analyst B sees
    print("WHAT ANALYST B SEES:")
    print("-" * 60)
    print()
    print("Inherited notes:")
    for note in session_b.notes:
        print(f"  [{note.priority}] {note.note_type}: {note.content[:60]}...")
    print()
    
    print("Handoff package summary:")
    print(json.dumps({
        "from": handoff["from_analyst"],
        "to": handoff["to_analyst"],
        "duration_hours": f"{handoff['duration_hours']:.1f}",
        "summary": handoff["summary"][:100] + "...",
    }, indent=2))


if __name__ == "__main__":
    demo_session_continuity()
