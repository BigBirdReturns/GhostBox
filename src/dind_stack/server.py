"""
GhostBox Server v0.7.0

FastAPI runtime for the Digital Independence stack.

New in v0.7:
- /attention - Get attention geometry map
- /session/* - Analyst session management
- /capture/* - Real capture source configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import db as dind_db
from .runtime import build_kg, build_topic_view, summarize
from .session import Session
from .tension import compute_tension

# Import new v0.7 modules
try:
    from screen_ghost.simulated import Event as GhostEvent
    from screen_ghost.attention import AttentionGeometry, AttentionMap
    from screen_ghost.session import SessionManager, AnalystSession, HandoffStatus
    from axiom_kg.core import ingest_events
    _GHOST_AVAILABLE = True
except Exception:
    GhostEvent = None
    AttentionGeometry = None
    SessionManager = None
    ingest_events = None
    _GHOST_AVAILABLE = False


DEFAULT_DB_PATH = Path("ghost.db")
SESSIONS_DB_PATH = Path("sessions.db")

app = FastAPI(
    title="GhostBox Runtime",
    version="0.7.0",
    description="Efficient human oversight for adversarial information environments"
)

# Global state
session = Session()
attention_engine = AttentionGeometry() if AttentionGeometry else None
session_manager = SessionManager(db_path=SESSIONS_DB_PATH) if SessionManager else None


# =============================================================================
# Request/Response Models
# =============================================================================

class EventIn(BaseModel):
    topic: str
    source: str
    text: str
    confidence: float


class NoteIn(BaseModel):
    note_type: str
    topic: str
    content: str
    priority: int = 0


class FocusIn(BaseModel):
    topic: str
    seconds: float
    attention_score: float = 0.5


class StartSessionIn(BaseModel):
    analyst_id: str
    target_description: str = ""
    inherit_from: Optional[str] = None


class EndSessionIn(BaseModel):
    summary: str
    to_analyst: Optional[str] = None


# =============================================================================
# Lifecycle
# =============================================================================

@app.on_event("startup")
def init_db() -> None:
    dind_db.init_db(db_path=DEFAULT_DB_PATH)


# =============================================================================
# Core Endpoints (v0.6)
# =============================================================================

@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Return current high level session summary."""
    return summarize(db_path=DEFAULT_DB_PATH, session=session)


@app.get("/topics")
def get_topics() -> List[Dict[str, Any]]:
    """List topics with their current tension scores."""
    topics = dind_db.list_topics(db_path=DEFAULT_DB_PATH)
    output: list[dict[str, Any]] = []
    for topic in topics:
        claims = dind_db.list_claims_by_topic(topic, db_path=DEFAULT_DB_PATH)
        tension = compute_tension(claims)
        output.append({"topic": topic, "tension": tension, "claims": len(claims)})
    return output


@app.get("/topic/{topic}")
def get_topic(topic: str) -> Dict[str, Any]:
    """Topic focused KG and tension view."""
    topics = dind_db.list_topics(db_path=DEFAULT_DB_PATH)
    if topic not in topics:
        raise HTTPException(status_code=404, detail="Unknown topic")
    return build_topic_view(topic=topic, db_path=DEFAULT_DB_PATH)


@app.get("/graph")
def get_graph() -> Dict[str, List[Dict[str, Any]]]:
    """Return a compact knowledge graph view over claims and alerts."""
    return build_kg(db_path=DEFAULT_DB_PATH)


@app.post("/event")
def post_event(ev: EventIn) -> Dict[str, Any]:
    """Ingest one event into the system."""
    global session
    
    # Store in DIND
    dind_db.insert_claim(
        topic=ev.topic,
        source=ev.source,
        text=ev.text,
        confidence=ev.confidence,
        db_path=DEFAULT_DB_PATH,
    )
    
    # Feed to attention engine
    if attention_engine and GhostEvent:
        event = GhostEvent(
            topic=ev.topic,
            source=ev.source,
            text=ev.text,
            confidence=ev.confidence,
        )
        attention_engine.ingest(event)
    
    session.events_ingested += 1
    return {"ok": True}


@app.post("/topic/{topic}/tension")
def recompute_tension(topic: str) -> Dict[str, Any]:
    """Recompute tension for one topic, create an alert if needed."""
    from .scribe import compute_and_alert_for_topic
    
    topics = dind_db.list_topics(db_path=DEFAULT_DB_PATH)
    if topic not in topics:
        raise HTTPException(status_code=404, detail="Unknown topic")
    
    score = compute_and_alert_for_topic(topic, db_path=DEFAULT_DB_PATH, session=session)
    return {"topic": topic, "tension": score}


# =============================================================================
# Attention Geometry Endpoints (v0.7)
# =============================================================================

@app.get("/attention")
def get_attention() -> Dict[str, Any]:
    """
    Get the current attention geometry map.
    
    Shows where analysts should focus based on:
    - Contradictions between sources
    - Velocity spikes
    - Convergence patterns
    - Confidence drops
    - Unexpected silence
    """
    if not attention_engine:
        return {"error": "Attention engine not available", "shapes": []}
    
    attention_map = attention_engine.compute()
    return attention_map.summary()


@app.get("/attention/critical")
def get_critical_attention(threshold: float = 0.7) -> Dict[str, Any]:
    """Get only critical attention shapes above threshold."""
    if not attention_engine:
        return {"error": "Attention engine not available", "shapes": []}
    
    attention_map = attention_engine.compute()
    critical = attention_map.critical(threshold)
    
    return {
        "threshold": threshold,
        "count": len(critical),
        "shapes": [
            {
                "type": s.shape_type.name,
                "topic": s.topic,
                "score": s.score,
                "description": s.description,
                "sources": s.sources,
            }
            for s in critical
        ]
    }


@app.get("/attention/topic/{topic}")
def get_attention_by_topic(topic: str) -> Dict[str, Any]:
    """Get attention shapes for a specific topic."""
    if not attention_engine:
        return {"error": "Attention engine not available", "shapes": []}
    
    attention_map = attention_engine.compute()
    shapes = attention_map.by_topic(topic)
    
    return {
        "topic": topic,
        "count": len(shapes),
        "shapes": [
            {
                "type": s.shape_type.name,
                "score": s.score,
                "description": s.description,
                "sources": s.sources,
            }
            for s in shapes
        ]
    }


# =============================================================================
# Analyst Session Endpoints (v0.7)
# =============================================================================

@app.post("/session/start")
def start_analyst_session(req: StartSessionIn) -> Dict[str, Any]:
    """
    Start a new analyst session.
    
    If inherit_from is provided, loads context from that session's handoff.
    """
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    analyst_session = session_manager.start_session(
        analyst_id=req.analyst_id,
        target_description=req.target_description,
        inherit_from=req.inherit_from,
    )
    
    return {
        "session_id": analyst_session.session_id,
        "analyst_id": analyst_session.analyst_id,
        "started_at": analyst_session.started_at.isoformat(),
        "inherited_notes": len(analyst_session.notes),
    }


@app.post("/session/{session_id}/note")
def add_session_note(session_id: str, note: NoteIn) -> Dict[str, Any]:
    """Add a note to an analyst session."""
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    analyst_session = session_manager.get_session(session_id)
    if not analyst_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    analyst_session.add_note(
        note_type=note.note_type,
        topic=note.topic,
        content=note.content,
        priority=note.priority,
    )
    session_manager._save_session(analyst_session)
    
    return {"ok": True, "notes_count": len(analyst_session.notes)}


@app.post("/session/{session_id}/focus")
def track_session_focus(session_id: str, focus: FocusIn) -> Dict[str, Any]:
    """Track analyst focus time on a topic."""
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    analyst_session = session_manager.get_session(session_id)
    if not analyst_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    analyst_session.track_focus(
        topic=focus.topic,
        seconds=focus.seconds,
        attention_score=focus.attention_score,
    )
    
    # Update attention state
    if attention_engine:
        analyst_session.update_attention(attention_engine.compute())
    
    session_manager._save_session(analyst_session)
    
    return {
        "ok": True,
        "topic": focus.topic,
        "total_seconds": analyst_session.focus_areas[focus.topic].time_spent_seconds,
    }


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """Get current state of an analyst session."""
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    analyst_session = session_manager.get_session(session_id)
    if not analyst_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return analyst_session.to_dict()


@app.post("/session/{session_id}/end")
def end_analyst_session(session_id: str, req: EndSessionIn) -> Dict[str, Any]:
    """
    End an analyst session and create handoff package.
    
    Returns the complete handoff package for the next analyst.
    """
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    try:
        handoff = session_manager.end_session(
            session_id=session_id,
            summary=req.summary,
            to_analyst=req.to_analyst,
        )
        return handoff
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/sessions")
def list_sessions(
    analyst_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List analyst sessions with optional filters."""
    if not session_manager:
        raise HTTPException(status_code=501, detail="Session manager not available")
    
    status_enum = None
    if status:
        try:
            status_enum = HandoffStatus[status.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    return session_manager.list_sessions(
        analyst_id=analyst_id,
        status=status_enum,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Convenience entrypoint for running the runtime with uvicorn."""
    import uvicorn
    uvicorn.run("dind_stack.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
