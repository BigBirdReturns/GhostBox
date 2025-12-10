"""
GhostBox Integration Layer v0.8.1

This module wires together:
- Screen Ghost (capture sources)
- Axiom-KG (semantic coordinates + 9 adapters)
- DIND Stack (tension detection)
- Attention Geometry (where to look)
- Session Continuity (thread doesn't reset)
- PLTR Analyzer (Field Zero domain)

The flow:
  Real data → Adapter → Semantic coordinates → Space → Tension → Attention shapes

This is Field Zero ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

# Axiom-KG core
from axiom.core import (
    Space, 
    Node, 
    SemanticID, 
    RelationType, 
    Fork,
    DeterministicWrapper,
    Strategy,
)

# Axiom-KG adapters
from axiom.adapters import (
    RSSAdapter,
    ICalAdapter,
    SchemaOrgAdapter,
    OpenAPIAdapter,
    PackageAdapter,
    XBRLAdapter,
)

# PLTR Field Zero domain
from ghostbox_pltr import PLTRAnalyzer

# Screen Ghost
from screen_ghost.simulated import Event
from screen_ghost.capture import (
    CaptureSource,
    JSONFileSource,
    CSVFileSource,
    SQLiteSource,
    MultiplexSource,
)
from screen_ghost.attention import (
    AttentionGeometry,
    AttentionMap,
    AttentionShape,
    ShapeType,
)
from screen_ghost.session import (
    SessionManager,
    AnalystSession,
    HandoffStatus,
)

# DIND Stack
from dind_stack.models import Claim, Alert
from dind_stack import db as dind_db


# =============================================================================
# SEMANTIC TENSION: Tension computed from coordinates, not formulas
# =============================================================================

@dataclass
class SemanticTension:
    """
    Tension computed from semantic coordinates.
    
    This replaces the simple formula in DIND v0.6 with
    actual geometric reasoning from Axiom-KG.
    """
    
    topic: str
    tension_score: float  # 0.0 = aligned, 1.0 = maximum disagreement
    
    # What's driving the tension
    fork_count: int = 0
    contradiction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    semantic_drift: float = 0.0
    source_divergence: float = 0.0
    
    # Coordinates involved
    node_codes: List[str] = field(default_factory=list)
    centroid: Optional[str] = None
    spread: float = 0.0  # How far nodes are from centroid
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "tension_score": self.tension_score,
            "fork_count": self.fork_count,
            "contradiction_pairs": self.contradiction_pairs,
            "semantic_drift": self.semantic_drift,
            "source_divergence": self.source_divergence,
            "node_count": len(self.node_codes),
            "spread": self.spread,
        }


def compute_semantic_tension(
    space: Space, 
    topic: str,
    nodes: List[Node],
) -> SemanticTension:
    """
    Compute tension from semantic coordinates.
    
    Tension increases when:
    - Nodes are far apart in coordinate space (spread)
    - Explicit CONTRADICTS relations exist
    - Forks exist on the topic
    - Sources diverge (same topic, different coordinates)
    """
    if not nodes:
        return SemanticTension(topic=topic, tension_score=0.0)
    
    # 1. Compute spread (average pairwise distance)
    total_distance = 0
    pair_count = 0
    contradiction_pairs = []
    
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            dist = a.id.distance(b.id)
            total_distance += dist
            pair_count += 1
            
            # Check for explicit contradictions
            if RelationType.CONTRADICTS in a.relations:
                if b.id.code in a.relations[RelationType.CONTRADICTS]:
                    contradiction_pairs.append((a.label[:50], b.label[:50]))
    
    avg_distance = total_distance / pair_count if pair_count > 0 else 0
    spread = avg_distance / 4.0  # Normalize to 0-1 (max distance is 4)
    
    # 2. Count forks on this topic
    fork_count = 0
    for fork in space.forks():
        if topic.lower() in fork.label.lower():
            fork_count += 1
    
    # 3. Source divergence (different sources, same topic, different coordinates)
    sources = {}
    for node in nodes:
        source = node.metadata.get("source") or node.metadata.get("feed_title") or "unknown"
        if source not in sources:
            sources[source] = []
        sources[source].append(node.id.code)
    
    source_divergence = 0.0
    if len(sources) > 1:
        # Check if sources cluster differently
        source_list = list(sources.values())
        for i, codes_a in enumerate(source_list):
            for codes_b in source_list[i+1:]:
                # Compare first node from each source
                if codes_a and codes_b:
                    id_a = SemanticID.parse(codes_a[0])
                    id_b = SemanticID.parse(codes_b[0])
                    source_divergence += id_a.distance(id_b) / 4.0
        
        source_divergence = source_divergence / (len(sources) * (len(sources) - 1) / 2)
    
    # 4. Combine into tension score
    tension_score = (
        spread * 0.3 +
        min(1.0, fork_count * 0.2) +
        min(1.0, len(contradiction_pairs) * 0.25) +
        source_divergence * 0.25
    )
    
    return SemanticTension(
        topic=topic,
        tension_score=min(1.0, tension_score),
        fork_count=fork_count,
        contradiction_pairs=contradiction_pairs,
        semantic_drift=spread,
        source_divergence=source_divergence,
        node_codes=[n.id.code for n in nodes],
        spread=spread,
    )


# =============================================================================
# GHOSTBOX ENGINE: The unified runtime
# =============================================================================

@dataclass
class GhostBoxEngine:
    """
    The unified GhostBox runtime.
    
    Integrates:
    - Capture sources → Events
    - Axiom adapters → Semantic coordinates
    - Space → Derivation and tension
    - Attention geometry → Where to look
    - Session management → Analyst continuity
    """
    
    # Core state
    space: Space = field(default_factory=Space)
    attention: AttentionGeometry = field(default_factory=AttentionGeometry)
    sessions: SessionManager = field(default_factory=lambda: SessionManager(Path("ghostbox_sessions.db")))
    
    # Adapters (lazy loaded)
    _rss_adapter: Optional[RSSAdapter] = None
    _ical_adapter: Optional[ICalAdapter] = None
    _schemaorg_adapter: Optional[SchemaOrgAdapter] = None
    
    # Sources
    sources: MultiplexSource = field(default_factory=MultiplexSource)
    
    # Metrics
    events_ingested: int = 0
    nodes_created: int = 0
    alerts_generated: int = 0
    
    # Configuration
    db_path: Path = Path("ghostbox.db")
    tension_threshold: float = 0.6
    
    def __post_init__(self):
        # Initialize DIND database
        dind_db.init_db(db_path=self.db_path)
    
    # -------------------------------------------------------------------------
    # Adapters
    # -------------------------------------------------------------------------
    
    @property
    def rss(self) -> RSSAdapter:
        if self._rss_adapter is None:
            self._rss_adapter = RSSAdapter(space=self.space)
        return self._rss_adapter
    
    @property
    def ical(self) -> ICalAdapter:
        if self._ical_adapter is None:
            self._ical_adapter = ICalAdapter(space=self.space)
        return self._ical_adapter
    
    @property
    def schemaorg(self) -> SchemaOrgAdapter:
        if self._schemaorg_adapter is None:
            self._schemaorg_adapter = SchemaOrgAdapter(space=self.space)
        return self._schemaorg_adapter
    
    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------
    
    def ingest_event(self, event: Event) -> Node:
        """
        Ingest a single event into the system.
        
        1. Create claim in DIND
        2. Create node in Axiom space
        3. Feed to attention geometry
        """
        # Store in DIND
        dind_db.insert_claim(
            topic=event.topic,
            source=event.source,
            text=event.text,
            confidence=event.confidence,
            db_path=self.db_path,
        )
        
        # Create semantic node
        # Events are Entities (Major 1), type based on topic hash
        type_ = (hash(event.topic) % 99) + 1
        subtype = (hash(event.text[:50]) % 99) + 1
        
        sem_id = SemanticID.create(
            major=1,
            type_=type_,
            subtype=subtype,
            instance=self.nodes_created + 1,
        )
        
        node = Node(
            id=sem_id,
            label=event.text[:100],
            metadata={
                "topic": event.topic,
                "source": event.source,
                "confidence": event.confidence,
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )
        
        self.space.add(node)
        self.nodes_created += 1
        
        # Feed to attention
        self.attention.ingest(event)
        
        self.events_ingested += 1
        
        return node
    
    def ingest_events(self, events: List[Event]) -> List[Node]:
        """Ingest multiple events."""
        return [self.ingest_event(e) for e in events]
    
    def ingest_rss(self, feed_url: str) -> List[Node]:
        """Ingest RSS feed directly into the space."""
        nodes = self.rss.parse_url(feed_url)
        
        for node in nodes:
            if self.space.get(node.id) is None:
                self.space.add(node)
                self.nodes_created += 1
                
                # Also feed to attention as event
                event = Event(
                    topic=node.metadata.get("categories", ["news"])[0] if node.metadata.get("categories") else "news",
                    source=node.metadata.get("feed_title", "rss"),
                    text=node.label,
                    confidence=0.7,
                )
                self.attention.ingest(event)
                self.events_ingested += 1
        
        return nodes
    
    def ingest_calendar(self, calendar_path: str) -> Tuple[List[Node], List[Dict]]:
        """Ingest iCal and return nodes + conflicts."""
        nodes = self.ical.parse(calendar_path)
        
        for node in nodes:
            if self.space.get(node.id) is None:
                self.space.add(node)
                self.nodes_created += 1
        
        conflicts = self.ical.find_conflicts(nodes)
        
        return nodes, conflicts
    
    def ingest_from_sources(self) -> List[Node]:
        """Ingest from all configured capture sources."""
        nodes = []
        
        for event in self.sources.events():
            node = self.ingest_event(event)
            nodes.append(node)
        
        return nodes
    
    # -------------------------------------------------------------------------
    # Tension
    # -------------------------------------------------------------------------
    
    def compute_tension(self, topic: str) -> SemanticTension:
        """Compute semantic tension for a topic."""
        # Find all nodes related to this topic
        topic_nodes = []
        for node in self.space.nodes():
            node_topic = node.metadata.get("topic", "")
            if topic.lower() in node_topic.lower() or topic.lower() in node.label.lower():
                topic_nodes.append(node)
        
        return compute_semantic_tension(self.space, topic, topic_nodes)
    
    def compute_all_tensions(self) -> Dict[str, SemanticTension]:
        """Compute tension for all topics."""
        # Get unique topics
        topics = set()
        for node in self.space.nodes():
            topic = node.metadata.get("topic")
            if topic:
                topics.add(topic)
        
        return {topic: self.compute_tension(topic) for topic in topics}
    
    def check_alerts(self) -> List[Dict]:
        """Check all topics and generate alerts for high tension."""
        tensions = self.compute_all_tensions()
        alerts = []
        
        for topic, tension in tensions.items():
            if tension.tension_score >= self.tension_threshold:
                alert = {
                    "topic": topic,
                    "tension": tension.tension_score,
                    "fork_count": tension.fork_count,
                    "contradictions": len(tension.contradiction_pairs),
                    "node_count": len(tension.node_codes),
                    "generated_at": datetime.utcnow().isoformat(),
                }
                alerts.append(alert)
                self.alerts_generated += 1
        
        return alerts
    
    # -------------------------------------------------------------------------
    # Attention
    # -------------------------------------------------------------------------
    
    def get_attention_map(self) -> AttentionMap:
        """Get current attention geometry."""
        return self.attention.compute()
    
    def where_to_look(self, top_n: int = 5) -> List[AttentionShape]:
        """Get top N places requiring attention."""
        attention_map = self.attention.compute()
        return attention_map.top(top_n)
    
    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------
    
    def start_session(
        self, 
        analyst_id: str, 
        target: str = "",
        inherit_from: Optional[str] = None,
    ) -> AnalystSession:
        """Start a new analyst session."""
        return self.sessions.start_session(
            analyst_id=analyst_id,
            target_description=target,
            inherit_from=inherit_from,
        )
    
    def end_session(self, session_id: str, summary: str) -> Dict[str, Any]:
        """End session and create handoff package."""
        return self.sessions.end_session(
            session_id=session_id,
            summary=summary,
        )
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    
    def compare_feeds(self, feed_a: str, feed_b: str) -> Dict[str, Any]:
        """Compare two RSS feeds for divergence."""
        return self.rss.compare_feeds(feed_a, feed_b)
    
    def find_calendar_conflicts(self, *calendars: str) -> Tuple[List[Node], List[Dict]]:
        """Find conflicts across multiple calendars."""
        return self.ical.merge_calendars(*calendars)
    
    def derivation_ratio(self) -> float:
        """How much more can we derive than we store?"""
        return self.space.derivation_ratio
    
    # -------------------------------------------------------------------------
    # PLTR Analysis (Field Zero)
    # -------------------------------------------------------------------------
    
    def run_pltr_analysis(self) -> Dict[str, Any]:
        """
        Run full PLTR divergence analysis.
        
        This is Field Zero: real SEC filings, real RSS feeds, 
        semantic tension computed from coordinates.
        """
        analyzer = PLTRAnalyzer(space=self.space)
        result = analyzer.run_full_analysis()
        
        # Update engine metrics
        self.nodes_created = len(list(self.space.nodes()))
        
        return {
            **result,
            "brief": analyzer.generate_brief(),
            "linkedin": analyzer.generate_linkedin_post(),
        }
    
    def get_pltr_brief(self) -> str:
        """Get PLTR intelligence brief (runs analysis if needed)."""
        analyzer = PLTRAnalyzer(space=self.space)
        analyzer.run_full_analysis()
        return analyzer.generate_brief()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    def summary(self) -> Dict[str, Any]:
        """Complete system summary."""
        attention_map = self.attention.compute()
        
        return {
            "version": "0.8.1",
            "space": self.space.summary(),
            "events_ingested": self.events_ingested,
            "nodes_created": self.nodes_created,
            "alerts_generated": self.alerts_generated,
            "derivation_ratio": f"{self.derivation_ratio():.0f}x",
            "attention": {
                "total_shapes": len(attention_map.shapes),
                "critical": len(attention_map.critical()),
            },
            "tensions": {
                topic: t.tension_score 
                for topic, t in self.compute_all_tensions().items()
            },
        }


# =============================================================================
# CONVENIENCE: Quick start helpers
# =============================================================================

def create_engine(db_path: str = "ghostbox.db") -> GhostBoxEngine:
    """Create a new GhostBox engine."""
    return GhostBoxEngine(db_path=Path(db_path))


def demo_integration():
    """Demonstrate the integrated stack."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  GHOSTBOX v0.8.0 - Integrated Stack Demo                          ║")
    print("║                                                                   ║")
    print("║  Screen Ghost + Axiom-KG + DIND + Attention + Sessions            ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create engine
    engine = create_engine(db_path="/tmp/ghostbox_demo.db")
    
    # Ingest some events
    events = [
        Event("election_integrity", "feed_a", "Audit confirms no irregularities found.", 0.9),
        Event("election_integrity", "feed_b", "Independent review finds access anomalies.", 0.8),
        Event("election_integrity", "feed_c", "Officials deny remote access occurred.", 0.7),
        Event("election_integrity", "feed_d", "Whistleblower claims systems were compromised.", 0.6),
        Event("breaking_news", "wire_a", "Major policy announcement expected today.", 0.9),
        Event("breaking_news", "wire_b", "Sources confirm policy shift imminent.", 0.85),
    ]
    
    print("Ingesting events...")
    nodes = engine.ingest_events(events)
    print(f"  Created {len(nodes)} nodes in semantic space")
    print()
    
    # Check tension
    print("Computing semantic tension...")
    tension = engine.compute_tension("election_integrity")
    print(f"  Topic: {tension.topic}")
    print(f"  Tension score: {tension.tension_score:.2f}")
    print(f"  Source divergence: {tension.source_divergence:.2f}")
    print(f"  Semantic spread: {tension.spread:.2f}")
    print()
    
    # Get attention
    print("Attention geometry...")
    for shape in engine.where_to_look(3):
        print(f"  [{shape.shape_type.name}] {shape.topic}: {shape.score:.2f}")
    print()
    
    # Summary
    print("System summary:")
    summary = engine.summary()
    print(f"  Nodes: {summary['space']['nodes']}")
    print(f"  Derivation ratio: {summary['derivation_ratio']}")
    print(f"  Critical attention shapes: {summary['attention']['critical']}")
    print()
    
    print("This is Field Zero ready.")
    print("Connect real sources. Run 30 days. Document what it catches.")


if __name__ == "__main__":
    demo_integration()
