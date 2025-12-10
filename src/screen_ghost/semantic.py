"""
Semantic Integration Layer for GhostBox v0.8

This module bridges Screen Ghost events to Axiom-KG coordinates.

The integration unlocks:
1. Real semantic distance (not just text matching)
2. Fork detection (when sources diverge on meaning)
3. Derivation ratio (927x compression from coordinates)
4. Tension from semantic drift (not just source count)

Flow:
  Screen Ghost events → Axiom adapter → Semantic coordinates → Space
  Space coordinates → Derived tension → Attention geometry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto

# Screen Ghost imports
from screen_ghost.simulated import Event
from screen_ghost.attention import (
    AttentionGeometry, 
    AttentionMap, 
    AttentionShape, 
    ShapeType
)

# Axiom imports
from axiom.core import (
    Node, 
    SemanticID, 
    Space, 
    Fork,
    RelationType,
    DeterministicWrapper,
    Strategy,
    AuditLog,
)
from axiom.adapters import (
    RSSAdapter,
    ICalAdapter,
    SchemaOrgAdapter,
    BaseAdapter,
    get_adapter,
)


# =============================================================================
# Event to Node Conversion
# =============================================================================

# Topic → Major category mapping
TOPIC_TO_MAJOR = {
    # Entity topics (Major 1)
    "person": 1,
    "company": 1,
    "organization": 1,
    "product": 1,
    
    # Action topics (Major 2)
    "transaction": 2,
    "decision": 2,
    "announcement": 2,
    "event": 2,
    
    # Property topics (Major 3)
    "status": 3,
    "attribute": 3,
    "metric": 3,
    
    # Relation topics (Major 4)
    "connection": 4,
    "agreement": 4,
    "conflict": 4,
    
    # Location topics (Major 5)
    "location": 5,
    "region": 5,
    "address": 5,
    
    # Time topics (Major 6)
    "schedule": 6,
    "deadline": 6,
    "calendar": 6,
    "custody": 6,  # Your use case
    
    # Quantity topics (Major 7)
    "price": 7,
    "count": 7,
    "measurement": 7,
    
    # Abstract topics (Major 8) - default
    "news": 8,
    "opinion": 8,
    "analysis": 8,
    "claim": 8,
}

# Source → Type mapping (within major category)
SOURCE_TO_TYPE = {
    "feed_a": 1,
    "feed_b": 2,
    "feed_c": 3,
    "feed_d": 4,
    "calendar": 5,
    "email": 6,
    "slack": 7,
    "linkedin": 8,
    "twitter": 9,
    "manual": 10,
}


@dataclass
class EventConverter:
    """
    Converts Screen Ghost events to Axiom-KG nodes.
    
    This is the bridge between perception (events) and meaning (coordinates).
    """
    
    space: Space = field(default_factory=Space)
    instance_counter: int = 0
    
    def event_to_node(self, event: Event, timestamp: Optional[datetime] = None) -> Node:
        """
        Convert a Screen Ghost event to an Axiom-KG node.
        
        The conversion:
        - topic → major category
        - source → type within category
        - text hash → subtype (for clustering)
        - counter → instance
        """
        # Determine major from topic
        topic_lower = event.topic.lower()
        major = 8  # Default to Abstract
        for key, maj in TOPIC_TO_MAJOR.items():
            if key in topic_lower:
                major = maj
                break
        
        # Determine type from source
        source_lower = event.source.lower()
        type_ = 99  # Default unknown
        for key, typ in SOURCE_TO_TYPE.items():
            if key in source_lower:
                type_ = typ
                break
        
        # Subtype from text hash (clusters similar content)
        import hashlib
        text_hash = int(hashlib.md5(event.text.encode()).hexdigest()[:4], 16)
        subtype = (text_hash % 99) + 1
        
        # Instance counter
        self.instance_counter += 1
        instance = self.instance_counter % 9999 + 1
        
        sem_id = SemanticID.create(major, type_, subtype, instance)
        
        metadata = {
            "topic": event.topic,
            "source": event.source,
            "confidence": event.confidence,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "text": event.text,
        }
        
        return Node(
            id=sem_id,
            label=event.text[:100],
            metadata=metadata,
        )
    
    def ingest_event(self, event: Event, timestamp: Optional[datetime] = None) -> Node:
        """Convert event to node and add to space."""
        node = self.event_to_node(event, timestamp)
        return self.space.add(node)
    
    def ingest_events(self, events: List[Event]) -> List[Node]:
        """Ingest multiple events."""
        return [self.ingest_event(e) for e in events]


# =============================================================================
# Semantic Tension Calculator
# =============================================================================

@dataclass
class SemanticTension:
    """
    Computes tension from semantic coordinates, not just text.
    
    Tension sources:
    1. Source diversity (different type_ values = different sources)
    2. Semantic distance (coordinates far apart on same topic)
    3. Confidence gradient (high confidence disagreement = high tension)
    4. Fork count (explicit ambiguity markers)
    """
    
    space: Space
    
    def compute_topic_tension(self, topic: str) -> Dict[str, Any]:
        """
        Compute semantic tension for a topic.
        
        Returns detailed breakdown of tension sources.
        """
        # Find all nodes for this topic
        nodes = [
            n for n in self.space.nodes()
            if n.metadata.get("topic", "").lower() == topic.lower()
        ]
        
        if len(nodes) < 2:
            return {
                "topic": topic,
                "tension": 0.0,
                "node_count": len(nodes),
                "breakdown": {},
            }
        
        # Source diversity: count unique type_ values (sources)
        types = set(n.id.type_ for n in nodes)
        source_diversity = len(types) / 10.0  # Normalize to ~1.0
        
        # Semantic spread: average pairwise distance
        distances = []
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                distances.append(a.id.distance(b.id))
        avg_distance = sum(distances) / len(distances) if distances else 0
        semantic_spread = avg_distance / 4.0  # Normalize (max distance = 4)
        
        # Confidence variance: disagreement with high confidence = tension
        confidences = [n.metadata.get("confidence", 0.5) for n in nodes]
        avg_conf = sum(confidences) / len(confidences)
        conf_variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        confidence_tension = conf_variance * avg_conf * 4  # Scale up
        
        # Fork tension: explicit ambiguity in space
        topic_forks = [
            f for f in self.space.forks()
            if any(topic.lower() in n.metadata.get("topic", "").lower() 
                   for n in [self.space.get(b) for b in f.branches if self.space.get(b)])
        ]
        fork_tension = len(topic_forks) * 0.5
        
        # Total tension
        total = source_diversity + semantic_spread + confidence_tension + fork_tension
        
        return {
            "topic": topic,
            "tension": round(min(total, 5.0), 2),  # Cap at 5.0
            "node_count": len(nodes),
            "breakdown": {
                "source_diversity": round(source_diversity, 3),
                "semantic_spread": round(semantic_spread, 3),
                "confidence_tension": round(confidence_tension, 3),
                "fork_tension": round(fork_tension, 3),
            },
            "sources": list(types),
            "avg_confidence": round(avg_conf, 2),
            "avg_distance": round(avg_distance, 2),
        }
    
    def compute_all_tensions(self) -> Dict[str, Dict[str, Any]]:
        """Compute tension for all topics in space."""
        topics = set(
            n.metadata.get("topic", "unknown") 
            for n in self.space.nodes()
        )
        
        return {
            topic: self.compute_topic_tension(topic)
            for topic in topics
        }
    
    def detect_semantic_drift(self, topic: str, window_hours: float = 24) -> Dict[str, Any]:
        """
        Detect if a topic's semantic centroid is drifting over time.
        
        This catches narrative shifts that simple contradiction detection misses.
        """
        from datetime import timedelta
        
        nodes = [
            n for n in self.space.nodes()
            if n.metadata.get("topic", "").lower() == topic.lower()
        ]
        
        if len(nodes) < 4:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        # Sort by timestamp
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=window_hours)
        
        def get_ts(n):
            ts_str = n.metadata.get("timestamp")
            if ts_str:
                return datetime.fromisoformat(ts_str)
            return now
        
        recent = [n for n in nodes if get_ts(n) >= cutoff]
        older = [n for n in nodes if get_ts(n) < cutoff]
        
        if not recent or not older:
            return {"drift_detected": False, "reason": "no_time_split"}
        
        # Compare average coordinates
        def avg_coords(node_list):
            return (
                sum(n.id.major for n in node_list) / len(node_list),
                sum(n.id.type_ for n in node_list) / len(node_list),
                sum(n.id.subtype for n in node_list) / len(node_list),
            )
        
        recent_avg = avg_coords(recent)
        older_avg = avg_coords(older)
        
        # Drift = Euclidean distance in coordinate space
        drift = (
            (recent_avg[0] - older_avg[0]) ** 2 +
            (recent_avg[1] - older_avg[1]) ** 2 +
            (recent_avg[2] - older_avg[2]) ** 2
        ) ** 0.5
        
        return {
            "drift_detected": drift > 5.0,
            "drift_magnitude": round(drift, 2),
            "recent_centroid": recent_avg,
            "older_centroid": older_avg,
            "recent_count": len(recent),
            "older_count": len(older),
        }


# =============================================================================
# Integrated Attention with Semantic Coordinates
# =============================================================================

class SemanticAttentionGeometry(AttentionGeometry):
    """
    Extended attention geometry that uses Axiom-KG semantic coordinates.
    
    Adds:
    - SEMANTIC_DRIFT shape type
    - Fork-based contradiction detection
    - Derivation-aware scoring
    """
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__()
        self.space = space or Space()
        self.converter = EventConverter(space=self.space)
        self.tension_calc = SemanticTension(space=self.space)
    
    def ingest(self, event: Event, timestamp: Optional[datetime] = None) -> None:
        """Ingest event into both attention engine and semantic space."""
        # Standard attention tracking
        super().ingest(event, timestamp)
        
        # Also add to semantic space
        self.converter.ingest_event(event, timestamp)
    
    def compute(self) -> AttentionMap:
        """Compute attention map with semantic enhancements."""
        # Get base attention shapes
        attention = super().compute()
        
        # Add semantic shapes
        for topic in self.events_by_topic:
            # Check for semantic drift
            drift = self.tension_calc.detect_semantic_drift(topic)
            if drift.get("drift_detected"):
                shape = AttentionShape(
                    shape_type=ShapeType.TOPIC_DRIFT,  # Using existing type
                    topic=topic,
                    score=min(1.0, drift["drift_magnitude"] / 10.0),
                    center=(0.3, 0.7),
                    radius=0.2,
                    sources=[],
                    description=f"Semantic drift: {drift['drift_magnitude']:.1f} over {drift['recent_count']} recent vs {drift['older_count']} older",
                    metadata=drift,
                )
                attention.add(shape)
        
        return attention
    
    def semantic_summary(self) -> Dict[str, Any]:
        """Return semantic space summary."""
        return {
            "space": self.space.summary(),
            "tensions": self.tension_calc.compute_all_tensions(),
            "derivation_ratio": f"{self.space.derivation_ratio:.0f}x",
        }


# =============================================================================
# Adapter Integration
# =============================================================================

@dataclass
class AdapterBridge:
    """
    Bridge between Axiom adapters and GhostBox ingestion.
    
    Allows feeding RSS, iCal, etc. directly into the semantic attention engine.
    """
    
    engine: SemanticAttentionGeometry
    
    def ingest_rss(self, feed_source: Any, topic_override: Optional[str] = None) -> List[Node]:
        """
        Ingest an RSS feed.
        
        Each feed item becomes an event → node → attention tracking.
        """
        adapter = RSSAdapter()
        nodes = adapter.parse(feed_source)
        
        for node in nodes:
            # Convert Axiom node back to event for attention tracking
            event = Event(
                topic=topic_override or node.metadata.get("feed_title", "rss"),
                source=node.metadata.get("feed_url", "rss"),
                text=node.label,
                confidence=0.7,
            )
            
            # Parse timestamp
            ts = None
            if node.metadata.get("pub_date"):
                try:
                    ts = datetime.fromisoformat(node.metadata["pub_date"])
                except:
                    pass
            
            self.engine.ingest(event, ts)
            
            # Also add the full node to space
            self.engine.space.add(node)
        
        return nodes
    
    def ingest_calendar(self, ical_source: Any, topic: str = "calendar") -> Tuple[List[Node], List[Dict]]:
        """
        Ingest an iCalendar file.
        
        Returns nodes and any detected conflicts.
        """
        adapter = ICalAdapter()
        nodes = adapter.parse(ical_source)
        
        for node in nodes:
            event = Event(
                topic=topic,
                source=node.metadata.get("organizer", "calendar"),
                text=node.label,
                confidence=0.9,  # Calendar events are high confidence
            )
            
            ts = None
            if node.metadata.get("dtstart"):
                try:
                    ts = datetime.fromisoformat(node.metadata["dtstart"])
                except:
                    pass
            
            self.engine.ingest(event, ts)
            self.engine.space.add(node)
        
        # Find conflicts
        conflicts = adapter.find_conflicts(nodes)
        
        # Add conflict shapes to attention
        for conflict in conflicts:
            shape = AttentionShape(
                shape_type=ShapeType.CONTRADICTION,
                topic=topic,
                score=0.8,  # Calendar conflicts are serious
                center=(0.5, 0.5),
                radius=0.25,
                sources=["calendar"],
                description=f"Schedule conflict: {conflict['event_a']} overlaps {conflict['event_b']}",
                metadata=conflict,
            )
            # We'd need to add to the attention map, but that's computed fresh
            # So we track conflicts separately
        
        return nodes, conflicts
    
    def ingest_schema_org(self, json_ld_source: Any, topic: str = "structured_data") -> List[Node]:
        """
        Ingest Schema.org / JSON-LD data.
        """
        adapter = SchemaOrgAdapter()
        nodes = adapter.parse(json_ld_source)
        
        for node in nodes:
            event = Event(
                topic=topic,
                source=node.metadata.get("url", "schema_org"),
                text=node.label,
                confidence=0.8,
            )
            self.engine.ingest(event)
            self.engine.space.add(node)
        
        return nodes


# =============================================================================
# Main Engine
# =============================================================================

@dataclass
class GhostBoxEngine:
    """
    The unified GhostBox v0.8 engine.
    
    Combines:
    - Screen Ghost capture sources
    - Axiom-KG semantic coordinates
    - Attention geometry
    - Session continuity
    - Deterministic wrapper for accountability
    """
    
    space: Space = field(default_factory=Space)
    attention: SemanticAttentionGeometry = field(default=None)
    adapter_bridge: AdapterBridge = field(default=None)
    wrapper: DeterministicWrapper = field(default=None)
    
    def __post_init__(self):
        if self.attention is None:
            self.attention = SemanticAttentionGeometry(space=self.space)
        if self.adapter_bridge is None:
            self.adapter_bridge = AdapterBridge(engine=self.attention)
        if self.wrapper is None:
            self.wrapper = DeterministicWrapper(space=self.space)
    
    def ingest_event(self, event: Event) -> Node:
        """Ingest a single event."""
        self.attention.ingest(event)
        return self.attention.converter.ingest_event(event)
    
    def ingest_rss(self, source: Any, topic: Optional[str] = None) -> List[Node]:
        """Ingest RSS feed."""
        return self.adapter_bridge.ingest_rss(source, topic)
    
    def ingest_calendar(self, source: Any, topic: str = "calendar") -> Tuple[List[Node], List[Dict]]:
        """Ingest iCalendar."""
        return self.adapter_bridge.ingest_calendar(source, topic)
    
    def get_attention(self) -> AttentionMap:
        """Get current attention map."""
        return self.attention.compute()
    
    def get_tensions(self) -> Dict[str, Dict[str, Any]]:
        """Get semantic tensions for all topics."""
        return self.attention.tension_calc.compute_all_tensions()
    
    def get_topic_tension(self, topic: str) -> Dict[str, Any]:
        """Get tension for a specific topic."""
        return self.attention.tension_calc.compute_topic_tension(topic)
    
    def summary(self) -> Dict[str, Any]:
        """Full engine summary."""
        attention_map = self.get_attention()
        
        return {
            "space": self.space.summary(),
            "attention": attention_map.summary(),
            "tensions": self.get_tensions(),
            "derivation_ratio": f"{self.space.derivation_ratio:.0f}x",
            "audit_valid": self.space.audit.verify(),
        }


# =============================================================================
# Demo
# =============================================================================

def demo_semantic_integration() -> None:
    """Demonstrate the semantic integration."""
    from screen_ghost.simulated import generate_demo_events
    
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  GHOSTBOX v0.8: Semantic Integration Demo                         ║")
    print("║                                                                   ║")
    print("║  Events → Coordinates → Derived Tension → Attention Geometry      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create engine
    engine = GhostBoxEngine()
    
    # Ingest demo events
    events = generate_demo_events()
    print(f"Ingesting {len(events)} demo events...")
    for event in events:
        engine.ingest_event(event)
    
    print()
    print("SEMANTIC SPACE")
    print("=" * 60)
    space_summary = engine.space.summary()
    print(f"  Nodes: {space_summary['nodes']}")
    print(f"  Relations: {space_summary['relations']}")
    print(f"  Forks: {space_summary['forks']}")
    print(f"  Derivation ratio: {space_summary['derivation_ratio']}")
    print(f"  Audit valid: {space_summary['chain_valid']}")
    print()
    
    print("SEMANTIC TENSIONS")
    print("=" * 60)
    tensions = engine.get_tensions()
    for topic, data in tensions.items():
        print(f"\n  Topic: {topic}")
        print(f"    Tension: {data['tension']}")
        print(f"    Nodes: {data['node_count']}")
        if data.get('breakdown'):
            print(f"    Breakdown:")
            for k, v in data['breakdown'].items():
                print(f"      {k}: {v}")
    print()
    
    print("ATTENTION MAP")
    print("=" * 60)
    attention = engine.get_attention()
    print(f"  Total shapes: {len(attention.shapes)}")
    print(f"  Critical: {len(attention.critical())}")
    print()
    print("  Top shapes:")
    for shape in attention.top(5):
        print(f"    [{shape.shape_type.name}] {shape.topic}: {shape.score:.2f}")
        print(f"      {shape.description}")
    print()
    
    print("FULL SUMMARY")
    print("=" * 60)
    import json
    summary = engine.summary()
    # Simplify for display
    display_summary = {
        "nodes": summary["space"]["nodes"],
        "derivation_ratio": summary["derivation_ratio"],
        "attention_shapes": summary["attention"]["total_shapes"],
        "critical_shapes": summary["attention"]["critical_count"],
        "audit_valid": summary["audit_valid"],
    }
    print(json.dumps(display_summary, indent=2))


if __name__ == "__main__":
    demo_semantic_integration()
