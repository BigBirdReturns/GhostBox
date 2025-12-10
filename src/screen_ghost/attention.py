"""
Attention Geometry for GhostBox v0.7

This module computes "attention shapes" - geometric patterns that tell
analysts where to focus in an information stream.

The core insight: human attention is finite. In adversarial environments,
you can't read everything. You need geometric signals that highlight:

1. Contradiction clusters (multiple sources disagreeing)
2. Velocity spikes (sudden increase in activity)
3. Source convergence (independent sources aligning)
4. Confidence gradients (certainty changing over time)
5. Topic drift (semantic shift in a feed)

These aren't metaphors. They're computable shapes that can be:
- Rendered visually (attention maps)
- Ranked numerically (priority scores)
- Used to trigger alerts (threshold crossings)

This is what your colleague saw: data annotation for analysts,
geometric signals showing where to look.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

from .simulated import Event


class ShapeType(Enum):
    """Classification of attention shapes."""
    CONTRADICTION = auto()    # Sources disagree
    VELOCITY_SPIKE = auto()   # Activity burst
    CONVERGENCE = auto()      # Sources align unexpectedly
    CONFIDENCE_DROP = auto()  # Certainty declining
    TOPIC_DRIFT = auto()      # Semantic shift
    SILENCE = auto()          # Expected activity missing


@dataclass
class AttentionShape:
    """
    A geometric signal indicating where attention is needed.
    
    Think of this as a "hot spot" on a map of information space.
    """
    
    shape_type: ShapeType
    topic: str
    score: float  # 0.0 = ignore, 1.0 = critical
    center: Tuple[float, float]  # (x, y) in abstract attention space
    radius: float  # How much context to pull in
    sources: List[str]  # Which sources contributed
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"[{self.shape_type.name}] {self.topic}: {self.score:.2f} - {self.description}"


@dataclass
class AttentionMap:
    """
    A collection of attention shapes forming a "where to look" map.
    
    Analysts can scan this instead of reading everything.
    """
    
    shapes: List[AttentionShape] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add(self, shape: AttentionShape) -> None:
        self.shapes.append(shape)
    
    def top(self, n: int = 5) -> List[AttentionShape]:
        """Get top N shapes by score."""
        return sorted(self.shapes, key=lambda s: s.score, reverse=True)[:n]
    
    def by_type(self, shape_type: ShapeType) -> List[AttentionShape]:
        """Filter shapes by type."""
        return [s for s in self.shapes if s.shape_type == shape_type]
    
    def by_topic(self, topic: str) -> List[AttentionShape]:
        """Filter shapes by topic."""
        return [s for s in self.shapes if s.topic == topic]
    
    def critical(self, threshold: float = 0.7) -> List[AttentionShape]:
        """Get shapes above critical threshold."""
        return [s for s in self.shapes if s.score >= threshold]
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary for API responses."""
        return {
            "total_shapes": len(self.shapes),
            "critical_count": len(self.critical()),
            "by_type": {
                t.name: len(self.by_type(t)) 
                for t in ShapeType
            },
            "top_5": [
                {
                    "type": s.shape_type.name,
                    "topic": s.topic,
                    "score": s.score,
                    "description": s.description,
                }
                for s in self.top(5)
            ],
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# Shape Detectors
# =============================================================================

@dataclass
class ContradictionDetector:
    """
    Detect when sources contradict each other on a topic.
    
    Contradiction score increases with:
    - Number of distinct sources
    - Semantic opposition in claims
    - High confidence on both sides
    """
    
    opposition_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "positive": ["confirmed", "verified", "true", "yes", "found", "detected"],
        "negative": ["denied", "refuted", "false", "no", "not found", "undetected"],
    })
    
    def detect(self, topic: str, events: List[Event]) -> Optional[AttentionShape]:
        if len(events) < 2:
            return None
        
        sources = list(set(e.source for e in events))
        if len(sources) < 2:
            return None
        
        # Check for semantic opposition
        positive_count = 0
        negative_count = 0
        
        for e in events:
            text_lower = e.text.lower()
            if any(kw in text_lower for kw in self.opposition_keywords["positive"]):
                positive_count += 1
            if any(kw in text_lower for kw in self.opposition_keywords["negative"]):
                negative_count += 1
        
        if positive_count == 0 or negative_count == 0:
            return None
        
        # Score based on balance and confidence
        balance = min(positive_count, negative_count) / max(positive_count, negative_count)
        avg_confidence = sum(e.confidence for e in events) / len(events)
        
        score = balance * avg_confidence * min(1.0, len(sources) / 3)
        
        if score < 0.3:
            return None
        
        return AttentionShape(
            shape_type=ShapeType.CONTRADICTION,
            topic=topic,
            score=score,
            center=(0.5, 0.5),  # Center of attention space
            radius=0.3,
            sources=sources,
            description=f"{len(sources)} sources disagree ({positive_count} affirm, {negative_count} deny)",
            metadata={
                "positive_count": positive_count,
                "negative_count": negative_count,
                "avg_confidence": avg_confidence,
            }
        )


@dataclass
class VelocityDetector:
    """
    Detect sudden spikes in activity on a topic.
    
    Compares recent activity to baseline.
    """
    
    window_seconds: float = 60.0
    spike_threshold: float = 3.0  # N times baseline
    
    def detect(
        self, 
        topic: str, 
        events: List[Event],
        timestamps: List[datetime],
        baseline_rate: float = 1.0,  # events per minute
    ) -> Optional[AttentionShape]:
        
        if not timestamps:
            return None
        
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        recent = sum(1 for t in timestamps if t >= window_start)
        rate = recent / (self.window_seconds / 60.0)  # events per minute
        
        if rate < baseline_rate * self.spike_threshold:
            return None
        
        score = min(1.0, rate / (baseline_rate * self.spike_threshold * 2))
        sources = list(set(e.source for e in events))
        
        return AttentionShape(
            shape_type=ShapeType.VELOCITY_SPIKE,
            topic=topic,
            score=score,
            center=(0.8, 0.5),  # Right side of attention space (urgent)
            radius=0.2,
            sources=sources,
            description=f"{rate:.1f} events/min (baseline: {baseline_rate:.1f})",
            metadata={
                "rate": rate,
                "baseline": baseline_rate,
                "multiplier": rate / baseline_rate,
            }
        )


@dataclass
class ConvergenceDetector:
    """
    Detect when independent sources unexpectedly align.
    
    This is often more significant than contradiction -
    independent agreement suggests something real happened.
    """
    
    similarity_threshold: float = 0.6
    
    def detect(self, topic: str, events: List[Event]) -> Optional[AttentionShape]:
        if len(events) < 2:
            return None
        
        sources = list(set(e.source for e in events))
        if len(sources) < 2:
            return None
        
        # Group by source
        by_source: Dict[str, List[Event]] = defaultdict(list)
        for e in events:
            by_source[e.source].append(e)
        
        # Check for high confidence alignment across sources
        high_conf_sources = [
            src for src, evts in by_source.items()
            if any(e.confidence > 0.8 for e in evts)
        ]
        
        if len(high_conf_sources) < 2:
            return None
        
        # Simple convergence: multiple high-confidence sources
        score = min(1.0, len(high_conf_sources) / 4)
        
        if score < 0.4:
            return None
        
        return AttentionShape(
            shape_type=ShapeType.CONVERGENCE,
            topic=topic,
            score=score,
            center=(0.5, 0.8),  # Top of attention space (significant)
            radius=0.25,
            sources=high_conf_sources,
            description=f"{len(high_conf_sources)} independent sources agree with high confidence",
            metadata={
                "high_conf_sources": high_conf_sources,
            }
        )


@dataclass
class ConfidenceDropDetector:
    """
    Detect declining confidence over time.
    
    When sources become less certain, something changed.
    """
    
    drop_threshold: float = 0.2  # 20% drop triggers
    
    def detect(
        self, 
        topic: str, 
        events: List[Event],
        timestamps: List[datetime],
    ) -> Optional[AttentionShape]:
        
        if len(events) < 3:
            return None
        
        # Sort by timestamp
        paired = sorted(zip(timestamps, events), key=lambda x: x[0])
        
        # Compare first half to second half
        mid = len(paired) // 2
        first_half = [e for _, e in paired[:mid]]
        second_half = [e for _, e in paired[mid:]]
        
        avg_first = sum(e.confidence for e in first_half) / len(first_half)
        avg_second = sum(e.confidence for e in second_half) / len(second_half)
        
        drop = avg_first - avg_second
        
        if drop < self.drop_threshold:
            return None
        
        score = min(1.0, drop / 0.5)
        sources = list(set(e.source for e in events))
        
        return AttentionShape(
            shape_type=ShapeType.CONFIDENCE_DROP,
            topic=topic,
            score=score,
            center=(0.2, 0.5),  # Left side (concerning)
            radius=0.2,
            sources=sources,
            description=f"Confidence dropped {drop:.0%} over time",
            metadata={
                "avg_first": avg_first,
                "avg_second": avg_second,
                "drop": drop,
            }
        )


@dataclass
class SilenceDetector:
    """
    Detect when expected activity is missing.
    
    Sometimes the absence of information is the signal.
    """
    
    expected_interval_seconds: float = 300.0  # 5 minutes
    
    def detect(
        self,
        topic: str,
        timestamps: List[datetime],
        sources: List[str],
    ) -> Optional[AttentionShape]:
        
        if not timestamps:
            return None
        
        now = datetime.utcnow()
        latest = max(timestamps)
        gap = (now - latest).total_seconds()
        
        if gap < self.expected_interval_seconds:
            return None
        
        score = min(1.0, gap / (self.expected_interval_seconds * 4))
        
        return AttentionShape(
            shape_type=ShapeType.SILENCE,
            topic=topic,
            score=score,
            center=(0.1, 0.1),  # Corner (anomaly)
            radius=0.15,
            sources=sources,
            description=f"No activity for {gap/60:.1f} minutes",
            metadata={
                "gap_seconds": gap,
                "expected_seconds": self.expected_interval_seconds,
            }
        )


# =============================================================================
# Attention Geometry Engine
# =============================================================================

@dataclass
class AttentionGeometry:
    """
    Main engine for computing attention shapes across all topics.
    
    Usage:
        engine = AttentionGeometry()
        
        # Add events
        for event in events:
            engine.ingest(event)
        
        # Get attention map
        attention = engine.compute()
        
        # See where to look
        for shape in attention.top(5):
            print(shape)
    """
    
    # Event storage by topic
    events_by_topic: Dict[str, List[Event]] = field(default_factory=lambda: defaultdict(list))
    timestamps_by_topic: Dict[str, List[datetime]] = field(default_factory=lambda: defaultdict(list))
    
    # Baseline rates for velocity detection
    baseline_rates: Dict[str, float] = field(default_factory=dict)
    default_baseline: float = 0.5  # events per minute
    
    # Detectors
    contradiction_detector: ContradictionDetector = field(default_factory=ContradictionDetector)
    velocity_detector: VelocityDetector = field(default_factory=VelocityDetector)
    convergence_detector: ConvergenceDetector = field(default_factory=ConvergenceDetector)
    confidence_drop_detector: ConfidenceDropDetector = field(default_factory=ConfidenceDropDetector)
    silence_detector: SilenceDetector = field(default_factory=SilenceDetector)
    
    def ingest(self, event: Event, timestamp: Optional[datetime] = None) -> None:
        """Add an event to the geometry engine."""
        ts = timestamp or datetime.utcnow()
        self.events_by_topic[event.topic].append(event)
        self.timestamps_by_topic[event.topic].append(ts)
    
    def ingest_batch(self, events: List[Event]) -> None:
        """Add multiple events."""
        for e in events:
            self.ingest(e)
    
    def set_baseline(self, topic: str, rate: float) -> None:
        """Set expected event rate for a topic."""
        self.baseline_rates[topic] = rate
    
    def compute(self) -> AttentionMap:
        """Compute attention shapes for all topics."""
        attention = AttentionMap()
        
        for topic in self.events_by_topic:
            events = self.events_by_topic[topic]
            timestamps = self.timestamps_by_topic[topic]
            sources = list(set(e.source for e in events))
            baseline = self.baseline_rates.get(topic, self.default_baseline)
            
            # Run all detectors
            shapes = [
                self.contradiction_detector.detect(topic, events),
                self.velocity_detector.detect(topic, events, timestamps, baseline),
                self.convergence_detector.detect(topic, events),
                self.confidence_drop_detector.detect(topic, events, timestamps),
                self.silence_detector.detect(topic, timestamps, sources),
            ]
            
            for shape in shapes:
                if shape is not None:
                    attention.add(shape)
        
        return attention
    
    def clear(self, topic: Optional[str] = None) -> None:
        """Clear stored events."""
        if topic:
            self.events_by_topic[topic] = []
            self.timestamps_by_topic[topic] = []
        else:
            self.events_by_topic.clear()
            self.timestamps_by_topic.clear()


# =============================================================================
# Demo
# =============================================================================

def demo_attention_geometry() -> None:
    """Demonstrate attention geometry with synthetic events."""
    from .simulated import generate_demo_events
    
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  ATTENTION GEOMETRY DEMO                                          ║")
    print("║                                                                   ║")
    print("║  Geometric shapes that show analysts where to look.               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    engine = AttentionGeometry()
    
    # Ingest the demo events (contradictory)
    events = generate_demo_events()
    engine.ingest_batch(events)
    
    # Add some velocity events
    for i in range(10):
        engine.ingest(Event(
            topic="breaking_news",
            source=f"feed_{i % 3}",
            text=f"Breaking: Event {i} confirmed",
            confidence=0.7,
        ))
    
    # Compute attention map
    attention = engine.compute()
    
    print("ATTENTION MAP")
    print("=" * 60)
    print()
    
    print(f"Total shapes detected: {len(attention.shapes)}")
    print(f"Critical (>0.7): {len(attention.critical())}")
    print()
    
    print("TOP ATTENTION SHAPES:")
    print("-" * 60)
    for shape in attention.top(5):
        print(f"  {shape}")
    print()
    
    print("BY TYPE:")
    print("-" * 60)
    for shape_type in ShapeType:
        shapes = attention.by_type(shape_type)
        if shapes:
            print(f"  {shape_type.name}: {len(shapes)}")
    print()
    
    print("SUMMARY:")
    print("-" * 60)
    import json
    print(json.dumps(attention.summary(), indent=2))


if __name__ == "__main__":
    demo_attention_geometry()
