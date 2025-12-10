"""
GhostBox v0.8.0

Efficient human oversight for adversarial information environments.

Integrated stack:
- Screen Ghost: Capture sources
- Axiom-KG: Semantic coordinates  
- DIND Stack: Tension detection
- Attention Geometry: Where to look
- Session Continuity: Thread doesn't reset

Usage:
    from ghostbox import create_engine
    
    engine = create_engine()
    engine.ingest_rss("https://feed.example.com/rss")
    
    for shape in engine.where_to_look():
        print(f"Look at: {shape.topic} ({shape.score:.2f})")
"""

__version__ = "0.8.0"

from .integration import (
    GhostBoxEngine,
    SemanticTension,
    compute_semantic_tension,
    create_engine,
    demo_integration,
)

__all__ = [
    "__version__",
    "GhostBoxEngine",
    "SemanticTension",
    "compute_semantic_tension",
    "create_engine",
    "demo_integration",
]
