"""
GhostBox Sources - Intake Layer

Two modes of data intake:

1. Photonic (Screen Ghost)
   - Screenshot → VLM → Structured state
   - Any app, any device, no API needed
   - The UI is the API

2. Structured (Axiom Adapters)
   - RSS, XBRL, iCal, JSON-LD, etc.
   - Standard schemas → Semantic coordinates
   - When structured data exists

Both produce Nodes that feed into the same Space.
Same tension detection. Same attention geometry.
"""

from .photonic import (
    ScreenGhostSource,
    ScreenState,
    UIElement,
    observe,
    screen_to_node,
)

__all__ = [
    "ScreenGhostSource",
    "ScreenState", 
    "UIElement",
    "observe",
    "screen_to_node",
]
