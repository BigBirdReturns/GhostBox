"""
GhostBox PLTR Analysis Module

Integrated with Axiom-KG for semantic tension analysis.

Usage:
    from ghostbox_pltr import PLTRAnalyzer
    
    analyzer = PLTRAnalyzer()
    result = analyzer.run_full_analysis()
    print(analyzer.generate_brief())
"""

from .analyzer import (
    PLTRAnalyzer,
    PLTRTension,
    ClaimNode,
    CLAIM_TYPE_COORDS,
    CLAIM_KEYWORDS,
    extract_claims,
    compute_pltr_tension,
)

__all__ = [
    "PLTRAnalyzer",
    "PLTRTension", 
    "ClaimNode",
    "CLAIM_TYPE_COORDS",
    "CLAIM_KEYWORDS",
    "extract_claims",
    "compute_pltr_tension",
]
