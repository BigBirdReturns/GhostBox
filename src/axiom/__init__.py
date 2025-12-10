"""
Axiom-KG: Semantic Coordinates for Derivable Knowledge

Store the minimal structure from which all answers can be derived.
Like mathematics stores axioms, not equations.
Like DNA stores instructions, not organisms.

v0.2: Now with 9 domain adapters that read existing schemas:
    - Schema.org / JSON-LD (ASW websites)
    - OpenAPI / Swagger (API specs)
    - RSS / Atom (news feeds)
    - iCalendar (calendars)
    - Package manifests (npm, pip, cargo)
    - FHIR (healthcare)
    - XBRL (financial)
    - EPUB (ebooks)
    - Akoma Ntoso (legal)

This is the seed. The tree grows from here.
"""

__version__ = "0.2.0"

from .core import (
    # Categories
    MAJOR_CATEGORIES,
    
    # Core types
    SemanticID,
    SemanticIDError,
    Node,
    Fork,
    RelationType,
    
    # The space
    Space,
    AuditLog,
    AuditEntry,
    
    # The wrapper
    Strategy,
    Decision,
    DeterministicWrapper,
    
    # Helpers
    create_space,
    create_wrapper,
    quick_node,
)

# Import adapters subpackage
from . import adapters

__all__ = [
    # Version
    "__version__",
    
    # Categories
    "MAJOR_CATEGORIES",
    
    # Core types
    "SemanticID",
    "SemanticIDError", 
    "Node",
    "Fork",
    "RelationType",
    
    # The space
    "Space",
    "AuditLog",
    "AuditEntry",
    
    # The wrapper
    "Strategy",
    "Decision",
    "DeterministicWrapper",
    
    # Helpers
    "create_space",
    "create_wrapper",
    "quick_node",
    
    # Adapters
    "adapters",
]
