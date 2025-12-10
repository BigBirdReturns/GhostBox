"""
axiom-kg Adapters

Domain adapters convert structured schemas into axiom-kg coordinates.
Each adapter reads a specific format and produces Nodes with SemanticIDs.

Available adapters:
    - SchemaOrgAdapter: JSON-LD / Schema.org (ASW sites)
    - OpenAPIAdapter: OpenAPI/Swagger API specs
    - RSSAdapter: RSS/Atom feeds
    - ICalAdapter: iCalendar files
    - PackageAdapter: package.json, requirements.txt, Cargo.toml
    - FHIRAdapter: HL7 FHIR healthcare resources
    - XBRLAdapter: XBRL financial reports
    - EPUBAdapter: EPUB/ebook metadata
    - AkomaNtosoAdapter: Legal/legislative documents

Usage:
    from axiom.adapters import SchemaOrgAdapter
    
    adapter = SchemaOrgAdapter()
    nodes = adapter.parse("https://example.com/ai.json")
    
    # Or compare two sources
    diff = adapter.compare_sites("https://site-a.com/ai.json", "https://site-b.com/ai.json")
"""

from .base import BaseAdapter, FileAdapter, JSONAdapter, XMLAdapter
from .schemaorg import SchemaOrgAdapter
from .openapi import OpenAPIAdapter
from .rss import RSSAdapter
from .ical import ICalAdapter
from .package import PackageAdapter
from .fhir import FHIRAdapter
from .xbrl import XBRLAdapter
from .epub import EPUBAdapter
from .akn import AkomaNtosoAdapter

__all__ = [
    # Base classes
    "BaseAdapter",
    "FileAdapter", 
    "JSONAdapter",
    "XMLAdapter",
    
    # Domain adapters
    "SchemaOrgAdapter",
    "OpenAPIAdapter",
    "RSSAdapter",
    "ICalAdapter",
    "PackageAdapter",
    "FHIRAdapter",
    "XBRLAdapter",
    "EPUBAdapter",
    "AkomaNtosoAdapter",
]

# Convenience mapping
ADAPTERS = {
    "schemaorg": SchemaOrgAdapter,
    "jsonld": SchemaOrgAdapter,
    "asw": SchemaOrgAdapter,
    "openapi": OpenAPIAdapter,
    "swagger": OpenAPIAdapter,
    "rss": RSSAdapter,
    "atom": RSSAdapter,
    "feed": RSSAdapter,
    "ical": ICalAdapter,
    "calendar": ICalAdapter,
    "npm": PackageAdapter,
    "pip": PackageAdapter,
    "cargo": PackageAdapter,
    "package": PackageAdapter,
    "fhir": FHIRAdapter,
    "healthcare": FHIRAdapter,
    "xbrl": XBRLAdapter,
    "financial": XBRLAdapter,
    "epub": EPUBAdapter,
    "ebook": EPUBAdapter,
    "akn": AkomaNtosoAdapter,
    "legal": AkomaNtosoAdapter,
}


def get_adapter(name: str) -> type:
    """
    Get adapter class by name.
    
    Args:
        name: Adapter name (e.g., "schemaorg", "openapi", "rss")
        
    Returns:
        Adapter class
        
    Example:
        AdapterClass = get_adapter("openapi")
        adapter = AdapterClass()
        nodes = adapter.parse("api.json")
    """
    name_lower = name.lower()
    if name_lower not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name_lower]
