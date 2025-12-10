"""
Schema.org / JSON-LD Adapter for axiom-kg

Parses Schema.org JSON-LD (as used by AI-Structured Web) into axiom-kg coordinates.

Schema.org has ~800 types in a hierarchy. We map top-level types to major categories
and derive type/subtype from the schema hierarchy.

Usage:
    adapter = SchemaOrgAdapter()
    nodes = adapter.parse({"@type": "LocalBusiness", "name": "Acme Corp"})
    
    # Or from URL
    nodes = adapter.parse_url("https://example.com/ai.json")
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import JSONAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Schema.org top-level types → axiom-kg major categories
SCHEMA_TYPE_TO_MAJOR = {
    # Entities (Major 1)
    "Thing": 1,
    "Person": 1,
    "Organization": 1,
    "LocalBusiness": 1,
    "Place": 1,
    "Product": 1,
    "CreativeWork": 1,
    "Event": 1,
    "MedicalEntity": 1,
    
    # Actions (Major 2)
    "Action": 2,
    "SearchAction": 2,
    "BuyAction": 2,
    "SellAction": 2,
    "CreateAction": 2,
    "UpdateAction": 2,
    "DeleteAction": 2,
    "MoveAction": 2,
    "TransferAction": 2,
    "InteractAction": 2,
    
    # Properties (Major 3)
    "Property": 3,
    "PropertyValue": 3,
    "QuantitativeValue": 3,
    "Rating": 3,
    "AggregateRating": 3,
    
    # Relations (Major 4)
    "Role": 4,
    "OrganizationRole": 4,
    "EmployeeRole": 4,
    "LinkRole": 4,
    
    # Locations (Major 5)
    "Place": 5,
    "PostalAddress": 5,
    "GeoCoordinates": 5,
    "GeoShape": 5,
    "AdministrativeArea": 5,
    "Country": 5,
    "State": 5,
    "City": 5,
    
    # Time (Major 6)
    "Event": 6,
    "DateTime": 6,
    "Date": 6,
    "Time": 6,
    "Duration": 6,
    "Schedule": 6,
    "OpeningHoursSpecification": 6,
    
    # Quantity (Major 7)
    "QuantitativeValue": 7,
    "MonetaryAmount": 7,
    "PriceSpecification": 7,
    "UnitPriceSpecification": 7,
    "Distance": 7,
    "Mass": 7,
    "Energy": 7,
    
    # Abstract (Major 8)
    "Intangible": 8,
    "Service": 8,
    "Offer": 8,
    "Demand": 8,
    "Invoice": 8,
    "Order": 8,
    "BroadcastChannel": 8,
    "ComputerLanguage": 8,
    "Language": 8,
}

# More specific type mappings for type_ coordinate
SCHEMA_TYPE_TO_TYPE = {
    # Organizations
    "Organization": 1,
    "Corporation": 2,
    "LocalBusiness": 3,
    "NGO": 4,
    "GovernmentOrganization": 5,
    "EducationalOrganization": 6,
    "MedicalOrganization": 7,
    "SportsOrganization": 8,
    
    # LocalBusiness subtypes
    "Restaurant": 10,
    "Store": 11,
    "ProfessionalService": 12,
    "FinancialService": 13,
    "HealthAndBeautyBusiness": 14,
    "HomeAndConstructionBusiness": 15,
    "LegalService": 16,
    "RealEstateAgent": 17,
    
    # Creative works
    "CreativeWork": 1,
    "Article": 2,
    "Book": 3,
    "Movie": 4,
    "MusicRecording": 5,
    "Photograph": 6,
    "SoftwareApplication": 7,
    "WebPage": 8,
    "WebSite": 9,
    
    # Products
    "Product": 1,
    "Vehicle": 2,
    "IndividualProduct": 3,
    "ProductModel": 4,
    "SomeProducts": 5,
}


class SchemaOrgAdapter(JSONAdapter):
    """
    Adapter for Schema.org JSON-LD.
    
    Parses JSON-LD structured data (as used by ASW sites) and converts
    to axiom-kg coordinates for cross-site reasoning.
    """
    
    DOMAIN_NAME = "schemaorg"
    MAJOR_CATEGORY = 1  # Default to Entity
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
        self._type_hash_cache: Dict[str, int] = {}
    
    def _hash_type(self, schema_type: str) -> int:
        """Consistent hash of schema type to type_ coordinate."""
        if schema_type not in self._type_hash_cache:
            # Use explicit mapping if available
            if schema_type in SCHEMA_TYPE_TO_TYPE:
                self._type_hash_cache[schema_type] = SCHEMA_TYPE_TO_TYPE[schema_type]
            else:
                # Hash to 1-99 range
                self._type_hash_cache[schema_type] = (hash(schema_type) % 99) + 1
        return self._type_hash_cache[schema_type]
    
    def _get_major(self, schema_type: str) -> int:
        """Get major category for a schema type."""
        # Check direct mapping
        if schema_type in SCHEMA_TYPE_TO_MAJOR:
            return SCHEMA_TYPE_TO_MAJOR[schema_type]
        
        # Check if it ends with a known suffix
        for known_type, major in SCHEMA_TYPE_TO_MAJOR.items():
            if schema_type.endswith(known_type):
                return major
        
        # Default to Entity
        return 1
    
    def _schema_type_to_id(self, schema_type: str) -> SemanticID:
        """Convert Schema.org type to SemanticID."""
        major = self._get_major(schema_type)
        type_ = self._hash_type(schema_type)
        subtype = 1  # Could derive from schema hierarchy if needed
        
        return self.create_id(major, type_, subtype)
    
    def _extract_label(self, item: Dict) -> str:
        """Extract human-readable label from JSON-LD item."""
        # Try common label fields
        for field in ["name", "title", "headline", "alternateName", "@id"]:
            if field in item and item[field]:
                value = item[field]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    return value.get("@value", str(value))
        
        # Fall back to type
        return item.get("@type", "Unknown")
    
    def _parse_item(self, item: Dict, source_url: Optional[str] = None) -> List[Node]:
        """Parse a single JSON-LD item into nodes."""
        nodes = []
        
        schema_type = item.get("@type")
        if not schema_type:
            return nodes
        
        # Handle array of types
        if isinstance(schema_type, list):
            schema_type = schema_type[0]
        
        # Create main node
        sem_id = self._schema_type_to_id(schema_type)
        label = self._extract_label(item)
        
        metadata = {
            "schema_type": schema_type,
            "source_url": source_url,
            **{k: v for k, v in item.items() if not k.startswith("@")}
        }
        
        node = Node(id=sem_id, label=label, metadata=metadata)
        nodes.append(node)
        
        # Parse nested items
        for key, value in item.items():
            if key.startswith("@"):
                continue
            
            if isinstance(value, dict) and "@type" in value:
                nested_nodes = self._parse_item(value, source_url)
                if nested_nodes:
                    # Add relation from parent to child
                    child = nested_nodes[0]
                    node.add_relation(RelationType.HAS_PROPERTY, child)
                    nodes.extend(nested_nodes)
            
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict) and "@type" in v:
                        nested_nodes = self._parse_item(v, source_url)
                        if nested_nodes:
                            child = nested_nodes[0]
                            node.add_relation(RelationType.HAS_PROPERTY, child)
                            nodes.extend(nested_nodes)
        
        return nodes
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse Schema.org JSON-LD into axiom-kg nodes.
        
        Args:
            source: JSON-LD as dict, string, or file path
            
        Returns:
            List of Node objects
        """
        data = self.load_json(source)
        source_url = None
        
        if isinstance(source, (str, Path)):
            source_url = str(source)
        
        nodes = []
        
        # Handle @graph array
        if "@graph" in data:
            for item in data["@graph"]:
                nodes.extend(self._parse_item(item, source_url))
        else:
            nodes.extend(self._parse_item(data, source_url))
        
        return nodes
    
    def parse_url(self, url: str) -> List[Node]:
        """
        Fetch and parse JSON-LD from a URL.
        
        Args:
            url: URL to fetch (e.g., "https://example.com/ai.json")
            
        Returns:
            List of Node objects
        """
        import urllib.request
        import json
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        nodes = []
        
        if "@graph" in data:
            for item in data["@graph"]:
                nodes.extend(self._parse_item(item, url))
        else:
            nodes.extend(self._parse_item(data, url))
        
        return nodes
    
    def compare_sites(self, url_a: str, url_b: str) -> Dict[str, Any]:
        """
        Compare two ASW sites and find similarities/differences.
        
        Returns:
            Dict with shared types, unique types, and potential forks
        """
        nodes_a = self.parse_url(url_a)
        nodes_b = self.parse_url(url_b)
        
        types_a = {n.metadata.get("schema_type") for n in nodes_a}
        types_b = {n.metadata.get("schema_type") for n in nodes_b}
        
        shared = types_a & types_b
        only_a = types_a - types_b
        only_b = types_b - types_a
        
        # Find potential forks (same type, different properties)
        forks = []
        for type_name in shared:
            nodes_of_type_a = [n for n in nodes_a if n.metadata.get("schema_type") == type_name]
            nodes_of_type_b = [n for n in nodes_b if n.metadata.get("schema_type") == type_name]
            
            for na in nodes_of_type_a:
                for nb in nodes_of_type_b:
                    if na.label != nb.label:
                        forks.append({
                            "type": type_name,
                            "site_a": {"label": na.label, "url": url_a},
                            "site_b": {"label": nb.label, "url": url_b},
                        })
        
        return {
            "shared_types": list(shared),
            "only_in_a": list(only_a),
            "only_in_b": list(only_b),
            "potential_forks": forks,
            "node_count_a": len(nodes_a),
            "node_count_b": len(nodes_b),
        }
