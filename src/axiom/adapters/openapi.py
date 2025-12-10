"""
OpenAPI / Swagger Adapter for axiom-kg

Parses OpenAPI 3.x and Swagger 2.x specifications into axiom-kg coordinates.

Use cases:
- "Which of my microservices overlap?"
- "Which endpoints do the same thing?"
- "What schemas are shared across APIs?"
- "Detect breaking changes between API versions"

Usage:
    adapter = OpenAPIAdapter()
    nodes = adapter.parse("openapi.json")
    
    # Compare two APIs
    diff = adapter.compare_specs("api_v1.json", "api_v2.json")
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from .base import JSONAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# HTTP methods → type_ coordinates
METHOD_TO_TYPE = {
    "get": 1,
    "post": 2,
    "put": 3,
    "patch": 4,
    "delete": 5,
    "head": 6,
    "options": 7,
    "trace": 8,
}

# Common path patterns → subtype coordinates
PATH_PATTERN_TO_SUBTYPE = {
    "auth": 1,
    "login": 1,
    "logout": 1,
    "users": 2,
    "user": 2,
    "accounts": 2,
    "products": 3,
    "items": 3,
    "orders": 4,
    "checkout": 4,
    "cart": 4,
    "search": 5,
    "query": 5,
    "admin": 6,
    "config": 7,
    "settings": 7,
    "health": 8,
    "status": 8,
    "metrics": 9,
    "webhooks": 10,
    "callbacks": 10,
}


class OpenAPIAdapter(JSONAdapter):
    """
    Adapter for OpenAPI/Swagger specifications.
    
    Converts API specs into axiom-kg coordinates:
    - Endpoints → Action nodes (Major 2)
    - Schemas → Entity nodes (Major 1)
    - Parameters → Property nodes (Major 3)
    - Tags → Abstract nodes (Major 8)
    """
    
    DOMAIN_NAME = "openapi"
    SUPPORTED_EXTENSIONS = [".json", ".yaml", ".yml"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def load_spec(self, source: Any) -> Dict:
        """Load OpenAPI spec from JSON or YAML."""
        if isinstance(source, dict):
            return source
        
        path = Path(source) if isinstance(source, str) else source
        
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                return yaml.safe_load(path.read_text())
            except ImportError:
                raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        
        return self.load_json(source)
    
    def _detect_version(self, spec: Dict) -> str:
        """Detect OpenAPI version."""
        if "openapi" in spec:
            return "3.x"
        if "swagger" in spec:
            return "2.x"
        return "unknown"
    
    def _path_to_subtype(self, path: str) -> int:
        """Derive subtype from path pattern."""
        path_lower = path.lower()
        for pattern, subtype in PATH_PATTERN_TO_SUBTYPE.items():
            if pattern in path_lower:
                return subtype
        return 99  # Unknown pattern
    
    def _parse_endpoint(self, path: str, method: str, operation: Dict, spec_name: str) -> Node:
        """Parse a single endpoint into a Node."""
        # Endpoints are Actions (Major 2)
        major = 2
        type_ = METHOD_TO_TYPE.get(method.lower(), 99)
        subtype = self._path_to_subtype(path)
        
        sem_id = self.create_id(major, type_, subtype)
        
        # Build label
        operation_id = operation.get("operationId", f"{method.upper()} {path}")
        label = operation_id
        
        metadata = {
            "path": path,
            "method": method.upper(),
            "operation_id": operation.get("operationId"),
            "summary": operation.get("summary"),
            "description": operation.get("description"),
            "tags": operation.get("tags", []),
            "parameters": [p.get("name") for p in operation.get("parameters", [])],
            "responses": list(operation.get("responses", {}).keys()),
            "spec_name": spec_name,
        }
        
        return Node(id=sem_id, label=label, metadata=metadata)
    
    def _parse_schema(self, name: str, schema: Dict, spec_name: str) -> Node:
        """Parse a schema definition into a Node."""
        # Schemas are Entities (Major 1)
        major = 1
        
        # Determine type from schema structure
        schema_type = schema.get("type", "object")
        type_map = {"object": 1, "array": 2, "string": 3, "number": 4, "integer": 5, "boolean": 6}
        type_ = type_map.get(schema_type, 1)
        
        subtype = 1
        
        sem_id = self.create_id(major, type_, subtype)
        
        # Extract properties
        properties = list(schema.get("properties", {}).keys())
        required = schema.get("required", [])
        
        metadata = {
            "schema_type": schema_type,
            "properties": properties,
            "required": required,
            "description": schema.get("description"),
            "spec_name": spec_name,
        }
        
        return Node(id=sem_id, label=name, metadata=metadata)
    
    def _parse_parameter(self, param: Dict, spec_name: str) -> Node:
        """Parse a parameter definition into a Node."""
        # Parameters are Properties (Major 3)
        major = 3
        
        # Type from location
        location_map = {"query": 1, "path": 2, "header": 3, "cookie": 4, "body": 5}
        type_ = location_map.get(param.get("in", "query"), 1)
        
        subtype = 1
        
        sem_id = self.create_id(major, type_, subtype)
        
        metadata = {
            "location": param.get("in"),
            "required": param.get("required", False),
            "schema": param.get("schema"),
            "description": param.get("description"),
            "spec_name": spec_name,
        }
        
        return Node(id=sem_id, label=param.get("name", "unknown"), metadata=metadata)
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse OpenAPI spec into axiom-kg nodes.
        
        Args:
            source: OpenAPI spec as dict, JSON/YAML file path
            
        Returns:
            List of Node objects
        """
        spec = self.load_spec(source)
        version = self._detect_version(spec)
        
        # Get spec identifier
        spec_name = spec.get("info", {}).get("title", str(source))
        
        nodes = []
        
        # Parse paths/endpoints
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                if method in path_item:
                    operation = path_item[method]
                    node = self._parse_endpoint(path, method, operation, spec_name)
                    nodes.append(node)
        
        # Parse schemas (OpenAPI 3.x)
        if version == "3.x":
            schemas = spec.get("components", {}).get("schemas", {})
            for name, schema in schemas.items():
                node = self._parse_schema(name, schema, spec_name)
                nodes.append(node)
        
        # Parse definitions (Swagger 2.x)
        elif version == "2.x":
            definitions = spec.get("definitions", {})
            for name, schema in definitions.items():
                node = self._parse_schema(name, schema, spec_name)
                nodes.append(node)
        
        return nodes
    
    def compare_specs(self, spec_a: Any, spec_b: Any) -> Dict[str, Any]:
        """
        Compare two OpenAPI specs.
        
        Finds:
        - Shared endpoints
        - Endpoints only in A or B
        - Schema differences
        - Potential breaking changes
        """
        nodes_a = self.parse(spec_a)
        nodes_b = self.parse(spec_b)
        
        # Group by type
        endpoints_a = {n.metadata["path"] + ":" + n.metadata["method"]: n 
                       for n in nodes_a if n.id.major == 2}
        endpoints_b = {n.metadata["path"] + ":" + n.metadata["method"]: n 
                       for n in nodes_b if n.id.major == 2}
        
        schemas_a = {n.label: n for n in nodes_a if n.id.major == 1}
        schemas_b = {n.label: n for n in nodes_b if n.id.major == 1}
        
        # Find differences
        shared_endpoints = set(endpoints_a.keys()) & set(endpoints_b.keys())
        only_in_a = set(endpoints_a.keys()) - set(endpoints_b.keys())
        only_in_b = set(endpoints_b.keys()) - set(endpoints_a.keys())
        
        shared_schemas = set(schemas_a.keys()) & set(schemas_b.keys())
        schemas_only_a = set(schemas_a.keys()) - set(schemas_b.keys())
        schemas_only_b = set(schemas_b.keys()) - set(schemas_a.keys())
        
        # Detect schema changes
        schema_changes = []
        for name in shared_schemas:
            props_a = set(schemas_a[name].metadata.get("properties", []))
            props_b = set(schemas_b[name].metadata.get("properties", []))
            
            if props_a != props_b:
                schema_changes.append({
                    "schema": name,
                    "added_properties": list(props_b - props_a),
                    "removed_properties": list(props_a - props_b),
                })
        
        # Potential breaking changes
        breaking_changes = []
        
        # Removed endpoints are breaking
        for endpoint in only_in_a:
            breaking_changes.append({
                "type": "endpoint_removed",
                "endpoint": endpoint,
            })
        
        # Removed schema properties might be breaking
        for change in schema_changes:
            if change["removed_properties"]:
                breaking_changes.append({
                    "type": "schema_properties_removed",
                    "schema": change["schema"],
                    "properties": change["removed_properties"],
                })
        
        return {
            "endpoints": {
                "shared": list(shared_endpoints),
                "only_in_a": list(only_in_a),
                "only_in_b": list(only_in_b),
            },
            "schemas": {
                "shared": list(shared_schemas),
                "only_in_a": list(schemas_only_a),
                "only_in_b": list(schemas_only_b),
                "changes": schema_changes,
            },
            "breaking_changes": breaking_changes,
            "summary": {
                "endpoints_a": len(endpoints_a),
                "endpoints_b": len(endpoints_b),
                "schemas_a": len(schemas_a),
                "schemas_b": len(schemas_b),
                "breaking_change_count": len(breaking_changes),
            }
        }
    
    def find_similar_endpoints(self, nodes: List[Node], threshold: int = 2) -> List[Dict]:
        """
        Find endpoints that might be doing the same thing.
        
        Uses coordinate distance to find similar endpoints.
        """
        endpoints = [n for n in nodes if n.id.major == 2]
        similar = []
        
        for i, a in enumerate(endpoints):
            for b in endpoints[i+1:]:
                dist = a.id.distance(b.id)
                if dist <= threshold:
                    similar.append({
                        "endpoint_a": a.metadata.get("operation_id", a.label),
                        "endpoint_b": b.metadata.get("operation_id", b.label),
                        "distance": dist,
                        "reason": "Same method and path pattern" if dist == 0 else "Similar pattern",
                    })
        
        return similar
