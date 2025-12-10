"""
Base Adapter Interface for axiom-kg

Adapters convert domain-specific schemas into axiom-kg coordinates.
The geometry engine stays domain-agnostic. Domain knowledge lives here.

Every adapter must:
1. Parse its native format
2. Assign SemanticID coordinates
3. Create Nodes with appropriate relations
4. Return a list of Nodes (or add directly to a Space)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from axiom.core import Node, SemanticID, Space, RelationType


class BaseAdapter(ABC):
    """
    Abstract base class for all axiom-kg adapters.
    
    Subclasses implement parsing logic for specific formats.
    The coordinate assignment can use the default or be overridden.
    """
    
    # Override in subclass
    DOMAIN_NAME: str = "base"
    MAJOR_CATEGORY: int = 8  # Default to Abstract
    
    def __init__(self, space: Optional[Space] = None):
        self.space = space or Space()
        self._instance_counter: Dict[tuple, int] = {}
    
    def _next_instance(self, major: int, type_: int, subtype: int) -> int:
        """Get next available instance number for a coordinate prefix."""
        key = (major, type_, subtype)
        self._instance_counter[key] = self._instance_counter.get(key, 0) + 1
        return self._instance_counter[key]
    
    def create_id(self, major: int, type_: int, subtype: int = 1) -> SemanticID:
        """Create a SemanticID with auto-incrementing instance."""
        instance = self._next_instance(major, type_, subtype)
        return SemanticID.create(major, type_, subtype, instance)
    
    @abstractmethod
    def parse(self, source: Any) -> List[Node]:
        """
        Parse source data into axiom-kg Nodes.
        
        Args:
            source: Domain-specific input (file path, dict, string, etc.)
            
        Returns:
            List of Node objects with assigned coordinates
        """
        pass
    
    def parse_to_space(self, source: Any) -> Space:
        """Parse source and add all nodes to the space."""
        nodes = self.parse(source)
        for node in nodes:
            if self.space.get(node.id) is None:
                self.space.add(node)
        return self.space
    
    def parse_many(self, sources: List[Any]) -> List[Node]:
        """Parse multiple sources."""
        all_nodes = []
        for source in sources:
            all_nodes.extend(self.parse(source))
        return all_nodes


class FileAdapter(BaseAdapter):
    """Base adapter for file-based sources."""
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def parse_file(self, path: Path) -> List[Node]:
        """Parse a file by path."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return self.parse(path)
    
    def parse_directory(self, directory: Path, recursive: bool = False) -> List[Node]:
        """Parse all supported files in a directory."""
        all_nodes = []
        
        pattern = "**/*" if recursive else "*"
        for ext in self.SUPPORTED_EXTENSIONS:
            for path in directory.glob(f"{pattern}{ext}"):
                try:
                    all_nodes.extend(self.parse_file(path))
                except Exception as e:
                    print(f"Warning: Failed to parse {path}: {e}")
        
        return all_nodes


class JSONAdapter(FileAdapter):
    """Base adapter for JSON-based formats."""
    
    SUPPORTED_EXTENSIONS = [".json"]
    
    def load_json(self, source: Any) -> Dict:
        """Load JSON from file path, string, or dict."""
        import json
        
        if isinstance(source, dict):
            return source
        
        if isinstance(source, str):
            # Try as file path first
            path = Path(source)
            if path.exists():
                return json.loads(path.read_text())
            # Try as JSON string
            return json.loads(source)
        
        if isinstance(source, Path):
            return json.loads(source.read_text())
        
        raise ValueError(f"Cannot load JSON from {type(source)}")


class XMLAdapter(FileAdapter):
    """Base adapter for XML-based formats."""
    
    SUPPORTED_EXTENSIONS = [".xml"]
    
    def load_xml(self, source: Any):
        """Load XML from file path or string."""
        import xml.etree.ElementTree as ET
        
        if isinstance(source, str):
            path = Path(source)
            if path.exists():
                return ET.parse(path).getroot()
            return ET.fromstring(source)
        
        if isinstance(source, Path):
            return ET.parse(source).getroot()
        
        raise ValueError(f"Cannot load XML from {type(source)}")
