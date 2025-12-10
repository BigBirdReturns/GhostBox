"""
Akoma Ntoso Legal Adapter for axiom-kg

Parses Akoma Ntoso XML format for legislative and legal documents.

Use cases:
- "How does this new regulation affect existing law?"
- "Track amendments to a statute over time"
- "Find related provisions across jurisdictions"
- "Build citation graph"

Akoma Ntoso: http://www.akomantoso.org/

Usage:
    adapter = AkomaNtosoAdapter()
    nodes = adapter.parse("legislation.xml")
    
    # Find related provisions
    related = adapter.find_citing(nodes, target_provision)
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime
import re

from .base import XMLAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Document types → type_ coordinates
DOC_TYPE_TO_TYPE = {
    # Primary legislation
    "act": 1,
    "bill": 2,
    "statute": 3,
    "law": 4,
    "constitution": 5,
    
    # Secondary legislation
    "regulation": 10,
    "rule": 11,
    "order": 12,
    "decree": 13,
    "ordinance": 14,
    
    # Judicial
    "judgment": 20,
    "decision": 21,
    "opinion": 22,
    "order": 23,
    
    # Other
    "amendment": 30,
    "resolution": 31,
    "treaty": 32,
    "agreement": 33,
}

# Structural elements → subtype coordinates
ELEMENT_TO_SUBTYPE = {
    "article": 1,
    "section": 2,
    "paragraph": 3,
    "clause": 4,
    "subclause": 5,
    "chapter": 6,
    "part": 7,
    "title": 8,
    "schedule": 9,
    "annex": 10,
}

# Akoma Ntoso namespace
AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


class AkomaNtosoAdapter(XMLAdapter):
    """
    Adapter for Akoma Ntoso legal documents.
    
    Converts legal documents into axiom-kg coordinates:
    - Documents → Entity nodes (Major 1)
    - Structural elements → Entity nodes with hierarchy
    - References/citations → Relation edges
    - Metadata → Property nodes
    """
    
    DOMAIN_NAME = "akn"
    SUPPORTED_EXTENSIONS = [".xml", ".akn"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _get_doc_type(self, root) -> int:
        """Determine document type from root element or metadata."""
        root_tag = root.tag.split('}')[-1].lower() if '}' in root.tag else root.tag.lower()
        
        for doc_type, type_ in DOC_TYPE_TO_TYPE.items():
            if doc_type in root_tag:
                return type_
        
        # Check children for document type
        for child in root:
            child_tag = child.tag.split('}')[-1].lower() if '}' in child.tag else child.tag.lower()
            for doc_type, type_ in DOC_TYPE_TO_TYPE.items():
                if doc_type in child_tag:
                    return type_
        
        return 99  # Unknown
    
    def _extract_metadata(self, root) -> Dict:
        """Extract document metadata from identification block."""
        metadata = {
            "title": None,
            "date": None,
            "uri": None,
            "country": None,
            "language": None,
            "keywords": [],
        }
        
        # Look for meta/identification
        for elem in root.iter():
            tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
            
            if tag == "frbrwork":
                # FRBR Work identification
                for child in elem.iter():
                    child_tag = child.tag.split('}')[-1].lower() if '}' in child.tag else child.tag.lower()
                    if child_tag == "frbruri":
                        metadata["uri"] = child.get("value")
                    elif child_tag == "frbrdate":
                        metadata["date"] = child.get("date")
                    elif child_tag == "frbrcountry":
                        metadata["country"] = child.get("value")
            
            elif tag == "doctitle":
                metadata["title"] = elem.text
            
            elif tag == "keyword":
                kw = elem.get("value") or elem.text
                if kw:
                    metadata["keywords"].append(kw)
            
            elif tag == "frbrlanguage":
                metadata["language"] = elem.get("language")
        
        return metadata
    
    def _parse_structure(self, element, parent_path: str = "", doc_metadata: Dict = None) -> List[Node]:
        """Recursively parse document structure into nodes."""
        nodes = []
        
        tag = element.tag.split('}')[-1].lower() if '}' in element.tag else element.tag.lower()
        
        # Check if this is a structural element
        subtype = ELEMENT_TO_SUBTYPE.get(tag, 0)
        
        if subtype > 0:
            # This is a structural element
            eId = element.get("eId", element.get("id", ""))
            num_elem = element.find(f".//{{{AKN_NS}}}num") or element.find(".//num")
            heading_elem = element.find(f".//{{{AKN_NS}}}heading") or element.find(".//heading")
            
            num = num_elem.text if num_elem is not None else ""
            heading = heading_elem.text if heading_elem is not None else ""
            
            # Build path
            current_path = f"{parent_path}/{tag}" if parent_path else tag
            if num:
                current_path += f"[{num}]"
            
            # Create node
            sem_id = self.create_id(1, DOC_TYPE_TO_TYPE.get(tag, 1), subtype)
            
            label = f"{tag.title()} {num}".strip()
            if heading:
                label += f": {heading}"
            
            # Extract text content (simplified)
            text_content = ""
            for p in element.findall(f".//{{{AKN_NS}}}p") + element.findall(".//p"):
                if p.text:
                    text_content += p.text + " "
            
            metadata = {
                "element_type": tag,
                "eId": eId,
                "num": num,
                "heading": heading,
                "path": current_path,
                "text_preview": text_content[:500].strip() if text_content else None,
                "doc_title": doc_metadata.get("title") if doc_metadata else None,
                "doc_uri": doc_metadata.get("uri") if doc_metadata else None,
            }
            
            node = Node(id=sem_id, label=label, metadata=metadata)
            nodes.append(node)
            
            # Parse children
            for child in element:
                child_nodes = self._parse_structure(child, current_path, doc_metadata)
                nodes.extend(child_nodes)
        else:
            # Not a structural element, but might have structural children
            for child in element:
                child_nodes = self._parse_structure(child, parent_path, doc_metadata)
                nodes.extend(child_nodes)
        
        return nodes
    
    def _extract_references(self, root) -> List[Dict]:
        """Extract all references/citations from document."""
        references = []
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
            
            if tag == "ref":
                ref = {
                    "href": elem.get("href"),
                    "text": elem.text,
                    "source_eId": None,
                }
                
                # Try to find parent eId
                parent = elem
                while parent is not None:
                    if parent.get("eId"):
                        ref["source_eId"] = parent.get("eId")
                        break
                    parent = parent.getparent() if hasattr(parent, 'getparent') else None
                
                references.append(ref)
        
        return references
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse Akoma Ntoso document into axiom-kg nodes.
        
        Args:
            source: Path to .xml file or XML string
            
        Returns:
            List of Node objects
        """
        root = self.load_xml(source)
        
        # Extract document metadata
        metadata = self._extract_metadata(root)
        
        # Create document node
        doc_type = self._get_doc_type(root)
        doc_id = self.create_id(1, doc_type, 1)
        
        doc_node = Node(
            id=doc_id,
            label=metadata.get("title", "Untitled Document"),
            metadata={
                "type": "document",
                **metadata,
            }
        )
        
        nodes = [doc_node]
        
        # Parse structure
        structural_nodes = self._parse_structure(root, "", metadata)
        nodes.extend(structural_nodes)
        
        # Extract references
        references = self._extract_references(root)
        
        # Add reference metadata to document
        doc_node.metadata["reference_count"] = len(references)
        doc_node.metadata["references"] = [r["href"] for r in references if r["href"]][:20]
        
        return nodes
    
    def find_provision(self, nodes: List[Node], path: str) -> Optional[Node]:
        """Find a specific provision by path."""
        for node in nodes:
            if node.metadata.get("path") == path:
                return node
            if node.metadata.get("eId") == path:
                return node
        return None
    
    def compare_documents(self, doc_a: Any, doc_b: Any) -> Dict[str, Any]:
        """
        Compare structure of two legal documents.
        
        Useful for tracking amendments or comparing jurisdictions.
        """
        nodes_a = self.parse(doc_a)
        nodes_b = self.parse(doc_b)
        
        # Get structural elements
        elements_a = {n.metadata.get("path"): n for n in nodes_a if n.metadata.get("path")}
        elements_b = {n.metadata.get("path"): n for n in nodes_b if n.metadata.get("path")}
        
        shared_paths = set(elements_a.keys()) & set(elements_b.keys())
        only_a = set(elements_a.keys()) - set(elements_b.keys())
        only_b = set(elements_b.keys()) - set(elements_a.keys())
        
        # Compare headings for shared paths
        heading_changes = []
        for path in shared_paths:
            heading_a = elements_a[path].metadata.get("heading", "")
            heading_b = elements_b[path].metadata.get("heading", "")
            if heading_a != heading_b:
                heading_changes.append({
                    "path": path,
                    "heading_a": heading_a,
                    "heading_b": heading_b,
                })
        
        return {
            "structure": {
                "shared_paths": len(shared_paths),
                "only_in_a": list(only_a),
                "only_in_b": list(only_b),
                "heading_changes": heading_changes,
            },
            "element_count_a": len(elements_a),
            "element_count_b": len(elements_b),
            "similarity": len(shared_paths) / len(set(elements_a.keys()) | set(elements_b.keys())) if elements_a or elements_b else 0,
        }
    
    def build_citation_graph(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Build citation graph across multiple documents.
        """
        # Parse all documents
        doc_refs = {}  # doc_uri -> list of outgoing references
        all_uris = set()
        
        for doc in documents:
            nodes = self.parse(doc)
            
            # Find document node
            doc_node = None
            for n in nodes:
                if n.metadata.get("type") == "document":
                    doc_node = n
                    break
            
            if doc_node:
                uri = doc_node.metadata.get("uri", doc_node.label)
                all_uris.add(uri)
                doc_refs[uri] = doc_node.metadata.get("references", [])
        
        # Build graph
        edges = []
        for source_uri, refs in doc_refs.items():
            for target_uri in refs:
                edges.append({
                    "source": source_uri,
                    "target": target_uri,
                })
        
        # Find most cited
        citation_counts = {}
        for edge in edges:
            target = edge["target"]
            citation_counts[target] = citation_counts.get(target, 0) + 1
        
        most_cited = sorted(
            citation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            "documents": len(doc_refs),
            "total_citations": len(edges),
            "edges": edges[:100],  # Limit output
            "most_cited": [{"uri": uri, "count": count} for uri, count in most_cited],
        }
    
    def timeline(self, nodes: List[Node]) -> List[Dict]:
        """
        Build chronological timeline of legal documents.
        """
        timeline = []
        
        for node in nodes:
            if node.metadata.get("type") != "document":
                continue
            
            date = node.metadata.get("date")
            if date:
                timeline.append({
                    "date": date,
                    "title": node.label,
                    "uri": node.metadata.get("uri"),
                    "coordinate": node.id.code,
                })
        
        timeline.sort(key=lambda x: x["date"])
        
        return timeline
