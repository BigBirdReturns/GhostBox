"""
EPUB / Ebook Adapter for axiom-kg

Parses EPUB and ebook metadata into axiom-kg coordinates.

Use cases:
- "How do these books relate?"
- "Same author different pen name?"
- "Build a publishing graph"
- "Find thematically similar works"

EPUB uses OPF (Open Packaging Format) for metadata.

Usage:
    adapter = EPUBAdapter()
    nodes = adapter.parse("book.epub")
    
    # Find related books
    related = adapter.find_related(nodes, library_nodes)
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import zipfile
import hashlib

from .base import XMLAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Book genres/subjects → type_ coordinates
GENRE_TO_TYPE = {
    # Fiction
    "fiction": 1,
    "novel": 1,
    "literary fiction": 2,
    "science fiction": 3,
    "fantasy": 4,
    "mystery": 5,
    "thriller": 6,
    "romance": 7,
    "horror": 8,
    "historical fiction": 9,
    
    # Non-fiction
    "non-fiction": 20,
    "nonfiction": 20,
    "biography": 21,
    "autobiography": 22,
    "history": 23,
    "science": 24,
    "technology": 25,
    "business": 26,
    "self-help": 27,
    "philosophy": 28,
    "religion": 29,
    "politics": 30,
    
    # Academic
    "textbook": 40,
    "reference": 41,
    "academic": 42,
    
    # Other
    "poetry": 50,
    "drama": 51,
    "children": 52,
    "young adult": 53,
}


# Dublin Core namespace
DC_NS = "http://purl.org/dc/elements/1.1/"
OPF_NS = "http://www.idpf.org/2007/opf"


class EPUBAdapter(XMLAdapter):
    """
    Adapter for EPUB ebooks and OPF metadata.
    
    Converts book metadata into axiom-kg coordinates:
    - Books → Entity nodes (Major 1)
    - Authors → Entity nodes (Major 1, type 2)
    - Publishers → Entity nodes (Major 1, type 3)
    - Subjects → Abstract nodes (Major 8)
    """
    
    DOMAIN_NAME = "epub"
    SUPPORTED_EXTENSIONS = [".epub", ".opf"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _genre_to_type(self, subjects: List[str]) -> int:
        """Map subjects/genres to type_ coordinate."""
        for subject in subjects:
            subject_lower = subject.lower()
            for genre, type_ in GENRE_TO_TYPE.items():
                if genre in subject_lower:
                    return type_
        return 99  # Unknown
    
    def _extract_opf_from_epub(self, epub_path: Path) -> str:
        """Extract OPF content from EPUB file."""
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Find container.xml to locate OPF
            container = zf.read("META-INF/container.xml").decode('utf-8')
            
            # Parse to find rootfile
            import xml.etree.ElementTree as ET
            container_root = ET.fromstring(container)
            
            # Find rootfile path
            for rootfile in container_root.iter():
                if 'rootfile' in rootfile.tag:
                    opf_path = rootfile.get('full-path')
                    if opf_path:
                        return zf.read(opf_path).decode('utf-8')
            
            # Fallback: look for .opf file
            for name in zf.namelist():
                if name.endswith('.opf'):
                    return zf.read(name).decode('utf-8')
        
        raise ValueError("Could not find OPF file in EPUB")
    
    def _parse_opf(self, opf_content: str, source_name: str = "") -> List[Node]:
        """Parse OPF metadata into nodes."""
        import xml.etree.ElementTree as ET
        root = ET.fromstring(opf_content)
        
        nodes = []
        
        # Find metadata section
        metadata = None
        for elem in root.iter():
            if 'metadata' in elem.tag.lower():
                metadata = elem
                break
        
        if metadata is None:
            return nodes
        
        # Extract Dublin Core metadata
        title = ""
        creators = []
        subjects = []
        publisher = ""
        date = ""
        description = ""
        language = ""
        identifier = ""
        rights = ""
        
        for elem in metadata:
            tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
            text = elem.text or ""
            
            if tag == "title":
                title = text
            elif tag == "creator":
                creators.append({
                    "name": text,
                    "role": elem.get(f"{{{OPF_NS}}}role", elem.get("role", "author")),
                })
            elif tag == "subject":
                subjects.append(text)
            elif tag == "publisher":
                publisher = text
            elif tag == "date":
                date = text
            elif tag == "description":
                description = text
            elif tag == "language":
                language = text
            elif tag == "identifier":
                identifier = text
            elif tag == "rights":
                rights = text
        
        # Create book node
        major = 1  # Entity
        type_ = self._genre_to_type(subjects)
        subtype = 1  # Book
        
        book_id = self.create_id(major, type_, subtype)
        
        book_metadata = {
            "title": title,
            "creators": creators,
            "subjects": subjects,
            "publisher": publisher,
            "date": date,
            "description": description[:500] if description else None,
            "language": language,
            "identifier": identifier,
            "rights": rights,
            "source": source_name,
        }
        
        book_node = Node(id=book_id, label=title or "Untitled", metadata=book_metadata)
        nodes.append(book_node)
        
        # Create author nodes
        for creator in creators:
            if creator.get("role", "").lower() in ["author", "aut", ""]:
                author_id = self.create_id(1, 2, 1)  # Entity > Person > Author
                author_node = Node(
                    id=author_id,
                    label=creator["name"],
                    metadata={
                        "type": "author",
                        "name": creator["name"],
                    }
                )
                nodes.append(author_node)
                book_node.add_relation(RelationType.PART_OF, author_node)
        
        # Create publisher node if present
        if publisher:
            pub_id = self.create_id(1, 3, 1)  # Entity > Organization > Publisher
            pub_node = Node(
                id=pub_id,
                label=publisher,
                metadata={
                    "type": "publisher",
                    "name": publisher,
                }
            )
            nodes.append(pub_node)
            book_node.add_relation(RelationType.PART_OF, pub_node)
        
        # Create subject nodes
        for subject in subjects:
            subj_id = self.create_id(8, 1, 1)  # Abstract > Category > Subject
            subj_node = Node(
                id=subj_id,
                label=subject,
                metadata={
                    "type": "subject",
                    "name": subject,
                }
            )
            nodes.append(subj_node)
            book_node.add_relation(RelationType.HAS_PROPERTY, subj_node)
        
        return nodes
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse EPUB or OPF file into axiom-kg nodes.
        
        Args:
            source: Path to .epub or .opf file
            
        Returns:
            List of Node objects
        """
        path = Path(source) if isinstance(source, str) else source
        
        if path.suffix.lower() == ".epub":
            opf_content = self._extract_opf_from_epub(path)
        elif path.suffix.lower() == ".opf":
            opf_content = path.read_text()
        else:
            # Try as raw OPF content
            opf_content = source if isinstance(source, str) else str(source)
        
        return self._parse_opf(opf_content, str(path))
    
    def parse_library(self, directory: Path) -> List[Node]:
        """Parse all EPUBs in a directory."""
        all_nodes = []
        
        for epub_path in directory.glob("**/*.epub"):
            try:
                nodes = self.parse(epub_path)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"Warning: Failed to parse {epub_path}: {e}")
        
        return all_nodes
    
    def find_by_author(self, nodes: List[Node], author_name: str) -> List[Node]:
        """Find all books by a specific author."""
        author_lower = author_name.lower()
        
        books = []
        for node in nodes:
            if node.metadata.get("type") == "author":
                continue
            
            creators = node.metadata.get("creators", [])
            for creator in creators:
                if author_lower in creator.get("name", "").lower():
                    books.append(node)
                    break
        
        return books
    
    def find_related(self, book: Node, library: List[Node], min_shared_subjects: int = 1) -> List[Dict]:
        """
        Find books related to a given book by shared subjects.
        """
        book_subjects = set(s.lower() for s in book.metadata.get("subjects", []))
        
        if not book_subjects:
            return []
        
        related = []
        
        for node in library:
            # Skip non-book nodes
            if node.metadata.get("type") in ["author", "publisher", "subject"]:
                continue
            
            # Skip same book
            if node.label == book.label:
                continue
            
            node_subjects = set(s.lower() for s in node.metadata.get("subjects", []))
            shared = book_subjects & node_subjects
            
            if len(shared) >= min_shared_subjects:
                related.append({
                    "title": node.label,
                    "shared_subjects": list(shared),
                    "shared_count": len(shared),
                    "coordinate_distance": book.id.distance(node.id),
                })
        
        return sorted(related, key=lambda x: x["shared_count"], reverse=True)
    
    def author_graph(self, nodes: List[Node]) -> Dict[str, Any]:
        """
        Build author relationship graph.
        
        Finds:
        - Authors with multiple books
        - Potential pen names (same coordinates, different names)
        - Co-authors
        """
        author_books = {}  # author_name -> list of books
        
        for node in nodes:
            if node.metadata.get("type") in ["author", "publisher", "subject"]:
                continue
            
            creators = node.metadata.get("creators", [])
            for creator in creators:
                name = creator.get("name", "")
                if name:
                    if name not in author_books:
                        author_books[name] = []
                    author_books[name].append(node.label)
        
        # Find prolific authors
        prolific = [
            {"name": name, "book_count": len(books), "books": books}
            for name, books in author_books.items()
            if len(books) > 1
        ]
        prolific.sort(key=lambda x: x["book_count"], reverse=True)
        
        return {
            "total_authors": len(author_books),
            "prolific_authors": prolific[:20],
        }
    
    def subject_analysis(self, nodes: List[Node]) -> Dict[str, Any]:
        """
        Analyze subject distribution across library.
        """
        subject_counts = {}
        
        for node in nodes:
            if node.metadata.get("type") in ["author", "publisher", "subject"]:
                continue
            
            for subject in node.metadata.get("subjects", []):
                subject_lower = subject.lower()
                subject_counts[subject_lower] = subject_counts.get(subject_lower, 0) + 1
        
        # Sort by count
        sorted_subjects = sorted(
            subject_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "unique_subjects": len(subject_counts),
            "top_subjects": [
                {"subject": s, "count": c}
                for s, c in sorted_subjects[:30]
            ],
        }
