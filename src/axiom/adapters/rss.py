"""
RSS/Atom Feed Adapter for axiom-kg

Parses RSS 2.0 and Atom feeds into axiom-kg coordinates.

Use cases:
- "Show me how this news source's coverage shifted over time"
- "Track topic drift across multiple feeds"
- "Detect when sources converge on a narrative"
- "Compare coverage of same event across outlets"

Usage:
    adapter = RSSAdapter()
    nodes = adapter.parse("feed.xml")
    
    # Track topics over time
    timeline = adapter.topic_timeline(nodes)
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from .base import XMLAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Content categories → type_ coordinates
CATEGORY_TO_TYPE = {
    # News categories
    "news": 1,
    "politics": 2,
    "business": 3,
    "technology": 4,
    "science": 5,
    "health": 6,
    "sports": 7,
    "entertainment": 8,
    "opinion": 9,
    "editorial": 9,
    
    # Content types
    "article": 10,
    "video": 11,
    "podcast": 12,
    "image": 13,
    "gallery": 14,
}


class RSSAdapter(XMLAdapter):
    """
    Adapter for RSS and Atom feeds.
    
    Converts feed items into axiom-kg coordinates:
    - Feed items → Entity nodes (Major 1, type based on category)
    - Feed source → Entity node (Major 1, type 1 = source)
    - Categories/tags → Abstract nodes (Major 8)
    - Timestamps → Time nodes (Major 6)
    """
    
    DOMAIN_NAME = "rss"
    SUPPORTED_EXTENSIONS = [".xml", ".rss", ".atom"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _detect_format(self, root) -> str:
        """Detect if RSS or Atom."""
        if root.tag == "rss" or root.tag == "channel":
            return "rss"
        if "atom" in root.tag.lower() or root.tag == "feed":
            return "atom"
        # Check for RSS inside
        if root.find("channel") is not None:
            return "rss"
        return "unknown"
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
        
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",        # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _category_to_type(self, categories: List[str]) -> int:
        """Map categories to type_ coordinate."""
        for cat in categories:
            cat_lower = cat.lower()
            for key, type_ in CATEGORY_TO_TYPE.items():
                if key in cat_lower:
                    return type_
        return 99  # Unknown
    
    def _text_to_hash_subtype(self, text: str) -> int:
        """Hash text content to subtype for clustering similar content."""
        if not text:
            return 1
        # Simple hash to 1-99 range
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return (h % 99) + 1
    
    def _extract_text(self, element) -> str:
        """Extract text from element, handling CDATA."""
        if element is None:
            return ""
        text = element.text or ""
        return text.strip()
    
    def _parse_rss_item(self, item, feed_title: str, feed_url: str) -> Node:
        """Parse RSS item element."""
        # Items are Entities (Major 1)
        major = 1
        
        # Get categories
        categories = [self._extract_text(c) for c in item.findall("category")]
        type_ = self._category_to_type(categories)
        
        # Hash title for subtype clustering
        title = self._extract_text(item.find("title"))
        subtype = self._text_to_hash_subtype(title)
        
        sem_id = self.create_id(major, type_, subtype)
        
        # Extract metadata
        link = self._extract_text(item.find("link"))
        description = self._extract_text(item.find("description"))
        pub_date = self._parse_date(self._extract_text(item.find("pubDate")))
        guid = self._extract_text(item.find("guid"))
        author = self._extract_text(item.find("author")) or self._extract_text(item.find("{http://purl.org/dc/elements/1.1/}creator"))
        
        metadata = {
            "feed_title": feed_title,
            "feed_url": feed_url,
            "link": link,
            "description": description[:500] if description else None,
            "pub_date": pub_date.isoformat() if pub_date else None,
            "guid": guid,
            "author": author,
            "categories": categories,
        }
        
        return Node(id=sem_id, label=title or "Untitled", metadata=metadata)
    
    def _parse_atom_entry(self, entry, feed_title: str, feed_url: str) -> Node:
        """Parse Atom entry element."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        # Entries are Entities (Major 1)
        major = 1
        
        # Get categories
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns) + entry.findall("category")]
        type_ = self._category_to_type(categories)
        
        # Get title
        title_elem = entry.find("atom:title", ns) or entry.find("title")
        title = self._extract_text(title_elem)
        subtype = self._text_to_hash_subtype(title)
        
        sem_id = self.create_id(major, type_, subtype)
        
        # Extract metadata
        link_elem = entry.find("atom:link", ns) or entry.find("link")
        link = link_elem.get("href") if link_elem is not None else ""
        
        summary_elem = entry.find("atom:summary", ns) or entry.find("summary")
        summary = self._extract_text(summary_elem)
        
        content_elem = entry.find("atom:content", ns) or entry.find("content")
        content = self._extract_text(content_elem)
        
        updated_elem = entry.find("atom:updated", ns) or entry.find("updated")
        pub_date = self._parse_date(self._extract_text(updated_elem))
        
        id_elem = entry.find("atom:id", ns) or entry.find("id")
        entry_id = self._extract_text(id_elem)
        
        author_elem = entry.find("atom:author/atom:name", ns) or entry.find("author/name")
        author = self._extract_text(author_elem)
        
        metadata = {
            "feed_title": feed_title,
            "feed_url": feed_url,
            "link": link,
            "summary": summary[:500] if summary else None,
            "content": content[:500] if content else None,
            "pub_date": pub_date.isoformat() if pub_date else None,
            "entry_id": entry_id,
            "author": author,
            "categories": categories,
        }
        
        return Node(id=sem_id, label=title or "Untitled", metadata=metadata)
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse RSS/Atom feed into axiom-kg nodes.
        
        Args:
            source: Feed XML as file path or string
            
        Returns:
            List of Node objects
        """
        root = self.load_xml(source)
        format_type = self._detect_format(root)
        
        nodes = []
        
        if format_type == "rss":
            channel = root.find("channel") or root
            feed_title = self._extract_text(channel.find("title"))
            feed_url = self._extract_text(channel.find("link"))
            
            for item in channel.findall("item"):
                node = self._parse_rss_item(item, feed_title, feed_url)
                nodes.append(node)
        
        elif format_type == "atom":
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            title_elem = root.find("atom:title", ns) or root.find("title")
            feed_title = self._extract_text(title_elem)
            
            link_elem = root.find("atom:link", ns) or root.find("link")
            feed_url = link_elem.get("href") if link_elem is not None else ""
            
            entries = root.findall("atom:entry", ns) + root.findall("entry")
            for entry in entries:
                node = self._parse_atom_entry(entry, feed_title, feed_url)
                nodes.append(node)
        
        return nodes
    
    def parse_url(self, url: str) -> List[Node]:
        """Fetch and parse feed from URL."""
        import urllib.request
        
        with urllib.request.urlopen(url) as response:
            xml_str = response.read().decode()
        
        return self.parse(xml_str)
    
    def topic_timeline(self, nodes: List[Node]) -> Dict[str, List[Dict]]:
        """
        Group items by topic/category over time.
        
        Returns:
            Dict mapping category → list of items with timestamps
        """
        timeline = {}
        
        for node in nodes:
            categories = node.metadata.get("categories", ["uncategorized"])
            pub_date = node.metadata.get("pub_date")
            
            for cat in categories:
                if cat not in timeline:
                    timeline[cat] = []
                
                timeline[cat].append({
                    "title": node.label,
                    "date": pub_date,
                    "link": node.metadata.get("link"),
                    "coordinate": node.id.code,
                })
        
        # Sort by date
        for cat in timeline:
            timeline[cat].sort(key=lambda x: x["date"] or "")
        
        return timeline
    
    def compare_feeds(self, feed_a: Any, feed_b: Any) -> Dict[str, Any]:
        """
        Compare two feeds for topic overlap and divergence.
        """
        nodes_a = self.parse(feed_a)
        nodes_b = self.parse(feed_b)
        
        # Get categories
        cats_a = set()
        cats_b = set()
        
        for n in nodes_a:
            cats_a.update(n.metadata.get("categories", []))
        for n in nodes_b:
            cats_b.update(n.metadata.get("categories", []))
        
        shared_topics = cats_a & cats_b
        only_a = cats_a - cats_b
        only_b = cats_b - cats_a
        
        # Find similar titles (potential same story)
        similar_stories = []
        for na in nodes_a:
            for nb in nodes_b:
                # Same coordinate subtype = similar content hash
                if na.id.subtype == nb.id.subtype:
                    similar_stories.append({
                        "feed_a_title": na.label,
                        "feed_b_title": nb.label,
                        "coordinate_distance": na.id.distance(nb.id),
                    })
        
        return {
            "shared_topics": list(shared_topics),
            "only_in_a": list(only_a),
            "only_in_b": list(only_b),
            "similar_stories": similar_stories[:20],  # Limit output
            "item_count_a": len(nodes_a),
            "item_count_b": len(nodes_b),
        }
