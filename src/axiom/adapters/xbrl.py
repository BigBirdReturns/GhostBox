"""
XBRL Financial Reporting Adapter for axiom-kg

Parses XBRL (eXtensible Business Reporting Language) financial statements.

Use cases:
- "Compare these two companies' financials structurally"
- "Track how a company's reporting changed over time"
- "Find companies with similar financial structure"
- "Detect unusual reporting patterns"

XBRL is XML-based. Most SEC filings use XBRL for structured data.

Usage:
    adapter = XBRLAdapter()
    nodes = adapter.parse("10k_filing.xml")
    
    # Compare companies
    diff = adapter.compare_filings(company_a, company_b)
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re

from .base import XMLAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# XBRL concept categories → axiom-kg coordinates
CONCEPT_CATEGORY_TO_COORDS = {
    # Assets (Major 1 - Entity, type 1)
    "Assets": (1, 1, 1),
    "CurrentAssets": (1, 1, 2),
    "NoncurrentAssets": (1, 1, 3),
    "Cash": (1, 1, 4),
    "Inventory": (1, 1, 5),
    "PropertyPlantEquipment": (1, 1, 6),
    
    # Liabilities (Major 1 - Entity, type 2)
    "Liabilities": (1, 2, 1),
    "CurrentLiabilities": (1, 2, 2),
    "NoncurrentLiabilities": (1, 2, 3),
    "Debt": (1, 2, 4),
    "AccountsPayable": (1, 2, 5),
    
    # Equity (Major 1 - Entity, type 3)
    "Equity": (1, 3, 1),
    "StockholdersEquity": (1, 3, 1),
    "RetainedEarnings": (1, 3, 2),
    "CommonStock": (1, 3, 3),
    
    # Revenue (Major 7 - Quantity, type 1)
    "Revenue": (7, 1, 1),
    "Revenues": (7, 1, 1),
    "SalesRevenue": (7, 1, 2),
    "ServiceRevenue": (7, 1, 3),
    
    # Expenses (Major 7 - Quantity, type 2)
    "Expenses": (7, 2, 1),
    "CostOfRevenue": (7, 2, 2),
    "OperatingExpenses": (7, 2, 3),
    "ResearchAndDevelopment": (7, 2, 4),
    "SellingGeneralAdministrative": (7, 2, 5),
    
    # Income (Major 7 - Quantity, type 3)
    "NetIncome": (7, 3, 1),
    "OperatingIncome": (7, 3, 2),
    "GrossProfit": (7, 3, 3),
    "EarningsPerShare": (7, 3, 4),
    
    # Cash Flow (Major 7 - Quantity, type 4)
    "CashFlow": (7, 4, 1),
    "OperatingCashFlow": (7, 4, 2),
    "InvestingCashFlow": (7, 4, 3),
    "FinancingCashFlow": (7, 4, 4),
}


# Common XBRL namespaces
XBRL_NAMESPACES = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "us-gaap": "http://fasb.org/us-gaap/2023",
    "dei": "http://xbrl.sec.gov/dei/2023",
    "link": "http://www.xbrl.org/2003/linkbase",
}


class XBRLAdapter(XMLAdapter):
    """
    Adapter for XBRL financial reports.
    
    Converts XBRL facts into axiom-kg coordinates:
    - Financial concepts → Entity/Quantity nodes
    - Contexts (periods) → Time metadata
    - Units → Property metadata
    """
    
    DOMAIN_NAME = "xbrl"
    SUPPORTED_EXTENSIONS = [".xml", ".xbrl"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
        self._concept_cache = {}
    
    def _get_coords(self, concept_name: str) -> Tuple[int, int, int]:
        """Get coordinates for an XBRL concept."""
        # Check direct mapping
        for key, coords in CONCEPT_CATEGORY_TO_COORDS.items():
            if key.lower() in concept_name.lower():
                return coords
        
        # Default to Quantity (financial data)
        return (7, 99, 1)
    
    def _clean_concept_name(self, name: str) -> str:
        """Clean concept name for display."""
        # Remove namespace prefix
        if ":" in name:
            name = name.split(":")[-1]
        
        # Add spaces before capitals
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        
        return name
    
    def _parse_contexts(self, root) -> Dict[str, Dict]:
        """Parse XBRL context elements."""
        contexts = {}
        
        # Try with namespace
        for ns_prefix, ns_uri in XBRL_NAMESPACES.items():
            for ctx in root.findall(f".//{{{ns_uri}}}context"):
                ctx_id = ctx.get("id")
                if ctx_id:
                    contexts[ctx_id] = self._extract_context(ctx, ns_uri)
        
        # Try without namespace
        for ctx in root.findall(".//context"):
            ctx_id = ctx.get("id")
            if ctx_id:
                contexts[ctx_id] = self._extract_context(ctx, None)
        
        return contexts
    
    def _extract_context(self, ctx, ns_uri: Optional[str]) -> Dict:
        """Extract context details."""
        result = {"id": ctx.get("id")}
        
        # Entity identifier
        if ns_uri:
            entity = ctx.find(f".//{{{ns_uri}}}identifier")
        else:
            entity = ctx.find(".//identifier")
        
        if entity is not None:
            result["entity"] = entity.text
        
        # Period
        if ns_uri:
            instant = ctx.find(f".//{{{ns_uri}}}instant")
            start = ctx.find(f".//{{{ns_uri}}}startDate")
            end = ctx.find(f".//{{{ns_uri}}}endDate")
        else:
            instant = ctx.find(".//instant")
            start = ctx.find(".//startDate")
            end = ctx.find(".//endDate")
        
        if instant is not None:
            result["instant"] = instant.text
            result["period_type"] = "instant"
        elif start is not None and end is not None:
            result["start_date"] = start.text
            result["end_date"] = end.text
            result["period_type"] = "duration"
        
        return result
    
    def _parse_units(self, root) -> Dict[str, str]:
        """Parse XBRL unit elements."""
        units = {}
        
        for ns_uri in XBRL_NAMESPACES.values():
            for unit in root.findall(f".//{{{ns_uri}}}unit"):
                unit_id = unit.get("id")
                measure = unit.find(f".//{{{ns_uri}}}measure")
                if unit_id and measure is not None:
                    units[unit_id] = measure.text
        
        # Without namespace
        for unit in root.findall(".//unit"):
            unit_id = unit.get("id")
            measure = unit.find(".//measure")
            if unit_id and measure is not None:
                units[unit_id] = measure.text
        
        return units
    
    def _parse_facts(self, root, contexts: Dict, units: Dict) -> List[Node]:
        """Parse XBRL fact elements into nodes."""
        nodes = []
        
        # Get all elements that have contextRef (these are facts)
        for elem in root.iter():
            ctx_ref = elem.get("contextRef")
            if ctx_ref is None:
                continue
            
            # Get concept name from tag
            tag = elem.tag
            if "}" in tag:
                # Remove namespace
                concept = tag.split("}")[-1]
            else:
                concept = tag
            
            # Skip non-financial elements
            if concept in ["context", "unit", "schemaRef"]:
                continue
            
            value = elem.text
            if value is None:
                continue
            
            # Get coordinates
            major, type_, subtype = self._get_coords(concept)
            sem_id = self.create_id(major, type_, subtype)
            
            # Get context info
            ctx = contexts.get(ctx_ref, {})
            
            # Get unit info
            unit_ref = elem.get("unitRef")
            unit = units.get(unit_ref, "")
            
            # Parse numeric value
            numeric_value = None
            try:
                # Handle decimals attribute
                decimals = elem.get("decimals", "0")
                if decimals == "INF":
                    numeric_value = float(value)
                else:
                    numeric_value = float(value)
            except (ValueError, TypeError):
                pass
            
            label = self._clean_concept_name(concept)
            
            metadata = {
                "concept": concept,
                "value": value,
                "numeric_value": numeric_value,
                "unit": unit,
                "context_id": ctx_ref,
                "period_type": ctx.get("period_type"),
                "instant": ctx.get("instant"),
                "start_date": ctx.get("start_date"),
                "end_date": ctx.get("end_date"),
                "entity": ctx.get("entity"),
                "decimals": elem.get("decimals"),
            }
            
            node = Node(id=sem_id, label=label, metadata=metadata)
            nodes.append(node)
        
        return nodes
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse XBRL document into axiom-kg nodes.
        
        Args:
            source: XBRL file path or XML string
            
        Returns:
            List of Node objects
        """
        root = self.load_xml(source)
        
        # Parse supporting structures
        contexts = self._parse_contexts(root)
        units = self._parse_units(root)
        
        # Parse facts
        nodes = self._parse_facts(root, contexts, units)
        
        return nodes
    
    def get_financial_summary(self, nodes: List[Node]) -> Dict[str, Any]:
        """
        Extract key financial metrics from parsed nodes.
        """
        summary = {
            "revenue": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "equity": None,
            "cash": None,
            "periods": set(),
        }
        
        for node in nodes:
            concept = node.metadata.get("concept", "").lower()
            value = node.metadata.get("numeric_value")
            period = node.metadata.get("end_date") or node.metadata.get("instant")
            
            if period:
                summary["periods"].add(period)
            
            if value is None:
                continue
            
            # Match key concepts
            if "revenue" in concept and "cost" not in concept:
                if summary["revenue"] is None or value > summary["revenue"]:
                    summary["revenue"] = value
            elif "netincome" in concept or "netloss" in concept:
                summary["net_income"] = value
            elif concept == "assets" or "totalassets" in concept:
                summary["total_assets"] = value
            elif concept == "liabilities" or "totalliabilities" in concept:
                summary["total_liabilities"] = value
            elif "stockholdersequity" in concept or "equity" in concept:
                summary["equity"] = value
            elif "cashandcashequivalents" in concept:
                summary["cash"] = value
        
        summary["periods"] = sorted(list(summary["periods"]))
        
        return summary
    
    def compare_filings(self, filing_a: Any, filing_b: Any) -> Dict[str, Any]:
        """
        Compare two XBRL filings structurally.
        """
        nodes_a = self.parse(filing_a)
        nodes_b = self.parse(filing_b)
        
        summary_a = self.get_financial_summary(nodes_a)
        summary_b = self.get_financial_summary(nodes_b)
        
        # Get concepts used
        concepts_a = {n.metadata.get("concept") for n in nodes_a}
        concepts_b = {n.metadata.get("concept") for n in nodes_b}
        
        shared_concepts = concepts_a & concepts_b
        only_a = concepts_a - concepts_b
        only_b = concepts_b - concepts_a
        
        # Compare key metrics
        comparisons = {}
        for key in ["revenue", "net_income", "total_assets", "total_liabilities", "equity", "cash"]:
            val_a = summary_a.get(key)
            val_b = summary_b.get(key)
            
            if val_a is not None and val_b is not None and val_a != 0:
                change_pct = ((val_b - val_a) / abs(val_a)) * 100
            else:
                change_pct = None
            
            comparisons[key] = {
                "filing_a": val_a,
                "filing_b": val_b,
                "change_pct": change_pct,
            }
        
        return {
            "concepts": {
                "shared": len(shared_concepts),
                "only_in_a": len(only_a),
                "only_in_b": len(only_b),
                "overlap_ratio": len(shared_concepts) / len(concepts_a | concepts_b) if concepts_a or concepts_b else 0,
            },
            "metrics": comparisons,
            "periods_a": summary_a["periods"],
            "periods_b": summary_b["periods"],
            "fact_count_a": len(nodes_a),
            "fact_count_b": len(nodes_b),
        }
    
    def track_concept_over_time(self, filings: List[Any], concept_name: str) -> List[Dict]:
        """
        Track a specific concept across multiple filings.
        """
        timeline = []
        
        for filing in filings:
            nodes = self.parse(filing)
            
            for node in nodes:
                if concept_name.lower() in node.metadata.get("concept", "").lower():
                    timeline.append({
                        "value": node.metadata.get("numeric_value"),
                        "period_end": node.metadata.get("end_date") or node.metadata.get("instant"),
                        "unit": node.metadata.get("unit"),
                        "concept": node.metadata.get("concept"),
                    })
        
        # Sort by date
        timeline.sort(key=lambda x: x["period_end"] or "")
        
        return timeline
    
    def find_similar_structure(self, target: Any, candidates: List[Any], threshold: float = 0.7) -> List[Dict]:
        """
        Find filings with similar concept structure to target.
        """
        target_nodes = self.parse(target)
        target_concepts = {n.metadata.get("concept") for n in target_nodes}
        
        similar = []
        
        for i, candidate in enumerate(candidates):
            candidate_nodes = self.parse(candidate)
            candidate_concepts = {n.metadata.get("concept") for n in candidate_nodes}
            
            # Jaccard similarity
            intersection = len(target_concepts & candidate_concepts)
            union = len(target_concepts | candidate_concepts)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                similar.append({
                    "candidate_index": i,
                    "similarity": similarity,
                    "shared_concepts": intersection,
                    "total_concepts": union,
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
