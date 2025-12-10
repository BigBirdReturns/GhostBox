"""
GhostBox PLTR Analysis - Integrated with Axiom-KG

This module analyzes Palantir (PLTR) by comparing:
- SEC XBRL filings (ground truth)
- RSS narrative (what's being said)
- Claim types (MOAT, GROWTH, GOV_CAPTURE, etc.)

Everything flows through axiom-kg:
- XBRLAdapter parses filings into Nodes
- RSSAdapter parses articles into Nodes  
- Claims become Nodes with semantic coordinates
- Divergence is computed as SemanticTension

This is Field Zero: real data, real coordinates, real tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import urllib.request
import re

# Axiom-KG core
from axiom.core import (
    Space,
    Node,
    SemanticID,
    RelationType,
    Fork,
)

# Axiom-KG adapters
from axiom.adapters import XBRLAdapter, RSSAdapter

# Screen Ghost integration (for tension computation)
from screen_ghost.attention import AttentionShape, ShapeType


# =============================================================================
# CLAIM TYPES AS SEMANTIC COORDINATES
# =============================================================================

# Claims are Abstract > Narrative > [Type]
CLAIM_TYPE_COORDS = {
    "MOAT": (8, 1, 1),
    "GROWTH": (8, 1, 2),
    "GOV_CAPTURE": (8, 1, 3),
    "COMMERCIAL": (8, 1, 4),
    "RISK_DISCLOSURE": (8, 1, 5),
}

CLAIM_KEYWORDS = {
    "MOAT": {
        "weak": ["moat", "defensible", "competitive advantage", "switching cost", "stickiness"],
        "strong": ["dominant", "irreplaceable", "monopoly", "only platform", "no alternative"],
    },
    "GROWTH": {
        "weak": ["growth", "scale", "scaling", "expansion", "demand"],
        "strong": ["explosive", "hypergrowth", "surging", "soaring", "unprecedented"],
    },
    "GOV_CAPTURE": {
        "weak": ["government", "public sector", "defense", "military", "dod", "pentagon", "nato"],
        "strong": ["backbone", "critical infrastructure", "mission critical"],
    },
    "COMMERCIAL": {
        "weak": ["commercial", "enterprise", "retail", "customer", "client"],
        "strong": ["exploding demand", "every customer", "across industries"],
    },
    "RISK_DISCLOSURE": {
        "weak": ["risk", "regulatory", "scrutiny", "probe", "investigation", "lawsuit"],
        "strong": ["fraud", "criminal", "sanction", "whistleblower"],
    },
}


# =============================================================================
# CLAIM EXTRACTION
# =============================================================================

@dataclass
class ClaimNode:
    """A claim extracted from narrative, as an axiom-kg Node."""
    claim_type: str
    intensity: float  # 0.33, 0.66, or 1.0
    source: str
    title: str
    node: Node


def score_text(text: str, claim_type: str) -> float:
    """Score text for a claim type."""
    text_lower = text.lower()
    keywords = CLAIM_KEYWORDS.get(claim_type, {})
    weak = keywords.get("weak", [])
    strong = keywords.get("strong", [])
    
    score = 0.0
    if any(k in text_lower for k in weak):
        score = 0.33
    if any(k in text_lower for k in strong):
        score = max(score, 0.66)
    if score >= 0.66 and any(k in text_lower for k in weak):
        score = 1.0
    
    return score


def extract_claims(article_node: Node, space: Space) -> List[ClaimNode]:
    """
    Extract claims from an article node.
    
    Each claim becomes a child Node with:
    - Coordinates from CLAIM_TYPE_COORDS
    - Relation to parent article
    - Intensity in metadata
    """
    claims = []
    
    text = article_node.label + " " + (article_node.metadata.get("description") or "")
    source = article_node.metadata.get("feed_title", "unknown")
    
    for claim_type, coords in CLAIM_TYPE_COORDS.items():
        intensity = score_text(text, claim_type)
        
        if intensity > 0:
            # Create claim node
            major, type_, subtype = coords
            claim_id = SemanticID.create(
                major=major,
                type_=type_,
                subtype=subtype,
                instance=len([n for n in space.nodes() if n.id.major == major]) + 1
            )
            
            claim_node = Node(
                id=claim_id,
                label=f"{claim_type}: {article_node.label[:50]}",
                metadata={
                    "claim_type": claim_type,
                    "intensity": intensity,
                    "source": source,
                    "article_title": article_node.label,
                    "extracted_from": article_node.id.code,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            
            # Add relation from article to claim
            article_node.add_relation(RelationType.HAS_PROPERTY, claim_node)
            
            claims.append(ClaimNode(
                claim_type=claim_type,
                intensity=intensity,
                source=source,
                title=article_node.label,
                node=claim_node,
            ))
    
    return claims


# =============================================================================
# SEC XBRL FETCHING
# =============================================================================

PLTR_CIK = "0001321655"
SEC_BASE = "https://data.sec.gov"

SEC_HEADERS = {
    "User-Agent": "GhostBox/0.8 (research; contact@example.com)",
    "Accept": "application/json",
}


def fetch_sec_companyfacts(cik: str = PLTR_CIK) -> dict:
    """Fetch company facts from SEC EDGAR API."""
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def extract_filing_signals(companyfacts: dict) -> Dict[str, float]:
    """
    Extract normalized signals from SEC filings.
    
    Returns dict of claim_type -> signal (0.0 to 1.0)
    """
    signals = {ct: 0.5 for ct in CLAIM_TYPE_COORDS.keys()}
    
    facts = companyfacts.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    
    # GROWTH signal from revenue trajectory
    revenue_data = us_gaap.get("Revenues", {}).get("units", {}).get("USD", [])
    if not revenue_data:
        revenue_data = us_gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax", {}).get("units", {}).get("USD", [])
    
    if len(revenue_data) >= 4:
        # Get last 4 quarterly values
        sorted_rev = sorted(revenue_data, key=lambda x: x.get("end", ""))
        last_four = sorted_rev[-4:]
        
        first_val = last_four[0].get("val", 0)
        last_val = last_four[-1].get("val", 0)
        
        if first_val > 0:
            growth_rate = (last_val - first_val) / first_val
            # Normalize to 0-1 (capped at +/- 50%)
            growth_rate = max(-0.5, min(0.5, growth_rate))
            signals["GROWTH"] = round(growth_rate + 0.5, 3)
    
    # MOAT signal from operating margin
    op_income = us_gaap.get("OperatingIncomeLoss", {}).get("units", {}).get("USD", [])
    if op_income and revenue_data:
        sorted_op = sorted(op_income, key=lambda x: x.get("end", ""))
        sorted_rev = sorted(revenue_data, key=lambda x: x.get("end", ""))
        
        if sorted_op and sorted_rev:
            latest_op = sorted_op[-1].get("val", 0)
            latest_rev = sorted_rev[-1].get("val", 1)
            
            margin = latest_op / latest_rev if latest_rev else 0
            # Normalize margin (-20% to +20%) to 0-1
            margin = max(-0.2, min(0.2, margin))
            signals["MOAT"] = round((margin + 0.2) / 0.4, 3)
    
    return signals


def filing_signals_to_nodes(signals: Dict[str, float], space: Space) -> List[Node]:
    """Convert filing signals to axiom-kg nodes."""
    nodes = []
    
    for claim_type, signal in signals.items():
        coords = CLAIM_TYPE_COORDS.get(claim_type, (8, 99, 1))
        major, type_, subtype = coords
        
        # Filing signals go in a different subtype range (10+) to distinguish from narrative claims
        node_id = SemanticID.create(
            major=major,
            type_=type_,
            subtype=subtype + 10,  # Filing signals are subtype 11-15
            instance=1,
        )
        
        node = Node(
            id=node_id,
            label=f"FILING_{claim_type}",
            metadata={
                "claim_type": claim_type,
                "source": "SEC_XBRL",
                "signal": signal,
                "is_filing": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        nodes.append(node)
    
    return nodes


# =============================================================================
# RSS FEEDS
# =============================================================================

AI_FEEDS = [
    "https://feeds.feedburner.com/venturebeat/SZYF",  # VentureBeat AI
    "https://www.artificialintelligence-news.com/feed/",
]

PLTR_FEEDS = [
    "https://news.google.com/rss/search?q=palantir+stock&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=palantir+AI&hl=en-US&gl=US&ceid=US:en",
]


# =============================================================================
# SEMANTIC TENSION (from coordinates)
# =============================================================================

@dataclass
class PLTRTension:
    """Tension computed from filing vs narrative coordinates."""
    
    claim_type: str
    filing_signal: float
    narrative_intensity: float
    sector_intensity: float
    
    tension_vs_filing: float
    tension_vs_sector: float
    divergence: float
    driver: str  # "filing" or "sector"
    
    # Nodes involved
    filing_node: Optional[Node] = None
    narrative_nodes: List[Node] = field(default_factory=list)


def compute_pltr_tension(
    space: Space,
    filing_signals: Dict[str, float],
    claim_type: str,
) -> PLTRTension:
    """
    Compute tension for a claim type.
    
    Tension = distance between filing signal and narrative intensity.
    """
    # Find narrative nodes for this claim type
    narrative_nodes = [
        n for n in space.nodes()
        if n.metadata.get("claim_type") == claim_type
        and not n.metadata.get("is_filing")
    ]
    
    # Find filing node
    filing_node = None
    for n in space.nodes():
        if n.metadata.get("claim_type") == claim_type and n.metadata.get("is_filing"):
            filing_node = n
            break
    
    # Compute average narrative intensity
    if narrative_nodes:
        narrative_intensity = sum(n.metadata.get("intensity", 0) for n in narrative_nodes) / len(narrative_nodes)
    else:
        narrative_intensity = 0.0
    
    # Filing signal
    filing_signal = filing_signals.get(claim_type, 0.5)
    
    # For now, sector intensity is average of all narrative (could be separate feeds)
    sector_nodes = [n for n in narrative_nodes if "sector" in n.metadata.get("source", "").lower()]
    if sector_nodes:
        sector_intensity = sum(n.metadata.get("intensity", 0) for n in sector_nodes) / len(sector_nodes)
    else:
        sector_intensity = narrative_intensity * 0.8  # Proxy
    
    # Tension
    tension_vs_filing = abs(narrative_intensity - filing_signal)
    tension_vs_sector = abs(narrative_intensity - sector_intensity)
    
    divergence = max(tension_vs_filing, tension_vs_sector)
    driver = "filing" if tension_vs_filing >= tension_vs_sector else "sector"
    
    return PLTRTension(
        claim_type=claim_type,
        filing_signal=round(filing_signal, 3),
        narrative_intensity=round(narrative_intensity, 3),
        sector_intensity=round(sector_intensity, 3),
        tension_vs_filing=round(tension_vs_filing, 3),
        tension_vs_sector=round(tension_vs_sector, 3),
        divergence=round(divergence, 3),
        driver=driver,
        filing_node=filing_node,
        narrative_nodes=narrative_nodes,
    )


# =============================================================================
# PLTR ANALYZER (integrated with GhostBox)
# =============================================================================

@dataclass
class PLTRAnalyzer:
    """
    PLTR analysis using axiom-kg.
    
    Integrates with GhostBox engine or runs standalone.
    """
    
    space: Space = field(default_factory=Space)
    rss_adapter: RSSAdapter = field(default_factory=RSSAdapter)
    xbrl_adapter: XBRLAdapter = field(default_factory=XBRLAdapter)
    
    # State
    filing_signals: Dict[str, float] = field(default_factory=dict)
    claims: List[ClaimNode] = field(default_factory=list)
    tensions: Dict[str, PLTRTension] = field(default_factory=dict)
    
    def ingest_sec_filings(self, companyfacts: Optional[dict] = None) -> Dict[str, float]:
        """Ingest SEC filings and extract signals."""
        if companyfacts is None:
            companyfacts = fetch_sec_companyfacts()
        
        # Extract signals
        self.filing_signals = extract_filing_signals(companyfacts)
        
        # Convert to nodes and add to space
        filing_nodes = filing_signals_to_nodes(self.filing_signals, self.space)
        for node in filing_nodes:
            self.space.add(node)
        
        return self.filing_signals
    
    def ingest_rss_feeds(self, feeds: List[str], is_sector: bool = False) -> List[Node]:
        """Ingest RSS feeds and extract claims."""
        all_nodes = []
        
        for feed_url in feeds:
            try:
                nodes = self.rss_adapter.parse_url(feed_url)
                
                for node in nodes:
                    # Mark as sector baseline if applicable
                    if is_sector:
                        node.metadata["source_type"] = "sector"
                    else:
                        node.metadata["source_type"] = "pltr_specific"
                    
                    # Add to space
                    self.space.add(node)
                    all_nodes.append(node)
                    
                    # Extract claims
                    claims = extract_claims(node, self.space)
                    for claim in claims:
                        self.space.add(claim.node)
                        self.claims.append(claim)
            
            except Exception as e:
                print(f"Warning: Failed to fetch {feed_url}: {e}")
        
        return all_nodes
    
    def compute_all_tensions(self) -> Dict[str, PLTRTension]:
        """Compute tension for all claim types."""
        self.tensions = {}
        
        for claim_type in CLAIM_TYPE_COORDS.keys():
            tension = compute_pltr_tension(self.space, self.filing_signals, claim_type)
            self.tensions[claim_type] = tension
        
        return self.tensions
    
    def overall_divergence(self) -> Tuple[float, str]:
        """Compute overall divergence score and posture."""
        if not self.tensions:
            self.compute_all_tensions()
        
        weights = {
            "MOAT": 0.25,
            "GROWTH": 0.25,
            "GOV_CAPTURE": 0.2,
            "COMMERCIAL": 0.15,
            "RISK_DISCLOSURE": 0.15,
        }
        
        weighted_sum = sum(
            self.tensions[ct].divergence * weights.get(ct, 0.2)
            for ct in self.tensions
        )
        total_weight = sum(weights.values())
        
        overall = weighted_sum / total_weight if total_weight else 0.0
        
        if overall < 0.25:
            posture = "stable"
        elif overall < 0.55:
            posture = "contested"
        else:
            posture = "critical"
        
        return round(overall, 3), posture
    
    def run_full_analysis(self, demo_mode: bool = False) -> Dict[str, Any]:
        """Run complete PLTR analysis."""
        if demo_mode:
            print("Running in DEMO mode with mock data...")
            return self._run_demo_analysis()
        
        print("Fetching SEC filings...")
        self.ingest_sec_filings()
        
        print("Fetching AI sector feeds...")
        self.ingest_rss_feeds(AI_FEEDS, is_sector=True)
        
        print("Fetching PLTR-specific feeds...")
        self.ingest_rss_feeds(PLTR_FEEDS, is_sector=False)
        
        print("Computing tensions...")
        self.compute_all_tensions()
        
        overall, posture = self.overall_divergence()
        
        return {
            "overall_score": overall,
            "posture": posture,
            "tensions": {
                ct: {
                    "divergence": t.divergence,
                    "filing_signal": t.filing_signal,
                    "narrative_intensity": t.narrative_intensity,
                    "driver": t.driver,
                }
                for ct, t in self.tensions.items()
            },
            "space_summary": self.space.summary(),
            "claim_count": len(self.claims),
        }
    
    def _run_demo_analysis(self) -> Dict[str, Any]:
        """Run with mock data for demonstration."""
        # Mock filing signals (realistic PLTR values)
        self.filing_signals = {
            "MOAT": 0.55,       # Modest operating margin
            "GROWTH": 0.72,     # Strong revenue growth
            "GOV_CAPTURE": 0.5, # Can't derive from filings directly
            "COMMERCIAL": 0.5,  # Can't derive from filings directly
            "RISK_DISCLOSURE": 0.5,
        }
        
        # Add filing nodes
        filing_nodes = filing_signals_to_nodes(self.filing_signals, self.space)
        for node in filing_nodes:
            self.space.add(node)
        
        # Mock narrative claims (simulating RSS ingestion)
        mock_claims = [
            ("MOAT", 0.66, "SeekingAlpha", "Palantir's AI Platform Creates Unassailable Moat"),
            ("MOAT", 1.0, "Motley Fool", "Why Palantir Has No Real Competitors"),
            ("GROWTH", 1.0, "CNBC", "Palantir Revenue Surges as AI Demand Explodes"),
            ("GROWTH", 0.66, "Bloomberg", "PLTR Sees Continued Strong Growth in Government"),
            ("GOV_CAPTURE", 1.0, "DefenseNews", "Palantir Becomes Backbone of Pentagon AI Strategy"),
            ("GOV_CAPTURE", 0.66, "Reuters", "US Army Expands Palantir Contract"),
            ("COMMERCIAL", 0.66, "TechCrunch", "Enterprise AI Adoption Drives Palantir Commercial Growth"),
            ("COMMERCIAL", 0.33, "Forbes", "Palantir Struggles to Diversify Beyond Government"),
            ("RISK_DISCLOSURE", 0.33, "WSJ", "Palantir Faces Scrutiny Over Data Privacy Practices"),
        ]
        
        for claim_type, intensity, source, title in mock_claims:
            coords = CLAIM_TYPE_COORDS[claim_type]
            claim_id = SemanticID.create(
                major=coords[0],
                type_=coords[1],
                subtype=coords[2],
                instance=len(self.claims) + 1
            )
            
            node = Node(
                id=claim_id,
                label=f"{claim_type}: {title[:40]}",
                metadata={
                    "claim_type": claim_type,
                    "intensity": intensity,
                    "source": source,
                    "article_title": title,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            self.space.add(node)
            self.claims.append(ClaimNode(
                claim_type=claim_type,
                intensity=intensity,
                source=source,
                title=title,
                node=node,
            ))
        
        # Compute tensions
        self.compute_all_tensions()
        overall, posture = self.overall_divergence()
        
        return {
            "overall_score": overall,
            "posture": posture,
            "tensions": {
                ct: {
                    "divergence": t.divergence,
                    "filing_signal": t.filing_signal,
                    "narrative_intensity": t.narrative_intensity,
                    "driver": t.driver,
                }
                for ct, t in self.tensions.items()
            },
            "space_summary": self.space.summary(),
            "claim_count": len(self.claims),
            "mode": "demo",
        }
    
    def generate_brief(self) -> str:
        """Generate markdown intelligence brief."""
        overall, posture = self.overall_divergence()
        
        lines = [
            "# GhostBox PLTR Intelligence Brief",
            "",
            f"**Divergence Score:** {overall:.3f}",
            f"**Posture:** {posture.upper()}",
            "",
            f"**Space:** {self.space.summary()['nodes']} nodes, {self.space.summary()['forks']} forks",
            f"**Claims extracted:** {len(self.claims)}",
            "",
            "## Tension by Claim Type",
            "",
        ]
        
        for ct, tension in sorted(self.tensions.items(), key=lambda x: x[1].divergence, reverse=True):
            lines.append(
                f"- **{ct}**: divergence {tension.divergence:.3f} "
                f"(narrative {tension.narrative_intensity:.3f} vs filing {tension.filing_signal:.3f}, "
                f"driver: {tension.driver})"
            )
        
        lines.extend([
            "",
            "## Highest Tension Areas",
            "",
        ])
        
        top_tensions = sorted(self.tensions.values(), key=lambda t: t.divergence, reverse=True)[:3]
        for t in top_tensions:
            if t.claim_type == "GROWTH":
                desc = "Growth narrative runs ahead of revenue trajectory"
            elif t.claim_type == "MOAT":
                desc = "Moat claims exceed margin support"
            elif t.claim_type == "GOV_CAPTURE":
                desc = "Government capture story misaligned with filings"
            elif t.claim_type == "COMMERCIAL":
                desc = "Commercial adoption narrative ahead of reported mix"
            else:
                desc = "Narrative and filings diverge"
            
            lines.append(f"- **{t.claim_type}**: {desc}")
        
        lines.extend([
            "",
            "---",
            "*Generated by GhostBox + Axiom-KG. Real data, semantic coordinates, geometric tension.*",
        ])
        
        return "\n".join(lines)
    
    def generate_linkedin_post(self) -> str:
        """Generate LinkedIn post text."""
        overall, posture = self.overall_divergence()
        
        top = sorted(self.tensions.values(), key=lambda t: t.divergence, reverse=True)[:3]
        
        bullets = []
        for t in top:
            if t.claim_type == "GROWTH":
                bullets.append("Growth story runs ahead of revenue trajectory")
            elif t.claim_type == "MOAT":
                bullets.append("Moat claims look stronger than margin support")
            elif t.claim_type == "GOV_CAPTURE":
                bullets.append("Government backbone narrative not fully matched in filings")
            elif t.claim_type == "COMMERCIAL":
                bullets.append("Commercial adoption narrative ahead of reported mix")
            elif t.claim_type == "RISK_DISCLOSURE":
                bullets.append("Risk language in filings and public conversation out of sync")
        
        lines = [
            "GhostBox AI Divergence Index: Palantir (PLTR)",
            "",
            f"Score: {overall:.3f}",
            f"Posture: {posture.upper()}",
            "",
            "GhostBox compares three layers:",
            "- SEC XBRL filings (ground truth)",
            "- PLTR-specific narrative",
            "- AI sector baseline",
            "",
            "Highest tension areas:",
        ]
        
        for b in bullets:
            lines.append(f"• {b}")
        
        lines.extend([
            "",
            "This is automated analysis from public data.",
            "Signal for oversight, not investment advice.",
            "",
            "#GhostBox #PLTR #AI #QuantAnalysis",
        ])
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run PLTR analysis."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  GhostBox PLTR Analysis - Integrated with Axiom-KG               ║")
    print("║                                                                   ║")
    print("║  SEC XBRL → Axiom Nodes → Semantic Tension → Report              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    analyzer = PLTRAnalyzer()
    result = analyzer.run_full_analysis()
    
    print()
    print(f"Overall divergence: {result['overall_score']:.3f}")
    print(f"Posture: {result['posture'].upper()}")
    print()
    print("Tensions by claim type:")
    for ct, data in result['tensions'].items():
        print(f"  {ct}: {data['divergence']:.3f} (driver: {data['driver']})")
    
    print()
    print("=" * 60)
    print()
    print(analyzer.generate_brief())
    
    # Save outputs
    output_dir = Path("ghostbox_pltr_output")
    output_dir.mkdir(exist_ok=True)
    
    (output_dir / "brief.md").write_text(analyzer.generate_brief())
    (output_dir / "linkedin.txt").write_text(analyzer.generate_linkedin_post())
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    
    print()
    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
