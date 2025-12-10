# GhostBox v0.8.1

**Integrated intelligence stack for semantic tension analysis.**

```
Real Data → Axiom-KG Adapters → Semantic Coordinates → Tension → Attention → Decision
```

## What's Integrated

| Component | Function |
|-----------|----------|
| **Axiom-KG** | Semantic coordinate system with 9 domain adapters |
| **Screen Ghost** | Capture sources (JSON, CSV, SQLite, webhooks) |
| **DIND Stack** | Tension detection and alert generation |
| **Attention Geometry** | Where to look (contradiction, velocity, convergence) |
| **Session Continuity** | Analyst handoff without context loss |
| **PLTR Analyzer** | Field Zero domain: SEC filings vs narrative |

## Quick Start

```python
from ghostbox.integration import create_engine

# Create engine
engine = create_engine()

# Run PLTR analysis (Field Zero)
result = engine.run_pltr_analysis()
print(f"Divergence: {result['overall_score']:.3f}")
print(f"Posture: {result['posture']}")
```

## PLTR Analysis

The PLTR module demonstrates the full stack:

1. **SEC XBRL filings** parsed by `XBRLAdapter` → financial ground truth
2. **RSS feeds** parsed by `RSSAdapter` → narrative claims
3. **Claims** become `Nodes` with semantic coordinates
4. **Divergence** computed as `SemanticTension` from coordinate distance
5. **Report** generated automatically

```bash
# Run PLTR analysis
python -m ghostbox_pltr.analyzer
```

Output:
```
Overall divergence: 0.218
Posture: STABLE

Tensions by claim type:
  GOV_CAPTURE: 0.330 (narrative runs ahead of filings)
  MOAT: 0.280 (moat claims exceed margin support)
  GROWTH: 0.166 (growth story vs revenue trajectory)
```

## Architecture

```
ghostbox-integrated/
├── src/
│   ├── axiom/                    # Semantic coordinate system
│   │   ├── core.py               # Space, Node, SemanticID, Fork
│   │   └── adapters/             # 9 domain adapters
│   │       ├── schemaorg.py      # JSON-LD / ASW
│   │       ├── openapi.py        # API specs
│   │       ├── rss.py            # RSS/Atom feeds
│   │       ├── ical.py           # Calendars
│   │       ├── package.py        # npm/pip/cargo
│   │       ├── fhir.py           # Healthcare
│   │       ├── xbrl.py           # Financial (SEC filings)
│   │       ├── epub.py           # Ebooks
│   │       └── akn.py            # Legal
│   │
│   ├── screen_ghost/             # Capture layer
│   │   ├── capture.py            # Sources (JSON, CSV, SQLite)
│   │   ├── attention.py          # Attention geometry
│   │   └── session.py            # Analyst continuity
│   │
│   ├── dind_stack/               # Tension detection
│   │   ├── tension.py            # Tension computation
│   │   └── server.py             # FastAPI endpoints
│   │
│   ├── ghostbox/                 # Integration layer
│   │   └── integration.py        # GhostBoxEngine
│   │
│   └── ghostbox_pltr/            # Field Zero: PLTR analysis
│       └── analyzer.py           # SEC vs narrative tension
```

## The Thesis

**Knowledge should not be stored. It should be derivable.**

Everything flows through semantic coordinates:
- Same schema → Same coordinates
- Different sources → Comparable by distance
- Contradictions → Forks with explicit disambiguation
- Tension → Geometric, not heuristic

## Field Zero

This is the first deployed domain: PLTR financial analysis.

- **Real data**: SEC EDGAR API, Google News RSS
- **Real stakes**: Investment signal
- **Real output**: Divergence score, tension report, LinkedIn post

Run it for 30 days. Track what it catches. Document the decisions.

## What's Next

- XREV integration (reversible decision traces)
- More Field Zero domains (custody tracking, LinkedIn attention)
- Web interface for real-time monitoring

## License

MIT

## Author

Jonathan Sandhu / Sandhu Consulting Group

---

*This is how it's done.*
