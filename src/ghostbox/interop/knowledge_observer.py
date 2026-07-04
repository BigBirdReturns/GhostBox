"""GhostBox observer over axm-core (axm-forge) knowledge shards.

axm-core knowledge output is a genesis-sealed shard (proven live). This observer
lets GhostBox add findings over that knowledge WITHOUT becoming the knowledge or
custody owner. The order is fixed and non-negotiable:

    KnowledgeShardRef -> its SealedShard -> custody verification (through the
    LANDED GenesisCustodySpoke) -> read the sealed knowledge graph -> GhostBox
    findings.

Enforced boundaries:
  - Reuses the landed custody seam for verification. It does NOT call
    ``axm-verify`` and does NOT duplicate genesis logic -- ``GenesisCustodySpoke``
    already owns that path.
  - Never mints or overrides ``shard_id`` -- identity is the sealed shard's,
    carried verbatim (as are the genesis content ids ``c1_`` / ``e1_``).
  - Never treats producer-asserted knowledge metadata (``compiler`` /
    ``source_refs``) as custody. Knowledge is read from the *verified* sealed
    graph, never from the ref's metadata.
  - Never annotates an unverified shard as trusted: knowledge findings are
    emitted ONLY after custody returns VERIFIED. A non-PASS shard yields no
    knowledge findings (the custody spoke has already recorded the failure).
  - Never writes into the sealed shard: it reads ``graph/*.jsonl`` and nothing
    else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .contracts import AttentionFinding, KnowledgeShardRef, ProvenanceState, VerifyStatus
from .genesis_spoke import CustodyOutcome, GenesisCustodySpoke


@dataclass(frozen=True)
class ObservationResult:
    """Outcome of observing one knowledge shard. GhostBox-owned.

    Holds only the genesis-assigned ``shard_id`` (a reference, verbatim), the
    custody verdict, and GhostBox's own findings -- no shard body, no manifest,
    no signature, no custody material.
    """

    shard_id: str
    verified: bool
    status: VerifyStatus
    findings: Tuple[AttentionFinding, ...]


class KnowledgeObserver:
    """Observe verified knowledge shards; annotate, never own."""

    def __init__(self, custody: GenesisCustodySpoke) -> None:
        self._custody = custody
        self._findings: List[AttentionFinding] = []

    @property
    def findings(self) -> Tuple[AttentionFinding, ...]:
        return tuple(self._findings)

    def observe(self, ref: KnowledgeShardRef) -> ObservationResult:
        # 1) Verify the underlying sealed shard THROUGH the landed custody seam.
        #    No direct axm-verify, no duplicated genesis logic -- custody owns it.
        entry = self._custody.ingest_sealed_shard(ref.sealed)

        # 2) Fail closed: only a VERIFIED custody outcome unlocks observation.
        if entry.outcome is not CustodyOutcome.VERIFIED:
            # The custody spoke already recorded its unverified_seal finding; the
            # observer adds NO knowledge findings over an untrusted shard, and
            # never reads its graph.
            return ObservationResult(
                shard_id=ref.shard_id, verified=False, status=entry.status, findings=()
            )

        # 3) Only now read the verified sealed knowledge graph and surface findings.
        findings = tuple(self._surface_knowledge(ref))
        self._findings.extend(findings)
        return ObservationResult(
            shard_id=ref.shard_id, verified=True, status=entry.status, findings=findings
        )

    # -- reading the *verified* sealed knowledge graph (read-only) -----------

    def _surface_knowledge(self, ref: KnowledgeShardRef) -> List[AttentionFinding]:
        shard_dir = Path(ref.sealed.shard_dir)
        labels = self._entity_labels(shard_dir / "graph" / "entities.jsonl")
        claims_path = shard_dir / "graph" / "claims.jsonl"
        findings: List[AttentionFinding] = []
        if not claims_path.exists():
            return findings
        for row in self._read_jsonl(claims_path):
            subject = labels.get(row.get("subject", ""), row.get("subject", ""))
            predicate = row.get("predicate", "")
            obj = row.get("object", "")
            if not str(row.get("object_type", "")).startswith("literal"):
                obj = labels.get(obj, obj)
            claim_id = row.get("claim_id", "")
            statement = " ".join(str(x) for x in (subject, predicate, obj) if str(x)).strip()
            # input_refs are genesis-owned ids (shard + claim), carried verbatim;
            # never a minted id and never producer metadata.
            input_refs = [ref.shard_id] + ([claim_id] if claim_id else [])
            findings.append(
                AttentionFinding(
                    tension_type="knowledge_claim",
                    score=0.5,
                    summary=f"Verified knowledge claim in {ref.shard_id}: {statement}",
                    input_refs=input_refs,
                    claims=[statement] if statement else [],
                    provenance=ProvenanceState.UNTESTED,  # surfaced for testing; not asserted true
                )
            )
        return findings

    def _entity_labels(self, path: Path) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        if path.exists():
            for row in self._read_jsonl(path):
                eid = row.get("entity_id")
                if eid:
                    labels[eid] = row.get("label", eid)
        return labels

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
