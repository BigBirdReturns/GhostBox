"""GhostBox observer over axm-chat conversation shards.

axm-chat conversation output is a genesis-sealed shard (proven live: a real
``axm-chat import`` was probed and verified detached on this branch). This
observer lets GhostBox add findings over that conversation record WITHOUT
becoming the conversation or custody owner. The order is fixed and
non-negotiable:

    ConversationShardRef -> its SealedShard -> custody verification (through
    the LANDED GenesisCustodySpoke) -> confirm the sealed namespace is
    chat/conversation -> read the sealed conversation graph -> GhostBox
    findings.

Enforced boundaries:
  - Reuses the landed custody seam for verification. It does NOT call
    ``axm-verify`` and does NOT duplicate genesis logic -- ``GenesisCustodySpoke``
    already owns that path.
  - Never mints or overrides ``shard_id`` -- identity is the sealed shard's,
    carried verbatim (as are the genesis content ids ``c1_`` / ``e1_``).
  - Never treats producer-asserted conversation metadata (``spoke`` /
    ``export_ref``) as custody. The conversation is read from the *verified*
    sealed graph, never from the ref's metadata.
  - Never annotates an unverified shard as trusted: conversation findings are
    emitted ONLY after custody returns VERIFIED. A non-PASS shard yields no
    conversation findings (the custody spoke has already recorded the failure).
  - Never interprets conversation content: no summarization, no sentiment, no
    classification. Claims are surfaced verbatim for downstream testing --
    what the sealed record says, never what it means.
  - Never writes into the sealed shard: it reads ``manifest.json`` (namespace
    only) and ``graph/*.jsonl``, nothing else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .contracts import AttentionFinding, ConversationShardRef, ProvenanceState, VerifyStatus
from .genesis_spoke import CustodyOutcome, GenesisCustodySpoke

# The namespace the real axm-chat import stamps on every conversation shard
# (constant across conversations; per-conversation identity lives in the
# conversation/{id} entity label, not the namespace).
CONVERSATION_NAMESPACE = "chat/conversation"


@dataclass(frozen=True)
class ConversationObservationResult:
    """Outcome of observing one conversation shard. GhostBox-owned.

    Holds only the genesis-assigned ``shard_id`` (a reference, verbatim), the
    custody verdict, the sealed namespace that was checked, and GhostBox's own
    findings -- no shard body, no manifest, no signature, no custody material.
    """

    shard_id: str
    verified: bool
    status: VerifyStatus
    namespace: Optional[str]
    findings: Tuple[AttentionFinding, ...]


class ConversationObserver:
    """Observe verified conversation shards; annotate, never own."""

    def __init__(self, custody: GenesisCustodySpoke) -> None:
        self._custody = custody
        self._findings: List[AttentionFinding] = []

    @property
    def findings(self) -> Tuple[AttentionFinding, ...]:
        return tuple(self._findings)

    def observe(self, ref: ConversationShardRef) -> ConversationObservationResult:
        # 1) Verify the underlying sealed shard THROUGH the landed custody seam.
        #    No direct axm-verify, no duplicated genesis logic -- custody owns it.
        entry = self._custody.ingest_sealed_shard(ref.sealed)

        # 2) Fail closed: only a VERIFIED custody outcome unlocks observation.
        if entry.outcome is not CustodyOutcome.VERIFIED:
            # The custody spoke already recorded its unverified_seal finding;
            # the observer adds NO conversation findings over an untrusted
            # shard, and never reads its manifest or graph.
            return ConversationObservationResult(
                shard_id=ref.shard_id,
                verified=False,
                status=entry.status,
                namespace=None,
                findings=(),
            )

        # 3) Confirm the verified shard IS a conversation shard. The namespace
        #    is the real spoke's only kind marker (there is no metadata.kind);
        #    a verified shard of another kind is a finding, not a silent pass.
        namespace = self._sealed_namespace(Path(ref.sealed.shard_dir))
        if namespace != CONVERSATION_NAMESPACE:
            finding = AttentionFinding(
                tension_type="not_conversation_shard",
                score=1.0,
                summary=(
                    f"Shard {ref.shard_id} verified but its sealed namespace is "
                    f"{namespace!r}, not {CONVERSATION_NAMESPACE!r}; no conversation "
                    f"findings were read from it"
                ),
                input_refs=[ref.shard_id],
                provenance=ProvenanceState.UNTESTED,
            )
            self._findings.append(finding)
            return ConversationObservationResult(
                shard_id=ref.shard_id,
                verified=True,
                status=entry.status,
                namespace=namespace,
                findings=(finding,),
            )

        # 4) Only now read the verified sealed conversation graph and surface
        #    findings.
        findings = tuple(self._surface_conversation(ref))
        self._findings.extend(findings)
        return ConversationObservationResult(
            shard_id=ref.shard_id,
            verified=True,
            status=entry.status,
            namespace=namespace,
            findings=findings,
        )

    # -- reading the *verified* sealed conversation record (read-only) --------

    @staticmethod
    def _sealed_namespace(shard_dir: Path) -> Optional[str]:
        """metadata.namespace from the verified sealed manifest, read-only."""
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        metadata = manifest.get("metadata") or {}
        value = metadata.get("namespace")
        return value if isinstance(value, str) else None

    def _surface_conversation(self, ref: ConversationShardRef) -> List[AttentionFinding]:
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
                    tension_type="conversation_claim",
                    score=0.5,
                    summary=f"Verified conversation claim in {ref.shard_id}: {statement}",
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
