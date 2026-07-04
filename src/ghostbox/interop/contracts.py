"""
AXM Level-2 interoperability contracts.

This module defines the shared object boundary between the AXM repos and
GhostBox as the level-2 attention layer. It is deliberately small. It holds
data shapes and protocols, not behavior.

Design rules enforced here:
  - No salted Python hash for persistent IDs. Python's built-in hash() is
    per-process salted (PYTHONHASHSEED) and must never be used for stable IDs.
    All derived IDs are content-addressed over canonical bytes using BLAKE3
    when available, SHA-256 otherwise.
  - Every boundary distinguishes proven, simulated, frozen, and untested.
  - Refs carry externally assigned IDs. GhostBox references shards, it does
    not store or re-mint them.

Placement note:
  The spec asks for this module "in GhostBox." It lives at
  ghostbox/interop/contracts.py. Because every repo imports the same shapes,
  it is written to be lifted out into a standalone `axm_contracts` package
  later with no code change. Nothing here imports GhostBox internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Hashing / stable content-addressed IDs
# ---------------------------------------------------------------------------

try:
    import blake3 as _blake3  # type: ignore

    _HASH_TAG = "b3"

    def _digest(data: bytes) -> str:
        return _blake3.blake3(data).hexdigest()

except ImportError:  # pragma: no cover - fallback path
    import hashlib

    _HASH_TAG = "sha256"

    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic serialization for content addressing.

    Sorted keys, no insignificant whitespace, utf-8. Two payloads that are
    semantically equal produce identical bytes and therefore identical IDs.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError(f"non-canonicalizable type: {type(value).__name__}")


def content_id(prefix: str, identity: dict[str, Any]) -> str:
    """Return a stable, content-addressed ID of the form <prefix>:<tag>:<hex>.

    prefix names the object kind (evt, find, rcpt...). tag names the hash
    algorithm actually used so consumers can verify without guessing.
    """
    return f"{prefix}:{_HASH_TAG}:{_digest(canonical_bytes(identity))[:32]}"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Status vocabularies
# ---------------------------------------------------------------------------


class ProvenanceState(str, Enum):
    """State of an artifact against reality.

    PROVEN     Faithfully captured or cryptographically verified. Note that a
               proven capture means the record is trustworthy, not that the
               content it captured is true.
    SIMULATED  Generated, modeled, or synthetic. Never treated as record.
    FROZEN     Sealed and immutable. Change requires a new artifact, not an
               edit. axm-genesis kernel state is frozen.
    UNTESTED   Present but not yet checked against evidence.
    """

    PROVEN = "proven"
    SIMULATED = "simulated"
    FROZEN = "frozen"
    UNTESTED = "untested"


class ClaimVerdict(str, Enum):
    """Result of testing a claim against available evidence.

    This is distinct from ProvenanceState. ProvenanceState describes an
    artifact. ClaimVerdict describes a claim's standing after contact with
    evidence.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    FROZEN = "frozen"
    UNTESTED = "untested"


# ---------------------------------------------------------------------------
# Boundary objects
# ---------------------------------------------------------------------------


@dataclass
class EvidenceEvent:
    """What the surface showed. Emitted by ScreenGhost.

    Intake, not truth. provenance PROVEN here asserts faithful capture of the
    surface state, nothing about whether the captured content is accurate.
    """

    source: str
    surface: str  # chat | screen | ui | ...
    observation: dict[str, Any]
    captured_at: str = field(default_factory=now_utc)
    raw_ref: Optional[str] = None
    provenance: ProvenanceState = ProvenanceState.PROVEN
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = content_id(
                "evt",
                {
                    "source": self.source,
                    "surface": self.surface,
                    "observation": self.observation,
                    "captured_at": self.captured_at,
                    "raw_ref": self.raw_ref,
                },
            )


@dataclass
class KnowledgeShardRef:
    """Reference to a compiled knowledge shard produced by axm-core.

    A reference, not the shard. GhostBox points at knowledge, it does not own
    it. shard_id is assigned by axm-core and carried verbatim.
    """

    shard_id: str
    compiler: str = "axm-core"
    source_refs: list[str] = field(default_factory=list)
    summary: str = ""
    compiled_at: str = field(default_factory=now_utc)
    provenance: ProvenanceState = ProvenanceState.UNTESTED


@dataclass
class ConversationShardRef:
    """Reference to a conversation-memory shard produced by axm-chat.

    Added beyond the strict dataclass list in the spec so the contract is
    complete: axm-chat is a named spoke and GhostBox must be able to ingest
    its output. Modeled as a Ref for the same reason as KnowledgeShardRef.
    Flag for audit confirmation.
    """

    shard_id: str
    spoke: str = "axm-chat"
    export_ref: Optional[str] = None
    summary: str = ""
    recorded_at: str = field(default_factory=now_utc)
    provenance: ProvenanceState = ProvenanceState.UNTESTED


@dataclass
class ShardReceipt:
    """Seal-and-verify record produced by the axm-genesis trust kernel.

    axm-genesis is authoritative for sealing. GhostBox consumes receipts and
    verifies against them. It never mints signatures. receipt_id is derived
    for stable reference; the signature is the authority.
    """

    shard_id: str
    content_hash: str
    signature: str
    signature_algorithm: str = "ML-DSA-44"
    hash_algorithm: str = "BLAKE3"
    sealed_at: str = field(default_factory=now_utc)
    verified: bool = False
    provenance: ProvenanceState = ProvenanceState.FROZEN
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_id:
            self.receipt_id = content_id(
                "rcpt",
                {
                    "shard_id": self.shard_id,
                    "content_hash": self.content_hash,
                    "signature_algorithm": self.signature_algorithm,
                    "sealed_at": self.sealed_at,
                },
            )


@dataclass
class AttentionFinding:
    """Where the tension is. Emitted by GhostBox.

    The level-2 output. Carries the inputs that produced it, a tension type,
    and a magnitude. It surfaces claims for downstream testing. It does not
    decide whether those claims are true.
    """

    tension_type: str  # semantic_tension | contradiction | gap | field_zero | ...
    score: float
    summary: str
    input_refs: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    provenance: ProvenanceState = ProvenanceState.UNTESTED
    created_at: str = field(default_factory=now_utc)
    finding_id: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = content_id(
                "find",
                {
                    "tension_type": self.tension_type,
                    "score": self.score,
                    "summary": self.summary,
                    "input_refs": sorted(self.input_refs),
                    "created_at": self.created_at,
                },
            )


@dataclass
class ClaimCheckResult:
    """What a claim did on contact with the evidence. Emitted by
    axm-capability-claim-test.

    Evaluates claims. Does not create authority. verdict is one of
    supported, contradicted, frozen, untested.
    """

    claim: str
    verdict: ClaimVerdict
    supporting_refs: list[str] = field(default_factory=list)
    contradicting_refs: list[str] = field(default_factory=list)
    rationale: str = ""
    checked_at: str = field(default_factory=now_utc)
    check_id: str = ""

    def __post_init__(self) -> None:
        if not self.check_id:
            self.check_id = content_id(
                "chk",
                {
                    "claim": self.claim,
                    "verdict": self.verdict,
                    "supporting_refs": sorted(self.supporting_refs),
                    "contradicting_refs": sorted(self.contradicting_refs),
                    "checked_at": self.checked_at,
                },
            )


@dataclass
class PhysicalEvidenceEvent:
    """What changed in the physical world fast enough to require proof.
    Emitted by axm-embodied.

    Event-triggered high-resolution capture with forensic continuity.
    continuity_ref anchors the non-selective recording chain; a break in that
    chain is itself a finding, not a silent gap.
    """

    trigger: str
    sensor: str
    continuity_ref: str
    content_hash: str
    captured_at: str = field(default_factory=now_utc)
    fidelity: str = "high"
    provenance: ProvenanceState = ProvenanceState.PROVEN
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = content_id(
                "phys",
                {
                    "trigger": self.trigger,
                    "sensor": self.sensor,
                    "continuity_ref": self.continuity_ref,
                    "content_hash": self.content_hash,
                    "captured_at": self.captured_at,
                },
            )


# ---------------------------------------------------------------------------
# Protocols (roles, one per repo)
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceSource(Protocol):
    """ScreenGhost. Surface intake into structured evidence."""

    def emit_evidence(self) -> list[EvidenceEvent]: ...


@runtime_checkable
class ConversationSpoke(Protocol):
    """axm-chat. Conversation record into queryable shards."""

    def emit_conversation_shards(self) -> list[ConversationShardRef]: ...


@runtime_checkable
class KnowledgeCompiler(Protocol):
    """axm-core. Source documents into deterministic knowledge shards."""

    def compile(self, source_refs: list[str]) -> list[KnowledgeShardRef]: ...


@runtime_checkable
class TrustKernel(Protocol):
    """axm-genesis. Frozen. Seals and verifies. Nothing else mints receipts."""

    def seal(self, shard_id: str, content_hash: str) -> ShardReceipt: ...

    def verify(self, receipt: ShardReceipt) -> bool: ...


@runtime_checkable
class AttentionLayer(Protocol):
    """GhostBox. Level 2. Ingests shards and events, emits findings."""

    def ingest(
        self,
        events: list[EvidenceEvent],
        knowledge: list[KnowledgeShardRef],
        conversation: list[ConversationShardRef],
        physical: list[PhysicalEvidenceEvent],
    ) -> None: ...

    def find_tension(self) -> list[AttentionFinding]: ...


@runtime_checkable
class ClaimHarness(Protocol):
    """axm-capability-claim-test. Evaluates claims against evidence."""

    def check(self, claim: str, evidence_refs: list[str]) -> ClaimCheckResult: ...


@runtime_checkable
class EmbodiedSource(Protocol):
    """axm-embodied. Physical-world evidence spoke."""

    def emit_physical(self) -> list[PhysicalEvidenceEvent]: ...


# ---------------------------------------------------------------------------
# Self-test: determinism and shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fixed = "2026-07-04T00:00:00+00:00"

    e1 = EvidenceEvent(
        source="screenghost",
        surface="chat",
        observation={"text": "vendor claims mandatory integration"},
        captured_at=fixed,
    )
    e2 = EvidenceEvent(
        source="screenghost",
        surface="chat",
        observation={"text": "vendor claims mandatory integration"},
        captured_at=fixed,
    )
    assert e1.event_id == e2.event_id, "content addressing must be deterministic"
    assert ":" in e1.event_id and e1.event_id.startswith("evt:")

    finding = AttentionFinding(
        tension_type="contradiction",
        score=0.82,
        summary="Claim of open standard contradicts sole-source integration path.",
        input_refs=[e1.event_id],
        claims=["integration layer is open"],
    )

    result = ClaimCheckResult(
        claim="integration layer is open",
        verdict=ClaimVerdict.CONTRADICTED,
        contradicting_refs=[e1.event_id],
        rationale="Surface evidence shows sole-source dependency.",
    )

    receipt = ShardReceipt(
        shard_id="know:example",
        content_hash="deadbeef",
        signature="<opaque>",
        sealed_at=fixed,
    )

    phys = PhysicalEvidenceEvent(
        trigger="motion",
        sensor="cam-01",
        continuity_ref="chain:0001",
        content_hash="cafebabe",
        captured_at=fixed,
    )

    for obj in (e1, finding, result, receipt, phys):
        print(json.dumps(asdict(obj), default=_json_default, indent=2))
        print("-" * 60)

    print(f"hash algorithm in use: {_HASH_TAG}")
    print("self-test passed")
