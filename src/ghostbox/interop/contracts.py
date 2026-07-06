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


class VerifyStatus(str, Enum):
    """Result of asking axm-genesis to verify a sealed shard.

    This is the genesis verification vocabulary, kept AT the boundary rather
    than projected to a bool. axm-verify's exit codes are frozen (axm-genesis
    COMPATIBILITY.md): ``0`` PASS, ``1`` FAIL, ``2`` MALFORMED. Plus one
    caller-side state that never reaches the kernel: ``NO_TRUSTED_KEY``, when no
    out-of-band anchor is available to verify against.

    Only ``PASS`` is trust; every other state blocks the shard. A consumer that
    wants a bool derives ``status is VerifyStatus.PASS`` locally -- the boundary
    never hides the failure class (a malformed shard and a bad signature are
    different findings).
    """

    PASS = "pass"
    FAIL = "fail"
    MALFORMED = "malformed"
    NO_TRUSTED_KEY = "no_trusted_key"


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


@dataclass(frozen=True)
class KnowledgeShardRef:
    """axm-core's knowledge overlay ON a genesis-sealed shard.

    Reconciled against the real axm-core surface (probe on branch
    ``claude/axm-core-knowledge-edge``): axm-core produces knowledge by handing
    candidates to the genesis compiler (``axm_build.compiler_generic.
    compile_generic_shard``), so its output *is* a genesis-sealed ``SealedShard``.
    This ref therefore sits ABOVE that sealed shard, it does not stand beside it
    and it is not a second custody object.

    Ownership:
      - genesis owns ``shard_id``, manifest, signatures, Merkle root, the content
        ids (``e1_`` / ``c1_`` / ``ea_`` / ``s1_`` / ``p1_``), and verification --
        all on the embedded ``SealedShard``;
      - axm-core owns the knowledge selection and this metadata;
      - GhostBox observes and annotates later, and never here.

    There is deliberately NO independent ``shard_id`` field: identity comes from
    the embedded sealed shard (see the ``shard_id`` property), so a knowledge ref
    can never mint or override a custody id. Verification is not a path of its
    own -- ``verify`` delegates to the custody seam on the underlying sealed
    shard. There is no ``summary`` (a summary is not axm-core authority; if one is
    wanted it is a later GhostBox-side annotation, never core or manifest truth).
    ``namespace`` / ``title`` / ``publisher`` already live in the sealed manifest
    and are read through ``sealed``, never duplicated here. Knowledge-content
    addressing (entities/claims/provenance/spans) is genesis-derived and read
    from the sealed shard when observing, not stored on the ref.
    """

    sealed: "SealedShard"  # the genesis-sealed custody object; the ONLY identity/verification source
    compiler: str = "axm-forge"  # knowledge metadata: the producing pipeline (the forge path)
    # Producer-asserted input-document references (what axm-core extracted from).
    # These are metadata, NOT custody-anchored identity: a minimal forge run emits
    # no ``ext/locators@1``, so they are not recoverable or verifiable from the
    # sealed shard and must never be trusted as custody. The sealed manifest's own
    # ``sources[]`` ARE custody material -- read them through ``sealed``, never
    # duplicated into this overlay as authority.
    source_refs: list[str] = field(default_factory=list)
    compiled_at: Optional[str] = None  # maps to the sealed manifest's metadata.created_at

    @property
    def shard_id(self) -> str:
        """Identity comes from the sealed shard -- never minted or overridden here."""
        return self.sealed.shard_id

    def verify(self, kernel: "TrustKernel", *, trusted_key: str) -> "VerifyStatus":
        """Delegate verification to the custody seam on the underlying sealed shard.

        A KnowledgeShardRef is not a verification path of its own: it verifies the
        SAME object genesis sealed, through the same ``TrustKernel.verify``
        boundary. Custody and trust never move up into the knowledge layer.
        """
        return kernel.verify(self.sealed, trusted_key=trusted_key)

    @classmethod
    def over(
        cls,
        sealed: "SealedShard",
        *,
        compiler: str = "axm-forge",
        source_refs: Optional[list[str]] = None,
        compiled_at: Optional[str] = None,
    ) -> "KnowledgeShardRef":
        """Build a knowledge overlay over a sealed shard.

        ``compiled_at`` defaults to the sealed shard's ``sealed_at`` (the manifest
        ``metadata.created_at``), so the compile timestamp is never a second
        source of truth.
        """
        return cls(
            sealed=sealed,
            compiler=compiler,
            source_refs=list(source_refs or []),
            compiled_at=compiled_at if compiled_at is not None else sealed.sealed_at,
        )


@dataclass(frozen=True)
class ConversationShardRef:
    """axm-chat's conversation overlay ON a genesis-sealed shard.

    Reconciled against the real axm-chat surface (probe on branch
    ``claude/session-planning-1h0xlw``): ``axm-chat import`` extracts a
    conversation and hands candidates to the genesis compiler
    (``axm_build.compiler_generic.compile_generic_shard``) in ONE pass, so its
    output *is* a genesis-sealed ``SealedShard`` — namespace
    ``chat/conversation``, publisher ``@axm_chat``, one ``conversation/{id}``
    entity plus ``turn/{n}`` entities, and ``has_title`` / ``message_count`` /
    ``has_turn`` claims, verified detached with an out-of-band key. This ref
    therefore sits ABOVE that sealed shard, exactly as ``KnowledgeShardRef``
    sits above axm-core's output: one custody pattern, not a second.

    Ownership:
      - genesis owns ``shard_id``, manifest, signatures, Merkle root, the
        content ids (``e1_`` / ``c1_``), and verification -- all on the
        embedded ``SealedShard``;
      - axm-chat owns the conversation extraction and this metadata;
      - GhostBox observes and annotates later, and never here.

    Reconciliation (guess -> reality):
      - the independent ``shard_id`` field is gone -- identity comes from the
        embedded sealed shard (see the ``shard_id`` property) and can never be
        minted or overridden here;
      - ``summary`` is gone -- the real spoke emits no summary anywhere; the
        conversation title lives in the sealed manifest and is read through
        ``sealed``, never duplicated as ref authority;
      - ``provenance`` is gone -- it lives on the embedded ``sealed``;
      - ``recorded_at`` no longer self-mints ``now_utc()``; it maps to the
        sealed manifest's ``metadata.created_at`` (via ``over``);
      - ``export_ref`` stays as producer metadata: which raw export file the
        conversation came from. The real import records it NOWHERE in the
        sealed shard, so it is not recoverable or verifiable from the shard
        and must never be trusted as custody.
    """

    sealed: "SealedShard"  # the genesis-sealed custody object; the ONLY identity/verification source
    spoke: str = "axm-chat"  # conversation metadata: the producing spoke
    export_ref: Optional[str] = None  # producer-asserted raw-export reference; metadata, never custody
    recorded_at: Optional[str] = None  # maps to the sealed manifest's metadata.created_at

    @property
    def shard_id(self) -> str:
        """Identity comes from the sealed shard -- never minted or overridden here."""
        return self.sealed.shard_id

    def verify(self, kernel: "TrustKernel", *, trusted_key: str) -> "VerifyStatus":
        """Delegate verification to the custody seam on the underlying sealed shard.

        A ConversationShardRef is not a verification path of its own: it
        verifies the SAME object genesis sealed, through the same
        ``TrustKernel.verify`` boundary. Custody and trust never move up into
        the conversation layer.
        """
        return kernel.verify(self.sealed, trusted_key=trusted_key)

    @classmethod
    def over(
        cls,
        sealed: "SealedShard",
        *,
        spoke: str = "axm-chat",
        export_ref: Optional[str] = None,
        recorded_at: Optional[str] = None,
    ) -> "ConversationShardRef":
        """Build a conversation overlay over a sealed shard.

        ``recorded_at`` defaults to the sealed shard's ``sealed_at`` (the
        manifest ``metadata.created_at``), so the record timestamp is never a
        second source of truth.
        """
        return cls(
            sealed=sealed,
            spoke=spoke,
            export_ref=export_ref,
            recorded_at=recorded_at if recorded_at is not None else sealed.sealed_at,
        )


@dataclass(frozen=True)
class SealedShard:
    """A shard sealed by the axm-genesis trust kernel.

    Genesis is authoritative for the seal, and the seal IS the on-disk shard
    directory -- ``manifest.json`` + ``sig/manifest.sig`` + ``sig/publisher.pub``
    -- not a GhostBox-minted object. This is a *reference* to that sealed
    directory, carried verbatim. (It replaces the earlier ``ShardReceipt``,
    which modelled a genesis "receipt" that genesis does not actually produce.)

    Identity is the genesis-assigned, derived ``shard_id`` (``sh1_`` + BLAKE3 of
    the canonical manifest bytes); the signature context -- the ``axm-hybrid1``
    suite, Ed25519 + ML-DSA-44, both of which must verify, over the
    domain-separated manifest message -- is the authority. GhostBox never mints,
    stores, or re-derives that identity, and never copies the signature bytes:
    those stay in the shard's ``sig/`` directory, which is what genesis verifies.

    ``merkle_root`` and ``sealed_at`` are read-only projections of manifest
    fields (``integrity.merkle_root`` / ``metadata.created_at``) for display and
    correlation only. There is deliberately NO ``verified`` field: verification
    is an *act* that returns a :class:`VerifyStatus`, never a property stored on
    the artifact.
    """

    shard_id: str                          # genesis-assigned sh1_..., verbatim
    shard_dir: str                         # the sealed directory genesis verifies
    suite: str = "axm-hybrid1"             # Ed25519 + ML-DSA-44; both must verify
    merkle_root: Optional[str] = None      # manifest integrity.merkle_root (BLAKE3)
    sealed_at: Optional[str] = None        # manifest metadata.created_at
    provenance: ProvenanceState = ProvenanceState.FROZEN


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
    """axm-chat. Conversation record into queryable shards.

    Reconciled note: axm-chat itself exports NO shard-reference API — its real
    surface is ``import_export()``, which seals shards into a store directory
    and returns only counts. This protocol is therefore the shape of a
    *consumer-side adapter* over that store (scan sealed shard directories,
    build ``ConversationShardRef.over(sealed)`` for each), not an interface the
    axm-chat package implements or is expected to grow.
    """

    def emit_conversation_shards(self) -> list[ConversationShardRef]: ...


@runtime_checkable
class KnowledgeCompiler(Protocol):
    """axm-core. Source documents into deterministic knowledge shards."""

    def compile(self, source_refs: list[str]) -> list[KnowledgeShardRef]: ...


@runtime_checkable
class TrustKernel(Protocol):
    """axm-genesis. Frozen. Seals and verifies; nothing else seals.

    ``verify`` is the real verification boundary, not a convenience bool. It
    takes the sealed shard and an out-of-band ``trusted_key`` -- supplied by the
    caller, never read from the shard's own ``sig/publisher.pub`` -- and returns
    a :class:`VerifyStatus` that preserves genesis's frozen exit taxonomy
    (PASS / FAIL / MALFORMED, plus NO_TRUSTED_KEY when the caller has no anchor).
    Consumers derive a bool locally as ``status is VerifyStatus.PASS``; the
    boundary itself never collapses the failure class.
    """

    def seal(self, shard_dir: str) -> SealedShard: ...

    def verify(self, shard: SealedShard, *, trusted_key: str) -> VerifyStatus: ...


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

    sealed = SealedShard(
        shard_id="sh1_example",
        shard_dir="/shards/gold/example",
        merkle_root="deadbeef",
        sealed_at=fixed,
    )
    assert sealed.provenance is ProvenanceState.FROZEN
    assert not hasattr(sealed, "verified"), "verification is an act, not a field"

    phys = PhysicalEvidenceEvent(
        trigger="motion",
        sensor="cam-01",
        continuity_ref="chain:0001",
        content_hash="cafebabe",
        captured_at=fixed,
    )

    for obj in (e1, finding, result, sealed, phys):
        print(json.dumps(asdict(obj), default=_json_default, indent=2))
        print("-" * 60)

    print(f"hash algorithm in use: {_HASH_TAG}")
    print("self-test passed")
