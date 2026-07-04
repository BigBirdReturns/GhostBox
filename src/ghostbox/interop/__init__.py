"""GhostBox level-2 interoperability surface.

Import shared contract shapes and protocols from here so repos depend on the
boundary, not on each other's internals.
"""

from .contracts import (
    # status vocabularies
    ProvenanceState,
    ClaimVerdict,
    VerifyStatus,
    # boundary objects
    EvidenceEvent,
    KnowledgeShardRef,
    ConversationShardRef,
    SealedShard,
    AttentionFinding,
    ClaimCheckResult,
    PhysicalEvidenceEvent,
    # roles
    EvidenceSource,
    ConversationSpoke,
    KnowledgeCompiler,
    TrustKernel,
    AttentionLayer,
    ClaimHarness,
    EmbodiedSource,
    # helpers
    content_id,
    canonical_bytes,
    now_utc,
)

__all__ = [
    "ProvenanceState",
    "ClaimVerdict",
    "VerifyStatus",
    "EvidenceEvent",
    "KnowledgeShardRef",
    "ConversationShardRef",
    "SealedShard",
    "AttentionFinding",
    "ClaimCheckResult",
    "PhysicalEvidenceEvent",
    "EvidenceSource",
    "ConversationSpoke",
    "KnowledgeCompiler",
    "TrustKernel",
    "AttentionLayer",
    "ClaimHarness",
    "EmbodiedSource",
    "content_id",
    "canonical_bytes",
    "now_utc",
]
