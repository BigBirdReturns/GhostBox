# AXM Interoperability Contracts

Status: draft for audit confirmation
Reference implementation: `ghostbox/interop/contracts.py`

This document is the human-readable version of the boundary. The Python module
is authoritative for field names and types. Where they disagree, the module
wins and this document is the bug.

## Identity and hashing

Persistent IDs are content-addressed. Never use Python's built-in `hash()` for
a stable ID. It is per-process salted through PYTHONHASHSEED and will not
survive a restart, let alone cross a repo boundary.

IDs take the form `<prefix>:<algorithm>:<hex>`, for example
`evt:b3:1c703656...`. The algorithm tag names what actually produced the digest,
BLAKE3 when the `blake3` package is present, SHA-256 otherwise. Consumers verify
against the stated algorithm rather than guessing. Canonicalization is sorted
keys, no insignificant whitespace, utf-8.

Refs carry externally assigned IDs verbatim. A KnowledgeShardRef does not
re-mint the shard id that axm-core assigned. GhostBox points at shards. It does
not rename them.

## Objects

### EvidenceEvent
Emitter: ScreenGhost. Answers: what did the surface show.
Fields: source, surface, observation, captured_at, raw_ref, provenance,
event_id.
Contract: intake only. provenance PROVEN asserts faithful capture of surface
state, not that the captured content is accurate. event_id is derived from
source, surface, observation, captured_at, raw_ref.

### KnowledgeShardRef
Emitter: axm-core (referenced by GhostBox). Answers: what knowledge compiled
from sources.
Fields: shard_id, compiler, source_refs, summary, compiled_at, provenance.
Contract: a reference, not the shard body. shard_id is assigned by axm-core.

### ConversationShardRef
Emitter: axm-chat (referenced by GhostBox). Answers: what the conversation
record contained.
Fields: shard_id, spoke, export_ref, summary, recorded_at, provenance.
Contract: a spoke, not the whole memory system. Modeled as a Ref for the same
reason as KnowledgeShardRef. This object is an addition beyond the strict
dataclass list in the original spec, added so the contract is complete because
axm-chat is a named core spoke. Flagged for audit confirmation.

### ShardReceipt
Emitter: axm-genesis. Answers: can this record be verified.
Fields: shard_id, content_hash, signature, signature_algorithm (ML-DSA-44),
hash_algorithm (BLAKE3), sealed_at, verified, provenance (FROZEN), receipt_id.
Contract: genesis is the only issuer. GhostBox and others verify against a
receipt, they never produce a signature. receipt_id is derived for stable
reference; the signature is the authority.

### AttentionFinding
Emitter: GhostBox. Answers: where is the tension.
Fields: tension_type, score, summary, input_refs, claims, provenance,
created_at, finding_id.
Contract: the level-2 output. Carries the input references that produced it and
the claims it surfaces for downstream testing. It surfaces claims. It does not
decide whether they are true.

### ClaimCheckResult
Emitter: axm-capability-claim-test. Answers: what claims survive contact with
the evidence.
Fields: claim, verdict, supporting_refs, contradicting_refs, rationale,
checked_at, check_id.
Contract: verdict is one of SUPPORTED, CONTRADICTED, FROZEN, UNTESTED. Evaluates
claims. Does not create authority.

### PhysicalEvidenceEvent
Emitter: axm-embodied. Answers: what changed in the physical world fast enough
to require forensic capture.
Fields: trigger, sensor, continuity_ref, content_hash, captured_at, fidelity,
provenance (PROVEN), event_id.
Contract: event-triggered high-resolution capture. continuity_ref anchors the
non-selective recording chain. A break in that chain is itself a finding, not a
silent gap.

## Protocols

Each repo implements one role protocol. The protocol is the seam; internals stay
private.

EvidenceSource.emit_evidence -> list[EvidenceEvent]  (ScreenGhost)
ConversationSpoke.emit_conversation_shards -> list[ConversationShardRef]  (axm-chat)
KnowledgeCompiler.compile(source_refs) -> list[KnowledgeShardRef]  (axm-core)
TrustKernel.seal(shard_id, content_hash) -> ShardReceipt; verify(receipt) -> bool  (axm-genesis)
AttentionLayer.ingest(...); find_tension -> list[AttentionFinding]  (GhostBox)
ClaimHarness.check(claim, evidence_refs) -> ClaimCheckResult  (claim-test)
EmbodiedSource.emit_physical -> list[PhysicalEvidenceEvent]  (axm-embodied)

## Invariants

Storage authority stays out of GhostBox.
Sealing authority stays in axm-genesis and nowhere else.
Refs are never re-minted downstream.
Every object declares a provenance or verdict. There is no implicit state.
Proven, simulated, frozen, and untested are always distinguishable at the
boundary.
