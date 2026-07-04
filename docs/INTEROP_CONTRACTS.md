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
`evt:sha256:c34679d7...`. The algorithm is **pinned to SHA-256 as a fixed
contract constant** — not an environment-dependent choice. An optional backend
(e.g. BLAKE3 when the `blake3` package happens to be installed) would make the
same canonical object hash to `evt:b3:...` on one machine and `evt:sha256:...` on
another, silently breaking the cross-repo promise that one observation yields one
`event_id`. Algorithm agility, if ever required, must be an explicit new contract
version carrying its own tag — never a silent import-driven switch. The tag stays
in the id so consumers verify against the stated algorithm rather than guessing.
Canonicalization is sorted keys, no insignificant whitespace, utf-8.

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
hash_algorithm, sealed_at, signer, verified, provenance (FROZEN),
receipt_body_id, receipt_id (a.k.a. attestation_id).
Contract: genesis is the only issuer. GhostBox and others verify against a
receipt; they never produce a signature. The signature is the authority — no
derived id replaces it.

Identity is split into two objects with different lifetimes, so distinct
authority records never alias and no id is circular:
- `receipt_body_id` identifies the *unsigned body* — `shard_id`, `content_hash`,
  `hash_algorithm`, `sealed_at`. Same body → same body id; it never includes a
  signature.
- `receipt_id` / `attestation_id` identifies the *signed authority record* —
  body id + `signer` + `signature_algorithm` + `signature`. Two signatures over
  the same body are two authority events with two ids. It is derived *after*
  signing and is never part of the signed payload, so the id can depend on the
  signature without the signature depending on the id.

Verify path: state which question you ask, because they are different objects —
"is this body sealed at all?" keys on `receipt_body_id`; "is this specific
authority act valid?" keys on `receipt_id`. `TrustKernel.verify` answers the
second. Collapsing them reintroduces the aliasing this split removes.

**PROPOSED AMENDMENT.** This two-id split refines identification of a
genesis-owned shape. It is offered from the interop layer and must be confirmed
by the axm-genesis kernel owner before it is treated as authoritative. It changes
identity derivation only; it does not change sealing or authority semantics.

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
