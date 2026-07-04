# AXM Level-2 Alignment

Status: draft for audit confirmation
Scope: alignment, not feature expansion

## Purpose

This document aligns the AXM ecosystem around GhostBox as the level-2 attention
layer. It does not rewrite repos, merge repos, or introduce a monolith. It
defines a thin interoperability contract so each repo can evolve without
drifting into duplicate claims, duplicate formats, or duplicate authority.

## The layering

Level 1 is record and knowledge. It answers what happened and what can be
compiled from sources. It is where storage, sealing, and compilation live.

Level 2 is attention. It answers where the tension is. GhostBox sits here. It
ingests shards and events from level 1, computes semantic tension and attention
geometry, and emits findings. It reads the record. It does not become the
record.

The single load-bearing rule: GhostBox is the attention layer, not the storage
layer. If GhostBox ever starts holding the canonical copy of a shard, minting
its own receipts, or deciding what is true, the layering has failed.

## Roles, one job each

ScreenGhost is intake, not truth. It turns messy UI, chat, and screen state
into structured evidence events.

axm-genesis is the frozen trust kernel. It seals and verifies. It is the only
component that mints receipts.

axm-core compiles knowledge. It turns source documents into deterministic
knowledge shards. It does not decide attention.

axm-chat is a spoke, not the whole memory system. It turns conversation records
into queryable, reviewable shards.

axm-capability-claim-test is the reality harness. It evaluates whether a claim
is supported, contradicted, frozen, or untested. It evaluates claims. It does
not create authority.

axm-embodied is the physical-world evidence spoke. It performs event-triggered
high-resolution capture with forensic continuity when the world changes fast
enough to require proof.

GhostBox is level 2. It ingests the objects the others emit and computes where
the tension is.

## The mental model

ScreenGhost answers: what did the surface show.
axm-genesis answers: can this record be verified.
axm-core answers: what knowledge can be compiled from sources.
axm-chat answers: what did the conversation record contain.
GhostBox answers: where is the tension.
Claim test answers: what claims survive contact with the evidence.
axm-embodied answers: what changed in the physical world fast enough to require
forensic capture.

## The four states every boundary must distinguish

Proven. Faithfully captured or cryptographically verified. A proven capture
means the record is trustworthy, not that its content is true.

Simulated. Generated, modeled, or synthetic. Never treated as record.

Frozen. Sealed and immutable. Change requires a new artifact, not an edit.

Untested. Present but not yet checked against evidence.

Claims carry a parallel but distinct vocabulary once tested: supported,
contradicted, frozen, untested. A claim verdict is about a claim standing
against evidence. A provenance state is about an artifact. They are not the
same field and must not be collapsed.

## Data flow

1. ScreenGhost emits EvidenceEvent. axm-embodied emits PhysicalEvidenceEvent
   when the physical world triggers capture.
2. axm-core emits KnowledgeShard. axm-chat emits ConversationShard.
3. axm-genesis seals shards and issues ShardReceipt. Anyone can verify against
   a receipt. Only genesis issues one.
4. GhostBox ingests events and shard references, computes attention, and emits
   AttentionFinding. Findings carry the input references and surface claims.
5. axm-capability-claim-test consumes surfaced claims plus evidence references
   and emits ClaimCheckResult with a verdict.

Nothing in this flow lets a downstream component overwrite an upstream record.
Attention points at the record. It does not edit it.

## Non-goals for this pass

No repo merges. No monolith. No feature expansion. No new storage authority in
GhostBox. No relocation of sealing out of axm-genesis. The first useful output
is a repo role map and a contract layer. Once those exist, repos evolve
independently against a stable boundary.
