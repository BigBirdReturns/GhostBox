# AXM Repo Role Map

Status: draft for audit confirmation

One repo, one job. The core stack is the first-pass dependency set for the
level-2 alignment. The support stack is real but later. It is not a first-pass
dependency and should not be wired in during this session.

## Core stack

### 1. GhostBox — Level 2
Attention geometry, semantic tension, Field Zero. Ingests shards and events,
emits AttentionFinding. Reads the record, does not become the record.
Emits: AttentionFinding.

### 2. ScreenGhost — Surface intake
Messy UI, chat, and screen state into structured evidence. Intake, not truth.
Emits: EvidenceEvent.

### 3. axm-genesis — Trust kernel
Signed, content-addressed shards and verification. Frozen. The only issuer of
receipts.
Emits: ShardReceipt.

### 4. axm-core — Knowledge compiler
Source documents into deterministic knowledge artifacts. Compiles knowledge,
does not decide attention.
Emits: KnowledgeShard (referenced as KnowledgeShardRef).

### 5. axm-chat — Conversation spoke
Chat exports into queryable, reviewable shards. A spoke, not the whole memory
system.
Emits: ConversationShard (referenced as ConversationShardRef).

### 6. axm-capability-claim-test — Reality harness
Claim compatibility, boundary testing, falsifiability. Evaluates claims, does
not create authority.
Emits: ClaimCheckResult.

### 7. axm-embodied — Physical evidence spoke
Event-triggered high-resolution capture and forensic continuity.
Emits: PhysicalEvidenceEvent.

## Support stack (later, not first-pass)

### 8. axm-tools
Public utility surfaces and demo distribution.

### 9. axm-zombie-adapter
Edge compute and distributed inference substrate.

### 10. axm-fleet / axm-sfn
Physical node custody, attestation, deployment, fabrication and edge control.

### 11. axm-world / axm-arc
Legibility, simulation, and playable cartridge or runtime surface.

## Boundary summary

| Repo | Role | Emits | Authority it does NOT hold |
| --- | --- | --- | --- |
| GhostBox | attention (L2) | AttentionFinding | storage, sealing, truth |
| ScreenGhost | intake | EvidenceEvent | truth |
| axm-genesis | trust kernel | ShardReceipt | attention |
| axm-core | knowledge compiler | KnowledgeShardRef | attention |
| axm-chat | conversation spoke | ConversationShardRef | whole memory system |
| axm-capability-claim-test | reality harness | ClaimCheckResult | authority creation |
| axm-embodied | physical spoke | PhysicalEvidenceEvent | truth |

## Audit checklist for the Claude Code session

For each core-stack repo, confirm:

- The repo actually emits the object listed above, or note the gap.
- No repo re-mints an ID that another repo assigned.
- Sealing exists only in axm-genesis.
- GhostBox holds no canonical storage.
- Every emitted object carries an explicit provenance or verdict.
- No persistent ID is derived from Python's built-in hash().
- Proven, simulated, frozen, and untested are distinguishable at each boundary.

Record findings against these seven checks per repo. That table is the real
first output of the audit.
