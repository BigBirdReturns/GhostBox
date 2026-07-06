# Genesis custody seam — decision record

Status: the interop contract is **corrected to genesis's real surface** for the
custody seam. Decision on the record: **reality is authority** — the contract
describes the real genesis custody seam, so where the mirror diverged from the
kernel the *contract* was corrected (once), not the adapter (N times). This
supersedes the earlier open-mismatch list: M1–M3 below are now **resolved in
`contracts.py`**, and the adapter conforms to the corrected seam instead of
routing around it.

## The decision

The alternative was to label the contract a mirror-side projection and let every
spoke preserve the kernel's semantics locally. That is drift by construction — N
private corrections, no shared truth, a second custody model. Rejected. The
correction is written down once, in the shared contract, so there is one seam.

## What changed in the contract — genesis seam only

| Was (mismatch) | Now (corrected) |
|---|---|
| **M1** `ShardReceipt`, a genesis "receipt" genesis does not produce, with a GhostBox-minted `receipt_id` | **`SealedShard`**: a frozen *reference* to the sealed shard directory (`manifest.json` + `sig/`). Authoritative identity is the genesis-assigned `shard_id` (`sh1_` + BLAKE3 of the manifest) + signature context. No `receipt_id`. |
| **M2** `signature_algorithm="ML-DSA-44"` (drops the hybrid half) | `suite="axm-hybrid1"` — Ed25519 + ML-DSA-44, both must verify. |
| **M3** `TrustKernel.verify(receipt) -> bool` (no anchor, no failure class, no shard body) | `TrustKernel.verify(shard, *, trusted_key) -> VerifyStatus` — out-of-band anchor required; preserves PASS / FAIL / MALFORMED / NO_TRUSTED_KEY. Bool is derived locally as `status is PASS`, never at the boundary. |
| `verified: bool` field on the sealed object | Removed. Verification is an *act* returning `VerifyStatus`, never a property stored on the artifact. |

## The adapter now conforms (it does not route around)

`genesis_spoke.py` implements the corrected `TrustKernel`
(`isinstance(kernel, TrustKernel)` is asserted in the suite), ingests a
`SealedShard`, delegates verification to an injected genesis verifier — it never
opens the manifest or the signature itself — fails closed on any non-PASS, and
retains only the `shard_id` reference. `tests/test_genesis_custody_spoke.py`:
11/11 against the frozen exit codes.

## M0 stays clean

GhostBox retains no shard-body field. `CustodyLedgerEntry` holds `shard_id`
(a reference, verbatim) plus its own outcome / status / finding ids —
structurally nothing that could carry `suite`, `merkle_root`, or a signature,
even by mistake. No write-back path exists.

## Scope — what is NOT claimed

This corrects the **genesis seam only**. In the contract, `KnowledgeShardRef`
(axm-core) and `ConversationShardRef` (axm-chat) were annotated **NOT YET
VERIFIED** — mirror-side guesses until each edge is wired and reconciled against
its real surface on its own branch. Correcting genesis is not a claim that the
stack is corrected. The remaining core objects (`EvidenceEvent`,
`AttentionFinding`, `ClaimCheckResult`, `PhysicalEvidenceEvent`) are untouched by
this pass. The `KnowledgeShardRef` edge, when it is built, inherits this one
documented custody pattern — it does not get to choose a second.

**Status updates since this pass:**

- `KnowledgeShardRef` (axm-core): reconciled on `claude/axm-core-knowledge-edge`
  (PR #4). Composes over `SealedShard`; observer verifies through the landed
  custody seam. One custody pattern, as required above.
- `ConversationShardRef` (axm-chat): reconciled on
  `claude/session-planning-1h0xlw`. The probe ran a real `axm-chat import`
  (generic export → sealed shard, namespace `chat/conversation`, publisher
  `@axm_chat`) and verified it detached with an out-of-band key. Reality: the
  spoke's output IS a genesis-sealed shard, so the ref now composes over
  `SealedShard` exactly like `KnowledgeShardRef` (independent `shard_id`,
  `summary`, and `provenance` fields removed; `recorded_at` maps to the sealed
  manifest's `metadata.created_at`). One contract-vs-reality mismatch found and
  documented: axm-chat exports NO shard-reference API (`import_export()` returns
  counts only), so `ConversationSpoke` is annotated as a consumer-side adapter
  role, not an axm-chat interface. `conversation_observer.py` inherits the one
  custody pattern; live-proven against the real probe shard (6 findings, all
  keyed to genesis ids).
- `PhysicalEvidenceEvent` / `EmbodiedSource` (axm-embodied): still mirror-side
  guesses — the last unreconciled edge.
