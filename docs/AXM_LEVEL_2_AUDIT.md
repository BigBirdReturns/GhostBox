# AXM Level-2 Alignment Audit

Status: first-pass audit output. Scope-limited and honest about it.

This is the "real first output" the role map asks for: the seven checks from
[`REPO_ROLE_MAP.md`](REPO_ROLE_MAP.md) run against each core-stack repo, recorded
as a table with findings. It is **not** a certification. It records what was
verified against source, what is a gap, and what was out of reach this session.

## Scope discipline

Two of the seven core repos are present in this session and were audited against
source: **GhostBox** and **ScreenGhost**. The other five (axm-genesis, axm-core,
axm-chat, axm-capability-claim-test, axm-embodied) were **not in session scope**
and are recorded as `not-verified`. No category borrows trust: a repo not
audited gets no green mark.

## The seven checks

1. Emits the object the role map assigns (or the gap is noted).
2. No repo re-mints an ID another repo assigned.
3. Sealing exists only in axm-genesis.
4. GhostBox holds no canonical storage.
5. Every emitted object carries an explicit provenance or verdict.
6. No persistent ID is derived from Python's built-in `hash()`.
7. Proven / simulated / frozen / untested are distinguishable at each boundary.

## Results

| Repo | 1 emits | 2 no re-mint | 3 sealing only in genesis | 4 no canonical store | 5 explicit provenance | 6 no salted-hash IDs | 7 four states distinguishable |
|---|---|---|---|---|---|---|---|
| **GhostBox** | ✅ interop `AttentionFinding` | ✅ | ✅ (no sealing present) | ✅ | ✅ at boundary | ✅ **fixed** (was gap; see F1) | ✅ |
| **ScreenGhost** | ✅ `EvidenceEvent` (new shim) | ✅ | ✅ (no sealing present) | ✅ | ✅ | ✅ **guarded** | ✅ |
| axm-genesis | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-core | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-chat | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-capability-claim-test | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-embodied | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |

## Findings, with evidence

### F1 — GhostBox derived persistent semantic coordinates from salted `hash()` — **FIXED** (check 6)

Status: **fixed and verified.** Opened as a gap in the first pass; closed in the
hardening pass after the regression tests below passed.

The interop **boundary** was already clean: `src/ghostbox/interop/contracts.py`
content-addresses every ID (`content_id` over canonical bytes, BLAKE3 when
present, SHA-256 fallback, algorithm tag embedded in the ID).

The GhostBox **internals** were not. Node placement — which *is* a persistent
semantic identity — was derived from Python's per-process-salted `hash()` in
three places. Because `PYTHONHASHSEED` randomizes those per process, the same
input yielded different coordinates across restarts, so a `SemanticID` was not
reproducible and would not survive a repo boundary — the exact failure the
contract warns about, one layer below the clean seam.

**Fix applied:**

- Added `stable_hash_int()` to `src/axiom/core.py` — a SHA-256-derived,
  unsalted integer over the utf-8 bytes.
- Replaced the salted derivations at all three sites:
  - `src/ghostbox/integration.py` — `type_` / `subtype` now via `stable_hash_int`.
  - `src/ghostbox/sources/photonic.py` — screen `instance` now via `stable_hash_int`.
  - `src/axiom/adapters/schemaorg.py` — the `_hash_type` bucket now via `stable_hash_int`.
- Added `tests/test_stable_hash.py`. The load-bearing case
  (`test_coordinates_stable_across_hashseed`) derives the coordinates in three
  separate Python processes started with `PYTHONHASHSEED` = `0`, `12345`, and
  `random`, and asserts they agree — with a sensitivity guard asserting that
  builtin `hash()` of the same string genuinely *differs* across those seeds, so
  the test cannot pass on a no-op. A source-level guard
  (`test_no_salted_hash_in_persistent_coordinate_paths`) prevents regression.
  **4/4 tests pass.**

Not changed (correctly): `src/axiom/core.py:165,260` define `__hash__` dunders
(`return hash(self.code)` / `hash(self.id)`). Those are legitimate in-memory
hashability for set/dict membership, not persistent IDs, and must keep builtin
`hash()`.

**Field Zero reproducibility note.** Before this fix, Field Zero was functionally
demonstrated but not fully reproducible at the semantic-coordinate layer. After
this fix, Field Zero can be described as reproducible on the informational
substrate, subject to the remaining limits documented in
[`THREAT_GEOMETRY.md`](THREAT_GEOMETRY.md) (single-substrate support is not
invariance; physical substrates remain untested).

### F2 — ScreenGhost already hardened against F1's bug class (check 6, pass with guard)

ScreenGhost treats salted `hash()` for stable seeds as a known, fixed defect:

- `core/population.py:190` explicitly uses a process-independent seed and
  comments that "builtin hash() is salted per process."
- `tests/test_adapter_conformance.py:237-239` asserts "stable hashing only:
  sha256 via hashlib, never builtin hash() for seeds" and greps its own source
  to forbid `random.Random(hash(`.
- `tests/test_release.py:47` keeps "the RC0 bug" (salted `hash()`) on the record
  as a regression guard.

The intake repo passes the check the attention repo fails — with a test that
prevents the regression. Worth propagating that guard into GhostBox.

### F3 — Sealing is absent, not misplaced (check 3, pass by absence)

Neither GhostBox nor ScreenGhost mints receipts or signatures. The
`ShardReceipt` shape lives in the interop contract as a *consumed* object;
nothing in either repo produces one. This is correct: sealing stays reserved to
axm-genesis. The check passes because the authority is absent where it should be
absent, not because it was verified present-and-correct in genesis (genesis was
not in scope).

### F4 — GhostBox holds no canonical storage across the boundary (check 4, pass)

GhostBox persists working state (SQLite for DIND claims/alerts, session DBs), but
it does not hold the canonical copy of any upstream shard, nor does it re-issue
upstream IDs. `KnowledgeShardRef` / `ConversationShardRef` carry externally
assigned `shard_id`s verbatim; GhostBox points, it does not own. The level-2
invariant holds at the seam.

### F5 — `ShardReceipt.hash_algorithm` is a declared constant, not a detected one (note; folded into F6)

The contract's `ShardReceipt` declares `hash_algorithm` (the algorithm used to
produce `content_hash`) as a field, while the id-derivation digest is a separate,
now-pinned SHA-256 (F6/P1). These are different things and were previously easy
to conflate. As of F6 the `hash_algorithm` field participates in the identity
directly (it is part of `receipt_body_id`), so the declared value can no longer
drift silently from what the id covers. A consumer still reads the algorithm tag
*inside* the id when verifying the id itself.

### F6 — Review-driven hardening: id pin (P1) and receipt-identity split (P2) — **applied**

Two automated-review findings on the PRs were both real and are fixed.

**P1 — boundary ids pinned to SHA-256.** The id digest previously used BLAKE3
when the `blake3` package was importable and SHA-256 otherwise, with the tag
baked into the id. So the same canonical object produced `evt:b3:...` on one
machine and `evt:sha256:...` on another, breaking the contract's own promise that
one observation yields one `event_id` across ScreenGhost and GhostBox. Fixed by
pinning the algorithm to SHA-256 as a fixed contract constant on **both** sides
(`ghostbox/interop/contracts.py` and ScreenGhost's `core/evidence_event.py`).
Guarded by `tests/test_interop_ids.py` and the ScreenGhost conformance test
(`_HASH_TAG == "sha256"`, no optional-backend branch).

**P2 — receipt identity split to stop authority aliasing.** `receipt_id`
previously omitted the signature, so two distinct signatures over the same
shard/content/`sealed_at` (deterministic replay, retry, key rotation) aliased to
one id. Fixed by splitting identity: `receipt_body_id` (unsigned body:
`shard_id`, `content_hash`, `hash_algorithm`, `sealed_at`) and `receipt_id` /
`attestation_id` (the signed authority record, including `signer`,
`signature_algorithm`, `signature`). The body id never depends on a signature and
`receipt_id` is derived after signing and never part of the signed payload, so
there is **no circular hash/signature dependency**. `TrustKernel.verify` is
documented to key on the signed record (`receipt_id`); "is this body sealed" keys
on `receipt_body_id`. Guarded by `tests/test_interop_ids.py`.

**Governance flag (open, non-code).** The receipt-identity split refines how a
**genesis-owned** shape is identified. It is landed here as a **proposed
amendment** to the genesis contract and is marked as such in code and in
`INTEROP_CONTRACTS.md`. It must be confirmed by the axm-genesis kernel owner
before it is treated as authoritative. This is the one gate on the GhostBox PR
that is not machine-checkable; it should be answered before that PR merges.

### F7 — Unsigned commit history, inconsistent with the project's provenance claims (open, human-side)

The commits carrying this work are not cryptographically signed: the ephemeral
build environment holds no signing key. The author identity is correct
(`Claude <noreply@anthropic.com>`), but GitHub shows the commits as "Unverified."

For a normal repo this is cosmetic. For this stack it is not. The product is
verifiable, signed, timestamped provenance — axm-genesis seals with ML-DSA-44,
and the Threat Geometry paper turns on the difference between *proven* and merely
*asserted*. An unattested change history is therefore a live inconsistency: the
receipts for the receipts-machine are missing. This is the same class of gap as
F1 (salted hash), one level up — F1 meant a coordinate could not prove it was the
same coordinate across processes; an unsigned commit means the history cannot
prove who authored it or that it was not altered. On a repo whose product is "you
can verify this record," an unverifiable commit log is the first thing a hostile
reader points at.

Not blockable from here (no key in the environment), and holding the PRs hostage
to signing would be the wrong call. But the accurate status is "unattested,
inconsistent with what the project claims, smallest fix is human-side commit
signing" — not "nothing to fix." Resolution is signing keyed to the human owner,
not the container. The clean form maps onto the F6 genesis gate: unsigned commits
from automated sessions may land on a branch, but merge to `main` only via a
**signed merge commit from the owner** — which makes "the kernel owner" and "me
at 3am" the same *verifiable* identity instead of an honor-system claim.

Held open, like F6. Decide the signing policy **before either PR merges**: the
merge commit is the first chance to attest `main` top-down, and merging unsigned
starts the provenance chain with an unverifiable first link.

## Bottom line

The level-2 **boundary** is sound: roles are separated, sealing is reserved,
GhostBox owns no record, IDs are content-addressed and now algorithm-pinned, and
the four provenance states are distinguishable at every emitted object. **F1**
(salted-hash coordinates internal to GhostBox) is **fixed and verified** with a
cross-process determinism test; **F6** closed two review findings — the id
algorithm is pinned (P1) and receipt identity is split to stop authority aliasing
without circular math (P2). With F1 closed, Field Zero is reproducible at the
semantic-coordinate layer on the informational substrate. Three open items remain
non-code: the **genesis-owner sign-off** on the receipt-identity amendment (F6),
the **commit-signing policy** for an attested history (F7, decide before either PR
merges), and five repos still `not-verified` — none may be treated as passing
until audited against source.
