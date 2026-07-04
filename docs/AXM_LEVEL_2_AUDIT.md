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

### F5 — `ShardReceipt.hash_algorithm` is a declared constant, not a detected one (note, not a gap)

The contract's `ShardReceipt` defaults `hash_algorithm="BLAKE3"` while, in an
environment without the `blake3` package, the derived `receipt_id` tag falls back
to `sha256`. These are consistent by design — the field names the sealing
algorithm axm-genesis is expected to use; the `receipt_id` is only a local stable
reference and the signature is the authority — but a consumer should read the
algorithm tag *inside* the ID, not the declared field, when verifying. Logged so
it is not mistaken for a contradiction.

## Bottom line

The level-2 **boundary** is sound: roles are separated, sealing is reserved,
GhostBox owns no record, IDs are content-addressed, and the four provenance
states are distinguishable at every emitted object. The one substantive gap —
**F1: salted-hash coordinates internal to GhostBox** — has been **fixed and
verified** with a cross-process determinism regression test; the same fix was
already tested in a sibling repo (F2). With F1 closed, Field Zero is reproducible
at the semantic-coordinate layer on the informational substrate. Five repos
remain `not-verified` and must not be treated as passing until audited against
source.
