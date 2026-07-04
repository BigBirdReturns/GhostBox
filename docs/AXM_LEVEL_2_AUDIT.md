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
| **GhostBox** | ✅ interop `AttentionFinding` | ✅ | ✅ (no sealing present) | ✅ | ✅ at boundary | ⚠️ **gap in internals** | ✅ |
| **ScreenGhost** | ✅ `EvidenceEvent` (new shim) | ✅ | ✅ (no sealing present) | ✅ | ✅ | ✅ **guarded** | ✅ |
| axm-genesis | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-core | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-chat | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-capability-claim-test | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |
| axm-embodied | not-verified | not-verified | not-verified | n/a | not-verified | not-verified | not-verified |

## Findings, with evidence

### F1 — GhostBox derives persistent semantic coordinates from salted `hash()` (check 6, gap)

The interop **boundary** is clean: `src/ghostbox/interop/contracts.py` content-
addresses every ID (`content_id` over canonical bytes, BLAKE3 when present,
SHA-256 fallback, with the algorithm tag embedded in the ID). Verified by running
the module's self-test.

The GhostBox **internals** are not clean. Node placement — which *is* a
persistent semantic identity — is derived from Python's per-process-salted
`hash()` in three places:

- `src/ghostbox/integration.py:374-375` — `type_ = (hash(event.topic) % 99) + 1`
  and `subtype = (hash(event.text[:50]) % 99) + 1`.
- `src/ghostbox/sources/photonic.py:363` — `instance = hash(state.screen) % 9999 + 1`.
- `src/axiom/adapters/schemaorg.py:167` — type bucket from `hash(schema_type)`.

Because `PYTHONHASHSEED` randomizes these per process, the same event yields
different coordinates across restarts, so a `SemanticID` is not reproducible and
will not survive a repo boundary. This is exactly the failure the contract warns
about, present one layer below the clean seam.

Not flagged: `src/axiom/core.py:165,260` define `__hash__` dunders
(`return hash(self.code)` / `hash(self.id)`). Those are legitimate in-memory
hashability for set/dict membership, not persistent IDs, and are correct.

**Recommendation (out of scope for this pass, logged):** derive coordinates via
the same content-addressing used at the boundary — e.g. route these through a
stable digest of canonical bytes rather than builtin `hash()`. This is an
internals fix, not a contract change.

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
states are distinguishable at every emitted object. The one substantive gap is
**internal to GhostBox** (F1: salted-hash coordinates), and the fix already
exists, tested, in a sibling repo (F2). Five repos remain `not-verified` and must
not be treated as passing until audited against source.
