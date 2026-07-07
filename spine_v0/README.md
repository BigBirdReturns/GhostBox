# AXM Sovereign Spine v0

A small system that proves the **opposite primitive** to a collapsed vendor
operating layer (Palantir/Maven-shaped "mavenOS"): the record survives the
tool, the AI cannot rewrite custody, and findings stay downstream of verifiable
evidence.

This is **not** a backend-replacement claim. It is a working sovereign-record
primitive: **seal → verify (outside the AI) → annotate → export → remove**, and
the record still verifies after the AI is gone.

## The pipeline

```
seal (real axm-genesis)                     spine_v0/genesis_cli.py
  → SealedShard (genesis-derived sh1_ id)   axm_verify.crypto.derive_shard_id
  → verify through GhostBox custody seam     ghostbox.interop.genesis_spoke (PR #2)
        real axm-verify CLI injected as the verifier
  → GhostBox AttentionFinding (downstream)   spine_v0/attention.py
  → reviewable packet                        spine_v0/packet.py
  → EXIT TEST: verify with GhostBox removed  spine_v0/exit_test.py
```

GhostBox's PR #2 custody code is used **unchanged**. The harness injects the
real `axm-verify` CLI as the verifier and hands GhostBox a `SealedShard`;
GhostBox decides trust exactly as before.

## Run it

Prerequisite — the real genesis kernel must be installed (this is the honest
environment dependency; the tests skip cleanly without it):

```bash
pip install -e '/path/to/axm-genesis[dev]'   # brings axm-verify / axm-build + dilithium-py backend
```

Then, from the repo root:

```bash
PYTHONPATH=src python -m spine_v0.run_spine --out spine_v0_out
PYTHONPATH=src python -m pytest tests/test_spine_v0.py -q     # real-kernel suite (skips if absent)
```

A captured run is in [`example_packet.md`](example_packet.md).

## The boundary (inherited from PR #2, held against the real kernel)

- GhostBox **does not seal** — `seal()` is refused; sealing authority stays in axm-genesis.
- GhostBox **delegates verification** to the real `axm-verify` CLI; it never opens the manifest or signature.
- Trust is anchored to an **out-of-band key**; the shard's embedded `sig/publisher.pub` is never used as authority.
- Missing trusted key is a GhostBox-owned **`NO_TRUSTED_KEY`**, decided *before* invoking the CLI — because `axm-verify` exit 2 is ambiguous between a usage error and a malformed shard (confirmed in the probe).
- GhostBox **retains only** the genesis-assigned `shard_id`; no shard body, no signature, no receipt object.
- The `AttentionFinding` is **downstream of verification** (`UNTESTED`); it surfaces content for claim-testing, it does not assert truth or rewrite custody.

## Scope

Genesis seam only. `KnowledgeShardRef` (axm-core) and `ConversationShardRef`
(axm-chat) are future feeds — untouched here. ScreenGhost is a future intake
feed — untouched. No genesis mutation, no PR #1, no mavenOS-replacement claim.

Status update (2026-07-06): ConversationShardRef and KnowledgeShardRef have since been reconciled against their real surfaces elsewhere in the repo (see docs/GENESIS_EDGE_MISMATCHES.md), and ScreenGhost's pixel evidence has its own observer (docs/PIXEL_EVIDENCE_OBSERVER_V0.md). All remain untouched within spine_v0 itself, which stays scoped to the genesis seam only.

## Genesis live-probe receipts (carried forward, honest)

Established directly against the real kernel in this environment:

| Check | Result |
|---|---|
| Gold shard verify (out-of-band key) | **PASS** (exit 0) |
| Fresh disposable shard seal + verify | **PASS** (exit 0) |
| Wrong out-of-band key | **FAIL** / `E_SIG_INVALID` (exit 1) |
| Missing trusted key | **refused** (exit 2, never PASS) |
| Detached verification (bytes + oob pub only) | **PASS** |
| Crypto backend | `dilithium-py` fallback active; `liboqs` (`oqs`) absent |

**Backend caveat, stated plainly:** the ML-DSA-44 path runs on the pure-Python
`dilithium-py` fallback. It is functional in this environment; it is **not
load-proven**.

## Control question

> Can a user seal a real record, verify it with an out-of-band key, let GhostBox
> annotate it without rewriting custody, export the evidence, and still verify
> the record after GhostBox is removed?

**v0 answer: yes** — demonstrated live (`spine v0: OK — verified=pass,
detached-verify=PASS, ghostbox-in-exit-path=False`), at real-kernel tier, on a
single small record with the fallback crypto backend. The record survives the
attention service.
