# AXM Sovereign Spine v0 — reviewable packet

**Product claim:** Custody survives tool removal: seal, verify outside the AI, annotate, export, remove.

## Record identity
- `shard_id`: `sh1_d479ba41d22da1a5eb36e238689aa656ab767c68f626fbfe56419fe0131806db`
- genesis-assigned (sh1_ = BLAKE3 of the manifest); carried verbatim, never minted by GhostBox

## Signature context
- suite: `axm-hybrid1` — Ed25519 || ML-DSA-44 (both must verify)
- merkle_root: `629fba200da041e7a7fd8e4a104ca7574e948197337a5926c2064e801dfe5d72`
- sealed_at: `2026-07-04T00:00:00Z`
- authority is the out-of-band trusted key, NOT the shard's embedded sig/publisher.pub

## Trust anchor
- source: `/tmp/axm_spine_v0_elxqtjdc/keys/spine_publisher.pub` (out-of-band)

## Verification
- status: **pass**
- verifier: `axm-verify shard <dir> --trusted-key <oob_pub> (real kernel)`
- genesis said: `{"shard": "/tmp/axm_spine_v0_elxqtjdc/shard", "status": "PASS", "error_count": 0, "errors": [], "profiles_checked": [], "profiles_unchecked": []}`

## Custody (GhostBox)
- outcome: **verified** (status `pass`)
- trusted_key_id: `spine_publisher`
- retained by GhostBox: shard_id reference only

## GhostBox attention finding
- `find:b3:bcbee454057252833f10acd197d6252c` — surfaced_record (provenance: untested)
- Verified genesis record sh1_d479ba41d22da1a5eb36e238689aa656ab767c68f626fbfe56419fe0131806db entered attention. Surface content is available for downstream claim-testing; attention asserts nothing about whether the content is true.
- input_refs: ['sh1_d479ba41d22da1a5eb36e238689aa656ab767c68f626fbfe56419fe0131806db']

## Boundary (what GhostBox did NOT do)
- GhostBox did not seal (seal() is refused; sealing authority stays in axm-genesis).
- GhostBox delegated verification to the real axm-verify CLI; it never opened the manifest or signature.
- Trust was anchored to an out-of-band key; the shard's embedded sig/publisher.pub was never used as authority.
- Missing trusted key is GhostBox-owned NO_TRUSTED_KEY, decided before invoking the CLI (exit 2 is ambiguous between a usage error and a malformed shard).
- GhostBox retained only the genesis-assigned shard_id; no shard body, no signature, no receipt object.
- The AttentionFinding is downstream of verification (UNTESTED); it does not assert the content is true and does not rewrite custody.

## Exit test — record survives GhostBox removal
- GhostBox involved: **False**
- detached verify status: **PASS** (exit 0)
- verified with only the shard bytes + the out-of-band public key.

## Genesis live-probe receipts (carried forward)
- gold_shard_verify: PASS (exit 0)
- fresh_disposable_shard_verify: PASS (exit 0)
- wrong_out_of_band_key: FAIL / E_SIG_INVALID (exit 1)
- missing_trusted_key: refused (exit 2, never PASS)
- detached_verification: PASS (shard bytes + out-of-band pub only)
- crypto_backend: dilithium-py fallback active; liboqs (oqs) absent
- backend_caveat: functional in this environment; NOT load-proven
