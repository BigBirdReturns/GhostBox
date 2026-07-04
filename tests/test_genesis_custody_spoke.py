"""Custody-edge conformance: GhostBox ingests a genesis-sealed shard, verifies
it *through genesis*, preserves ownership boundaries, and adds attention state
without rewriting the seal.

The control question turned into assertions:

    Can GhostBox point at a sealed AXM record, verify it through genesis,
    preserve ownership boundaries, and then add attention state without
    rewriting the receipt?

The contract is now corrected to genesis's real surface (SealedShard +
VerifyStatus + TrustKernel.verify(shard, *, trusted_key) -> VerifyStatus), so
the adapter *conforms to* it rather than routing around it. Genesis is not live
here (evidence tier: design-doc + test-claim); the genesis verifier is injected
as a fake returning the frozen exit codes. The boundary behavior -- not the
crypto -- is under test.
"""
from dataclasses import asdict

import pytest

from ghostbox.interop.contracts import SealedShard, TrustKernel, VerifyStatus
from ghostbox.interop.genesis_spoke import (
    CustodyOutcome,
    GenesisCustodySpoke,
    GenesisTrustKernel,
)

FIXED = "2026-07-04T00:00:00+00:00"
OOB_KEY = "/out-of-band/keys/publisher.pub"  # NOT the shard's own sig/publisher.pub


class CountingVerifier:
    """Fake genesis verifier. Returns a fixed frozen exit code and records every
    call so tests can assert single-shot (no-retry) behavior and which key was
    used as the trust anchor."""

    def __init__(self, code: int) -> None:
        self.code = code
        self.calls: list[tuple[str, str]] = []

    def __call__(self, shard_dir: str, trusted_key: str) -> int:
        self.calls.append((shard_dir, trusted_key))
        return self.code


def make_shard(shard_id: str = "sh1_deadbeef", shard_dir: str = "/shards/gold") -> SealedShard:
    return SealedShard(
        shard_id=shard_id,
        shard_dir=shard_dir,
        merkle_root="merkle-root-beef",
        sealed_at=FIXED,
    )


def spoke_with(code: int, *, trusted_key=OOB_KEY):
    verifier = CountingVerifier(code)
    kernel = GenesisTrustKernel(verifier, trusted_key=trusted_key, trusted_key_id="gold-v2-provisional")
    return GenesisCustodySpoke(kernel), verifier


# --- the adapter conforms to the corrected contract -------------------------


def test_adapter_conforms_to_trustkernel_protocol():
    # M3 is resolved: the adapter satisfies the real verify boundary rather than
    # routing around a bool convenience method.
    kernel = GenesisTrustKernel(CountingVerifier(0), trusted_key=OOB_KEY)
    assert isinstance(kernel, TrustKernel)


def test_sealed_shard_has_no_verified_field():
    # Verification is an act (VerifyStatus), never a property stored on the
    # sealed artifact.
    assert not hasattr(make_shard(), "verified")


# --- happy path -------------------------------------------------------------


def test_verified_shard_enters_custody_as_trusted():
    spoke, verifier = spoke_with(0)  # PASS
    shard = make_shard()

    entry = spoke.ingest_sealed_shard(shard)

    assert entry.outcome is CustodyOutcome.VERIFIED
    assert entry.status is VerifyStatus.PASS
    assert spoke.is_trusted(shard.shard_id)
    assert spoke.findings == ()          # nothing to flag on a clean seal
    assert len(verifier.calls) == 1      # exactly one verify, no retry


def test_trust_anchor_is_the_out_of_band_key_not_the_shard():
    spoke, verifier = spoke_with(0)
    spoke.ingest_sealed_shard(make_shard())
    # The key handed to genesis is the out-of-band anchor, never derived from
    # the shard being verified.
    assert verifier.calls[0][1] == OOB_KEY


# --- fail closed ------------------------------------------------------------


def test_verify_false_emits_unverified_finding_and_blocks_trust():
    spoke, verifier = spoke_with(1)  # FAIL (e.g. E_SIG_INVALID)
    shard = make_shard()

    entry = spoke.ingest_sealed_shard(shard)

    assert entry.outcome is CustodyOutcome.UNVERIFIED
    assert entry.status is VerifyStatus.FAIL
    assert not spoke.is_trusted(shard.shard_id)         # trust is BLOCKED
    assert len(spoke.findings) == 1
    finding = spoke.findings[0]
    assert finding.tension_type == "unverified_seal"
    assert finding.finding_id == entry.finding_id
    assert len(verifier.calls) == 1                     # recorded, not retried


def test_malformed_shard_is_unverified_and_distinct_from_failed():
    spoke, verifier = spoke_with(2)  # MALFORMED
    shard = make_shard()

    entry = spoke.ingest_sealed_shard(shard)

    assert entry.outcome is CustodyOutcome.UNVERIFIED
    assert entry.status is VerifyStatus.MALFORMED       # distinct from FAIL
    assert not spoke.is_trusted(shard.shard_id)
    assert len(spoke.findings) == 1


def test_no_trusted_key_refuses_and_never_reaches_genesis():
    spoke, verifier = spoke_with(0, trusted_key=None)  # verifier would PASS if called
    shard = make_shard()

    entry = spoke.ingest_sealed_shard(shard)

    assert entry.status is VerifyStatus.NO_TRUSTED_KEY
    assert entry.outcome is CustodyOutcome.UNVERIFIED
    assert not spoke.is_trusted(shard.shard_id)         # never PASS without an anchor
    assert verifier.calls == []                          # short-circuits before genesis


# --- ownership boundary: ids verbatim, no re-mint ---------------------------


def test_shard_id_preserved_verbatim_no_remint():
    spoke, _ = spoke_with(0)
    shard = make_shard(shard_id="sh1_cafef00d")

    entry = spoke.ingest_sealed_shard(shard)

    assert entry.shard_id == "sh1_cafef00d"  # genesis-assigned id, verbatim


def test_unverified_finding_carries_shard_id_verbatim():
    spoke, _ = spoke_with(1)
    spoke.ingest_sealed_shard(make_shard(shard_id="sh1_abc123"))
    assert spoke.findings[0].input_refs == ["sh1_abc123"]


# --- no write-back: sealed shard is immutable input, body not retained ------


def test_sealed_shard_is_not_mutated_and_body_is_not_retained():
    spoke, _ = spoke_with(0)
    shard = make_shard()
    before = asdict(shard)

    entry = spoke.ingest_sealed_shard(shard)

    # The sealed shard handed in is unchanged (it is a frozen dataclass).
    assert asdict(shard) == before
    # GhostBox's ledger holds only the reference id, never the shard body:
    # no suite, no merkle root, no signature crossed the boundary into storage.
    assert not hasattr(entry, "suite")
    assert not hasattr(entry, "merkle_root")
    assert not hasattr(entry, "signature")


# --- sealing authority stays in genesis -------------------------------------


def test_ghostbox_cannot_seal():
    kernel = GenesisTrustKernel(CountingVerifier(0), trusted_key=OOB_KEY)
    with pytest.raises(PermissionError):
        kernel.seal("/shards/gold")
