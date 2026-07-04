"""AXM Sovereign Spine v0 — end-to-end against the REAL genesis kernel.

Unlike tests/test_genesis_custody_spoke.py (which injects a fake verifier), these
drive the real ``axm-verify`` / ``axm-build`` CLIs. They skip cleanly where the
kernel is not installed, so the environment dependency is explicit, never faked.

Control question, as assertions: seal a real record, verify it outside the AI,
let GhostBox annotate without rewriting custody, export, and still verify after
GhostBox is removed.
"""
from __future__ import annotations

import pytest

from ghostbox.interop.contracts import VerifyStatus
from ghostbox.interop.genesis_spoke import (
    CustodyOutcome,
    GenesisCustodySpoke,
    GenesisTrustKernel,
)

from spine_v0.attention import surface_finding
from spine_v0.exit_test import verify_detached
from spine_v0.genesis_cli import (
    RealGenesisVerifier,
    keygen,
    kernel_available,
    seal_sample_record,
    sealed_shard_from_dir,
)

pytestmark = pytest.mark.skipif(
    not kernel_available(),
    reason="axm-genesis kernel (axm-verify / axm-build) not on PATH",
)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    base = tmp_path_factory.mktemp("spine")
    shard_dir, oob_pub, key_id = seal_sample_record(base)
    return sealed_shard_from_dir(shard_dir), oob_pub, key_id, shard_dir, base


def _spoke(oob_pub, key_id):
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key=str(oob_pub), trusted_key_id=key_id)
    return GenesisCustodySpoke(kernel)


def test_seal_identity_is_genesis_derived(sealed):
    s, *_ = sealed
    assert s.shard_id.startswith("sh1_")
    assert s.suite == "axm-hybrid1"
    assert s.merkle_root  # projected from the manifest


def test_real_verify_takes_custody_and_trusts(sealed):
    s, oob_pub, key_id, *_ = sealed
    spoke = _spoke(oob_pub, key_id)
    entry = spoke.ingest_sealed_shard(s)
    assert entry.outcome is CustodyOutcome.VERIFIED
    assert entry.status is VerifyStatus.PASS
    assert spoke.is_trusted(s.shard_id)
    assert spoke.findings == ()  # clean seal → nothing flagged


def test_wrong_out_of_band_key_fails_closed(sealed):
    s, _oob, _kid, _shard_dir, base = sealed
    _, foreign_pub = keygen(base / "foreign", name="attacker")  # a key that did NOT sign this shard
    spoke = _spoke(foreign_pub, "attacker")
    entry = spoke.ingest_sealed_shard(s)
    assert entry.outcome is CustodyOutcome.UNVERIFIED
    assert entry.status is VerifyStatus.FAIL          # E_SIG_INVALID from the real kernel
    assert not spoke.is_trusted(s.shard_id)           # trust BLOCKED
    assert len(spoke.findings) == 1
    assert spoke.findings[0].tension_type == "unverified_seal"


def test_no_key_is_ghostbox_owned_never_calls_cli(sealed):
    s, *_ = sealed
    verifier = RealGenesisVerifier()
    spoke = GenesisCustodySpoke(GenesisTrustKernel(verifier, trusted_key=None))
    entry = spoke.ingest_sealed_shard(s)
    assert entry.status is VerifyStatus.NO_TRUSTED_KEY
    assert entry.outcome is CustodyOutcome.UNVERIFIED
    assert verifier.last_code is None                  # the real CLI was never invoked
    assert not spoke.is_trusted(s.shard_id)


def test_attention_is_downstream_and_untested(sealed):
    s, *_ = sealed
    finding = surface_finding(s)
    assert finding.input_refs == [s.shard_id]          # genesis id, verbatim
    assert finding.provenance.value == "untested"      # surfaces, does not assert truth
    assert finding.tension_type == "surfaced_record"


def test_custody_retains_only_the_reference(sealed):
    s, oob_pub, key_id, *_ = sealed
    entry = _spoke(oob_pub, key_id).ingest_sealed_shard(s)
    assert entry.shard_id == s.shard_id
    assert not hasattr(entry, "merkle_root")           # no shard body in GhostBox's record
    assert not hasattr(entry, "signature")


def test_exit_property_record_survives_ghostbox_removal(sealed):
    _s, oob_pub, _kid, shard_dir, _base = sealed
    res = verify_detached(shard_dir, oob_pub)
    assert res["status"] == "PASS"
    assert res["exit_code"] == 0
    assert res["ghostbox_involved"] is False           # verified with only bytes + oob pub


def test_ghostbox_cannot_seal():
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key="x")
    with pytest.raises(PermissionError):
        kernel.seal("/some/shard/dir")
