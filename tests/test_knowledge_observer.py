"""GhostBox knowledge observer — end-to-end against a REAL genesis-sealed shard.

Uses spine_v0's real forge/seal helpers to produce a genesis-sealed knowledge
shard (2 entities, 1 claim), wraps it in a KnowledgeShardRef, and drives the
observer through the LANDED custody seam. Skips cleanly without the kernel.

Control question, as assertions: GhostBox observes a real knowledge shard ONLY
after verifying its underlying genesis-sealed record, then adds findings without
becoming the knowledge or custody owner.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from ghostbox.interop.contracts import KnowledgeShardRef, VerifyStatus
from ghostbox.interop.genesis_spoke import GenesisCustodySpoke, GenesisTrustKernel
from ghostbox.interop import knowledge_observer as ko_module
from ghostbox.interop.knowledge_observer import KnowledgeObserver

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
def kref(tmp_path_factory):
    base = tmp_path_factory.mktemp("kobs")
    shard_dir, oob_pub, key_id = seal_sample_record(base)
    sealed = sealed_shard_from_dir(shard_dir)
    ref = KnowledgeShardRef.over(sealed, source_refs=["docs/sample.md"])
    return ref, oob_pub, key_id, shard_dir, base


def _observer(oob_pub, key_id):
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key=str(oob_pub), trusted_key_id=key_id)
    return KnowledgeObserver(GenesisCustodySpoke(kernel))


def _digest_tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --- verified -> findings downstream of the verified shard_id ---------------


def test_verified_shard_produces_findings_downstream_of_shard_id(kref):
    ref, oob_pub, key_id, *_ = kref
    obs = _observer(oob_pub, key_id)
    res = obs.observe(ref)

    assert res.verified is True and res.status is VerifyStatus.PASS
    assert res.shard_id == ref.shard_id
    assert len(res.findings) >= 1
    for f in res.findings:
        assert ref.shard_id in f.input_refs          # downstream of the verified shard_id
        assert f.tension_type == "knowledge_claim"
        assert f.provenance.value == "untested"      # observation, not truth


# --- fail closed: unverified shards are never observed as trusted -----------


def test_wrong_key_blocks_trusted_observation(kref):
    ref, _oob, _kid, _sd, base = kref
    _, foreign_pub = keygen(base / "foreign", name="attacker")
    obs = _observer(foreign_pub, "attacker")
    res = obs.observe(ref)

    assert res.verified is False and res.status is VerifyStatus.FAIL
    assert res.findings == ()                        # NO knowledge findings over an untrusted shard
    assert obs.findings == ()


def test_missing_key_is_no_anchor_before_verification(kref):
    ref, *_ = kref
    verifier = RealGenesisVerifier()
    obs = KnowledgeObserver(GenesisCustodySpoke(GenesisTrustKernel(verifier, trusted_key=None)))
    res = obs.observe(ref)

    assert res.status is VerifyStatus.NO_TRUSTED_KEY
    assert res.verified is False and res.findings == ()
    assert verifier.last_code is None                # the CLI was never reached


# --- producer metadata is never treated as custody --------------------------


def test_source_refs_and_compiler_are_metadata_not_custody(kref):
    ref, oob_pub, key_id, *_ = kref
    assert ref.compiler == "axm-forge"
    assert ref.source_refs == ["docs/sample.md"]

    res = _observer(oob_pub, key_id).observe(ref)
    for f in res.findings:
        # findings key to genesis-owned ids only, never to producer source_refs
        assert "docs/sample.md" not in f.input_refs
        assert all(r.startswith(("sh1_", "c1_", "e1_")) for r in f.input_refs)


# --- GhostBox owns only findings + the external reference -------------------


def test_ghostbox_retains_only_findings_and_shard_id_reference(kref):
    ref, oob_pub, key_id, *_ = kref
    obs = _observer(oob_pub, key_id)
    res = obs.observe(ref)

    assert res.shard_id == ref.shard_id
    for custody_field in ("manifest", "signature", "merkle_root", "suite"):
        assert not hasattr(res, custody_field)       # no custody material in the result
    assert obs.findings == res.findings              # only its own finding state


def test_observation_never_rewrites_the_sealed_shard(kref):
    ref, oob_pub, key_id, _sd, _base = kref
    shard_dir = Path(ref.sealed.shard_dir)
    before = _digest_tree(shard_dir)
    _observer(oob_pub, key_id).observe(ref)
    assert _digest_tree(shard_dir) == before         # observation writes nothing into the shard


# --- reuses the custody seam; no duplicated genesis verification ------------


def test_observer_does_not_reimplement_or_call_verification():
    # Verification is the custody seam's job: the observer must not shell out to
    # the CLI or import genesis's crypto machinery. (Checks imports/calls, not the
    # docstring prose, which legitimately *names* axm-verify to say it isn't called.)
    src = inspect.getsource(ko_module)
    assert "import subprocess" not in src
    assert "axm_verify" not in src                   # no genesis verifier import
    assert "axm_build" not in src                    # no genesis compiler import
    assert "GenesisCustodySpoke" in src              # it reuses the landed custody seam
