"""KnowledgeShardRef contract shape — reconciled against the real axm-core surface.

The probe established that axm-core knowledge output IS a genesis-sealed shard
(compiled through the genesis compiler). So KnowledgeShardRef must sit ABOVE
SealedShard: it carries axm-core knowledge metadata only, mints no identity, and
delegates verification to the landed custody seam. These are contract-level tests
(a fake kernel stands in for the real one); no genesis kernel is required.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from ghostbox.interop.contracts import KnowledgeShardRef, SealedShard, VerifyStatus


def make_sealed(shard_id: str = "sh1_deadbeef", shard_dir: str = "/shards/k", created_at: str = "2026-07-04T00:00:00Z") -> SealedShard:
    return SealedShard(shard_id=shard_id, shard_dir=shard_dir, merkle_root="merkle-root", sealed_at=created_at)


class _RecordingKernel:
    """Fake TrustKernel: records what it was asked to verify, refuses to seal."""

    def __init__(self, status: VerifyStatus) -> None:
        self.status = status
        self.verified = None
        self.key = None

    def seal(self, shard_dir):  # pragma: no cover - must never be called by a ref
        raise PermissionError("knowledge layer never seals")

    def verify(self, shard, *, trusted_key):
        self.verified = shard
        self.key = trusted_key
        return self.status


# --- identity comes from the sealed shard, never minted ---------------------


def test_identity_comes_from_sealed_not_minted():
    s = make_sealed("sh1_cafef00d")
    ref = KnowledgeShardRef.over(s)
    assert ref.shard_id == "sh1_cafef00d"
    assert ref.shard_id == ref.sealed.shard_id
    # there is no independent shard_id field to become a second source of truth
    assert "shard_id" not in {f.name for f in fields(ref)}


def test_shard_id_cannot_be_overridden():
    ref = KnowledgeShardRef.over(make_sealed("sh1_real"))
    with pytest.raises((AttributeError, FrozenInstanceError)):
        ref.shard_id = "sh1_forged"  # read-only property on a frozen ref


def test_construction_requires_a_sealed_shard():
    # A knowledge ref cannot exist without the underlying custody object.
    with pytest.raises(TypeError):
        KnowledgeShardRef()  # type: ignore[call-arg]


# --- it is a ref above SealedShard, not a custody object --------------------


def test_ref_carries_no_custody_material():
    ref = KnowledgeShardRef.over(make_sealed())
    names = {f.name for f in fields(ref)}
    for custody_field in ("signature", "merkle_root", "manifest", "suite"):
        assert custody_field not in names  # custody lives on the sealed shard
    assert names == {"sealed", "compiler", "source_refs", "compiled_at"}


def test_is_not_the_sealed_shard_itself():
    ref = KnowledgeShardRef.over(make_sealed())
    assert not isinstance(ref, SealedShard)  # references custody, is not custody


# --- verification delegates through the sealed shard seam -------------------


def test_verification_delegates_to_sealed_through_the_seam():
    s = make_sealed()
    ref = KnowledgeShardRef.over(s)
    kernel = _RecordingKernel(VerifyStatus.PASS)

    result = ref.verify(kernel, trusted_key="/out-of-band/pub")

    assert result is VerifyStatus.PASS
    assert kernel.verified is s          # verified the underlying SealedShard...
    assert kernel.verified is ref.sealed # ...the exact same object, not a copy
    assert kernel.key == "/out-of-band/pub"  # out-of-band anchor passed through


def test_ref_has_no_verification_logic_of_its_own():
    # Non-PASS from the seam is returned verbatim; the ref adds no custody logic.
    ref = KnowledgeShardRef.over(make_sealed())
    assert ref.verify(_RecordingKernel(VerifyStatus.FAIL), trusted_key="k") is VerifyStatus.FAIL
    assert ref.verify(_RecordingKernel(VerifyStatus.NO_TRUSTED_KEY), trusted_key="") is VerifyStatus.NO_TRUSTED_KEY


# --- field reconciliation ---------------------------------------------------


def test_no_summary_field():
    assert "summary" not in {f.name for f in fields(KnowledgeShardRef.over(make_sealed()))}


def test_compiled_at_maps_to_manifest_created_at():
    ref = KnowledgeShardRef.over(make_sealed(created_at="2026-07-04T12:34:56Z"))
    assert ref.compiled_at == "2026-07-04T12:34:56Z"  # == sealed.sealed_at (manifest created_at)


def test_compiler_default_is_the_forge_pipeline():
    # Reconciled against the live forge proof: the producing surface is the forge.
    assert KnowledgeShardRef.over(make_sealed()).compiler == "axm-forge"


def test_compiler_is_free_knowledge_metadata():
    assert KnowledgeShardRef.over(make_sealed(), compiler="axm-core").compiler == "axm-core"
