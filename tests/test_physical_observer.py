"""GhostBox physical observer — end-to-end against a REAL genesis-sealed shard.

Seals a physical-capture shard with the real kernel, replicating exactly the
shape the probed axm-embodied ``compile_frame_capsule`` produces (namespace
``embodied/capture``, sealed ``capture_manifest.json`` with the
``physical_capture`` tier + limits, an opaque ``frames.bin`` payload, event-log
source, ``declared_trigger`` / ``trigger_source`` / ``opened_capture_window`` /
``closed_capture_window`` / ``content_sha256`` claims, and the judge-verified
``ext/streams@1.jsonl`` index), wraps it in a SealedShard, and drives the
observer through the LANDED custody seam. Skips cleanly without the kernel.

The fixture seals through the kernel's own compiler entry point
(``axm_build.compiler_generic.compile_generic_shard``) because the frozen CLI
has no extension-table flag — this is the SAME call the real axm-embodied
compiler makes, with the same config vocabulary. The frame payload is a labeled
sim blob: the observer never opens ``frames.bin``, so its bytes are sealed but
unread.

Control question, as assertions: GhostBox observes real physical evidence ONLY
after verifying its genesis-sealed record, never reads frame bytes, never
trusts a shard whose declared counts disagree with its own sealed index, and
adds bounded findings without becoming the evidence or custody owner.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from ghostbox.interop.contracts import VerifyStatus
from ghostbox.interop.genesis_spoke import GenesisCustodySpoke, GenesisTrustKernel
from ghostbox.interop import physical_observer as po_module
from ghostbox.interop.physical_observer import PHYSICAL_TIER, PhysicalEvidenceObserver

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

_TIER_LIMITS = [
    "opaque sensor bytes within declared trigger windows only",
    "not identity",
    "not activity or semantic classification",
    "not continuous coverage (gaps between windows are declared, not hidden)",
    "not platform truth",
    "not legal-grade provenance by itself",
]


def seal_physical_record(
    workdir: Path,
    *,
    tier: str = PHYSICAL_TIER,
    frames_kept_declared: int | None = None,
    with_index: bool = True,
) -> tuple[Path, Path, str]:
    """Seal a physical-capture record shaped exactly like real axm-embodied output.

    Same sealed content set (``frames.bin`` sim blob + ``capture_manifest.json``),
    same event-log source, same claim vocabulary, same ``streams@1`` index rows,
    sealed through the same kernel compiler entry point the real
    ``compile_frame_capsule`` uses. Returns (shard_dir, out_of_band_public_key_path,
    publisher_name). ``tier`` / ``frames_kept_declared`` / ``with_index`` exist to
    build the untrusted variants.
    """
    from axm_build.compiler_generic import CompilerConfig, compile_generic_shard
    from axm_build.sign import hybrid1_keygen

    workdir.mkdir(parents=True, exist_ok=True)
    sensor = "sim-doorcam-01"
    session = "ghostbox-test-session"
    frames = [b"SIMFRAME-1" * 8, b"SIMFRAME-2" * 8]
    frame_hashes = [hashlib.sha256(b).hexdigest() for b in frames]

    events = [
        {"evt": "capture_trigger", "frame_id": 1, "reason": "motion",
         "source": "sim-pir-3", "sensor_id": sensor},
        {"evt": "capture_window_opened", "frame_id": 1, "first_kept_frame_id": 1,
         "sensor_id": sensor},
        {"evt": "frame_kept", "frame_id": 1, "sensor_id": sensor,
         "stream_refs": {"frames": {"file": "frames.bin", "offset": 4, "length": 157}},
         "content_sha256": frame_hashes[0], "chain": "aa" * 32},
        {"evt": "frame_kept", "frame_id": 2, "sensor_id": sensor,
         "stream_refs": {"frames": {"file": "frames.bin", "offset": 161, "length": 157}},
         "content_sha256": frame_hashes[1], "chain": "bb" * 32},
        {"evt": "capture_window_closed", "frame_id": 2, "sensor_id": sensor},
    ]
    event_lines = [json.dumps(e) for e in events]
    events_path = workdir / "events.jsonl"
    events_path.write_text("\n".join(event_lines) + "\n", encoding="utf-8")

    # Candidates exactly as the real frame compiler extracts them.
    candidates = [
        {"subject": sensor, "predicate": "declared_trigger", "object": "motion",
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[0]},
        {"subject": "trigger/frame-1", "predicate": "trigger_source", "object": "sim-pir-3",
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[0]},
        {"subject": sensor, "predicate": "opened_capture_window", "object": "frame-1",
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[1]},
        {"subject": "frame-1", "predicate": "content_sha256", "object": frame_hashes[0],
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[2]},
        {"subject": "frame-2", "predicate": "content_sha256", "object": frame_hashes[1],
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[3]},
        {"subject": sensor, "predicate": "closed_capture_window", "object": "frame-2",
         "object_type": "literal:string", "tier": 1, "evidence": event_lines[4]},
    ]
    candidates_path = workdir / "candidates.jsonl"
    candidates_path.write_text(
        "\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8"
    )

    frames_bin = workdir / "frames.bin"
    frames_bin.write_bytes(b"AXFF" + b"".join(frames))  # sim blob; sealed but never read

    capture_manifest = workdir / "capture_manifest.json"
    capture_manifest.write_text(json.dumps({
        "evidence_tier": tier,
        "evidence_tier_limits": _TIER_LIMITS,
        "session_id": session,
        "sensor_id": sensor,
        "started_at": "2026-07-06T00:00:00Z",
        "frames_observed": 4,
        "frames_kept": frames_kept_declared if frames_kept_declared is not None else len(frames),
        "triggers": 1,
        "chain_genesis": "cc" * 32,
        "note": "sim capture for the GhostBox physical observer suite",
    }, indent=2), encoding="utf-8")

    streams_rows = [
        {"frame_id": 1, "stream": "frames", "file": "frames.bin", "offset": 4,
         "length": 157, "status": "VERIFIED", "content_hash": frame_hashes[0]},
        {"frame_id": 2, "stream": "frames", "file": "frames.bin", "offset": 161,
         "length": 157, "status": "VERIFIED", "content_hash": frame_hashes[1]},
    ]

    a, b = hybrid1_keygen()
    sk, pk = (a, b) if len(a) == 3904 else (b, a)
    pub_path = workdir / "publisher.pub"
    pub_path.write_bytes(pk)

    shard_dir = workdir / "shard"
    cfg = CompilerConfig(
        source_path=events_path,
        candidates_path=candidates_path,
        out_dir=shard_dir,
        private_key=sk,
        publisher_id="@axm_embodied",
        publisher_name="AXM Embodied",
        namespace="embodied/capture",
        created_at="2026-07-06T00:00:00Z",
        title="Frame capture capsule (GhostBox suite)",
        license_spdx="Apache-2.0",
        profiles=(),
        extra_content=(
            ("frames.bin", frames_bin),
            ("capture_manifest.json", capture_manifest),
        ),
        extra_ext={"streams@1": streams_rows} if with_index else None,
    )
    assert compile_generic_shard(cfg), "kernel rejected the physical fixture shard"
    return shard_dir, pub_path, "physical_publisher"


@pytest.fixture(scope="module")
def pshard(tmp_path_factory):
    base = tmp_path_factory.mktemp("pobs")
    shard_dir, oob_pub, key_id = seal_physical_record(base)
    sealed = sealed_shard_from_dir(shard_dir)
    return sealed, oob_pub, key_id, shard_dir, base


def _observer(oob_pub, key_id):
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key=str(oob_pub), trusted_key_id=key_id)
    return PhysicalEvidenceObserver(GenesisCustodySpoke(kernel))


def _digest_tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --- verified + consistent -> bounded findings downstream of shard_id -------


def test_verified_physical_shard_is_trusted_with_bounded_findings(pshard):
    sealed, oob_pub, key_id, *_ = pshard
    obs = _observer(oob_pub, key_id)
    res = obs.observe(sealed)

    assert res.verified is True and res.status is VerifyStatus.PASS
    assert res.trusted is True
    assert res.evidence_tier == PHYSICAL_TIER
    assert res.frames_kept == 2
    assert res.chain_genesis == "cc" * 32
    kinds = [f.tension_type for f in res.findings]
    assert kinds == [
        "physical_evidence_available",
        "opaque_sensor_capture",
        "capture_continuity_available",
        "human_review_required",
    ]
    for f in res.findings:
        assert sealed.shard_id in f.input_refs       # downstream of the verified shard_id


def test_trusted_findings_ship_the_real_tier_non_assertions(pshard):
    sealed, oob_pub, key_id, *_ = pshard
    res = _observer(oob_pub, key_id).observe(sealed)

    available = res.findings[0]
    for limit in ("not identity", "not platform truth",
                  "not legal-grade provenance by itself"):
        assert limit in available.summary            # attention, never adjudication
    review = res.findings[-1]
    assert review.provenance.value == "untested"     # a flag, not a proven fact


# --- fail closed: unverified shards are never observed as trusted -----------


def test_wrong_key_blocks_trusted_observation(pshard):
    sealed, _oob, _kid, _sd, base = pshard
    _, foreign_pub = keygen(base / "foreign", name="attacker")
    obs = _observer(foreign_pub, "attacker")
    res = obs.observe(sealed)

    assert res.verified is False and res.status is VerifyStatus.FAIL
    assert res.trusted is False
    assert res.evidence_tier is None                 # manifest never read
    assert res.findings == ()
    assert obs.findings == ()


def test_missing_key_is_no_anchor_before_verification(pshard):
    sealed, *_ = pshard
    verifier = RealGenesisVerifier()
    obs = PhysicalEvidenceObserver(GenesisCustodySpoke(GenesisTrustKernel(verifier, trusted_key=None)))
    res = obs.observe(sealed)

    assert res.status is VerifyStatus.NO_TRUSTED_KEY
    assert res.verified is False and res.findings == ()
    assert verifier.last_code is None                # the CLI was never reached


# --- a verified NON-physical shard is a finding, not content ----------------


def test_verified_non_physical_shard_is_flagged_not_read(tmp_path):
    # A real sealed shard from another surface (the spine sample) verifies, but
    # it carries no capture manifest: one bounded finding, nothing trusted.
    shard_dir, oob_pub, key_id = seal_sample_record(tmp_path)
    sealed = sealed_shard_from_dir(shard_dir)
    res = _observer(oob_pub, key_id).observe(sealed)

    assert res.verified is True and res.trusted is False
    assert len(res.findings) == 1
    assert res.findings[0].tension_type == "not_physical_evidence"


def test_unexpected_tier_blocks_trusted_observation(tmp_path):
    shard_dir, oob_pub, key_id = seal_physical_record(tmp_path, tier="pixel_capture")
    sealed = sealed_shard_from_dir(shard_dir)
    res = _observer(oob_pub, key_id).observe(sealed)

    assert res.verified is True and res.trusted is False
    assert res.evidence_tier == "pixel_capture"
    assert len(res.findings) == 1
    assert res.findings[0].tension_type == "unexpected_evidence_tier"


# --- the sealed index must agree with the declared counts -------------------


def test_missing_stream_index_blocks_trusted_observation(tmp_path):
    shard_dir, oob_pub, key_id = seal_physical_record(tmp_path, with_index=False)
    sealed = sealed_shard_from_dir(shard_dir)
    res = _observer(oob_pub, key_id).observe(sealed)

    assert res.verified is True and res.trusted is False
    assert len(res.findings) == 1
    assert res.findings[0].tension_type == "physical_stream_index_missing"


def test_declared_count_disagreeing_with_index_blocks_trust(tmp_path):
    # frames_kept says 3, the sealed judge index carries 2 rows: the
    # log/disk-disagreement class is never trusted past.
    shard_dir, oob_pub, key_id = seal_physical_record(tmp_path, frames_kept_declared=3)
    sealed = sealed_shard_from_dir(shard_dir)
    res = _observer(oob_pub, key_id).observe(sealed)

    assert res.verified is True and res.trusted is False
    assert len(res.findings) == 1
    assert res.findings[0].tension_type == "physical_stream_index_mismatch"


# --- GhostBox owns only findings + external refs -----------------------------


def test_result_carries_no_custody_material(pshard):
    sealed, oob_pub, key_id, *_ = pshard
    res = _observer(oob_pub, key_id).observe(sealed)

    assert res.shard_id == sealed.shard_id
    for custody_field in ("manifest", "signature", "merkle_root", "suite"):
        assert not hasattr(res, custody_field)


def test_observation_never_rewrites_the_sealed_shard(pshard):
    sealed, oob_pub, key_id, shard_dir, _base = pshard
    before = _digest_tree(Path(shard_dir))
    _observer(oob_pub, key_id).observe(sealed)
    assert _digest_tree(Path(shard_dir)) == before   # observation writes nothing


def test_frame_bytes_are_never_opened():
    # The observer must not read frames.bin at all — frames are opaque sensor
    # bytes, and even hashing them here would start a second judge.
    src = inspect.getsource(po_module)
    assert "frames.bin" not in src.replace("``frames.bin``", "")  # prose only


# --- reuses the custody seam; no duplicated genesis or spoke logic ----------


def test_observer_does_not_reimplement_or_call_verification():
    src = inspect.getsource(po_module)
    assert "import subprocess" not in src
    assert "axm_verify" not in src                   # no genesis verifier import
    assert "axm_build" not in src                    # no genesis compiler import
    assert "GenesisCustodySpoke" in src              # it reuses the landed custody seam


def test_observer_never_imports_the_embodied_spoke():
    src = inspect.getsource(po_module)
    assert "axm_embodied" not in src
    assert "import hashlib" in src                   # only for the manifest digest ref
