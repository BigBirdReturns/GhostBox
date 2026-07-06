"""GhostBox Pixel Review Layer — human review strictly downstream of the observer.

Pure tests construct the observer's ``PixelObservationResult`` directly (it is
GhostBox-owned state), so they run without the kernel. One end-to-end test seals
a REAL pixel shard, observes it through the landed custody seam, then reviews it
-- it skips cleanly without axm-build/axm-verify.

Control question, as assertions: a human can record a bounded attention
disposition over a trusted pixel observation, and nothing in this layer can
touch the PNG, restate the evidence tier, or turn a review into a truth claim.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from ghostbox.interop.contracts import AttentionFinding, ProvenanceState, VerifyStatus
from ghostbox.interop.pixel_observer import PixelObservationResult
from ghostbox.interop.pixel_review import (
    PixelReviewLayer,
    PixelReviewRecord,
    ReviewDisposition,
    ReviewRefused,
)

IMG = "e0" * 32
MAN = "c2" * 32
FIXED = "2026-07-04T00:00:00+00:00"


def _review_finding(shard_id: str = "sh1_test") -> AttentionFinding:
    return AttentionFinding(
        tension_type="human_review_required",
        score=1.0,
        summary=f"Pixel evidence for shard {shard_id} requires human review.",
        input_refs=[shard_id, f"img:sha256:{IMG}"],
        created_at=FIXED,
    )


def _trusted_observation(shard_id: str = "sh1_test", **over) -> PixelObservationResult:
    kw = dict(
        shard_id=shard_id, verified=True, status=VerifyStatus.PASS, trusted=True,
        evidence_tier="pixel_capture", image_sha256=IMG, manifest_sha256=MAN,
        findings=(_review_finding(shard_id),),
    )
    kw.update(over)
    return PixelObservationResult(**kw)


# === intake: only trusted pixel observations are reviewable =================


def test_trusted_observation_is_admitted_and_pending():
    layer = PixelReviewLayer()
    sid = layer.submit(_trusted_observation())
    assert sid == "sh1_test" and layer.pending == ("sh1_test",)


def test_unverified_observation_is_refused():
    layer = PixelReviewLayer()
    obs = _trusted_observation(verified=False, trusted=False, status=VerifyStatus.NO_TRUSTED_KEY, findings=())
    with pytest.raises(ReviewRefused) as ei:
        layer.submit(obs)
    assert ei.value.reason == "not_verified" and layer.pending == ()


def test_untrusted_observation_is_refused():
    # verified seal but blocked by tier/hash discipline upstream -> not reviewable
    obs = _trusted_observation(trusted=False, findings=())
    with pytest.raises(ReviewRefused) as ei:
        PixelReviewLayer().submit(obs)
    assert ei.value.reason == "not_trusted"


def test_wrong_tier_is_refused_defense_in_depth():
    obs = _trusted_observation(evidence_tier="clipboard_capture")
    with pytest.raises(ReviewRefused) as ei:
        PixelReviewLayer().submit(obs)
    assert ei.value.reason == "wrong_tier"


def test_observation_without_review_finding_is_refused():
    obs = _trusted_observation(findings=())
    with pytest.raises(ReviewRefused) as ei:
        PixelReviewLayer().submit(obs)
    assert ei.value.reason == "no_review_finding"


# === the human decision ======================================================


def test_review_produces_record_with_verbatim_downstream_refs():
    layer = PixelReviewLayer()
    obs = _trusted_observation()
    layer.submit(obs)
    rec = layer.review("sh1_test", reviewer="analyst-1", disposition="escalate",
                       note="quote card differs from copied text", reviewed_at=FIXED)
    assert rec.shard_id == "sh1_test"                      # genesis id, verbatim
    assert rec.image_sha256 == IMG and rec.manifest_sha256 == MAN
    assert rec.answers_finding_id == obs.findings[0].finding_id  # answers THAT finding
    assert rec.disposition is ReviewDisposition.ESCALATE
    assert rec.reviewer == "analyst-1"
    assert rec.review_id.startswith("rev:")
    assert layer.pending == ()                             # resolved


def test_evidence_tier_is_carried_verbatim_and_immutable():
    layer = PixelReviewLayer()
    layer.submit(_trusted_observation())
    rec = layer.review("sh1_test", reviewer="analyst-1", disposition="dismiss")
    assert rec.evidence_tier == "pixel_capture"            # verbatim, never upgraded
    with pytest.raises((AttributeError, TypeError)):       # frozen: cannot be restated
        rec.evidence_tier = "platform_record"  # type: ignore[misc]
    # and review() exposes no parameter that could change the tier
    import inspect as _i
    assert "tier" not in _i.signature(PixelReviewLayer.review).parameters
    assert "evidence_tier" not in _i.signature(PixelReviewLayer.review).parameters


def test_truth_shaped_verdicts_are_not_representable():
    layer = PixelReviewLayer()
    layer.submit(_trusted_observation())
    for bad in ("authentic", "true", "verified_content", "platform_authentic", "legal_provenance"):
        with pytest.raises(ReviewRefused) as ei:
            layer.review("sh1_test", reviewer="analyst-1", disposition=bad)
        assert ei.value.reason == "invalid_disposition"
    assert {d.value for d in ReviewDisposition} == {"escalate", "dismiss", "needs_context"}


def test_review_requires_human_attribution():
    layer = PixelReviewLayer()
    layer.submit(_trusted_observation())
    for bad in ("", "   ", None):
        with pytest.raises(ReviewRefused) as ei:
            layer.review("sh1_test", reviewer=bad, disposition="dismiss")  # type: ignore[arg-type]
        assert ei.value.reason == "no_reviewer"
    assert layer.records == ()                             # nothing recorded


def test_unsubmitted_shard_cannot_be_reviewed():
    with pytest.raises(ReviewRefused) as ei:
        PixelReviewLayer().review("sh1_ghost", reviewer="analyst-1", disposition="dismiss")
    assert ei.value.reason == "not_submitted"


def test_review_history_is_append_only():
    layer = PixelReviewLayer()
    layer.submit(_trusted_observation())
    r1 = layer.review("sh1_test", reviewer="analyst-1", disposition="needs_context", reviewed_at=FIXED)
    r2 = layer.review("sh1_test", reviewer="analyst-2", disposition="escalate",
                      reviewed_at="2026-07-05T00:00:00+00:00")
    assert layer.records == (r1, r2)                       # both kept; nothing replaced
    assert layer.records_for("sh1_test") == (r1, r2)
    assert r1.review_id != r2.review_id


def test_review_record_is_an_opinion_not_a_fact():
    layer = PixelReviewLayer()
    layer.submit(_trusted_observation())
    rec = layer.review("sh1_test", reviewer="analyst-1", disposition="escalate",
                       note="this is definitely fake")  # reviewer-attributed text only
    assert rec.provenance is ProvenanceState.UNTESTED     # never proven
    assert "not page truth" in rec.non_assertions[0] or "not page truth" in rec.non_assertions
    assert "not OCR text" in rec.non_assertions


def test_review_ids_are_deterministic():
    def run():
        layer = PixelReviewLayer()
        layer.submit(_trusted_observation())
        return layer.review("sh1_test", reviewer="analyst-1", disposition="dismiss",
                            note="n", reviewed_at=FIXED)
    assert run().review_id == run().review_id


# === strictly downstream: this layer cannot touch the shard or the PNG ======


def test_review_layer_performs_no_filesystem_access():
    from ghostbox.interop import pixel_review

    src = inspect.getsource(pixel_review)
    for token in ("Path(", "open(", "read_bytes", "read_text", "write_bytes",
                  "write_text", "os.path", "import os", "import subprocess",
                  "import hashlib"):
        assert token not in src, f"review layer must not use {token!r} (downstream only)"


def test_review_layer_imports_no_screenghost_ocr_dom_or_browser():
    code = (
        "import importlib, sys\n"
        "importlib.import_module('ghostbox.interop.pixel_review')\n"
        "bad=[m for m in ('screenghost','pytesseract','PIL','cv2','torch','selenium',"
        "'playwright','lxml','bs4','xml.etree.ElementTree') "
        "if any(k==m or k.startswith(m+'.') for k in sys.modules)]\n"
        "print('BAD:'+','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    assert [l for l in out.stdout.splitlines() if l.startswith("BAD:")][0] == "BAD:"


# === end-to-end: seal -> custody -> observe -> human review =================


def test_end_to_end_real_shard_review(tmp_path):
    from spine_v0.genesis_cli import kernel_available

    if not kernel_available():
        pytest.skip("axm-genesis kernel not on PATH")
    from test_pixel_observer import _seal_pixel_shard, _spoke  # tests/ is on sys.path
    from ghostbox.interop.pixel_observer import PixelEvidenceObserver

    sealed, pub, png = _seal_pixel_shard(tmp_path, sidecar={"url": "https://example.social/x"})
    obs = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    layer = PixelReviewLayer()
    layer.submit(obs)
    rec = layer.review(sealed.shard_id, reviewer="analyst-1", disposition="escalate",
                       note="needs a second look")
    assert rec.shard_id == sealed.shard_id                # genesis id end to end
    assert rec.evidence_tier == "pixel_capture"
    # the PNG is untouched by the whole chain
    assert (Path(sealed.shard_dir) / "content" / "capture.png").read_bytes() == png
    # and the sealed record still verifies with GhostBox nowhere in the loop
    proc = subprocess.run(["axm-verify", "shard", sealed.shard_dir, "--trusted-key", str(pub)],
                          capture_output=True, text=True)
    assert proc.returncode == 0
