"""GhostBox Pixel Evidence Observer — over ScreenGhost Pixel Evidence v0 shards.

Drives the observer through the LANDED ``GenesisCustodySpoke`` against REAL
genesis-sealed pixel shards (capture.png + pixel_capture_manifest.json). The
pixel shard is sealed here with the genesis CLI directly -- GhostBox never
imports ScreenGhost. Kernel-dependent tests skip cleanly without axm-build /
axm-verify.

Control question, as assertions: GhostBox observes a verified pixel-evidence
shard, emits bounded attention findings, and leaves the screenshot, manifest, and
custody record owned entirely outside GhostBox.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from ghostbox.interop.contracts import ProvenanceState, SealedShard, VerifyStatus
from ghostbox.interop.genesis_spoke import (
    CustodyOutcome,
    GenesisCustodySpoke,
    GenesisTrustKernel,
)
from ghostbox.interop.pixel_observer import PixelEvidenceObserver, PixelObservationResult
from spine_v0.genesis_cli import RealGenesisVerifier, kernel_available, keygen, sealed_shard_from_dir

requires_kernel = pytest.mark.skipif(
    not kernel_available(), reason="axm-genesis kernel (axm-build / axm-verify) not on PATH"
)

AXM_BUILD = "axm-build"
SIDECAR = {"url": "https://example.social/status/123", "page_title": "a post that looked edited"}


def _png(w: int = 4, h: int = 3) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x11\x22\x33" * w for _ in range(h))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _seal_pixel_shard(work: Path, *, tier: str = "pixel_capture", sidecar: dict | None = None,
                      wrong_image_hash: bool = False, png: bytes | None = None):
    """Seal a pixel-evidence shard with the real genesis CLI. Returns
    (SealedShard, pub_key_path, png_bytes). No ScreenGhost import."""
    png = png if png is not None else _png()
    content = work / "content"
    content.mkdir(parents=True, exist_ok=True)
    (content / "capture.png").write_bytes(png)
    manifest = {
        "image_sha256": ("de" * 32) if wrong_image_hash else hashlib.sha256(png).hexdigest(),
        "image_bytes": len(png),
        "source_label": "Chrome",
        "capture_method": "sharex_scrolling",
        "evidence_tier": tier,
        "image_format": "png",
        "png_filename": "capture.png",
    }
    if sidecar:
        manifest.update(sidecar)
    (content / "pixel_capture_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    src = "capture.png is a pixel_capture of rendered-surface\n"
    (content / "source.txt").write_text(src, encoding="utf-8")
    cands = [
        {"type": "entity", "namespace": "screenghost/pixel", "label": "capture.png", "entity_type": "pixel_capture"},
        {"type": "entity", "namespace": "screenghost/pixel", "label": "rendered-surface", "entity_type": "rendered_surface"},
        {"type": "claim", "subject_label": "capture.png", "predicate": "is_pixel_capture_of",
         "object_label": "rendered-surface", "object_type": "entity", "tier": 1,
         "evidence": {"source_file": "source.txt", "byte_start": 0, "byte_end": len(src.encode()) - 1, "text": src.strip()}},
    ]
    (work / "cand.jsonl").write_text("\n".join(json.dumps(c) for c in cands) + "\n", encoding="utf-8")
    key_path, pub_path = keygen(work / "keys", name="publisher")
    shard_dir = work / "shard"
    subprocess.run(
        [AXM_BUILD, "compile", str(work / "cand.jsonl"), str(content), str(shard_dir),
         "--private-key", str(key_path), "--namespace", "screenghost/pixel",
         "--title", "pixel", "--created-at", "2026-07-04T00:00:00Z"],
        check=True, capture_output=True, text=True,
    )
    return sealed_shard_from_dir(shard_dir), pub_path, png


def _spoke(pub_key_path) -> GenesisCustodySpoke:
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key=str(pub_key_path) if pub_key_path else None)
    return GenesisCustodySpoke(kernel)


def _kinds(result: PixelObservationResult) -> set:
    return {f.tension_type for f in result.findings}


# === requirement: verified pixel evidence shard produces findings ===========


@requires_kernel
def test_verified_pixel_shard_produces_findings(tmp_path):
    sealed, pub, _png_bytes = _seal_pixel_shard(tmp_path, sidecar=SIDECAR)
    obs = PixelEvidenceObserver(_spoke(pub))
    result = obs.observe(sealed)
    assert result.verified is True and result.trusted is True
    assert result.status is VerifyStatus.PASS
    assert result.evidence_tier == "pixel_capture"
    assert {"pixel_evidence_available", "rendered_surface_capture", "human_review_required",
            "capture_context_available"} <= _kinds(result)
    # bounded: no finding asserts page truth / OCR / platform authenticity
    for f in result.findings:
        assert "not OCR text" in f.summary or f.tension_type in {"human_review_required"}
        assert f.provenance in (ProvenanceState.PROVEN, ProvenanceState.UNTESTED)


@requires_kernel
def test_no_sidecar_omits_capture_context_finding(tmp_path):
    sealed, pub, _ = _seal_pixel_shard(tmp_path)  # no sidecar
    result = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    assert result.trusted is True
    assert "capture_context_available" not in _kinds(result)
    assert "pixel_evidence_available" in _kinds(result)


# === requirement: wrong key blocks observation ==============================


@requires_kernel
def test_wrong_key_blocks_observation(tmp_path):
    sealed, _pub, _ = _seal_pixel_shard(tmp_path)
    subprocess.run([AXM_BUILD, "keygen", str(tmp_path / "atk"), "--name", "attacker"],
                   check=True, capture_output=True, text=True)
    spoke = _spoke(tmp_path / "atk" / "attacker.pub")
    result = PixelEvidenceObserver(spoke).observe(sealed)
    assert result.verified is False and result.trusted is False
    assert result.status is VerifyStatus.FAIL
    assert result.findings == ()                        # observer adds no pixel findings
    # custody recorded the unverified_seal itself
    assert spoke.ledger[-1].outcome is CustodyOutcome.UNVERIFIED


# === requirement: missing key blocks via the existing no-anchor path ========


def test_missing_key_blocks_observation(tmp_path):
    # No kernel needed: NO_TRUSTED_KEY is decided before any CLI call.
    sealed = SealedShard(shard_id="sh1_dummy", shard_dir=str(tmp_path))
    spoke = _spoke(None)  # no out-of-band anchor
    result = PixelEvidenceObserver(spoke).observe(sealed)
    assert result.verified is False and result.trusted is False
    assert result.status is VerifyStatus.NO_TRUSTED_KEY
    assert result.findings == ()


# === requirement: malformed shard blocks observation ========================


@requires_kernel
def test_malformed_shard_blocks_observation(tmp_path):
    sealed, pub, _ = _seal_pixel_shard(tmp_path)
    # Corrupt the sealed manifest so genesis reports malformed/failed.
    (Path(sealed.shard_dir) / "manifest.json").write_text("{ this is not valid json", encoding="utf-8")
    result = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    assert result.verified is False and result.trusted is False
    assert result.status in (VerifyStatus.MALFORMED, VerifyStatus.FAIL)
    assert result.findings == ()


# === requirement: non-pixel_capture tier -> untrusted finding only ==========


@requires_kernel
def test_non_pixel_tier_emits_untrusted_finding_only(tmp_path):
    sealed, pub, _ = _seal_pixel_shard(tmp_path, tier="clipboard_capture")
    result = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    assert result.verified is True                      # the seal itself is fine
    assert result.trusted is False
    assert _kinds(result) == {"unexpected_evidence_tier"}
    assert "pixel_evidence_available" not in _kinds(result)


# === requirement: hash mismatch blocks trusted observation ==================


@requires_kernel
def test_hash_mismatch_blocks_trusted_observation(tmp_path):
    sealed, pub, _ = _seal_pixel_shard(tmp_path, wrong_image_hash=True)
    result = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    assert result.verified is True                      # signature verifies...
    assert result.trusted is False                      # ...but manifest hash != PNG bytes
    assert _kinds(result) == {"pixel_manifest_hash_mismatch"}


# === requirement: findings reference shard_id + pixel hash verbatim =========


@requires_kernel
def test_findings_reference_shard_id_and_pixel_hash_verbatim(tmp_path):
    sealed, pub, png = _seal_pixel_shard(tmp_path)
    result = PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    img_hash = hashlib.sha256(png).hexdigest()
    avail = next(f for f in result.findings if f.tension_type == "pixel_evidence_available")
    assert sealed.shard_id in avail.input_refs                     # genesis id, verbatim
    assert f"img:sha256:{img_hash}" in avail.input_refs            # pixel hash, verbatim
    assert result.image_sha256 == img_hash


# === requirement: PNG bytes remain unchanged ================================


@requires_kernel
def test_png_bytes_unchanged_by_observation(tmp_path):
    sealed, pub, png = _seal_pixel_shard(tmp_path)
    png_path = Path(sealed.shard_dir) / "content" / "capture.png"
    before = png_path.read_bytes()
    PixelEvidenceObserver(_spoke(pub)).observe(sealed)
    assert png_path.read_bytes() == before == png                 # never rewritten


# === requirement: reads manifest only after verification ====================


def test_observer_reads_manifest_only_after_verification(tmp_path, monkeypatch):
    # If _read_pixel_manifest / _sealed_png_sha256 were reached on a non-verified
    # shard, this blows up. A refusal must precede any content read.
    sealed = SealedShard(shard_id="sh1_dummy", shard_dir=str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("content was read before custody returned VERIFIED")

    obs = PixelEvidenceObserver(_spoke(None))  # NO_TRUSTED_KEY -> never verified
    monkeypatch.setattr(obs, "_read_pixel_manifest", boom)
    monkeypatch.setattr(obs, "_sealed_png_sha256", boom)
    result = obs.observe(sealed)
    assert result.verified is False and result.findings == ()


def test_fail_verdict_reads_no_content(tmp_path, monkeypatch):
    # Fake verifier returns FAIL (exit 1) with a key present -> still no read.
    kernel = GenesisTrustKernel(lambda d, k: 1, trusted_key="oob-key")
    obs = PixelEvidenceObserver(GenesisCustodySpoke(kernel))
    monkeypatch.setattr(obs, "_read_pixel_manifest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read on FAIL")))
    result = obs.observe(SealedShard(shard_id="sh1_x", shard_dir=str(tmp_path)))
    assert result.status is VerifyStatus.FAIL and result.findings == ()


# === requirement: detached record still verifies without GhostBox ===========


@requires_kernel
def test_detached_record_verifies_without_ghostbox(tmp_path):
    sealed, pub, _ = _seal_pixel_shard(tmp_path)
    # Verify with only the shard bytes + the oob pub, no GhostBox in the loop.
    proc = subprocess.run(["axm-verify", "shard", sealed.shard_dir, "--trusted-key", str(pub)],
                          capture_output=True, text=True)
    assert proc.returncode == 0


# === requirement: no ScreenGhost / OCR / DOM / clipboard / browser imports ==

_FORBIDDEN = (
    "screenghost", "core.pixel_evidence", "core.pixel_seal",
    "pytesseract", "PIL", "cv2", "easyocr", "torch", "torchvision",
    "selenium", "playwright", "pyppeteer",
    "xml.etree.ElementTree", "lxml", "bs4",
)


def test_observer_imports_no_screenghost_ocr_dom_clipboard_or_browser():
    # Import ONLY the observer in a clean subprocess; assert none of the
    # forbidden modules (ScreenGhost, OCR, image model, DOM, browser) were pulled.
    code = (
        "import importlib, sys\n"
        "importlib.import_module('ghostbox.interop.pixel_observer')\n"
        f"bad=[m for m in {_FORBIDDEN!r} "
        "if any(k==m or k.startswith(m+'.') for k in sys.modules)]\n"
        "print('BAD:'+','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("BAD:")][0]
    assert line == "BAD:", f"observer pulled in forbidden modules: {line}"


def test_observer_source_has_no_forbidden_imports():
    # Check import statements, not docstring prose (which names OCR/ScreenGhost/DOM
    # only to say they are excluded).
    from ghostbox.interop import pixel_observer

    src = inspect.getsource(pixel_observer)
    for token in ("import screenghost", "from screenghost", "import pytesseract",
                  "from PIL", "import cv2", "import selenium", "from selenium",
                  "import playwright", "from playwright", "import lxml", "from lxml",
                  "xml.etree", "import torch", "clipboard"):
        assert token not in src, f"observer must not use {token!r}"
    # It cannot call axm-verify directly: it imports no subprocess (custody owns
    # verification). Check the import statement, not the docstring prose.
    assert "import subprocess" not in src
