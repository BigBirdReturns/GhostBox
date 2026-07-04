"""GhostBox observer over ScreenGhost Pixel Evidence v0 records.

ScreenGhost seals a rendered-surface screenshot (``capture.png`` + a bounded
``pixel_capture_manifest.json``) into a genesis ``axm-hybrid1`` shard. This
observer lets GhostBox add attention findings over that pixel evidence WITHOUT
becoming its owner. The order is fixed and non-negotiable, and mirrors the landed
``KnowledgeObserver``:

    SealedShard -> custody verification (through the LANDED GenesisCustodySpoke)
    -> confirm evidence tier is ``pixel_capture`` -> confirm the manifest image
    hash matches the sealed PNG bytes -> GhostBox-owned AttentionFindings,
    strictly downstream of the verified ``shard_id``.

Enforced boundaries:
  - Reuses the landed custody seam for verification. It does NOT call
    ``axm-verify`` and does NOT duplicate genesis logic -- ``GenesisCustodySpoke``
    already owns that path.
  - Observation happens ONLY after a VERIFIED custody outcome. FAIL / MALFORMED /
    NO_TRUSTED_KEY yield no pixel findings (the custody spoke has already recorded
    the failure); the manifest is not read until custody returns VERIFIED.
  - The PNG is never created, modified, OCR'd, classified, or rewritten. It is
    read once, only to confirm its hash matches the manifest, and never retained.
  - No ScreenGhost import: GhostBox consumes the sealed shard shape + manifest,
    never ScreenGhost as an authority.
  - Findings are bounded: they never assert page truth, author identity, platform
    authenticity, legal provenance, OCR text, or the semantic content of the
    screenshot. Custody identity stays the genesis ``shard_id``; GhostBox stores
    only its own finding state and external refs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import AttentionFinding, ProvenanceState, VerifyStatus
from .genesis_spoke import CustodyOutcome, GenesisCustodySpoke

PIXEL_TIER = "pixel_capture"
MANIFEST_NAME = "pixel_capture_manifest.json"
CONTENT_DIR = "content"

# Sidecar-supplied context keys that, when present, unlock capture_context_available.
_CONTEXT_KEYS = ("url", "page_title", "app_name", "capture_tool", "captured_at")

# What GhostBox will NEVER assert about a pixel capture. Shipped in the finding so
# a downstream reader cannot mistake attention for adjudication.
_NON_ASSERTIONS = (
    "not page truth",
    "not author identity",
    "not platform authenticity",
    "not legal provenance",
    "not OCR text",
    "not the semantic content of the screenshot",
)


@dataclass(frozen=True)
class PixelObservationResult:
    """Outcome of observing one pixel-evidence shard. GhostBox-owned.

    Holds only the genesis-assigned ``shard_id`` (a reference, verbatim), the
    custody verdict, the (external) hashes read from the verified shard, and
    GhostBox's own findings -- no PNG bytes, no manifest body, no signature, no
    custody material.
    """

    shard_id: str
    verified: bool                     # custody returned PASS
    status: VerifyStatus
    trusted: bool                      # verified AND pixel_capture tier AND hash matches
    evidence_tier: Optional[str]
    image_sha256: Optional[str]
    manifest_sha256: Optional[str]
    findings: Tuple[AttentionFinding, ...]


class PixelEvidenceObserver:
    """Observe verified ScreenGhost pixel-evidence shards; annotate, never own."""

    def __init__(self, custody: GenesisCustodySpoke) -> None:
        self._custody = custody
        self._findings: List[AttentionFinding] = []

    @property
    def findings(self) -> Tuple[AttentionFinding, ...]:
        return tuple(self._findings)

    def observe(self, sealed) -> PixelObservationResult:
        # 1) Verify the sealed shard THROUGH the landed custody seam. No direct
        #    axm-verify, no duplicated genesis logic -- custody owns verification.
        entry = self._custody.ingest_sealed_shard(sealed)

        # 2) Fail closed: only a VERIFIED custody outcome unlocks observation. On
        #    FAIL / MALFORMED / NO_TRUSTED_KEY the custody spoke already recorded
        #    its unverified_seal finding; the observer reads NOTHING and adds no
        #    pixel findings.
        if entry.outcome is not CustodyOutcome.VERIFIED:
            return PixelObservationResult(
                shard_id=sealed.shard_id, verified=False, status=entry.status,
                trusted=False, evidence_tier=None, image_sha256=None,
                manifest_sha256=None, findings=(),
            )

        # 3) Only now -- after verification -- read the sealed pixel manifest.
        shard_dir = Path(sealed.shard_dir)
        manifest, manifest_sha256 = self._read_pixel_manifest(shard_dir)
        if manifest is None:
            f = self._untrusted(
                sealed.shard_id, "not_pixel_evidence",
                f"Verified shard {sealed.shard_id} has no readable "
                f"{MANIFEST_NAME}; not observed as pixel evidence.",
            )
            return self._result(sealed, entry.status, None, None, [f], trusted=False)

        tier = manifest.get("evidence_tier")
        image_sha256 = manifest.get("image_sha256")

        # 4) Confirm the evidence tier. A non-pixel_capture tier emits ONLY an
        #    untrusted finding -- never the trusted pixel findings.
        if tier != PIXEL_TIER:
            f = self._untrusted(
                sealed.shard_id, "unexpected_evidence_tier",
                f"Shard {sealed.shard_id} manifest tier is {tier!r}, not "
                f"{PIXEL_TIER!r}; blocked from trusted pixel observation.",
                extra_refs=[_img_ref(image_sha256)] if image_sha256 else None,
            )
            return self._result(sealed, entry.status, tier, image_sha256, [f], trusted=False)

        # 5) Confirm the manifest's image hash matches the sealed PNG bytes. The
        #    PNG is read once, only to hash it; it is never rewritten or retained.
        actual = self._sealed_png_sha256(shard_dir, manifest.get("png_filename", "capture.png"))
        if actual is None or actual != image_sha256:
            f = self._untrusted(
                sealed.shard_id, "pixel_manifest_hash_mismatch",
                f"Manifest image hash {image_sha256} does not match the sealed PNG "
                f"bytes ({actual}); blocked from trusted pixel observation.",
                extra_refs=[_img_ref(image_sha256)] if image_sha256 else None,
            )
            return self._result(sealed, entry.status, tier, image_sha256, [f], trusted=False)

        # 6) Verified pixel_capture with a matching hash -> bounded findings,
        #    strictly downstream of the verified shard_id.
        findings = self._surface_pixel_findings(sealed.shard_id, manifest, image_sha256, manifest_sha256)
        self._findings.extend(findings)
        return self._result(sealed, entry.status, tier, image_sha256, findings, trusted=True,
                            manifest_sha256=manifest_sha256)

    # -- finding construction (bounded, downstream of shard_id) --------------

    def _surface_pixel_findings(
        self, shard_id: str, manifest: Dict[str, Any], image_sha256: str, manifest_sha256: str
    ) -> List[AttentionFinding]:
        capture_method = manifest.get("capture_method")
        source_label = manifest.get("source_label")
        # External refs, carried verbatim. Never a minted id.
        refs = [shard_id, _img_ref(image_sha256), _manifest_ref(manifest_sha256)]

        available = AttentionFinding(
            tension_type="pixel_evidence_available",
            score=1.0,
            summary=(
                f"A genesis-sealed pixel_capture record is available for shard {shard_id} "
                f"(capture method: {capture_method}; source label: {source_label}). "
                f"This is a rendered-surface screenshot record: {', '.join(_NON_ASSERTIONS)}."
            ),
            input_refs=list(refs),
            claims=[f"shard {shard_id} carries a verified pixel_capture record"],
            provenance=ProvenanceState.PROVEN,  # the SEAL is verified; not a claim about page content
        )

        rendered = AttentionFinding(
            tension_type="rendered_surface_capture",
            score=1.0,
            summary=(
                "The evidence is the rendered surface the user visually captured "
                "(pixel_capture tier): rendered pixels only. It records what was on "
                f"screen, {', '.join(_NON_ASSERTIONS)}."
            ),
            input_refs=list(refs),
            claims=["evidence tier is pixel_capture: rendered surface only"],
            provenance=ProvenanceState.PROVEN,
        )

        review = AttentionFinding(
            tension_type="human_review_required",
            score=1.0,
            summary=(
                f"Pixel evidence for shard {shard_id} requires human review. GhostBox "
                "does not adjudicate the screenshot's meaning; it is bounded pixel "
                "evidence, not machine-established truth."
            ),
            input_refs=list(refs),
            claims=["a human must review this pixel evidence before any downstream claim"],
            provenance=ProvenanceState.UNTESTED,  # an attention flag, not a proven fact
        )

        findings = [available, rendered, review]

        # optional: capture context, ONLY if sidecar fields are present. It is
        # user/tool-supplied context, explicitly NOT verified as accurate.
        present = {k: manifest.get(k) for k in _CONTEXT_KEYS if manifest.get(k)}
        if present:
            desc = ", ".join(f"{k}={v!r}" for k, v in sorted(present.items()))
            findings.append(
                AttentionFinding(
                    tension_type="capture_context_available",
                    score=0.5,
                    summary=(
                        f"User/tool-supplied capture context is present for shard {shard_id} "
                        f"({desc}). This context is supplied, NOT verified as accurate: "
                        f"{', '.join(_NON_ASSERTIONS)}."
                    ),
                    input_refs=list(refs),
                    claims=[f"supplied capture context present: {sorted(present)}"],
                    provenance=ProvenanceState.UNTESTED,  # supplied, not verified
                )
            )
        return findings

    def _untrusted(
        self, shard_id: str, kind: str, summary: str, *, extra_refs: Optional[List[str]] = None
    ) -> AttentionFinding:
        f = AttentionFinding(
            tension_type=kind,
            score=1.0,
            summary=summary,
            input_refs=[shard_id] + list(extra_refs or []),
            claims=[f"shard {shard_id} pixel observation blocked: {kind}"],
            provenance=ProvenanceState.UNTESTED,
        )
        self._findings.append(f)
        return f

    # -- read the *verified* sealed shard (read-only) ------------------------

    def _read_pixel_manifest(self, shard_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        path = shard_dir / CONTENT_DIR / MANIFEST_NAME
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, None
        return manifest, hashlib.sha256(raw).hexdigest()

    def _sealed_png_sha256(self, shard_dir: Path, png_filename: str) -> Optional[str]:
        """Hash the sealed PNG bytes once. Read-only: the PNG is never decoded,
        re-encoded, rewritten, or retained -- only hashed for the match check."""
        path = shard_dir / CONTENT_DIR / png_filename
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _result(
        self, sealed, status, tier, image_sha256, findings, *, trusted, manifest_sha256=None
    ) -> PixelObservationResult:
        return PixelObservationResult(
            shard_id=sealed.shard_id, verified=True, status=status, trusted=trusted,
            evidence_tier=tier, image_sha256=image_sha256, manifest_sha256=manifest_sha256,
            findings=tuple(findings),
        )


def _img_ref(image_sha256: Optional[str]) -> str:
    return f"img:sha256:{image_sha256}"


def _manifest_ref(manifest_sha256: Optional[str]) -> str:
    return f"manifest:sha256:{manifest_sha256}"
