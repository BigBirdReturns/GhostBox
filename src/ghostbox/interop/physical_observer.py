"""GhostBox observer over axm-embodied physical-capture shards.

axm-embodied seals an event-triggered frame-capture capsule (``frames.bin`` +
``capture_manifest.json`` + the capture log, with a judge-verified byte-level
stream index published as ``ext/streams@1.jsonl``) into a genesis
``axm-hybrid1`` shard (proven live: a real recorder session was driven and
sealed on this branch, then verified detached). This observer lets GhostBox add
attention findings over that physical evidence WITHOUT becoming its owner. The
order is fixed and non-negotiable, and mirrors the landed pixel observer:

    SealedShard -> custody verification (through the LANDED GenesisCustodySpoke)
    -> confirm evidence tier is ``physical_capture`` -> confirm the sealed
    stream index agrees with the declared kept-frame count -> GhostBox-owned
    AttentionFindings, strictly downstream of the verified ``shard_id``.

Enforced boundaries:
  - Reuses the landed custody seam for verification. It does NOT call
    ``axm-verify`` and does NOT duplicate genesis logic -- ``GenesisCustodySpoke``
    already owns that path.
  - Observation happens ONLY after a VERIFIED custody outcome. FAIL / MALFORMED /
    NO_TRUSTED_KEY yield no physical findings (the custody spoke has already
    recorded the failure); nothing is read until custody returns VERIFIED.
  - Frames are never decoded, filtered, classified, or interpreted: the observer
    never opens ``frames.bin`` at all. It reads only the sealed capture manifest
    and the sealed stream index, and it never recomputes the continuity chain --
    that is the recorder-side judge's job before sealing, and re-deriving it
    here would duplicate spoke logic as a second authority.
  - No axm-embodied import: GhostBox consumes the sealed shard shape, never
    axm-embodied as an authority.
  - Findings are bounded to the real declared tier limits: they never assert
    identity, activity or semantic classification, continuous coverage,
    platform truth, or legal-grade provenance. Custody identity stays the
    genesis ``shard_id``; GhostBox stores only its own finding state and
    external refs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import AttentionFinding, ProvenanceState, VerifyStatus
from .genesis_spoke import CustodyOutcome, GenesisCustodySpoke

PHYSICAL_TIER = "physical_capture"
MANIFEST_NAME = "capture_manifest.json"
CONTENT_DIR = "content"
STREAM_INDEX = "ext/streams@1.jsonl"

# What GhostBox will NEVER assert about a physical capture. Reconciled verbatim
# from the real sealed capture manifest's evidence_tier_limits. Shipped in the
# finding so a downstream reader cannot mistake attention for adjudication.
_NON_ASSERTIONS = (
    "not identity",
    "not activity or semantic classification",
    "not continuous coverage (gaps between windows are declared, not hidden)",
    "not platform truth",
    "not legal-grade provenance by itself",
)


@dataclass(frozen=True)
class PhysicalObservationResult:
    """Outcome of observing one physical-capture shard. GhostBox-owned.

    Holds only the genesis-assigned ``shard_id`` (a reference, verbatim), the
    custody verdict, the external facts it checked (tier, declared kept-frame
    count, the continuity anchor), and GhostBox's own findings -- no frame
    bytes, no manifest body, no signature, no custody material.
    """

    shard_id: str
    verified: bool                     # custody returned PASS
    status: VerifyStatus
    trusted: bool                      # verified AND physical_capture tier AND index agrees
    evidence_tier: Optional[str]
    frames_kept: Optional[int]
    chain_genesis: Optional[str]
    manifest_sha256: Optional[str]
    findings: Tuple[AttentionFinding, ...]


class PhysicalEvidenceObserver:
    """Observe verified axm-embodied physical-capture shards; annotate, never own."""

    def __init__(self, custody: GenesisCustodySpoke) -> None:
        self._custody = custody
        self._findings: List[AttentionFinding] = []

    @property
    def findings(self) -> Tuple[AttentionFinding, ...]:
        return tuple(self._findings)

    def observe(self, sealed) -> PhysicalObservationResult:
        # 1) Verify the sealed shard THROUGH the landed custody seam. No direct
        #    axm-verify, no duplicated genesis logic -- custody owns verification.
        entry = self._custody.ingest_sealed_shard(sealed)

        # 2) Fail closed: only a VERIFIED custody outcome unlocks observation.
        if entry.outcome is not CustodyOutcome.VERIFIED:
            return PhysicalObservationResult(
                shard_id=sealed.shard_id, verified=False, status=entry.status,
                trusted=False, evidence_tier=None, frames_kept=None,
                chain_genesis=None, manifest_sha256=None, findings=(),
            )

        # 3) Only now -- after verification -- read the sealed capture manifest.
        shard_dir = Path(sealed.shard_dir)
        manifest, manifest_sha256 = self._read_capture_manifest(shard_dir)
        if manifest is None:
            f = self._untrusted(
                sealed.shard_id, "not_physical_evidence",
                f"Verified shard {sealed.shard_id} has no readable "
                f"{MANIFEST_NAME}; not observed as physical evidence.",
            )
            return self._result(sealed, entry.status, None, None, None, None, [f])

        tier = manifest.get("evidence_tier")
        frames_kept = manifest.get("frames_kept")
        chain_genesis = manifest.get("chain_genesis")

        # 4) Confirm the evidence tier. A non-physical_capture tier emits ONLY
        #    an untrusted finding -- never the trusted physical findings.
        if tier != PHYSICAL_TIER:
            f = self._untrusted(
                sealed.shard_id, "unexpected_evidence_tier",
                f"Shard {sealed.shard_id} capture manifest tier is {tier!r}, not "
                f"{PHYSICAL_TIER!r}; blocked from trusted physical observation.",
            )
            return self._result(sealed, entry.status, tier, frames_kept, chain_genesis,
                                manifest_sha256, [f])

        # 5) Confirm the sealed judge-verified stream index agrees with the
        #    declared kept-frame count. The real compiler always publishes it;
        #    a physical_capture shard without a consistent index is the
        #    log/disk-disagreement class the recorder-side judge treats as
        #    FATAL, so GhostBox never trusts past it.
        index_rows = self._read_stream_index(shard_dir)
        if index_rows is None:
            f = self._untrusted(
                sealed.shard_id, "physical_stream_index_missing",
                f"Shard {sealed.shard_id} declares {PHYSICAL_TIER!r} but carries no "
                f"readable {STREAM_INDEX}; blocked from trusted physical observation.",
            )
            return self._result(sealed, entry.status, tier, frames_kept, chain_genesis,
                                manifest_sha256, [f])
        frame_rows = [r for r in index_rows if r.get("stream") == "frames"]
        unverified = [r for r in frame_rows if r.get("status") != "VERIFIED"]
        if unverified or frames_kept is None or len(frame_rows) != int(frames_kept):
            f = self._untrusted(
                sealed.shard_id, "physical_stream_index_mismatch",
                f"Shard {sealed.shard_id} declares frames_kept={frames_kept!r} but its "
                f"sealed stream index carries {len(frame_rows)} frame rows "
                f"({len(unverified)} not VERIFIED); blocked from trusted physical "
                f"observation.",
            )
            return self._result(sealed, entry.status, tier, frames_kept, chain_genesis,
                                manifest_sha256, [f])

        # 6) Verified physical_capture with a consistent index -> bounded
        #    findings, strictly downstream of the verified shard_id.
        findings = self._surface_physical_findings(
            sealed.shard_id, manifest, chain_genesis, manifest_sha256, len(frame_rows)
        )
        self._findings.extend(findings)
        return self._result(sealed, entry.status, tier, frames_kept, chain_genesis,
                            manifest_sha256, findings, trusted=True)

    # -- finding construction (bounded, downstream of shard_id) --------------

    def _surface_physical_findings(
        self,
        shard_id: str,
        manifest: Dict[str, Any],
        chain_genesis: Optional[str],
        manifest_sha256: Optional[str],
        kept_rows: int,
    ) -> List[AttentionFinding]:
        sensor_id = manifest.get("sensor_id")
        session_id = manifest.get("session_id")
        frames_observed = manifest.get("frames_observed")
        triggers = manifest.get("triggers")
        # External refs, carried verbatim. Never a minted id.
        refs = [shard_id]
        if chain_genesis:
            refs.append(_chain_ref(chain_genesis))
        if manifest_sha256:
            refs.append(_manifest_ref(manifest_sha256))

        available = AttentionFinding(
            tension_type="physical_evidence_available",
            score=1.0,
            summary=(
                f"A genesis-sealed physical_capture record is available for shard "
                f"{shard_id} (sensor: {sensor_id}; session: {session_id}). This is an "
                f"event-triggered opaque-sensor-bytes record: {', '.join(_NON_ASSERTIONS)}."
            ),
            input_refs=list(refs),
            claims=[f"shard {shard_id} carries a verified physical_capture record"],
            provenance=ProvenanceState.PROVEN,  # the SEAL is verified; not a claim about frame content
        )

        opaque = AttentionFinding(
            tension_type="opaque_sensor_capture",
            score=1.0,
            summary=(
                "The evidence is opaque sensor bytes within declared trigger windows "
                "(physical_capture tier): never decoded, filtered, classified, or "
                f"interpreted. It records what the sensor emitted, {', '.join(_NON_ASSERTIONS)}."
            ),
            input_refs=list(refs),
            claims=["evidence tier is physical_capture: opaque sensor bytes in declared windows only"],
            provenance=ProvenanceState.PROVEN,
        )

        continuity = AttentionFinding(
            tension_type="capture_continuity_available",
            score=1.0,
            summary=(
                f"The sealed judge-verified stream index for shard {shard_id} carries "
                f"{kept_rows} kept frames (declared: {frames_observed} observed, "
                f"{triggers} triggers), continuity-chained from {_chain_ref(chain_genesis)}. "
                f"Gaps between capture windows are declared, not hidden; a chain break "
                f"never seals."
            ),
            input_refs=list(refs),
            claims=[f"shard {shard_id} carries a sealed continuity-chained frame index"],
            provenance=ProvenanceState.PROVEN,
        )

        review = AttentionFinding(
            tension_type="human_review_required",
            score=1.0,
            summary=(
                f"Physical evidence for shard {shard_id} requires human review. GhostBox "
                "does not adjudicate what the frames show; it is bounded physical "
                "evidence, not machine-established truth."
            ),
            input_refs=list(refs),
            claims=["a human must review this physical evidence before any downstream claim"],
            provenance=ProvenanceState.UNTESTED,  # an attention flag, not a proven fact
        )

        return [available, opaque, continuity, review]

    def _untrusted(self, shard_id: str, kind: str, summary: str) -> AttentionFinding:
        f = AttentionFinding(
            tension_type=kind,
            score=1.0,
            summary=summary,
            input_refs=[shard_id],
            claims=[f"shard {shard_id} physical observation blocked: {kind}"],
            provenance=ProvenanceState.UNTESTED,
        )
        self._findings.append(f)
        return f

    # -- read the *verified* sealed shard (read-only) ------------------------

    def _read_capture_manifest(self, shard_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        path = shard_dir / CONTENT_DIR / MANIFEST_NAME
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, None
        if not isinstance(manifest, dict):
            return None, None
        return manifest, hashlib.sha256(raw).hexdigest()

    def _read_stream_index(self, shard_dir: Path) -> Optional[List[Dict[str, Any]]]:
        path = shard_dir / STREAM_INDEX
        if not path.exists():
            return None
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None
        return rows

    def _result(
        self, sealed, status, tier, frames_kept, chain_genesis, manifest_sha256,
        findings, *, trusted: bool = False,
    ) -> PhysicalObservationResult:
        return PhysicalObservationResult(
            shard_id=sealed.shard_id, verified=True, status=status, trusted=trusted,
            evidence_tier=tier, frames_kept=frames_kept, chain_genesis=chain_genesis,
            manifest_sha256=manifest_sha256, findings=tuple(findings),
        )


def _chain_ref(chain_genesis: Optional[str]) -> str:
    return f"chain:sha256:{chain_genesis}"


def _manifest_ref(manifest_sha256: Optional[str]) -> str:
    return f"manifest:sha256:{manifest_sha256}"
