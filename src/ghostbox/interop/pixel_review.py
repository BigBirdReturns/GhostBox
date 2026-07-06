"""GhostBox Pixel Review Layer v0 -- HUMAN review, strictly downstream of the
pixel observer.

The observer (``pixel_observer.py``) already emitted ``human_review_required``
over every trusted pixel_capture observation. This layer is where that review
happens: a human looks at the evidence and records an ATTENTION DISPOSITION.
GhostBox records the human's decision as its own review state; it decides
nothing itself.

Placement in the chain (fixed):

    SealedShard -> GenesisCustodySpoke (custody) -> PixelEvidenceObserver
    (tier + hash discipline) -> THIS LAYER (human review disposition)

Enforced boundaries:
  - Strictly downstream of the observer: the only input is a
    ``PixelObservationResult``. This module performs NO filesystem access at
    all -- no shard read, no manifest read, no PNG read. It cannot touch the
    image even by accident (asserted at source level: no Path/open/read).
  - Only trusted observations are reviewable: ``verified`` and ``trusted`` must
    both be True and the tier must be ``pixel_capture`` (re-checked here,
    defense in depth). Anything else is refused, never queued.
  - The evidence tier is IMMUTABLE. A review never upgrades pixel evidence to
    DOM truth, API truth, platform authenticity, or legal provenance. The
    record carries the tier verbatim and there is no parameter to change it.
  - The disposition vocabulary is attention-only (escalate / dismiss /
    needs_context). Truth-shaped verdicts ("authentic", "true", ...) are not
    representable and are refused if attempted.
  - The decision is HUMAN-attributed: a non-empty reviewer identity is
    required, and the note is recorded as reviewer-attributed text, never as a
    GhostBox assertion. Review records are ``UNTESTED`` provenance -- a human
    opinion on file, not a proven fact.
  - Review history is append-only: re-review adds a record, it never edits or
    replaces one.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .contracts import ProvenanceState, content_id, now_utc
from .pixel_observer import PIXEL_TIER, PixelObservationResult, _NON_ASSERTIONS


class ReviewDisposition(str, Enum):
    """What the human reviewer wants the attention layer to do next.

    Deliberately attention-shaped, never truth-shaped: there is no
    "authentic", no "true", no "verified_content" -- a review moves attention,
    it does not adjudicate the screenshot's meaning or upgrade its tier.
    """

    ESCALATE = "escalate"            # a human wants more eyes / downstream action
    DISMISS = "dismiss"              # a human saw it and stands down the attention
    NEEDS_CONTEXT = "needs_context"  # a human cannot decide without more supplied context


class ReviewRefused(RuntimeError):
    """The review layer refused. ``reason`` is a stable machine-readable tag."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PixelReviewRecord:
    """One human review decision over one trusted pixel observation.

    GhostBox-owned state, downstream refs only: the genesis ``shard_id``, the
    external image/manifest hashes, and the ``human_review_required`` finding it
    answers -- all carried verbatim. No PNG bytes, no manifest body, no custody
    material. ``evidence_tier`` is copied verbatim from the observation and the
    dataclass is frozen: nothing here can restate the evidence as anything more
    than a pixel capture.
    """

    review_id: str
    shard_id: str                     # genesis-assigned, verbatim
    image_sha256: str                 # external ref, verbatim
    manifest_sha256: str              # external ref, verbatim
    answers_finding_id: str           # the human_review_required finding, verbatim
    evidence_tier: str                # verbatim from the observation; NEVER upgraded
    disposition: ReviewDisposition
    reviewer: str                     # human attribution, required
    note: str                         # reviewer-attributed text, NOT a GhostBox assertion
    reviewed_at: str
    non_assertions: Tuple[str, ...] = _NON_ASSERTIONS
    provenance: ProvenanceState = ProvenanceState.UNTESTED  # a human opinion on file


@dataclass(frozen=True)
class _QueuedItem:
    """Internal: one trusted observation awaiting (or past) its first review."""

    shard_id: str
    image_sha256: str
    manifest_sha256: str
    evidence_tier: str
    review_finding_id: str


class PixelReviewLayer:
    """Queue trusted pixel observations for human review; record the decisions.

    ``submit`` admits only trusted observations. ``review`` records a
    human-attributed disposition and resolves the pending item; further reviews
    of the same shard append to the history, never rewrite it.
    """

    def __init__(self) -> None:
        self._items: Dict[str, _QueuedItem] = {}   # everything ever admitted
        self._pending: List[str] = []              # shard_ids awaiting first review
        self._records: List[PixelReviewRecord] = []

    # -- intake ---------------------------------------------------------------

    def submit(self, observation: PixelObservationResult) -> str:
        """Admit a trusted pixel observation for human review.

        Refuses (never queues) anything the observer did not mark trusted, and
        re-checks the tier here -- defense in depth, not trust in the caller.
        Returns the shard_id now pending review.
        """
        if not observation.verified:
            raise ReviewRefused(
                f"shard {observation.shard_id} was not custody-verified "
                f"(status={observation.status.value}); nothing to review.",
                reason="not_verified",
            )
        if not observation.trusted:
            raise ReviewRefused(
                f"shard {observation.shard_id} is not a trusted pixel observation; "
                f"untrusted evidence is not reviewable as pixel evidence.",
                reason="not_trusted",
            )
        if observation.evidence_tier != PIXEL_TIER:
            raise ReviewRefused(
                f"shard {observation.shard_id} tier is {observation.evidence_tier!r}, "
                f"not {PIXEL_TIER!r}; this layer reviews pixel captures only.",
                reason="wrong_tier",
            )
        review_finding = next(
            (f for f in observation.findings if f.tension_type == "human_review_required"),
            None,
        )
        if review_finding is None:
            raise ReviewRefused(
                f"shard {observation.shard_id} carries no human_review_required "
                f"finding; the observer did not ask for review.",
                reason="no_review_finding",
            )
        item = _QueuedItem(
            shard_id=observation.shard_id,
            image_sha256=observation.image_sha256 or "",
            manifest_sha256=observation.manifest_sha256 or "",
            evidence_tier=observation.evidence_tier,
            review_finding_id=review_finding.finding_id,
        )
        if observation.shard_id not in self._items:
            self._items[observation.shard_id] = item
            self._pending.append(observation.shard_id)
        return observation.shard_id

    # -- the human decision ---------------------------------------------------

    def review(
        self,
        shard_id: str,
        *,
        reviewer: str,
        disposition: "ReviewDisposition | str",
        note: str = "",
        reviewed_at: Optional[str] = None,
    ) -> PixelReviewRecord:
        """Record one human review decision. Append-only; tier immutable."""
        item = self._items.get(shard_id)
        if item is None:
            raise ReviewRefused(
                f"shard {shard_id} was never admitted for review; submit a trusted "
                f"observation first.",
                reason="not_submitted",
            )
        if not (isinstance(reviewer, str) and reviewer.strip()):
            raise ReviewRefused(
                "a review must be attributed to a human reviewer; empty reviewer refused.",
                reason="no_reviewer",
            )
        try:
            disp = ReviewDisposition(disposition)
        except ValueError:
            raise ReviewRefused(
                f"disposition {disposition!r} is not an attention disposition "
                f"({[d.value for d in ReviewDisposition]}); truth-shaped verdicts "
                f"are not representable here.",
                reason="invalid_disposition",
            )
        when = reviewed_at or now_utc()
        record = PixelReviewRecord(
            review_id=content_id(
                "rev",
                {
                    "shard_id": item.shard_id,
                    "image_sha256": item.image_sha256,
                    "answers_finding_id": item.review_finding_id,
                    "disposition": disp.value,
                    "reviewer": reviewer,
                    "note": note,
                    "reviewed_at": when,
                },
            ),
            shard_id=item.shard_id,
            image_sha256=item.image_sha256,
            manifest_sha256=item.manifest_sha256,
            answers_finding_id=item.review_finding_id,
            evidence_tier=item.evidence_tier,  # verbatim; no parameter can change it
            disposition=disp,
            reviewer=reviewer,
            note=note,
            reviewed_at=when,
        )
        self._records.append(record)
        if shard_id in self._pending:
            self._pending.remove(shard_id)
        return record

    # -- GhostBox-owned state, read-only views --------------------------------

    @property
    def pending(self) -> Tuple[str, ...]:
        """shard_ids admitted but not yet reviewed by any human."""
        return tuple(self._pending)

    @property
    def records(self) -> Tuple[PixelReviewRecord, ...]:
        """Append-only review history."""
        return tuple(self._records)

    def records_for(self, shard_id: str) -> Tuple[PixelReviewRecord, ...]:
        return tuple(r for r in self._records if r.shard_id == shard_id)
