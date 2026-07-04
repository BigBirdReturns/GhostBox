"""GhostBox <-> axm-genesis custody edge: a read-only spoke adapter.

This is the level-2 consumer side of the trust seam. GhostBox points at a
genesis-sealed shard, verifies it *through genesis* (never re-implementing the
kernel, never parsing the seal itself), records the custody outcome in its own
reviewable ledger, and may add attention state -- without ever owning,
mutating, re-emitting, or re-minting the sealed record.

The seam this conforms to is the corrected genesis boundary in
``contracts.py``: ``SealedShard`` (the seal is the on-disk shard directory, not
a GhostBox-minted receipt), ``VerifyStatus`` (PASS / FAIL / MALFORMED /
NO_TRUSTED_KEY, kept at the boundary, never flattened to a bool), and
``TrustKernel.verify(shard, *, trusted_key) -> VerifyStatus``. This adapter now
*conforms to* that contract rather than routing around it.

Boundary rules enforced here:

  - Kernel untouched. Verification is delegated to a genesis-owned verifier
    handed in at construction (dependency injection). GhostBox never signs,
    seals, re-implements crypto, or opens the shard's manifest/sig itself.
    ``seal()`` is refused.
  - Out-of-band trust. The trusted publisher key is supplied out of band at
    construction, never taken from the shard's own ``sig/publisher.pub``. With
    no trusted key, verification refuses (``NO_TRUSTED_KEY``) and never yields a
    PASS -- the same anti-pattern the axm-chat/axm-core perimeter sweep (F1,
    HIGH) closed.
  - Fail closed. A verify result that is not PASS (failed OR malformed OR
    no-key) emits an UNVERIFIED ``AttentionFinding``, records the custody entry
    as unverified, and blocks trust. No retry, no quarantine-and-proceed, no
    log-and-continue. A failed verification is a result to record, not an error
    to route around.
  - No write-back. The ``SealedShard`` reference is immutable input (a frozen
    dataclass). GhostBox persists only the genesis-assigned ``shard_id``
    (a reference, carried verbatim) plus its own outcome / attention state --
    never the shard body, never a sealed object re-emitted with GhostBox fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .contracts import (
    AttentionFinding,
    ProvenanceState,
    SealedShard,
    VerifyStatus,
    now_utc,
)


class CustodyOutcome(str, Enum):
    """What GhostBox recorded about a shard it was handed."""

    VERIFIED = "verified"      # PASS: genesis-sealed, trusted, FROZEN
    UNVERIFIED = "unverified"  # everything else: blocked, never trusted


# The genesis-owned verifier. Signature mirrors the frozen CLI contract:
#   axm-verify shard <shard_dir> --trusted-key <oob_pubkey>  -> exit 0 | 1 | 2
# We hand it the shard directory and the out-of-band trusted key and expect the
# frozen integer exit code back. GhostBox interprets no crypto and opens no
# manifest; it maps the frozen code to a custody decision. In tests this is a
# fake; in production it is a thin wrapper over ``axm_verify.cli.verify_shard``.
GenesisVerifier = Callable[[str, str], int]


@dataclass(frozen=True)
class CustodyLedgerEntry:
    """A reviewable record of one custody decision. GhostBox-owned.

    Holds only the genesis-assigned ``shard_id`` (a reference, carried verbatim)
    plus the verification outcome, the id of the out-of-band key that anchored
    it, and -- when unverified -- the finding that recorded it. It deliberately
    holds NO field from the sealed shard body (no suite, merkle root, or
    signature): that stays in genesis's record. This is the "no write-back"
    invariant made structural -- there is no field here that could carry the
    body even by mistake.
    """

    shard_id: str
    outcome: CustodyOutcome
    status: VerifyStatus
    trusted_key_id: Optional[str]
    decided_at: str
    finding_id: Optional[str] = None


class GenesisTrustKernel:
    """Read-only adapter over the genesis verifier. Conforms to ``TrustKernel``.

    Implements the verify half of the role and refuses the seal half: only
    axm-genesis seals. The out-of-band trusted key is injected at construction;
    without it, verification returns ``NO_TRUSTED_KEY`` and never falls back to
    the shard's own embedded key to manufacture trust.
    """

    def __init__(
        self,
        verifier: GenesisVerifier,
        *,
        trusted_key: Optional[str] = None,
        trusted_key_id: Optional[str] = None,
    ) -> None:
        self._verify_shard = verifier
        self._trusted_key = trusted_key
        self._trusted_key_id = trusted_key_id or trusted_key

    @property
    def trusted_key_id(self) -> Optional[str]:
        return self._trusted_key_id

    def seal(self, shard_dir: str) -> "SealedShard":
        raise PermissionError(
            "GhostBox is the attention layer, not the trust kernel: it never "
            "seals. Sealing authority stays in axm-genesis."
        )

    def verify(
        self, shard: SealedShard, *, trusted_key: Optional[str] = None
    ) -> VerifyStatus:
        """Ask genesis whether this sealed shard verifies. One call, no retry.

        ``trusted_key`` defaults to the out-of-band anchor injected at
        construction. With no anchor from either source, we return
        ``NO_TRUSTED_KEY`` and never call genesis -- the anchor is never taken
        from inside the shard (that is the F1 vacuous-trust bug).
        """
        key = trusted_key if trusted_key is not None else self._trusted_key
        if not key:
            return VerifyStatus.NO_TRUSTED_KEY
        code = self._verify_shard(shard.shard_dir, key)
        if code == 0:
            return VerifyStatus.PASS
        if code == 2:
            return VerifyStatus.MALFORMED
        return VerifyStatus.FAIL  # exit 1 (or any other nonzero) -> failed, not trusted


class GenesisCustodySpoke:
    """GhostBox's custody edge onto genesis.

    ``ingest_sealed_shard`` points at a genesis-sealed shard, verifies it
    through the kernel adapter, and records the decision. A PASS marks the shard
    trusted (by pointer only) and available to the attention layer as a FROZEN
    input. Anything else emits an unverified finding and blocks trust.
    """

    def __init__(self, kernel: GenesisTrustKernel) -> None:
        self._kernel = kernel
        self._ledger: list[CustodyLedgerEntry] = []
        self._findings: list[AttentionFinding] = []
        # Pointers only. GhostBox records *that* a shard verified, by its
        # genesis-assigned id -- never a copy of the shard.
        self._trusted_shard_ids: set[str] = set()

    @property
    def ledger(self) -> tuple[CustodyLedgerEntry, ...]:
        return tuple(self._ledger)

    @property
    def findings(self) -> tuple[AttentionFinding, ...]:
        return tuple(self._findings)

    def is_trusted(self, shard_id: str) -> bool:
        """Has this shard been verified through genesis in this session?"""
        return shard_id in self._trusted_shard_ids

    def ingest_sealed_shard(self, shard: SealedShard) -> CustodyLedgerEntry:
        status = self._kernel.verify(shard)
        decided_at = now_utc()

        if status is VerifyStatus.PASS:
            # Genesis says sealed and trusted. Record a pointer; copy nothing.
            self._trusted_shard_ids.add(shard.shard_id)
            entry = CustodyLedgerEntry(
                shard_id=shard.shard_id,  # genesis-assigned, verbatim
                outcome=CustodyOutcome.VERIFIED,
                status=status,
                trusted_key_id=self._kernel.trusted_key_id,
                decided_at=decided_at,
            )
            self._ledger.append(entry)
            return entry

        # Fail closed. Not PASS -> unverified. Surface a finding, block trust,
        # and do NOT retry, quarantine-and-proceed, or treat it as sealed.
        finding = AttentionFinding(
            tension_type="unverified_seal",
            score=1.0,
            summary=(
                f"Genesis verification did not return PASS for shard "
                f"{shard.shard_id} (status={status.value}). The shard is NOT "
                f"treated as sealed and is blocked from trusted use."
            ),
            input_refs=[shard.shard_id],  # verbatim; no re-mint
            claims=[f"shard {shard.shard_id} is genesis-sealed and verifiable"],
            provenance=ProvenanceState.UNTESTED,
            created_at=decided_at,
        )
        self._findings.append(finding)
        entry = CustodyLedgerEntry(
            shard_id=shard.shard_id,
            outcome=CustodyOutcome.UNVERIFIED,
            status=status,
            trusted_key_id=self._kernel.trusted_key_id,
            decided_at=decided_at,
            finding_id=finding.finding_id,
        )
        self._ledger.append(entry)
        return entry
