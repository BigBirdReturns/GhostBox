"""GhostBox level-2 attention over a genesis-verified record.

The seam's whole point: findings are *downstream of verifiable evidence*.
This runs only after custody has verified the shard, and it observes the
record's public content to emit one ``AttentionFinding``. It:

  - reads only ``content/source.txt`` (the observable document, byte-addressable
    per the genesis spec) -- never the manifest, signature, or graph internals,
    so GhostBox does not parse the seal;
  - references the record by its genesis-assigned ``shard_id`` (verbatim);
  - marks the finding ``UNTESTED`` -- it surfaces content for downstream
    claim-testing, it does not assert the content is true;
  - writes nothing, seals nothing, rewrites no custody material.
"""
from __future__ import annotations

from pathlib import Path

from ghostbox.interop.contracts import AttentionFinding, ProvenanceState, SealedShard


def surface_finding(sealed: SealedShard) -> AttentionFinding:
    """Observe a verified record and surface it as attention state."""
    source = Path(sealed.shard_dir) / "content" / "source.txt"
    excerpt = ""
    if source.exists():
        excerpt = source.read_text(encoding="utf-8")[:280].strip()

    return AttentionFinding(
        tension_type="surfaced_record",
        score=0.5,
        summary=(
            f"Verified genesis record {sealed.shard_id} entered attention. "
            f"Surface content is available for downstream claim-testing; "
            f"attention asserts nothing about whether the content is true."
        ),
        input_refs=[sealed.shard_id],  # genesis-assigned id, verbatim
        claims=[excerpt] if excerpt else [],
        provenance=ProvenanceState.UNTESTED,
    )
