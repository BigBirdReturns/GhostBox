"""The reviewable packet: what a human inspects instead of trusting a platform.

The packet shows the record identity, the signature context, where trust was
anchored, the verification status, GhostBox's finding, the enforced boundary,
and the exit-test result -- plus the genesis probe receipts carried forward.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ghostbox.interop.contracts import AttentionFinding, SealedShard
from ghostbox.interop.genesis_spoke import CustodyLedgerEntry

# Genesis live-probe receipts (see docs/GENESIS_EDGE_MISMATCHES.md and the probe
# ledger). Carried forward honestly: functional now, not load-proven.
PROBE_RECEIPTS = {
    "gold_shard_verify": "PASS (exit 0)",
    "fresh_disposable_shard_verify": "PASS (exit 0)",
    "wrong_out_of_band_key": "FAIL / E_SIG_INVALID (exit 1)",
    "missing_trusted_key": "refused (exit 2, never PASS)",
    "detached_verification": "PASS (shard bytes + out-of-band pub only)",
    "crypto_backend": "dilithium-py fallback active; liboqs (oqs) absent",
    "backend_caveat": "functional in this environment; NOT load-proven",
}

BOUNDARY_NOTES = [
    "GhostBox did not seal (seal() is refused; sealing authority stays in axm-genesis).",
    "GhostBox delegated verification to the real axm-verify CLI; it never opened the manifest or signature.",
    "Trust was anchored to an out-of-band key; the shard's embedded sig/publisher.pub was never used as authority.",
    "Missing trusted key is GhostBox-owned NO_TRUSTED_KEY, decided before invoking the CLI "
    "(exit 2 is ambiguous between a usage error and a malformed shard).",
    "GhostBox retained only the genesis-assigned shard_id; no shard body, no signature, no receipt object.",
    "The AttentionFinding is downstream of verification (UNTESTED); it does not assert the content is true "
    "and does not rewrite custody.",
]


def build_packet(
    *,
    sealed: SealedShard,
    trusted_key_source: str,
    verify_status: str,
    genesis_result: Optional[dict],
    finding: AttentionFinding,
    custody_entry: CustodyLedgerEntry,
    exit_test: Optional[dict] = None,
) -> Dict[str, Any]:
    return {
        "artifact": "AXM Sovereign Spine v0",
        "claim": "Custody survives tool removal: seal, verify outside the AI, annotate, export, remove.",
        "shard_identity": {
            "shard_id": sealed.shard_id,
            "note": "genesis-assigned (sh1_ = BLAKE3 of the manifest); carried verbatim, never minted by GhostBox",
        },
        "signature_context": {
            "suite": sealed.suite,
            "signature_algorithm": "Ed25519 || ML-DSA-44 (both must verify)",
            "merkle_root": sealed.merkle_root,
            "sealed_at": sealed.sealed_at,
            "authority_note": "authority is the out-of-band trusted key, NOT the shard's embedded sig/publisher.pub",
        },
        "trusted_key_source": {
            "source": trusted_key_source,
            "kind": "out-of-band",
        },
        "verification": {
            "status": verify_status,
            "genesis_result": genesis_result,
            "verifier": "axm-verify shard <dir> --trusted-key <oob_pub> (real kernel)",
        },
        "custody": {
            "outcome": custody_entry.outcome.value,
            "status": custody_entry.status.value,
            "trusted_key_id": custody_entry.trusted_key_id,
            "shard_id": custody_entry.shard_id,
            "retained_by_ghostbox": "shard_id reference only",
        },
        "ghostbox_finding": {
            "finding_id": finding.finding_id,
            "tension_type": finding.tension_type,
            "score": finding.score,
            "summary": finding.summary,
            "input_refs": finding.input_refs,
            "claims": finding.claims,
            "provenance": finding.provenance.value,
        },
        "boundary_notes": BOUNDARY_NOTES,
        "exit_test": exit_test,
        "genesis_probe_receipts": PROBE_RECEIPTS,
    }


def render_markdown(packet: Dict[str, Any]) -> str:
    p = packet
    lines = [
        f"# {p['artifact']} — reviewable packet",
        "",
        f"**Product claim:** {p['claim']}",
        "",
        "## Record identity",
        f"- `shard_id`: `{p['shard_identity']['shard_id']}`",
        f"- {p['shard_identity']['note']}",
        "",
        "## Signature context",
        f"- suite: `{p['signature_context']['suite']}` — {p['signature_context']['signature_algorithm']}",
        f"- merkle_root: `{p['signature_context']['merkle_root']}`",
        f"- sealed_at: `{p['signature_context']['sealed_at']}`",
        f"- {p['signature_context']['authority_note']}",
        "",
        "## Trust anchor",
        f"- source: `{p['trusted_key_source']['source']}` ({p['trusted_key_source']['kind']})",
        "",
        "## Verification",
        f"- status: **{p['verification']['status']}**",
        f"- verifier: `{p['verification']['verifier']}`",
        f"- genesis said: `{json.dumps(p['verification']['genesis_result'])}`",
        "",
        "## Custody (GhostBox)",
        f"- outcome: **{p['custody']['outcome']}** (status `{p['custody']['status']}`)",
        f"- trusted_key_id: `{p['custody']['trusted_key_id']}`",
        f"- retained by GhostBox: {p['custody']['retained_by_ghostbox']}",
        "",
        "## GhostBox attention finding",
        f"- `{p['ghostbox_finding']['finding_id']}` — {p['ghostbox_finding']['tension_type']} "
        f"(provenance: {p['ghostbox_finding']['provenance']})",
        f"- {p['ghostbox_finding']['summary']}",
        f"- input_refs: {p['ghostbox_finding']['input_refs']}",
        "",
        "## Boundary (what GhostBox did NOT do)",
    ]
    lines += [f"- {n}" for n in p["boundary_notes"]]
    if p.get("exit_test") is not None:
        et = p["exit_test"]
        lines += [
            "",
            "## Exit test — record survives GhostBox removal",
            f"- GhostBox involved: **{et['ghostbox_involved']}**",
            f"- detached verify status: **{et['status']}** (exit {et['exit_code']})",
            "- verified with only the shard bytes + the out-of-band public key.",
        ]
    lines += ["", "## Genesis live-probe receipts (carried forward)"]
    lines += [f"- {k}: {v}" for k, v in p["genesis_probe_receipts"].items()]
    lines += [""]
    return "\n".join(lines)
