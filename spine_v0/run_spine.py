"""Run the AXM Sovereign Spine v0 end to end.

    seal (real genesis)  ->  verify through GhostBox's custody seam (real kernel)
    ->  GhostBox attention finding  ->  reviewable packet  ->  exit test.

GhostBox's PR #2 code is used unchanged: the harness injects the *real*
axm-verify CLI as the verifier and hands GhostBox a SealedShard. GhostBox
decides trust, fails closed, retains only the reference, and cannot seal.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ghostbox.interop.contracts import VerifyStatus
from ghostbox.interop.genesis_spoke import (
    CustodyOutcome,
    GenesisCustodySpoke,
    GenesisTrustKernel,
)

from spine_v0.attention import surface_finding
from spine_v0.exit_test import verify_detached
from spine_v0.genesis_cli import (
    RealGenesisVerifier,
    kernel_available,
    seal_sample_record,
    sealed_shard_from_dir,
)
from spine_v0.packet import build_packet, render_markdown


def run(out_dir: Path) -> dict:
    """Run the spine once; write packet.{json,md} to out_dir; return the packet."""
    if not kernel_available():
        raise SystemExit(
            "axm-genesis kernel not on PATH (need `axm-verify` and `axm-build`).\n"
            "Install it, e.g.:  pip install -e '/path/to/axm-genesis[dev]'"
        )

    workdir = Path(tempfile.mkdtemp(prefix="axm_spine_v0_"))

    # 1) Seal a disposable real record with genesis tooling.
    shard_dir, oob_pub, key_id = seal_sample_record(workdir)

    # 2) Build the SealedShard from genesis-derived identity (harness computes it;
    #    GhostBox receives it verbatim).
    sealed = sealed_shard_from_dir(shard_dir)

    # 3) Verify + take custody through GhostBox's PR #2 seam, with the REAL kernel
    #    injected. The out-of-band pub is the trust anchor.
    verifier = RealGenesisVerifier()
    kernel = GenesisTrustKernel(verifier, trusted_key=str(oob_pub), trusted_key_id=key_id)
    spoke = GenesisCustodySpoke(kernel)
    entry = spoke.ingest_sealed_shard(sealed)

    if entry.outcome is not CustodyOutcome.VERIFIED:
        raise SystemExit(f"custody did not verify a freshly-sealed shard: {entry.status.value}")
    assert spoke.is_trusted(sealed.shard_id), "verified shard must be trusted"
    assert entry.status is VerifyStatus.PASS

    # 4) Attention downstream of verification.
    finding = surface_finding(sealed)

    # 5) Exit test: verify the record with GhostBox removed from the loop.
    exit_result = verify_detached(shard_dir, oob_pub)

    # 6) Reviewable packet.
    packet = build_packet(
        sealed=sealed,
        trusted_key_source=str(oob_pub),
        verify_status=entry.status.value,
        genesis_result=verifier.last_result,
        finding=finding,
        custody_entry=entry,
        exit_test=exit_result,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "packet.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    (out_dir / "packet.md").write_text(render_markdown(packet), encoding="utf-8")
    return packet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run AXM Sovereign Spine v0.")
    ap.add_argument("--out", default="spine_v0_out", help="directory for packet.json / packet.md")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    packet = run(out_dir)
    print(render_markdown(packet))
    print(f"\n[packet written to {out_dir}/packet.json and {out_dir}/packet.md]")
    et = packet["exit_test"]
    ok = packet["verification"]["status"] == "pass" and et["status"] == "PASS" and et["ghostbox_involved"] is False
    print(f"[spine v0: {'OK' if ok else 'INCOMPLETE'} — verified={packet['verification']['status']}, "
          f"detached-verify={et['status']}, ghostbox-in-exit-path={et['ghostbox_involved']}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
