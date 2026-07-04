"""Real axm-genesis tooling wrappers for the spine harness.

The important one is ``RealGenesisVerifier``: it is the ``GenesisVerifier``
callable that PR #2's ``GenesisTrustKernel`` injects, but backed by the real
``axm-verify`` CLI instead of a fake. The adapter boundary is unchanged --
this returns genesis's frozen exit code (0 PASS / 1 FAIL / 2 MALFORMED) and
GhostBox maps it. GhostBox never opens the manifest or signature itself.

genesis imports are lazy so this module loads even where the kernel is absent
(so tests can skip cleanly and the packet/attention code stays importable).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# GhostBox contract shape (from the custody branch / PR #2). No GhostBox
# *behavior* is imported here -- only the boundary object the harness fills.
from ghostbox.interop.contracts import SealedShard

AXM_VERIFY = "axm-verify"
AXM_BUILD = "axm-build"


class GenesisKernelUnavailable(RuntimeError):
    pass


def kernel_available() -> bool:
    """True iff the real genesis CLIs are on PATH."""
    return shutil.which(AXM_VERIFY) is not None and shutil.which(AXM_BUILD) is not None


class RealGenesisVerifier:
    """The real ``axm-verify`` CLI as the injected GenesisVerifier.

    Signature matches PR #2's ``GenesisVerifier = Callable[[str, str], int]``.
    Returns the frozen exit code; GhostBox maps it to VerifyStatus. Records the
    last raw JSON result so the harness can put genesis's own words in the
    packet -- GhostBox never sees that, it only gets the int.

    The kernel adapter only calls this when it *has* a trusted key (it owns
    ``NO_TRUSTED_KEY`` itself), so ``axm-verify`` is always invoked *with*
    ``--trusted-key``; the usage-error exit 2 (missing key) never reaches here,
    leaving exit 2 to mean a malformed shard.
    """

    def __init__(self, axm_verify: str = AXM_VERIFY) -> None:
        self._bin = axm_verify
        self.last_code: Optional[int] = None
        self.last_result: Optional[dict] = None

    def __call__(self, shard_dir: str, trusted_key: str) -> int:
        proc = subprocess.run(
            [self._bin, "shard", shard_dir, "--trusted-key", trusted_key],
            capture_output=True,
            text=True,
        )
        self.last_code = proc.returncode
        self.last_result = _parse_verifier_json(proc.stdout, proc.stderr)
        return proc.returncode


def _parse_verifier_json(stdout: str, stderr: str) -> dict:
    body = stdout.strip()
    if body:
        try:
            return json.loads(body.splitlines()[-1])
        except json.JSONDecodeError:
            pass
    return {"raw_stdout": stdout, "raw_stderr": stderr}


def keygen(outdir: Path, name: str = "spine_publisher") -> Tuple[Path, Path]:
    """Generate a throwaway axm-hybrid1 keypair via real ``axm-build keygen``.

    Returns (secret_key_path, public_key_path). The public key is the
    out-of-band trust anchor; the secret stays in the disposable workdir.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [AXM_BUILD, "keygen", str(outdir), "--name", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return outdir / f"{name}.key", outdir / f"{name}.pub"


def seal_sample_record(
    workdir: Path,
    *,
    namespace: str = "spine/v0",
    title: str = "AXM Sovereign Spine v0 sample record",
    created_at: str = "2026-07-04T00:00:00Z",
) -> Tuple[Path, Path, str]:
    """Create a disposable sample record and seal it with the real kernel.

    Returns (shard_dir, out_of_band_public_key_path, publisher_name).
    """
    content_dir = workdir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    source = content_dir / "source.txt"
    source.write_text("Tourniquet stops Severe Bleeding.", encoding="utf-8")
    nbytes = len(source.read_bytes())

    candidates = workdir / "candidates.jsonl"
    rows = [
        {"type": "entity", "namespace": namespace, "label": "Tourniquet", "entity_type": "concept"},
        {"type": "entity", "namespace": namespace, "label": "Severe Bleeding", "entity_type": "concept"},
        {
            "type": "claim",
            "subject_label": "Tourniquet",
            "predicate": "stops",
            "object_label": "Severe Bleeding",
            "object_type": "entity",
            "tier": 1,
            "evidence": {
                "source_file": "source.txt",
                "byte_start": 0,
                "byte_end": nbytes,
                "text": "Tourniquet stops Severe Bleeding.",
            },
        },
    ]
    candidates.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    key_path, pub_path = keygen(workdir / "keys", name="spine_publisher")
    shard_dir = workdir / "shard"
    subprocess.run(
        [
            AXM_BUILD, "compile", str(candidates), str(content_dir), str(shard_dir),
            "--private-key", str(key_path),
            "--namespace", namespace, "--title", title, "--created-at", created_at,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return shard_dir, pub_path, "spine_publisher"


def sealed_shard_from_dir(shard_dir: Path) -> SealedShard:
    """Build the GhostBox ``SealedShard`` from a real sealed shard directory.

    The identity is derived by *genesis's own* function
    (``axm_verify.crypto.derive_shard_id``); the harness computes it and hands
    it to GhostBox verbatim. GhostBox never derives it. ``merkle_root`` /
    ``sealed_at`` are read-only projections of manifest fields.
    """
    from axm_verify.crypto import derive_shard_id  # lazy: kernel-only

    manifest_bytes = (shard_dir / "manifest.json").read_bytes()
    shard_id = derive_shard_id(manifest_bytes)
    m = json.loads(manifest_bytes)
    return SealedShard(
        shard_id=shard_id,
        shard_dir=str(shard_dir),
        suite=m.get("suite", "axm-hybrid1"),
        merkle_root=(m.get("integrity") or {}).get("merkle_root"),
        sealed_at=(m.get("metadata") or {}).get("created_at"),
    )
