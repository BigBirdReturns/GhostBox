"""The exit property: the record survives GhostBox removal.

Verify a sealed shard using ONLY its bytes plus the out-of-band public key,
through the genesis verifier CLI. This module imports **no GhostBox code** on
purpose -- it is the proof that the record's verifiability does not depend on
the attention service. That detachment is the core product claim.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def verify_detached(shard_dir: str | Path, trusted_key: str | Path, axm_verify: str = "axm-verify") -> Dict[str, Any]:
    """Verify with only shard bytes + out-of-band pub. No GhostBox in the loop."""
    proc = subprocess.run(
        [axm_verify, "shard", str(shard_dir), "--trusted-key", str(trusted_key)],
        capture_output=True,
        text=True,
    )
    result: Dict[str, Any] = {}
    body = proc.stdout.strip()
    if body:
        try:
            result = json.loads(body.splitlines()[-1])
        except json.JSONDecodeError:
            result = {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}
    return {
        "exit_code": proc.returncode,
        "status": result.get("status"),
        "ghostbox_involved": False,
        "genesis_result": result,
    }
