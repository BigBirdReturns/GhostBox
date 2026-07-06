#!/bin/bash
# SessionStart bootstrap for Claude Code on the web.
#
# Brings up the pinned axm-genesis kernel WITHOUT pip-from-git (which some
# sandboxes block): clone the kernel repo at the pinned v1.0.0 commit, install
# only PyPI dependencies, and expose axm-build / axm-verify as thin wrappers so
# spine_v0.genesis_cli.kernel_available() sees them on PATH. After this hook,
# the full 108-test suite runs for real (no kernel skips) with:
#     python -m pytest tests/ -q
#
# Idempotent; safe to re-run. Web sessions only.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Pinned by full commit hash (axm-genesis v1.0.0) per the ledger release
# policy — the same pin as .github/workflows/ci.yml. Bump deliberately.
GENESIS_PIN="9074e7fb2e9cedde692b248cdd0c6a805e77d8ac"
BOOT="$HOME/.axm-bootstrap"
GEN="$BOOT/axm-genesis"
BIN="$BOOT/bin"
mkdir -p "$BIN"

# 1. Kernel source at the pin. Full clone (the pin is not the branch tip, so a
#    shallow clone cannot reach it); reused if already present.
if [ ! -f "$GEN/src/axm_build/cli.py" ]; then
  rm -rf "$GEN"
  git clone --quiet https://github.com/BigBirdReturns/axm-genesis "$GEN"
fi
git -C "$GEN" -c advice.detachedHead=false checkout --quiet "$GENESIS_PIN"

# 2. PyPI dependencies only — the kernel's runtime deps plus pytest.
python3 -m pip install --quiet --user \
  blake3 pynacl click 'dilithium-py>=0.5.0' pytest

# 3. Thin CLI wrappers (the kernel is run from source, not installed).
for spec in axm-build:axm_build axm-verify:axm_verify; do
  name="${spec%%:*}"
  mod="${spec##*:}"
  cat > "$BIN/$name" <<WRAP
#!/bin/sh
export PYTHONPATH="$GEN/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec python3 -c "from $mod.cli import main; main()" "\$@"
WRAP
  chmod +x "$BIN/$name"
done

# 4. Persist the environment for the whole session.
{
  echo "export PATH=\"$BIN:\$PATH\""
  echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR/src:$CLAUDE_PROJECT_DIR:$GEN/src\${PYTHONPATH:+:\$PYTHONPATH}\""
} >> "$CLAUDE_ENV_FILE"

echo "axm bootstrap: kernel @ ${GENESIS_PIN:0:7} from source, axm-build/axm-verify on PATH" >&2
