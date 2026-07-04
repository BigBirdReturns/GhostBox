"""Regression: persistent semantic coordinates must be reproducible.

Python's builtin ``hash()`` is salted per process (PYTHONHASHSEED), so deriving a
persistent coordinate from it means the same input lands on a different
coordinate after every restart. That is the bug these tests guard against.

The load-bearing test is ``test_coordinates_stable_across_hashseed``: it derives
the same coordinates in three separate Python processes started with different
PYTHONHASHSEED values and asserts they agree — and asserts that builtin ``hash()``
of the same string genuinely *differs* across those processes, so the test is
provably sensitive and cannot pass on a no-op.
"""
import inspect
import json
import os
import subprocess
import sys

from axiom.core import stable_hash_int

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Golden SHA-256-derived integer. Pins the algorithm: a change to the derivation
# (or an accidental fall-back to builtin hash) breaks this.
_GOLDEN_ELECTION = 90132080595682441782743617509416252186657455951464683392316550921892852662291


def test_stable_hash_int_is_deterministic_and_nonnegative():
    assert stable_hash_int("x") == stable_hash_int("x")
    v = stable_hash_int("election_integrity")
    assert isinstance(v, int) and v >= 0
    assert v == _GOLDEN_ELECTION


def test_derived_coordinates_are_in_range():
    for s in ("election_integrity", "Display", "LocalBusiness", "", "ก", "🔥"):
        assert 1 <= stable_hash_int(s) % 99 + 1 <= 99
        assert 1 <= stable_hash_int(s) % 9999 + 1 <= 9999


# Snippet run in a child process. Derives the same three persistent coordinates
# the production code derives, plus a builtin-hash contrast value.
_CHILD = r"""
import json, sys
sys.path.insert(0, {src!r})
from axiom.core import stable_hash_int
print(json.dumps({{
    "type_topic": stable_hash_int("election_integrity") % 99 + 1,
    "instance_screen": stable_hash_int("Display") % 9999 + 1,
    "schema_type": stable_hash_int("LocalBusiness") % 99 + 1,
    "builtin": hash("election_integrity"),
}}))
"""


def _derive(seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": _SRC}
    out = subprocess.check_output(
        [sys.executable, "-c", _CHILD.format(src=_SRC)], env=env, text=True
    )
    return json.loads(out)


def test_coordinates_stable_across_hashseed():
    a = _derive("0")
    b = _derive("12345")
    c = _derive("random")

    for key in ("type_topic", "instance_screen", "schema_type"):
        assert a[key] == b[key] == c[key], f"{key} drifted across PYTHONHASHSEED"

    # Sensitivity guard: builtin h() of a str IS salted, so it must differ across
    # the two fixed-but-different seeds. If this ever stops differing, the test
    # above is no longer proving anything.
    assert not (a["builtin"] == b["builtin"]), "builtin hash unexpectedly stable; test not sensitive"


def test_no_salted_hash_in_persistent_coordinate_paths():
    # Source-level guard so the fix can't silently regress. The exact salted-hash
    # derivations that were the bug must be gone, and each site must now derive
    # its coordinate via stable_hash_int. (The legitimate __hash__ dunders live
    # in axiom/core.py and are untouched.)
    import ghostbox.integration as integ
    import ghostbox.sources.photonic as photonic
    import axiom.adapters.schemaorg as schemaorg

    integ_src = inspect.getsource(integ)
    assert "hash(event.topic)" not in integ_src
    assert "hash(event.text" not in integ_src
    assert "hash(state.screen)" not in inspect.getsource(photonic)
    assert "hash(schema_type)" not in inspect.getsource(schemaorg)

    for mod in (integ, photonic, schemaorg):
        assert "stable_hash_int" in inspect.getsource(mod), (
            f"{mod.__name__} should derive coordinates via stable_hash_int"
        )
