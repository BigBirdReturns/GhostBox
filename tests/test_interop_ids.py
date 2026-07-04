"""Interop boundary id guarantees after the P1/P2 review fixes.

P1 (pin): every boundary id is SHA-256, a fixed contract constant, so the same
canonical object hashes identically across repos and machines — no optional
dependency variance.

P2 (split): ShardReceipt identity is split into an unsigned *body* id and a
signed *attestation* id, so two distinct signatures over the same body never
alias, and the id math is not circular (the body id never depends on a
signature).
"""
import re

from ghostbox.interop.contracts import EvidenceEvent, ShardReceipt, _HASH_TAG

_FIXED = "2026-07-04T00:00:00+00:00"
_EVT_RE = re.compile(r"^evt:sha256:[0-9a-f]{32}$")


# --- P1: pinned SHA-256 ------------------------------------------------------

def test_hash_algorithm_is_pinned_to_sha256():
    # A fixed contract constant, not an environment-dependent choice.
    assert _HASH_TAG == "sha256"


def test_event_id_is_sha256_tagged_and_deterministic():
    obs = {"app": "Settings", "screen": "Display"}
    a = EvidenceEvent(source="sg", surface="screen", observation=obs, captured_at=_FIXED)
    b = EvidenceEvent(source="sg", surface="screen", observation=obs, captured_at=_FIXED)
    assert a.event_id == b.event_id
    assert _EVT_RE.match(a.event_id), a.event_id


# --- P2: receipt identity split ----------------------------------------------

def _receipt(signature, *, shard="know:x", content="deadbeef", sealed=_FIXED):
    return ShardReceipt(shard_id=shard, content_hash=content, signature=signature, sealed_at=sealed)


def test_same_body_shares_body_id_regardless_of_signature():
    # The body id must not move when only the signature changes. This is also
    # the no-circularity guarantee: the signature is not part of the body
    # identity, so signing cannot depend on an id that depends on the signature.
    assert _receipt("sigA").receipt_body_id == _receipt("sigB").receipt_body_id


def test_distinct_signatures_do_not_alias():
    a, b = _receipt("sigA"), _receipt("sigB")
    assert a.receipt_id != b.receipt_id
    assert a.attestation_id == a.receipt_id  # attestation_id is the receipt_id


def test_body_id_changes_when_body_changes():
    assert _receipt("s", shard="know:x").receipt_body_id != _receipt("s", shard="know:y").receipt_body_id
    assert _receipt("s", content="aa").receipt_body_id != _receipt("s", content="bb").receipt_body_id
    assert _receipt("s", sealed="2026-01-01T00:00:00+00:00").receipt_body_id != _receipt("s").receipt_body_id


def test_ids_carry_explicit_sha256_prefix():
    r = _receipt("sig")
    assert r.receipt_body_id.startswith("rbody:sha256:")
    assert r.receipt_id.startswith("rcpt:sha256:")
