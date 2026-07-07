"""GhostBox conversation observer — end-to-end against a REAL genesis-sealed shard.

Seals a conversation-shaped shard with the real kernel, replicating exactly the
shape the probed ``axm-chat import`` produces (namespace ``chat/conversation``,
one ``conversation/{id}`` entity plus ``turn/{n}`` entities, ``has_title`` /
``message_count`` literals and ``has_turn`` claims over turn-block evidence),
wraps it in a ConversationShardRef, and drives the observer through the LANDED
custody seam. Skips cleanly without the kernel.

Control question, as assertions: GhostBox observes a real conversation shard
ONLY after verifying its underlying genesis-sealed record, then adds findings
without becoming the conversation or custody owner — and never reads a shard
that verified but is not a conversation shard.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from ghostbox.interop.contracts import ConversationShardRef, VerifyStatus
from ghostbox.interop.genesis_spoke import GenesisCustodySpoke, GenesisTrustKernel
from ghostbox.interop import conversation_observer as co_module
from ghostbox.interop.conversation_observer import (
    CONVERSATION_NAMESPACE,
    ConversationObserver,
)

from spine_v0.genesis_cli import (
    AXM_BUILD,
    RealGenesisVerifier,
    keygen,
    kernel_available,
    seal_sample_record,
    sealed_shard_from_dir,
)

pytestmark = pytest.mark.skipif(
    not kernel_available(),
    reason="axm-genesis kernel (axm-verify / axm-build) not on PATH",
)


def seal_conversation_record(workdir: Path) -> tuple[Path, Path, str]:
    """Seal a conversation record shaped exactly like real axm-chat output.

    Same turn-block source layout, same entity labels (``conversation/{id}``,
    ``turn/{n}``), same claim vocabulary (``has_title`` / ``message_count``
    literals at tier 0, ``has_turn`` entity claims at tier 1), same constant
    namespace. Returns (shard_dir, out_of_band_public_key_path, publisher_name).
    """
    conv_id = "conv-observer-test"
    title = "Observer test conversation"
    turns = [
        ("HUMAN", "What must be verified first?"),
        ("ASSISTANT", "Custody, before any content is read."),
    ]

    blocks = [f"{role}:\n{text}" for role, text in turns]
    source_text = (
        f"=== CONVERSATION: {title} ===\nID: {conv_id}\nStarted:\n\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    content_dir = workdir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / "source.txt").write_text(source_text, encoding="utf-8")
    source_bytes = source_text.encode("utf-8")

    def evidence(text: str) -> dict:
        start = source_bytes.index(text.encode("utf-8"))
        return {
            "source_file": "source.txt",
            "byte_start": start,
            "byte_end": start + len(text.encode("utf-8")),
            "text": text,
        }

    conv_label = f"conversation/{conv_id}"
    rows: list[dict] = [
        {"type": "entity", "namespace": CONVERSATION_NAMESPACE, "label": conv_label,
         "entity_type": "concept"},
    ]
    for n in range(len(turns)):
        rows.append({"type": "entity", "namespace": CONVERSATION_NAMESPACE,
                     "label": f"turn/{n}", "entity_type": "concept"})
    header = f"=== CONVERSATION: {title} ==="
    rows.append({"type": "claim", "subject_label": conv_label, "predicate": "has_title",
                 "object_label": title, "object_type": "literal:string", "tier": 0,
                 "evidence": evidence(header)})
    rows.append({"type": "claim", "subject_label": conv_label, "predicate": "message_count",
                 "object_label": str(len(turns)), "object_type": "literal:integer", "tier": 0,
                 "evidence": evidence(f"ID: {conv_id}")})
    for n, block in enumerate(blocks):
        rows.append({"type": "claim", "subject_label": conv_label, "predicate": "has_turn",
                     "object_label": f"turn/{n}", "object_type": "entity", "tier": 1,
                     "evidence": evidence(block)})

    candidates = workdir / "candidates.jsonl"
    candidates.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    key_path, pub_path = keygen(workdir / "keys", name="chat_publisher")
    shard_dir = workdir / "shard"
    subprocess.run(
        [
            AXM_BUILD, "compile", str(candidates), str(content_dir), str(shard_dir),
            "--private-key", str(key_path),
            "--namespace", CONVERSATION_NAMESPACE,
            "--title", title,
            "--created-at", "2026-07-06T00:00:00Z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return shard_dir, pub_path, "chat_publisher"


@pytest.fixture(scope="module")
def cref(tmp_path_factory):
    base = tmp_path_factory.mktemp("cobs")
    shard_dir, oob_pub, key_id = seal_conversation_record(base)
    sealed = sealed_shard_from_dir(shard_dir)
    ref = ConversationShardRef.over(sealed, export_ref="exports/conversations.json")
    return ref, oob_pub, key_id, shard_dir, base


def _observer(oob_pub, key_id):
    kernel = GenesisTrustKernel(RealGenesisVerifier(), trusted_key=str(oob_pub), trusted_key_id=key_id)
    return ConversationObserver(GenesisCustodySpoke(kernel))


def _digest_tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --- verified -> findings downstream of the verified shard_id ---------------


def test_verified_shard_produces_findings_downstream_of_shard_id(cref):
    ref, oob_pub, key_id, *_ = cref
    obs = _observer(oob_pub, key_id)
    res = obs.observe(ref)

    assert res.verified is True and res.status is VerifyStatus.PASS
    assert res.shard_id == ref.shard_id
    assert res.namespace == CONVERSATION_NAMESPACE
    assert len(res.findings) >= 1
    for f in res.findings:
        assert ref.shard_id in f.input_refs          # downstream of the verified shard_id
        assert f.tension_type == "conversation_claim"
        assert f.provenance.value == "untested"      # observation, not truth


def test_findings_surface_the_real_conversation_vocabulary(cref):
    ref, oob_pub, key_id, *_ = cref
    res = _observer(oob_pub, key_id).observe(ref)

    predicates = {c.split(" ")[1] for f in res.findings for c in f.claims if len(c.split(" ")) > 1}
    # The probed axm-chat vocabulary, surfaced verbatim — never interpreted.
    assert {"has_title", "message_count", "has_turn"} <= predicates


# --- fail closed: unverified shards are never observed as trusted -----------


def test_wrong_key_blocks_trusted_observation(cref):
    ref, _oob, _kid, _sd, base = cref
    _, foreign_pub = keygen(base / "foreign", name="attacker")
    obs = _observer(foreign_pub, "attacker")
    res = obs.observe(ref)

    assert res.verified is False and res.status is VerifyStatus.FAIL
    assert res.namespace is None                     # manifest never read
    assert res.findings == ()                        # NO conversation findings over an untrusted shard
    assert obs.findings == ()


def test_missing_key_is_no_anchor_before_verification(cref):
    ref, *_ = cref
    verifier = RealGenesisVerifier()
    obs = ConversationObserver(GenesisCustodySpoke(GenesisTrustKernel(verifier, trusted_key=None)))
    res = obs.observe(ref)

    assert res.status is VerifyStatus.NO_TRUSTED_KEY
    assert res.verified is False and res.findings == ()
    assert verifier.last_code is None                # the CLI was never reached


# --- a verified NON-conversation shard is a finding, not content ------------


def test_verified_non_conversation_shard_is_flagged_not_read(tmp_path):
    # A real sealed shard from another namespace (the spine sample) verifies,
    # but it is not a conversation shard: one bounded finding, no claim reads.
    shard_dir, oob_pub, key_id = seal_sample_record(tmp_path)
    ref = ConversationShardRef.over(sealed_shard_from_dir(shard_dir))
    obs = _observer(oob_pub, key_id)
    res = obs.observe(ref)

    assert res.verified is True and res.status is VerifyStatus.PASS
    assert res.namespace == "spine/v0"
    assert len(res.findings) == 1
    f = res.findings[0]
    assert f.tension_type == "not_conversation_shard"
    assert ref.shard_id in f.input_refs
    assert f.claims == []                            # no claim was read from the graph


# --- producer metadata is never treated as custody --------------------------


def test_spoke_and_export_ref_are_metadata_not_custody(cref):
    ref, oob_pub, key_id, *_ = cref
    assert ref.spoke == "axm-chat"
    assert ref.export_ref == "exports/conversations.json"

    res = _observer(oob_pub, key_id).observe(ref)
    for f in res.findings:
        # findings key to genesis-owned ids only, never to producer export refs
        assert "exports/conversations.json" not in f.input_refs
        assert all(r.startswith(("sh1_", "c1_", "e1_")) for r in f.input_refs)


# --- GhostBox owns only findings + the external reference -------------------


def test_ghostbox_retains_only_findings_and_shard_id_reference(cref):
    ref, oob_pub, key_id, *_ = cref
    obs = _observer(oob_pub, key_id)
    res = obs.observe(ref)

    assert res.shard_id == ref.shard_id
    for custody_field in ("manifest", "signature", "merkle_root", "suite"):
        assert not hasattr(res, custody_field)       # no custody material in the result
    assert obs.findings == res.findings              # only its own finding state


def test_observation_never_rewrites_the_sealed_shard(cref):
    ref, oob_pub, key_id, _sd, _base = cref
    shard_dir = Path(ref.sealed.shard_dir)
    before = _digest_tree(shard_dir)
    _observer(oob_pub, key_id).observe(ref)
    assert _digest_tree(shard_dir) == before         # observation writes nothing into the shard


# --- reuses the custody seam; no duplicated genesis verification ------------


def test_observer_does_not_reimplement_or_call_verification():
    # Verification is the custody seam's job: the observer must not shell out to
    # the CLI or import genesis's crypto machinery. (Checks imports/calls, not the
    # docstring prose, which legitimately *names* axm-verify to say it isn't called.)
    src = inspect.getsource(co_module)
    assert "import subprocess" not in src
    assert "axm_verify" not in src                   # no genesis verifier import
    assert "axm_build" not in src                    # no genesis compiler import
    assert "GenesisCustodySpoke" in src              # it reuses the landed custody seam


def test_observer_never_imports_the_chat_spoke():
    # The observer reads the VERIFIED sealed record; it never reaches into
    # axm-chat's package for a second view of the conversation.
    src = inspect.getsource(co_module)
    assert "axm_chat" not in src
