"""PhysicalEvidenceEvent contract shape — reconciled against the real axm-embodied surface.

The probe drove a real FrameCaptureRecorder session (sim frames, labeled as sim)
and sealed it with compile_frame_capsule. Reality: triggers carry BOTH a
caller-declared reason and source (never inferred), every kept record is keyed
by a session-monotonic frame_id, continuity is a chain hash anchored at a
session-bound chain_genesis, and nothing anywhere carries a "fidelity" field.
These are contract-level tests; no genesis kernel is required.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from ghostbox.interop.contracts import PhysicalEvidenceEvent, ProvenanceState


def make_event(**overrides) -> PhysicalEvidenceEvent:
    kwargs = dict(
        trigger="motion",
        trigger_source="sim-pir-3",
        sensor="sim-doorcam-01",
        continuity_ref="aa" * 32,
        content_hash="bb" * 32,
        frame_id=3,
        captured_at="2026-07-06T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return PhysicalEvidenceEvent(**kwargs)


# --- field reconciliation ----------------------------------------------------


def test_no_fidelity_field():
    # Nothing in the real surface carries fidelity; "high" asserted a quality
    # nothing measured.
    assert "fidelity" not in {f.name for f in fields(make_event())}


def test_trigger_source_is_required():
    # The real recorder refuses a trigger without an explicit reason AND source.
    with pytest.raises(TypeError):
        PhysicalEvidenceEvent(  # type: ignore[call-arg]
            trigger="motion", sensor="cam", continuity_ref="c", content_hash="h"
        )


def test_frame_id_keys_the_kept_record():
    assert make_event(frame_id=7).frame_id == 7
    assert "frame_id" in {f.name for f in fields(make_event())}


def test_reconciled_field_set():
    names = {f.name for f in fields(make_event())}
    assert names == {
        "trigger", "trigger_source", "sensor", "continuity_ref", "content_hash",
        "frame_id", "captured_at", "provenance", "event_id",
    }


# --- identity is content-addressed and deterministic -------------------------


def test_event_id_is_deterministic_and_covers_the_reconciled_fields():
    assert make_event().event_id == make_event().event_id
    assert make_event().event_id.startswith("phys:")
    # the new reality-bearing fields participate in the identity
    assert make_event(trigger_source="other").event_id != make_event().event_id
    assert make_event(frame_id=99).event_id != make_event().event_id


def test_provenance_asserts_faithful_capture_only():
    # PROVEN here asserts faithful capture of opaque sensor bytes, nothing about
    # what the frames mean; the custody boundary stays the sealed shard.
    assert make_event().provenance is ProvenanceState.PROVEN
