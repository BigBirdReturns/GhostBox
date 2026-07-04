"""AXM Sovereign Spine v0.

A small, runnable proof of the opposite primitive to a collapsed vendor
operating layer: a real record is sealed by axm-genesis, verified through the
authority that owns it (an out-of-band key), annotated by GhostBox without
GhostBox ever owning or rewriting custody, exported as a reviewable packet --
and still verifiable after GhostBox is removed from the loop.

This package is the demo *harness*. It may drive genesis tooling directly
(keygen / compile / derive_shard_id). GhostBox's custody boundary is untouched:
the harness hands GhostBox a ``SealedShard`` and a real verifier, and GhostBox's
PR #2 code decides trust exactly as before.
"""
