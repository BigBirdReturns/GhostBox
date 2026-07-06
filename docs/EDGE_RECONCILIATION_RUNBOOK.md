# Edge reconciliation runbook

Status: durable procedure, extracted from four completed reconciliations
(genesis, axm-core, axm-chat, axm-embodied). Every edge the interop contract
(`src/ghostbox/interop/contracts.py`) currently declares has been through this
procedure once. This document is what a future reconciliation — a new spoke,
or a re-verification of an old one — follows instead of re-deriving the method
from commit messages.

Audience: a reader with the repos checked out and nothing else. No session
context, no memory of who did the four prior passes or why. If you are that
reader: read this whole document before touching a contract file.

## Why this exists

`contracts.py` declares boundary objects for every AXM spoke GhostBox talks to.
Each object started life as a **mirror-side guess**: a plausible shape,
written before anyone drove the spoke's real production path. Four times, the
guess was wrong in a specific, recurring way — fields nothing real backs,
identity re-invented instead of composed, an API the protocol assumed but the
spoke never shipped. Each time, the fix was the same five-step procedure. This
document is that procedure, generalized past the four specific edges so it
transfers to the fifth, the tenth, or the one reconciled decades from now by
someone who has never heard of this session.

The core decision, already made and not up for re-litigation per edge: **when
the contract and reality disagree, the contract is wrong.** Reality is
authority. The correction is written once, in the shared contract, not
N times as private adapter-side workarounds. See `GENESIS_EDGE_MISMATCHES.md`
("The decision") for the record of why the alternative — let every spoke
preserve its own semantics locally — was rejected: it is drift by construction,
N private corrections, no shared truth, a second custody model per spoke.

## The five steps

Follow these in order. Do not skip a step because the spoke "looks simple."
Every one of the four precedent edges looked simple until it was probed.

### 1. Probe reality first

Drive the spoke's REAL production path end to end — a real import, a real
capture, a real compile — producing an actual genesis-sealed shard. This is
not optional and it is not a documentation exercise: until a real artifact
exists, every field on the contract object is a guess, no matter how
confident it looks.

Rules for the probe:

- **Sims are allowed, guesses are not.** If the spoke's real input is a live
  sensor, a simulated sensor feeding the REAL recorder/compiler code path is a
  valid probe — but it must be labeled as a sim in the contract docstring and
  in the commit message. A hand-built JSON blob that merely *looks like* the
  spoke's output is not a probe of anything; it probes your own assumptions.
- **Drive the real code, not a mock of it.** The axm-chat probe ran the real
  `axm_chat.spoke.import_export()`. The axm-embodied probe ran a real
  `FrameCaptureRecorder` session and the real `compile_frame_capsule`. Neither
  probe hand-assembled a shard directory and called it done.
- **Record, at minimum, from the probe:**
  - namespace (the sealed manifest's `metadata.namespace`)
  - publisher id (`metadata.publisher_id` / equivalent)
  - the entity and claim vocabulary actually emitted (exact predicate names,
    not paraphrases)
  - any extension tables published (e.g. `ext/streams@1.jsonl`) and why
  - the derived `sh1_` shard id (genesis's own derivation, never
    self-computed — see "boundaries that never move" below)
  - a **detached** `axm-verify` PASS against the shard, using an
    **out-of-band** trusted key
- **The out-of-band key is not optional and not decorative.** Verify using a
  key supplied independently of the shard directory — never the shard's own
  `sig/publisher.pub`. Trusting the embedded key is the anchor trap: it makes
  every shard verify against itself, which is not verification, it is an
  echo. This is perimeter finding F1 (see `src/ghostbox/interop/genesis_spoke.py`,
  `GenesisTrustKernel.verify`) and it is closed structurally, not by
  discipline: `NO_TRUSTED_KEY` is returned, and the kernel is never called,
  when no external key is supplied.

If the spoke exposes no way to drive its real path in your environment (no
kernel on `PATH`, no test fixture, no sample data), the reconciliation cannot
proceed to step 2. Say so and stop; do not guess further and call it
"probed."

### 2. Correct the contract TO reality, never reality to the contract

With a real probed shard in hand, diff every field the contract currently
declares against what the probe actually produced or actually required.
Three outcomes per field:

- **Nothing real backs it → delete it.** Precedents: `KnowledgeShardRef.summary`
  (no summary anywhere in the real axm-core output — a summary, if wanted
  later, is a GhostBox-side annotation, never core authority),
  `ConversationShardRef.provenance` (redundant with the embedded `SealedShard`'s
  own provenance — the field doubled a source of truth),
  `PhysicalEvidenceEvent.fidelity` (nothing in the real recorder measures a
  "high"/"low" quality; the field asserted something nobody computed).
- **Reality requires it but the guess omitted it → add it.** Precedents:
  `trigger_source` (the real `FrameCaptureRecorder` refuses a trigger without
  an explicit reason AND source — the guess only had `trigger`),
  `frame_id` (the real recorder keys every kept record, gaps included, by a
  session-monotonic id — the guess had no way to expose what was NOT kept).
- **Identity is never re-invented.** If the spoke's real output is itself a
  genesis-sealed shard (true for every edge reconciled so far), the contract
  object composes OVER `SealedShard` — it does not carry its own `shard_id`
  field. Identity is a read-only property that reads through to
  `self.sealed.shard_id`; it is never minted, never overridden, never a
  second source of truth. If a future spoke's real output is NOT itself a
  sealed shard, its identity still stays downstream of whatever `SealedShard`
  eventually custodies it — it does not get an independent identity scheme
  either. This is non-negotiable per the standing decision above: one custody
  pattern, not a second one per edge.
- **Producer metadata is kept, but never mistaken for custody.** Fields like
  `compiler`, `source_refs`, `export_ref`, `spoke` describe what the producer
  claims about itself. Keep them if the guess had a legitimate reason to want
  them, but document explicitly, in the dataclass docstring, that they are
  metadata: not recoverable from the sealed shard, not verifiable, never to
  be treated as custody-anchored, and never placed in an observer finding's
  `input_refs`.
- **If the protocol claimed an API the spoke doesn't have → re-document it as
  a consumer-side adapter role, not silently drop it or silently leave it
  aspirational.** Precedents: `ConversationSpoke.emit_conversation_shards`
  (axm-chat's real surface is `import_export()`, which returns counts, not
  shard refs — there is no shard-reference API to call) and
  `EmbodiedSource.emit_physical` (axm-embodied's real surface is
  `FrameCaptureRecorder` + `compile_frame_capsule`, not an event-emitting
  method). Both `Protocol` classes stay in `contracts.py` — the shape they
  describe is still useful, as the shape a GhostBox-side consumer adapter
  should have when it scans sealed shard directories and builds refs over
  them. What changes is the docstring: it says, explicitly, "this is not an
  interface the spoke package implements or is expected to grow."

Write the reconciliation into the dataclass's own docstring (see
`KnowledgeShardRef`, `ConversationShardRef`, `PhysicalEvidenceEvent` in
`contracts.py` for the pattern): which branch/commit probed it, what the real
surface produces, and a "guess → reality" list of every field that moved.
The docstring is the permanent record; do not rely on the commit message
alone — commit history can be squashed, rebased, or simply not checked out by
whoever reads the contract next.

### 3. Land the observer on the ONE custody pattern

Every reconciled edge gets exactly one observer, and every observer is
structured identically:

1. **Constructor takes a `GenesisCustodySpoke`.** Never a raw kernel, never a
   direct verifier. The observer delegates verification to the landed custody
   seam (`src/ghostbox/interop/genesis_spoke.py`) — it does not call
   `axm-verify`, does not import `axm_verify`/`axm_build`/`subprocess`, and
   does not re-implement any part of genesis's crypto or CLI. (Tests enforce
   this with `inspect.getsource(observer_module)` string checks — see
   `test_observer_does_not_reimplement_or_call_verification` in both
   `tests/test_conversation_observer.py` and `tests/test_physical_observer.py`.
   Do the same for a new observer: assert the module source contains
   `GenesisCustodySpoke` and does not contain the forbidden imports.)
2. **`observe()` verifies custody FIRST and fails closed.** On any outcome
   that is not `CustodyOutcome.VERIFIED` (i.e. genesis did not return `PASS`),
   return immediately with empty findings and nothing read. No retry, no
   quarantine-and-proceed, no partial trust. The custody spoke has already
   recorded the failure in its own ledger; the observer's only job on failure
   is to add nothing.
3. **Only after VERIFIED, bounded read-only reads of sealed content.** Read
   exactly the files needed (a manifest, a graph directory, a stream index)
   and nothing else. Never write into the sealed shard directory — verify
   this with a before/after digest of the shard tree across the observe call
   (see `_digest_tree` + `test_observation_never_rewrites_the_sealed_shard` in
   both observer test files).
4. **Kind/tier checks emit bounded untrusted findings, never a silent pass.**
   If the verified shard is not the kind expected (wrong namespace, wrong
   evidence tier, a stream index that disagrees with a declared count), that
   is itself a finding — `not_conversation_shard`, `unexpected_evidence_tier`,
   `physical_stream_index_mismatch`, etc. — not a quiet no-op and not a
   crash. See `conversation_observer.py`'s namespace check and
   `physical_observer.py`'s tier + stream-index-agreement checks for the
   pattern.
5. **Findings key to genesis-owned ids verbatim.** `input_refs` on every
   emitted `AttentionFinding` carry the `sh1_`/`c1_`/`e1_` ids read from the
   verified shard, never a GhostBox-minted id, never a producer metadata
   field (an export path, a compiler name). Test this directly: assert
   producer metadata strings never appear in `input_refs`, and that every ref
   matches the genesis id prefixes (see
   `test_spoke_and_export_ref_are_metadata_not_custody`).
6. **The observer never imports the spoke's own package.** It consumes the
   sealed shard shape, never the spoke as a second authority
   (`test_observer_never_imports_the_chat_spoke`,
   `test_observer_never_imports_the_embodied_spoke`). If the spoke's package
   name would ever need to appear in the observer module, that is a sign the
   observer is reaching around the custody seam instead of trusting it.
7. **The sealed shard is never written, and content that isn't needed is
   never opened.** The physical observer never opens `frames.bin` even to
   hash it — hashing untrusted content for a second time would itself be
   starting a second, unauthorized judge over the evidence. Bound reads to
   exactly what the finding needs.

A new observer for a new edge should read as a structural clone of
`knowledge_observer.py` / `conversation_observer.py` / `physical_observer.py`
/ `pixel_observer.py`, with only the shard-kind check and the finding
vocabulary changed. If a new edge seems to need a different shape (a second
verification path, an observer that writes back, an observer that imports the
spoke package), that is a signal to stop and re-examine the design, not a
signal to add a new pattern.

### 4. Live-prove against the probe artifact

Before an edge is called reconciled, the new observer must run — successfully,
producing the expected custody PASS and the expected findings — against the
ACTUAL probe artifact from step 1, not only against the test suite's replica
shard (step 5 below covers the replica; it is necessary but not sufficient).
This is what makes the whole exercise more than an internally-consistent
fiction: the test replica is pinned to the probe's shape, but only a run
against the real probe artifact confirms the pin is honest.

Record, in the docstring and/or commit message: the probe shard's id, the
custody verdict obtained, and the finding count/kinds obtained. See the
appendix below for the two most recent examples of this record.

### 5. Record it

- Update `GENESIS_EDGE_MISMATCHES.md`'s "Status updates since this pass"
  section (or the equivalent living status doc at reconciliation time) with:
  which edge, which branch/commit, what the probe found, what changed in the
  contract, and the live-proof result.
- The commit message states: ownership of the change (which edge, whose
  surface), scope as a literal file-count claim ("diff limited to the N edge
  files" — precedent: `2438cbf`'s "diff limited to the 5 seam files"), the
  live-proof numbers, and an honest caveat line. The caveat line is not
  boilerplate — precedent caveats: "real kernel, single small shard,
  dilithium-py fallback — functional, not load-proven" and an explicit list of
  what was NOT touched by this pass. A reconciliation that claims more than it
  did is worse than one that admits its limits.
- Do not claim other edges are reconciled because one edge is. Each edge is
  reconciled independently and the status doc says so per edge.

## Test-fixture strategy

Two distinct test layers exist per edge, and they test different things:

- **Contract-shape tests** (e.g. `tests/test_conversation_shard_ref.py`,
  `tests/test_knowledge_shard_ref.py`) run kernel-free, always, on any
  machine. They use a fake `TrustKernel` (a `_RecordingKernel` that records
  what it was asked to verify and refuses to seal) to check the dataclass
  shape itself: identity reads through to the embedded `SealedShard` and
  can't be overridden, the ref can't be constructed without a sealed shard,
  `verify()` delegates rather than re-implementing, and so on. These tests
  carry no opinion about what a real spoke produces — they only enforce that
  the contract object's shape can't drift from "compose over `SealedShard`,
  never mint identity."
- **Observer tests** (`tests/test_conversation_observer.py`,
  `tests/test_physical_observer.py`) are kernel-gated:
  `pytestmark = pytest.mark.skipif(not kernel_available(), ...)`, where
  `kernel_available()` (`spine_v0/genesis_cli.py`) checks that the real
  `axm-verify` and `axm-build` CLIs are on `PATH`. These tests seal a
  **replica** of the probed shape — not the probe artifact itself, which is
  disposable — through the real kernel, then drive the real observer against
  that real, freshly-sealed shard.
  - When the replica's shape needs no extension tables, seal it via the
    frozen `axm-build compile` CLI as a subprocess (see
    `seal_conversation_record` in `test_conversation_observer.py`, calling
    `AXM_BUILD, "compile", ...`).
  - When the replica needs an extension table the frozen CLI has no flag for
    (e.g. `ext/streams@1.jsonl`), seal it via the kernel's own Python API,
    `axm_build.compiler_generic.compile_generic_shard`, passing
    `extra_ext={...}` directly (see `seal_physical_record` in
    `test_physical_observer.py`). This is the SAME code path the real spoke's
    compiler calls — it is not a second, weaker sealing route, it is the only
    route available for shapes the CLI doesn't expose yet.
  - Either way, the replica's entity labels, claim/predicate vocabulary,
    namespace, and (where relevant) extension-table rows must match the
    step-1 probe exactly — not "close enough." The replica is pinned to the
    probe; if the probe used `has_title`/`message_count`/`has_turn` claims,
    the replica uses exactly those, not paraphrases.
  - The replica being sealed through the real kernel is what makes these
    tests meaningfully different from a pure mock — but the replica is still
    a re-creation, not the original probe. **The live-proof in step 4, run
    once against the actual probe artifact, is what makes the replica's
    pinning honest** rather than a second, unverified guess about what the
    probe looked like. Do not skip step 4 because the replica tests pass;
    they test different things.

## Boundaries that never move

These hold across every edge, past and future. If a reconciliation would
require violating one of these, the reconciliation is wrong, not the
boundary.

- **Never mint `sh1_`.** GhostBox (and any consumer-side adapter) never
  derives or assigns a shard id. It is read verbatim from genesis's own
  derivation (`axm_verify.crypto.derive_shard_id` at probe/test-fixture time;
  the manifest's own identity at runtime).
- **Never trust the embedded publisher key.** Verification always uses a
  trusted key supplied out of band, independent of the shard directory being
  verified. A shard that verifies only against its own embedded key has not
  been verified against anything (perimeter finding F1).
- **Fail closed.** Any non-`PASS` `VerifyStatus` (`FAIL`, `MALFORMED`,
  `NO_TRUSTED_KEY`) blocks all downstream reads and all findings. No retry,
  no partial trust, no "probably fine."
- **No interpretation of sealed content.** No OCR, no vision models, no
  sentiment/semantic classifiers, no summarization of what a claim "means."
  Findings surface what the sealed record verbatim says (a claim triple, a
  declared tier, a declared count), never an inference about what it means or
  whether it's true.
- **Reviews never say "true."** `AttentionFinding.provenance` distinguishes a
  verified SEAL (`PROVEN` — the capture or signature is faithful) from a
  claim about the world (`UNTESTED` — surfaced for downstream testing, not
  asserted). An observer proving a shard's custody never upgrades a content
  claim to "true" as a side effect. Human-review findings stay `UNTESTED` by
  construction; see `physical_observer.py`'s `human_review_required` finding
  and `pixel_observer.py`'s identical pattern.

## Appendix: probe records (two most recent reconciliations)

These are the step-1/step-4 records for the axm-chat and axm-embodied
reconciliations, both run in the same live session on 2026-07-06. Included as
worked examples of the format step 5 asks for.

**axm-chat probe (2026-07-06).** Drove `axm_chat.spoke.import_export()` on a
minimal generic export. Result: sealed shard
`sh1_5e17a54b84c88f32a17bbec64b43f6b592f888d5417870015428549c3286f40f`,
namespace `chat/conversation`, publisher `@axm_chat`, claims `has_title` /
`message_count` / `has_turn`. Verified with
`axm-verify shard <dir> --trusted-key <out-of-band publisher.pub>` → status
PASS, exit 0. Live-proof: `ConversationObserver` run against this shard
returned verified PASS, 6 findings, all `input_refs` genesis-owned (no
producer metadata leaked into a finding ref).

**axm-embodied probe (2026-07-06).** Drove a real `FrameCaptureRecorder`
session with simulated sensor frames (labeled as sim throughout — 6 frames
observed, 4 kept, 1 declared trigger), then sealed the session via the real
`compile_frame_capsule`. Result: sealed shard
`sh1_207c54cbaef7f12247678cf7727a3f3f00589d5f6b0ff4e271e3729b45d7abb6`,
namespace `embodied/capture`, publisher `@axm_embodied`, evidence tier
`physical_capture`, judge-verified `ext/streams@1.jsonl` index published.
Detached `axm-verify` with an out-of-band key → PASS. Live-proof:
`PhysicalEvidenceObserver` run against this shard returned verified PASS,
trusted (tier confirmed, stream index agreed with the declared kept-frame
count), 4 bounded findings: `physical_evidence_available`,
`opaque_sensor_capture`, and `capture_continuity_available` all `PROVEN`
(the seal and the declared structure are verified); `human_review_required`
stays `UNTESTED` by construction — a flag, not a proven fact.

Both probe shards were sealed with throwaway keys in disposable work
directories that no longer exist. **The durable evidence is this written
record plus the re-runnable procedure in this document — not the probe
binaries themselves**, which were never meant to persist. A future
re-verification re-runs step 1 fresh; it does not go looking for these
specific `sh1_` directories.

## Durability rules for this document

This runbook is meant to still be correct and followable long after the
session and machine that wrote it are gone. Maintaining it (or writing the
next edge's reconciliation) means keeping these:

- No absolute filesystem paths tied to one machine or one session. Every path
  in this document is repo-relative (`src/ghostbox/interop/contracts.py`, not
  `/home/user/GhostBox/src/...`).
- No session-specific URLs (no links to a particular chat session, ticket, or
  ephemeral CI run). A commit hash or a file path outlives all of those.
- Pin tool/kernel versions by commit hash, not by a moving branch name or a
  "latest" assumption. Precedent: axm-genesis kernel pinned at `v1.0.0` @
  commit `9074e7fb2e9cedde692b248cdd0c6a805e77d8ac`. When the kernel a future
  reconciliation probes against has moved past this commit, that is worth
  noting explicitly in the new reconciliation's docstring/commit message —
  don't assume frozen CLI behavior (exit codes, flags) carried forward
  unchanged.
- Commands are written repo-relative and tool-name-relative (`axm-build
  compile ...`, `axm-verify shard <dir> --trusted-key <key>`), never as a
  literal path into someone's checkout.
- When adding a new precedent to this document (a fifth edge, a sixth), add
  it as a new example next to the existing ones — do not rewrite the
  existing precedents to match new terminology. The four-then-two precedents
  above are historical record; a later reader relies on them describing what
  actually happened, not what the vocabulary later evolved into.
