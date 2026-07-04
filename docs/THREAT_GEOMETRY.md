# Threat Geometry

**Distributed Threat Intelligence as Resource Allocation Over Sparse Anomaly Fields**

Status: field-theory paper. Draft for audit confirmation.
Layer: this is the theory of GhostBox as the level-2 attention layer. It sits
above [`AXM_LEVEL_2_ALIGNMENT.md`](AXM_LEVEL_2_ALIGNMENT.md) and is grounded in
the boundary objects defined in [`INTEROP_CONTRACTS.md`](INTEROP_CONTRACTS.md).

---

## 0. The one-sentence version

> Do not search the beach for one suspicious grain of sand.
> Model the shoreline. Watch the topology. Detect erosion.
> Move resources before the collapse.

Detection tells you a point changed. Threat geometry tells you *how the field is
deforming*, *what hypothesis explains the deformation*, and *where scarce
resources should move next*. The first is a classification problem on a sample.
The second is a topology-and-allocation problem on a field. This paper argues
the second is the durable object, and that the category labels we usually attach
to it — "drone detection," "crowd monitoring," "border security," "wildfire
watch," "financial narrative divergence" — are surface names for one problem.

## 1. The claim, stated so it can be argued with

There is an invariant architecture underneath distributed threat intelligence.
The **modality** changes. The **signature library** changes. The **fusion,
hypothesis tracking, resource optimization, and feedback loop do not.**

If that is true, then a system built correctly for one modality is not a
one-off. It is an instance of a substrate-invariant engine, and moving to a new
modality is a matter of swapping the sensor front-end and the signature library,
not rebuilding the intelligence layer.

This is falsifiable. It is false if any of the following hold:

- The layers below (baseline, anomaly, hypothesis, allocation, feedback) require
  modality-specific logic that cannot be factored out of the sensor front-end.
- Two modalities that this paper claims share the engine turn out to need
  incompatible hypothesis representations or incompatible objective functions.
- The "field deformation" framing produces no decision a per-alert detector
  would have missed — i.e., the topology buys nothing over the point classifier.

The paper's job is to state the invariant precisely enough that these tests can
be run, and to be honest about which instances are **proven**, which are
**simulated**, and which are **untested**.

## 2. Why "geometry": the five invariant properties of a threat field

A threat is not a labeled object sitting in a bin. It is a deformation in a
physical or informational field that has five properties, and these five are
what make the problem geometric rather than categorical.

1. **Spatial distribution.** Threats occupy and move through space. A launch
   site, a crowd density gradient, a border segment, a fire front, a convoy
   corridor — each is a region, not a pixel. The signal is the shape of the
   region and how it moves.
2. **Temporal precursors.** Threats emerge over time and emit weak signals
   *before* the event. The pre-launch RF chatter, the pre-crush density
   buildup, the pre-intrusion pattern-of-life change, the pre-ignition fuel and
   wind state. Precursors are the whole point: attention that arrives at the
   event is late.
3. **Multimodal signatures.** The precursor is rarely loud in one channel. It is
   a faint correlated deflection across RF, acoustic, thermal, visual, radar,
   and environmental channels at once. Any single channel, thresholded alone,
   either misses it or drowns in false positives.
4. **Base-rate sparsity.** Real threats are rare against an enormous, noisy,
   mostly-benign background. This is the property that kills naive detectors:
   at a low base rate, a detector with a "good" false-positive rate still buries
   the operator in false alarms. Sparsity forces you to reason about the field,
   not the sample.
5. **Resource constraint.** You cannot cover everything. Officers, interceptors,
   fire crews, analyst attention, and sensor dwell time are all finite. The
   output of the system is therefore not a label — it is *a placement of scarce
   resources against a deforming field*.

These five are why the correct primitive is a **field with a baseline and a
gradient**, and why the correct output is **an allocation**, not an alert. Every
subsequent design choice follows from taking these five seriously.

## 3. The substrate-invariant architecture

Six layers. The first and last are modality-specific at their edges. The middle
four are the invariant engine.

```
                 modality-specific edge
   ┌──────────────────────────────────────────────┐
   │  1. Sensor abstraction                        │  RF / acoustic / thermal /
   │     raw readings  →  SignatureEvent           │  visual / radar / env /
   └──────────────────────────────────────────────┘  screen / filing / feed
                          │
   ┌──────────────────────┼───────────────────────┐
   │  2. Baseline field   ▼                        │
   │     learn "normal" per (location, time,       │   ── invariant engine ──
   │     season) cell                              │   the same code regardless
   ├──────────────────────────────────────────────┤   of what feeds layer 1
   │  3. Anomaly gradient                          │
   │     deviation of a cell from its own baseline │
   ├──────────────────────────────────────────────┤
   │  4. Hypothesis topology                       │
   │     correlate weak anomalies across sensors   │
   │     and time into emerging threat hypotheses  │
   ├──────────────────────────────────────────────┤
   │  5. Resource geometry                         │
   │     place limited assets to minimize expected │
   │     damage under hypotheses + constraints     │
   └──────────────────────┬───────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────┐
   │  6. Feedback / erosion model                  │  learn from TP / FP / miss /
   │     update baselines and priors from outcomes │  response outcome
   └──────────────────────────────────────────────┘
```

### Layer 1 — Sensor abstraction

Every modality converts its raw reading into a common **signature event**: a
timestamped, located, typed observation with a provenance state. RF spectra,
acoustic spectrograms, thermal frames, view-trees, SEC filings, and RSS items
all reduce to the same shape at this seam. This is the *only* layer that knows
what a drone or a filing is. In the AXM contract this event is one of:

- `EvidenceEvent` — a surface observation from ScreenGhost (the informational
  substrate; *proven* today).
- `PhysicalEvidenceEvent` — an event-triggered high-resolution physical capture
  from axm-embodied (the physical substrate; *untested* today).

Both carry an explicit `provenance` (`proven` / `simulated` / `frozen` /
`untested`). A signature event never asserts that a threat is real. It asserts
only that a channel deflected and that the capture of that deflection is
faithful. The distinction between *the capture is trustworthy* and *the content
is true* is load-bearing and is preserved at every layer above.

### Layer 2 — Baseline field

Normal is not global; it is local. Traffic at a border crossing at 3am in
February is a different "normal" than noon in July. The baseline is learned per
spatiotemporal cell — `(location, time-of-day, season)` — so that anomaly is
always measured against the cell's *own* history, not a global average that
would flag every night shift and miss every quiet anomaly.

This is the layer that makes sparsity tractable. You are not asking "is this
reading unusual for the world," you are asking "is this reading unusual *here,
now, for this cell*." A weak signal that is invisible against the global
distribution can be a large local deviation.

### Layer 3 — Anomaly gradient

Anomaly is the deviation of a cell from its own baseline, and — crucially — the
**gradient** of that deviation across neighboring cells and successive times.
A single high-anomaly cell is a point. A gradient — anomaly rising along a
corridor, a density front advancing, divergence widening between two sources —
is field deformation. This is the "watch the erosion" layer. It is where the
grain-of-sand framing is abandoned in favor of the shoreline.

### Layer 4 — Hypothesis topology

Weak anomalies across different sensors and times are correlated into
**hypotheses**: structured, evolving explanations of *what deformation is
underway*. A faint RF deflection here, an acoustic anomaly there, a thermal
gradient upwind — none decisive alone — are fused into "possible pre-launch
activity in sector 7, confidence rising." Hypotheses are spawned, strengthened,
weakened, split (forked when the evidence stops cohering), and retired. This is
a tracker over *explanations of the field*, not over objects.

In the AXM contract, the output of this layer is the `AttentionFinding`: a
`tension_type`, a `score`, the `input_refs` that produced it, and the `claims`
it surfaces. It says *where the tension is*. It explicitly does **not** decide
whether the surfaced claim is true — that is the claim harness's job
(`ClaimCheckResult`, verdict ∈ {supported, contradicted, frozen, untested}).
Attention points at the field. It does not adjudicate it.

### Layer 5 — Resource geometry

This is where the beach metaphor becomes doctrine. The objective is **not**
"detect more things." It is: given the active hypotheses, the protected assets,
the travel and equipment constraints, and the expected damage under each
hypothesis, *place limited defensive assets to minimize expected loss.* The
output is a deployment — officers, interceptors, fire crews, sensor dwell,
analyst attention — plus a coverage map showing what is now exposed.

The same allocator drives every modality because the objective is written over
hypotheses and constraints, not over drones or filings. Change the asset type
and the damage model; the optimization is the same shape.

### Layer 6 — Feedback / erosion model

Outcomes flow back. True positive, false positive, miss, and response outcome
each update the baselines (layer 2) and the hypothesis priors (layer 4). A miss
is not just a bad day — it is a correction signal about where the baseline was
wrong or the fusion was too conservative. "Erosion" is literal here: the field
model is continuously eroded and re-deposited by contact with reality, which is
exactly the discipline that keeps a sparse-signal system from drifting into
either paranoia or complacency.

## 4. What is actually built, and on which substrate

Honesty is the whole game in a sparse-signal domain, so this section is explicit
about what exists versus what is projected. No category borrows trust from
another.

GhostBox today implements the invariant engine (layers 2–5, with the feedback
loop partial) over an **informational / semantic** substrate:

| Layer | Invariant role | GhostBox implementation today | Status |
|---|---|---|---|
| 1 | Sensor abstraction | ScreenGhost photonic + Axiom-KG adapters (RSS, XBRL, iCal, schema.org, OpenAPI, …) → `EvidenceEvent` / Node | **proven** (informational) |
| 2 | Baseline field | Semantic coordinate space (Axiom-KG `SemanticID`, `Space`); "normal" = where comparable claims cluster | **proven** (informational) |
| 3 | Anomaly gradient | `SemanticTension`: spread, source divergence, semantic drift as distance in coordinate space | **proven** (informational) |
| 4 | Hypothesis topology | `AttentionGeometry` shapes — contradiction, velocity, convergence — and forks; emitted as `AttentionFinding` | **proven** (informational) |
| 5 | Resource geometry | `where_to_look(top_n)` / `AttentionMap.top()` — allocation of the scarcest asset, *analyst attention* | **partial** — ranks attention; does not yet optimize multi-asset placement under travel/equipment constraints |
| 6 | Feedback / erosion | Session continuity + alert thresholds; outcome-driven baseline update | **untested** — the loop exists structurally; outcome learning is not yet closed |

**Field Zero** is the first deployed instance of this engine: PLTR financial
analysis. SEC XBRL filings are the baseline (financial ground truth); Google
News RSS is the narrative field; `SemanticTension` computes the divergence
between them; the output is a divergence score and a posture. This is a real,
running instance of "watch the field deform" on an informational substrate —
narrative eroding away from filings — and it is the honest proof that the engine
works at all. It is **not** proof that any physical modality works.

The **physical** substrate — drone launch, crowd crush, border intrusion,
wildfire, convoy ambush — is the projection of this same engine onto sensor data
carried by `PhysicalEvidenceEvent` from axm-embodied. As of this paper, every
physical modality is **simulated or untested**. The engine is shared by design;
the sensor front-ends and signature libraries are not built. Claiming otherwise
would be exactly the grain-of-sand-to-shoreline overreach this paper is trying
to make disciplined.

The single durable claim, then, is narrow and defensible: *the middle four
layers are substrate-invariant, demonstrated on an informational substrate, and
the physical instances are future modalities behind the same seam — not shipped
capability.*

## 5. Where this sits in the AXM layering

Threat Geometry is the theory of **level 2**. The layering is fixed by
[`AXM_LEVEL_2_ALIGNMENT.md`](AXM_LEVEL_2_ALIGNMENT.md):

- **Level 1 — record and knowledge.** What happened, what can be compiled,
  what can be sealed. ScreenGhost (`EvidenceEvent`), axm-embodied
  (`PhysicalEvidenceEvent`), axm-core (`KnowledgeShardRef`), axm-chat
  (`ConversationShardRef`), axm-genesis (`ShardReceipt`, the only sealer).
- **Level 2 — attention.** *Where is the tension.* GhostBox ingests level-1
  objects and emits `AttentionFinding`. It reads the record; it does not become
  the record.
- **Testing — reality harness.** axm-capability-claim-test consumes the claims a
  finding surfaces and returns `ClaimCheckResult`. Attention proposes; the harness
  disposes.

The load-bearing invariant: **GhostBox is the attention layer, not the storage
layer.** The moment attention starts holding the canonical copy, minting its own
receipts, or deciding what is true, the layering has failed and the honesty
guarantees collapse with it. Threat geometry is a theory of *where to look and
where to move resources*, computed over a record it never owns.

## 6. The four states, carried to the edge

Every object at every boundary declares one of four provenance states —
**proven, simulated, frozen, untested** — and a claim, once tested, carries a
parallel verdict — **supported, contradicted, frozen, untested**. These are
different fields and must never be collapsed. In a threat-geometry system this
is not bureaucracy; it is the difference between a defensible intelligence
product and a hallucinated one:

- A `proven` capture of a physical event means *the recording is faithful*, not
  that a threat is present.
- A `simulated` field exercise (a modeled crowd, a synthetic RF stream) is never
  promoted to record, no matter how realistic.
- A hypothesis is an `AttentionFinding`, always `untested` until the claim
  harness rules; a resource is never moved on the strength of attention alone
  without that state being explicit to the operator.

This is the same discipline ScreenGhost already enforces on the informational
substrate ("no category borrows trust"; synthetic ≠ device ≠ real-app ≠
business) carried up into the physical one.

## 7. What follows for the operator

The operator does not want every grain of sand. The operator wants:

1. the **shoreline** — the current field and its baseline,
2. the **erosion pattern** — where and how fast the field is deforming,
3. the **exposed assets** — what the deformation threatens, and
4. the **next best placement** — where to move limited resources now.

A drone launch, a crowd crush, a border intrusion, a wildfire, and a convoy
ambush are different signatures on the same substrate: sparse anomalies
propagating through constrained physical space. The engine that serves the
operator is indifferent to which one it is. That indifference is the thesis.

## 8. Scope, non-goals, and the competitive question

This is a theory paper. It does not ship a physical sensor stack, and it does not
claim one. It defines an invariant and shows one honest, running instance of it.

The competitive implication — that centralized-ontology systems are built to
explain records *after* collection while this architecture redirects attention
and resources *while the field is still changing* — is real but secondary, and
is deliberately demoted to a separate memo so it cannot inflate the paper:
[`THREAT_GEOMETRY_COMPETITIVE_MEMO.md`](THREAT_GEOMETRY_COMPETITIVE_MEMO.md).
The durable object is the field theory. The competitive story follows from it;
it does not lead.

---

*ScreenGhost was the intake fight. GhostBox is the attention layer. Threat
Geometry is the field theory.*
