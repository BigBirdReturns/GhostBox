# Threat Geometry — Competitive Memo (demoted appendix)

Status: positioning memo, not a proof. Deliberately separated from
[`THREAT_GEOMETRY.md`](THREAT_GEOMETRY.md) so the competitive argument cannot
inflate the field theory. Read the paper first; this follows from it, it does
not lead it.

## Why this is a memo and not a section

The durable object is the field theory: an invariant attention-and-allocation
engine over sparse anomaly fields. A general theory survives a decade of vendor
churn. "We beat vendor X" survives one product cycle. So the competitive claim
is racked here, where it can be argued on its own terms and cannot borrow the
credibility of the paper's proven parts.

Everything below is **argument and positioning**, carrying no `proven` status.
The only `proven` artifact in this program is Field Zero (semantic tension
between filings and narrative), documented in the paper. This memo claims none
of that certainty.

## The actual distinction: post-hoc explanation vs in-flight redirection

Centralized ontology systems are built to **collect centrally and correlate
everything, then explain records after collection.** Their strength is the
after-action picture: pull every source into one ontology, and answer "what
happened and how does it all connect." That is a genuine strength and this memo
does not pretend otherwise.

Threat geometry is built for a different moment. It processes at the edge, fuses
weak signals across modes, detects deviation from local baseline *geometrically*,
and its output is **a redirection of scarce resources while the physical field is
still changing.** The question it answers is not "what happened" but "where is the
field deforming and where should the next asset go, now."

These are not the same product wearing different logos. They optimize different
objectives at different times in the event:

| | Centralized ontology | Threat geometry (level 2) |
|---|---|---|
| Primary moment | after collection | while the field deforms |
| Primitive | record → correlation | field → gradient → allocation |
| Output | explanation / link chart | deployment + coverage map |
| Where compute lives | central | edge, then fuse |
| Failure it fights | "we couldn't connect the dots afterward" | "we moved the asset too late" |
| Authority over truth | holds the canonical store | holds none — points at a record it never owns |

The last row is the real architectural fork, and it is the one enforced in code,
not just asserted: the AXM alignment forbids GhostBox from holding canonical
storage, minting receipts, or deciding truth (see
[`AXM_LEVEL_2_ALIGNMENT.md`](AXM_LEVEL_2_ALIGNMENT.md)). A system that must own the
central store to function is structurally a collect-then-correlate system. A
system that is forbidden from owning it is structurally an attention layer. That
is a design commitment, not a marketing adjective.

## What this memo does NOT claim

- It does **not** claim a shipped physical-threat capability. Every physical
  modality in the paper is simulated or untested.
- It does **not** claim centralized systems are bad at what they are for. They
  are good at post-hoc correlation; this is a different job.
- It does **not** claim a benchmark win. There is no head-to-head number here,
  and inventing one would violate the program's denominator discipline.

The honest competitive statement is narrow: *for the in-flight
resource-redirection problem on a deforming field, a centralized
collect-then-correlate architecture is solving a different problem, and an
edge-first attention layer that is forbidden from owning the record is the more
natural fit.* Whether that matters commercially is a market question this memo
does not resolve.

## The one line worth keeping

Centralized ontology explains the beach after the tide goes out. Threat geometry
watches the shoreline erode and moves the crew before the collapse. Both are
real. They are not the same job.
