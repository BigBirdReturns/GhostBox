# GhostBox Pixel Review Layer v0

Human review, **strictly downstream** of the Pixel Evidence Observer. The
observer flags every trusted `pixel_capture` observation with
`human_review_required`; this layer is where a human answers that flag — and
GhostBox records the answer as its own review state, deciding nothing itself.

## Placement in the chain (fixed)

```
SealedShard
  → GenesisCustodySpoke            custody verification (landed seam)
  → PixelEvidenceObserver          tier + hash discipline, bounded findings
  → PixelReviewLayer               ← THIS: human attention disposition
```

The only input is a `PixelObservationResult`. The layer performs **no filesystem
access at all** — no shard read, no manifest read, no PNG read (asserted at
source level: no `Path`/`open`/`read_bytes` anywhere in the module). It cannot
touch the image even by accident.

## The decision vocabulary (attention-shaped, never truth-shaped)

| Disposition | Meaning |
|---|---|
| `escalate` | a human wants more eyes / downstream action |
| `dismiss` | a human saw it and stands down the attention |
| `needs_context` | a human cannot decide without more supplied context |

There is deliberately no `authentic`, no `true`, no `verified_content`: a review
**moves attention**, it does not adjudicate the screenshot's meaning. Truth-shaped
verdicts are not representable and are refused (`invalid_disposition`).

## The record

`PixelReviewRecord` carries: content-addressed `review_id` (`rev:…`), the genesis
`shard_id` **verbatim**, the image/manifest hashes **verbatim**, the
`human_review_required` finding it answers, the disposition, the required human
`reviewer` attribution, the reviewer-attributed `note` (never a GhostBox
assertion), and the `evidence_tier` **copied verbatim and frozen** — a review can
never upgrade `pixel_capture` to DOM truth, API truth, platform authenticity, or
legal provenance. Provenance is `UNTESTED`: a human opinion on file, not a proven
fact. The non-assertions ship inside every record.

## Rules (enforced + tested)

- **Only trusted observations are reviewable.** `verified` + `trusted` +
  `pixel_capture` tier all re-checked here (defense in depth); anything else is
  refused with a stable reason (`not_verified` / `not_trusted` / `wrong_tier` /
  `no_review_finding`).
- **Human attribution required.** Empty reviewer → refused; nothing recorded.
- **Append-only history.** Re-review adds a record; nothing is edited or
  replaced.
- **Deterministic ids.** Same decision inputs → same `review_id`.
- **No ScreenGhost / OCR / image model / DOM / browser import** (subprocess-
  isolated) and no `subprocess` in the layer itself.

## Live receipts (this environment)

| Check | Result |
|---|---|
| Trusted observation admitted → pending | **yes** |
| Unverified / untrusted / wrong-tier / no-finding | **refused**, never queued |
| Review record refs `shard_id` + hashes + finding | **verbatim** |
| Evidence tier verbatim + immutable (frozen, no API to change) | **yes** |
| Truth-shaped verdicts (`authentic`, `true`, …) | **not representable — refused** |
| Empty reviewer | **refused**, nothing recorded |
| History append-only; ids deterministic | **yes** |
| No filesystem access in the layer | **yes** (source-asserted) |
| End-to-end: seal → custody → observe → review | **PASS** (PNG untouched; detached verify still exit 0) |
| Test suite | **16/16** (repo 67/67 at the time of this slice; the current total is tracked by CI and docs/EDGE_RECONCILIATION_RUNBOOK.md) |

**Evidence tier of this slice:** review-workflow-over-verified-observation, proven
against a real genesis-sealed pixel shard end to end. No OCR, no vision models, no
browser automation — unchanged from the observer's bans.

## Control question

Can a human record a bounded attention disposition over a trusted pixel
observation, while nothing in the review layer can touch the PNG, restate the
evidence tier, or turn a review into a truth claim?

**v0 answer: yes** — downstream-only input, tier frozen verbatim, attention-shaped
vocabulary, human-attributed and append-only.
