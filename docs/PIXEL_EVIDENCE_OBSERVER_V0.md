# GhostBox Pixel Evidence Observer v0

A level-2 attention observer over **ScreenGhost Pixel Evidence v0** records. It
observes a genesis-sealed pixel-evidence shard **only after custody
verification**, emits bounded GhostBox-owned `AttentionFinding`s downstream of the
verified `shard_id`, and leaves the screenshot, manifest, and custody record owned
entirely outside GhostBox.

## Order (fixed, non-negotiable)

```
SealedShard (capture.png + pixel_capture_manifest.json)
   │  verify THROUGH the landed GenesisCustodySpoke   (no direct axm-verify)
   ▼
custody VERIFIED?  ── no (FAIL / MALFORMED / NO_TRUSTED_KEY) → no pixel findings; manifest never read
   │ yes
   ▼  read pixel_capture_manifest.json  (only now)
evidence_tier == pixel_capture?  ── no → untrusted finding only (unexpected_evidence_tier)
   │ yes
   ▼  manifest image hash == sealed PNG bytes?  ── no → untrusted finding only (pixel_manifest_hash_mismatch)
   │ yes
   ▼
GhostBox-owned AttentionFindings, downstream of shard_id:
   pixel_evidence_available · rendered_surface_capture · human_review_required
   (+ capture_context_available if a sidecar supplied context)
```

Verification is reused from the **landed** `GenesisCustodySpoke` — the observer
does not call `axm-verify`, does not import `subprocess`, and does not duplicate
genesis logic.

## Findings (bounded)

| Finding | Provenance | Meaning |
|---|---|---|
| `pixel_evidence_available` | `proven` | a verified `pixel_capture` record exists for this `shard_id` |
| `rendered_surface_capture` | `proven` | the evidence is the rendered surface (pixels only) |
| `human_review_required` | `untested` | attention flag: a human must review; GhostBox does not adjudicate |
| `capture_context_available` | `untested` | user/tool-supplied context present (url / page_title / …), **not verified** |

`proven` means the **seal** is verified — never that the screenshot content is
true. Every finding ships the non-assertions verbatim: **not** page truth, **not**
author identity, **not** platform authenticity, **not** legal provenance, **not**
OCR text, **not** the semantic content of the screenshot. Findings reference the
genesis `shard_id`, the image hash (`img:sha256:…`), and the manifest hash
(`manifest:sha256:…`) verbatim.

## Boundaries (enforced + tested)

- **Verify first, through the landed seam.** Custody owns verification; a non-PASS
  outcome yields no pixel findings and the manifest is never read.
- **Read only after VERIFIED.** Asserted: the content readers are never reached on
  a non-verified shard.
- **The PNG is never created, modified, OCR'd, classified, or rewritten.** It is
  read once, only to confirm its hash matches the manifest, and never retained.
- **No ScreenGhost import.** GhostBox consumes the sealed shard shape + manifest,
  never ScreenGhost as an authority (subprocess-isolated + source-checked).
- **No OCR, no image model, no DOM parser, no clipboard parser, no browser.**
  (`pytesseract` / `PIL` / `cv2` / `torch` / `selenium` / `playwright` /
  `xml.etree` / `lxml` never imported.)
- **No custody material retained.** `PixelObservationResult` holds only the
  `shard_id`, the custody verdict, the external hashes, and GhostBox's findings —
  no PNG bytes, no manifest body, no signature.
- **Tier + hash discipline.** A non-`pixel_capture` tier or a manifest/PNG hash
  mismatch emits an untrusted finding **only** and blocks trusted observation.

## Live receipts (this environment)

| Check | Result |
|---|---|
| Verified pixel shard → findings | **PASS** (4 bounded findings, `trusted=True`) |
| Wrong key | **blocked** — `verified=False`, no pixel findings (custody records `unverified_seal`) |
| Missing key | **blocked** — `NO_TRUSTED_KEY` via the existing no-anchor path |
| Malformed shard | **blocked** — `MALFORMED`/`FAIL`, no findings |
| Non-`pixel_capture` tier | **untrusted finding only** (`unexpected_evidence_tier`) |
| Manifest/PNG hash mismatch | **untrusted finding only** (`pixel_manifest_hash_mismatch`) |
| Findings reference `shard_id` + pixel hash | **verbatim** |
| PNG bytes unchanged by observation | **yes** |
| No ScreenGhost / OCR / DOM / clipboard / browser import | **yes** (subprocess-isolated) |
| Manifest read only after verification | **yes** |
| Detached record verifies without GhostBox | **PASS** (`axm-verify` on shard bytes + oob pub) |
| Test suite | **14/14** (repo 51/51) |

**Evidence tier of this slice:** observer-over-verified-bundle, proven against a
real genesis-sealed pixel shard built in-test with the genesis CLI. No OCR, no
vision intelligence — that can be a later observer layer once custody and tier
discipline are landed.

## Control question

Can GhostBox observe a verified ScreenGhost pixel evidence shard, emit bounded
attention findings, and still leave the screenshot, manifest, and custody record
owned entirely outside GhostBox?

**v0 answer: yes** — verified first through the landed custody seam, findings
strictly downstream of the genesis `shard_id`, the PNG read-once-and-never-kept,
and nothing about page truth, identity, or platform authenticity asserted.
