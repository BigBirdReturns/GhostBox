# Interoperable Foundry exit and DevCon 7 reproduction

**Status: runnable, bounded proof.** This is the public MIT-licensed recipe for replacing selected Foundry services while retaining an authorized data/contract exit. It extends the existing [GhostBox Ship of Theseus](SHIP_OF_THESEUS.md). That system seals nine migration surfaces at explicit FULL, CONTRACT, SOURCE or ATTESTED tiers. This addition executes a selected read interface after the vendor disappears.

**What interoperable means here:** the same documented Ontology API v2 read routes and JSON shapes can be sourced from Foundry or an independently held, integrity-checked capture. OSDK clients may still need authentication/client configuration changes. Writes, Functions, Actions, derived properties, authorization, Workshop, AIP applications, GPU scheduling and adapter hot-swap are not claimed compatible. Unsupported requests return errors. This is not a Palantir product.

## 1. Run everything on synthetic data

~~~bash
git clone https://github.com/BigBirdReturns/GhostBox.git
cd GhostBox
python -m foundry_exit.interop_acceptance --out ./new-cutover-proof
python -m foundry_exit.synthetic_devcon7 --out ./demo-traces
python -m foundry_exit.trace_curation \
  --otlp-json ./demo-traces/otlp.json \
  --reviews-jsonl ./demo-traces/reviews.jsonl \
  --out ./demo-curated
~~~

The acceptance script makes an **invented** Ontology v2 capture, serves it over authenticated HTTP loopback, compares paginated reads, primary-key lookup and type/link metadata, and rejects unauthenticated access or writes. Inspect the acceptance receipt. The curation script joins OTLP JSON trace IDs to *independent* reviewed outcomes, requires an explicit matching source/review classification and policy version, rejects unapproved or incorrect behavior, deduplicates and splits by business case (avoiding random-trace leakage). It emits train/eval JSONL plus source hashes and a manifest.

For optional GPU supervised fine-tuning, install the standard PyTorch, transformers and PEFT packages in an isolated CUDA environment (not needed for the stdlib-only exporter and bridge):

~~~bash
python -m foundry_exit.portable_sft \
  --train ./demo-curated/train.jsonl \
  --eval ./demo-curated/eval.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out ./demo-trained --max-steps 300
~~~

Outputs: compatible PEFT LoRA adapter and tokenizer, train/eval hashes, base-versus-adapter predictions and accuracy, wall time, GPU identity, peak allocated VRAM and training settings. The adapter still requires the exact base-model family and a separately deployed inference server. This is SFT, not Palantir's regional GPU hub or RL runtime.

## 2. Connect a tenant with permission to export

Use your own authorized Foundry HTTPS origin and an OAuth token with the documented api:ontologies-read scope. Select only explicitly approved types. Supply the token out of band, never in a repository:

~~~bash
export FOUNDRY_TOKEN='<your authorized scoped token>'
python -m foundry_exit.interop_bridge capture \
  --base-url https://YOUR-FOUNDRY-ORIGIN.example \
  --ontology your-ontology-api-name \
  --types Ticket,Decision \
  --out ./tenant-approved-capture
unset FOUNDRY_TOKEN
~~~

The exporter uses Palantir's published GET Ontology v2 paths: List Object Types, List Outgoing Link Types and List Objects. It follows all nextPageToken values, checks totalCount when provided, detects duplicate primary keys and pagination loops, and stores paginated JSON responses with SHA-256 hashes and a selected-type coverage manifest. It refuses overwrites. Dataset bytes are a *separate* plane handled by the existing GhostBox S3-compatible import (see [README](README.md)).

**Consistency boundary:** Palantir says List Objects does not guarantee consistency under concurrent writes. A completed page walk is not a transaction snapshot. A real migration must reconcile version, counts and changes, and use a supported alternative when the documented Object Storage V1 10,000-row listing cap applies. Do not infer tenant-wide coverage from a selected-type capture.

## 3. Seal, verify and serve independently

The new capture structure feeds the existing [AXM Genesis](https://github.com/BigBirdReturns/axm-genesis) exit path:

~~~bash
axm-exit ./tenant-approved-capture --out ./tenant-sealed
axm-verify shard ./tenant-sealed --trusted-key PATH_TO_TRUSTED_PUBLISHER_PUBLIC_KEY
~~~

Consult [FIRST_HOUR](FIRST_HOUR.md) for the Genesis install and detached-verification procedure. The bridge's SHA-256 manifest detects changes but is not a digital signature; Genesis custody is separate. Other planks remain available through axm-pipeline-exit, axm-logic-exit, axm-residual-exit and axm-exit-ship.

Serve the captured read interface on loopback with a separate local bearer secret:

~~~bash
export AXM_EXIT_LOCAL_TOKEN='<generate a long random token>'
python -m foundry_exit.interop_bridge serve \
  --capture ./tenant-approved-capture \
  --host 127.0.0.1 --port 8787

curl -H "Authorization: Bearer $AXM_EXIT_LOCAL_TOKEN" \
  "http://127.0.0.1:8787/api/v2/ontologies/your-ontology-api-name/objects/Ticket?pageSize=100"
~~~

The implementation handles these documented read routes: list object types, get one object type, list its outgoing link types, list objects and get an object by primary key. Pagination tokens are bound to the immutable capture manifest and authenticated. Unsupported requests and query semantics fail visibly, and writes return HTTP 501. The demonstration server has one local token and does not implement per-row policy, corporate SSO or TLS. **Do not expose it to any untrusted network or treat its authorization as equivalent to Foundry.**

## 4. Feed Palantir logs into the open training path

Foundry offers organization-level log export to streaming datasets with an OpenTelemetry protobuf option, subject to permissions and enrollment support. An authorized decoder or OpenTelemetry Collector converts that format to OTLP JSON. Then explicitly map the export's observed fields into the open adapter contract used by this sample:

| Span attribute | Purpose |
|---|---|
| axm.case_id | Stable business-case join key |
| axm.policy_version | Policy version in force at decision time |
| axm.classification | Approved sensitivity marking from the actual source |
| gen_ai.input.messages | Recorded model input |
| gen_ai.output.messages | Recorded model response |

Join to independently reviewed JSONL with trace_id, case_id, policy_version, classification, approved, expected_response and reviewer_id. These attribute names are an **adapter contract**, not a claim about every native Foundry log schema. The curation program consumes OTLP **JSON**, not protobuf directly.

Critical security fact: [Palantir's log-permissioning documentation](https://www.palantir.com/docs/foundry/aip-observability/log-permissioning) states that log markings are configured explicitly and do **not** automatically inherit every source-executor/input marking. Traces may contain sensitive prompts, object values and outputs. Review maximum reachable sensitivity, data-controller authorization, retention and training purpose. The curator defaults to PUBLIC only; unknown or mismatched marking excludes a span.

## 5. Actual capability ledger

| Product function | Independent mechanism | State |
|---|---|---|
| Ontology data and type metadata | API v2 capture → Genesis seal → local API | **Executable synthetic HTTP proof**, not live tenant |
| Dataset bytes | Foundry S3-compatible/REST → existing GhostBox import | Existing simulator proof; no live tenant |
| Trace curation | OTLP converter + reviewed outcomes + Cantos/Second Run records | **Executed** on synthetic traces |
| Model Studio SFT | PyTorch/PEFT, source hashes and independent eval | **Executed** on W03 RTX 8000 |
| Query/custody | AXM Spectra and Genesis | Existing verified components and receipts |
| GPU dispatch | TierDesk/Estate native jobs | Real scheduler, not Foundry GPU hub wire compatibility |
| LoRA inference deployment | Portable adapter into compatible model server | Adapter produced; server/hot-swap not yet qualified |
| Actions, Functions and Queries | Captured contracts and source, customer-owned executable replacements | **CONTRACT/SOURCE only**; no equivalent write engine |
| Workshop and AIP apps | Own source or rebuild from attested evidence | **No complete documented portable export** |
| Authorization | Customer-owned IAM and row/field policies | **Not ported**; explicit new security qualification required |

[Cantos/Second Run](https://github.com/BigBirdReturns/cantos) retains recorded procedures and dependency requalification, [Aperture](https://github.com/BigBirdReturns/aperture) qualifies local model configurations, and [AXM Core](https://github.com/BigBirdReturns/axm-core) supplies Spectra query. None is a mandated single-vendor dependency for preserving the exported data. No private API compatibility is assumed.

## 6. Acceptance and exit conditions

A real customer exits an operational surface only after the selection is complete and integrity-sealed; object/link and data consistency reconcile; every invoked action/function/agent step has an executed replacement; identity and row/field policy are enforced; current-policy business-outcome regressions pass; compute, review labor, ingress and serving costs are measured; and a cutover/backout rehearsal succeeds without the proprietary engine. Portability of source code and data alone does **not** prove application parity or instantaneous escape.

**Observed on November 13, 2026, synthetic Estate run:** 432 OTLP spans, 317 admitted, 253 training and 64 case-disjoint evaluation records. Qwen2.5-0.5B-Instruct on an RTX 8000 trained a 540,672-parameter rank-8 adapter in 25.531 seconds, used 2.63 GiB peak allocated GPU memory, and improved exact-match task accuracy from 16/64 to 64/64 on held-out unchanged-policy cases. These are NOT Palantir measurements or proof of real-world migration parity. Local hashes, metrics, and adapter remain in the Estate's scratch execution record.

Re-run targeted conformance:

~~~bash
python -m foundry_exit.interop_acceptance --out ./fresh-acceptance
python -m pytest -q tests/test_foundry_interop_bridge.py tests/test_foundry_trace_curation.py
~~~

## Primary documentation

- [List Objects, including consistency and 10,000-row constraints](https://www.palantir.com/docs/foundry/api/v2/ontologies-v2-resources/ontology-objects/list-objects)
- [Object type API](https://www.palantir.com/docs/foundry/api/ontologies-v2-resources/object-types/list-object-types)
- [Outgoing link type API](https://www.palantir.com/docs/foundry/api/ontologies-v2-resources/object-types/list-outgoing-link-types)
- [S3-compatible datasets](https://www.palantir.com/docs/foundry/data-integration/foundry-s3-api)
- [Organization log exports](https://www.palantir.com/docs/foundry/administration/configure-logging)
- [Ontology SDK](https://www.palantir.com/docs/foundry/ontology-sdk)
- [DevCon 7 presentation](https://www.youtube.com/watch?v=qxeQQFSyY6k)
