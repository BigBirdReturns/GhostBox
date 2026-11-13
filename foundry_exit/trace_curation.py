"""Curate independently reviewed OpenTelemetry trace data for portable SFT.

Input: OTLP JSON (OpenTelemetry ExportTraceServiceRequest mapping), plus
reviewed independent outcomes (JSONL). Source OTel protobuf must first be
decoded by an authorized OTel transform or Collector. This utility never
queries private Palantir trace endpoints or guesses labels from model output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timezone


class CurationError(ValueError):
    pass


def _attr(attrs):
    out = {}
    for a in attrs:
        if not isinstance(a, dict) or not isinstance(a.get("key"), str):
            raise CurationError("Invalid OTLP attribute")
        v = a.get("value") or {}
        if not isinstance(v, dict):
            raise CurationError("Invalid OTLP attribute value")
        allowed = [k for k in ("stringValue", "intValue", "boolValue", "doubleValue") if k in v]
        if len(allowed) != 1:
            continue
        out[a["key"]] = str(v[allowed[0]]) if allowed[0] == "intValue" else v[allowed[0]]
    return out


def flatten_otlp(doc):
    """Yield individual spans; no synthetic prompt/output or permissions."""
    if not isinstance(doc, dict) or not isinstance(doc.get("resourceSpans"), list):
        raise CurationError("Expected OTLP JSON with resourceSpans[]")
    for res in doc["resourceSpans"]:
        for scope in res.get("scopeSpans", []):
            for span in scope.get("spans", []):
                if isinstance(span, dict):
                    yield span


def _read_reviews(path):
    records = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise CurationError(f"Review line {number}: invalid JSON") from e
        tid = r.get("trace_id")
        if not isinstance(tid, str) or len(tid) != 32 or any(x not in "0123456789abcdef" for x in tid.lower()):
            raise CurationError(f"Review line {number}: invalid OTLP trace id")
        if tid in records:
            raise CurationError("Duplicate review trace id; ambiguous independent judgment")
        if any(not r.get(k) for k in ("case_id","expected_response","policy_version","reviewer_id","classification")):
            raise CurationError(f"Review line {number}: missing independent judgment field")
        records[tid] = r
    return records


def curate(traces_file: Path, reviews_file: Path, out_dir: Path,
           *, allowed_classifications=("PUBLIC",), holdout_fraction=0.20):
    """Write deterministic train and held-out JSONL, plus source hashes.

    Review admission demands independently recorded expected behavior and
    explicit matching source/review marking, policy version and case id.
    Group split by case id, never random trace row.
    """
    if not 0 < holdout_fraction < 1:
        raise CurationError("holdout_fraction must be strictly between 0 and 1")
    if out_dir.exists():
        raise CurationError("Output exists; refusing to overwrite a prior qualification")
    source_bytes = traces_file.read_bytes()
    review_bytes = reviews_file.read_bytes()
    doc = json.loads(source_bytes)
    reviews = _read_reviews(reviews_file)
    reasons = {}
    seen = set()
    seen_decisions = set()
    accepted = []
    span_count = 0

    def rejected(why):
        reasons[why] = reasons.get(why, 0) + 1

    for span in flatten_otlp(doc):
        span_count += 1
        tid = span.get("traceId")
        if not isinstance(tid,str) or len(tid)!=32 or any(x not in "0123456789abcdef" for x in tid.lower()):
            rejected("invalid_trace_id"); continue
        if tid in seen:
            rejected("duplicate_trace_id"); continue
        seen.add(tid)
        review = reviews.get(tid)
        if review is None:
            rejected("no_independent_review"); continue
        if review.get("approved") is not True:
            rejected("outcome_not_approved"); continue
        attrs = _attr(span.get("attributes", []))
        classification = attrs.get("axm.classification")
        if classification is None or classification != review["classification"]:
            rejected("missing_or_mismatched_classification"); continue
        if classification not in allowed_classifications:
            rejected("classification_not_authorized"); continue
        if attrs.get("axm.case_id") != review["case_id"] or attrs.get("axm.policy_version") != review["policy_version"]:
            rejected("case_or_policy_version_disagreement"); continue
        prompt = attrs.get("gen_ai.input.messages")
        output = attrs.get("gen_ai.output.messages")
        if not isinstance(prompt,str) or not prompt.strip() or not isinstance(output,str) or not output.strip():
            rejected("missing_prompt_or_output"); continue
        if output.strip() != review["expected_response"].strip():
            rejected("incorrect_behavior"); continue
        start = span.get("startTimeUnixNano")
        if not str(start).isdigit():
            rejected("missing_start_time"); continue
        decision_key=(review["case_id"],review["policy_version"],prompt.strip(),review["expected_response"].strip())
        if decision_key in seen_decisions:
            rejected("duplicate_reviewed_decision"); continue
        seen_decisions.add(decision_key)
        trace_sha = hashlib.sha256(json.dumps(span,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
        accepted.append({
            "trace_id":tid,
            "case_id":review["case_id"],
            "policy_version":review["policy_version"],
            "reviewer_id":review["reviewer_id"],
            "classification":classification,
            "source_sha256":trace_sha,
            "source_start_nanos":str(start),
            "messages":[{"role":"user","content":prompt},{"role":"assistant","content":review["expected_response"]}],
        })
    accepted.sort(key=lambda r:(r["case_id"],r["trace_id"]))
    train, evals = [], []
    for r in accepted:
        h=int(hashlib.sha256(r["case_id"].encode()).hexdigest()[:16],16)/2**64
        (evals if h < holdout_fraction else train).append(r)
    manifest = {
        "format":"axm-otel-curation/1","source":"otlp-json-plus-independent-reviewed-outcomes",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "input_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "reviews_sha256":hashlib.sha256(review_bytes).hexdigest(),
        "total_spans":span_count,
        "review_count":len(reviews),
        "accepted":len(accepted),
        "rejected":reasons,
        "train_cases":len({r["case_id"] for r in train}),"eval_cases":len({r["case_id"] for r in evals}),
        "train_records":len(train),"eval_records":len(evals),
        "split":"sha256(case_id) group holdout; no trace-level leakage",
        "classifications_allowed":list(allowed_classifications),
        "limitations":[
            "Training rows only after independent review and matching source marking.",
            "OTel log markings may require explicit administrative controls at capture source.",
            "No source transaction snapshot guarantee or model quality claim.",
            "The consumer must enforce source-equivalent access policies for any nonpublic training data.",
        ],
    }
    if not train or not evals:
        raise CurationError("Missing nonempty train or eval partition; adjust source scope/holdout with a new run")
    if {x["case_id"] for x in train} & {x["case_id"] for x in evals}:
        raise CurationError("Case leakage across split")
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=".otel-curation-",dir=str(out_dir.parent)))
    try:
        for name, rows in (("train.jsonl",train),("eval.jsonl",evals)):
            with (tmp/name).open("w",encoding="utf-8") as fd:
                for row in rows:
                    fd.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
        for file in ("train.jsonl","eval.jsonl"):
            manifest[file+"_sha256"]=hashlib.sha256((tmp/file).read_bytes()).hexdigest()
        (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        tmp.rename(out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description="Independent OTel trace curation, no proprietary Foundry runtime")
    p.add_argument("--otlp-json",type=Path,required=True)
    p.add_argument("--reviews-jsonl",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allowed-classification",action="append",default=["PUBLIC"])
    p.add_argument("--holdout",type=float,default=0.2)
    a=p.parse_args(argv)
    try:
        result=curate(a.otlp_json,a.reviews_jsonl,a.out,
           allowed_classifications=tuple(set(a.allowed_classification)),holdout_fraction=a.holdout)
    except (OSError,ValueError,KeyError) as e:
        print(json.dumps({"error":type(e).__name__,"message":str(e)}))
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
