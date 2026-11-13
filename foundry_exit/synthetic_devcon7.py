"""Reproducible, wholly synthetic enterprise AIP/OTel training demonstration.

No patient, employee, client, Estate or Foundry production data is used.
Generates an OTLP JSON trace payload, independently reviewed outcomes JSONL,
and two deliberately different operational policy versions for drift testing.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

SITES=("CINDER","HARBOR","PALISADE","ORCHARD")
ISSUES=("D17","F22","Q09")
TEAMS=("ENGINEERING","SAFETY","PROCUREMENT","SCHEDULING")
NOTES=(
    "Field operator requested review",
    "Previous work item was reopened",
    "Supervisor requested clarification",
    "Original request has a new attachment",
    "Queue reference was updated",
    "The work package changed state",
)
SEED=20261113


def make(out):
    out=Path(out)
    if out.exists():
        raise ValueError("Refusing to replace existing synthetic evidence")
    out.mkdir(parents=True)
    rng=random.Random(SEED)
    labels=list(TEAMS)*3
    rng.shuffle(labels)
    pairs=[(s,c) for s in SITES for c in ISSUES]
    policy1=dict(zip((s+":"+c for s,c in pairs),labels))
    policy2=dict(policy1)
    for i,(site,code) in enumerate(pairs):
        key=site+":"+code
        if i%2==0:
            policy2[key]=TEAMS[(TEAMS.index(policy1[key])+1)%4]

    spans=[]; reviews=[]
    for pair_idx,(site,code) in enumerate(pairs):
        correct=policy1[site+":"+code]
        for n in range(36):
            ticket=f"{site}-{code}-{n:03}"
            note=NOTES[(n+pair_idx)%len(NOTES)]
            prompt=(f"Route this enterprise operations ticket. Site: {site}. "
                    f"Issue code: {code}. Note: {note}. "
                    "Reply with exactly one team tag: ENGINEERING, SAFETY, PROCUREMENT, or SCHEDULING.")
            accepted=rng.random()>0.15
            actual=correct if accepted else rng.choice([x for x in TEAMS if x!=correct])
            marking="RESTRICTED" if rng.random()<0.11 else "PUBLIC"
            tid=hashlib.sha256(("synthetic|"+ticket).encode()).hexdigest()[:32]
            sid=hashlib.sha256(("span|"+ticket).encode()).hexdigest()[:16]
            attrs={
              "axm.case_id":ticket,
              "axm.policy_version":"v1",
              "axm.classification":marking,
              "gen_ai.input.messages":prompt,
              "gen_ai.output.messages":actual
            }
            span={
                "traceId":tid,"spanId":sid,"name":"synthetic.route-ticket",
                "startTimeUnixNano":str(1793500000_000000000+n+pair_idx*100),
                "attributes":[{"key":k,"value":{"stringValue":v}} for k,v in attrs.items()],
            }
            spans.append(span)
            reviews.append({
                "trace_id":tid,"case_id":ticket,
                "expected_response":correct,"policy_version":"v1",
                "reviewer_id":"independent-synthetic-oracle","approved":accepted,
                "classification":marking,
            })
    doc={"resourceSpans":[{"resource":{"attributes":[{"key":"service.name",
           "value":{"stringValue":"synthetic-devcon7-fixture"}}]},
           "scopeSpans":[{"scope":{"name":"ghostbox.synthetic"},
                          "spans":spans}]}]}
    files={
      "otlp.json":json.dumps(doc,indent=2),
      "reviews.jsonl":"\n".join(json.dumps(r,sort_keys=True) for r in reviews)+"\n",
      "policy-v1.json":json.dumps(policy1,sort_keys=True,indent=2),
      "policy-v2.json":json.dumps(policy2,sort_keys=True,indent=2),
    }
    for name, contents in files.items():
        (out/name).write_text(contents+"\n" if not contents.endswith("\n") else contents,encoding="utf-8")
    receipt={
       "source":"fully synthetic, generated from seed; NOT a Foundry tenant",
       "seed":SEED,"spans":len(spans),"reviews":len(reviews),
       "policy_keys":len(policy1),"changed_policy_keys":sum(policy1[k]!=policy2[k] for k in policy1),
       "files":{k:hashlib.sha256((out/k).read_bytes()).hexdigest() for k in files},
    }
    (out/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    make(a.out)
