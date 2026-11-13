"""Test the OTel trace->independent review->portable training dataset boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from foundry_exit.trace_curation import curate, CurationError


def attr(key, value):
    return {"key": key, "value": {"stringValue": value}}


def fixture():
    spans, reviews = [], []
    for i in range(40):
        tid=f"{i+1:032x}"
        klass="RESTRICTED" if i%11==0 else "PUBLIC"
        case=f"case-{i//2}"
        output="approved" if i%7 else "incorrect"
        span={"traceId":tid,"spanId":f"{i+1:016x}","startTimeUnixNano":str(1794590000_000000000+i),
              "attributes":[attr("axm.case_id",case),attr("axm.policy_version","v1"),
                attr("axm.classification",klass),
                attr("gen_ai.input.messages",f"Decide case {case}"),
                attr("gen_ai.output.messages",output)]}
        spans.append(span)
        reviews.append({"trace_id":tid,"case_id":case,"policy_version":"v1","reviewer_id":"reviewer-1",
                        "expected_response":"approved","approved": i%13 != 0,"classification":klass})
    # Missing review, unknown marking and duplicate trace should remain excluded.
    spans.append({"traceId":f"{999:032x}","spanId":"0123456789abcdef",
                  "startTimeUnixNano":"1794590000000000111",
                  "attributes":[attr("axm.classification","PUBLIC")]})
    spans.append(spans[-2])
    doc={"resourceSpans":[{"resource":{"attributes":[]},"scopeSpans":[{"scope":{"name":"axm"},
          "spans":spans}]}]}
    return doc,reviews


class TraceCurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        doc,reviews=fixture()
        self.source=self.root/"otel.json"
        self.judgments=self.root/"reviews.jsonl"
        self.source.write_text(json.dumps(doc))
        self.judgments.write_text("\n".join(json.dumps(x) for x in reviews)+"\n")
        self.out=self.root/"out"

    def test_synthetic_curation_acceptance(self):
        m=curate(self.source,self.judgments,self.out)
        train=[json.loads(x) for x in (self.out/"train.jsonl").read_text().splitlines()]
        evalu=[json.loads(x) for x in (self.out/"eval.jsonl").read_text().splitlines()]
        self.assertGreater(len(train),0)
        self.assertGreater(len(evalu),0)
        self.assertEqual(m["accepted"],len(train)+len(evalu))
        self.assertFalse({x["case_id"] for x in train}&{x["case_id"] for x in evalu})
        self.assertTrue(all(r["classification"]=="PUBLIC" for r in train+evalu))
        self.assertEqual(m["input_sha256"],hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertIn("classification_not_authorized",m["rejected"])
        self.assertIn("incorrect_behavior",m["rejected"])
        self.assertIn("outcome_not_approved",m["rejected"])
        self.assertIn("no_independent_review",m["rejected"])
        self.assertIn("duplicate_trace_id",m["rejected"])
        self.assertTrue(all("source_sha256" in r and "reviewer_id" in r for r in train+evalu))

    def test_rejects_missing_classification_even_if_review_approved(self):
        doc,reviews=fixture()
        first=doc["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
        first["attributes"]=[a for a in first["attributes"] if a["key"]!="axm.classification"]
        self.source.write_text(json.dumps(doc))
        m=curate(self.source,self.judgments,self.out)
        self.assertGreater(m["rejected"]["missing_or_mismatched_classification"],0)

    def test_fails_closed_on_duplicate_review(self):
        text=self.judgments.read_text()
        self.judgments.write_text(text+text.splitlines()[0]+"\n")
        with self.assertRaises(CurationError):
            curate(self.source,self.judgments,self.out)
        self.assertFalse(self.out.exists())

    def test_refuses_to_overwrite(self):
        curate(self.source,self.judgments,self.out)
        with self.assertRaises(CurationError):
            curate(self.source,self.judgments,self.out)


if __name__=="__main__":
    unittest.main()
