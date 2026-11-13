"""End-to-end synthetic provider-switch acceptance test for bounded Ontology reads.

Authenticates to the local HTTP bridge, reads the documented Foundry v2
wire shapes from a synthetic source and from the replacement, checks
page-by-page content parity, and verifies that writes fail closed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .interop_bridge import CapturedOntology, capture_ontology, make_handler
from .synthetic_ontology import fixture_fetch


def prove(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    capture_ontology(fixture_fetch,"synthetic",out/"capture",["Ticket","Decision"])
    snap=CapturedOntology(out/"capture")
    secret="synthetic-acceptance-token-20261113"
    server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(snap,secret))
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    def get(path):
        req=Request(f"http://127.0.0.1:{server.server_port}{path}",
                    headers={"Authorization":"Bearer "+secret})
        with urlopen(req,timeout=5) as r:
            return json.load(r)
    tested=[]
    try:
        path="/api/v2/ontologies/synthetic/objects/Ticket"
        rows=[]
        page=get(path+"?pageSize=3")
        while True:
            rows.extend(page["data"])
            token=page.get("nextPageToken")
            if not token: break
            from urllib.parse import quote
            page=get(path+"?pageSize=3&pageToken="+quote(token))
        expected=[r for r in snap.rows["objects/Ticket.json"]]
        if rows!=expected:
            raise AssertionError("Cutover lost or changed ticket rows")
        tested.append("paginated object-read parity")
        obj=get(path+"/SYN-T-004")
        if obj!=expected[4]:
            raise AssertionError("Single-object mismatch")
        tested.append("single object read parity")
        ty=get("/api/v2/ontologies/synthetic/objectTypes/Ticket")
        if ty["apiName"]!="Ticket":
            raise AssertionError("Type mismatch")
        tested.append("object type metadata")
        links=get("/api/v2/ontologies/synthetic/objectTypes/Ticket/outgoingLinkTypes")
        if links["data"]!=snap.rows["linkTypes/Ticket.json"]:
            raise AssertionError("Link type metadata mismatch")
        tested.append("outgoing link metadata")
        try:
            urlopen(f"http://127.0.0.1:{server.server_port}/healthz",timeout=5)
            raise AssertionError("Unauthenticated health check exposed")
        except HTTPError as e:
            if e.code!=401:raise
        tested.append("local authorization denial")
        req=Request(f"http://127.0.0.1:{server.server_port}{path}",
                    data=b"{}", method="POST",headers={"Authorization":"Bearer "+secret})
        try:
            urlopen(req,timeout=5)
            raise AssertionError("Unexpected write acceptance")
        except HTTPError as e:
            if e.code!=501:raise
        tested.append("unported write explicit 501")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=4)
    receipt={
      "evidence_tier":"synthetic API-v2 wire-shape conformance; NOT live Foundry",
      "source":"official-shape fixture constructed locally",
      "capture_digest":snap.digest,"compared_ticket_rows":len(expected),
      "tests":tested,"accepted":len(tested),"failed":0,
      "limits":"Read-only subset; actions, functions, ACL semantics, application runtime and telemetry export require independent migration.",
      "executed_at":datetime.now(timezone.utc).isoformat(),
    }
    (out/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    prove(args.out)
