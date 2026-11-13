"""Generate a strictly synthetic Foundry API v2 capture for local bridge demo.

This uses the same capture function called with real authorized HTTPS responses.
No Foundry connection or credentials are made or needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .interop_bridge import capture_ontology


SCHEMAS=[
  {"apiName":"Ticket","displayName":"Ticket","primaryKey":"id",
   "properties":{"id":{"dataType":{"type":"string"}},
                 "status":{"dataType":{"type":"string"}}}},
  {"apiName":"Decision","displayName":"Decision","primaryKey":"id",
   "properties":{"id":{"dataType":{"type":"string"}},
                 "result":{"dataType":{"type":"string"}}}},
]
OBJECTS={
 "Ticket":[{"__apiName":"Ticket","__primaryKey":f"SYN-T-{i:03}",
            "id":f"SYN-T-{i:03}","status":"pending" if i%2==0 else "completed"}
           for i in range(8)],
 "Decision":[{"__apiName":"Decision","__primaryKey":"SYN-D-001",
              "id":"SYN-D-001","result":"reviewed"}],
}
LINKS={
 "Ticket":[{"apiName":"lastDecision","objectTypeApiName":"Decision","cardinality":"ONE"}],
 "Decision":[],
}


def fixture_fetch(path,params):
    if path.endswith("/objectTypes"):
        rows=SCHEMAS
    elif path.endswith("/outgoingLinkTypes"):
        rows=LINKS[path.rsplit("/",2)[-2]]
    elif "/objects/" in path:
        rows=OBJECTS[path.rsplit("/",1)[-1]]
    else:
        raise ValueError("Unsupported synthetic endpoint "+path)
    start=int(params.get("pageToken") or 0)
    page={"data":rows[start:start+3],"totalCount":str(len(rows))}
    if start+3<len(rows):
        page["nextPageToken"]=str(start+3)
    return page


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    m=capture_ontology(fixture_fetch,"synthetic",a.out,["Ticket","Decision"])
    print(json.dumps({"source":"SYNTHETIC official-shape fixture, NOT Palantir",
        "capture":str(a.out),"files":len(m["files"]),
        "status":"complete selected-type pagination"},indent=2))
