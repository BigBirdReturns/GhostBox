"""Conformance tests for the bounded Foundry Ontology API v2 read bridge.

Fixtures implement the documented JSON shapes. They do NOT represent an
authorized tenant or Palantir-provided certification.
"""
import json
from pathlib import Path
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest

from foundry_exit.interop_bridge import (
    BridgeError, CapturedOntology, _pages, authorized_fetch,
    capture_ontology, make_handler,
)
from http.server import ThreadingHTTPServer


class FakeFoundryV2:
    def __init__(self):
        self.calls = []
        self.schemas = {
            "Ticket": {"apiName":"Ticket","displayName":"Ticket",
                      "primaryKey":"id","properties":{"id":{"dataType":{"type":"string"}}}},
            "Decision": {"apiName":"Decision","displayName":"Decision",
                      "primaryKey":"id","properties":{"id":{"dataType":{"type":"string"}}}},
        }
        self.records = {
            "Ticket": [
                {"__primaryKey":f"T-{i}","__apiName":"Ticket","id":f"T-{i}","decision":"open"}
                for i in range(5)
            ],
            "Decision": [
                {"__primaryKey":"D-1","__apiName":"Decision","id":"D-1"}
            ]
        }
        self.links = {
            "Ticket": [{"apiName":"selectedDecision", "objectTypeApiName":"Decision","cardinality":"ONE"}],
            "Decision": []
        }

    def __call__(self, path, params):
        self.calls.append((path,params))
        if path.endswith("/objectTypes"):
            rows = list(self.schemas.values())
        elif path.endswith("/outgoingLinkTypes"):
            name = path.split("/")[-2]
            rows = self.links[name]
        elif "/objects/" in path:
            name = path.split("/")[-1]
            rows = self.records[name]
        else:
            raise ValueError(path)
        token = params.get("pageToken")
        offset = int(token) if token else 0
        size = 2
        page = {"data":rows[offset:offset+size], "totalCount":str(len(rows))}
        if offset+size < len(rows):
            page["nextPageToken"] = str(offset+size)
        return page


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "capture"
        self.fake = FakeFoundryV2()
        self.manifest = capture_ontology(self.fake,"demo",self.root,["Ticket","Decision"])
        self.capture = CapturedOntology(self.root)
        self.secret = "local-only-test-secret"

    def test_capture_complete_and_hashed(self):
        self.assertEqual(len(self.manifest["files"]),5)
        self.assertEqual(self.capture.type_names,{"Ticket","Decision"})
        self.assertEqual(len(self.capture.rows["objects/Ticket.json"]),5)
        self.assertEqual(self.manifest["consistency"].split(";")[0],"nontransactional-list-objects")
        self.assertEqual(self.capture.total["objects/Ticket.json"],5)
        self.assertGreaterEqual(len([x for x in self.fake.calls if x[0].endswith("/objects/Ticket")]),3)

    def test_list_objects_pagination_and_lookup(self):
        path="/api/v2/ontologies/demo/objects/Ticket"
        first=self.capture.get(path, {"pageSize":["2"]}, self.secret)
        self.assertEqual([r["id"] for r in first["data"]],["T-0","T-1"])
        self.assertEqual(first["totalCount"],"5")
        seen=first["data"][:]
        nxt=first["nextPageToken"]
        while nxt:
            resp=self.capture.get(path,{"pageSize":["2"],"pageToken":[nxt]},self.secret)
            seen+=resp["data"]
            nxt=resp.get("nextPageToken")
        self.assertEqual([r["id"] for r in seen],[f"T-{i}" for i in range(5)])
        self.assertEqual(self.capture.get(path+"/T-3",{},self.secret)["id"],"T-3")
        self.assertEqual(self.capture.get("/api/v2/ontologies/demo/objectTypes/Ticket",{},self.secret)["apiName"],"Ticket")
        self.assertEqual(self.capture.get("/api/v2/ontologies/demo/objectTypes/Ticket/outgoingLinkTypes",{},self.secret)["data"][0]["apiName"],"selectedDecision")

    def test_token_is_scoped_and_fails_on_corruption(self):
        path="/api/v2/ontologies/demo/objects/Ticket"
        token=self.capture.get(path,{"pageSize":["2"]},self.secret)["nextPageToken"]
        with self.assertRaises(BridgeError) as c:
            self.capture.get("/api/v2/ontologies/demo/objects/Decision",{"pageToken":[token]},self.secret)
        self.assertEqual(c.exception.code,"INVALID_PAGE_TOKEN")
        with self.assertRaises(BridgeError):
            self.capture.get(path,{"pageToken":[token[:-1]+("a" if token[-1]!="a" else "b")]},self.secret)
        with self.assertRaises(BridgeError):
            self.capture.get(path,{"pageToken":[token]},self.secret+"rotated")
        with self.assertRaises(BridgeError) as c:
            self.capture.get(path,{"pageToken":["%%%"]},self.secret)
        self.assertEqual(c.exception.code,"INVALID_PAGE_TOKEN")

    def test_hash_mismatch_fails_closed(self):
        original=(self.root/"objects/Ticket.json").read_bytes()
        (self.root/"objects/Ticket.json").write_bytes(original+b"\n")
        with self.assertRaises(BridgeError) as c:
            CapturedOntology(self.root)
        self.assertEqual(c.exception.code,"HASH_MISMATCH")

    def test_capture_refuses_overwrite_and_missing_scope(self):
        with self.assertRaises(BridgeError) as c:
            capture_ontology(self.fake,"demo",self.root,["Ticket"])
        self.assertEqual(c.exception.code,"DEST_EXISTS")
        with self.assertRaises(BridgeError) as c:
            capture_ontology(self.fake,"demo",Path(self.temp.name)/"missing",["Nope"])
        self.assertEqual(c.exception.code,"SCOPE_NOT_FOUND")

    def test_partial_and_duplicate_exports_refused(self):
        f=FakeFoundryV2()
        def bad_total(path,params):
            value=f(path,params)
            if path.endswith("/objects/Ticket"):
                value["totalCount"]="999"
            return value
        with self.assertRaises(BridgeError) as c:
            capture_ontology(bad_total,"demo",Path(self.temp.name)/"bad_count",["Ticket"])
        self.assertEqual(c.exception.code,"INCOMPLETE_CAPTURE")
        f2=FakeFoundryV2()
        f2.records["Ticket"][4]["__primaryKey"]="T-0"
        with self.assertRaises(BridgeError) as c:
            capture_ontology(f2,"demo",Path(self.temp.name)/"duplicate",["Ticket"])
        self.assertEqual(c.exception.code,"MISSING_OR_DUPLICATE_KEY")

    def test_source_security_and_loop(self):
        with self.assertRaises(BridgeError) as c:
            authorized_fetch("http://example.com","abc")
        self.assertEqual(c.exception.code,"INVALID_SOURCE")
        with self.assertRaises(BridgeError):
            authorized_fetch("https://example.com","")
        def loop(path,params):
            return {"data":[{"id":1}],"nextPageToken":"one"}
        with self.assertRaises(BridgeError) as c:
            _pages(loop,"/api/v2/ontologies/demo/objects/Ticket")
        self.assertEqual(c.exception.code,"PAGE_TOKEN_LOOP")
        with self.assertRaises(BridgeError):
            capture_ontology(self.fake,"demo",Path(self.temp.name)/"traversal",["../evil"])

    def test_unported_queries_are_not_silently_ignored(self):
        with self.assertRaises(BridgeError) as c:
            self.capture.get("/api/v2/ontologies/demo/objects/Ticket",
                             {"branch":["alternate"]}, self.secret)
        self.assertEqual(c.exception.status,501)

    def test_unknown_actions_explicit_501(self):
        with self.assertRaises(BridgeError) as c:
            self.capture.get("/api/v2/ontologies/demo/actions/approve/apply",{},self.secret)
        self.assertEqual(c.exception.status,501)

    def test_real_loopback_http_auth_read_and_write_refusal(self):
        handler=make_handler(self.capture,self.secret)
        server=ThreadingHTTPServer(("127.0.0.1",0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url=f"http://127.0.0.1:{server.server_port}/api/v2/ontologies/demo/objects/Ticket"
        with self.assertRaises(HTTPError) as c:
            urlopen(url)
        self.assertEqual(c.exception.code,401)
        req=Request(url,headers={"Authorization":f"Bearer {self.secret}"})
        with urlopen(req) as r:
            page=json.load(r)
        self.assertEqual(len(page["data"]),5)
        req=Request(url,method="POST",headers={"Authorization":f"Bearer {self.secret}"},data=b"{}")
        with self.assertRaises(HTTPError) as c:
            urlopen(req)
        self.assertEqual(c.exception.code,501)


if __name__=="__main__":
    unittest.main()
