"""Read-only Foundry Ontology API v2 interoperability bridge.

Consumes a complete authorized capture made through Palantir's *published*
Ontology API. Provides a deliberately bounded, API-shaped local replacement.
Every unsupported behavior fails visibly rather than pretending to have ported
Palantir's proprietary execution, authorization, or policy engines.

Stdlib-only. Never exports or logs credentials. This is not a Foundry server.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlsplit, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler


class BridgeError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise BridgeError("INVALID_NAME", "Unsafe ontology or object-type name")
    return value


def _path(*segs: str) -> str:
    return "/api/v2/ontologies/" + "/".join(quote(s, safe="") for s in segs)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def authorized_fetch(base_url: str, token: str, timeout: float = 25.0) -> Callable[[str, dict], dict]:
    """Factory for bounded HTTPS-only GETs to a specifically authorized tenant."""
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise BridgeError("INVALID_SOURCE", "Source must be an explicit HTTPS tenant origin")
    if not token:
        raise BridgeError("NO_CREDENTIAL", "Set the scoped read-only Foundry token out of band")
    origin = base_url.rstrip("/")
    opener = build_opener(_NoRedirect())

    def fetch(path: str, params: dict) -> dict:
        if not path.startswith("/api/v2/ontologies/") or ".." in path:
            raise BridgeError("INVALID_ROUTE", "Only known Ontology API v2 read paths are permitted")
        url = origin + path + (("?" + urlencode(params)) if params else "")
        req = Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"}, method="GET")
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise BridgeError("SOURCE_TOO_LARGE", "API response exceeds 32 MiB bound")
                doc = json.loads(raw)
        except HTTPError as e:
            raise BridgeError("UPSTREAM_HTTP", f"Tenant returned HTTP {e.code}", status=502) from None
        except (URLError, TimeoutError) as e:
            raise BridgeError("UPSTREAM_UNAVAILABLE", type(e).__name__, status=502) from None
        except (ValueError, UnicodeError):
            raise BridgeError("UPSTREAM_INVALID_JSON", "Upstream did not return JSON", status=502) from None
        if not isinstance(doc, dict):
            raise BridgeError("UPSTREAM_INVALID_SHAPE", "Expected a JSON object from the tenant", status=502)
        return doc

    return fetch


def _pages(fetch: Callable[[str, dict], dict], path: str, max_pages: int = 20000) -> tuple[list[dict], list[dict], int | None]:
    pages: list[dict] = []
    rows: list[dict] = []
    total: int | None = None
    seen = set()
    token: str | None = None
    for _ in range(max_pages):
        params = {"pageSize": 1000}
        if token is not None:
            params["pageToken"] = token
        page = fetch(path, params)
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise BridgeError("INVALID_PAGE", f"Missing data[] in {path}")
        if "totalCount" in page and page["totalCount"] is not None:
            try:
                declared = int(page["totalCount"])
            except (TypeError, ValueError):
                raise BridgeError("INVALID_TOTAL", "totalCount is not an integer") from None
            if total is not None and total != declared:
                raise BridgeError("CHANGING_TOTAL", "Source totalCount changed during paging")
            total = declared
        rows.extend(page["data"])
        pages.append(page)
        nxt = page.get("nextPageToken")
        if not nxt:
            if total is not None and len(rows) != total:
                raise BridgeError("INCOMPLETE_CAPTURE", f"Source declared {total} but returned {len(rows)} rows")
            return pages, rows, total
        if not isinstance(nxt, str) or nxt in seen:
            raise BridgeError("PAGE_TOKEN_LOOP", "Duplicate or invalid nextPageToken")
        seen.add(nxt)
        token = nxt
    raise BridgeError("PAGE_LIMIT", "Exceeded bounded pagination limit")


def capture_ontology(
    fetch: Callable[[str, dict], dict],
    ontology: str,
    out_dir: Path,
    allowed_types: list[str],
) -> dict:
    """Capture selected object-type metadata, link types, and full paginated data.

    Captures the source responses verbatim AS JSON VALUES, not raw HTTP bytes.
    No point-in-time transaction guarantee is assumed for object listing.
    """
    ontology = _name(ontology)
    requested = {_name(x) for x in allowed_types}
    if not requested:
        raise BridgeError("SCOPE_REQUIRED", "Specify one or more --types")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise BridgeError("DEST_EXISTS", "Refusing to overwrite an existing capture")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".foundry-capture-", dir=str(out_dir.parent)))
    manifest: dict = {
        "format": "axm-foundry-ontology-v2-capture/1",
        "source": "palantir-documented-ontology-api-v2",
        "ontology": ontology,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "consistency": "nontransactional-list-objects; independent reconciliation required",
        "coverage": "complete pagination for selected types; other types excluded",
        "files": [],
    }

    def record(relative: str, pages: list[dict], *, row_count: int, total: int | None):
        dest = tmp / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = _json_bytes(pages)
        dest.write_bytes(payload)
        manifest["files"].append({
            "path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": row_count, "declared_total": total, "pages": len(pages),
            "exhausted": True,
        })

    try:
        # Palantir's published ontology metadata API lists all names. Keep
        # only selected types, never silently claim tenant-wide coverage.
        pages, all_types, _ = _pages(fetch, _path(ontology, "objectTypes"))
        types = [t for t in all_types if isinstance(t, dict) and t.get("apiName") in requested]
        found = {t["apiName"] for t in types}
        if found != requested:
            raise BridgeError("SCOPE_NOT_FOUND", "Missing object types: " + ", ".join(sorted(requested-found)))
        record("objectTypes.json", [{"data": types, "totalCount": str(len(types))}],
               row_count=len(types), total=len(types))
        for name in sorted(requested):
            links, lr, lt = _pages(fetch, _path(ontology, "objectTypes", name, "outgoingLinkTypes"))
            record(f"linkTypes/{name}.json", links, row_count=len(lr), total=lt)
            objects, rows, total = _pages(fetch, _path(ontology, "objects", name))
            keys = [str(r["__primaryKey"]) for r in rows if isinstance(r, dict) and "__primaryKey" in r]
            if len(keys) != len(rows) or len(set(keys)) != len(keys):
                raise BridgeError("MISSING_OR_DUPLICATE_KEY", f"Missing/duplicate primary keys: {name}")
            record(f"objects/{name}.json", objects, row_count=len(rows), total=total)
        (tmp / "AXM_CAPTURE_MANIFEST.json").write_bytes(_json_bytes(manifest))
        tmp.rename(out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_pages_file(path: Path) -> tuple[list[dict], list[dict]]:
    value = json.loads(path.read_bytes())
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise BridgeError("INVALID_CAPTURE", f"Bad capture at {path}")
    rows = []
    for page in value:
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise BridgeError("INVALID_CAPTURE", f"Bad page at {path}")
        rows.extend(page["data"])
    return value, rows


class CapturedOntology:
    def __init__(self, root: Path, *, allow_partial: bool = False):
        self.root = Path(root)
        mf = self.root / "AXM_CAPTURE_MANIFEST.json"
        if not mf.is_file():
            raise BridgeError("MISSING_MANIFEST", "An audited capture manifest is required")
        self.manifest = json.loads(mf.read_bytes())
        if self.manifest.get("format") != "axm-foundry-ontology-v2-capture/1":
            raise BridgeError("WRONG_FORMAT", "Capture version mismatch")
        self.ontology = _name(self.manifest["ontology"])
        self.parts: dict[str, list[dict]] = {}
        self.rows: dict[str, list[dict]] = {}
        self.total: dict[str, int] = {}
        for f in self.manifest["files"]:
            relative = f.get("path")
            if not isinstance(relative, str):
                raise BridgeError("INVALID_MANIFEST", "Missing file path")
            parts = relative.split("/")
            if not parts or len(parts) > 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) or x in (".", "..") for x in parts):
                raise BridgeError("INVALID_MANIFEST", "Unsafe file path")
            path = self.root.joinpath(*parts)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != f.get("sha256"):
                raise BridgeError("HASH_MISMATCH", f"Capture modified: {relative}")
            pages, rows = _load_pages_file(path)
            if len(rows) != f.get("rows") or len(pages) != f.get("pages") or not f.get("exhausted"):
                raise BridgeError("INCOMPLETE_CAPTURE", f"Incomplete capture: {relative}")
            declared = f.get("declared_total")
            if declared is not None and len(rows) != int(declared) and not allow_partial:
                raise BridgeError("INCOMPLETE_CAPTURE", f"Count mismatch: {relative}")
            self.parts[relative] = pages
            self.rows[relative] = rows
            self.total[relative] = len(rows)
        if "objectTypes.json" not in self.rows:
            raise BridgeError("MISSING_TYPES", "Capture has no object types")
        self.type_names = {x["apiName"] for x in self.rows["objectTypes.json"]}
        for name in self.type_names:
            if f"objects/{name}.json" not in self.rows or f"linkTypes/{name}.json" not in self.rows:
                raise BridgeError("INCOMPLETE_CAPTURE", f"Missing object or link capture for {name}")
        self.digest = hashlib.sha256(mf.read_bytes()).hexdigest()

    def get(self, path: str, query: Mapping[str, list[str]], token_secret: str) -> dict:
        segments = [unquote(x) for x in path.split("/") if x]
        if len(segments) < 5 or segments[:3] != ["api", "v2", "ontologies"]:
            raise BridgeError("NOT_PORTED", "Only documented Ontology v2 read routes are implemented", 501)
        ontology = segments[3]
        if ontology != self.ontology:
            raise BridgeError("ONTOLOGY_NOT_FOUND", "Unknown ontology", 404)
        # A wire-compatible response must not silently ignore selectors,
        # branches or consistency options whose semantics we have not ported.
        unsupported = set(query) - {"pageSize", "pageToken"}
        if unsupported:
            raise BridgeError("NOT_PORTED", "Unsupported query semantics: " + ", ".join(sorted(unsupported)), 501)
        relative: str
        rows: list[dict]
        if segments[4:] == ["objectTypes"]:
            relative = "objectTypes.json"
        elif len(segments) == 7 and segments[4] == "objectTypes" and segments[6] == "outgoingLinkTypes":
            relative = "linkTypes/" + _name(segments[5]) + ".json"
        elif len(segments) == 6 and segments[4] == "objectTypes":
            name = _name(segments[5])
            found = next((t for t in self.rows["objectTypes.json"] if t["apiName"] == name), None)
            if found is None:
                raise BridgeError("OBJECT_TYPE_NOT_FOUND", "Unknown object type", 404)
            return found
        elif len(segments) == 6 and segments[4] == "objects":
            relative = "objects/" + _name(segments[5]) + ".json"
        elif len(segments) == 7 and segments[4] == "objects":
            relative = "objects/" + _name(segments[5]) + ".json"
            rows = self.rows.get(relative, [])
            key = segments[6]
            found = next((r for r in rows if str(r.get("__primaryKey")) == key), None)
            if found is None:
                raise BridgeError("OBJECT_NOT_FOUND", "Unknown object primary key", 404)
            return found
        else:
            raise BridgeError("NOT_PORTED", "Actions, functions and writes require independently deployed execution engines", 501)
        rows = self.rows.get(relative)
        if rows is None:
            raise BridgeError("TYPE_NOT_FOUND", "Object or link type not in selected capture", 404)
        raw_size = (query.get("pageSize") or ["1000"])[0]
        try:
            size = int(raw_size)
        except ValueError:
            raise BridgeError("BAD_PAGE_SIZE", "pageSize must be an integer") from None
        if size < 1 or size > 1000:
            raise BridgeError("BAD_PAGE_SIZE", "pageSize must be between 1 and 1000")
        offset = 0
        supplied = (query.get("pageToken") or [None])[0]
        if supplied:
            try:
                data_s, sig = supplied.split(".", 1)
                data = base64.urlsafe_b64decode(data_s + "="*((-len(data_s))%4))
                expected = hmac.new(token_secret.encode(), data, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(sig, expected):
                    raise ValueError("HMAC")
                state = json.loads(data)
                if state["path"] != relative or state["digest"] != self.digest:
                    raise ValueError("scope")
                offset = int(state["offset"])
            except (ValueError, KeyError, TypeError):
                raise BridgeError("INVALID_PAGE_TOKEN", "Page token is stale or invalid")
        if offset < 0 or offset >= len(rows) and supplied:
            raise BridgeError("INVALID_PAGE_TOKEN", "Page offset outside captured range")
        page = rows[offset:offset+size]
        result = {"data": page}
        if relative.startswith("objects/"):
            result["totalCount"] = str(len(rows))
        nxt = offset+len(page)
        if nxt < len(rows):
            data = _json_bytes({"digest": self.digest, "path": relative, "offset": nxt})
            sig = hmac.new(token_secret.encode(), data, hashlib.sha256).hexdigest()
            result["nextPageToken"] = base64.urlsafe_b64encode(data).decode().rstrip("=")+"."+sig
        return result


def make_handler(capture: CapturedOntology, local_token: str):
    if len(local_token) < 16:
        raise BridgeError("MISSING_LOCAL_AUTH", "Local bearer token must have at least 16 characters")
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, data: dict):
            body = _json_bytes(data)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authenticated(self) -> bool:
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + local_token)

        def do_GET(self):
            if not self._authenticated():
                self._send(401, {"errorCode": "UNAUTHORIZED", "errorName": "UNAUTHORIZED"})
                return
            split = urlsplit(self.path)
            if split.path == "/healthz":
                self._send(200, {"status": "ready", "source": "captured", "digest": capture.digest, "readOnly": True})
                return
            try:
                self._send(200, capture.get(split.path, parse_qs(split.query), local_token))
            except BridgeError as e:
                self._send(e.status, {"errorCode": e.code, "errorName": e.code, "message": str(e)})

        def do_POST(self):
            if not self._authenticated():
                self._send(401, {"errorCode": "UNAUTHORIZED"})
            else:
                self._send(501, {"errorCode": "NOT_PORTED", "message": "Action/write engine is not implemented"})

        def do_PUT(self):
            self.do_POST()

        def do_DELETE(self):
            self.do_POST()

        def log_message(self, fmt, *args):
            # Do not record potentially sensitive ontology keys, paths or tokens.
            pass

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only interoperable Ontology API capture and local endpoint")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="Pull selected object types from an authorized Foundry tenant")
    cap.add_argument("--base-url", required=True, help="Explicit HTTPS Foundry origin, no /api suffix")
    cap.add_argument("--ontology", required=True)
    cap.add_argument("--types", required=True, help="Comma-separated type names; explicit scope")
    cap.add_argument("--out", required=True, type=Path)
    cap.add_argument("--token-env", default="FOUNDRY_TOKEN")
    srv = sub.add_parser("serve", help="Serve local read-only Ontology v2 API")
    srv.add_argument("--capture", type=Path, required=True)
    srv.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    srv.add_argument("--port", type=int, default=8787)
    srv.add_argument("--local-token-env", default="AXM_EXIT_LOCAL_TOKEN")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            fetch = authorized_fetch(args.base_url, os.getenv(args.token_env, ""))
            result = capture_ontology(fetch, args.ontology, args.out, args.types.split(","))
            print(json.dumps({"capture": str(args.out), "coverage": result["coverage"],
                              "files": len(result["files"]), "consistency": result["consistency"]}, indent=2))
        else:
            capture = CapturedOntology(args.capture)
            handler = make_handler(capture, os.getenv(args.local_token_env, ""))
            server = ThreadingHTTPServer((args.host, args.port), handler)
            print(f"Read-only Ontology v2 bridge on {args.host}:{args.port}, digest {capture.digest}", flush=True)
            server.serve_forever()
    except BridgeError as e:
        print(json.dumps({"error": e.code, "message": str(e)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
