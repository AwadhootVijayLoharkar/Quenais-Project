#!/usr/bin/env python
"""
Upload a file to an existing Zenodo draft (no curl needed).

    python tools/zenodo_upload.py <draft-id> <file> [<file> ...]

<draft-id> is the number in the draft's address, zenodo.org/uploads/<draft-id>.
The access token (Zenodo -> Applications -> Personal access tokens, scopes
deposit:write and deposit:actions) is read from $ZENODO_TOKEN if set, otherwise
asked for without echo. It is never written to disk.
"""

import getpass
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://zenodo.org/api/deposit/depositions"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dep_id, files = sys.argv[1], sys.argv[2:]
    token = os.environ.get("ZENODO_TOKEN") or getpass.getpass("Zenodo token: ")
    hdr = {"Authorization": f"Bearer {token}"}
    try:
        req = urllib.request.Request(f"{API}/{dep_id}", headers=hdr)
        bucket = json.load(urllib.request.urlopen(req))["links"]["bucket"]
    except urllib.error.HTTPError as exc:
        sys.exit(f"could not open draft {dep_id}: HTTP {exc.code} {exc.reason}\n"
                 f"{exc.read().decode(errors='replace')[:500]}")
    print("bucket:", bucket)
    for path in files:
        name = os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            req = urllib.request.Request(
                f"{bucket}/{name}", data=fh, method="PUT",
                headers={**hdr, "Content-Type": "application/octet-stream",
                         "Content-Length": str(size)})
            try:
                resp = urllib.request.urlopen(req)
            except urllib.error.HTTPError as exc:
                sys.exit(f"upload of {name} failed: HTTP {exc.code} {exc.reason}\n"
                         f"{exc.read().decode(errors='replace')[:500]}")
        print(f"HTTP {resp.status}  {name}  {size / 2**20:.1f} MB uploaded")


if __name__ == "__main__":
    main()
