"""Extract named members from a huge remote tar with HTTP range requests (no full download).

    python tools/remote_tar.py URL OUT_DIR NAME_SUBSTRING [NAME_SUBSTRING ...] [--list-only]
Walks the 512-byte tar headers by range-reading each one and skipping over member data, so only headers and
the wanted members are transferred. Handles GNU long names ('L') and pax headers ('x').
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


def octal(b: bytes) -> int:
    s = b.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def main(a):
    client = httpx.Client(timeout=120, follow_redirects=True)
    total = int(client.head(a.url).headers["content-length"])
    off, pending_name, found, wanted = 0, None, {}, set(a.names)
    n = 0
    while off + 512 <= total and wanted - set(found):
        hdr = client.get(a.url, headers={"Range": f"bytes={off}-{off + 511}"}).content
        if hdr == b"\0" * 512:
            break
        size, kind = octal(hdr[124:136]), hdr[156:157]
        name = pending_name or (hdr[345:500].split(b"\0", 1)[0] + b"/" if hdr[345] else b"") + hdr[:100].split(b"\0", 1)[0]
        data_off = off + 512
        if kind == b"L":                       # GNU long name: the next member's name is the data
            pending_name = client.get(a.url, headers={"Range": f"bytes={data_off}-{data_off + size - 1}"}).content.split(b"\0", 1)[0]
        elif kind == b"x":                     # pax header: look for path=
            pax = client.get(a.url, headers={"Range": f"bytes={data_off}-{data_off + size - 1}"}).content
            pending_name = next((ln.split(b"=", 1)[1] for ln in pax.split(b"\n") if b" path=" in ln), None)
        else:
            pending_name = None
            path = name.decode(errors="replace")
            n += 1
            if a.verbose or n % 200 == 0:
                print(f"[{n}] {off/1e9:7.2f} GB {path} ({size/1e6:.1f} MB)", flush=True)
            hit = next((w for w in wanted if w in path and w not in found), None)
            if hit and kind in (b"0", b"\0") and not a.list_only:
                dest = Path(a.out) / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with client.stream("GET", a.url, headers={"Range": f"bytes={data_off}-{data_off + size - 1}"}) as r, \
                        open(dest, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                found.setdefault(hit, []).append(str(dest))
                print(f"saved {dest} ({size/1e6:.1f} MB)", flush=True)
        off = data_off + ((size + 511) // 512) * 512
    print("found:", found, file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("names", nargs="+", help="substrings; each is satisfied by the first member that contains it, "
                                           "so pass exact file paths to fetch specific files")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
