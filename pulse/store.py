"""Persistent memory: SQLite for facts, segments and episodes; content-addressed files for images.

Tiers (see docs/design.md):
- L1 episodes: kept frames (thumbnail blob + caption + vector), rolled off after `keep_days` or when the
  blob store exceeds `max_bytes`, oldest first. The caption and vector stay; only pixels are dropped.
- L2 evidence: blobs referenced by a live fact are pinned and never rolled off.
- L3 semantic: facts, segments and episode text/vectors stay.
Raw audio is never stored.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments(seg_id TEXT PRIMARY KEY, rev INT, speaker TEXT, text TEXT, t REAL);
CREATE TABLE IF NOT EXISTS facts(key TEXT PRIMARY KEY, value TEXT, evidence TEXT, source TEXT, t REAL, version INT);
CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, kind TEXT, t REAL, text TEXT, blob TEXT, vec BLOB,
                                    created REAL);
CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, t REAL, op TEXT, detail TEXT);
"""


class Store:
    def __init__(self, root: str, keep_days: float = 14.0, max_bytes: int = 1 << 30):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "memory.sqlite")
        self.db.executescript(SCHEMA)
        self.keep_days, self.max_bytes = keep_days, max_bytes

    # ---- blobs ---------------------------------------------------------------------------------
    def put_blob(self, path: str) -> str:
        data = Path(path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        dest = self.blob_path(sha)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return sha

    def blob_path(self, sha: str) -> Path:
        return self.root / "blobs" / sha[:2] / f"{sha}.jpg"

    # ---- rows ----------------------------------------------------------------------------------
    def save_segment(self, seg) -> None:
        self.db.execute("REPLACE INTO segments VALUES(?,?,?,?,?)", (seg.seg_id, seg.rev, seg.speaker, seg.text, seg.t))
        self.db.commit()

    def save_fact(self, fact) -> None:
        ev = [{"kind": e.kind, "ref": e.ref, "rev": e.rev, "quote": e.quote} for e in fact.evidence]
        self.db.execute("REPLACE INTO facts VALUES(?,?,?,?,?,?)",
                        (fact.key, json.dumps(fact.value, ensure_ascii=False, default=str), json.dumps(ev),
                         fact.source, fact.t, fact.version))
        self.db.commit()

    def delete_fact(self, key: str) -> None:
        self.db.execute("DELETE FROM facts WHERE key=?", (key,))
        self.db.commit()

    def save_episode(self, ep, vec: np.ndarray | None, blob: str | None) -> None:
        self.db.execute("REPLACE INTO episodes VALUES(?,?,?,?,?,?,?)",
                        (ep.id, ep.kind, ep.t, ep.text, blob,
                         None if vec is None else np.asarray(vec, np.float32).tobytes(), time.time()))
        self.db.commit()

    def log(self, op: str, detail: dict) -> None:
        self.db.execute("INSERT INTO log(t,op,detail) VALUES(?,?,?)", (time.time(), op, json.dumps(detail, default=str)))
        self.db.commit()

    # ---- retention -----------------------------------------------------------------------------
    def pinned(self) -> set[str]:
        out = set()
        for (value,) in self.db.execute("SELECT value FROM facts"):
            blob = json.loads(value).get("blob") if value.startswith("{") else None
            if blob:
                out.add(blob)
        return out

    def usage(self) -> dict:
        files = list((self.root / "blobs").rglob("*.jpg"))
        db = (self.root / "memory.sqlite").stat().st_size
        return {"blob_files": len(files), "blob_bytes": sum(f.stat().st_size for f in files), "db_bytes": db,
                "episodes": self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "facts": self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]}

    def enforce_retention(self, now: float | None = None) -> dict:
        """Drop pixels of old or excess L1 episodes; pinned (L2) blobs and all text/vectors stay."""
        now = now or time.time()
        pinned = self.pinned()
        rows = self.db.execute("SELECT id, blob, created FROM episodes WHERE blob IS NOT NULL ORDER BY created").fetchall()
        sizes = {b: self.blob_path(b).stat().st_size for _, b, _ in rows if self.blob_path(b).exists()}
        total = sum(sizes.values())
        dropped = 0
        for ep_id, blob, created in rows:
            if blob in pinned:
                continue
            old = now - created > self.keep_days * 86400
            if not old and total <= self.max_bytes:
                continue
            if self.db.execute("SELECT COUNT(*) FROM episodes WHERE blob=? AND id<>?", (blob, ep_id)).fetchone()[0] == 0:
                path = self.blob_path(blob)
                if path.exists():
                    total -= sizes.get(blob, 0)
                    path.unlink()
            self.db.execute("UPDATE episodes SET blob=NULL WHERE id=?", (ep_id,))
            dropped += 1
        self.db.commit()
        return {"dropped_pixels": dropped, "blob_bytes": total, "pinned": len(pinned)}

    def wipe(self) -> None:
        self.db.close()
        shutil.rmtree(self.root)
