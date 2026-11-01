import time
from pathlib import Path

import numpy as np
from PIL import Image

from pulse.longterm import Episode
from pulse.memory import Evidence, WorkingMemory
from pulse.store import Store
from pulse.video import ChangeFilter


def img(path: Path, value: int) -> Path:
    Image.fromarray(np.full((48, 64, 3), value, np.uint8)).save(path)
    return path


def test_retention_drops_old_pixels_but_keeps_pinned_evidence_and_text(tmp_path):
    store = Store(str(tmp_path / "mem"), keep_days=14)
    old_frame, evidence_frame = img(tmp_path / "a.jpg", 10), img(tmp_path / "b.jpg", 200)
    pinned = store.put_blob(str(evidence_frame))
    m = WorkingMemory(store=store)
    m.put("item.keys", {"place": "bowl", "blob": pinned}, [Evidence("vision", "f2")], "observed", 1.0)
    store.save_episode(Episode("f1", "frame", 0.0, "old frame"), np.ones(4), store.put_blob(str(old_frame)))
    store.save_episode(Episode("f2", "frame", 1.0, "keys in bowl"), np.ones(4), pinned)
    out = store.enforce_retention(now=time.time() + 15 * 86400)
    assert out["dropped_pixels"] == 1 and out["pinned"] == 1
    assert store.blob_path(pinned).exists()
    rows = dict(store.db.execute("SELECT id, text FROM episodes").fetchall())
    assert rows == {"f1": "old frame", "f2": "keys in bowl"}        # text survives, pixels gone for f1


def test_change_filter_skips_static_frames_with_floor(tmp_path):
    flt = ChangeFilter(threshold=6.0, floor_s=30.0)
    same = img(tmp_path / "s.jpg", 100)
    changed = img(tmp_path / "c.jpg", 180)
    assert flt.keep(same, 0)[0]          # first frame always kept
    assert not flt.keep(same, 1)[0]      # unchanged
    assert flt.keep(changed, 2)[0]       # changed
    assert not flt.keep(changed, 3)[0]
    assert flt.keep(changed, 33)[0]      # floor interval forces a keep
