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


def test_edge_and_state_triggers():
    from pulse.triggers import EdgeTrigger, StateTracker
    e = EdgeTrigger(0.9, refractory_s=5)
    out = [e.update(t, p) for t, p in enumerate([0.1, 0.95, 0.97, 0.2, 0.95, 0.1, 0.1, 0.1, 0.95])]
    assert out == [None, "event", None, None, None, None, None, None, "event"]   # quiet during refractory
    s = StateTracker(on=0.95, off=0.7, min_on=3)
    out = [s.update(t, p) for t, p in enumerate([0.96, 0.97, 0.99, 0.8, 0.75, 0.6, 0.99])]
    assert out == [None, None, "start", None, None, "stop", None] and s.started_at == 0


def test_ring_clip_and_keep_policy(tmp_path):
    from pulse.stream import FrameRing, KeepPolicy, make_clip
    ring = FrameRing(seconds=3)
    paths = [img(tmp_path / f"f{i}.jpg", 12 * i) for i in range(20)]
    for i, p in enumerate(paths):
        ring.add(i * 0.25, str(p))
    assert ring.frames[0][0] >= 4.75 - 3 and len(ring.last(2)) == 8
    assert len(ring.sampled(1.0)) == 4
    url = make_clip(ring.last(2), 4, tmp_path / "c.mp4")
    assert url.startswith("data:video/mp4;base64,") and (tmp_path / "c.mp4").stat().st_size > 0
    k = KeepPolicy(keyframe_s=30)
    assert [k.decide(t, n) for t, n in [(0, False), (1, False), (2, True), (31, False)]] == ["keyframe", None, "evidence", "keyframe"]


def test_confirmed_edge_needs_agreeing_reads():
    from pulse.triggers import ConfirmedEdge
    c = ConfirmedEdge(0.8, confirm=2, refractory_s=8)
    seq = [0.9, 0.1, 0.85, 0.9, 0.95, 0.2, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]
    out = [c.update(t, p) for t, p in enumerate(seq)]
    # one-clip spikes never fire; agreeing pairs inside the 8 s refractory are ignored; t=14 fires again
    assert [t for t, o in enumerate(out) if o] == [3, 14]
