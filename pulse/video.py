"""Turn a video file into heartbeat events the way a live camera would: sample, then skip unchanged frames.

Frames are extracted once with ffmpeg (1 fps, 640 px). A cheap grayscale-thumbnail difference decides
which frames are worth a decider read; a floor interval guarantees at least one frame every few seconds
so slow changes are not missed. The same filter runs in the browser for live input.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .heartbeat import Event

FFMPEG = os.environ.get("PULSE_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"   # set PULSE_FFMPEG to override


def extract_frames(video: str, out_dir: str, fps: float = 1.0, width: int = 640, start: float = 0.0,
                   duration: float | None = None, ffmpeg: str = FFMPEG) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("f_*.jpg"))
    if existing:
        return existing
    cmd = [ffmpeg, "-v", "error", "-ss", str(start), "-i", video]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps},scale={width}:-2", "-q:v", "4", str(out / "f_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(out.glob("f_*.jpg"))


def frame_time(path: Path, fps: float = 1.0, start: float = 0.0) -> float:
    """ffmpeg numbers frames from 1; frame n covers time start + (n - 1) / fps."""
    return start + (int(path.stem.split("_")[1]) - 1) / fps


@dataclass
class ChangeFilter:
    threshold: float = 6.0      # mean absolute difference on a 32x24 grayscale thumbnail (0-255)
    floor_s: float = 30.0       # always keep one frame at least this often
    _last: np.ndarray | None = None
    _last_t: float = -1e9

    def keep(self, image: Path, t: float) -> tuple[bool, float]:
        thumb = np.asarray(Image.open(image).convert("L").resize((32, 24)), dtype=np.float32)
        diff = 255.0 if self._last is None else float(np.abs(thumb - self._last).mean())
        if diff >= self.threshold or t - self._last_t >= self.floor_s:
            self._last, self._last_t = thumb, t
            return True, diff
        return False, diff


def video_events(frames: list[Path], fps: float = 1.0, start: float = 0.0, flt: ChangeFilter | None = None):
    """Yield (event or None, frame path, t, diff); None means the change filter dropped the frame."""
    flt = flt or ChangeFilter()
    for path in frames:
        t = frame_time(path, fps, start)
        kept, diff = flt.keep(path, t)
        ev = Event("vision", t, data={"frame": path.stem, "image": str(path)}) if kept else None
        yield ev, path, t, diff
