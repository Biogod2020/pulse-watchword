"""Live camera stream: everything seen stays in memory briefly; only meaningful frames reach disk.

- FrameRing: the last `seconds` of captured frames (L0). Look-back recall scans it; nothing here is persisted.
- make_clip: the newest `clip_s` seconds as a small H.264 clip for one video read by the judge
  (2 s clips every 1 s matched 4 fps frame-by-frame accuracy at under half the GPU time; eval_video_windows.py).
- KeepPolicy: which frames get stored (L1/L2): frames where something was noticed, frames used as evidence for
  an answer, and one keyframe at least every `keyframe_s` seconds.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

FFMPEG = os.environ.get("PULSE_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"   # set PULSE_FFMPEG to override


@dataclass
class FrameRing:
    seconds: float = 180.0
    frames: deque = field(default_factory=deque)     # (t, path)

    def add(self, t: float, path: str) -> None:
        self.frames.append((t, path))
        while self.frames and t - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    def last(self, seconds: float) -> list[tuple[float, str]]:
        if not self.frames:
            return []
        now = self.frames[-1][0]
        return [(t, p) for t, p in self.frames if t > now - seconds]

    def sampled(self, every_s: float = 1.0) -> list[tuple[float, str]]:
        """One frame per `every_s` seconds, for look-back scans."""
        out, last = [], -1e9
        for t, p in self.frames:
            if t - last >= every_s - 1e-6:
                out.append((t, p))
                last = t
        return out


def make_clip(frames: list[tuple[float, str]], fps: float, out: Path, width: int = 448) -> str:
    """Encode frames (in order) into an mp4 and return it as a data URL for the judge."""
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(p).resolve()}'\nduration {1 / fps:.4f}\n" for _, p in frames))
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", f"fps={fps},scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return "data:video/mp4;base64," + base64.b64encode(out.read_bytes()).decode()


@dataclass
class KeepPolicy:
    keyframe_s: float = 30.0
    _last_key: float = -1e9
    kept: int = 0

    def decide(self, t: float, noticed: bool) -> str | None:
        """'evidence' when something was noticed, 'keyframe' on the floor interval, else None (memory only)."""
        if noticed:
            self.kept += 1
            return "evidence"
        if t - self._last_key >= self.keyframe_s:
            self._last_key = t
            self.kept += 1
            return "keyframe"
        return None
