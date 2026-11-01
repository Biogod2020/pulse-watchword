"""Online temporal triggers over per-frame judge probabilities (look only at the present and the past).

EdgeTrigger: fire once when a score first crosses its threshold, then stay quiet for `refractory_s`.
  Use for brief actions (washing a cup, turning a knob).
StateTracker: hysteresis on/off for sustained activities (stirring): on after `min_on` consecutive frames at or
  above `on`, off when a frame drops below `off`. Emits "start"/"stop".
Settings come from EPIC-KITCHENS P12_101 (eval_activity_cues.py): washing edge@0.9 86% precise; tap/knob
edge@0.5 58%; stirring state 0.95/0.7/3 covers 36/37 narrated stir events.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdgeTrigger:
    threshold: float
    refractory_s: float = 5.0
    _prev: float = 0.0
    _last: float = -1e9

    def update(self, t: float, p: float) -> str | None:
        fired = p >= self.threshold > self._prev and t - self._last >= self.refractory_s
        self._prev = p
        if fired:
            self._last = t
            return "event"
        return None


@dataclass
class StateTracker:
    on: float
    off: float
    min_on: int = 3
    active: bool = False
    _run: int = 0
    started_at: float | None = None

    def update(self, t: float, p: float) -> str | None:
        if not self.active:
            self._run = self._run + 1 if p >= self.on else 0
            if self._run >= self.min_on:
                self.active, self.started_at = True, t - self.min_on + 1
                return "start"
            return None
        if p < self.off:
            self.active, self._run = False, 0
            return "stop"
        return None


@dataclass
class ConfirmedEdge:
    """Fire once when `confirm` consecutive reads are at or above `threshold`, then stay quiet for `refractory_s`.
    Overlapping clips (2 s every 1 s) make two agreeing reads cheap evidence against one-clip flickers."""
    threshold: float
    confirm: int = 2
    refractory_s: float = 8.0
    _run: int = 0
    _last: float = -1e9

    def update(self, t: float, p: float) -> str | None:
        self._run = self._run + 1 if p >= self.threshold else 0
        if self._run == self.confirm and t - self._last >= self.refractory_s:
            self._last = t
            return "event"
        return None
