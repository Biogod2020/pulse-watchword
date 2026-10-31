"""Working memory: what is currently true, where it came from, and which input version it rests on.

Raw inputs (transcript segments with revisions, camera observations) are kept separately from the
facts derived from them. Every fact carries evidence pointing at a specific input revision, so a
corrected transcript invalidates exactly the facts built on the old wording, and a fact the agent
proposed is never mistaken for an independent observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    kind: str        # speech | vision | rule
    ref: str         # segment id, frame id or rule id
    rev: int = 1
    quote: str = ""


@dataclass
class Segment:
    seg_id: str
    rev: int
    speaker: str
    text: str
    t: float


@dataclass
class Fact:
    key: str
    value: object
    evidence: list[Evidence]
    source: str      # observed (direct input) | proposed (agent) | derived (rule)
    t: float
    version: int


@dataclass
class WorkingMemory:
    segments: dict[str, Segment] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    facts: dict[str, Fact] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    version: int = 0

    def _bump(self) -> int:
        self.version += 1
        return self.version

    def add_segment(self, seg_id: str, rev: int, speaker: str, text: str, t: float) -> list[str]:
        """Insert or revise a segment; returns the keys of facts invalidated by the revision."""
        old = self.segments.get(seg_id)
        if old and rev <= old.rev:
            return []
        if old and old.text == text:
            old.rev = rev
            return []
        self.segments[seg_id] = Segment(seg_id, rev, speaker, text, t)
        if not old:
            self.order.append(seg_id)
        self._bump()
        stale = [k for k, f in self.facts.items()
                 if any(e.kind == "speech" and e.ref == seg_id and e.rev < rev for e in f.evidence)]
        for k in stale:
            self.log.append({"op": "invalidate", "key": k, "because": f"{seg_id} revised to r{rev}", "t": t})
            del self.facts[k]
        return stale

    def is_current(self, ev: Evidence) -> bool:
        if ev.kind != "speech":
            return True
        seg = self.segments.get(ev.ref)
        return bool(seg and seg.rev == ev.rev)

    def put(self, key: str, value: object, evidence: list[Evidence], source: str, t: float) -> bool:
        """Commit a fact; refuses facts whose evidence was revised while they were being produced."""
        if not all(self.is_current(e) for e in evidence):
            self.log.append({"op": "reject_stale", "key": key, "t": t})
            return False
        self.facts[key] = Fact(key, value, evidence, source, t, self._bump())
        self.log.append({"op": "put", "key": key, "value": value, "source": source, "t": t})
        return True

    def get(self, key: str, default=None):
        f = self.facts.get(key)
        return f.value if f else default

    def recent(self, n: int = 6) -> list[Segment]:
        return [self.segments[s] for s in self.order[-n:]]

    def facts_with_prefix(self, prefix: str) -> list[Fact]:
        return sorted((f for k, f in self.facts.items() if k.startswith(prefix)), key=lambda f: f.key)

    def search(self, query: str, k: int = 3) -> list[Segment]:
        """Keyword overlap search over the transcript; an embedding index can replace this."""
        words = {w for w in query.lower().split() if len(w) > 2}
        scored = [(len(words & set(s.text.lower().split())), i, s)
                  for i, s in enumerate(self.segments[x] for x in self.order)]
        return [s for score, _, s in sorted(scored, key=lambda x: (-x[0], -x[1])) if score][:k]
