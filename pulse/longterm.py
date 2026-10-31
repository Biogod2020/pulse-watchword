"""Long-term (episodic) memory: every utterance and kept camera frame, searchable by meaning.

Text and images share one EmbeddingGemma 2 space, so a spoken question ("where did I put my
glasses?") can retrieve a frame nobody described in words. Embeddings only find candidates;
deciding what a retrieved frame or line means is left to the decider or the agent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    name: str

    async def texts(self, texts: list[str], prompt: str | None) -> np.ndarray: ...

    async def images(self, paths: list[str]) -> np.ndarray: ...


class RemoteEmbedder:
    """Client for tools/embed_server.py."""

    def __init__(self, url: str = "http://127.0.0.1:18820", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.name = "embeddinggemma-2"
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def _post(self, body: dict) -> np.ndarray:
        r = await self.client.post(self.url + "/embed", json=body)
        r.raise_for_status()
        return np.asarray(r.json()["vectors"], dtype=np.float32)

    async def texts(self, texts: list[str], prompt: str | None) -> np.ndarray:
        return await self._post({"texts": texts, "prompt": prompt})

    async def images(self, paths: list[str]) -> np.ndarray:
        return await self._post({"images": [str(Path(x).resolve()) for x in paths]})


@dataclass
class Episode:
    id: str
    kind: str                # speech | frame
    t: float
    text: str                # utterance, or the frame caption
    image: str | None = None
    meta: dict = field(default_factory=dict)


class LongTermMemory:
    def __init__(self, embedder: Embedder | None):
        self.embedder = embedder
        self.items: list[Episode] = []
        self.vecs: list[np.ndarray] = []
        self.stats = {"embed_ms": 0.0, "embedded": 0}

    async def add(self, ep: Episode) -> None:
        vec = None
        if self.embedder:
            t = time.perf_counter()
            if ep.image:
                vec = (await self.embedder.images([ep.image]))[0]
            else:
                vec = (await self.embedder.texts([ep.text], "document"))[0]
            self.stats["embed_ms"] += (time.perf_counter() - t) * 1e3
            self.stats["embedded"] += 1
        # replace an older version of the same episode (an ASR revision)
        for i, old in enumerate(self.items):
            if old.id == ep.id:
                self.items[i], self.vecs[i] = ep, vec
                return
        self.items.append(ep)
        self.vecs.append(vec)

    async def search(self, query: str, k: int = 3, kind: str | None = None) -> list[dict]:
        pool = [i for i, ep in enumerate(self.items) if kind is None or ep.kind == kind]
        if not pool:
            return []
        if self.embedder and all(self.vecs[i] is not None for i in pool):
            q = (await self.embedder.texts([query], "query"))[0]
            scores = {i: float(self.vecs[i] @ q) for i in pool}
        else:
            words = {w for w in query.lower().split() if len(w) > 2}
            scores = {i: float(len(words & set(self.items[i].text.lower().split()))) for i in pool}
        # Text-text similarity runs higher than text-image, so frames and utterances are ranked
        # separately and the top k of each are returned (the modality gap would otherwise hide frames).
        best = []
        for kd in sorted({self.items[i].kind for i in pool}):
            best += sorted((i for i in pool if self.items[i].kind == kd), key=lambda i: -scores[i])[:k]
        return [{"id": self.items[i].id, "kind": self.items[i].kind, "t": self.items[i].t,
                 "text": self.items[i].text, "image": self.items[i].image, "score": round(scores[i], 3)}
                for i in best]
