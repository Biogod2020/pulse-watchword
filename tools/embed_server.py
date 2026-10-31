"""Tiny EmbeddingGemma 2 service: text and image embeddings in one 768-d space.

    python tools/embed_server.py --port 18820 --device cpu
POST /embed {"texts": [...], "prompt": "query"|"document"|null} or {"images": ["/abs/path.png", ...]}
"""
from __future__ import annotations

import argparse
import time

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
MODEL: SentenceTransformer | None = None


class EmbedIn(BaseModel):
    texts: list[str] = []
    images: list[str] = []
    prompt: str | None = None


@app.get("/health")
def health():
    return {"ok": MODEL is not None}


@app.post("/embed")
def embed(req: EmbedIn):
    t = time.perf_counter()
    if req.images:
        vecs = MODEL.encode([Image.open(p).convert("RGB") for p in req.images], normalize_embeddings=True)
    else:
        vecs = MODEL.encode(req.texts, prompt_name=req.prompt, normalize_embeddings=True)
    return {"vectors": vecs.tolist(), "ms": (time.perf_counter() - t) * 1e3}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18820)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    # float32: the model card warns that fp16 activations overflow
    MODEL = SentenceTransformer("google/embeddinggemma-2", device=a.device, model_kwargs={"torch_dtype": torch.float32})
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
