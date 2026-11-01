"""Automatic scenario packs: from a one-sentence description to calibrated heartbeat questions.

1. draft(): the brain writes narrow yes/no watch questions in the style validated on real video (a fixed
   "Does this video itself show ..." prefix, one concrete action per question, explicit exclusions).
2. calibrate(): a teacher (a stronger, slower judge) labels a small sample of clips; each question's threshold
   for the cheap judge is the one that agrees best with the teacher (F1). No human labels are used.
The cheap judge then runs continuously with those thresholds.
"""
from __future__ import annotations

import json
import re

import httpx

PREFIX = "Does this video itself show "

DRAFT_SYSTEM = """You design the always-on "watch list" of a camera assistant worn by a person doing a task.
A fast judge answers yes/no questions about every 2-second first-person video clip. Write questions that it can
answer reliably:
- each question asks about ONE concrete, visible action or state (e.g. "a hand tightening a bolt with a wrench"),
- never broad ones ("is something important happening?", "is the person making a mistake?"),
- add explicit exclusions for the closest confusable case ("Not just holding the wrench."),
- cover the steps that matter most in this task and the visible signs of the most likely mistakes.
Return JSON: {"questions": [{"id": "snake_case", "ask": "<text that completes: Does this video itself show ...?>",
"why": "<one short reason>"}]}"""


def _chat(base_url: str, model: str, key: str, system: str, user: str, max_tokens: int = 1500) -> str:
    r = httpx.post(base_url.rstrip("/") + "/chat/completions", timeout=180, trust_env=False,
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0])
    return json.loads(text[start:])


def draft(scenario: str, n: int, base_url: str, model: str, key: str, steps: list[str] | None = None) -> list[dict]:
    """steps: the work instruction / SOP for the task, when one exists (a professional pack starts from it)."""
    user = f"Task: {scenario}\n"
    if steps:
        user += "Work instruction (steps in order):\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(steps)) + "\n"
    out = _json(_chat(base_url, model, key, DRAFT_SYSTEM, user + f"Write {n} questions."))
    qs = []
    for q in out["questions"][:n]:
        ask = q["ask"].strip()
        ask = ask[len(PREFIX):] if ask.startswith(PREFIX) else ask[0].lower() + ask[1:]
        qs.append({"id": re.sub(r"[^a-z0-9_]", "_", q["id"].lower())[:40], "ask": PREFIX + ask.rstrip("?") + "?",
                   "why": q.get("why", "")})
    return qs


def map_to_labels(questions: list[dict], vocabulary: list[str], base_url: str, model: str, key: str) -> dict:
    """For evaluation only: which annotated action labels does each question correspond to?"""
    listing = "\n".join(vocabulary)
    text = _chat(base_url, model, key,
                 "Match each question to the annotated action labels (verb noun) it would be true for. Use only "
                 "labels from the list; an empty list if none match. Return JSON {\"<question id>\": [labels]}.",
                 "QUESTIONS:\n" + "\n".join(f"{q['id']}: {q['ask']}" for q in questions) + "\n\nLABELS:\n" + listing,
                 max_tokens=3000)
    m = _json(text)
    vocab = set(vocabulary)
    return {q["id"]: [x for x in m.get(q["id"], []) if x in vocab] for q in questions}


def best_threshold(scores: list[float], teacher: list[bool], min_positives: int = 3) -> tuple[float, float]:
    """Threshold for the cheap judge that best agrees with the teacher's yes/no (F1). With fewer than
    `min_positives` teacher positives there is nothing to calibrate against, so the default 0.5 is kept."""
    best = (0.5, 0.0)
    if sum(teacher) < min_positives:
        return best
    for th in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pred = [s >= th for s in scores]
        tp = sum(p and t for p, t in zip(pred, teacher))
        fp = sum(p and not t for p, t in zip(pred, teacher))
        fn = sum(t and not p for p, t in zip(pred, teacher))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (th, f1)
    return best
