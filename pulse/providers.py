"""Pluggable model providers.

Two roles, each behind a small interface so any vendor can fill it:
- DecisionProvider: the cheap "intuition" read. Many multiple-choice questions over one context,
  answered in a single call. Calibrated providers return per-option probabilities.
- AgentProvider: the slow, deliberate worker. A chat model with tool calls, woken only when the
  heartbeat decides a capability is worth running.
"""
from __future__ import annotations

import json
import re
import asyncio
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import httpx


@dataclass
class Question:
    id: str
    ask: str
    options: list[str] = field(default_factory=lambda: ["yes", "no"])
    kind: str = "choice"            # choice | noul (native yes/no read) | span (short free text, e.g. a caption)
    requires: str | None = None     # hierarchical: asked only after this question fired
    scope: str = "context"          # which state the provider reads: context | event
    criteria: dict[str, str] | None = None   # optional definition per option (djev choice criteria)
    threshold: float | None = None           # overrides the capability threshold for this question


@dataclass
class Decision:
    id: str
    choice: str
    probs: dict[str, float] | None = None   # None when the provider cannot expose probabilities


@dataclass
class DecisionBatch:
    decisions: dict[str, Decision]
    latency_ms: float
    provider: str
    calibrated: bool


class DecisionProvider(Protocol):
    name: str

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch: ...


class ProviderError(RuntimeError):
    pass


def _strip_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ProviderError(f"no JSON object in reply: {text[:200]}")
    return json.loads(text[start:end + 1])


DECIDE_SYSTEM = (
    "You are the fast intuition layer of an always-on assistant. You never act and never give advice. "
    "You only answer multiple-choice questions about the CONTEXT, judging from what is written there. "
    "Reply with one JSON object mapping each question id to exactly one of its options, nothing else.")


class OpenAICompatDecider:
    """Any OpenAI-compatible chat endpoint. Probabilities are not available through every gateway
    (the local opencode proxy drops logprobs), so this provider reports calibrated=False and the
    heartbeat falls back to the argmax choice."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", extra: dict | None = None,
                 timeout: float = 60.0, name: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.extra = extra or {}
        self.name = name or f"openai:{model}"
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False,
                                        headers={"Authorization": f"Bearer {api_key}"})

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch:
        if images or videos:
            raise ProviderError(f"{self.name} is text-only here; route image/video questions to a vision decider")
        listing = "\n".join(f"- {q.id}: {q.ask} Options: {', '.join(q.options)}" for q in questions)
        body = {"model": self.model, "temperature": 0, "max_tokens": 40 * len(questions) + 40,
                "messages": [{"role": "system", "content": DECIDE_SYSTEM},
                             {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTIONS:\n{listing}"}],
                **self.extra}
        t = time.perf_counter()
        try:
            r = await self.client.post(self.base_url + "/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise ProviderError(str(e)) from e
        if r.status_code != 200:
            raise ProviderError(f"{r.status_code}: {r.text[:300]}")
        raw = _strip_json(r.json()["choices"][0]["message"].get("content") or "")
        decisions = {}
        for q in questions:
            value = str(raw.get(q.id, "")).strip().lower()
            choice = next((o for o in q.options if o.lower() == value), q.options[-1])
            decisions[q.id] = Decision(q.id, choice)
        return DecisionBatch(decisions, (time.perf_counter() - t) * 1e3, self.name, calibrated=False)


class DjevDecider:
    """djev structured_server (POST /v1/systemone): one denoising read, probabilities per option.
    Pre-existing component, used here through a new client."""

    CHUNK = 10  # djev changes its answer format above ten questions per read

    @staticmethod
    def _wire(q: Question) -> dict:
        if q.kind == "noul":
            return {"type": "noul", "instructions": q.ask}
        if q.kind == "span":
            return {"type": "span", "instructions": q.ask, "criteria": {"max_tokens": 32}}
        return {"type": "choice", "instructions": q.ask,
                "criteria": {o: (q.criteria or {}).get(o) for o in q.options}}

    def __init__(self, url: str, model: str = "dgemma", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.model = model
        self.name = "djev"
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch:
        t = time.perf_counter()
        decisions: dict[str, Decision] = {}
        for i in range(0, len(questions), self.CHUNK):
            part = questions[i:i + self.CHUNK]
            body = {"model": self.model, "state": context, "samples": 1,
                    "questions": {q.id: self._wire(q) for q in part}}
            if images:
                body["images"] = images
            if videos:
                body["videos"] = videos
            try:
                r = await self.client.post(self.url + "/v1/systemone", json=body)
            except httpx.HTTPError as e:
                raise ProviderError(str(e)) from e
            if r.status_code != 200:
                raise ProviderError(f"{r.status_code}: {r.text[:300]}")
            for q in part:
                a = r.json()["answers"][q.id]
                if q.kind == "span":
                    decisions[q.id] = Decision(q.id, (a.get("text") or "").strip() if a.get("found", True) else "")
                elif q.kind == "noul":
                    p = float(a["noul"])
                    decisions[q.id] = Decision(q.id, "yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})
                else:
                    decisions[q.id] = Decision(q.id, a["choice"], {k: float(v) for k, v in a["probabilities"].items()})
        return DecisionBatch(decisions, (time.perf_counter() - t) * 1e3, self.name, calibrated=True)


class ChatYesNoDecider:
    """Yes/no probabilities from any OpenAI-compatible vision chat endpoint, one request per question (run
    concurrently; the server batches them and shares the image prefix).

    Used for Meta Muse models: Muse Glimmer served locally with vLLM (`tools/serve_glimmer.sh`) as the always-on
    judge, and Muse Spark through the Meta Model API (https://api.meta.ai/v1) as the second look.
    probability: "logprobs" reads P(yes) from the first answer token (needs logprobs; vLLM has them);
    "json" asks for {"yes_probability": x} when an API does not return logprobs.
    Glimmer's chat template lets the assistant pick a channel (to=self reasoning, to=user answer); `prefill`
    continues an assistant message addressed to the user, so the next token is the answer.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "none", name: str | None = None,
                 max_images: int = 4, probability: str = "logprobs", prefill: bool = True, timeout: float = 60.0,
                 effort: str = "low"):
        self.base = base_url.rstrip("/"); self.model = model; self.key = api_key; self.effort = effort
        self.name = name or model; self.max_images = max_images; self.probability = probability; self.prefill = prefill
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    def _messages(self, context: str, ask: str, images: list[str]) -> list[dict]:
        step = max(1, len(images) // self.max_images) if images else 1
        imgs = (images or [])[::step][:self.max_images]
        content = [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        tail = {"logprobs": "Answer with exactly one word, yes or no.",
                "reason": "Look carefully at every frame, then answer with exactly one word, yes or no.",
                "json": 'Reply with JSON only: {"yes_probability": <number between 0 and 1>}.'}[self.probability]
        content.append({"type": "text", "text": f"{context}\nThe images are frames in time order.\nQuestion: {ask}\n{tail}"})
        msgs = [{"role": "system", "content": f"You are a careful visual judge.\n\nReasoning strength: {self.effort}."},
                {"role": "user", "content": content}]
        if self.prefill and self.probability == "logprobs":
            msgs.append({"role": "assistant", "content": "Answer:"})
        return msgs

    async def _one(self, context: str, q: Question, images: list[str]) -> Decision:
        body = {"model": self.model, "messages": self._messages(context, q.ask, images), "temperature": 0}
        if self.probability == "logprobs":
            body.update(max_tokens=1, logprobs=True, top_logprobs=20)
            if self.prefill:
                body.update(add_generation_prompt=False, continue_final_message=True)
        elif self.probability == "reason":     # think first (Glimmer reasoning channel), then a plain yes/no
            body.update(max_tokens=1200)
        else:
            body.update(max_tokens=40, response_format={"type": "json_object"})
        h = {"Authorization": f"Bearer {self.key}"}
        try:
            r = await self.client.post(self.base + "/chat/completions", json=body, headers=h)
        except httpx.HTTPError as e:
            raise ProviderError(str(e)) from e
        if r.status_code != 200:
            raise ProviderError(f"{r.status_code}: {r.text[:300]}")
        ch = r.json()["choices"][0]
        if self.probability == "logprobs":
            yes = no = 0.0
            for t in ch["logprobs"]["content"][0]["top_logprobs"]:
                w = t["token"].strip().lower()
                if w in ("yes", "y"):
                    yes += math.exp(t["logprob"])
                elif w in ("no", "n"):
                    no += math.exp(t["logprob"])
            p = yes / (yes + no) if yes + no > 0 else 0.0
        elif self.probability == "reason":
            words = re.findall(r"[a-z]+", (ch["message"].get("content") or "").lower())
            p = 1.0 if words and words[-1] == "yes" else 0.0
        else:
            p = float(_strip_json(ch["message"]["content"]).get("yes_probability", 0.0))
        return Decision(q.id, "yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch:
        t = time.perf_counter()
        out = await asyncio.gather(*[self._one(context, q, images or []) for q in questions])
        return DecisionBatch({d.id: d for d in out}, (time.perf_counter() - t) * 1e3, self.name, calibrated=True)


def _image_url(path: str) -> str:
    import base64
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


image_url = _image_url


class ScriptedDecider:
    """Deterministic provider for tests: answers come from a function of (context, question)."""

    def __init__(self, fn: Callable[[str, Question], dict[str, float] | str], name: str = "scripted"):
        self.fn = fn
        self.name = name

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch:
        decisions = {}
        calibrated = True
        for q in questions:
            out = self.fn(context, q)
            if isinstance(out, str):
                decisions[q.id] = Decision(q.id, out)
                calibrated = False
            else:
                decisions[q.id] = Decision(q.id, max(out, key=out.get), out)
        return DecisionBatch(decisions, 0.0, self.name, calibrated)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Awaitable[dict]]

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class AgentRun:
    text: str
    tool_calls: list[dict]
    latency_ms: float
    steps: int


class AgentProvider(Protocol):
    name: str

    async def run(self, system: str, user: str, tools: list[Tool], max_steps: int = 4,
                  stop_after_tools: bool = False) -> AgentRun: ...


class OpenAICompatAgent:
    """Plain tool-call loop over an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", extra: dict | None = None,
                 timeout: float = 120.0, name: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.extra = extra or {}
        self.name = name or f"openai:{model}"
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False,
                                        headers={"Authorization": f"Bearer {api_key}"})

    async def run(self, system: str, user: str, tools: list[Tool], max_steps: int = 4,
                  stop_after_tools: bool = False) -> AgentRun:
        """stop_after_tools: return once the first batch of tool calls has executed, for capabilities
        whose output is the tool effect itself (saves a round trip)."""
        by_name = {t.name: t for t in tools}
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        trace: list[dict] = []
        t = time.perf_counter()
        for step in range(1, max_steps + 1):
            body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 600, **self.extra}
            if tools:
                body["tools"] = [x.spec() for x in tools]
            try:
                r = await self.client.post(self.base_url + "/chat/completions", json=body)
            except httpx.HTTPError as e:
                raise ProviderError(str(e)) from e
            if r.status_code != 200:
                raise ProviderError(f"{r.status_code}: {r.text[:300]}")
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return AgentRun(msg.get("content") or "", trace, (time.perf_counter() - t) * 1e3, step)
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool = by_name.get(fn["name"])
                result = await tool.handler(args) if tool else {"error": f"unknown tool {fn['name']}"}
                trace.append({"tool": fn["name"], "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            if stop_after_tools:
                return AgentRun("", trace, (time.perf_counter() - t) * 1e3, step)
        return AgentRun("", trace, (time.perf_counter() - t) * 1e3, max_steps)

class BudgetExceeded(ProviderError):
    pass


class LunaDecider:
    """OpenAI Decisions API (POST /v1/decisions) with gpt-6-luna: typed choices with probabilities.

    Cost is tracked from returned usage at the published rate ($0.10 / M input tokens, $0.01 cached, output free,
    retrieved 2026-10-29); calls stop with BudgetExceeded once `budget_usd` would be crossed.
    """
    RATE_IN, RATE_CACHED = 0.10 / 1e6, 0.01 / 1e6

    def __init__(self, key_file: str = "~/.config/pulse/openai_key", model: str = "gpt-6-luna",
                 budget_usd: float = 8.0, timeout: float = 60.0, ledger: str | None = None):
        key = Path(key_file).expanduser().read_text().strip()
        self.model, self.name, self.budget = model, f"openai:{model}", budget_usd
        self.client = httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=timeout,
                                        headers={"Authorization": f"Bearer {key}"})
        self.ledger = Path(ledger).expanduser() if ledger else Path("~/.config/pulse/luna_spend.json").expanduser()
        self.spent = json.loads(self.ledger.read_text())["usd"] if self.ledger.exists() else 0.0
        self.calls = 0

    @staticmethod
    def _wire(q: Question) -> dict:
        options = q.options if q.kind != "noul" else ["yes", "no"]
        return {"type": "choice", "name": q.id, "instructions": q.ask,
                "choices": [{"value": o, "description": (q.criteria or {}).get(o) or o.capitalize()} for o in options]}

    async def decide(self, context: str, questions: list[Question],
                     images: list[str] | None = None, videos: list[str] | None = None) -> DecisionBatch:
        if self.spent >= self.budget:
            raise BudgetExceeded(f"Luna budget reached: ${self.spent:.4f} of ${self.budget}")
        if videos:
            raise ProviderError("the Decisions API path here takes images, not video; send sampled frames instead")
        inp: object = context
        if images:
            inp = [{"role": "user", "content": [{"type": "input_text", "text": context}] +
                    [{"type": "input_image", "image_url": u} for u in images]}]
        t = time.perf_counter()
        r = await self.client.post("/decisions", json={"model": self.model, "input": inp,
                                                        "questions": [self._wire(q) for q in questions]})
        if r.status_code != 200:
            raise ProviderError(f"{r.status_code}: {r.text[:300]}")
        d = r.json()
        u = d.get("usage", {})
        cached = u.get("input_tokens_details", {}).get("cached_tokens", 0)
        self.spent += (u.get("input_tokens", 0) - cached) * self.RATE_IN + cached * self.RATE_CACHED
        self.calls += 1
        self.ledger.write_text(json.dumps({"usd": self.spent, "calls_this_process": self.calls}))
        decisions = {}
        for a in d.get("answers", []):
            probs = {p["value"]: float(p["probability"]) for p in a.get("probabilities", [])}
            decisions[a["name"]] = Decision(a["name"], a.get("choice") or max(probs, key=probs.get), probs)
        return DecisionBatch(decisions, (time.perf_counter() - t) * 1e3, self.name, calibrated=True)
