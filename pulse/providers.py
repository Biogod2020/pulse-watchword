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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import httpx


@dataclass
class Question:
    id: str
    ask: str
    options: list[str] = field(default_factory=lambda: ["yes", "no"])
    kind: str = "choice"            # choice | noul (djev's native yes/no read; options must be yes/no)
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

    async def decide(self, context: str, questions: list[Question]) -> DecisionBatch: ...


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

    async def decide(self, context: str, questions: list[Question]) -> DecisionBatch:
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

    def __init__(self, url: str, model: str = "dgemma", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.model = model
        self.name = "djev"
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def decide(self, context: str, questions: list[Question]) -> DecisionBatch:
        t = time.perf_counter()
        decisions: dict[str, Decision] = {}
        for i in range(0, len(questions), self.CHUNK):
            part = questions[i:i + self.CHUNK]
            body = {"model": self.model, "state": context, "samples": 1,
                    "questions": {q.id: ({"type": "noul", "instructions": q.ask} if q.kind == "noul" else
                                         {"type": "choice", "instructions": q.ask,
                                          "criteria": {o: (q.criteria or {}).get(o) for o in q.options}})
                                  for q in part}}
            try:
                r = await self.client.post(self.url + "/v1/systemone", json=body)
            except httpx.HTTPError as e:
                raise ProviderError(str(e)) from e
            if r.status_code != 200:
                raise ProviderError(f"{r.status_code}: {r.text[:300]}")
            for q in part:
                a = r.json()["answers"][q.id]
                if q.kind == "noul":
                    p = float(a["noul"])
                    decisions[q.id] = Decision(q.id, "yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})
                else:
                    decisions[q.id] = Decision(q.id, a["choice"], {k: float(v) for k, v in a["probabilities"].items()})
        return DecisionBatch(decisions, (time.perf_counter() - t) * 1e3, self.name, calibrated=True)


class ScriptedDecider:
    """Deterministic provider for tests: answers come from a function of (context, question)."""

    def __init__(self, fn: Callable[[str, Question], dict[str, float] | str], name: str = "scripted"):
        self.fn = fn
        self.name = name

    async def decide(self, context: str, questions: list[Question]) -> DecisionBatch:
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
