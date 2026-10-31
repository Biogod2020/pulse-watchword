"""Replay the everyday scenario: speech + first-person frames through one heartbeat.

    python run_daily.py   # djev heartbeat + vision, DeepSeek stand-in agent, EmbeddingGemma 2 memory
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pulse.heartbeat import Event, Heartbeat
from pulse.longterm import LongTermMemory, RemoteEmbedder
from pulse.memory import WorkingMemory
from pulse.packs.daily import context, item_locations, make_perceive, make_remember, open_tasks, pack
from pulse.providers import DjevDecider, OpenAICompatAgent

PROXY = "http://127.0.0.1:4002/v1"


async def main(args):
    scenario = json.loads(Path(args.scenario).read_text())
    djev = DjevDecider(args.djev_url)
    agent = OpenAICompatAgent(PROXY, args.model, extra={"thinking": {"type": "disabled"}},
                              name=f"{args.model} (stand-in)")
    ltm = LongTermMemory(RemoteEmbedder(args.embed_url) if args.embed_url else None)
    caps, rules = pack(agent, djev, ltm)
    hb = Heartbeat(djev, WorkingMemory(), caps, rules, context,
                   perceive=make_perceive(args.djev_url), remember=make_remember(ltm))
    for raw in scenario["events"]:
        tick = await hb.on_event(Event(**raw))
        e = tick["event"]
        what = (f"[{e['seg_id']}] {e['speaker']}: {e['text']}" if e["kind"] == "speech"
                else f"<frame {e['data']['frame']}> {e['data'].get('caption', '')}")
        fired = [a["capability"] for a in tick["asked"] if a["fired"]]
        probs = " ".join(f"{a['question']}={a['p_yes']:.2f}" for a in tick["asked"] if a["p_yes"] is not None)
        print(f"t={tick['t']:>4} {what}")
        print(f"       perceive {tick['perceive_ms']:.0f} ms | read {tick['read_ms']:.0f} ms | remember "
              f"{tick['remember_ms']:.0f} ms | {probs} | fired={fired or '-'}"
              + (f" ERROR {tick['decider_error']}" if tick["decider_error"] else ""))
        for name, res in tick["results"].items():
            print(f"       -> {name}: {json.dumps(res, ensure_ascii=False)[:260]}")
    print("\nOPEN TASKS", json.dumps(open_tasks(hb.memory), ensure_ascii=False))
    print("ITEMS", json.dumps(item_locations(hb.memory), ensure_ascii=False))
    print("LTM", ltm.stats)
    out = Path(args.out or f"runs/{scenario['id']}_{int(time.time())}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scenario": scenario["id"], "ticks": hb.trace, "memory_log": hb.memory.log},
                              ensure_ascii=False, indent=2, default=str))
    print("trace saved to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenarios/daily/daily.json")
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--embed-url", default="http://127.0.0.1:18820")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--out")
    asyncio.run(main(p.parse_args()))
