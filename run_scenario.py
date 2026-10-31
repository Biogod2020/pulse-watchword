"""Replay a scripted scenario through the heartbeat and save the full trace.

    python run_scenario.py --decider deepseek --agent deepseek
    python run_scenario.py --decider djev --djev-url http://127.0.0.1:18811 --agent deepseek
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pulse.heartbeat import Event, Heartbeat
from pulse.memory import WorkingMemory
from pulse.providers import DjevDecider, OpenAICompatAgent, OpenAICompatDecider
from pulse.packs.surgical_count import FixtureVision, context, ledger, pack

PROXY = "http://127.0.0.1:4002/v1"
NO_THINKING = {"thinking": {"type": "disabled"}}


def build_decider(args):
    if args.decider == "djev":
        return DjevDecider(args.djev_url)
    return OpenAICompatDecider(PROXY, args.model, extra=NO_THINKING, name=f"{args.model} (stand-in)")


async def main(args):
    scenario = json.loads(Path(args.scenario).read_text())
    agent = OpenAICompatAgent(PROXY, args.model, extra=NO_THINKING, name=f"{args.model} (stand-in)")
    caps, rules = pack(agent, FixtureVision())
    hb = Heartbeat(build_decider(args), WorkingMemory(), caps, rules, context)
    for raw in scenario["events"]:
        tick = await hb.on_event(Event(**raw))
        e = tick["event"]
        what = f"[{e['seg_id']} r{e['rev']}] {e['speaker']}: {e['text']}" if e["kind"] == "speech" else f"<{e['kind']} {e['data']}>"
        fired = [a["capability"] for a in tick["asked"] if a["fired"]]
        print(f"t={tick['t']:>4} {what}")
        print(f"       read {tick['read_ms']:.0f} ms ({tick['provider']}) fired={fired or '-'}"
              + (f" invalidated={tick['invalidated']}" if tick["invalidated"] else "")
              + (f" ERROR {tick['decider_error']}" if tick["decider_error"] else ""))
        for name, res in tick["results"].items():
            short = {k: v for k, v in res.items() if k not in ("calls",)}
            print(f"       -> {name}: {json.dumps(short, ensure_ascii=False)[:220]}")
        for a in tick["alerts_raised"]:
            print(f"       !! {a['severity'].upper()} {a['text']}")
    final = ledger(hb.memory)
    print("\nFINAL LEDGER", json.dumps({k: v for k, v in final.items() if k != "events"}))
    out = Path(args.out or f"runs/{scenario['id']}_{args.decider}_{int(time.time())}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scenario": scenario["id"], "decider": args.decider, "model": args.model,
                               "ticks": hb.trace, "final_ledger": final, "memory_log": hb.memory.log},
                              ensure_ascii=False, indent=2))
    print("trace saved to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenarios/or_sponge_count.json")
    p.add_argument("--decider", choices=["deepseek", "djev"], default="deepseek")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--out")
    asyncio.run(main(p.parse_args()))
