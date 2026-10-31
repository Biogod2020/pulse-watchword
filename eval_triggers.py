"""Compare trigger-question designs on a labeled utterance set.

    python eval_triggers.py --djev-url http://127.0.0.1:18811
Each utterance is judged with a realistic preceding transcript (sponge talk earlier in the case),
which is what confused the first flat design.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pulse.providers import DjevDecider, Question

LABELS = ["sponge", "added", "out", "verify", "closing", "assistant"]
HISTORY = ["Scrub nurse: Initial count, ten lap sponges on the field.",
           "Circulator: Two sponges out, they're in the counter bag.",
           "Surgeon: Suction, please."]

# A: the first flat design (compound wording, choice yes/no, "NEWEST line" inside a transcript)
FLAT_V1 = {
    "added": "Does the NEWEST line state that surgical sponges were added to the field, taken out into the counter "
             "bag, or give a sponge count?",
    "closing": "Does the NEWEST line say the team is starting to close (fascia, wound or skin closure)?",
    "assistant": "Is the NEWEST line addressed to the assistant, asking it for information?",
}

PREFIX = "Does this utterance itself "
ATOMIC = {
    "sponge": "refer to surgical sponges (lap sponges, laps, raytecs, gauze) or the sponge count? Not a sponge "
              "stick, which is an instrument, and not needles, sutures or other instruments.",
    "added": "report new sponges being opened, handed onto the field or added to the count, including an initial "
             "count? Not using or asking for a sponge already on the field, and not a restated total.",
    "out": "report used sponges being taken off the field into the counter bag? Not sponges packed or left inside "
           "the patient, and not a restated total.",
    "verify": "ask for the sponge count to be checked, verified or reported?",
    "closing": "say the surgical team is starting to close the incision (fascia, wound or skin closure)? Not "
               "closing a door, and not 'close' meaning near.",
    "assistant": "directly address the assistant, asking it for information or to record something? Not a "
                 "question or request to another team member.",
}
TOPICS = {
    "sponges": "Surgical sponges (laps, raytecs, gauze) or the sponge count; not a sponge stick or other instrument",
    "closure": "Starting to close the incision",
    "assistant": "Speaking directly to the assistant",
    "other": "Anything else: instruments, needles, sutures, vitals, doors, routine requests",
}
GATE = {"added": "sponge", "out": "sponge", "verify": "sponge"}


def transcript_state(item: dict) -> str:
    return "\n".join(["TRANSCRIPT:"] + HISTORY + [f"NEWEST LINE: {item['speaker']}: {item['text']}"])


def anchored_state(item: dict) -> str:
    return "\n".join(["EARLIER TRANSCRIPT (context only):"] + HISTORY +
                     ["", f"THIS UTTERANCE: {item['speaker']}: {item['text']}"])


async def run_variant(decider, items, variant):
    rows, calls, ms = [], 0, []
    for it in items:
        t = time.perf_counter()
        if variant == "A_flat_v1":
            qs = [Question(k, v) for k, v in FLAT_V1.items()]
            b = await decider.decide(transcript_state(it), qs)
            calls += 1
            p = {k: d.probs["yes"] for k, d in b.decisions.items()}
            # the old compound question stood in for every sponge sub-question
            p.update({"sponge": p["added"], "out": p["added"], "verify": p["added"]})
        elif variant == "B_flat_noul":
            qs = [Question(k, PREFIX + v, kind="noul") for k, v in ATOMIC.items()]
            b = await decider.decide(anchored_state(it), qs)
            calls += 1
            p = {k: d.probs["yes"] for k, d in b.decisions.items()}
        elif variant == "D_flat_specific":
            qs = [Question(k, PREFIX + ATOMIC[k], kind="noul") for k in ATOMIC if k != "sponge"]
            b = await decider.decide(anchored_state(it), qs)
            calls += 1
            p = {k: d.probs["yes"] for k, d in b.decisions.items()}
            p["sponge"] = max(p["added"], p["out"], p["verify"])
        elif variant == "E_topic_choice":
            topic = Question("topic", "What is this utterance itself mainly about?", list(TOPICS))
            topic.criteria = TOPICS
            b = await decider.decide(anchored_state(it), [topic])
            calls += 1
            probs = b.decisions["topic"].probs
            p = {"sponge": probs["sponges"], "closing": 0.0, "assistant": probs["assistant"]}
            p.update({k: 0.0 for k in GATE})
            second = []
            if probs["sponges"] >= 0.5 or probs["assistant"] >= 0.5:
                second += [Question(k, PREFIX + ATOMIC[k], kind="noul") for k in GATE]
            if probs["assistant"] >= 0.5:
                second.append(Question("assistant", PREFIX + ATOMIC["assistant"], kind="noul"))
            if probs["closure"] >= 0.3:
                second.append(Question("closing", PREFIX + ATOMIC["closing"], kind="noul"))
            if second:
                b2 = await decider.decide(anchored_state(it), second)
                calls += 1
                p.update({k: d.probs["yes"] for k, d in b2.decisions.items()})
        else:
            first = [Question(k, PREFIX + ATOMIC[k], kind="noul") for k in ("sponge", "closing", "assistant")]
            b = await decider.decide(anchored_state(it), first)
            calls += 1
            p = {k: d.probs["yes"] for k, d in b.decisions.items()}
            if p["sponge"] >= 0.5:
                second = [Question(k, PREFIX + ATOMIC[k], kind="noul") for k in GATE]
                b2 = await decider.decide(anchored_state(it), second)
                calls += 1
                p.update({k: d.probs["yes"] for k, d in b2.decisions.items()})
            else:
                p.update({k: 0.0 for k in GATE})
        ms.append((time.perf_counter() - t) * 1e3)
        rows.append({"text": it["text"], "trap": it.get("trap"), "p": p})
    return rows, calls, ms


def score(items, rows, keys):
    out = {}
    for k in keys:
        tp = fp = fn = tn = 0
        for it, r in zip(items, rows):
            y, yhat = bool(it.get(k)), r["p"][k] >= 0.5
            tp += y and yhat; fp += (not y) and yhat; fn += y and not yhat; tn += (not y) and not yhat
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "acc": (tp + tn) / len(items)}
    return out


async def main(args):
    items = json.loads(Path("scenarios/trigger_labels.json").read_text())["items"]
    decider = DjevDecider(args.djev_url)
    report = {}
    for variant in args.variants:
        rows, calls, ms = await run_variant(decider, items, variant)
        keys = ["added", "closing", "assistant"] if variant == "A_flat_v1" else LABELS
        s = score(items, rows, keys)
        ms_sorted = sorted(ms)
        report[variant] = {"scores": s, "calls": calls, "p50_ms": ms_sorted[len(ms) // 2],
                           "p95_ms": ms_sorted[int(len(ms) * 0.95) - 1], "rows": rows}
        errs = sum(v["fp"] + v["fn"] for v in s.values())
        print(f"\n{variant}: {calls} reads, p50 {report[variant]['p50_ms']:.0f} ms, "
              f"p95 {report[variant]['p95_ms']:.0f} ms, total errors {errs}")
        for k, v in s.items():
            print(f"  {k:9} acc {v['acc']:.2f}  fp {v['fp']}  fn {v['fn']}")
        for it, r in zip(items, rows):
            wrong = [k for k in keys if (r["p"][k] >= 0.5) != bool(it.get(k))]
            if wrong:
                print(f"    x {it['text'][:52]:52} " + " ".join(f"{k}={r['p'][k]:.2f}" for k in wrong))
    Path("runs").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--out", default="runs/trigger_eval.json")
    p.add_argument("--variants", nargs="+", default=["A_flat_v1", "B_flat_noul", "C_hierarchical",
                                                       "D_flat_specific", "E_topic_choice"])
    asyncio.run(main(p.parse_args()))
