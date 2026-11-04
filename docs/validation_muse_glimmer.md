# Muse Glimmer as Pulse's only model (2026-11-03)

Setup: Muse Glimmer 30B (`meta-models/Muse-Glimmer-30B`, BF16; checkpoint sha256 matches the hub) on vLLM 0.32, one H100
(`tools/serve_glimmer.sh`, prefix caching, `muse_glimmer` tool and reasoning parsers).

## How a yes/no probability is read
System: "You are a careful visual judge. Reasoning strength: low." The user turn holds 4 frames and the question. The
assistant turn is pre-filled with `Answer:` (`continue_final_message=True`). The chat template then renders
`<|start|>assistant to=user<|message|>Answer:`, so the model answers directly with no reasoning channel. We request
`max_tokens=1, top_logprobs=20` and compute P(yes) = Σp(yes-like) / (Σp(yes-like) + Σp(no-like)) over the tokens
" yes", "yes", " Yes", " no", and so on.
Second look: the same request without the pre-fill at "Reasoning strength: medium". The model reasons first, and its
final word (yes/no) is the verdict.

## Stock clips, every second (`tools/glimmer_scan.py`, runs/glimmer_scan.json)
| clip | own question, max P(yes) | a different clip's question, max |
|---|---|---|
| cat across laptop | 0.84 | 0.02 |
| cat at cake | 0.13 | 0.00 |
| coffee | 0.74 | 0.00 |
| cat on table | 0.93 | 0.08 |
| wave (wave only in the first 3 s, then a blown kiss) | 0.33 | 0.01 |
| pill | 0.38 | 0.02 |
Second look (medium reasoning) on the top-scoring seconds: it confirmed the cat across the laptop, coffee and the cat on
the table. It rejected the wave and the pill. Its reasoning: "hand over mouth, pill not visible, could be a cough" and
"hand mostly static". It also rejected all near-miss seconds we tried (score 0.1–0.2). Frames at 896 px did not change
these verdicts.
Consequence: in Pulse Live the Glimmer candidate line is 0.15 (`--candidate-threshold`), not 0.5.

## Held-out EPIC-KITCHENS P12_101 (`eval_glimmer_windows.py`, `eval_heldout.py`)
2,044 two-second clips (4 of the 8 frames at 4 fps, 640×360), 6 cues, one request per cue: 495 s wall time. Part of
that time overlapped with demo recording on the same GPU. Thresholds were picked on minutes 0–10 (best event-level F1,
at least 3 fires, rising-edge rule with a 4 s cooldown). They were frozen and scored on minutes 10–34. The baseline is
the djev DiffusionGemma 2-s video clips from section 7 of `validation_epic_P12_101.md`, rescored with the same
procedure.

| cue | Glimmer AUC | th | fires | precision | caught | baseline AUC | th | fires | precision | caught | chance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| washing | 0.97 | 0.2 | 23 | 0.78 | 16/20 | 0.96 | 0.05 | 37 | 0.54 | 18/20 | 0.07 |
| stirring | 0.89 | 0.95 | 43 | 0.74 | 19/29 | 0.89 | 0.95 | 89 | 0.18 | 12/29 | 0.36 |
| pouring | 0.92 | 0.7 | 11 | 0.73 | 8/20 | 0.79 | 0.95 | 29 | 0.35 | 9/20 | 0.09 |
| opening | 0.81 | 0.7 | 6 | 0.67 | 8/28 | 0.80 | 0.9 | 38 | 0.24 | 10/28 | 0.10 |
| tap_or_knob | 0.90 | 0.15 | 34 | 0.50 | 18/36 | 0.94 | 0.4 | 27 | 0.63 | 18/36 | 0.09 |
| putting_down | 0.87 | 0.6 | 41 | 0.42 | 17/36 | 0.71 | 0.6 | 126 | 0.20 | 24/36 | 0.11 |
"Caught" counts held-out events that overlap at least one fire. For stirring, the earlier report used a state tracker,
not a rising edge. This table uses the rising edge for every cue, so stirring precision here is not comparable to that
report.

## Pulse Live takes (`start_muse.sh`, recorded with film2/record2.py into pulse_data/film4/ui)
cat across laptop 0.33 → confirmed; cat on table 0.35 → confirmed; coffee 0.54 → confirmed (first attempt: candidate
rejected); cat at cake 0.16 → confirmed; "where did I leave my keys?" → correct place, 21 frames scanned, 7.3 s.
Brain: writing a watch question takes about 10–12 s at reasoning strength low.
Latency (single request, warm): quick look 0.09–0.23 s; second look 10–14 s while recording ran alongside, 4–7 s alone.
