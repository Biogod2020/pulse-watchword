# Narration audio for the submission film (2026-11-03)

Final film: `pulse_data/film4/cut/pulse_muse.mp4` (v4). The previous narration is kept as `pulse_muse_v1_qwenvo.mp4`.
Before/after listening pairs: `pulse_data/film4/cut/ab_listen/`.

## What sounded "strange" (diagnosis of v1)
- Gemini, as three listeners (voice-over director, mixing engineer, viewer), named these problems: sentence endings
  dropping flat; list cadence; music ducking that pumped between sentences; a voice that was very dry against the music;
  sibilance.
- Measurements confirmed four of them:
  - every line was a separate clone with no direction;
  - ASR heard "Pulse" as "Pse", "Meta's" as "metatas", and "Glimmer" as "Gimmer";
  - mid-phrase pauses lasted up to 0.78 s, and the final "Pulse." was followed by 2.2 s;
  - music came back up in every pause longer than 0.3 s (the duck was a 0.3 s moving average).

## Engine bake-off (`film4/tts_lab.py`, `film4/oai_tts.py`, `film4/oai_audio.py`)
The test script was the first three lines (~45 s).

Candidates:
- Qwen3-TTS VoiceDesign (whole paragraph, and per line with a delivery note);
- Qwen3-TTS clone (paragraph);
- MOSS VoiceGenerator and MOSS-TTS clone (paragraph);
- OpenAI `gpt-4o-mini-tts` and `gpt-audio-1.5` (several voices);
- v1.

Blind ranking by Gemini (all clips in one prompt, two orders):
- Qwen VoiceDesign per line with a delivery note won both rounds (7/10);
- v1 scored 5–6;
- MOSS came last.

**How far the AI judges can be trusted:**
- **Pairwise A/B is unusable.** Gemini picked the first clip in 36 of 42 pairs, and only 6 of 21 pairs agreed across the two orders.
- **Absolute single-clip scoring (Gemini and gpt-audio, 3–5 repeats) does not separate good TTS from a human.** Three human
  Spoken-Wikipedia readings scored 4.8–5.2. Every good TTS take scored 5.5–6.8, and repeats vary by about ±0.5.
- **Selection was therefore based on defects that can be checked:**
  - ASR word error rate (`tts_lab.py score`);
  - the clip fitting its picture budget;
  - a per-clip Gemini defect list: mispronunciations, dropped or added words, glitches, mid-phrase pauses (`film4/defect_check.py`).

**Chosen:**
- `gpt-audio-1.5`, voice "cedar". It placed first or tied first with both judges, and a preset voice keeps the same
  timbre on every line.
- Each line carries its own delivery note (`film4/vo4_notes.json`), with 3–8 takes per line.
- Two lines were reworded slightly, because dense commas produced choppy delivery:
  - M4: "Here it is, running live on real footage";
  - M10: "Next come glasses".

## Post and mix
- **`film4/pause_fix_en.py`** shortens over-long pauses; nothing is lengthened and no speech is edited.
  - SenseVoice timestamps are aligned to the script words, so every silence is placed after a specific word and its punctuation.
  - Targets: inside a phrase 0.10 s, comma 0.30 s, colon 0.38 s, sentence end 0.55 s, and 0.8 s in the short slogan.
  - Pauses are now 0.26–0.56 s; they were up to 0.9 s, plus 2.2 s in the slogan.
- **`film4/vo_post.py`** processes each line at 48 kHz:
  - high-pass filter, -2 dB at 300 Hz, +1.5 dB at 3 kHz, +2 dB above 10 kHz;
  - de-esser on 5–10 kHz, compression 2.5:1, a 6% small room;
  - -20 LUFS per line.
  - A variant with no high-frequency boost and a stronger de-esser cut sibilance peaks by 4–8 dB, but scored slightly
    lower with both judges (5.7 vs 6.0), so the brighter chain was kept.
- **`film4/cut4.py duck_curve`:**
  - Music stays 11 dB down through pauses shorter than 2.5 s, so it no longer swells between sentences.
  - Ducking reaches full depth in 0.25 s and releases over 0.9 s.
  - Measured: constant -11 dB from 0 to 52 s, where v1 swelled in every pause.
- **Final level:** -13.9 LUFS integrated, -1.3 dBFS peak.
- **Narration inside the final mix:** ASR word error rate is 0–6.5% for M1–M10. For M11, ASR merges "Pulse. Always" into one word.

## Objective evidence (v6, the version in the film)
**Tools.** In `/cwStorage/home/jijh/.venvs/mos` (PyPI):
- `tools/mos_score.py`: UTMOS22, the VoiceMOS 2022 winner, and Microsoft Distill-MOS. Both are trained to predict
  human MOS ratings.
- `tools/prosody_compare.py`: prosody compared with 20 clean human read-speech clips (LibriSpeech).
- `tools/end_fall.py`: pitch fall at the end of each phrase.

| | UTMOS | Distill-MOS | phrase-end fall (st) | prosody distance to human |
|---|---|---|---|---|
| human read speech (LibriSpeech, clean) | 3.82 | 4.48 | 1.68 | 0 (reference) |
| human Spoken-Wikipedia (home recordings) | 3.44 | 4.29 | - | 1.26 |
| v1 (Qwen clone, the film before this work) | 4.52 | 4.59 | 2.26 | 1.00 |
| v5 (gpt-audio, first pick) | 4.40 | 4.62 | 2.54 | 0.92 |
| **v6 (final)** | **4.42** | **4.60** | **1.97** | **0.76** |

- **Acoustic quality:** above clean human recordings on both predictors, and equal to v1 (within noise).
  - The first post chain cost 0.1 UTMOS / 0.26 Distill-MOS. An ablation found the cause: a limiter, and to a lesser
    degree the presence/air EQ and the compressor.
  - The final chain is high-pass, -2 dB at 300 Hz, a light de-esser, a 6% room and -20 LUFS, with no limiter.
- **Prosody:** what still sounded "strange" was the sentence endings. TTS sinks too far, which listeners hear as flat,
  dropping endings.
  - v6 picks, for each line, the take whose phrase-end fall is closest to human read speech (`film4/oai_audio.py`
    with an explicit note: "do not let the pitch sink into a low, creaky voice").
  - Pauses are trimmed to human-like lengths (`film4/pause_fix_en.py`): comma 0.25 s, sentence end 0.45 s.
  - Prosody distance to human dropped from 1.00 (v1) to 0.76. Softening endings with PSOLA instead was tried and
    rejected: it cost 0.09 UTMOS and barely changed the fall.
- **Defects:** a per-line Gemini check on the final processed lines found no clicks, clipped, dropped or changed
  words, mispronunciations or creaky endings (0 of 11 lines). Naturalness under the same prompt is 4.8 raw and 4.7
  processed, a difference within noise.
- **Final mix:**
  - -14.5 LUFS integrated, -0.9 dBFS peak;
  - narration ASR word error rate inside the mix 0–8%;
  - the music duck holds -11 dB through gaps shorter than 2.5 s.

## Final version: v9, Qwen3-TTS VoiceDesign (female), calibrated against professional narration
**The reference.** Audiobook-style reading is not keynote narration, so the reference became professional narration:
- 39 utterances from five public-domain NASA narrated films, one of them read by Octavia Spencer;
- the narration was separated from the music with Demucs (`htdemucs`);
- our film mixes went through the identical pipeline: Demucs, then utterance split (`tools/split_utts.py`).

**What professional narrators do that our TTS did not:**
- a much wider pitch range: 4.6 st spread, against 2.7 for v1;
- phrase endings that stay up: 0.7 st fall, against 2.3 for v1;
- a slightly slower pace: 4.18 syllables/s, against 4.55 for v1.

**v9:**
- Qwen3-TTS VoiceDesign, with a per-line delivery note plus an "expressive documentary narrator" style and an
  "endings stay buoyant" note (`tts_lab.py prod`);
- 8 seeds per line, gated by ASR word error rate <= 0.08 and speaker similarity >= 0.97 to the anchor voice;
- ranked by prosody distance to the professional set (`tools/take_features.py`), with lines whose ending fall
  exceeded 3 st replaced from the top 3 by a lower-fall take;
- then pause trim, the post chain, and the 5%-slower "problem" animation (M2 is longer).

**Results**, all measured in the final film mix after Demucs separation:

| | prosody distance to pro | f0 spread | clip-end fall | syll/s | strict Gemini naturalness (clips x reps) |
|---|---|---|---|---|---|
| professional narration (NASA) | 0 | 4.57 | 1.16 | 4.18 | 6.50 (16x2) |
| v1 (Qwen clone, before) | 0.89 | 2.71 | 2.41 | 4.55 | 5.38 (8x1) |
| v7 (gpt-audio cedar) | 0.57 | 3.26 | 1.93 | 4.51 | 5.69 (12x3) |
| v8 (Qwen design) | 0.39 | 4.04 | 4.69 | 4.30 | 6.20 (11x3) |
| **v9 (final)** | **0.37** | 3.68 | **1.85** | 4.37 | **6.20 (10x3)** |

**The "strict Gemini" test.** A single clip per call, with the prompt "from a finished film, separated from its
music". It asks for defects, naturalness 1-10, and a human/AI verdict.
- **It discriminates:** on the first 8 clips per group, professional narration scored 7.25 and 6/8 human, our v7
  5.75 and 0/8.
- **Large sample:**
  - v9 is 0.30 below professional narration, with a bootstrap 95% CI of [-1.22, +0.59]. v9 is not distinguishable
    from professional narration on this measure.
  - v7 is -0.81, CI [-1.76, +0.15].
- **The human/AI verdict is too noisy to rank versions.** Identical v8 audio got 0/8 in one run and 3/8 in another,
  and Octavia Spencer was called "AI" in 2 of 8 clips.
- **Other checks on v9:**
  - UTMOS 4.49 / Distill-MOS 4.52 (clean human LibriSpeech: 3.82 / 4.48);
  - zero defects on all 11 lines (per-line Gemini check);
  - narration ASR word error rate in the mix 0–9%. M11 reads correctly once 0.5 s of context is included; the
    error is the ASR cutting at the first phoneme.
  - final mix -14.3 LUFS, -1.3 dBFS peak.
- **Kept for comparison:**
  - `cut/pulse_muse_v1_qwenvo.mp4` (before), `pulse_muse_v7_cedar.mp4`, `pulse_muse_v8_qwen.mp4`, `pulse_muse_v9_qwen.mp4` (= `pulse_muse.mp4`);
  - `vo_final_v7/` holds the cedar lines.

## Final version: MOSS-TTS, one continuous read (user's preference) + classical acoustic tools
**1. Generation.** MOSS-TTS v1.5 voice clone reads all 11 lines in ONE pass (`film4/tts_lab.py mclone` with
`REF_WAV` = a 14.7 s line of the Qwen3-TTS documentary narrator).
- 4 seeds, about 2:10 each, whole-take ASR word error rate 2.3–5.6%.
- MOSS-VoiceGenerator (voice described in words) stopped after about 33 s on long text, so it was not used.

**2. Splitting** (`film4/align_split.py`):
- SenseVoice word times are aligned to the script.
- Each boundary is cut in a silence. Every candidate silence within ±1.5 s is then tried, keeping the one whose two
  neighbouring pieces transcribe closest to their own lines.
- Timing alone had cut "Just | sentences" and "does | it" in the wrong place.

**3. Choosing the take:** seed 3 is closest to professional narration. Lines M1–M10 come from seed 3.
- f0 spread 4.72 st vs pro 4.57, 4.19 syll/s vs 4.18, prosody distance 0.30.
- The tagline M11 comes from seed 4: phrase-end fall 2.1 st instead of 5.9; Gemini once flagged a creaky
  "thinking" in the seed-3 tagline.

**4. Classical acoustic tools, tested before use:**
- **Kept:**
  - pause normalisation by punctuation (`pause_fix_en.py`);
  - the post chain (`vo_post.py`): high-pass, -2 dB at 300 Hz, light de-esser, 6% room, -20 LUFS, no limiter.
- **Rejected after measurement:**
  - WORLD-vocoder ending-lift (`film4/world_fix.py`, harvest/cheaptrick/d4c, edits spliced only into phrase tails)
    did cut the ending fall from 4.8 to 2.4 / 1.9 st. But the strict Gemini test scored it lower:
    -0.21, CI [-0.52, +0.06], and -0.30, CI [-0.61, -0.03].
  - Praat-PSOLA ending-lift and pitch-range expansion: no gain (paired CIs include 0), and slightly lower MOS.

**Results**, final film mix, Demucs-separated, same pipeline as every other version:

| | prosody dist. to pro | f0 spread | syll/s | pause mean | strict naturalness | judged human | mean defects |
|---|---|---|---|---|---|---|---|
| professional narration (NASA) | 0 | 4.57 | 4.18 | 0.39 | 6.50 | 22/32 | 1.94 |
| v9 (Qwen per line) | 0.37 | 3.68 | 4.37 | 0.29 | 6.20 | 0/30 | 1.93 |
| **MOSS one pass (final)** | **0.29** | **4.50** | **4.24** | **0.37** | 6.13 | **4/30** | **1.73** |

These figures were measured on the seed-3 cut before the boundary and tagline fixes. On the final version:
- **Defects:** per-line Gemini check, 2 repeats per line plus 3 for the new tagline: zero defects.
- **Mix:** narration ASR word error rate 0–6.5%; -14.1 LUFS, -1.2 dBFS peak.
- **Versus professional narration:** naturalness is -0.37, CI [-1.30, +0.57], so not significantly different.

**Other versions kept:** v9 lines in `vo_final_v9/`, `cut/pulse_muse_v9_qwen.mp4`, and `pulse_muse_v7_cedar.mp4`.

## Honest status
- Both AI judges score the final narration about 6/10. They give human readings about 5.
- Their remaining comments repeat ("list cadence", "ducking"), and the ducking comment contradicts the measured gain
  curve. They no longer point to defects we can verify.
- A human listen to `ab_listen/` is the real test.
- The final voice is female (MOSS-TTS v1.5, cloned from the Qwen3-TTS narrator design, open weights). The end card credits it as an AI-generated voice.
