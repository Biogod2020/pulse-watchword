# X post (Most Viral prize) — attach the 30–45 s vertical cut or the first 40 s of the film

**Main post**
I told my camera, in plain words: "tell me when the cat walks across my laptop."

It watched every second. It caught him. 🐈

Pulse runs entirely on @Meta's open-weights Muse Glimmer, on one GPU, with no cloud API:
⚡ a 0.2 s "quick look" every second
🧠 a slower "second look" only when something seems to happen

Built for the #MetaAIHackathon 👇

**Reply 1 (how)**
The trick: Glimmer answers each watch with ONE token, and we read P(yes) from the log-probs. That makes it cheap enough
to run all day. When the score rises, the same model looks again at higher reasoning strength. Always watching, rarely
thinking.

**Reply 2 (proof)**
On 34 min of real head-camera kitchen video, with thresholds set on the first 10 min and scored on the rest, 73–78% of
Glimmer's alerts for washing, stirring and pouring were right (chance: 7–36%). It still misses small quick gestures;
the limits are in the repo.

**Reply 3 (next)**
Next: Ray-Ban Meta glasses through the Wearables Device Access Toolkit (not built yet). Code: https://github.com/Biogod2020/pulse-watchword
