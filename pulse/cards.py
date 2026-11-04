"""Shareable "moment" cards: the frame, what Pulse noticed, when, and a small brand line.

Portrait 1080x1350 (feeds on Instagram/X/WeChat Moments crop it cleanly).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

def _font(env, names):
    """Cards carry bilingual (Chinese/English) labels, so they need a CJK font: env var, then common install paths."""
    cands = [os.environ.get(env)] + [str(Path(d).expanduser() / n) for d in
             ("~/.fonts/noto-cjk", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/noto-cjk",
              "/usr/share/fonts/google-noto-cjk", "/System/Library/Fonts") for n in names]
    return next((c for c in cands if c and Path(c).exists()), names[0])


FONT_BOLD = _font("PULSE_FONT_BOLD", ["NotoSansCJKsc-Bold.otf", "NotoSansCJK-Bold.ttc", "PingFang.ttc"])
FONT_REG = _font("PULSE_FONT_REGULAR", ["NotoSansCJKsc-Regular.otf", "NotoSansCJK-Regular.ttc", "PingFang.ttc"])
W, H = 1080, 1350


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _tokens(text: str) -> list[str]:
    """Latin words stay whole; CJK characters are their own tokens."""
    out, word = [], ""
    for ch in text:
        if ch.isascii() and not ch.isspace():
            word += ch
            continue
        if word:
            out.append(word)
            word = ""
        out.append(ch)
    return out + ([word] if word else [])


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for tok in _tokens(text):
        trial = cur + tok
        if draw.textlength(trial, font=font) > width and cur.strip():
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur = trial
    return lines + ([cur.rstrip()] if cur.strip() else [])


def strip_emoji(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) < 0x2190 or 0x2E80 <= ord(ch) <= 0xFFEF).strip()


def moment_card(frame: str, title: str, subtitle: str, stamp: str, out: str, accent: str = "#22d3ee") -> str:
    img = Image.open(frame).convert("RGB")
    bg = img.resize((W, H)).filter(ImageFilter.GaussianBlur(40))
    bg = Image.blend(bg, Image.new("RGB", (W, H), "#0b1020"), 0.55)
    card = bg.copy()
    # photo with rounded corners
    pw = W - 120
    ph = int(img.height * pw / img.width)
    ph = 820
    scaled = img.resize((pw, max(ph, int(img.height * pw / img.width)))) if img.height * pw / img.width >= ph else \
        img.resize((int(img.width * ph / img.height), ph))
    left = (scaled.width - pw) // 2
    top = (scaled.height - ph) // 2
    photo = scaled.crop((left, top, left + pw, top + ph))
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw, ph), 36, fill=255)
    card.paste(photo, (60, 150), mask)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((60, 150, 60 + pw, 150 + ph), 36, outline=accent, width=4)
    # header
    d.ellipse((60, 62, 88, 90), fill=accent)
    d.text((104, 52), "Pulse noticed", font=_font(FONT_BOLD, 40), fill="white")
    d.text((W - 60, 60), stamp, font=_font(FONT_REG, 32), fill="#cbd5e1", anchor="ra")
    # text block
    y = 150 + ph + 46
    for line in _wrap(d, strip_emoji(title), _font(FONT_BOLD, 64), W - 120)[:1]:
        d.text((60, y), line, font=_font(FONT_BOLD, 64), fill="white")
        y += 86
    sub = _wrap(d, strip_emoji(subtitle), _font(FONT_REG, 34), W - 120)
    for i, line in enumerate(sub[:2]):
        if i == 1 and len(sub) > 2:
            line = line.rstrip(" ,.;") + "…"
        d.text((60, y), line, font=_font(FONT_REG, 34), fill="#cbd5e1")
        y += 48
    # footer
    d.line((60, H - 118, W - 60, H - 118), fill="#334155", width=2)
    d.text((60, H - 96), "pulse", font=_font(FONT_BOLD, 34), fill="white")
    d.text((160, H - 92), "always watching, rarely thinking", font=_font(FONT_REG, 28), fill="#94a3b8")
    d.text((W - 60, H - 92), "told it in plain words", font=_font(FONT_REG, 28), fill=accent, anchor="ra")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    card.save(out, quality=92)
    return out


def moment_reel(clip: str, title: str, stamp: str, out: str, loops: int = 3) -> str:
    """A 9:16 shareable reel: the moment clip looped, padded on a blurred copy, with the label and brand burned in."""
    import subprocess
    import tempfile
    ffmpeg = os.environ.get("PULSE_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    tmp = Path(tempfile.mkdtemp(prefix="pulse_reel_"))
    (tmp / "title.txt").write_text(strip_emoji(title))
    (tmp / "brand.txt").write_text(f"pulse noticed · {stamp}")
    (tmp / "tag.txt").write_text("always watching, rarely thinking")
    t = lambda name: str(tmp / name).replace(":", "\\:")
    vf = (f"[0:v]split=2[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5,"
          f"eq=brightness=-0.25[bg];[b]scale=1000:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2-80,"
          f"drawtext=fontfile={FONT_BOLD}:textfile={t('brand.txt')}:x=60:y=120:fontsize=44:fontcolor=0x22d3ee,"
          f"drawtext=fontfile={FONT_BOLD}:textfile={t('title.txt')}:x=(w-text_w)/2:y=h-560:fontsize=72:fontcolor=white,"
          f"drawtext=fontfile={FONT_REG}:textfile={t('tag.txt')}:x=(w-text_w)/2:y=h-200:fontsize=38:fontcolor=0x94a3b8")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-stream_loop", str(loops - 1), "-i", clip, "-filter_complex", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags",
                    "+faststart", "-an", out], check=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return out
