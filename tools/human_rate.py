"""Summarise strict single-clip Gemini verdicts (naturalness, defects, judged human) per clip-name prefix.
  human_rate.py DIR prefix [prefix ...]"""
import glob, json, re, sys
D = sys.argv[1]
for g in sys.argv[2:]:
    rows = []
    for f in sorted(glob.glob(f"{D}/{g}*.md")):
        m = re.findall(r'\{[^{}]*"naturalness"[^{}]*\}', open(f).read())
        try:
            rows.append(json.loads(m[-1]))
        except Exception:
            pass
    if rows:
        print(f"{g:14s} n={len(rows):2d} naturalness {sum(r['naturalness'] for r in rows) / len(rows):.2f} "
              f"defects {sum(r['n_defects'] for r in rows) / len(rows):.2f} judged human {sum(r['verdict'] == 'human' for r in rows)}/{len(rows)}")
