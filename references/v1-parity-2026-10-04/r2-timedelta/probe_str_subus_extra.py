"""follow-ups to probe_str_subus: v1's per-end text vs half-even rounding (how many ends, which disagree),
and a v2 workaround reproducing v1's rounded text (timedelta(microseconds=round(exact*10**6)))"""
import sys
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-timedelta']
import random
import datetime as dt
from fractions import Fraction
import probe_str_subus as P  # re-runs the base probe's prints first (cheap)

rng = random.Random(20261004)
tot = rhe = 0
bad = []
wk_ok = wk_n = 0
for i in range(600):
    m = rng.choice([18, 20, 21, 22, 23, 24, 26])
    span = rng.choice([4, 64, 2 ** 12, 2 ** 30, 2 ** 40])
    pieces = []
    for _ in range(rng.randint(1, 3)):
        a = Fraction(rng.randint(-span, span), 2 ** m)
        b = a + Fraction(rng.randint(0, span // 2 + 1), 2 ** m)
        sc, ec = rng.random() < .5, rng.random() < .5
        if a == b:
            sc = ec = True
        pieces.append((a, b, sc, ec))
    a1, a2 = P.build1(pieces), P.build2(pieces)
    for (o1, x1, y1, z1), (lo, lc, hi, hc) in zip(P.split_v1(str(a1)), P.canon2(a2)):
        for t1, exact in ((x1, lo), (y1, hi)):
            tot += 1
            ok = P.dec1(t1) == P.round_half_even_us(exact)
            rhe += ok
            if not ok and len(bad) < 4:
                bad.append((t1, float(exact), float(P.dec1(t1) - exact) * 10 ** 6))
    # workaround: v2 piece ends rounded half-even to the us, printed by python, == v1's text
    for p2, (o1, x1, y1, z1) in zip(a2, P.split_v1(str(a1))):
        wk_n += 1
        r = lambda v: str(dt.timedelta(microseconds=round(Fraction(v) * 10 ** 6)))
        wk_ok += (r(p2.inf_seconds), r(p2.sup_seconds)) == (x1, y1)
print(f'v1 end texts: {tot}; equal to the exact end rounded half-even to the us: {rhe}; others e.g. {bad}')
print(f'workaround timedelta(microseconds=round(p.inf_seconds*10**6)) reproduces v1 end text: {wk_ok}/{wk_n} pieces')
