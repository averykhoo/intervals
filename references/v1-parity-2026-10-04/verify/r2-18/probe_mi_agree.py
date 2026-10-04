import sys, warnings, random
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
from intervals import MultiInterval as MI
import numpy as np
from fractions import Fraction as F
def v1set(m):  # endpoints -> (lo, lo_closed, hi, hi_closed)
    e = m.endpoints; out = []
    for i in range(0, len(e), 2):
        (lo, le), (hi, he) = e[i], e[i+1]
        out.append((lo, le == 0, hi, he == 0))
    return out
def v2set(m): return [(lo, bool(lc), hi, bool(hc)) for lo, lc, hi, hc in m.pieces()] if hasattr(m,'pieces') else None
import intervals.kernel as K
def v2p(m): return [(lo, lc, hi, hc) for lo, lc, hi, hc in K.pieces(m.cuts)]
flags = ['no', 'yes', '', None, 0, 1, 2, [], [0], np.bool_(False), np.bool_(True), True, False, 'False', 0.0, F(0)]
rng = random.Random(18); n = 0; bad = []
for _ in range(400):
    a, b = sorted(rng.sample(range(-5, 6), 2)); sf, ef = rng.choice(flags), rng.choice(flags)
    try: r1 = v1set(v1.MultiInterval(a, b, start_closed=sf, end_closed=ef))
    except Exception as e: r1 = type(e).__name__
    try: r2 = v2p(MI(a, b, start_closed=sf, end_closed=ef))
    except Exception as e: r2 = type(e).__name__
    n += 1
    if r1 != r2: bad.append((a, b, sf, ef, r1, r2))
print('ranged cases', n, 'disagree', len(bad), bad[:3])
for sf in flags:
    for ef in flags:
        try: r1 = v1set(v1.MultiInterval(5, start_closed=sf, end_closed=ef))
        except Exception as e: r1 = type(e).__name__
        try: r2 = v2p(MI(5, start_closed=sf, end_closed=ef))
        except Exception as e: r2 = type(e).__name__
        if r1 != r2: print('point differs', repr(sf), repr(ef), 'v1', r1, 'v2', r2)
# sabotage: a deliberately wrong expectation must be caught
print('sabotage caught:', v2p(MI(0, 1, start_closed='no')) != [(0, False, 1, True)])
