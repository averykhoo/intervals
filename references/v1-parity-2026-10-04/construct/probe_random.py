"""random_multi_interval: v1's generator vs a v2-side re-spelling (same random draws), and hypothesis strategies"""
from common import *
import random

def v2_random_multi_interval(start, end, n, decimals=2, prob_neg_inf=0.25, prob_pos_inf=0.25):
    """the same draws as v1's, built with MultiInterval.from_pieces (v1's default flag: no closed inf)"""
    pts = set()
    if random.random() < prob_neg_inf and n > 1:
        pts.add(-math.inf)
    if random.random() < prob_pos_inf and n > 1:
        pts.add(math.inf)
    if decimals:
        while len(pts) < 2 * n:
            pts.add(round(start + (end - start) * random.random(), decimals))
    else:
        pts.update(range(start, end + 1)); pts = sorted(pts); random.shuffle(pts); pts = set(pts[:2 * n])
    e = sorted(pts); pieces = []
    for i in range(0, len(e), 2):
        lo, hi, x = e[i], e[i + 1], random.random()
        lo_inf, hi_inf = math.isinf(lo), math.isinf(hi)
        if not lo_inf and (x < .2 or lo == hi):
            pieces.append((lo, lo, True, True))
        elif x < .4 and not lo_inf and not hi_inf:
            pieces.append((lo, hi, True, True))
        elif x < .6 and not lo_inf:
            pieces.append((lo, hi, True, False))
        elif x < .8 and not hi_inf:
            pieces.append((lo, hi, False, True))
        else:
            pieces.append((lo, hi, False, False))
    return M2.from_pieces(pieces)

t = Tally('v1 random_multi_interval == v2 re-spelling (same seed)')
tv = Tally('v1 output valid (consistency check) and v2 parses its str')
for seed in range(500):
    args = (random.Random(seed).choice([-100, -10, 0]), random.Random(seed + 1).choice([10, 100]),
            seed % 6, [0, 1, 2, 3][seed % 4])
    random.seed(seed); o1 = outcome(lambda: v1.random_multi_interval(*args))
    random.seed(seed); o2 = outcome(lambda: v2_random_multi_interval(*args))
    if o1[0] == 'raise' or o2[0] == 'raise':
        t.check(o1[0] == o2[0], (seed, args, o1, o2)); continue
    t.check(v1_to_v2(o1[1]) == o2[1], (seed, args, str(o1[1]), str(o2[1])))
    c = outcome(o1[1]._consistency_check)
    tv.check(c[0] == 'ok' and M2.parse(str(o1[1])) == o2[1], (seed, args, c))
t.report(); tv.report()

print('=== hypothesis strategies (the documented replacement)')
sys.path.insert(0, ROOT)
from tests import strategies as S
from hypothesis import given, settings, HealthCheck
seen = []
@settings(max_examples=200, database=None, suppress_health_check=list(HealthCheck))
@given(S.cut_tuples())
def collect(c):
    seen.append(M2.from_cuts(c))
collect()
print(f'  drew {len(seen)} sets from tests/strategies.py::cut_tuples; with an inf member:',
      sum(1 for s in seen if math.inf in s or -math.inf in s), '; multi-piece:', sum(1 for s in seen if len(s) > 1))

print('=== v1 failure modes')
random.seed(0); print('  decimals=0, range too small for n:', outcome(lambda: v1.random_multi_interval(0, 3, 5, 0)))
print('  decimals=1, range [0, 0.5] (6 values) with n=5 (needs 10 distinct): run separately with a timeout (probe_random_hang.py)')

print('=== sabotage'); s = Tally('sab')
random.seed(1); a = v1.random_multi_interval(-10, 10, 3, 1); random.seed(2); b = v2_random_multi_interval(-10, 10, 3, 1)
s.check(v1_to_v2(a) == b, 'different seeds must differ'); assert s.report() == 1
