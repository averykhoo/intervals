import sys, time, random
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
from intervals.kernel import Builder
from fractions import Fraction
N = 50000
random.seed(9)
# v1 library: build N disjoint pieces [2i, 2i+1]
m1 = v1.MultiInterval.merge(*[v1.MultiInterval(2*i, 2*i+1) for i in range(N)])
assert len(m1.endpoints) == 2*N
b = Builder()
for i in range(N): b.add_piece(2*i, 2*i+1)
m2 = v2.MultiInterval._from_cuts(b.build()) if hasattr(v2.MultiInterval, '_from_cuts') else None
if m2 is None:
    m2 = v2.MultiInterval.from_pieces([(2*i, 2*i+1) for i in range(N)]) if hasattr(v2.MultiInterval,'from_pieces') else None
print('v2 type', type(m2))
news = [random.randrange(0, 2*N) + Fraction(1,4) for _ in range(10)]
t = time.perf_counter()
for x in news:
    m1.add(v1.MultiInterval(x, x + Fraction(1,2)))   # v1's library incremental insert
t1 = time.perf_counter() - t
t = time.perf_counter()
for x in news:
    m2 = m2 | v2.MultiInterval(x, x + Fraction(1,2))
t2 = time.perf_counter() - t
print(f'10 library inserts into {N}: v1 MultiInterval.add {t1:.3f}s, v2 mi | piece {t2:.3f}s')
# compare results as sets: membership at a set of probes
probes = [Fraction(k, 8) for k in range(-4, 8*2*N+8, 997)] + [x for x in news] + [x + Fraction(1,2) for x in news] + [x + Fraction(5,8) for x in news]
bad = [p for p in probes if (p in m1) != (p in m2)]
print('membership disagreements', len(bad), 'of', len(probes))
# can-fail check: deliberately wrong expectation
wrong = [p for p in probes if (p in m1) != (not (p in m2))]
print('deliberately wrong expectation catches', len(wrong))
import compare
print('compare.py imported by v1 modules?', any('compare' in (getattr(m,'__name__','')) for m in [v1]))
